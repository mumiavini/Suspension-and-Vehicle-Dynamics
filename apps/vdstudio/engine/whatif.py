"""The sweep / what-if engine -- the primary interface CLAUDE.md asks for.

The designer names the inputs to vary and the consequences to report; the
engine fills a table with one row per combination. It never ranks rows, never
highlights a "best" one and never writes a value back into the design. Telling
the designer a combination is infeasible (a failed solve, a band left) is
information, not a decision.

Three queries:

* :func:`run_grid` -- full-factorial table over 1..N inputs.
* :func:`sensitivity` -- d(output)/d(input) by central difference about the
  current design, for a trade-off matrix.
* :func:`target_query` -- "which values of this input put that output at this
  target?" Every root in the range is returned, or none. The target is an
  argument to the query, never a property of the design (CLAUDE.md).

Points can be evaluated in a process pool. Each worker rebuilds the geometry
from a pickled :class:`Project`, so nothing is shared between processes.
"""

from __future__ import annotations

import contextlib
import itertools
import math
import multiprocessing
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .geometry import GeometryError, resolve
from .kpis import DYNAMIC_GROUPS, KpiSet, compute_dynamic, compute_static
from .project import CORNERS, Project, get_path, with_path

Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Param:
    """One design input the engine may vary. ``path`` is a dotted project path;
    ``mirror`` applies the same value to the right-side corner of a hardpoint
    (Y mirrored), so a symmetric car stays symmetric."""

    path: str
    label: str
    unit: str
    group: str
    step: float
    mode: str = "synthesis"          # "synthesis", "hardpoints" or "any"
    mirror: bool = False

    def get(self, project: Project) -> float:
        return float(get_path(project, self.path))

    def apply(self, project: Project, value: float) -> Project:
        p = with_path(project, self.path, float(value))
        if self.mirror:
            parts = self.path.split(".")
            # hardpoints.points.<corner>.<point>.<axis>
            corner, axis = parts[2], int(parts[4])
            other = {"FL": "FR", "RL": "RR"}.get(corner)
            if other:
                v = -float(value) if axis == 1 else float(value)
                p = with_path(p, ".".join([parts[0], parts[1], other, parts[3], parts[4]]), v)
        return p


def _axle_params(axle: str) -> list[Param]:
    a = axle.title()
    g = f"{a} axle (synthesis)"
    spec = [
        ("track_mm", "track (contact patches)", "mm", 10.0),
        ("loaded_radius_mm", "tyre loaded radius", "mm", 2.0),
        ("static_camber_deg", "static camber", "deg", 0.25),
        ("rc_height_mm", "roll centre height target", "mm", 5.0),
        ("fvsa_length_mm", "FVSA length target", "mm", 100.0),
        ("lbj_y_mm", "LBJ y (outboard)", "mm", 5.0),
        ("lbj_z_mm", "LBJ z", "mm", 5.0),
        ("kpi_deg", "KPI", "deg", 1.0),
        ("kingpin_length_mm", "kingpin length", "mm", 5.0),
        ("inner_pickup_y_mm", "inboard pickup plane y", "mm", 5.0),
        ("lca_base_mm", "LCA base", "mm", 10.0),
        ("lca_sweep_mm", "LCA sweep", "mm", 10.0),
        ("uca_base_mm", "UCA base", "mm", 10.0),
        ("uca_sweep_mm", "UCA sweep", "mm", 10.0),
        ("dz_lca_mm", "LCA pivot-axis rise", "mm", 2.0),
        ("dz_uca_mm", "UCA pivot-axis rise", "mm", 2.0),
    ]
    out = [Param(f"{axle}.{f}", f"{a} {lab}", u, g, s) for f, lab, u, s in spec]
    out.append(Param(f"{axle}.travel_bump_mm", f"{a} bump travel (sweep range)", "mm", "Sweep "
                     "settings", 5.0, "any"))
    out.append(Param(f"{axle}.roll_reference_deg", f"{a} reference roll angle", "deg",
                     "Sweep settings", 0.25, "any"))
    return out


def _steering_params() -> list[Param]:
    g = "Steering (synthesis)"
    spec = [
        ("caster_deg", "caster", "deg", 1.0),
        ("caster_offset_mm", "caster offset", "mm", 5.0),
        ("tro_height_along_kingpin_mm", "steering arm height on kingpin", "mm", 5.0),
        ("steer_arm_length_mm", "steering arm length", "mm", 5.0),
        ("steer_arm_angle_deg", "steering arm angle", "deg", 2.0),
        ("rack_x_mm", "rack x (rearward)", "mm", 5.0),
        ("rack_z_mm", "rack z", "mm", 2.0),
        ("rack_half_length_mm", "rack half length", "mm", 5.0),
        ("static_toe_deg_per_side", "static toe per side", "deg", 0.1),
    ]
    out = [Param(f"steering.{f}", f"Steering {lab}", u, g, s) for f, lab, u, s in spec]
    hw = "Steering hardware"
    out += [
        Param("steering.pinion_radius_mm", "Pinion radius", "mm", hw, 1.0, "any"),
        Param("steering.max_rack_travel_mm", "Max rack travel (half stroke)", "mm", hw, 2.0,
              "any"),
        Param("steering.steering_wheel_diameter_mm", "Steering wheel diameter", "mm", hw, 10.0,
              "any"),
        Param("steering.mu_parking", "Parking friction μ (tyre assumption)", "-", hw, 0.1,
              "any"),
        Param("steering.ackermann_at_steer_deg", "Ackermann reference outer angle", "deg", hw,
              2.0, "any"),
    ]
    return out


def _vehicle_params() -> list[Param]:
    g = "Vehicle"
    spec = [
        ("total_mass_kg", "total mass", "kg", 5.0),
        ("unsprung_mass_kg", "unsprung mass", "kg", 2.0),
        ("cg_height_mm", "CG height", "mm", 10.0),
        ("wheelbase_mm", "wheelbase", "mm", 10.0),
        ("cg_from_front_axle_mm", "CG behind front axle", "mm", 10.0),
        ("target_roll_gradient_deg_per_g", "roll gradient target", "deg/g", 0.1),
        ("brake_bias_front", "brake bias front", "-", 0.02),
        ("lltd_front", "LLTD front (load-case assumption)", "-", 0.02),
        ("design_ay_g", "design lateral g", "g", 0.1),
    ]
    return [Param(f"vehicle.{f}", f"Vehicle {lab}", u, g, s, "any") for f, lab, u, s in spec]


def _toe_link_params() -> list[Param]:
    g = "Rear toe link (synthesis)"
    out = []
    for end, label in (("inboard", "inboard"), ("outboard", "outboard")):
        for i, ax in enumerate("xyz"):
            out.append(Param(f"rear_toe_link_{end}.{i}", f"Rear toe link {label} {ax.upper()}",
                             "mm", g, 2.0))
    return out


def hardpoint_params() -> list[Param]:
    """Hardpoint-mode inputs, left side, mirrored to the right."""
    out = []
    for corner in ("FL", "RL"):
        axle = "Front" if corner == "FL" else "Rear"
        for point in ("UCA_IN_FRONT", "UCA_IN_REAR", "UCA_OUT", "LCA_IN_FRONT", "LCA_IN_REAR",
                      "LCA_OUT", "TIE_ROD_IN", "TIE_ROD_OUT", "WHEEL_CENTER"):
            for i, ax in enumerate("XYZ"):
                out.append(Param(f"hardpoints.points.{corner}.{point}.{i}",
                                 f"{axle} {point} {ax} (mirrored)", "mm",
                                 f"{axle} hardpoints", 2.0, "hardpoints", mirror=True))
    return out


def all_params() -> list[Param]:
    return (_vehicle_params() + _axle_params("front") + _axle_params("rear")
            + _steering_params() + _toe_link_params() + hardpoint_params())


def params_for(project: Project) -> list[Param]:
    mode = project.geometry_mode
    return [p for p in all_params() if p.mode in ("any", mode)]


def param_by_path(path: str) -> Param:
    for p in all_params():
        if p.path == path:
            return p
    raise KeyError(path)


# --------------------------------------------------------------------------- #
# Which dynamic group produces which KPI key
# --------------------------------------------------------------------------- #

_KEY_GROUP: dict[str, str] | None = None


def key_groups(project: Project | None = None) -> dict[str, str]:
    """Map each KPI key to ``"static"`` or the dynamic group that produces it.

    Built once by running each group on the default car (a few seconds).
    """
    global _KEY_GROUP
    if _KEY_GROUP is None:
        geo = resolve(project or Project())
        out = {k: "static" for k in compute_static(geo).items}
        for g in DYNAMIC_GROUPS:
            for k in compute_dynamic(geo, [g]).items:
                out[k] = g
        _KEY_GROUP = out
    return _KEY_GROUP


def groups_for(keys: Iterable[str]) -> set[str]:
    kg = key_groups()
    gs_ = {kg.get(k, "rates") for k in keys}
    gs_.discard("static")
    if "steering.effort_sw_torque" in keys or any(k.startswith("steering.effort") for k in keys):
        gs_.add("steer")
    return gs_


# --------------------------------------------------------------------------- #
# Evaluation of one point (module level so a process pool can pickle it)
# --------------------------------------------------------------------------- #

@dataclass
class PointResult:
    assignment: tuple[float, ...]
    values: dict[str, float]
    status: dict[str, str]
    error: str = ""


def evaluate(project: Project, params: Sequence[str], values: Sequence[float],
             outputs: Sequence[str], groups: Sequence[str]) -> PointResult:
    p = project
    try:
        for path, v in zip(params, values, strict=True):
            p = param_by_path(path).apply(p, v)
        geo = resolve(p)
        ks: KpiSet = compute_static(geo)
        if groups:
            ks = ks.merge(compute_dynamic(geo, groups, static=ks))
    except (GeometryError, ValueError, RuntimeError, ZeroDivisionError) as exc:
        return PointResult(tuple(values), {o: float("nan") for o in outputs},
                           {o: "fail" for o in outputs}, str(exc))
    vals, stat = {}, {}
    for o in outputs:
        k = ks.get(o)
        vals[o] = k.value if k is not None and k.ok else float("nan")
        stat[o] = k.status if k is not None else "fail"
    return PointResult(tuple(values), vals, stat)


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #

@dataclass
class GridResult:
    params: list[str]
    outputs: list[str]
    axes: list[np.ndarray]
    points: list[PointResult] = field(default_factory=list)

    def array(self, output: str) -> np.ndarray:
        """Values reshaped to the grid (len(axis0), len(axis1), ...)."""
        shape = tuple(len(a) for a in self.axes)
        flat = np.array([pt.values.get(output, np.nan) for pt in self.points], dtype=float)
        return flat.reshape(shape)

    def rows(self) -> list[dict[str, object]]:
        out = []
        for pt in self.points:
            row: dict[str, object] = dict(zip(self.params, pt.assignment, strict=True))
            row.update(pt.values)
            row["error"] = pt.error
            out.append(row)
        return out


def _pool_size() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))


class Runner:
    """Evaluates many points, serially or on a process pool, with progress/cancel."""

    def __init__(self, parallel: bool = True) -> None:
        self.parallel = parallel
        self._pool: ProcessPoolExecutor | None = None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def map(self, project: Project, params: Sequence[str], combos: list[tuple[float, ...]],
            outputs: Sequence[str], progress: Progress | None = None,
            cancelled: Cancelled | None = None) -> list[PointResult]:
        tasks = [(tuple(params), c) for c in combos]
        return self.run_tasks(project, tasks, outputs, progress, cancelled)

    def run_tasks(self, project: Project, tasks: list[tuple[tuple[str, ...], tuple[float, ...]]],
                  outputs: Sequence[str], progress: Progress | None = None,
                  cancelled: Cancelled | None = None) -> list[PointResult]:
        """Evaluate ``(params, values)`` tasks; results come back in task order."""
        groups = sorted(groups_for(outputs))
        n = len(tasks)
        results: list[PointResult | None] = [None] * n
        if not self.parallel or n < 4:
            for i, (ps, vs) in enumerate(tasks):
                if cancelled is not None and cancelled():
                    raise InterruptedError
                results[i] = evaluate(project, ps, vs, outputs, groups)
                if progress:
                    progress((i + 1) / n, f"{i + 1}/{n}")
            return [r for r in results if r is not None]
        if self._pool is None:
            # spawn everywhere: forking a multi-threaded (Qt) process can deadlock,
            # and spawn is what Windows does anyway
            self._pool = ProcessPoolExecutor(max_workers=_pool_size(),
                                             mp_context=multiprocessing.get_context("spawn"))
        futs: dict[Future[PointResult], int] = {
            self._pool.submit(evaluate, project, list(ps), list(vs), list(outputs), groups): i
            for i, (ps, vs) in enumerate(tasks)}
        pending = set(futs)
        done = 0
        while pending:
            if cancelled is not None and cancelled():
                for f in pending:
                    f.cancel()
                raise InterruptedError
            finished, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for f in finished:
                results[futs[f]] = f.result()
                done += 1
                if progress:
                    progress(done / n, f"{done}/{n}")
        return [r for r in results if r is not None]


def run_grid(project: Project, params: Sequence[str], axes: Sequence[Sequence[float]],
             outputs: Sequence[str], runner: Runner | None = None,
             progress: Progress | None = None, cancelled: Cancelled | None = None) -> GridResult:
    if len(params) != len(axes):
        raise ValueError("one axis of values per parameter")
    combos = [tuple(float(v) for v in c) for c in itertools.product(*axes)]
    runner = runner or Runner(parallel=False)
    pts = runner.map(project, params, combos, outputs, progress, cancelled)
    return GridResult(list(params), list(outputs), [np.asarray(a, float) for a in axes], pts)


# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #

@dataclass
class SensitivityResult:
    params: list[str]
    outputs: list[str]
    base: dict[str, float]
    steps: dict[str, float]
    # d[output][param] = d(output)/d(param), per unit of the param
    derivative: dict[str, dict[str, float]]

    def per_step(self, output: str, param: str) -> float:
        return self.derivative[output][param] * self.steps[param]


def sensitivity(project: Project, params: Sequence[str], outputs: Sequence[str],
                steps: dict[str, float] | None = None, runner: Runner | None = None,
                progress: Progress | None = None,
                cancelled: Cancelled | None = None) -> SensitivityResult:
    """Central-difference d(output)/d(input) about the current design."""
    steps = {p: (steps or {}).get(p, param_by_path(p).step) for p in params}
    base = {p: param_by_path(p).get(project) for p in params}
    tasks: list[tuple[tuple[str, ...], tuple[float, ...]]] = [((), ())]
    for p in params:
        tasks.append(((p,), (base[p] + steps[p],)))
        tasks.append(((p,), (base[p] - steps[p],)))
    runner = runner or Runner(parallel=False)
    res = runner.run_tasks(project, tasks, outputs, progress, cancelled)
    deriv: dict[str, dict[str, float]] = {o: {} for o in outputs}
    for j, p in enumerate(params):
        up, dn = res[1 + 2 * j], res[2 + 2 * j]
        for o in outputs:
            deriv[o][p] = (up.values[o] - dn.values[o]) / (2.0 * steps[p])
    return SensitivityResult(list(params), list(outputs), res[0].values, steps, deriv)


# --------------------------------------------------------------------------- #
# Target query
# --------------------------------------------------------------------------- #

@dataclass
class TargetQueryResult:
    param: str
    output: str
    target: float
    scan_x: np.ndarray
    scan_y: np.ndarray
    roots: list[float]
    message: str


def target_query(project: Project, param: str, output: str, target: float, lo: float,
                 hi: float, n_scan: int = 9, runner: Runner | None = None,
                 progress: Progress | None = None,
                 cancelled: Cancelled | None = None) -> TargetQueryResult:
    """Every value of ``param`` in [lo, hi] where ``output`` crosses ``target``.

    A coarse scan brackets sign changes; each bracket is refined with ``brentq``.
    No crossing -> no root, stated as such. Nothing is applied to the design.
    """
    runner = runner or Runner(parallel=False)
    xs = np.linspace(lo, hi, n_scan)
    pts = runner.map(project, [param], [(float(x),) for x in xs], [output], progress, cancelled)
    ys = np.array([pt.values[output] for pt in pts], dtype=float)
    groups = sorted(groups_for([output]))

    def f(x: float) -> float:
        if cancelled is not None and cancelled():
            raise InterruptedError
        return evaluate(project, [param], [x], [output], groups).values[output] - target

    roots: list[float] = []
    for i in range(len(xs) - 1):
        y0, y1 = ys[i] - target, ys[i + 1] - target
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue
        if y0 == 0.0:
            roots.append(float(xs[i]))
        elif y0 * y1 < 0:
            tol = 1e-4 * max(1.0, abs(hi - lo))
            with contextlib.suppress(ValueError, RuntimeError):
                roots.append(float(brentq(f, xs[i], xs[i + 1], xtol=tol)))
    if math.isfinite(ys[-1]) and ys[-1] - target == 0.0:
        roots.append(float(xs[-1]))
    if roots:
        msg = (f"{len(roots)} value(s) of the input put the output at the target inside "
               f"[{lo:g}, {hi:g}]. Nothing has been applied to the design.")
    elif not np.isfinite(ys).any():
        msg = "Every scan point failed -- the range leaves the region where the geometry solves."
    else:
        msg = (f"No crossing of {target:g} inside [{lo:g}, {hi:g}] -- the target is not "
               "reachable with this input alone over this range.")
    return TargetQueryResult(param, output, target, xs, ys, roots, msg)


def corners() -> tuple[str, ...]:
    return CORNERS
