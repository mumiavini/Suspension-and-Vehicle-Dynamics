"""Compute the app's KPI table from Altair MotionSolve instead of DWSolver.

This produces the "Altair" column the Streamlit app shows beside its vdcore
column. The two columns are deliberately built by the SAME code: MotionSolve's
solved positions are replayed through ``vdcore``'s own ``sample_corner`` /
``axle_rates`` / roll-centre construction, so the only thing that differs
between the columns is *where the upright ended up*. A disagreement is
therefore a kinematics disagreement, never a difference of definition.

HOW THE TRAVELS ARE CHOSEN
    MotionSolve samples a uniform travel grid, and ``vdcore``'s analysis asks
    for specific travels. Rather than interpolate MotionSolve's answer -- which
    would quietly inject fitting error into a validation tool -- the grid is
    chosen so every travel vdcore requests lands exactly on a solved sample
    (see :func:`sweep_intervals`). :class:`MotionSolveReplay` raises if it is
    ever asked for a travel it does not hold.

THE ONE PLACE ALTAIR IS NOT INDEPENDENT
    ``axle_roll`` finds the wheel travel at which both contact patches sit on
    the tilted road by a ``brentq`` root search. Each probe of that search would
    cost a MotionSolve subprocess, so the roll travel is taken from vdcore and
    Altair is evaluated AT that travel. The residual -- how far Altair's own
    patches are from meeting there -- is returned as ``roll_patch_residual_mm``
    so the approximation is visible rather than hidden. On the 2027 geometry
    the two solvers agree on position to ~1e-5 mm, so this is immaterial, but
    it is reported, not assumed.

Frame: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up. Units: mm, deg.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from altair_model.msolve_driver import (
    AltairUnavailableError,
    build_corner,
    read_csv_points,
    run_motionsolve,
)

# ``_rot2`` and ``_line_intersection`` are private to vdcore.analysis.axle on
# purpose: they are the roll-frame arithmetic ``axle_roll`` uses. Reaching for
# them here is deliberate -- reimplementing the rotation or the line
# intersection would let a sign slip in this file masquerade as a solver
# disagreement, which is exactly what this cross-check exists to detect.
from vdcore.analysis.axle import (
    CornerSample,
    _line_intersection,
    _rot2,
    axle_rates,
    axle_roll,
    sample_corner,
)
from vdcore.analysis.roll_centre import roll_centre_height
from vdcore.geometry.derived import mechanical_trail_mm, scrub_radius_mm
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Axle, Corner, DerivedPoint

__all__ = [
    "AltairUnavailableError",
    "AxleAltairResult",
    "MotionSolveReplay",
    "altair_kpis_from_csv",
    "sweep_intervals",
]

# Travel lookups are keyed to the micrometre. MotionSolve echoes the travel we
# asked for (it is our own ramp expression evaluated at the output time), so an
# exact grid hit lands well inside this; anything outside it is a programming
# error in the grid choice, not a tolerance to be widened.
_TRAVEL_KEY_TOL_MM = 1e-6

# Roll half-amplitude for the roll-camber central difference. Mirrors
# ``legacy_app.analysis.vdcore_bridge._ROLL_CAMBER_PROBE_DEG`` so the Altair and
# vdcore roll-camber numbers are the same measurement.
_ROLL_CAMBER_PROBE_DEG = 2.0

# MotionSolve's KINEMATICS accuracy depends on the OUTPUT STEP SPACING, not
# just on the endpoints: each step starts its constraint solve from the
# previous one, so a coarse sweep never gets close to the true assembly.
# Measured on the 2027 FL corner over +/-25 mm against DWSolver, worst error
# over the sweep:
#
#     spacing    d(camber)    d(toe)
#     25.000 mm   1.6e-04     7.7e-04   deg
#     12.500 mm   1.3e-04     5.9e-04
#      6.250 mm   3.2e-05     1.6e-04
#      2.500 mm   1.1e-06     6.2e-06
#      1.250 mm   2.8e-08     7.4e-08   <- travel/20, what we use
#      0.625 mm   3.8e-07     1.7e-06
#
# Note it is NOT monotonic: 0.625 mm is worse than 1.250 mm, so this is a
# step-control interaction rather than a convergence trend, and cranking the
# step count up is not a free accuracy win. Coarse is reliably bad, though, so
# every run here -- including the short roll runs, which would otherwise sit at
# the 16-25 mm spacing at the top of that table -- is held at travel/20. That
# keeps the Altair column free of step-size artefacts a reader could mistake
# for a geometry disagreement.
_MAX_GRID_SPACING_DIVISOR = 20.0


def sweep_intervals(travel_mm: float, sweep_steps: int) -> int:
    """Number of MotionSolve intervals whose grid contains every needed travel.

    ``axle_rates`` samples ``linspace(-travel, +travel, sweep_steps)`` for the
    RC extremes and a central difference at ``+/- travel/20``. A uniform grid of
    ``n`` intervals over ``[-travel, +travel]`` has spacing ``2*travel/n``, so it
    contains both sets when ``n`` is a common multiple of ``sweep_steps - 1``
    (for the linspace) and ``40`` (for the ``travel/20`` step).

    With the defaults (``travel_mm=25``, ``sweep_steps=41``) this is 40 -- the
    two requirements coincide, and one 41-point sweep serves everything, at
    1.25 mm spacing.
    """
    if sweep_steps < 2:
        raise ValueError("sweep_steps must be at least 2")
    return int(np.lcm(sweep_steps - 1, 40))


def roll_intervals(roll_travel_mm: float, travel_mm: float) -> int:
    """Interval count for a short roll run of half-amplitude ``roll_travel_mm``.

    The run must solve exactly ``{-t, 0, +t}``, so the count is even and the
    endpoints are exact. It is then refined until the spacing is no coarser
    than the main sweep's (``travel_mm / 20``), because MotionSolve's answer
    depends on the spacing -- see :data:`_MAX_GRID_SPACING_DIVISOR`.
    """
    target = travel_mm / _MAX_GRID_SPACING_DIVISOR
    half_steps = max(1, math.ceil(abs(roll_travel_mm) / target))
    return 2 * half_steps


@dataclass
class MotionSolveReplay:
    """A :class:`vdcore.geometry.solver.CornerSolver` backed by solved MotionSolve states.

    Holds one :class:`SolverResult` per travel MotionSolve actually solved, in
    DWSolver's frame convention (chassis displaced by ``-wheel_travel_mm``, so
    the wheel centre stays near its static height). ``solve`` is a lookup, never
    an interpolation: asking for a travel that was not solved raises.
    """

    corner_id: str
    results: dict[int, SolverResult] = field(default_factory=dict)

    @staticmethod
    def _key(travel_mm: float) -> int:
        return int(round(travel_mm / _TRAVEL_KEY_TOL_MM))

    def add(self, travel_mm: float, result: SolverResult) -> None:
        self.results[self._key(travel_mm)] = result

    def travels_mm(self) -> list[float]:
        return sorted(k * _TRAVEL_KEY_TOL_MM for k in self.results)

    def solve(
        self,
        wheel_travel_mm: float = 0.0,
        roll_deg: float = 0.0,
        rack_mm: float = 0.0,
    ) -> SolverResult:
        """Return MotionSolve's solved state at this travel.

        ``roll_deg`` and ``rack_mm`` must be zero: ``msolve_corner.py`` grounds
        the chassis and the inner tie rod, so it models neither. Asking for them
        raises rather than silently returning the unrolled, unsteered answer.
        """
        if abs(roll_deg) > 1e-12:
            raise NotImplementedError(
                "MotionSolve corner model has a grounded chassis -- it cannot "
                "apply chassis roll. Roll is handled by combining per-corner "
                "travels; see kpi_runner._roll_state_from_samples."
            )
        if abs(rack_mm) > 1e-12:
            raise NotImplementedError(
                "MotionSolve corner model grounds the inner tie rod -- it has "
                "no rack travel, so steering KPIs are not available from it."
            )
        try:
            return self.results[self._key(wheel_travel_mm)]
        except KeyError:
            raise KeyError(
                f"{self.corner_id}: MotionSolve has no solved state at "
                f"{wheel_travel_mm:+.6f} mm travel. Solved travels: "
                f"{[f'{t:+.3f}' for t in self.travels_mm()]}. The travel grid "
                f"must contain every travel vdcore asks for -- see "
                f"kpi_runner.sweep_intervals."
            ) from None


def _result_from_msolve_row(
    corner: Corner, solver: DWSolver, row: dict[str, float], travel_mm: float
) -> SolverResult:
    """Turn one MotionSolve output row into a DWSolver-framed ``SolverResult``.

    Two conversions happen here:

    1. **Frame.** MotionSolve grounds the chassis and lifts the wheel by
       ``travel``; DWSolver holds the wheel and drops the chassis. The two
       differ by a rigid vertical translation, so ``travel`` is subtracted from
       every point to express MotionSolve's answer the way DWSolver would.
       Angles are unaffected by a translation.
    2. **Contact patch.** MotionSolve tracks only rigid points on the upright;
       the patch is a *derived* point. It is reconstructed here exactly as
       ``DWSolver.solve`` does (loaded radius is the vertical drop, negative
       camber moves the patch outboard), so the roll-centre construction sees
       the same definition on both sides.

    The angles come from ``DWSolver._extract_angles`` on purpose: restating
    those formulas would test only whether they were copied correctly, and a
    sign slip in a second copy would masquerade as a solver disagreement.
    """
    shift = np.array([0.0, 0.0, travel_mm])
    ubj = np.array([row["UCA_OUT_x"], row["UCA_OUT_y"], row["UCA_OUT_z"]]) - shift
    lbj = np.array([row["LCA_OUT_x"], row["LCA_OUT_y"], row["LCA_OUT_z"]]) - shift
    tro = np.array([row["TIE_ROD_OUT_x"], row["TIE_ROD_OUT_y"], row["TIE_ROD_OUT_z"]]) - shift
    wc = np.array([
        row["WHEEL_CENTER_x"], row["WHEEL_CENTER_y"], row["WHEEL_CENTER_z"],
    ]) - shift

    # The spin probe is offset from the wheel centre along the static spin axis
    # and rides the upright, so probe - WC is the spin axis at every step. The
    # frame shift cancels in the difference.
    spin = np.array([row["SPIN_x"], row["SPIN_y"], row["SPIN_z"]]) - shift - wc
    spin = spin / float(np.linalg.norm(spin))

    camber_deg, toe_deg, caster_deg, kpi_deg = solver._extract_angles(ubj, lbj, spin)

    # Contact patch, mirroring vdcore.geometry.solver.DWSolver.solve. Keep the
    # two in step: the patch feeds scrub, trail and the roll centre.
    gamma = math.radians(camber_deg)
    r = corner.tire.loaded_radius_mm
    lateral_shift = -r * math.tan(gamma)
    is_left = corner.corner_id in ("FL", "RL")
    cp_y = wc[1] + lateral_shift if is_left else wc[1] - lateral_shift
    cp_z = wc[2] - r

    def point(v: np.ndarray) -> DerivedPoint:
        return DerivedPoint(x_mm=float(v[0]), y_mm=float(v[1]), z_mm=float(v[2]))

    return SolverResult(
        ubj=point(ubj),
        lbj=point(lbj),
        tro=point(tro),
        wheel_center=point(wc),
        contact_patch=DerivedPoint(x_mm=float(wc[0]), y_mm=float(cp_y), z_mm=float(cp_z)),
        camber_deg=camber_deg,
        toe_deg_per_side=toe_deg,
        caster_deg=caster_deg,
        kpi_deg=kpi_deg,
        # MotionSolve reports its own convergence by exiting non-zero and by
        # "Kinematic DoF: 0"; a row that reached this CSV was solved.
        converged=True,
        residual_norm=0.0,
        nfev=0,
        njev=0,
    )


def _replay_for_corner(
    corner: Corner,
    csv_path: Path,
    droop_mm: float,
    bump_mm: float,
    intervals: int,
    workdir: Path,
) -> MotionSolveReplay:
    """Run one MotionSolve sweep and wrap it as a replay solver."""
    solver = DWSolver(corner)
    # vdcore's own static spin axis, so both sides share one definition of the
    # wheel plane. Reaching in for the definition (not the answer) is deliberate.
    spin_static = solver._reconstruct_spin_axis(
        solver._ubj_0, solver._lbj_0, solver._tro_0
    )
    rows = run_motionsolve(
        csv_path, corner.corner_id, spin_static,
        droop_mm, bump_mm, intervals, workdir,
    )
    replay = MotionSolveReplay(corner_id=corner.corner_id)
    for row in rows:
        travel = row["travel_mm"]
        replay.add(travel, _result_from_msolve_row(corner, solver, row, travel))
    return replay


def _roll_state_from_samples(
    outer: CornerSample, inner: CornerSample, roll_deg: float
) -> tuple[float, float, float, float]:
    """Roll-centre and cambers from two already-solved corner samples.

    Mirrors the post-root-find half of ``vdcore.analysis.axle.axle_roll``: tilt
    both sides into the road frame, put the outer patch back on the ground, and
    intersect the two contact-patch-to-instant-centre lines.

    Split out (rather than calling ``axle_roll``) because ``axle_roll`` finds
    the wheel travel by ``brentq``, and every probe of that search would cost a
    MotionSolve subprocess. ``tests/unit/test_altair_kpi_runner.py`` pins this
    against ``axle_roll`` on DWSolver samples so the two cannot drift apart.

    Returns ``(outer_camber_deg, inner_camber_deg, rc_height_mm, rc_lateral_mm)``,
    cambers road-relative and the roll centre ground-referenced.
    """
    phi = math.radians(roll_deg)
    rot = _rot2(-phi)
    cp_o = rot @ np.array([outer.cp_y_mm, outer.cp_z_mm])
    ic_o = rot @ np.array([outer.fvic_y_mm, outer.fvic_z_mm])
    cp_i = rot @ np.array([inner.cp_y_mm, inner.cp_z_mm])
    ic_i = rot @ np.array([inner.fvic_y_mm, inner.fvic_z_mm])

    shift = np.array([0.0, -cp_o[1]])
    cp_o, ic_o, cp_i, ic_i = cp_o + shift, ic_o + shift, cp_i + shift, ic_i + shift

    rc = _line_intersection(cp_o, ic_o, cp_i, ic_i)
    rc_y = float(rc[0]) if rc is not None else math.nan
    rc_z = float(rc[1]) if rc is not None else math.nan
    return (
        outer.camber_deg + roll_deg,
        inner.camber_deg - roll_deg,
        rc_z,
        rc_y,
    )


def _patch_residual_mm(
    outer: CornerSample, inner: CornerSample, roll_deg: float
) -> float:
    """How far the two contact patches are from meeting on the tilted road.

    Zero is what ``axle_roll``'s root search drives to. Reported so that using
    vdcore's roll travel for the Altair evaluation is a visible approximation.
    """
    rot = _rot2(-math.radians(roll_deg))
    cp_o = rot @ np.array([outer.cp_y_mm, outer.cp_z_mm])
    cp_i = rot @ np.array([inner.cp_y_mm, inner.cp_z_mm])
    return float(cp_o[1] - cp_i[1])


@dataclass(frozen=True)
class AxleAltairResult:
    """Altair-sourced KPIs for one axle, keyed as the app's setup sheet expects."""

    label: str
    values: dict[str, float]
    roll_patch_residual_mm: float
    solved_travels: int


def _axle_result(
    axle: Axle,
    label: str,
    csv_path: Path,
    workdir: Path,
    *,
    roll_deg: float,
    travel_mm: float,
    sweep_steps: int,
    roll_travels: dict[float, float],
) -> AxleAltairResult:
    """Every Altair KPI for one axle.

    ``roll_travels`` maps a roll angle to the wheel travel vdcore found for it;
    the Altair evaluation reuses those travels (see the module docstring).
    """
    intervals = sweep_intervals(travel_mm, sweep_steps)

    # The bump sweep covers axle_rates and the static row on each side.
    left = _replay_for_corner(
        axle.left, csv_path, travel_mm, travel_mm, intervals, workdir
    )
    right = _replay_for_corner(
        axle.right, csv_path, travel_mm, travel_mm, intervals, workdir
    )

    # Roll travels are arbitrary reals and will not sit on the sweep grid, so
    # each gets its own run over {-t .. +t}, whose even interval count puts
    # exact samples at -t, 0 and +t. Both signs are needed: the outer wheel
    # goes into bump, the inner into droop.
    for magnitude in sorted({abs(t) for t in roll_travels.values() if abs(t) > 1e-9}):
        steps = roll_intervals(magnitude, travel_mm)
        for corner, replay in ((axle.left, left), (axle.right, right)):
            extra = _replay_for_corner(
                corner, csv_path, magnitude, magnitude, steps, workdir
            )
            for travel in extra.travels_mm():
                replay.add(travel, extra.solve(wheel_travel_mm=travel))

    values: dict[str, float] = {}

    # ---- static rows (travel 0) ------------------------------------------- #
    static_l = left.solve(wheel_travel_mm=0.0)
    static_r = right.solve(wheel_travel_mm=0.0)
    values["camber_l"] = static_l.camber_deg
    values["camber_r"] = static_r.camber_deg
    values["caster_l"] = static_l.caster_deg
    values["caster_r"] = static_r.caster_deg
    values["kpi_l"] = static_l.kpi_deg
    values["kpi_r"] = static_r.kpi_deg
    values["sum_toe"] = static_l.toe_deg_per_side + static_r.toe_deg_per_side
    for side, res in (("l", static_l), ("r", static_r)):
        try:
            values[f"scrub_{side}"] = scrub_radius_mm(res)
        except ValueError:
            values[f"scrub_{side}"] = math.nan
        try:
            values[f"trail_{side}"] = mechanical_trail_mm(res)
        except ValueError:
            values[f"trail_{side}"] = math.nan
    try:
        values["rc_static"] = roll_centre_height(axle, static_l, static_r).rc_height_mm
    except (RuntimeError, ValueError):
        values["rc_static"] = math.nan

    # ---- bump-sweep rates, through vdcore's own axle_rates ----------------- #
    # axle_rates only ever solves axle.left, so a factory that always hands
    # back the left replay is exact rather than a convenience.
    def factory_left(corner: Corner) -> MotionSolveReplay:
        return left

    rates = axle_rates(
        axle,
        travel_bump_mm=travel_mm,
        travel_droop_mm=travel_mm,
        sweep_steps=sweep_steps,
        solver_factory=factory_left,
    )
    values["camber_gain"] = rates.camber_gain_deg_per_mm
    values["ride_camber_dpm"] = rates.camber_gain_deg_per_mm * 1000.0
    values["rc_migration_mm_per_mm"] = rates.rc_migration_mm_per_mm
    values["half_track_change_mm_per_mm"] = rates.half_track_change_mm_per_mm
    values["camber_full_bump_deg"] = rates.camber_full_bump_deg
    values["camber_full_droop_deg"] = rates.camber_full_droop_deg
    values["rc_dz"] = rates.rc_max_mm - rates.rc_min_mm
    # Lateral RC migration under PARALLEL travel is zero for a symmetric axle;
    # matching vdcore_bridge, which reports it as such rather than as drift.
    values["rc_dy"] = 0.0

    # ---- roll ------------------------------------------------------------- #
    def roll_state(angle: float) -> tuple[float, float, float, float, float]:
        travel = roll_travels[angle]
        outer = sample_corner(axle.left, left, +travel)
        inner = sample_corner(axle.right, right, -travel)
        o_cam, i_cam, rc_z, rc_y = _roll_state_from_samples(outer, inner, angle)
        return o_cam, i_cam, rc_z, rc_y, _patch_residual_mm(outer, inner, angle)

    o_cam, i_cam, rc_z, rc_y, residual = roll_state(roll_deg)
    values["rc_1g_z"] = rc_z
    values["rc_1g_y"] = rc_y
    values["outer_camber_deg"] = o_cam
    values["inner_camber_deg"] = i_cam

    # Roll camber: chassis-referenced central difference, mirroring
    # vdcore_bridge.roll_camber_deg_per_deg (which subtracts the road tilt back
    # out of axle_roll's road-relative camber).
    hi_o, _, _, _, res_hi = roll_state(+_ROLL_CAMBER_PROBE_DEG)
    lo_o, _, _, _, res_lo = roll_state(-_ROLL_CAMBER_PROBE_DEG)
    values["roll_camber"] = (
        (hi_o - _ROLL_CAMBER_PROBE_DEG) - (lo_o + _ROLL_CAMBER_PROBE_DEG)
    ) / (2.0 * _ROLL_CAMBER_PROBE_DEG)

    worst_residual = max(abs(residual), abs(res_hi), abs(res_lo))
    return AxleAltairResult(
        label=label,
        values=values,
        roll_patch_residual_mm=worst_residual,
        solved_travels=len(left.results) + len(right.results),
    )


def altair_kpis_from_csv(
    csv_path: Path,
    *,
    roll_deg: float = 1.5,
    travel_mm: float = 25.0,
    sweep_steps: int = 41,
    static_toe_deg: float = 0.0,
    static_camber_deg: float | None = None,
    loaded_radius_mm: float | None = None,
    workdir: Path | None = None,
) -> dict[str, AxleAltairResult]:
    """Run MotionSolve over a hardpoint CSV and return the app's KPI keys.

    Args:
        static_camber_deg: Static camber to build into the upright. When None,
            it is recovered from the CSV's ``CONTACT_PATCH`` row.
        loaded_radius_mm: Tyre loaded radius. When None, likewise from the CSV.

    Both overrides exist so this column can be built from the SAME design
    inputs as the vdcore column it sits beside. Static camber is a design
    variable the hardpoint file only records indirectly, via a contact patch
    rounded to the file's precision: on the 2027 CSV that rounding reads back
    as -1.499938 deg rather than -1.500000, which is a difference of INPUT, not
    of solver, and would otherwise show up as a fake 1.6e-3 deg disagreement in
    the toe row. Pass the app's value here and the two columns differ only in
    the kinematics.

    Returns ``{"front": AxleAltairResult, "rear": AxleAltairResult}``.

    Raises:
        AltairUnavailableError: Altair is not installed on this machine.
        RuntimeError: a MotionSolve run failed.
    """
    csv_path = Path(csv_path)
    points = read_csv_points(csv_path)

    axles = {
        "front": ("Front", "FL", "FR"),
        "rear": ("Rear", "RL", "RR"),
    }
    roll_angles = (roll_deg, +_ROLL_CAMBER_PROBE_DEG, -_ROLL_CAMBER_PROBE_DEG)

    out: dict[str, AxleAltairResult] = {}
    tmp = tempfile.TemporaryDirectory(prefix="altair_kpis_") if workdir is None else None
    work = Path(tmp.name) if tmp is not None else Path(workdir)  # type: ignore[arg-type]
    try:
        for key, (label, left_id, right_id) in axles.items():
            axle = Axle(
                left=build_corner(
                    left_id, points[left_id], static_toe_deg,
                    static_camber_deg=static_camber_deg,
                    loaded_radius_mm=loaded_radius_mm,
                ),
                right=build_corner(
                    right_id, points[right_id], static_toe_deg,
                    static_camber_deg=static_camber_deg,
                    loaded_radius_mm=loaded_radius_mm,
                ),
            )
            # vdcore supplies the roll travel; Altair is evaluated there. See
            # the module docstring for why, and roll_patch_residual_mm for the
            # size of the approximation.
            roll_travels = {
                angle: axle_roll(axle, angle).wheel_travel_mm
                for angle in roll_angles
            }
            out[key] = _axle_result(
                axle, label, csv_path, work,
                roll_deg=roll_deg, travel_mm=travel_mm,
                sweep_steps=sweep_steps, roll_travels=roll_travels,
            )
    finally:
        if tmp is not None:
            tmp.cleanup()
    return out
