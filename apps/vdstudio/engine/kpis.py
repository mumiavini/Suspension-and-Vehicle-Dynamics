"""Every number the app quotes, with its unit, source, band and failure state.

A :class:`KPI` is a value plus everything needed to defend it: which code
produced it, which design-intent band (if any) it is compared against and
where that band came from, and -- if the computation failed -- why. A failed
KPI carries ``NaN`` and an error string; it is never replaced by a
plausible-looking number (CLAUDE.md, numerical conventions).

Bands are INFORMATION, not verdicts. "Outside band" means "outside the
design-intent window recorded in CheckLimits / SteeringLimits", which the
designer may have reasons to leave. Nothing here picks a geometry.

Two tiers:

* :func:`compute_static` -- one ``DWSolver`` solve per corner plus closed-form
  geometry. ~30 ms; runs on every edit.
* :func:`compute_dynamic` -- sweeps (rates, roll, bump steer, steering,
  Ackermann, rod-end envelope). ~2-3 s; runs in a background worker. Groups
  can be requested individually so the trade-off engine only pays for what it
  reports.

Frames: ISO 8855 (X+ forward, Y+ LEFT, Z+ up). Rate and roll Z values are
CHASSIS-referenced (vdcore.analysis.axle).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

import geometry_summary as gs
import numpy as np
from scipy.optimize import brentq

from vdcore.analysis.axle import AxleRollState, axle_rates, axle_roll
from vdcore.analysis.roll_centre import (
    front_view_instant_centre,
    roll_centre_height,
)
from vdcore.analysis.toe import ToeSweepResult, toe_sweep
from vdcore.geometry.derived import mechanical_trail_mm, scrub_radius_mm
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Axle, Corner, DerivedPoint

from .geometry import AXLE_OF, CarGeometry

GRAVITY = 9.81
FSAE_MIN_WHEELBASE_MM = gs.FSAE_MIN_WHEELBASE_MM
FSAE_MIN_TRACK_RATIO = gs.FSAE_MIN_TRACK_RATIO
ROLL_CAMBER_PROBE_DEG = 2.0

Band = tuple[float, float]
Progress = Callable[[float, str], None]
Cancelled = Callable[[], bool]


@dataclass(frozen=True)
class KPI:
    key: str
    label: str
    group: str
    scope: str            # "vehicle", "front", "rear", "steering"
    value: float
    unit: str = ""
    fmt: str = ".3f"
    source: str = ""
    band: Band | None = None
    band_source: str = ""
    note: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return (not self.error and isinstance(self.value, (int, float))
                and math.isfinite(self.value))

    @property
    def status(self) -> str:
        """``fail`` / ``in`` / ``out`` / ``info`` -- information, not a verdict."""
        if not self.ok:
            return "fail"
        if self.band is None:
            return "info"
        lo, hi = self.band
        return "in" if lo - 1e-9 <= self.value <= hi + 1e-9 else "out"

    def text(self) -> str:
        if not self.ok:
            return "—"
        s = format(self.value, self.fmt)
        # "-0.000" reads as a sign error; it is a rounding artefact.
        if s.startswith("-") and not any(ch in "123456789" for ch in s):
            s = s[1:]
        return s

    def band_text(self) -> str:
        if self.band is None:
            return ""
        lo, hi = self.band
        if math.isinf(hi):
            return f"≥ {lo:g}"
        if math.isinf(lo):
            return f"≤ {hi:g}"
        return f"{lo:g} … {hi:g}"


@dataclass
class KpiSet:
    items: dict[str, KPI] = field(default_factory=dict)

    def add(self, k: KPI) -> None:
        self.items[k.key] = k

    def get(self, key: str) -> KPI | None:
        return self.items.get(key)

    def value(self, key: str) -> float:
        k = self.items.get(key)
        return k.value if k is not None and k.ok else float("nan")

    def merge(self, other: KpiSet) -> KpiSet:
        out = KpiSet(dict(self.items))
        out.items.update(other.items)
        return out

    def groups(self) -> list[str]:
        return list(dict.fromkeys(k.group for k in self.items.values()))

    def by_group(self, group: str) -> list[KPI]:
        return [k for k in self.items.values() if k.group == group]

    def counts(self) -> dict[str, int]:
        out = {"in": 0, "out": 0, "info": 0, "fail": 0}
        for k in self.items.values():
            out[k.status] += 1
        return out

    def rows(self) -> list[dict[str, str]]:
        return [{
            "key": k.key, "group": k.group, "scope": k.scope, "label": k.label,
            "value": k.text(), "unit": k.unit, "band": k.band_text(),
            "status": k.status, "source": k.source, "band_source": k.band_source,
            "note": k.note, "error": k.error,
        } for k in self.items.values()]


class _Builder:
    """Collects KPIs; each ``guard`` turns an exception into failed KPIs."""

    def __init__(self) -> None:
        self.out = KpiSet()

    def add(self, key: str, label: str, group: str, scope: str, value: float, unit: str = "",
            fmt: str = ".3f", source: str = "", band: Band | None = None,
            band_source: str = "", note: str = "", error: str = "") -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        if not error and not math.isfinite(v):
            error = "not finite (degenerate geometry or failed solve)"
        self.out.add(KPI(key, label, group, scope, v, unit, fmt, source, band,
                         band_source, note, error))

    def fail(self, specs: Iterable[tuple[str, str, str, str, str]], error: str) -> None:
        for key, label, group, scope, unit in specs:
            self.out.add(KPI(key, label, group, scope, float("nan"), unit, error=error))


_SLA = "sla_geometry.CheckLimits (design_intent)"
_STG = "steering_geometry.SteeringLimits (design_intent)"
_RULES = "FSAE rules"
_DW = "vdcore DWSolver"


# --------------------------------------------------------------------------- #
# Static helpers
# --------------------------------------------------------------------------- #

def solve_static(geo: CarGeometry) -> dict[str, SolverResult]:
    out = {}
    for cid in ("FL", "FR", "RL", "RR"):
        out[cid] = DWSolver(geo.corner(cid)).solve()
    return out


def _effective_pivot(front: np.ndarray, rear: np.ndarray, x: float) -> np.ndarray:
    axis = rear - front
    if abs(axis[0]) < 1e-10:
        return 0.5 * (front + rear)
    return front + (x - front[0]) / axis[0] * axis


def front_view_arm_length(geo: CarGeometry, cid: str, arm: str) -> float:
    """Front-view (Y-Z) arm length: ball joint to the pivot axis at the BJ's X.

    This is the projection the FVSA construction uses -- not a member length.
    Reduces to ``sla_geometry``'s ``lca_length_mm`` / ``uca_length_mm`` for an
    unswept arm.
    """
    bj = geo.p(cid, f"{arm}_OUT")
    piv = _effective_pivot(geo.p(cid, f"{arm}_IN_FRONT"), geo.p(cid, f"{arm}_IN_REAR"), bj[0])
    return float(math.hypot(bj[1] - piv[1], bj[2] - piv[2]))


def side_view_anti(geo: CarGeometry, axle: str, brake_bias_front: float,
                   cg_height_mm: float, wheelbase_mm: float) -> tuple[float, float, float]:
    """Anti-dive (front) / anti-squat (rear) %, plus SVIC (x, z) in ISO.

    Pivot-axis-rake construction: an arm rotating about an inboard axis of
    side-view slope m moves its ball joint with velocity slope -m, so the
    instant centre lies on the line of slope m through the BALL JOINT. This is
    ``sla_geometry._side_view_anti`` / ``legacy model_3d.anti_dive_percent``,
    pinned against the 3D linkage in tests/benchmarks.
    Parallel axes: SVIC at infinity, tan(theta) -> shared slope (not zero).
    Returns (anti_pct, svic_x, svic_z); SVIC is NaN when at infinity.
    """
    cid = "FL" if axle == "front" else "RL"

    def slope(arm: str) -> float | None:
        f, r = geo.p(cid, f"{arm}_IN_FRONT"), geo.p(cid, f"{arm}_IN_REAR")
        run = r[0] - f[0]
        if abs(run) < 1e-9:
            return None
        return float((r[2] - f[2]) / run)

    m_u, m_l = slope("UCA"), slope("LCA")
    if m_u is None or m_l is None or cg_height_mm <= 0:
        return float("nan"), float("nan"), float("nan")
    ubj, lbj = geo.p(cid, "UCA_OUT"), geo.p(cid, "LCA_OUT")
    cp_x = geo.p(cid, "CONTACT_PATCH")[0]
    if abs(m_u - m_l) < 1e-15:
        tan_theta = m_l
        sx = sz = float("nan")
    else:
        sx = ((ubj[2] - lbj[2]) + m_l * lbj[0] - m_u * ubj[0]) / (m_l - m_u)
        sz = lbj[2] + m_l * (sx - lbj[0])
        dx = sx - cp_x
        if abs(dx) < 1e-9:
            return 0.0, float(sx), float(sz)
        tan_theta = sz / dx
    if axle == "front":
        # A front IC must sit BEHIND the axle (dx < 0) to resist dive.
        tan_theta = -tan_theta + 0.0
        return float(brake_bias_front * tan_theta * wheelbase_mm / cg_height_mm * 100.0) + 0.0, \
            float(sx), float(sz)
    return float(tan_theta * wheelbase_mm / cg_height_mm * 100.0) + 0.0, float(sx), float(sz)


def steering_arm_length(geo: CarGeometry, cid: str) -> float:
    """Perpendicular distance from the outer tie-rod joint to the kingpin axis."""
    ubj, lbj, tro = geo.p(cid, "UCA_OUT"), geo.p(cid, "LCA_OUT"), geo.p(cid, "TIE_ROD_OUT")
    kp = ubj - lbj
    n = float(np.linalg.norm(kp))
    if n < 1e-12:
        return float("nan")
    u = kp / n
    v = tro - lbj
    return float(np.linalg.norm(v - np.dot(v, u) * u))


def member_lengths(geo: CarGeometry, cid: str) -> dict[str, float]:
    p = geo.p
    lbj, ubj = p(cid, "LCA_OUT"), p(cid, "UCA_OUT")
    d = ubj - lbj
    lca_mid = 0.5 * (p(cid, "LCA_IN_FRONT")[0] + p(cid, "LCA_IN_REAR")[0])
    uca_mid = 0.5 * (p(cid, "UCA_IN_FRONT")[0] + p(cid, "UCA_IN_REAR")[0])
    lca_base = abs(float(p(cid, "LCA_IN_FRONT")[0] - p(cid, "LCA_IN_REAR")[0]))
    uca_base = abs(float(p(cid, "UCA_IN_FRONT")[0] - p(cid, "UCA_IN_REAR")[0]))
    lca_sweep = abs(float(lbj[0] - lca_mid))
    uca_sweep = abs(float(ubj[0] - uca_mid))
    return {
        "lca_front_leg": float(np.linalg.norm(lbj - p(cid, "LCA_IN_FRONT"))),
        "lca_rear_leg": float(np.linalg.norm(lbj - p(cid, "LCA_IN_REAR"))),
        "uca_front_leg": float(np.linalg.norm(ubj - p(cid, "UCA_IN_FRONT"))),
        "uca_rear_leg": float(np.linalg.norm(ubj - p(cid, "UCA_IN_REAR"))),
        "tie_rod": float(np.linalg.norm(p(cid, "TIE_ROD_OUT") - p(cid, "TIE_ROD_IN"))),
        "kingpin_3d": float(np.linalg.norm(d)),
        "kingpin_2d": float(math.hypot(d[1], d[2])),
        "lca_base": lca_base, "uca_base": uca_base,
        "lca_sweep": lca_sweep, "uca_sweep": uca_sweep,
        "lca_ea": lca_sweep / (lca_base / 2.0) if lca_base > 1e-9 else float("inf"),
        "uca_ea": uca_sweep / (uca_base / 2.0) if uca_base > 1e-9 else float("inf"),
    }


# --------------------------------------------------------------------------- #
# Static tier
# --------------------------------------------------------------------------- #

def compute_static(geo: CarGeometry) -> KpiSet:
    b = _Builder()
    veh = geo.vehicle_data
    static = solve_static(geo)

    # ---- vehicle -----------------------------------------------------------
    wb = geo.wheelbase_mm
    tf, tr = geo.track_mm("front"), geo.track_mm("rear")
    narrow, wide = min(tf, tr), max(tf, tr)
    g = "Vehicle & rules"
    b.add("vehicle.wheelbase", "Wheelbase", g, "vehicle", wb, "mm", ".1f", "hardpoints",
          (FSAE_MIN_WHEELBASE_MM, math.inf), _RULES)
    b.add("vehicle.track_front", "Front track (contact patches)", g, "vehicle", tf, "mm", ".1f",
          "hardpoints + camber")
    b.add("vehicle.track_rear", "Rear track (contact patches)", g, "vehicle", tr, "mm", ".1f",
          "hardpoints + camber")
    b.add("vehicle.track_ratio", "Narrow / wide track", g, "vehicle", 100.0 * narrow / wide, "%",
          ".1f", "hardpoints", (100.0 * FSAE_MIN_TRACK_RATIO, 100.0), _RULES)
    b.add("vehicle.track_wheelbase", "Mean track / wheelbase", g, "vehicle",
          0.5 * (tf + tr) / wb, "-", ".3f", "hardpoints",
          note="Concept heuristic 0.75-0.80 (random_calcs.ipynb) -- information only.")
    tilt_min = 2.0 * veh.cg_height_mm * math.tan(math.radians(veh.tilt_test_angle_deg))
    b.add("vehicle.tilt_margin", f"{veh.tilt_test_angle_deg:.0f}° tilt: narrow track margin", g,
          "vehicle", narrow - tilt_min, "mm", ".1f", "2·h_cg·tan(tilt) vs narrow track",
          (0.0, math.inf), _RULES,
          note=f"needs {tilt_min:.0f} mm; narrow track is the {'front' if tf < tr else 'rear'}")
    b.add("vehicle.front_mass_fraction", "Static front mass fraction", g, "vehicle",
          100.0 * veh.front_mass_fraction, "%", ".1f", "vehicle data (design_intent)")

    # ---- per corner static ---------------------------------------------------
    for cid, res in static.items():
        axle = AXLE_OF[cid]
        side = "L" if cid in ("FL", "RL") else "R"
        grp = "Static alignment"
        lim = geo.axle_inputs(axle).limits
        specs = [(f"{axle}.{n}_{side}", f"{lab} ({side})", grp, axle, u) for n, lab, u in (
            ("camber", "Camber", "deg"), ("toe", "Toe per side", "deg"),
            ("caster", "Caster", "deg"), ("kpi", "KPI", "deg"),
            ("scrub", "Scrub radius", "mm"), ("trail", "Mechanical trail", "mm"))]
        if not res.converged:
            b.fail(specs, f"static solve did not converge (residual {res.residual_norm:.2e})")
            continue
        b.add(f"{axle}.camber_{side}", f"Camber ({side})", grp, axle, res.camber_deg, "deg",
              ".3f", _DW, note="negative = top inboard")
        b.add(f"{axle}.toe_{side}", f"Toe per side ({side})", grp, axle,
              res.toe_deg_per_side, "deg", ".3f", _DW, note="positive = toe-in")
        b.add(f"{axle}.caster_{side}", f"Caster ({side})", grp, axle, res.caster_deg, "deg",
              ".3f", _DW)
        b.add(f"{axle}.kpi_{side}", f"KPI ({side})", grp, axle, res.kpi_deg, "deg", ".3f", _DW,
              lim.kpi_deg, _SLA)
        try:
            b.add(f"{axle}.scrub_{side}", f"Scrub radius ({side})", grp, axle,
                  scrub_radius_mm(res), "mm", ".2f", _DW + " + derived",
                  lim.scrub_radius_mm, _SLA, note="positive = patch outboard of kingpin")
        except ValueError as exc:
            b.add(f"{axle}.scrub_{side}", f"Scrub radius ({side})", grp, axle, math.nan, "mm",
                  error=str(exc))
        trail_band = geo.steering_inputs.limits.mechanical_trail_mm if axle == "front" else None
        try:
            b.add(f"{axle}.trail_{side}", f"Mechanical trail ({side})", grp, axle,
                  mechanical_trail_mm(res), "mm", ".2f", _DW + " + derived",
                  trail_band, _STG if trail_band else "",
                  note="positive = kingpin ground point ahead of patch")
        except ValueError as exc:
            b.add(f"{axle}.trail_{side}", f"Mechanical trail ({side})", grp, axle, math.nan,
                  "mm", error=str(exc))

    for axle, (lc, rc) in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        l_res, r_res = static[lc], static[rc]
        inp = geo.axle_inputs(axle)
        lim = inp.limits
        if l_res.converged and r_res.converged:
            b.add(f"{axle}.total_toe", "Total toe (L + R)", "Static alignment", axle,
                  l_res.toe_deg_per_side + r_res.toe_deg_per_side, "deg", ".3f", _DW,
                  note="sum of per-side values, + = toe-in")
        _front_view_kpis(b, geo, axle, l_res, r_res)
        _member_kpis(b, geo, axle)
        # ball joints inside the rim
        r_rim = inp.rim_diameter_in * 25.4 / 2.0
        lo = geo.axles[axle].loaded_radius_mm - r_rim + lim.ball_joint_clearance_mm
        hi = geo.axles[axle].loaded_radius_mm + r_rim - lim.ball_joint_clearance_mm
        lbj_z, ubj_z = geo.p(lc, "LCA_OUT")[2], geo.p(lc, "UCA_OUT")[2]
        b.add(f"{axle}.lbj_z", "Lower ball joint height", "Packaging", axle, lbj_z, "mm", ".1f",
              "hardpoints", (lo, hi), f"rim window ± {lim.ball_joint_clearance_mm:g} mm clearance")
        b.add(f"{axle}.ubj_z", "Upper ball joint height", "Packaging", axle, ubj_z, "mm", ".1f",
              "hardpoints", (lo, hi), f"rim window ± {lim.ball_joint_clearance_mm:g} mm clearance")
        # anti-geometry
        bias = veh.brake_bias_front if axle == "front" else 1.0
        anti, sx, sz = side_view_anti(geo, axle, bias, veh.cg_height_mm, wb)
        at_inf = not (math.isfinite(sx) and math.isfinite(sz)) and math.isfinite(anti)
        note = (f"brake bias {veh.brake_bias_front:.2f}, h_cg {veh.cg_height_mm:.0f} mm"
                if axle == "front" else "all drive torque at the rear")
        if at_inf:
            note += "; pivot axes parallel in side view -> SVIC at infinity"
        b.add(f"{axle}.anti", "Anti-dive" if axle == "front" else "Anti-squat",
              "Anti-geometry", axle, anti, "%", ".2f",
              "pivot-axis rake construction (cross-checked vs 3D linkage)",
              lim.anti_percent, _SLA, note=note)
        if not at_inf:
            b.add(f"{axle}.svic_x", "Side-view IC x", "Anti-geometry", axle, sx, "mm", ".0f",
                  "pivot-axis rake construction")
            b.add(f"{axle}.svic_z", "Side-view IC z", "Anti-geometry", axle, sz, "mm", ".0f",
                  "pivot-axis rake construction")

    # steering arm / static steering
    for side, cid in (("L", "FL"), ("R", "FR")):
        b.add(f"steering.arm_{side}", f"Steering arm length ({side})", "Steering geometry",
              "steering", steering_arm_length(geo, cid), "mm", ".2f", "hardpoints",
              note="TRO to kingpin axis, perpendicular")
    tr_len = member_lengths(geo, "FL")["tie_rod"]
    b.add("steering.tie_rod", "Front tie rod length", "Steering geometry", "steering", tr_len,
          "mm", ".2f", "hardpoints", geo.steering_inputs.limits.tie_rod_length_mm, _STG)
    tri = geo.p("FL", "TIE_ROD_IN")
    b.add("steering.rack_z", "Rack height (tie-rod inboard z)", "Steering geometry", "steering",
          tri[2], "mm", ".2f", "hardpoints", geo.steering_inputs.limits.rack_z_window_mm, _STG)
    b.add("steering.rack_x", "Rack station (inboard x, rearward +)", "Steering geometry",
          "steering", -tri[0], "mm", ".2f", "hardpoints",
          geo.steering_inputs.limits.rack_x_window_mm, _STG,
          note="design frame: x positive REARWARD of the front axle")
    if geo.steering_geometry is not None:
        b.add("steering.geometric_ackermann", "Geometric Ackermann (plan-view construction)",
              "Steering geometry", "steering", geo.steering_geometry.geometric_ackermann_pct,
              "%", ".1f", "steering_geometry._geometric_ackermann_pct",
              note="small-angle construction; the solved value (Steering) is the rigorous one")

    # vehicle roll budget uses the SOLVED static roll centres
    rc_f, rc_r = b.out.value("front.rc_static"), b.out.value("rear.rc_static")
    _vehicle_roll_kpis(b, geo, rc_f, rc_r)
    return b.out


def _front_view_kpis(b: _Builder, geo: CarGeometry, axle: str, l_res: SolverResult,
                     r_res: SolverResult) -> None:
    lim = geo.axle_inputs(axle).limits
    g = "Front view"
    lc = "FL" if axle == "front" else "RL"
    ax = geo.axle(axle)
    if not (l_res.converged and r_res.converged):
        b.fail([(f"{axle}.rc_static", "Roll centre height (static)", g, axle, "mm")],
               "static solve did not converge")
        return
    try:
        rc = roll_centre_height(ax, l_res, r_res)
        b.add(f"{axle}.rc_static", "Roll centre height (static)", g, axle, rc.rc_height_mm, "mm",
              ".2f", _DW + " + FVIC construction", lim.rc_height_mm, _SLA)
        fv = rc.left_fvic
        fvsa = abs(fv.fvsa_mm)
        side = "inboard of the patch" if fv.fvsa_mm < 0 else "outboard of the patch"
        b.add(f"{axle}.fvsa", "FVSA length", g, axle, fvsa, "mm", ".1f",
              _DW + " + FVIC construction", lim.fvsa_length_mm, _SLA,
              note=f"magnitude; IC {side}")
        b.add(f"{axle}.fvic_y", "FVIC y (left corner)", g, axle, fv.fvic_y_mm, "mm", ".1f",
              "FVIC construction", note="negative = IC on the far side of the car")
        b.add(f"{axle}.fvic_z", "FVIC z (left corner)", g, axle, fv.fvic_z_mm, "mm", ".1f",
              "FVIC construction")
        b.add(f"{axle}.camber_gain_fvsa", "Camber gain from FVSA (57.3/FVSA)", g, axle,
              math.degrees(1.0) / fvsa, "deg/mm", ".4f", "closed form",
              lim.camber_gain_deg_per_mm, _SLA,
              note="textbook first-order rate magnitude; the solved, signed rate is in "
                   "Kinematic rates")
    except RuntimeError as exc:
        b.fail([(f"{axle}.rc_static", "Roll centre height (static)", g, axle, "mm"),
                (f"{axle}.fvsa", "FVSA length", g, axle, "mm")], str(exc))
    lca = front_view_arm_length(geo, lc, "LCA")
    uca = front_view_arm_length(geo, lc, "UCA")
    b.add(f"{axle}.lca_fv_length", "LCA front-view length", g, axle, lca, "mm", ".1f",
          "hardpoints", note="projection used by FVSA, not a member length")
    b.add(f"{axle}.uca_fv_length", "UCA front-view length", g, axle, uca, "mm", ".1f",
          "hardpoints")
    b.add(f"{axle}.uca_lca_ratio", "UCA / LCA (front view)", g, axle, uca / lca, "-", ".3f",
          "hardpoints", lim.uca_lca_ratio, _SLA)
    _ = front_view_instant_centre  # re-exported for curve code


def _member_kpis(b: _Builder, geo: CarGeometry, axle: str) -> None:
    lim = geo.axle_inputs(axle).limits
    lc = "FL" if axle == "front" else "RL"
    m = member_lengths(geo, lc)
    g = "Members (true 3D)"
    for key, lab in (("lca_front_leg", "LCA front leg"), ("lca_rear_leg", "LCA rear leg"),
                     ("uca_front_leg", "UCA front leg"), ("uca_rear_leg", "UCA rear leg")):
        b.add(f"{axle}.{key}", lab, g, axle, m[key], "mm", ".2f", "hardpoints",
              lim.lca_length_mm, _SLA)
    b.add(f"{axle}.tie_rod_len", "Tie rod / toe link", g, axle, m["tie_rod"], "mm", ".2f",
          "hardpoints")
    b.add(f"{axle}.kingpin_3d", "Kingpin length (true 3D)", g, axle, m["kingpin_3d"], "mm",
          ".2f", "hardpoints", lim.kingpin_length_mm, _SLA,
          note="the length that matters for the upright")
    b.add(f"{axle}.kingpin_2d", "Kingpin length (front view)", g, axle, m["kingpin_2d"], "mm",
          ".2f", "hardpoints", lim.kingpin_length_mm, _SLA)
    b.add(f"{axle}.lca_ea", "LCA sweep e/a", g, axle, m["lca_ea"], "-", ".3f", "hardpoints",
          (0.0, lim.ea_ratio_max), _SLA, note=f"base {m['lca_base']:.0f} / sweep "
          f"{m['lca_sweep']:.1f} mm (half-base reference)")
    b.add(f"{axle}.uca_ea", "UCA sweep e/a", g, axle, m["uca_ea"], "-", ".3f", "hardpoints",
          (0.0, lim.ea_ratio_max), _SLA, note=f"base {m['uca_base']:.0f} / sweep "
          f"{m['uca_sweep']:.1f} mm")


def _vehicle_roll_kpis(b: _Builder, geo: CarGeometry, rc_f: float, rc_r: float) -> None:
    veh = geo.vehicle_data
    g = "Roll budget"
    wb = geo.wheelbase_mm
    frac = veh.cg_from_front_axle_mm / veh.wheelbase_mm
    h_axis = rc_f + (rc_r - rc_f) * frac
    arm = veh.cg_height_mm - h_axis
    w_s = veh.sprung_mass_kg * GRAVITY
    grad = veh.target_roll_gradient_deg_per_g
    k_roll = w_s * (arm / 1000.0) / grad if grad > 0 else math.nan
    # Milliken Ch.16 with the pendulum term: phi/Ay = W d / (K - W d)
    grad_rad = math.radians(grad)
    k_pend = w_s * (arm / 1000.0) * (1.0 + 1.0 / grad_rad) * math.pi / 180.0 if grad > 0 \
        else math.nan
    b.add("vehicle.roll_axis_at_cg", "Roll axis height at CG", g, "vehicle", h_axis, "mm", ".1f",
          "solved static RCs, CG station")
    b.add("vehicle.roll_arm", "Roll moment arm", g, "vehicle", arm, "mm", ".1f",
          "h_cg − roll axis")
    b.add("vehicle.k_roll_required", "Required roll stiffness", g, "vehicle", k_roll, "N·m/deg",
          ".1f", "W_s·d / roll gradient (sla_geometry.solve_vehicle)",
          note=f"target roll gradient {grad:.2f} deg/g (design_intent)")
    b.add("vehicle.k_roll_required_pend", "Required roll stiffness (with pendulum term)", g,
          "vehicle", k_pend, "N·m/deg", ".1f", "W_s·d·(1 + 1/φ) (Milliken Ch.16)",
          note="includes the destabilising sprung-mass pendulum moment")
    b.add("vehicle.chassis_torsion_min", "Chassis torsion (min factor)", g, "vehicle",
          veh.chassis_factor_min * k_roll, "N·m/deg", ".0f",
          f"{veh.chassis_factor_min:g} × required roll stiffness")
    b.add("vehicle.chassis_torsion_target", "Chassis torsion (target factor)", g, "vehicle",
          veh.chassis_factor_target * k_roll, "N·m/deg", ".0f",
          f"{veh.chassis_factor_target:g} × required roll stiffness")
    W = veh.total_mass_kg * GRAVITY
    b.add("vehicle.fz_static_front", "Static load per front wheel", g, "vehicle",
          W * veh.front_mass_fraction / 2.0, "N", ".1f", "vehicle data")
    b.add("vehicle.fz_static_rear", "Static load per rear wheel", g, "vehicle",
          W * (1.0 - veh.front_mass_fraction) / 2.0, "N", ".1f", "vehicle data")
    _ = wb


# --------------------------------------------------------------------------- #
# Dynamic tier
# --------------------------------------------------------------------------- #

DYNAMIC_GROUPS: tuple[str, ...] = (
    "rates", "roll", "roll_camber", "bump_steer", "steer", "ackermann", "rod_end",
)

INSTANT_AXIS_STEP_MM = 0.05       # central difference for the upright's instantaneous twist
ASYMMETRY_TOL_MM = 1e-3           # |left - mirrored right| above this: evaluate both turns
EFFORT_STROKE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 0.95)
EFFORT_RACK_STEP_MM = 0.25

_RC3D = "instant screw axis of the upright (3D), DWSolver"
_PROJECTED = ("projected FVIC at each ball joint's X (vdcore); with caster the 3D "
              "instant-axis value differs, see rc_migration_3d")
_EFFORT_MODEL = ("lever-arm friction model μ·Fz·√(rs²+tm²), no patch-twist (bore) torque, "
                 "μ assumed, no TTC data")


def roll_camber_deg_per_deg(axle: Axle, probe_deg: float = ROLL_CAMBER_PROBE_DEG,
                            turn: str = "left") -> float:
    """Chassis-referenced roll-camber slope of the OUTER wheel, deg/deg.

    App convention: positive roll lifts the LEFT side (a left turn), so for
    ``turn="left"`` the outer wheel is the RIGHT one. ``turn="right"`` is the
    left wheel in a right turn -- the definition of
    ``legacy_app/analysis/vdcore_bridge.roll_camber_deg_per_deg``; the two agree
    on a symmetric axle. 0 = the wheel rolls with the body (no camber
    recovery), -1 = it stays upright to the road; road-referenced = 1 + value.
    """
    if turn == "left":
        # axle_roll(-phi) is ISO +phi; the RIGHT wheel is its "inner" corner
        hi, lo = axle_roll(axle, -probe_deg), axle_roll(axle, +probe_deg)
        return ((hi.inner_camber_deg + hi.roll_deg) - (lo.inner_camber_deg + lo.roll_deg)) \
            / (2.0 * probe_deg)
    hi, lo = axle_roll(axle, +probe_deg), axle_roll(axle, -probe_deg)
    return ((hi.outer_camber_deg - hi.roll_deg) - (lo.outer_camber_deg - lo.roll_deg)) \
        / (2.0 * probe_deg)


class RollState(NamedTuple):
    """An axle in roll in the app's turn convention, both wheels on the road."""

    outer_camber_deg: float       # road-referenced
    inner_camber_deg: float       # road-referenced
    rc_height_mm: float
    rc_lateral_mm: float          # ISO Y, + = LEFT, road frame
    outer_travel_mm: float        # outer wheel, + = bump


def turn_roll(axle: Axle, roll_deg: float, turn: str = "left") -> RollState:
    """``vdcore.analysis.axle.axle_roll`` mapped to a turn direction.

    ``axle_roll(phi)`` takes the LEFT wheel as outer and puts it in bump: that
    is ISO roll ``-phi`` (left side down), a RIGHT turn. The app's positive
    roll lifts the LEFT side (a left turn), so a left turn is
    ``axle_roll(-phi)`` with the roles swapped: outer = RIGHT wheel. RC lateral
    stays in ISO Y, so in a left turn + is toward the inner (left) wheel.
    """
    if turn == "left":
        s: AxleRollState = axle_roll(axle, -roll_deg)
        return RollState(s.inner_camber_deg, s.outer_camber_deg, s.rc_height_mm,
                         s.rc_lateral_mm, -s.wheel_travel_mm)
    s = axle_roll(axle, roll_deg)
    return RollState(s.outer_camber_deg, s.inner_camber_deg, s.rc_height_mm, s.rc_lateral_mm,
                     s.wheel_travel_mm)


def axle_asymmetry_mm(geo: CarGeometry, axle: str) -> float:
    """Largest |left − mirrored right| over the axle's static points, mm."""
    lc, rc = ("FL", "FR") if axle == "front" else ("RL", "RR")
    worst = 0.0
    for name, p in geo.points[lc].items():
        q = geo.points[rc].get(name)
        if q is None:
            return math.inf
        worst = max(worst, abs(p[0] - q[0]), abs(p[1] + q[1]), abs(p[2] - q[2]))
    return worst


def _skew(a: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def _in_chassis(p: DerivedPoint, travel: float) -> np.ndarray:
    # DWSolver drives travel by displacing the CHASSIS by -travel; add it back.
    return np.array([p.x_mm, p.y_mm, p.z_mm + travel])


def instant_axis_fvic(corner: Corner, solver: DWSolver, travel_mm: float = 0.0,
                      h: float = INSTANT_AXIS_STEP_MM) -> tuple[np.ndarray, np.ndarray]:
    """(contact patch, front-view IC), each (y, z) in the CHASSIS frame.

    The upright's instantaneous twist relative to the chassis comes from
    central differences (±h mm of travel) of UBJ, LBJ and TRO:
    v(p) = v_L + ω × (p − L), with ω fitted by least squares to v_U − v_L and
    v_T − v_L. The front-view IC is the point of the plane x = x_patch whose
    Y and Z velocities vanish: the 3D instant axis cut at the patch station.

    ``vdcore.analysis.roll_centre.front_view_instant_centre`` instead
    intersects each arm's projection through its own ball joint's X. The two
    agree when both ball joints share X or the upright does not pitch, and
    part with caster. vdcore is left unchanged; this is the app's 3D value.

    Raises:
        RuntimeError: a solve did not converge, or the instant axis has no
            roll component (front-view IC at infinity).
    """
    res: dict[float, SolverResult] = {}
    for d in (0.0, h, -h):
        r = solver.solve(wheel_travel_mm=travel_mm + d)
        if not r.converged:
            raise RuntimeError(f"{corner.corner_id} did not converge at {travel_mm + d:+.3f} mm "
                               f"travel (residual {r.residual_norm:.2e})")
        res[d] = r
    names = ("ubj", "lbj", "tro")
    p = {n: _in_chassis(getattr(res[0.0], n), travel_mm) for n in names}
    v = {n: (_in_chassis(getattr(res[h], n), travel_mm + h)
             - _in_chassis(getattr(res[-h], n), travel_mm - h)) / (2.0 * h) for n in names}
    # v_X − v_L = ω × (X − L) = −[X − L]× ω for X = UBJ, TRO
    a = np.vstack([-_skew(p["ubj"] - p["lbj"]), -_skew(p["tro"] - p["lbj"])])
    rhs = np.concatenate([v["ubj"] - v["lbj"], v["tro"] - v["lbj"]])
    w = np.linalg.lstsq(a, rhs, rcond=None)[0]          # rad per mm of travel
    if abs(w[0]) < 1e-9:
        raise RuntimeError(f"{corner.corner_id}: the upright's instant axis has no roll "
                           f"component at {travel_mm:+.3f} mm (front-view IC at infinity)")
    cp = _in_chassis(res[0.0].contact_patch, travel_mm)
    lbj, v_l = p["lbj"], v["lbj"]
    qx = cp[0] - lbj[0]
    ic = np.array([lbj[1] + (w[1] * qx - v_l[2]) / w[0], lbj[2] + (v_l[1] + w[2] * qx) / w[0]])
    return cp[1:], ic


def instant_axis_rc(axle: Axle, travel_mm: float = 0.0,
                    solvers: tuple[DWSolver, DWSolver] | None = None) -> tuple[float, float]:
    """(RC y, RC z) in the CHASSIS frame, both wheels at ``travel_mm``.

    The RC is where the left and right patch-to-IC lines cross, each IC from
    :func:`instant_axis_fvic`.
    """
    sl, sr = solvers if solvers is not None else (DWSolver(axle.left), DWSolver(axle.right))
    cl, il = instant_axis_fvic(axle.left, sl, travel_mm)
    cr, ir = instant_axis_fvic(axle.right, sr, travel_mm)
    d1, d2 = il - cl, ir - cr
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-12 * float(np.linalg.norm(d1) * np.linalg.norm(d2)):
        raise RuntimeError("patch-to-IC lines are parallel: roll centre at infinity")
    t = ((cr[0] - cl[0]) * d2[1] - (cr[1] - cl[1]) * d2[0]) / den
    rc = cl + t * d1
    return float(rc[0]), float(rc[1])


def _steer_pair(fl: DWSolver, fr: DWSolver, rack: float) -> tuple[SolverResult, SolverResult]:
    a, c = fl.solve(rack_mm=rack), fr.solve(rack_mm=rack)
    if not (a.converged and c.converged):
        raise RuntimeError(f"steer solve did not converge at rack {rack:+.2f} mm")
    return a, c


def steer_rate_sum(fl: DWSolver, fr: DWSolver, rack_mm: float,
                   h: float = EFFORT_RACK_STEP_MM) -> float:
    """|dδL/drack| + |dδR/drack| at ``rack_mm``, deg/mm (central difference ±h).

    By virtual work the rack force is M_kp·(this sum) for a kingpin friction
    moment M_kp per wheel; at straight ahead the sum is 2 / C-factor.
    """
    pl, pr = _steer_pair(fl, fr, rack_mm + h)
    ml, mr = _steer_pair(fl, fr, rack_mm - h)
    return (abs(pl.toe_deg_per_side - ml.toe_deg_per_side)
            + abs(pr.toe_deg_per_side - mr.toe_deg_per_side)) / (2.0 * h)


class AckermannSolve(NamedTuple):
    pct: float
    rack_mm: float
    track_mm: float               # static contact-patch track the reference used
    outer_deg: float
    inner_deg: float
    inner_ideal_deg: float


def solved_ackermann_pct(axle: Axle, wheelbase_mm: float, outer_steer_deg: float,
                         max_rack_mm: float) -> AckermannSolve:
    """Ackermann % from a real rack sweep on DWSolver, plus what it is measured against.

    Mirrors ``vdcore_bridge.solved_ackermann_pct`` (which reproduces
    ``steering_geometry._ackermann_at_steer`` to 3 decimals): the classical
    construction cot δo − cot δi = T/L with T = the STATIC contact-patch track
    and L = the static wheelbase. 0 % parallel, 100 % = that construction. The
    steered patches move, so another reference gives another %, and the
    denominator (δi,ideal − δo) is small; ``inner_deg − inner_ideal_deg`` is
    the less reference-sensitive reading. ``pct`` is NaN only for a
    degenerate construction.

    Raises:
        RuntimeError: the static or a steered solve did not converge.
        ValueError: the rack cannot reach the outer angle within its stroke.
    """
    left, right = DWSolver(axle.left), DWSolver(axle.right)
    l0, r0 = left.solve(), right.solve()
    if not (l0.converged and r0.converged):
        raise RuntimeError("static solve did not converge")
    track = abs(l0.contact_patch.y_mm - r0.contact_patch.y_mm)

    def steer(rack: float) -> tuple[float, float]:
        ls, rs = _steer_pair(left, right, rack)
        return (abs(ls.toe_deg_per_side - l0.toe_deg_per_side),
                abs(rs.toe_deg_per_side - r0.toe_deg_per_side))

    probe_l, probe_r = steer(1.0)
    left_is_outer = probe_l <= probe_r

    def outer_err(rack: float) -> float:
        sl, sr = steer(rack)
        return (sl if left_is_outer else sr) - outer_steer_deg

    lo, hi = 0.01, max_rack_mm
    e_lo, e_hi = outer_err(lo), outer_err(hi)
    if e_lo > 0.0:
        raise ValueError(f"outer {outer_steer_deg:g}° is below the steer at {lo:g} mm rack "
                         f"({e_lo + outer_steer_deg:.3f}°)")
    if e_hi < 0.0:
        raise ValueError(f"outer {outer_steer_deg:g}° not reachable within {max_rack_mm:g} mm "
                         f"rack (max outer {e_hi + outer_steer_deg:.2f}°)")
    rack = float(brentq(outer_err, lo, hi, xtol=1e-8))
    sl, sr = steer(rack)
    d_out = sl if left_is_outer else sr
    d_in = sr if left_is_outer else sl
    nan = float("nan")
    cot_ideal = 1.0 / math.tan(math.radians(d_out)) - track / wheelbase_mm
    if abs(cot_ideal) < 1e-12:
        return AckermannSolve(nan, rack, track, d_out, d_in, nan)
    d_ideal = math.degrees(math.atan(1.0 / cot_ideal))
    if abs(d_ideal - d_out) < 1e-12:
        return AckermannSolve(nan, rack, track, d_out, d_in, d_ideal)
    return AckermannSolve(float(100.0 * (d_in - d_out) / (d_ideal - d_out)), rack, track, d_out,
                          d_in, d_ideal)


def _toe_sweep_checked(corner: Corner, lo_mm: float, hi_mm: float,
                       steps: int = 21) -> ToeSweepResult:
    """``toe_sweep`` that raises instead of dropping non-converged points."""
    ts = toe_sweep(corner, lo_mm, hi_mm, steps)
    bad = [t for t, ok in zip(ts.wheel_travel_mm, ts.converged, strict=True) if not ok]
    if bad:
        raise RuntimeError(f"{len(bad)}/{steps} toe-sweep points did not converge "
                           f"(e.g. {bad[0]:+.1f} mm travel)")
    return ts


def compute_dynamic(geo: CarGeometry, groups: Iterable[str] | None = None,
                    progress: Progress | None = None,
                    cancelled: Cancelled | None = None,
                    static: KpiSet | None = None) -> KpiSet:
    """Swept KPIs. Raises ``InterruptedError`` if ``cancelled()`` turns true."""
    want = set(DYNAMIC_GROUPS if groups is None else groups)
    b = _Builder()
    steps = [g for g in DYNAMIC_GROUPS if g in want]
    total = max(len(steps) * 2, 1)
    done = 0

    def tick(msg: str) -> None:
        nonlocal done
        done += 1
        if cancelled is not None and cancelled():
            raise InterruptedError
        if progress is not None:
            progress(min(done / total, 1.0), msg)

    for axle in ("front", "rear"):
        ax = geo.axle(axle)
        inp = geo.axle_inputs(axle)
        lim = inp.limits
        lc = "FL" if axle == "front" else "RL"
        # vdcore's FVIC projects each arm through its own ball joint's X: exact only
        # when both share X (no caster) -- say so wherever that construction is quoted
        proj = _PROJECTED if abs(geo.p(lc, "UCA_OUT")[0] - geo.p(lc, "LCA_OUT")[0]) > 0.1 \
            else ""
        asym = axle_asymmetry_mm(geo, axle) > ASYMMETRY_TOL_MM
        if "rates" in want:
            g = "Kinematic rates"
            try:
                r = axle_rates(ax, travel_bump_mm=inp.travel_bump_mm,
                               travel_droop_mm=inp.travel_droop_mm)
                cg_band = (-lim.camber_gain_deg_per_mm[1], -lim.camber_gain_deg_per_mm[0])
                b.add(f"{axle}.camber_gain", "Camber gain (per mm bump)", g, axle,
                      r.camber_gain_deg_per_mm, "deg/mm", ".4f", "vdcore.analysis.axle",
                      cg_band, _SLA + " (sign: negative gain in bump)")
                b.add(f"{axle}.ride_camber", "Ride camber", g, axle,
                      r.camber_gain_deg_per_mm * 1000.0, "deg/m", ".2f", "vdcore.analysis.axle")
                b.add(f"{axle}.rc_migration", "RC migration (chassis-ref.)", g, axle,
                      r.rc_migration_mm_per_mm, "mm/mm", ".4f", "vdcore.analysis.axle",
                      note="ground-referenced differs by exactly 1 mm/mm"
                           + (f"; {proj}" if proj else ""))
                b.add(f"{axle}.half_track_change", "Half-track change", g, axle,
                      r.half_track_change_mm_per_mm, "mm/mm", ".4f", "vdcore.analysis.axle")
                b.add(f"{axle}.camber_full_bump", f"Camber at +{inp.travel_bump_mm:g} mm", g,
                      axle, r.camber_full_bump_deg, "deg", ".3f", "vdcore.analysis.axle")
                b.add(f"{axle}.camber_full_droop", f"Camber at −{inp.travel_droop_mm:g} mm", g,
                      axle, r.camber_full_droop_deg, "deg", ".3f", "vdcore.analysis.axle")
                b.add(f"{axle}.rc_min", "RC min over travel (chassis-ref.)", g, axle,
                      r.rc_min_mm, "mm", ".1f", "vdcore.analysis.axle", note=proj)
                b.add(f"{axle}.rc_max", "RC max over travel (chassis-ref.)", g, axle,
                      r.rc_max_mm, "mm", ".1f", "vdcore.analysis.axle", note=proj)
                b.add(f"{axle}.rc_span", "RC span over travel", g, axle,
                      r.rc_max_mm - r.rc_min_mm, "mm", ".1f", "vdcore.analysis.axle", note=proj)
            except (RuntimeError, ValueError) as exc:
                b.fail([(f"{axle}.camber_gain", "Camber gain (per mm bump)", g, axle, "deg/mm"),
                        (f"{axle}.rc_migration", "RC migration", g, axle, "mm/mm")], str(exc))
            try:
                solvers = (DWSolver(ax.left), DWSolver(ax.right))
                step = inp.travel_bump_mm / 20.0       # axle_rates' derivative step
                rc0 = instant_axis_rc(ax, 0.0, solvers)[1]
                mig = (instant_axis_rc(ax, +step, solvers)[1]
                       - instant_axis_rc(ax, -step, solvers)[1]) / (2.0 * step)
                b.add(f"{axle}.rc_static_3d", "Roll centre height, 3D instant axis (static)", g,
                      axle, rc0, "mm", ".2f", _RC3D,
                      note="IC where the upright's instant axis cuts the patch station, both "
                           "sides; equals the projected FVIC value without caster")
                b.add(f"{axle}.rc_migration_3d", "RC migration, 3D instant axis (chassis-ref.)",
                      g, axle, mig, "mm/mm", ".4f", _RC3D,
                      note=f"central difference ±{step:g} mm; ground-referenced differs by "
                           "exactly 1 mm/mm")
            except (RuntimeError, ValueError) as exc:
                b.fail([(f"{axle}.rc_static_3d", "Roll centre height, 3D instant axis (static)",
                         g, axle, "mm"),
                        (f"{axle}.rc_migration_3d", "RC migration, 3D instant axis", g, axle,
                         "mm/mm")], str(exc))
            tick(f"{axle} rates")
        if "roll" in want:
            g = f"Roll ({inp.roll_reference_deg:g}° reference)"
            src = "vdcore.analysis.axle.axle_roll (mapped to the turn)"
            for turn in ("left", "right") if asym else ("left",):
                sfx = "" if turn == "left" else "_right_turn"
                o, i = ("right", "left") if turn == "left" else ("left", "right")
                tn = f"{turn} turn"
                try:
                    rs = turn_roll(ax, inp.roll_reference_deg, turn)
                    b.add(f"{axle}.roll_outer_camber{sfx}",
                          f"Outer ({o}) wheel camber (road), {tn}", g, axle,
                          rs.outer_camber_deg, "deg", ".3f", src,
                          lim.outer_camber_in_roll_deg, _SLA,
                          note=("+roll = left turn (left side up): outer = RIGHT wheel, in bump"
                                if turn == "left" else
                                "right turn: outer = LEFT wheel, in bump (asymmetric axle)"))
                    b.add(f"{axle}.roll_inner_camber{sfx}",
                          f"Inner ({i}) wheel camber (road), {tn}", g, axle,
                          rs.inner_camber_deg, "deg", ".3f", src)
                    b.add(f"{axle}.roll_rc_height{sfx}", f"RC height in roll, {tn}", g, axle,
                          rs.rc_height_mm, "mm", ".1f", src, note=proj)
                    b.add(f"{axle}.roll_rc_lateral{sfx}", f"RC lateral in roll (ISO Y), {tn}", g,
                          axle, rs.rc_lateral_mm, "mm", ".1f", src,
                          note="ISO Y, + = toward the LEFT wheel (the "
                               f"{'inner' if turn == 'left' else 'outer'} one); road frame"
                               + (f"; {proj}" if proj else ""))
                    b.add(f"{axle}.roll_travel{sfx}", f"Outer wheel travel at that roll, {tn}", g,
                          axle, rs.outer_travel_mm, "mm", ".2f", src,
                          note=f"outer ({o}) wheel in bump")
                except (RuntimeError, ValueError) as exc:
                    b.fail([(f"{axle}.roll_outer_camber{sfx}", f"Outer wheel camber (road), {tn}",
                             g, axle, "deg")], str(exc))
            tick(f"{axle} roll")
        if "roll_camber" in want:
            g = "Kinematic rates"
            src = f"axle_roll ±{ROLL_CAMBER_PROBE_DEG:g}°, outer wheel"
            for turn in ("left", "right") if asym else ("left",):
                sfx = "" if turn == "left" else "_right_turn"
                tn = "" if turn == "left" else ", right turn"
                try:
                    rcam = roll_camber_deg_per_deg(ax, turn=turn)
                    b.add(f"{axle}.roll_camber{sfx}", f"Roll camber (chassis-ref.){tn}", g, axle,
                          rcam, "deg/deg", ".4f", src,
                          note="chassis-ref.: 0 = wheel rolls with the body (no recovery), "
                               "-1 = stays upright to the road; road-ref. coefficient = 1 + value")
                    b.add(f"{axle}.camber_recovery{sfx}", f"Camber recovery in roll{tn}", g,
                          axle, -100.0 * rcam, "%", ".1f", "−100 × roll camber (chassis-ref.)",
                          note="share of body roll the outer wheel's camber gives back; "
                               "0 % = rolls with the body, 100 % = stays upright to the road")
                except (RuntimeError, ValueError) as exc:
                    b.fail([(f"{axle}.roll_camber{sfx}", f"Roll camber{tn}", g, axle,
                             "deg/deg")], str(exc))
            tick(f"{axle} roll camber")
        if "bump_steer" in want:
            g = "Bump steer"
            try:
                bump, droop = inp.travel_bump_mm, inp.travel_droop_mm
                ts = _toe_sweep_checked(ax.left, -droop, bump)
                # linear rate over a SYMMETRIC range, so the quadratic term cannot leak in
                m = min(bump, droop)
                sym = ts if bump == droop else _toe_sweep_checked(ax.left, -m, m)
                linear = float(np.polyfit(sym.wheel_travel_mm, sym.toe_deg_per_side, 1)[0])
                toes = np.asarray(ts.toe_deg_per_side)
                b.add(f"{axle}.bump_steer", "Bump steer, linear (per side)", g, axle,
                      linear, "deg/mm", ".5f", "vdcore.analysis.toe",
                      (-gs.BUMP_STEER_LINEAR_LIMIT, gs.BUMP_STEER_LINEAR_LIMIT),
                      "geometry_summary.BUMP_STEER_LINEAR_LIMIT (design_intent)",
                      note=f"positive = toes IN in bump; least-squares over ±{m:g} mm")
                b.add(f"{axle}.bump_steer_peak", "Bump steer, peak |toe| (per side)", g, axle,
                      float(np.max(np.abs(toes))), "deg", ".4f", "vdcore.analysis.toe",
                      (0.0, gs.BUMP_STEER_PEAK_LIMIT),
                      "geometry_summary.BUMP_STEER_PEAK_LIMIT (design_intent)",
                      note=f"over −{droop:g} … +{bump:g} mm; nulling the linear term leaves a "
                           "parabola -- read both")
                b.add(f"{axle}.toe_full_bump", f"Toe change at +{bump:g} mm (per side)", g, axle,
                      float(toes[-1]), "deg", ".4f", "vdcore.analysis.toe")
                b.add(f"{axle}.toe_full_droop", f"Toe change at −{droop:g} mm (per side)", g,
                      axle, float(toes[0]), "deg", ".4f", "vdcore.analysis.toe")
            except (RuntimeError, ValueError) as exc:
                b.fail([(f"{axle}.bump_steer", "Bump steer, linear (per side)", g, axle,
                         "deg/mm"),
                        (f"{axle}.bump_steer_peak", "Bump steer, peak (per side)", g, axle,
                         "deg")], str(exc))
            tick(f"{axle} bump steer")
        if axle == "front":
            _steering_dynamic(b, geo, want, tick, static)
    return b.out


def _steering_dynamic(b: _Builder, geo: CarGeometry, want: set[str],
                      tick: Callable[[str], None], static: KpiSet | None) -> None:
    steer = geo.steering_inputs
    lim = steer.limits
    ax = geo.axle("front")
    g = "Steering kinematics"
    fl, fr = DWSolver(ax.left), DWSolver(ax.right)
    rack_per_deg = float("nan")
    if "steer" in want:
        ge = "Steering effort (parking)"
        # parking effort budget (steering_geometry.rack_force_parking, on solved scrub/trail)
        veh = geo.vehicle_data
        fz = veh.total_mass_kg * GRAVITY * veh.front_mass_fraction / 2.0
        st = static or KpiSet()
        rs = abs(st.value("front.scrub_L"))
        tm = abs(st.value("front.trail_L"))
        if not (math.isfinite(rs) and math.isfinite(tm)):
            rs = tm = float("nan")
        m_kp = steer.mu_parking * fz * math.sqrt(rs ** 2 + tm ** 2)   # N·mm
        try:
            s0l, s0r = _steer_pair(fl, fr, 0.0)
            eps = 0.5
            pl, _ = _steer_pair(fl, fr, +eps)
            ml, _ = _steer_pair(fl, fr, -eps)
            dtoe = pl.toe_deg_per_side - ml.toe_deg_per_side
            rack_per_deg = abs(2.0 * eps / dtoe) if abs(dtoe) > 1e-12 else float("inf")
            ratio = 360.0 * rack_per_deg / (2.0 * math.pi * steer.pinion_radius_mm) \
                if steer.pinion_radius_mm > 0 else float("nan")
            b.add("steering.rack_per_deg", "Rack travel per wheel degree", g, "steering",
                  rack_per_deg, "mm/deg", ".3f", _DW + " rack probe ±0.5 mm",
                  note="steering_geometry calls this the C-factor")
            b.add("steering.ratio", "Steering ratio, on-centre (SW : road wheel)", g,
                  "steering", ratio, ":1", ".2f",
                  f"rack/deg ÷ pinion (r = {steer.pinion_radius_mm:g} mm)",
                  lim.steering_ratio, _STG,
                  note="local ratio at straight ahead; it changes along the stroke "
                       "(Steering page, Ratio vs rack)")
            b.add("steering.rack_per_rev", "Rack travel per pinion revolution", g, "steering",
                  2.0 * math.pi * steer.pinion_radius_mm, "mm/rev", ".1f", "2π·r_pinion")
            mx_l, mx_r = _steer_pair(fl, fr, steer.max_rack_travel_mm)
            inner = abs(mx_l.toe_deg_per_side - s0l.toe_deg_per_side)
            outer = abs(mx_r.toe_deg_per_side - s0r.toe_deg_per_side)
            inner, outer = max(inner, outer), min(inner, outer)
            b.add("steering.max_steer_inner", "Max steer, inner wheel", g, "steering", inner,
                  "deg", ".2f", _DW + f" at ±{steer.max_rack_travel_mm:g} mm rack")
            b.add("steering.max_steer_outer", "Max steer, outer wheel", g, "steering", outer,
                  "deg", ".2f", _DW + f" at ±{steer.max_rack_travel_mm:g} mm rack")
            wb = geo.wheelbase_mm
            b.add("steering.turn_radius_outer", "Outer front wheel path radius at lock", g,
                  "steering", wb / math.sin(math.radians(outer)) / 1000.0 if outer > 1e-6
                  else float("nan"), "m", ".2f", "L / sin(δ_outer), kinematic (no slip)",
                  note="patch-centre path; the FSAE hairpin (9 m outside Ø) bounds the car's "
                       "OUTERMOST point, which lies outboard of the patch")
            sw_lock = 360.0 * steer.max_rack_travel_mm / (2.0 * math.pi * steer.pinion_radius_mm)
            b.add("steering.sw_lock", "Steering-wheel angle lock-to-centre", g, "steering",
                  sw_lock, "deg", ".0f", "rack stroke ÷ pinion")
            b.add("steering.ratio_overall", "Steering ratio, overall (SW lock : mean wheel lock)",
                  g, "steering", sw_lock / (0.5 * (inner + outer)) if inner + outer > 1e-9
                  else float("nan"), ":1", ".2f", "lock-to-centre",
                  note=f"SW {sw_lock:.0f}° ÷ ½({inner:.1f}° inner + {outer:.1f}° outer)")
            f_rack = 2.0 * m_kp * (math.pi / 180.0) / rack_per_deg if rack_per_deg > 1e-6 \
                else float("inf")
            t_sw = f_rack * steer.pinion_radius_mm / 1000.0
            f_rim = t_sw / (steer.steering_wheel_diameter_mm / 2000.0)
            src = "steering_geometry.rack_force_parking (virtual work)"
            b.add("steering.effort_kp_moment", "Kingpin moment per wheel", ge, "steering",
                  m_kp / 1000.0, "N·m", ".2f", src, note=f"μ = {steer.mu_parking:g}, "
                  f"Fz = {fz:.0f} N; {_EFFORT_MODEL}")
            b.add("steering.effort_rack_force", "Rack force (both wheels), on-centre", ge,
                  "steering", f_rack, "N", ".0f", src,
                  note="on-centre C-factor; see the max over the stroke")
            b.add("steering.effort_sw_torque", "Steering-wheel torque, on-centre", ge,
                  "steering", t_sw, "N·m", ".2f", src, (0.0, lim.steering_wheel_torque_Nm), _STG)
            b.add("steering.effort_rim_force", "Rim force, on-centre", ge, "steering", f_rim,
                  "N", ".0f", src, note=f"wheel Ø {steer.steering_wheel_diameter_mm:g} mm")
        except (RuntimeError, ValueError) as exc:
            b.fail([("steering.ratio", "Steering ratio", g, "steering", ":1"),
                    ("steering.rack_per_deg", "Rack travel per wheel degree", g, "steering",
                     "mm/deg")], str(exc))
        # the inner wheel's rate climbs toward lock, so on-centre understates parking effort
        spec = [("steering.effort_sw_torque_max", "Steering-wheel torque, max over the stroke",
                 ge, "steering", "N·m")]
        try:
            if not math.isfinite(m_kp):
                raise ValueError("kingpin moment unavailable (needs the solved static scrub "
                                 "and trail)")
            stroke = steer.max_rack_travel_mm
            racks = sorted({s * f * stroke for f in EFFORT_STROKE_FRACTIONS for s in (1.0, -1.0)})
            torque = [(m_kp * steer_rate_sum(fl, fr, r) * (math.pi / 180.0)
                       * steer.pinion_radius_mm / 1000.0, r) for r in racks]
            t_max, r_max = max(torque)
            pct = ", ".join(f"±{100 * f:g}" for f in EFFORT_STROKE_FRACTIONS if f > 0)
            b.add("steering.effort_sw_torque_max", "Steering-wheel torque, max over the stroke",
                  ge, "steering", t_max, "N·m", ".2f",
                  f"virtual work on DWSolver local steer rates (rack ±{EFFORT_RACK_STEP_MM:g} mm)",
                  (0.0, lim.steering_wheel_torque_Nm), _STG,
                  note=f"rack 0, {pct} % of {stroke:g} mm; max at {r_max:+.1f} mm; "
                       f"{_EFFORT_MODEL}")
        except (RuntimeError, ValueError) as exc:
            b.fail(spec, str(exc))
        tick("steering ratio")
    if "ackermann" in want:
        deg = steer.ackermann_at_steer_deg
        try:
            a = solved_ackermann_pct(ax, geo.wheelbase_mm, deg, steer.max_rack_travel_mm)
            ref = (f"classical cot δo − cot δi = T/L, T = static contact-patch track "
                   f"{a.track_mm:.0f} mm, L = static wheelbase {geo.wheelbase_mm:.0f} mm")
            b.add("steering.ackermann", f"Ackermann at {deg:g}° outer", g, "steering", a.pct,
                  "%", ".1f", _DW + " rack sweep", lim.ackermann_pct, _STG,
                  note=f"{ref}; 0 % parallel, 100 % = that construction; "
                       f"rack {a.rack_mm:.2f} mm")
            b.add("steering.ackermann_inner_error", f"Inner wheel vs that construction at "
                  f"{deg:g}° outer", g, "steering", a.inner_deg - a.inner_ideal_deg, "deg",
                  ".3f", _DW + " rack sweep",
                  note=f"δi − δi,ideal ({ref}); negative = less inner steer than the "
                       "construction; less sensitive to the reference than the %")
        except (RuntimeError, ValueError) as exc:
            b.fail([("steering.ackermann", "Ackermann (solved)", g, "steering", "%"),
                    ("steering.ackermann_inner_error", "Inner wheel vs the construction", g,
                     "steering", "deg")], str(exc))
        tick("Ackermann")
    if "rod_end" in want:
        try:
            b.add("steering.rod_end", "Worst tie-rod swing (bump × steer)", g, "steering",
                  worst_rod_end_swing(geo), "deg", ".2f",
                  _DW + " 9×9 envelope (steering_geometry method)",
                  (0.0, lim.rod_end_misalignment_deg), _STG,
                  note="screening: swing of the rod direction, not the joint's cone angle")
        except (RuntimeError, ValueError) as exc:
            b.fail([("steering.rod_end", "Worst tie-rod swing", g, "steering", "deg")], str(exc))
        tick("rod-end envelope")


def worst_rod_end_swing(geo: CarGeometry, n: int = 9) -> float:
    """Max angle of the FL tie-rod direction from static over the bump × steer box.

    Raises ``RuntimeError`` if any envelope point does not converge: the
    maximum of a partial grid would hide exactly the points that matter.
    """
    inp, steer = geo.front_inputs, geo.steering_inputs
    solver = DWSolver(geo.corner("FL"))
    s = solver.solve()
    if not s.converged:
        raise RuntimeError("static solve failed")
    tri0 = geo.p("FL", "TIE_ROD_IN")
    d0 = np.array([s.tro.x_mm, s.tro.y_mm, s.tro.z_mm]) - tri0
    d0 /= np.linalg.norm(d0)
    worst = 0.0
    failed: list[tuple[float, float]] = []
    for bump in np.linspace(-inp.travel_droop_mm, inp.travel_bump_mm, n):
        for rack in np.linspace(-steer.max_rack_travel_mm, steer.max_rack_travel_mm, n):
            r = solver.solve(wheel_travel_mm=float(bump), rack_mm=float(rack))
            if not r.converged:
                failed.append((float(bump), float(rack)))
                continue
            # DWSolver reports points with the chassis displaced by -bump; the
            # inboard joint moved with it.
            tri = tri0 + np.array([0.0, rack, -bump])
            d = np.array([r.tro.x_mm, r.tro.y_mm, r.tro.z_mm]) - tri
            d /= np.linalg.norm(d)
            worst = max(worst, math.degrees(math.acos(float(np.clip(np.dot(d, d0), -1, 1)))))
    if failed:
        raise RuntimeError(f"{len(failed)}/{n * n} rod-end envelope points did not converge "
                           f"(e.g. bump {failed[0][0]:+.1f} mm, rack {failed[0][1]:+.1f} mm)")
    return worst


def compute_all(geo: CarGeometry, groups: Iterable[str] | None = None,
                progress: Progress | None = None, cancelled: Cancelled | None = None) -> KpiSet:
    st = compute_static(geo)
    return st.merge(compute_dynamic(geo, groups, progress, cancelled, static=st))
