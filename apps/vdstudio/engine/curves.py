"""Kinematic sweeps for plotting: heave, roll and steer, on ``DWSolver``.

Every curve is a set of NumPy arrays sampled on the requested axis. A point
whose solve did not converge is ``NaN`` (never interpolated over), and the
``converged`` mask says which.

Conventions (ISO 8855, CLAUDE.md):
* wheel travel positive in BUMP (wheel up relative to the chassis);
* Z values (RC height, contact patch) CHASSIS-referenced, as in
  ``vdcore.analysis.axle.sample_corner``;
* toe per side, positive toe-IN; heave curves report toe CHANGE from ride
  height (bump steer), steer curves report steer angle from straight ahead;
* camber negative = top inboard; roll curves report camber relative to the
  ROAD, which is what the tyre sees.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from vdcore.analysis.axle import axle_roll
from vdcore.analysis.roll_centre import front_view_instant_centre
from vdcore.geometry.derived import mechanical_trail_mm, scrub_radius_mm
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import DerivedPoint

from .geometry import CarGeometry

Arr = np.ndarray
Cancelled = Callable[[], bool]

HEAVE_QUANTITIES: dict[str, tuple[str, str]] = {
    "camber": ("Camber", "deg"),
    "toe": ("Toe change per side (bump steer)", "deg"),
    "caster": ("Caster", "deg"),
    "kpi": ("KPI", "deg"),
    "scrub": ("Scrub radius", "mm"),
    "trail": ("Mechanical trail", "mm"),
    "half_track": ("Half-track change (outboard +)", "mm"),
    "wc_x": ("Wheel centre fore-aft change (fwd +)", "mm"),
    "fvsa": ("FVSA length", "mm"),
    "rc": ("Roll centre height (chassis-ref.)", "mm"),
}


@dataclass
class HeaveCurves:
    travel: Arr
    corners: dict[str, dict[str, Arr]] = field(default_factory=dict)
    converged: dict[str, Arr] = field(default_factory=dict)


def _nan(n: int) -> Arr:
    return np.full(n, np.nan)


def _rc_from_sample(corner, r: SolverResult, travel: float) -> tuple[float, float]:
    """(RC height, |FVSA|) for one solved corner, chassis frame (sample_corner logic)."""
    def to_chassis(p: DerivedPoint) -> DerivedPoint:
        return DerivedPoint(x_mm=p.x_mm, y_mm=p.y_mm, z_mm=p.z_mm + travel)

    cp = to_chassis(r.contact_patch)
    fv = front_view_instant_centre(corner, ubj=to_chassis(r.ubj), lbj=to_chassis(r.lbj),
                                   contact_patch=cp)
    if not fv.is_finite:
        return float("nan"), float("inf")
    dy = fv.fvic_y_mm - cp.y_mm
    if abs(dy) < 1e-12:
        return float("nan"), abs(fv.fvsa_mm)
    rc = cp.z_mm + (0.0 - cp.y_mm) * (fv.fvic_z_mm - cp.z_mm) / dy
    return float(rc), abs(fv.fvsa_mm)


def heave_curves(geo: CarGeometry, travel_min_mm: float, travel_max_mm: float, n: int = 41,
                 cancelled: Cancelled | None = None) -> HeaveCurves:
    travel = np.linspace(travel_min_mm, travel_max_mm, n)
    out = HeaveCurves(travel=travel)
    for cid in ("FL", "FR", "RL", "RR"):
        corner = geo.corner(cid)
        solver = DWSolver(corner)
        s0 = solver.solve()
        left = cid in ("FL", "RL")
        q = {k: _nan(n) for k in HEAVE_QUANTITIES}
        conv = np.zeros(n, dtype=bool)
        for i, t in enumerate(travel):
            if cancelled is not None and cancelled():
                raise InterruptedError
            r = solver.solve(wheel_travel_mm=float(t))
            if not (r.converged and s0.converged):
                continue
            conv[i] = True
            q["camber"][i] = r.camber_deg
            q["toe"][i] = r.toe_deg_per_side - s0.toe_deg_per_side
            q["caster"][i] = r.caster_deg
            q["kpi"][i] = r.kpi_deg
            try:
                q["scrub"][i] = scrub_radius_mm(r)
                q["trail"][i] = mechanical_trail_mm(r)
            except ValueError:
                pass
            dy = r.contact_patch.y_mm - s0.contact_patch.y_mm
            q["half_track"][i] = dy if left else -dy
            q["wc_x"][i] = r.wheel_center.x_mm - s0.wheel_center.x_mm
            rc, fvsa = _rc_from_sample(corner, r, float(t))
            q["rc"][i] = rc
            q["fvsa"][i] = fvsa
        out.corners[cid] = q
        out.converged[cid] = conv
    return out


@dataclass
class RollCurves:
    roll: Arr
    axles: dict[str, dict[str, Arr]] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


ROLL_QUANTITIES: dict[str, tuple[str, str]] = {
    "outer_camber": ("Outer wheel camber (road)", "deg"),
    "inner_camber": ("Inner wheel camber (road)", "deg"),
    "rc_height": ("Roll centre height", "mm"),
    "rc_lateral": ("Roll centre lateral", "mm"),
    "travel": ("Wheel travel (outer bump / inner droop)", "mm"),
}


def roll_curves(geo: CarGeometry, roll_max_deg: float, n: int = 13,
                cancelled: Cancelled | None = None) -> RollCurves:
    """Both wheels on the road at each chassis roll (``vdcore.analysis.axle.axle_roll``)."""
    roll = np.linspace(0.0, roll_max_deg, n)
    out = RollCurves(roll=roll)
    for axle in ("front", "rear"):
        q = {k: _nan(n) for k in ROLL_QUANTITIES}
        ax = geo.axle(axle)
        for i, phi in enumerate(roll):
            if cancelled is not None and cancelled():
                raise InterruptedError
            try:
                s = axle_roll(ax, float(phi))
            except (RuntimeError, ValueError) as exc:
                out.errors[axle] = str(exc)
                continue
            q["outer_camber"][i] = s.outer_camber_deg
            q["inner_camber"][i] = s.inner_camber_deg
            q["rc_height"][i] = s.rc_height_mm
            q["rc_lateral"][i] = s.rc_lateral_mm
            q["travel"][i] = s.wheel_travel_mm
        out.axles[axle] = q
    return out


@dataclass
class SteerCurves:
    rack: Arr
    steer_fl: Arr
    steer_fr: Arr
    camber_fl: Arr
    camber_fr: Arr
    caster_fl: Arr
    kpi_fl: Arr
    scrub_fl: Arr
    trail_fl: Arr
    # Ackermann view: outer vs inner on the positive-rack half
    ack_outer: Arr
    ack_inner: Arr
    ack_ideal: Arr
    ack_pct: Arr
    turn_radius_m: Arr


def steer_curves(geo: CarGeometry, n: int = 41, cancelled: Cancelled | None = None
                 ) -> SteerCurves:
    """Rack sweep over ± max rack travel. Steer angle = toe change from straight ahead,
    signed as in ``vdcore`` (toe-in positive per side). For a positive rack (+Y,
    left turn with the tie rods ahead of the axle) the left wheel toes out."""
    steer = geo.steering_inputs
    rack = np.linspace(-steer.max_rack_travel_mm, steer.max_rack_travel_mm, n)
    fl, fr = DWSolver(geo.corner("FL")), DWSolver(geo.corner("FR"))
    l0, r0 = fl.solve(), fr.solve()
    arrs = {k: _nan(n) for k in ("sfl", "sfr", "cfl", "cfr", "cas", "kpi", "scr", "trl")}
    for i, rk in enumerate(rack):
        if cancelled is not None and cancelled():
            raise InterruptedError
        a, c = fl.solve(rack_mm=float(rk)), fr.solve(rack_mm=float(rk))
        if not (a.converged and c.converged and l0.converged and r0.converged):
            continue
        arrs["sfl"][i] = a.toe_deg_per_side - l0.toe_deg_per_side
        arrs["sfr"][i] = c.toe_deg_per_side - r0.toe_deg_per_side
        arrs["cfl"][i] = a.camber_deg
        arrs["cfr"][i] = c.camber_deg
        arrs["cas"][i] = a.caster_deg
        arrs["kpi"][i] = a.kpi_deg
        try:
            arrs["scr"][i] = scrub_radius_mm(a)
            arrs["trl"][i] = mechanical_trail_mm(a)
        except ValueError:
            pass

    # Ackermann on |rack| > 0: inner wheel steers more.
    track = abs(l0.contact_patch.y_mm - r0.contact_patch.y_mm) if l0.converged else math.nan
    wb = geo.wheelbase_mm
    pos = rack > 1e-9
    a_l, a_r = np.abs(arrs["sfl"][pos]), np.abs(arrs["sfr"][pos])
    outer = np.minimum(a_l, a_r)
    inner = np.maximum(a_l, a_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        cot_i = 1.0 / np.tan(np.radians(outer)) - track / wb
        ideal = np.degrees(np.arctan(1.0 / cot_i))
        pct = 100.0 * (inner - outer) / (ideal - outer)
        radius = wb / np.sin(np.radians(outer)) / 1000.0
    return SteerCurves(
        rack=rack, steer_fl=arrs["sfl"], steer_fr=arrs["sfr"], camber_fl=arrs["cfl"],
        camber_fr=arrs["cfr"], caster_fl=arrs["cas"], kpi_fl=arrs["kpi"],
        scrub_fl=arrs["scr"], trail_fl=arrs["trl"], ack_outer=outer, ack_inner=inner,
        ack_ideal=ideal, ack_pct=pct, turn_radius_m=radius,
    )
