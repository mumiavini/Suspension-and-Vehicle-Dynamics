"""Pushrod / rocker / damper / springs: what the linkage does with the parts the team has.

Kinematics come from ``vdcore.analysis.rocker`` on each axle's LEFT corner
(DWSolver); the right corner is the mirror. On top of the linkage:

* damper usage: its length at ride, full bump and full droop against the
  damper's extended and compressed lengths (``DamperSpec``);
* rates: wheel rate ``k·MR²`` for every spring available, the sprung natural
  frequency (sprung corner mass, tyre not in series), and the roll stiffness
  the springs give, ``½·k_w·T²`` per axle;
* the roll budget: the roll stiffness a target roll gradient needs
  (``W_s·d·(1/gradient + π/180)``, the maneuver page's definition, with the
  pendulum term), what the springs give, and the anti-roll-bar stiffness that
  closes the gap at the chosen front/rear split. Information: the designer
  picks the gradient and the split;
* clearances: closest approach of the rod to the UCA legs and the tie rod /
  toe link over the travel (centrelines: subtract the tube radii);
* the rod force per unit wheel load (virtual work on the rocker).

Nothing here selects a part or a geometry. ``build_rocker`` is a construction
helper -- it places the rocker arms so each force acts along its pickup's motion
at ride height -- used to seed a layout the designer then owns.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

from vdcore.analysis.rocker import RockerLayout, RockerSolveError, RockerSweep, rocker_sweep
from vdcore.geometry.solver import DWSolver

from . import mass
from .geometry import CarGeometry
from .kpis import GRAVITY
from .project import Project, Pushrod, RockerSpec

LB_IN_TO_N_MM = 0.17512683
SWEEP_POINTS = 11
AXLE_CORNER = {"front": "FL", "rear": "RL"}


def n_mm(lb_in: float) -> float:
    return lb_in * LB_IN_TO_N_MM


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


def layout_for(spec: RockerSpec, cid: str) -> RockerLayout:
    """The vdcore layout of one corner (the spec is the LEFT corner; right = Y-mirror)."""
    s = 1.0 if cid in ("FL", "RL") else -1.0

    def m(p):
        return (float(p[0]), s * float(p[1]), float(p[2]))

    ax = spec.rocker_axis
    # an axis is an axial vector: the Y-mirror maps it to (−x, y, −z) (only psi's sign
    # depends on it)
    axis = (float(ax[0]), float(ax[1]), float(ax[2])) if s > 0 else \
        (-float(ax[0]), float(ax[1]), -float(ax[2]))
    return RockerLayout(mount_body=spec.mount_body,  # type: ignore[arg-type]
                        rod_mount=m(spec.rod_mount), rocker_pivot=m(spec.rocker_pivot),
                        rocker_axis=axis,
                        rocker_rod=m(spec.rocker_rod), rocker_damper=m(spec.rocker_damper),
                        damper_chassis=m(spec.damper_chassis))


def build_rocker(mount, pivot, damper_toward, rod_arm_mm: float, damper_arm_mm: float,
                 damper_ride_mm: float, mount_body: str = "lca",
                 rod_type: str = "pushrod") -> RockerSpec:
    """A rocker with both forces tangential at ride height (LEFT corner, chassis frame).

    ``damper_toward``: a point the damper points at from the rocker (its chassis
    end lies on that line, ``damper_ride_mm`` from the rocker pickup).
    """
    m, o = np.asarray(mount, float), np.asarray(pivot, float)
    ud = _unit(np.asarray(damper_toward, float) - o)
    rp, a = o.copy(), np.zeros(3)
    for _ in range(50):
        up = _unit(m - rp)
        a = _unit(np.cross(up, ud))
        new = o + rod_arm_mm * _unit(np.cross(up, a))
        if float(np.linalg.norm(new - rp)) < 1e-12:
            break
        rp = new
    rd = o + damper_arm_mm * _unit(np.cross(a, ud))
    dc = rd + damper_ride_mm * ud

    def t(v):
        return tuple(round(float(x), 4) for x in v)

    return RockerSpec(True, rod_type, mount_body, t(m), t(o), t(a), t(rp), t(rd), t(dc))


def travel_range(geo: CarGeometry, axle: str) -> tuple[float, float]:
    lim = geo.axle_inputs(axle)
    return -float(lim.travel_droop_mm), float(lim.travel_bump_mm)


def _seg_dist(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Closest distance between two segments (Ericson, Real-Time Collision Detection)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    c, b = d1 @ r, d1 @ d2
    den = a * e - b * b
    s = float(np.clip((b * f - c * e) / den, 0, 1)) if den > 1e-12 else 0.0
    t = (b * s + f) / e if e > 1e-12 else 0.0
    if t < 0:
        t, s = 0.0, float(np.clip(-c / a, 0, 1)) if a > 1e-12 else 0.0
    elif t > 1:
        t, s = 1.0, float(np.clip((b - c) / a, 0, 1)) if a > 1e-12 else 0.0
    return float(np.linalg.norm((p1 + s * d1) - (p2 + t * d2)))


@dataclass
class SpringOption:
    lb_in: float
    n_mm: float
    wheel_rate_n_mm: float
    frequency_hz: float
    roll_rate_nm_deg: float
    fitted: bool


@dataclass
class AxleSuspension:
    axle: str
    spec: RockerSpec
    error: str = ""
    sweep: RockerSweep | None = None
    droop_mm: float = math.nan
    bump_mm: float = math.nan
    mr_ride: float = math.nan
    mr_droop: float = math.nan
    mr_bump: float = math.nan
    damper_ride_mm: float = math.nan
    damper_droop_mm: float = math.nan       # longest
    damper_bump_mm: float = math.nan        # shortest
    droop_margin_mm: float = math.nan       # extended − length at full droop (≥ 0 fits)
    bump_margin_mm: float = math.nan        # length at full bump − compressed (≥ 0 fits)
    stroke_used_mm: float = math.nan
    stroke_usable_mm: float = math.nan
    psi_range_deg: tuple[float, float] = (math.nan, math.nan)
    rod_angle_max_deg: float = math.nan
    damper_angle_max_deg: float = math.nan
    rod_length_mm: float = math.nan
    rod_per_wheel: float = math.nan         # F_rod / F_wheel at ride
    sprung_corner_kg: float = math.nan
    spring_force_ride_N: float = math.nan   # fitted spring, static
    spring_compression_ride_mm: float = math.nan
    rod_force_ride_N: float = math.nan
    clearance_mm: float = math.nan
    clearance_with: str = ""
    springs: list[SpringOption] = field(default_factory=list)

    @property
    def progression_pct(self) -> float:
        return 100.0 * (self.mr_bump - self.mr_droop) / self.mr_ride

    @property
    def ok(self) -> bool:
        return not self.error


@dataclass
class RollBudget:
    gradient_deg_per_g: float
    required_nm_deg: float                  # total, with the pendulum term
    springs_nm_deg: dict[str, float]        # fitted springs, per axle
    front_share: float
    arb_needed_nm_deg: dict[str, float]     # per axle at that split (negative: springs exceed)
    gradient_springs_only: float            # deg/g with the fitted springs alone
    note: str = ""


def _axle(geo: CarGeometry, project: Project, axle: str) -> AxleSuspension:
    spec = project.rocker_front if axle == "front" else project.rocker_rear
    out = AxleSuspension(axle, spec)
    if not spec.enabled:
        out.error = "no rocker defined for this axle yet"
        return out
    cid = AXLE_CORNER[axle]
    droop, bump = travel_range(geo, axle)
    out.droop_mm, out.bump_mm = droop, bump
    zs = np.linspace(droop, bump, SWEEP_POINTS)
    if not np.any(np.isclose(zs, 0.0)):
        zs = np.sort(np.append(zs, 0.0))
    corner = geo.corner(cid)
    try:
        lay = layout_for(spec, cid)
        sw = rocker_sweep(corner, lay, zs)
    except (RockerSolveError, ValueError, RuntimeError) as exc:
        out.error = str(exc)
        return out
    out.sweep = sw
    i0, ilo, ihi = int(np.argmin(np.abs(zs))), 0, len(zs) - 1
    out.mr_ride, out.mr_droop, out.mr_bump = (float(sw.motion_ratio[i]) for i in (i0, ilo, ihi))
    L = sw.damper_length_mm
    out.damper_ride_mm, out.damper_droop_mm, out.damper_bump_mm = \
        float(L[i0]), float(L.max()), float(L.min())
    d = project.damper
    compressed = d.eye_to_eye_extended_mm - d.stroke_mm + d.bump_rubber_mm
    out.droop_margin_mm = d.eye_to_eye_extended_mm - out.damper_droop_mm
    out.bump_margin_mm = out.damper_bump_mm - compressed
    out.stroke_used_mm = out.damper_droop_mm - out.damper_bump_mm
    out.stroke_usable_mm = d.stroke_mm - d.bump_rubber_mm
    out.psi_range_deg = (float(sw.psi_deg.min()), float(sw.psi_deg.max()))
    out.rod_angle_max_deg = float(sw.rod_angle_deg.max())
    out.damper_angle_max_deg = float(sw.damper_angle_deg.max())
    st0 = sw.states[i0]
    out.rod_length_mm = st0.rod_length_mm
    out.rod_per_wheel = st0.rod_per_damper / out.mr_ride if out.mr_ride else math.nan

    # static loads through the fitted spring (no ARB load at ride)
    veh = geo.vehicle_data
    sm = mass.sprung_mass(veh.total_mass_kg, veh.unsprung_mass_kg, veh.cg_height_mm,
                          veh.cg_from_front_axle_mm, geo.wheelbase_mm,
                          geo.axles["front"].loaded_radius_mm,
                          geo.axles["rear"].loaded_radius_mm)
    share = sm.front_share if axle == "front" else 1.0 - sm.front_share
    out.sprung_corner_kg = sm.mass_kg * share / 2.0
    fz_s = out.sprung_corner_kg * GRAVITY
    k_fit = n_mm(project.springs.front_lb_in if axle == "front" else project.springs.rear_lb_in)
    out.spring_force_ride_N = fz_s / out.mr_ride
    out.spring_compression_ride_mm = out.spring_force_ride_N / k_fit
    out.rod_force_ride_N = fz_s * out.rod_per_wheel
    t_m = geo.track_mm(axle) / 1000.0
    for lb in sorted(set(project.springs.available_lb_in) | {k_fit / LB_IN_TO_N_MM}):
        k = n_mm(lb)
        kw = k * out.mr_ride ** 2
        out.springs.append(SpringOption(
            lb, k, kw, math.sqrt(kw * 1000.0 / out.sprung_corner_kg) / (2 * math.pi),
            0.5 * kw * 1000.0 * t_m * t_m * math.pi / 180.0, abs(k - k_fit) < 1e-6))

    # clearances of the rod to the other members (not the one it is mounted on)
    s = DWSolver(corner)
    members = []
    if spec.mount_body != "uca":
        members += [("UCA front leg", corner.uca_inboard_front, "ubj"),
                    ("UCA rear leg", corner.uca_inboard_rear, "ubj")]
    if spec.mount_body != "lca":
        members += [("LCA front leg", corner.lca_inboard_front, "lbj"),
                    ("LCA rear leg", corner.lca_inboard_rear, "lbj")]
    members.append(("tie rod / toe link", corner.tie_rod_inboard, "tro"))
    best, with_ = math.inf, ""
    for z, st in zip(zs, sw.states, strict=True):
        res = s.solve(wheel_travel_mm=float(z))
        lift = np.array([0.0, 0.0, float(z)])
        for name, p_in, end in members:
            a = np.array([p_in.x_mm, p_in.y_mm, p_in.z_mm])
            e = getattr(res, end)
            b = np.array([e.x_mm, e.y_mm, e.z_mm]) + lift
            dist = _seg_dist(st.rod_mount, st.rocker_rod, a, b)
            if dist < best:
                best, with_ = dist, f"{name} at {z:+.0f} mm"
    out.clearance_mm, out.clearance_with = best, with_
    return out


def roll_budget(geo: CarGeometry, project: Project, rc: dict[str, float],
                axles: dict[str, AxleSuspension]) -> RollBudget:
    veh = geo.vehicle_data
    grad = veh.target_roll_gradient_deg_per_g
    sm = mass.sprung_mass(veh.total_mass_kg, veh.unsprung_mass_kg, veh.cg_height_mm,
                          veh.cg_from_front_axle_mm, geo.wheelbase_mm,
                          geo.axles["front"].loaded_radius_mm,
                          geo.axles["rear"].loaded_radius_mm)
    h_axis = rc["front"] + (rc["rear"] - rc["front"]) * sm.cg_from_front_axle_mm / \
        geo.wheelbase_mm
    wsd = sm.mass_kg * GRAVITY * (sm.cg_height_mm - h_axis) / 1000.0     # N·m per g
    req = wsd * (1.0 / grad + math.pi / 180.0) if grad > 0 else math.inf
    springs = {}
    for ax, a in axles.items():
        fitted = [o for o in a.springs if o.fitted]
        springs[ax] = fitted[0].roll_rate_nm_deg if fitted else math.nan
    share = project.springs.roll_stiffness_front_share
    arb = {"front": share * req - springs["front"],
           "rear": (1.0 - share) * req - springs["rear"]}
    k_sp = springs["front"] + springs["rear"]
    denom = k_sp - wsd * math.pi / 180.0
    g_sp = wsd / denom if math.isfinite(k_sp) and denom > 0 else math.nan
    note = ("roll stiffness the target gradient needs, with the pendulum term W_s·d·φ; "
            "springs at the ride-height motion ratio; tyres not in series")
    return RollBudget(grad, req, springs, share, arb, g_sp, note)


@dataclass
class SuspensionReport:
    front: AxleSuspension
    rear: AxleSuspension
    budget: RollBudget | None


def suspension_report(geo: CarGeometry, project: Project, rc: dict[str, float]
                      ) -> SuspensionReport:
    f, r = _axle(geo, project, "front"), _axle(geo, project, "rear")
    budget = None
    if f.ok and r.ok and all(math.isfinite(v) for v in rc.values()):
        budget = roll_budget(geo, project, rc, {"front": f, "rear": r})
    return SuspensionReport(f, r, budget)


def pushrod_for_legs(geo: CarGeometry, project: Project, axle: str) -> Pushrod:
    """The pushrod line the link-force equilibrium uses: the rocker's when one is drawn."""
    spec = project.rocker_front if axle == "front" else project.rocker_rear
    rod = project.pushrod_front if axle == "front" else project.pushrod_rear
    if not spec.enabled:
        return rod
    lbj = geo.p(AXLE_CORNER[axle], "LCA_OUT")
    return Pushrod(tuple(np.asarray(spec.rod_mount) - lbj),  # type: ignore[arg-type]
                   tuple(np.asarray(spec.rocker_rod) - lbj),  # type: ignore[arg-type]
                   spec.source)


def tune_rocker(geo: CarGeometry, project: Project, axle: str, base: RockerSpec,
                target_mr: float, rod_arm_mm: float, damper_toward) -> RockerSpec:
    """Seed helper: damper arm for a target ride-height MR, then the damper's ride length
    set so the droop and bump margins are equal. Bisection; raises if not bracketed."""
    cid = AXLE_CORNER[axle]
    corner = geo.corner(cid)
    droop, bump = travel_range(geo, axle)

    def mr_for(arm: float) -> float:
        sp = build_rocker(base.rod_mount, base.rocker_pivot, damper_toward, rod_arm_mm, arm,
                          200.0, base.mount_body, base.rod_type)
        return float(rocker_sweep(corner, layout_for(sp, cid), [0.0]).motion_ratio[0])

    lo, hi = 10.0, 200.0
    if not (mr_for(lo) - target_mr) * (mr_for(hi) - target_mr) < 0:
        raise ValueError(f"MR {target_mr:g} not reachable with damper arms 10-200 mm")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (mr_for(lo) - target_mr) * (mr_for(mid) - target_mr) <= 0:
            hi = mid
        else:
            lo = mid
    arm = 0.5 * (lo + hi)
    d = project.damper
    compressed = d.eye_to_eye_extended_mm - d.stroke_mm + d.bump_rubber_mm
    ride = 0.5 * (d.eye_to_eye_extended_mm + compressed)
    for _ in range(6):
        sp = build_rocker(base.rod_mount, base.rocker_pivot, damper_toward, rod_arm_mm, arm,
                          ride, base.mount_body, base.rod_type)
        sw = rocker_sweep(corner, layout_for(sp, cid), [droop, bump])
        m_droop = d.eye_to_eye_extended_mm - float(sw.damper_length_mm[0])
        m_bump = float(sw.damper_length_mm[1]) - compressed
        ride += 0.5 * (m_droop - m_bump)
    return replace(sp, source="design_intent")
