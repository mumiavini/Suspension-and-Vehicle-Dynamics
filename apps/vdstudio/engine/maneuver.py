"""Quasi-static maneuver: one steady state of lateral and longitudinal g.

This is NOT a dynamics simulation (no transients, no tyre forces, no
frequency response -- CLAUDE.md). It answers: at a steady ``ay`` / ``ax``,
with the roll and pitch gradients the designer targets, where does the
kinematic geometry put each wheel, and how is the load split?

Load transfer (Milliken RCVD Ch.18 decomposition), per axle::

    ΔFz = ΔFz_geometric + ΔFz_elastic + ΔFz_unsprung
    ΔFz_geometric = W_s·(share of sprung mass on this axle)·ay·h_rc / T
    ΔFz_elastic   = (K_axle / K_total)·W_s·(ay + φ)·d / T  (d = h_s − roll axis at the sprung CG)
    ΔFz_unsprung  = W_u,axle·ay·h_u / T                  (h_u = loaded radius)

``W_s``, ``h_s`` and the sprung CG station come from :mod:`.mass` (the
whole-car CG with the unsprung masses removed), so the three terms add up to
the rigid-body moment ``Σ ΔFz·T = m·g·ay·h`` plus the pendulum moment
``W_s·d·φ`` (the rolled sprung CG moves ``d·φ`` outboard; ``φ`` in rad, small
angle). The springs react ``W_s·d·(ay + φ)``, so the implied roll stiffness is
``W_s·d·(1/gradient + π/180)`` -- the same definition as the concept page's
``concept.k_roll_pend``. Every term is linear in ``ay`` (``φ = gradient·ay``),
so the LLTD does not depend on ``ay`` and is evaluated at 1 g.

The elastic split ``K_front / K_total`` is a design input (springs + ARB are
not modelled here); the resulting LLTD is reported, never chosen.

Assumptions (stated on the page): the unsprung mass is split equally front /
rear (``mass.UNSPRUNG_ASSUMPTION``); pitch rotates about the CG station (equal
front/rear pitch frequency, anti-geometry folded into the pitch gradient).

Frames: ISO 8855. ``ay`` positive = acceleration toward +Y (a LEFT turn), so
load moves to the RIGHT wheels and the body rolls positive about +X (left side
up). ``ax`` positive = accelerating; braking is negative and moves load to the
front. Camber is reported relative to the ROAD.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import sla_geometry as sla
from vdcore.analysis.axle import axle_roll
from vdcore.geometry.solver import DWSolver

from . import mass
from .geometry import CarGeometry
from .kpis import GRAVITY

CORNER_SIDE = {"FL": ("front", +1), "FR": ("front", -1), "RL": ("rear", +1), "RR": ("rear", -1)}

ASSUMPTIONS = (f"{mass.UNSPRUNG_ASSUMPTION}; roll about the axis through the roll centres, "
               "taken at the sprung CG station; pitch rotates about the CG station (equal "
               "front/rear pitch frequency; anti-geometry is inside the pitch gradient)")


@dataclass
class ManeuverInputs:
    ay_g: float = 1.5
    ax_g: float = 0.0
    roll_gradient_deg_per_g: float = 1.0       # design_intent
    pitch_gradient_deg_per_g: float = 0.5      # design_intent
    roll_stiffness_front_share: float = 0.50   # design_intent (springs + ARB split)
    rc_source: str = "solved"                  # "solved" static RCs or "design" targets


@dataclass
class WheelState:
    cid: str
    fz_N: float
    travel_mm: float
    camber_road_deg: float
    camber_chassis_deg: float
    toe_deg_per_side: float
    converged: bool
    # False when the travel leaves the axle's design bump/droop range (no bump stop modelled)
    in_design_travel: bool = True


@dataclass
class ManeuverResult:
    inputs: ManeuverInputs
    roll_deg: float
    pitch_deg: float
    wheels: dict[str, WheelState]
    transfer: dict[str, dict[str, float]]      # per axle: geometric/elastic/unsprung/total
    lltd_front: float
    roll_axis_at_cg_mm: float                  # at the SPRUNG CG station
    roll_moment_arm_mm: float                  # sprung CG height − roll axis
    k_roll_total_Nm_per_deg: float             # incl. pendulum; inf = rigid, nan = undefined
    longitudinal_transfer_N: float
    rc_mm: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    sprung: mass.SprungMass | None = None
    pendulum_moment_Nm: float = 0.0            # W_s·d·φ, inside the elastic transfer


def run_maneuver(geo: CarGeometry, inp: ManeuverInputs,
                 rc_static: dict[str, float] | None = None) -> ManeuverResult:
    """One steady state. Wheel loads are always reported; kinematics may fail per wheel.

    Raises:
        ValueError: for an input outside its domain (front share outside [0, 1], a
            negative roll gradient, a non-finite acceleration), a roll-centre height
            that is not finite (e.g. ``rc_static`` from a failed KPI), or a sprung
            mass that cannot be formed.
    """
    share_f = inp.roll_stiffness_front_share
    grad = inp.roll_gradient_deg_per_g
    if not 0.0 <= share_f <= 1.0:
        raise ValueError(f"elastic roll stiffness front share must be in [0, 1], got {share_f:g}")
    if not grad >= 0.0:
        raise ValueError(f"roll gradient must be ≥ 0 deg/g, got {grad:g}")
    if not (math.isfinite(inp.ay_g) and math.isfinite(inp.ax_g)
            and math.isfinite(inp.pitch_gradient_deg_per_g)):
        raise ValueError("ay, ax and the pitch gradient must be finite")

    veh: sla.VehicleData = geo.vehicle_data
    warnings: list[str] = []
    L = geo.wheelbase_mm
    # one weight distribution: the vehicle data's front mass fraction, placed on the
    # wheelbase the points actually have (geometry.py already scales the vehicle
    # data to it; this keeps the two consistent if they ever differ)
    a = veh.cg_from_front_axle_mm * L / veh.wheelbase_mm
    b_ = L - a
    h = veh.cg_height_mm
    h_u = {axle: geo.axles[axle].loaded_radius_mm for axle in ("front", "rear")}
    sm = mass.sprung_mass(veh.total_mass_kg, veh.unsprung_mass_kg, h, a, L,
                          h_u["front"], h_u["rear"])
    W_s = sm.mass_kg * GRAVITY
    m_u_axle = {"front": sm.unsprung_front_kg, "rear": sm.unsprung_rear_kg}
    tf, tr = geo.track_mm("front"), geo.track_mm("rear")

    if inp.rc_source == "design" or not rc_static:
        rc = {"front": geo.front_inputs.rc_height_mm, "rear": geo.rear_inputs.rc_height_mm}
        design_rc = True
    else:
        rc = dict(rc_static)
        design_rc = False
    bad = [axle for axle in ("front", "rear") if not math.isfinite(rc[axle])]
    if bad and design_rc:
        raise ValueError(f"design roll-centre height is not finite ({', '.join(bad)})")
    if bad:
        keys = ", ".join(f"{x}.rc_static" for x in bad)
        raise ValueError(f"static roll centre unavailable ({keys} failed): use the design RCs "
                         "or fix the geometry")
    h_axis = rc["front"] + (rc["rear"] - rc["front"]) * sm.cg_from_front_axle_mm / L
    d = sm.cg_height_mm - h_axis
    if d <= 0:
        warnings.append(f"Roll axis at/above the sprung CG (d = {d:.1f} mm): the elastic roll "
                        "moment reverses, so a positive roll gradient is inconsistent with "
                        "these roll centres. The implied roll stiffness is undefined.")
        k_tot = float("nan")
    elif grad > 0:
        # springs react W_s·d·(ay + φ): K = W_s·d·(1/gradient + π/180)  [N·m/deg]
        k_tot = W_s * d / 1000.0 * (1.0 / grad + math.pi / 180.0)
    else:
        k_tot = float("inf")                   # rigid: zero gradient

    roll = grad * inp.ay_g
    pitch = -inp.pitch_gradient_deg_per_g * inp.ax_g + 0.0  # braking (ax<0) -> nose down (+)

    shares = {"front": share_f, "rear": 1.0 - share_f}
    mass_share = {"front": sm.front_share, "rear": 1.0 - sm.front_share}

    def lateral(ay_g: float) -> dict[str, dict[str, float]]:
        phi = math.radians(grad * ay_g)        # small angle: sin φ ≈ φ
        out: dict[str, dict[str, float]] = {}
        for axle, t in (("front", tf), ("rear", tr)):
            geo_t = W_s * mass_share[axle] * ay_g * rc[axle] / t
            el_t = shares[axle] * W_s * (ay_g + phi) * d / t
            un_t = m_u_axle[axle] * GRAVITY * ay_g * h_u[axle] / t
            out[axle] = {"geometric": geo_t, "elastic": el_t, "unsprung": un_t,
                         "total": geo_t + el_t + un_t}
        return out

    transfer = lateral(inp.ay_g)
    # every term is linear in ay, so the LLTD is the same at any ay; at 1 g it is
    # also defined when the state itself has ay = 0
    per_g = lateral(1.0)
    tot = per_g["front"]["total"] + per_g["rear"]["total"]
    lltd = per_g["front"]["total"] / tot if abs(tot) > 1e-12 else float("nan")
    m_pend = W_s * d * math.radians(roll) / 1000.0

    W = veh.total_mass_kg * GRAVITY
    d_long = W * inp.ax_g * h / L          # +ax moves load REARWARD
    static = {"front": W * (b_ / L) / 2.0, "rear": W * (a / L) / 2.0}

    # Wheel travel: roll part from the exact both-patches-on-road solve, pitch part
    # from the small-angle station offset about the CG.
    roll_travel: dict[str, float] = {}
    for axle in ("front", "rear"):
        try:
            roll_travel[axle] = axle_roll(geo.axle(axle), abs(roll)).wheel_travel_mm \
                if abs(roll) > 1e-9 else 0.0
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"{axle} roll solve failed: {exc}. Wheel loads still stand; "
                            f"{axle} cambers are not computed.")
            roll_travel[axle] = float("nan")
    x_cg = -a
    wheels: dict[str, WheelState] = {}
    sgn_roll = 1.0 if roll >= 0 else -1.0
    nan = float("nan")
    for cid, (axle, side) in CORNER_SIDE.items():
        # positive ay: right wheels (side -1) are outer and gain load
        lat = -side * transfer[axle]["total"]
        lon = (-d_long / 2.0) if axle == "front" else (d_long / 2.0)
        fz = static[axle] + lat + lon
        if not fz > 0:
            warnings.append(f"{cid} has lifted (Fz = {fz:.0f} N): the steady state is void "
                            "at these inputs.")
        x = geo.p(cid, "CONTACT_PATCH")[0]
        t_pitch = (x - x_cg) * math.tan(math.radians(pitch))
        # positive roll lifts the LEFT side: left droops, right bumps
        t_roll = -side * sgn_roll * roll_travel[axle]
        travel = t_roll + t_pitch
        if not math.isfinite(travel):
            # the roll solve failed (warned above): no kinematics for this wheel
            wheels[cid] = WheelState(cid, fz, travel, nan, nan, nan, False)
            continue
        lim = geo.axle_inputs(axle)
        inside = -lim.travel_droop_mm <= travel <= lim.travel_bump_mm
        if not inside:
            warnings.append(f"{cid} travel {travel:+.1f} mm is outside the {axle} design range "
                            f"(+{lim.travel_bump_mm:g} / −{lim.travel_droop_mm:g} mm); bump "
                            "stop and droop limit are not modelled.")
        try:
            r = DWSolver(geo.corner(cid)).solve(wheel_travel_mm=float(travel))
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"{cid} kinematic solve failed at {travel:+.1f} mm: {exc}")
            wheels[cid] = WheelState(cid, fz, travel, nan, nan, nan, False, inside)
            continue
        cam_road = r.camber_deg - side * roll   # outer (bump side) gains +roll
        wheels[cid] = WheelState(cid, fz, travel, cam_road if r.converged else nan,
                                 r.camber_deg if r.converged else nan,
                                 r.toe_deg_per_side if r.converged else nan,
                                 bool(r.converged), inside)
    return ManeuverResult(inp, roll, pitch, wheels, transfer, lltd, h_axis, d, k_tot,
                          d_long, rc, warnings, sm, m_pend)
