"""Quasi-static maneuver: one steady state of lateral and longitudinal g.

This is NOT a dynamics simulation (no transients, no tyre forces, no
frequency response -- CLAUDE.md). It answers: at a steady ``ay`` / ``ax``,
with the roll and pitch gradients the designer targets, where does the
kinematic geometry put each wheel, and how is the load split?

Load transfer (Milliken RCVD Ch.18 decomposition), per axle::

    ΔFz = ΔFz_geometric + ΔFz_elastic + ΔFz_unsprung
    ΔFz_geometric = W_s·(share of sprung mass on this axle)·ay·h_rc / T
    ΔFz_elastic   = (K_axle / K_total)·W_s·ay·d / T     (d = h_cg − roll axis at CG)
    ΔFz_unsprung  = W_u,axle·ay·h_u / T                  (h_u = loaded radius)

The elastic split ``K_front / K_total`` is a design input (springs + ARB are
not modelled here); the resulting LLTD is reported, never chosen.

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

from .geometry import CarGeometry
from .kpis import GRAVITY

CORNER_SIDE = {"FL": ("front", +1), "FR": ("front", -1), "RL": ("rear", +1), "RR": ("rear", -1)}


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


@dataclass
class ManeuverResult:
    inputs: ManeuverInputs
    roll_deg: float
    pitch_deg: float
    wheels: dict[str, WheelState]
    transfer: dict[str, dict[str, float]]      # per axle: geometric/elastic/unsprung/total
    lltd_front: float
    roll_axis_at_cg_mm: float
    roll_moment_arm_mm: float
    k_roll_total_Nm_per_deg: float
    longitudinal_transfer_N: float
    rc_mm: dict[str, float]
    warnings: list[str] = field(default_factory=list)


def run_maneuver(geo: CarGeometry, inp: ManeuverInputs,
                 rc_static: dict[str, float] | None = None) -> ManeuverResult:
    veh: sla.VehicleData = geo.vehicle_data
    warnings: list[str] = []
    L = geo.wheelbase_mm
    a = veh.cg_from_front_axle_mm
    b_ = L - a
    h = veh.cg_height_mm
    m_u_corner = veh.unsprung_mass_kg / 4.0
    W_s = veh.sprung_mass_kg * GRAVITY
    tf, tr = geo.track_mm("front"), geo.track_mm("rear")

    if inp.rc_source == "design" or not rc_static:
        rc = {"front": geo.front_inputs.rc_height_mm, "rear": geo.rear_inputs.rc_height_mm}
    else:
        rc = dict(rc_static)
    h_axis = rc["front"] + (rc["rear"] - rc["front"]) * a / L
    d = h - h_axis
    grad = inp.roll_gradient_deg_per_g
    k_tot = W_s * d / 1000.0 / grad if grad > 0 else float("inf")   # N·m/deg

    roll = grad * inp.ay_g
    pitch = -inp.pitch_gradient_deg_per_g * inp.ax_g + 0.0  # braking (ax<0) -> nose down (+)

    transfer: dict[str, dict[str, float]] = {}
    shares = {"front": inp.roll_stiffness_front_share,
              "rear": 1.0 - inp.roll_stiffness_front_share}
    mass_share = {"front": b_ / L, "rear": a / L}
    for axle, t in (("front", tf), ("rear", tr)):
        h_u = geo.axles[axle].loaded_radius_mm
        geo_t = W_s * mass_share[axle] * inp.ay_g * rc[axle] / t
        el_t = shares[axle] * W_s * inp.ay_g * d / t
        un_t = 2 * m_u_corner * GRAVITY * inp.ay_g * h_u / t
        transfer[axle] = {"geometric": geo_t, "elastic": el_t, "unsprung": un_t,
                          "total": geo_t + el_t + un_t}
    tot = transfer["front"]["total"] + transfer["rear"]["total"]
    lltd = transfer["front"]["total"] / tot if abs(tot) > 1e-12 else float("nan")

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
            warnings.append(f"{axle} roll solve failed: {exc}")
            roll_travel[axle] = float("nan")
    x_cg = -a
    wheels: dict[str, WheelState] = {}
    sgn_roll = 1.0 if roll >= 0 else -1.0
    for cid, (axle, side) in CORNER_SIDE.items():
        # positive ay: right wheels (side -1) are outer and gain load
        lat = -side * transfer[axle]["total"]
        lon = (-d_long / 2.0) if axle == "front" else (d_long / 2.0)
        fz = static[axle] + lat + lon
        x = geo.p(cid, "CONTACT_PATCH")[0]
        t_pitch = (x - x_cg) * math.tan(math.radians(pitch))
        # positive roll lifts the LEFT side: left droops, right bumps
        t_roll = -side * sgn_roll * roll_travel[axle]
        travel = t_roll + t_pitch
        r = DWSolver(geo.corner(cid)).solve(wheel_travel_mm=float(travel))
        cam_road = r.camber_deg - side * roll   # outer (bump side) gains +roll
        wheels[cid] = WheelState(cid, fz, travel, cam_road if r.converged else float("nan"),
                                 r.camber_deg if r.converged else float("nan"),
                                 r.toe_deg_per_side if r.converged else float("nan"),
                                 bool(r.converged))
        if fz <= 0:
            warnings.append(f"{cid} has lifted (Fz = {fz:.0f} N): the steady state is void "
                            "at these inputs.")
    return ManeuverResult(inp, roll, pitch, wheels, transfer, lltd, h_axis, d, k_tot,
                          d_long, rc, warnings)
