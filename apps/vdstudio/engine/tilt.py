"""Static tilt table: wheel loads, first wheel lift and the tip angle.

The car stands on a table tilted by θ about an axis parallel to its X axis;
gravity has a component ``W·cos θ`` normal to the table and ``W·sin θ`` along
it, toward the downhill side. Quasi-static: nothing moves but the suspension.

Tip angle (the car rotates about the line through its two downhill contact
patches)::

    tan θ_tip = (y_line(x_cg) − Δy(θ)) / (h_cg + Δh)

``y_line(x_cg)`` is the downhill tipping line's lateral distance from the CG
at the CG station (between the front and rear half-tracks), ``Δy`` the CG's
shift toward that line and ``Δh`` its rise. Rigid car: Δy = Δh = 0. Body
roll moves the sprung CG ``d·φ`` downhill (``d`` = sprung CG over the roll
axis), so ``Δy = (W_s/W)·d·φ`` with ``φ = gradient·sin θ`` (the roll gradient is
per g of lateral load; on the table the lateral load is ``sin θ`` g). The
springs extending under the reduced normal load (``cos θ``) and the tyres'
lateral deflection are not modelled -- no wheel rates or tyre data -- and
enter as the inputs ``cg_rise_mm`` and ``lateral_shift_mm`` to show their
effect.

Wheel loads (normal to the table) use the maneuver page's per-axle
decomposition (Milliken RCVD Ch.18): geometric (roll centre), elastic (roll
stiffness share, with the pendulum term) and unsprung, with the lateral load
``sin θ`` and the normal load ``cos θ``. One uphill wheel lifting is not the
tip: on three wheels the chassis still stands until the tip angle.

Nothing here judges the result: it reports angles, loads and margins.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from scipy.optimize import brentq

from . import mass
from .geometry import CarGeometry
from .kpis import GRAVITY

CORNER_SIDE = {"FL": ("front", +1), "FR": ("front", -1), "RL": ("rear", +1), "RR": ("rear", -1)}


@dataclass(frozen=True)
class TiltInputs:
    angle_deg: float = 60.0
    roll_gradient_deg_per_g: float = 1.0      # design_intent
    roll_stiffness_front_share: float = 0.50  # design_intent
    cg_rise_mm: float = 0.0                   # springs extending, not modelled
    lateral_shift_mm: float = 0.0             # tyre lateral deflection etc., not modelled
    cg_height_mm: float | None = None         # override the vehicle data's (sensitivity)


@dataclass
class TiltResult:
    inputs: TiltInputs
    wheel_fz_N: dict[str, float]              # normal to the table, downhill = RIGHT side
    roll_deg: float
    cg_shift_mm: float                        # Δy toward the downhill line
    tip_line_mm: float                        # y_line at the CG station
    cg_height_mm: float                       # h_cg + rise
    tip_angle_deg: float                      # with the compliance terms
    tip_angle_rigid_deg: float
    first_lift: tuple[str, float] | None      # (corner, angle) of the first uphill wheel lift
    max_cg_height_mm: float                   # h at which the car tips exactly at angle_deg
    rc_mm: dict[str, float]
    roll_arm_mm: float                        # d
    sprung_fraction: float                    # W_s / W
    notes: list[str] = field(default_factory=list)

    @property
    def margin_deg(self) -> float:
        return self.tip_angle_deg - self.inputs.angle_deg


def tilt_table(geo: CarGeometry, rc_static: dict[str, float], inp: TiltInputs) -> TiltResult:
    """The car on the table, downhill on its RIGHT (ISO: −Y). Symmetric car: either side."""
    veh = geo.vehicle_data
    L = geo.wheelbase_mm
    a = veh.cg_from_front_axle_mm * L / veh.wheelbase_mm
    h = veh.cg_height_mm if inp.cg_height_mm is None else inp.cg_height_mm
    if not h > 0:
        raise ValueError(f"CG height must be > 0 mm, got {h:g}")
    h_u = {ax: geo.axles[ax].loaded_radius_mm for ax in ("front", "rear")}
    # the rise is the body lifting on its springs: it moves the CG (and so the roll arm
    # d and the load transfer) as well as the tip condition
    h_eff = h + inp.cg_rise_mm
    sm = mass.sprung_mass(veh.total_mass_kg, veh.unsprung_mass_kg, h_eff, a, L,
                          h_u["front"], h_u["rear"])
    W = veh.total_mass_kg * GRAVITY
    W_s = sm.mass_kg * GRAVITY
    m_u = {"front": sm.unsprung_front_kg, "rear": sm.unsprung_rear_kg}
    t = {"front": geo.track_mm("front"), "rear": geo.track_mm("rear")}
    rc = dict(rc_static)
    if not all(math.isfinite(v) for v in rc.values()):
        raise ValueError("static roll centre heights are not finite")
    h_axis = rc["front"] + (rc["rear"] - rc["front"]) * sm.cg_from_front_axle_mm / L
    d = sm.cg_height_mm - h_axis
    share = {"front": inp.roll_stiffness_front_share,
             "rear": 1.0 - inp.roll_stiffness_front_share}
    mshare = {"front": sm.front_share, "rear": 1.0 - sm.front_share}
    grad = inp.roll_gradient_deg_per_g

    def phi_rad(theta: float) -> float:
        return math.radians(grad * math.sin(theta))

    def loads(theta: float) -> dict[str, float]:
        s, c = math.sin(theta), math.cos(theta)
        out = {}
        static = {"front": W * c * (1.0 - a / L) / 2.0, "rear": W * c * (a / L) / 2.0}
        for cid, (ax, side) in CORNER_SIDE.items():
            dfz = (W_s * mshare[ax] * s * rc[ax] / t[ax]
                   + share[ax] * W_s * (s + c * phi_rad(theta)) * d / t[ax]
                   + m_u[ax] * GRAVITY * s * h_u[ax] / t[ax])
            out[cid] = static[ax] + (dfz if side < 0 else -dfz)   # right side is downhill
        return out

    # tipping line through the downhill (right) patches, at the CG station
    pf, pr = geo.p("FR", "CONTACT_PATCH"), geo.p("RR", "CONTACT_PATCH")
    x_cg = -a
    f = (x_cg - pf[0]) / (pr[0] - pf[0])
    y_line = -(pf[1] + f * (pr[1] - pf[1]))          # distance from the centreline, mm

    def shift(theta: float) -> float:
        return sm.mass_kg / veh.total_mass_kg * d * phi_rad(theta) + inp.lateral_shift_mm

    def tip_residual(theta: float) -> float:
        return math.tan(theta) * h_eff - (y_line - shift(theta))

    lo, hi = math.radians(1e-3), math.radians(89.0)
    notes: list[str] = []
    try:
        tip = math.degrees(brentq(tip_residual, lo, hi, xtol=1e-12))
    except ValueError:
        tip = float("nan")
        notes.append("tip angle not bracketed in (0°, 89°)")
    rigid = math.degrees(math.atan2(y_line, h))

    first: tuple[str, float] | None = None
    for cid in ("FL", "RL"):                         # uphill wheels
        def fz(theta: float, cid: str = cid) -> float:
            return loads(theta)[cid]
        if fz(hi) < 0:
            ang = math.degrees(brentq(fz, lo, hi, xtol=1e-12))
            if first is None or ang < first[1]:
                first = (cid, ang)

    th = math.radians(inp.angle_deg)
    max_h = (y_line - shift(th)) / math.tan(th) - inp.cg_rise_mm
    if d <= 0:
        notes.append(f"roll axis at/above the sprung CG (d = {d:.1f} mm)")
    return TiltResult(inp, loads(th), grad * math.sin(th), shift(th), y_line, h_eff, tip, rigid,
                      first, max_h, rc, d, sm.mass_kg / veh.total_mass_kg, notes)
