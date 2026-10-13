"""Wishbone, tie-rod and pushrod forces of the outer wheels at a maneuver state.

Every case is solved by the six-link equilibrium of the upright
(:func:`vdcore.analysis.upright_loads.upright_link_forces`) on the merged
corner. That includes braking, so the old "not computed" rows are gone. Loads
are the tyre forces the road applies to the OUTER wheel of each axle, with the
Fz of the maneuver state and the friction coefficients from vehicle data
(assumptions: no tyre data):

    Cornering  (0, ±μy·Fz, Fz)         Fy toward the turn centre
    Braking    (−μx·Fz, 0, Fz)
    Combined   (−0.7·μx·Fz, ±0.7·μy·Fz, Fz)   (sla_geometry's combined case)
    Traction   (+μx·Fz, 0, Fz)          rear axle only (rear-wheel drive)

A longitudinal force acts at the contact patch when its torque reacts on the
upright (outboard brakes) and at the wheel centre when it reacts on the
chassis (inboard brakes, a chassis diff with driveshafts), per
``Project.layout``. Fy and Fz always act at the patch. The system is linear,
so the two parts are solved separately and added.

Static geometry (ride height); the pushrod line is an ``estimate``
(``Project.pushrod_front/rear``). Frame: ISO 8855; N; + = tension.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from vdcore.analysis.upright_loads import (
    LINKS,
    SingularLinkLayoutError,
    UprightLinkForces,
    upright_link_forces,
)

from .geometry import CarGeometry
from .maneuver import ManeuverResult
from .project import Project, Pushrod

COMBINED_SHARE = 0.7       # sla_geometry.load_cases' combined case

LINK_LABELS: dict[str, str] = {
    "lca_front": "LCA front", "lca_rear": "LCA rear", "uca_front": "UCA front",
    "uca_rear": "UCA rear", "tie_rod": "tie rod", "pushrod": "pushrod",
}
LINK_ORDER: tuple[str, ...] = ("lca_front", "lca_rear", "uca_front", "uca_rear", "tie_rod",
                               "pushrod")
assert set(LINK_ORDER) == set(LINKS)


@dataclass
class LegCase:
    axle: str
    case: str
    corner: str
    fz_N: float
    load_N: tuple[float, float, float]
    fx_point: str                         # "contact patch" / "wheel centre"
    forces: dict[str, float] = field(default_factory=dict)
    error: str = ""


def pushrod_line(geo: CarGeometry, cid: str, rod: Pushrod) -> tuple[np.ndarray, np.ndarray]:
    """(outboard, inboard) of the pushrod for this corner, ISO 8855."""
    side = np.array([1.0, 1.0 if cid in ("FL", "RL") else -1.0, 1.0])
    lbj = geo.p(cid, "LCA_OUT")
    return (lbj + side * np.asarray(rod.outboard_from_lbj_mm, float),
            lbj + side * np.asarray(rod.inboard_from_lbj_mm, float))


def _cases(axle: str) -> tuple[str, ...]:
    base = ("Cornering", "Braking", "Combined")
    return (*base, "Traction") if axle == "rear" else base


def leg_cases(geo: CarGeometry, project: Project, r: ManeuverResult) -> list[LegCase]:
    """Every load case on the outer (most loaded) wheel of each axle."""
    veh = geo.vehicle_data
    mu_y, mu_x = veh.mu_lateral, veh.mu_longitudinal
    out: list[LegCase] = []
    for axle, cids in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        outer = max(cids, key=lambda c: r.wheels[c].fz_N)
        fz = float(r.wheels[outer].fz_N)
        # the road pushes the outer tyre toward the turn centre: +Y on a right corner
        toward = 1.0 if outer in ("FR", "RR") else -1.0
        rod = project.pushrod_front if axle == "front" else project.pushrod_rear
        lay = geo.layout
        brake_wc = lay.front_brakes_inboard if axle == "front" else lay.rear_brakes_inboard
        cp, wc = geo.p(outer, "CONTACT_PATCH"), geo.p(outer, "WHEEL_CENTER")
        pr_o, pr_i = pushrod_line(geo, outer, rod)
        corner = geo.corner(outer)
        for case in _cases(axle):
            fy = {"Cornering": mu_y, "Combined": COMBINED_SHARE * mu_y}.get(case, 0.0) * fz
            fx = {"Braking": -mu_x, "Combined": -COMBINED_SHARE * mu_x,
                  "Traction": mu_x}.get(case, 0.0) * fz
            at_wc = (lay.rear_drive_inboard if case == "Traction" else brake_wc) and fx != 0.0
            lc = LegCase(axle, case, outer, fz, (fx, toward * fy, fz),
                         "wheel centre" if at_wc else "contact patch")
            if not fz > 0:
                lc.error = "not computed: outer wheel not loaded"
                out.append(lc)
                continue
            try:
                lat: UprightLinkForces = upright_link_forces(corner, (0.0, toward * fy, fz), cp,
                                                             pr_o, pr_i)
                lon = upright_link_forces(corner, (fx, 0.0, 0.0), wc if at_wc else cp, pr_o,
                                          pr_i)
            except SingularLinkLayoutError as exc:
                lc.error = f"not computed: {exc}"
                out.append(lc)
                continue
            a, b = lat.as_dict(), lon.as_dict()
            lc.forces = {k: a[k] + b[k] for k in LINK_ORDER}
            out.append(lc)
    return out
