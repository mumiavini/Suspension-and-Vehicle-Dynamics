"""Link forces from the equilibrium of one upright: 6 two-force members, 6 equations.

The upright (with its wheel) is held by six links, each pinned at both ends, so
each carries a pure axial force:

    UCA front leg, UCA rear leg   (UBJ -> UCA inboard pivots)
    LCA front leg, LCA rear leg   (LBJ -> LCA inboard pivots)
    tie rod / toe link            (TRO -> TRI)
    pushrod                       (outboard mount -> inboard / rocker point)

Force and moment balance of the upright under the tyre load give a 6x6 linear
system A f = -[F; M], with column i = [u_i; (p_i - o) x u_i] for the unit
vector u_i from the link's upright end p_i toward its chassis end. ``f_i > 0``
is TENSION: a link in tension pulls the upright toward the chassis.

This replaces the closure used by ``sla_geometry.solve_ball_joint_forces``
(UCA as a single two-force member in the front view, no tie rod and no
pushrod). That closure balances only the moment about X, so any longitudinal
force -- braking, traction, combined -- leaves the moments about Y and Z
unbalanced.

Load application point (where the tyre force effectively acts on the upright):

* the CONTACT PATCH when the brake / drive torque reacts on the upright
  (outboard brakes, hub motors);
* the WHEEL CENTRE when it reacts on the chassis through a driveshaft
  (inboard brakes, a chassis-mounted diff): the shaft couple goes to the
  chassis and only the force reaches the bearing.

Scope -- stated wherever the numbers are shown:

* Static geometry (ride height). Link directions at other travels differ by
  the arms' rotation.
* The pushrod is a member of the upright acting through the given outboard
  point. For a pushrod mounted on a wishbone near the ball joint, this is the
  usual approximation: the arm's bending and the load sharing it causes are
  not modelled. This is not an FEA tool.
* The tyre load is an input. No tyre data exists, so the friction
  coefficients behind it are assumptions.

Frame: ISO 8855 (X+ forward, Y+ LEFT, Z+ up), mm, N, N·mm.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from vdcore.models.hardpoint import Corner, Hardpoint

Arr = NDArray[np.float64]

__all__ = [
    "LINKS",
    "SingularLinkLayoutError",
    "UprightLinkForces",
    "upright_link_forces",
]

LINKS: tuple[str, ...] = (
    "uca_front", "uca_rear", "lca_front", "lca_rear", "tie_rod", "pushrod",
)

# Above this condition number the 6 links are (nearly) unable to hold the
# upright in some direction: the forces would be huge and meaningless.
MAX_CONDITION = 1e8


class SingularLinkLayoutError(RuntimeError):
    """The six links cannot react a general load (a mechanism, or nearly one).

    Raised rather than returning forces, so a near-singular layout can never
    reach a report as plausible numbers.
    """


class UprightLinkForces(BaseModel, frozen=True):
    """Axial force in each link, N; positive = tension."""

    uca_front_N: float
    uca_rear_N: float
    lca_front_N: float
    lca_rear_N: float
    tie_rod_N: float
    pushrod_N: float
    residual_force_N: float          # |sum F| after the solve (should be ~0)
    residual_moment_Nmm: float       # |sum M| after the solve (should be ~0)
    condition_number: float

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, f"{k}_N")) for k in LINKS}


def _vec(p: Hardpoint | tuple[float, float, float] | Arr) -> Arr:
    if isinstance(p, Hardpoint):
        return np.array([p.x_mm, p.y_mm, p.z_mm], dtype=float)
    return np.asarray(p, dtype=float).reshape(3)


def upright_link_forces(
    corner: Corner,
    load_N: tuple[float, float, float] | Arr,
    load_point_mm: tuple[float, float, float] | Arr,
    pushrod_outboard_mm: tuple[float, float, float] | Arr,
    pushrod_inboard_mm: tuple[float, float, float] | Arr,
) -> UprightLinkForces:
    """Solve the six link forces for one tyre load on one corner (static geometry).

    Args:
        corner: the corner, ISO 8855; its static hardpoints are used.
        load_N: the force the ROAD applies to the tyre, (Fx, Fy, Fz), ISO 8855,
            e.g. braking on any wheel is Fx < 0, the outer wheel of a left
            turn (a right corner) has Fy > 0, vertical support is Fz > 0.
        load_point_mm: where that force acts on the upright: the contact patch,
            or the wheel centre for torque reacted on the chassis (see the
            module docstring).
        pushrod_outboard_mm, pushrod_inboard_mm: the pushrod line, ISO 8855,
            for THIS corner (mirror Y for a right corner yourself).

    Raises:
        SingularLinkLayoutError: if the links cannot hold the upright (coincident
            or parallel lines making a mechanism), or a link has zero length.
    """
    links = (
        (corner.uca_outboard, corner.uca_inboard_front),
        (corner.uca_outboard, corner.uca_inboard_rear),
        (corner.lca_outboard, corner.lca_inboard_front),
        (corner.lca_outboard, corner.lca_inboard_rear),
        (corner.tie_rod_outboard, corner.tie_rod_inboard),
        (pushrod_outboard_mm, pushrod_inboard_mm),
    )
    origin = _vec(corner.lca_outboard)
    a = np.empty((6, 6))
    for i, (up_end, ch_end) in enumerate(links):
        p, q = _vec(up_end), _vec(ch_end)
        d = q - p
        n = float(np.linalg.norm(d))
        if not n > 1e-6:
            raise SingularLinkLayoutError(f"{corner.corner_id}: link {LINKS[i]} has zero length")
        u = d / n
        a[:3, i] = u
        a[3:, i] = np.cross(p - origin, u)

    force = np.asarray(load_N, dtype=float).reshape(3)
    moment = np.cross(_vec(load_point_mm) - origin, force)
    rhs = -np.concatenate([force, moment])
    # Moments are in N·mm, forces in N: scale the moment rows by a typical arm so
    # the condition number measures geometry, not units.
    scale = max(float(np.linalg.norm(_vec(corner.uca_outboard) - origin)), 1.0)
    w = np.diag([1.0, 1.0, 1.0, 1.0 / scale, 1.0 / scale, 1.0 / scale])
    cond = float(np.linalg.cond(w @ a))
    if not math.isfinite(cond) or cond > MAX_CONDITION:
        raise SingularLinkLayoutError(
            f"{corner.corner_id}: the six links cannot hold the upright (condition "
            f"number {cond:.3g}); check for parallel or coincident link lines")
    f = np.linalg.solve(a, rhs)
    res = a @ f - rhs
    return UprightLinkForces(
        uca_front_N=float(f[0]), uca_rear_N=float(f[1]),
        lca_front_N=float(f[2]), lca_rear_N=float(f[3]),
        tie_rod_N=float(f[4]), pushrod_N=float(f[5]),
        residual_force_N=float(np.linalg.norm(res[:3])),
        residual_moment_Nmm=float(np.linalg.norm(res[3:])),
        condition_number=cond,
    )
