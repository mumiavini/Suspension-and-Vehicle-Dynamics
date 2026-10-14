"""Roll centre from the upright's instantaneous 3D motion.

Coordinate system: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up. Units mm, rad.
Every position handed to the instant-centre functions must be in the
CHASSIS frame: the frame the Corner's inboard pickups are written in. A
``DWSolver`` result is in the world frame (the chassis is displaced by
``-wheel_travel_mm``), so its points are raised by the travel first; the
functions below that take a ``SolverResult`` do this themselves, given the
travel. Every Z they return is therefore CHASSIS-referenced (CLAUDE.md).

Definition -- the force-based roll centre
-----------------------------------------
Milliken RCVD Ch. 17 and SAE J670 define the roll centre as the point in the
transverse vertical plane through the contact patches at which a lateral
force applied to the sprung mass produces no suspension roll. By virtual
work, a force at the contact patch does no work on the suspension degree of
freedom when it is perpendicular to the patch's instantaneous velocity
relative to the chassis. Each side's force line therefore runs from the
contact patch (CP) through the point about which the upright instantaneously
rotates in front view AT THE PATCH STATION, and the roll centre is where the
left and right lines cross.

Front-view instant centre (:func:`instant_axis_fvic`)
-----------------------------------------------------
1. Instantaneous motion (twist) of the upright, from positions only -- no
   extra solves, so a replay solver that can only return states it has
   already solved (the Altair MotionSolve cross-check) still works:

   - UBJ ``U`` rotates about the chassis-fixed UCA pivot axis ``a_U``
     through ``P_U``:  ``v_U = th_U * a_U x (U - P_U)``;
   - LBJ ``L`` likewise: ``v_L = th_L * a_L x (L - P_L)``;
   - the upright is rigid: ``v_U = v_L + w x (U - L)``  (3 equations);
   - the tie rod / toe link keeps ``|T - TRI|`` constant:
     ``(v_L + w x (T - L)) . (T - TRI) = 0``  (1 equation). ``TRI`` is the
     chassis-fixed inboard point, displaced along Y by the rack travel.

   Four homogeneous equations in five unknowns ``(th_U, th_L, w)``: a
   one-degree-of-freedom mechanism. The twist is the null vector of that
   4 x 5 matrix, found by SVD with a rank check; a rank below 4 means the
   linkage is singular and :class:`InstantKinematicsError` is raised.
2. Upright points in the plane ``x = x_CP`` have the front-view (Y, Z)
   velocity ``v(p) = v_L + w x (p - L)``, which is a planar rotation at rate
   ``w_x`` about one point: the FVIC, where ``v_y = v_z = 0``. Over all X
   these zero-front-view-velocity points form a line parallel to ``w``
   (the instantaneous screw axis itself when the twist has no pitch), so the
   front-view pole moves with X by ``(w_y / w_x, w_z / w_x)`` mm per mm.
   Taking it at the CP station is what makes the CP-to-FVIC line the
   force line above. ``w_x = 0`` puts the FVIC at infinity (parallel arms);
   that is reported, never replaced by a number.

Projected construction (:func:`projected_fvic`), kept as a cross-check
---------------------------------------------------------------------
Intersect, in front view, each arm's line from its pivot axis (taken at its
own ball joint's X) to the ball joint. Each such line is the normal to that
ball joint's front-view velocity at that joint's X station, so it passes
through the front-view pole OF THAT STATION. The construction is therefore
exact only when both ball joints and the contact patch share one X station,
or the upright's rotation has no pitch or yaw component (``w_y = w_z = 0``).
With caster the two ball joints sit at different X and the two lines pass
through different points; on the 2027 front (UBJ x = -12.3, LBJ x = +10.1,
CP x = 0) the projected RC migrates -0.311 mm/mm against the instant-axis
-0.200 mm/mm.

All functions require solved joint positions. There is no silent fallback to
static hardpoints -- a call site that forgets to pass solved state gets a
TypeError, not a plausible wrong answer.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Axle, Corner, DerivedPoint, Hardpoint

Arr = NDArray[np.float64]

__all__ = [
    "FVICResult",
    "InstantKinematicsError",
    "RollCentreMigrationResult",
    "RollCentreResult",
    "UprightTwist",
    "front_view_instant_centre",
    "instant_axis_fvic",
    "projected_fvic",
    "roll_centre_height",
    "roll_centre_migration",
    "upright_twist",
]

# Smallest-to-largest singular value ratio of the 4 x 5 velocity-constraint
# matrix below which its rank is taken as < 4 (a singular linkage, whose
# instantaneous motion is not unique). Every entry of the matrix is a length
# in mm, so the ratio is dimensionless.
_RANK_TOL = 1e-9

# |w_x| relative to the unit null vector (th_U, th_L, w) below which the
# upright has no front-view rotation and the FVIC is at infinity. At 1e-9 the
# FVIC would already lie ~1e11 mm away; anything larger is a real (if long)
# swing arm. DWSolver's positions are good to ~1e-10 mm in practice (the
# analytic twist matches finite differences to 2e-9), which puts |w_x| near
# 1e-12 on a truly parallel geometry. A source at its convergence bound
# (~1e-5 mm, e.g. a replayed MotionSolve state) can reach ~1e-8: the FVIC then
# lies ~1e10 mm out along the arms, and each patch-to-FVIC line runs parallel
# to its arms -- the parallel-arm limit, not an invented number.
_PARALLEL_TOL = 1e-9

# |vertical wheel-centre velocity| relative to the largest point velocity of
# the twist below which the wheel centre is taken not to move vertically (a
# locked suspension that can only steer), so the twist has no per-mm-of-travel
# form. A real corner sits near 1; rounding on a truly zero value near 1e-16.
_NO_TRAVEL_TOL = 1e-9

# How far the solved positions may sit from the corner's link lengths before
# they are rejected as being in the wrong frame (world instead of chassis, or
# the wrong rack). DWSolver declares convergence at a residual norm of
# ~1.4e-5 mm, so this leaves a margin of ~70.
_FRAME_TOL_MM = 1e-3


class InstantKinematicsError(RuntimeError):
    """The linkage is singular: its instantaneous motion is not unique.

    Raised instead of returning an instant centre picked from a
    multi-dimensional null space.
    """


class FVICResult(BaseModel, frozen=True):
    """Front-view instant centre of one corner.

    ISO 8855 (Y+ left, Z+ up), CHASSIS frame, mm.

    fvic_y_mm, fvic_z_mm: the instant centre in the Y-Z plane. For
        :func:`instant_axis_fvic` it is the point of the plane
        ``x = x_CP`` whose instantaneous velocity relative to the chassis has
        no Y or Z component; for :func:`projected_fvic` it is the
        intersection of the projected arm lines.
    is_finite: False when the FVIC is at infinity (no front-view rotation,
        e.g. parallel equal-length arms). The coordinates and ``fvsa_mm``
        are then ``inf``.
    fvsa_mm: front-view swing arm, the distance from the contact patch to
        the FVIC in the Y-Z plane. Negative when the FVIC is INBOARD of the
        patch -- the usual SLA layout, including an FVIC beyond the
        centreline; positive when it is outboard of the patch. ``inf`` when
        the FVIC is at infinity.
    """

    fvic_y_mm: float
    fvic_z_mm: float
    is_finite: bool
    fvsa_mm: float
    corner_id: str


class UprightTwist(BaseModel, frozen=True):
    """Instantaneous motion of the upright relative to the chassis.

    ISO 8855, CHASSIS frame. Normalised per mm of wheel travel: the wheel
    centre's chassis-frame vertical velocity is +1 (bump positive), so every
    rate is a derivative with respect to ``wheel_travel_mm`` as
    :class:`~vdcore.geometry.solver.DWSolver` defines it.

    The velocity of an upright point ``p`` is
    ``lbj_velocity + omega x (p - lbj_mm)``.

    omega_rad_per_mm: angular velocity (rad per mm of travel).
    lbj_mm: the reference point (the LBJ position the twist was taken at).
    lbj_velocity: velocity of the LBJ (mm per mm of travel).
    uca_rate_rad_per_mm, lca_rate_rad_per_mm: rotation rates of the arms
        about their pivot axes, positive by the right-hand rule about the
        front-to-rear pivot direction.
    """

    omega_rad_per_mm: tuple[float, float, float]
    lbj_mm: tuple[float, float, float]
    lbj_velocity: tuple[float, float, float]
    uca_rate_rad_per_mm: float
    lca_rate_rad_per_mm: float

    def velocity_at(self, p: DerivedPoint) -> tuple[float, float, float]:
        """Velocity (mm per mm of travel) of the upright point at ``p``.

        ISO 8855, chassis frame.
        """
        w = np.array(self.omega_rad_per_mm, dtype=float)
        v = np.array(self.lbj_velocity, dtype=float) + np.cross(
            w, _vec(p) - np.array(self.lbj_mm, dtype=float)
        )
        return float(v[0]), float(v[1]), float(v[2])


class RollCentreResult(BaseModel, frozen=True):
    """Roll centre of an axle, from the two CP-to-FVIC force lines.

    ISO 8855 (Y+ left, Z+ up), CHASSIS frame, mm.

    rc_height_mm: Z where the left and right CP-to-FVIC lines cross.
        Chassis-referenced: at zero travel the chassis and ground frames
        coincide, so the static value is also the height above the road.
        Can be negative (below ground) -- physically valid.
    rc_y_mm: Y of the roll centre. Zero for a symmetric axle at symmetric
        travel; under asymmetric travel (one side in bump, the other in
        droop) the RC migrates laterally -- the lateral migration that causes
        jacking.
    """

    rc_height_mm: float
    rc_y_mm: float
    left_fvic: FVICResult
    right_fvic: FVICResult


class RollCentreMigrationResult(BaseModel, frozen=True):
    """RC height and lateral position vs symmetric wheel travel.

    ISO 8855, CHASSIS-referenced Z (mm). Ground-referenced RC height differs
    by exactly the wheel travel.
    """

    wheel_travel_mm: list[float]
    rc_height_mm: list[float]
    rc_y_mm: list[float]
    converged: list[bool]


def _vec(p: DerivedPoint | Hardpoint) -> Arr:
    """Point as a float array (ISO 8855, mm)."""
    return np.array([p.x_mm, p.y_mm, p.z_mm], dtype=float)


def _unit(v: Arr, what: str) -> Arr:
    """Unit vector along ``v``. Raises ValueError for a zero-length vector."""
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"{what} has zero length -- its direction is undefined.")
    out: Arr = v / n
    return out


def _skew(d: Arr) -> Arr:
    """Matrix ``S`` with ``S @ w == np.cross(d, w)``."""
    return np.array(
        [
            [0.0, -d[2], d[1]],
            [d[2], 0.0, -d[0]],
            [-d[1], d[0], 0.0],
        ],
        dtype=float,
    )


def _radial_and_axial(point: Arr, pivot: Arr, axis_unit: Arr) -> tuple[float, float]:
    """Distance of ``point`` from the axis, and its position along it (mm)."""
    rel = point - pivot
    axial = float(np.dot(rel, axis_unit))
    radial = float(np.linalg.norm(rel - axial * axis_unit))
    return radial, axial


def _check_chassis_frame(corner: Corner, ubj: Arr, lbj: Arr, tro: Arr, tri: Arr) -> None:
    """Reject positions that do not fit the corner's chassis-frame pickups.

    The instantaneous kinematics pair the solved ball joints with the
    Corner's chassis-fixed pivots, so both must be in the chassis frame. A
    world-frame ``DWSolver`` result at non-zero travel (or a wrong rack)
    violates the link lengths by roughly the frame offset; returning an
    instant centre from it would be a plausible-looking wrong number.

    Raises:
        ValueError: if any link length or axial position is off by more
            than ``_FRAME_TOL_MM``.
    """
    checks: list[tuple[str, float, float]] = []
    for name, front, rear, outboard_0, solved in (
        ("UCA", corner.uca_inboard_front, corner.uca_inboard_rear, corner.uca_outboard, ubj),
        ("LCA", corner.lca_inboard_front, corner.lca_inboard_rear, corner.lca_outboard, lbj),
    ):
        p_front = _vec(front)
        axis = _unit(_vec(rear) - p_front, f"{corner.corner_id} {name} pivot axis")
        r0, a0 = _radial_and_axial(_vec(outboard_0), p_front, axis)
        r1, a1 = _radial_and_axial(solved, p_front, axis)
        checks.append((f"{name} arm radius", r0, r1))
        checks.append((f"{name} position along its pivot axis", a0, a1))
    tr0 = float(np.linalg.norm(_vec(corner.tie_rod_outboard) - _vec(corner.tie_rod_inboard)))
    checks.append(("tie-rod length", tr0, float(np.linalg.norm(tro - tri))))
    for label, i, j, p, q in (
        ("UBJ-LBJ", "uca_outboard", "lca_outboard", ubj, lbj),
        ("UBJ-TRO", "uca_outboard", "tie_rod_outboard", ubj, tro),
        ("LBJ-TRO", "lca_outboard", "tie_rod_outboard", lbj, tro),
    ):
        d0 = float(np.linalg.norm(_vec(getattr(corner, i)) - _vec(getattr(corner, j))))
        checks.append((f"upright {label} distance", d0, float(np.linalg.norm(p - q))))

    bad = [(n, a, b) for n, a, b in checks if abs(a - b) > _FRAME_TOL_MM]
    if bad:
        detail = "; ".join(f"{n} {b:.4f} vs {a:.4f} mm" for n, a, b in bad)
        raise ValueError(
            f"{corner.corner_id}: solved positions do not fit this corner's "
            f"chassis-frame pickups ({detail}). Pass CHASSIS-frame points "
            "(DWSolver output raised by wheel_travel_mm) and the rack travel "
            "the solve used."
        )


def _twist_null_vector(corner: Corner, ubj: Arr, lbj: Arr, tro: Arr, rack_mm: float) -> Arr:
    """Unit null vector ``(th_U, th_L, w_x, w_y, w_z)`` of the velocity constraints.

    All inputs CHASSIS frame, ISO 8855, mm. See the module docstring for the
    four equations. Each row is scaled to units of length (the tie-rod row is
    divided by the tie-rod length), so the singular values are lengths and
    the rank tolerance is dimensionless. The sign of the result is arbitrary.

    Raises:
        ValueError: if the positions are not in this corner's chassis frame.
        InstantKinematicsError: if the matrix rank is below 4.
    """
    tri = _vec(corner.tie_rod_inboard) + np.array([0.0, rack_mm, 0.0])
    _check_chassis_frame(corner, ubj, lbj, tro, tri)

    p_u = _vec(corner.uca_inboard_front)
    a_u = _unit(_vec(corner.uca_inboard_rear) - p_u, f"{corner.corner_id} UCA pivot axis")
    p_l = _vec(corner.lca_inboard_front)
    a_l = _unit(_vec(corner.lca_inboard_rear) - p_l, f"{corner.corner_id} LCA pivot axis")

    g_u = np.cross(a_u, ubj - p_u)  # v_U per unit th_U
    g_l = np.cross(a_l, lbj - p_l)  # v_L per unit th_L
    e_tr = _unit(tro - tri, f"{corner.corner_id} tie rod")

    a = np.zeros((4, 5), dtype=float)
    # Rigid upright: th_U g_U - th_L g_L - w x (U - L) = 0, and
    # -w x d = d x w = skew(d) w.
    a[0:3, 0] = g_u
    a[0:3, 1] = -g_l
    a[0:3, 2:5] = _skew(ubj - lbj)
    # Tie rod: (th_L g_L + w x (T - L)) . e = 0, and
    # (w x d) . e = w . (d x e).
    a[3, 1] = float(np.dot(g_l, e_tr))
    a[3, 2:5] = np.cross(tro - lbj, e_tr)

    _, s, vt = np.linalg.svd(a)
    if s[0] <= 0.0 or s[3] < _RANK_TOL * s[0]:
        raise InstantKinematicsError(
            f"{corner.corner_id}: the linkage is singular at this position "
            f"(velocity-constraint singular values {s[3]:.3e} vs {s[0]:.3e} mm) "
            "-- its instantaneous motion, and so its instant centre, is not unique."
        )
    null: Arr = vt[-1]
    return null


def _fvsa_signed(corner_id: str, cp_y: float, cp_z: float, ic_y: float, ic_z: float) -> float:
    """CP-to-FVIC distance, negative when the FVIC is inboard of the patch."""
    fvsa = math.hypot(ic_y - cp_y, ic_z - cp_z)
    inboard = ic_y < cp_y if corner_id in ("FL", "RL") else ic_y > cp_y
    return -fvsa if inboard else fvsa


def _infinite(corner_id: str) -> FVICResult:
    inf = float("inf")
    return FVICResult(
        fvic_y_mm=inf, fvic_z_mm=inf, is_finite=False, fvsa_mm=inf, corner_id=corner_id
    )


def upright_twist(
    corner: Corner,
    ubj: DerivedPoint,
    lbj: DerivedPoint,
    tro: DerivedPoint,
    wheel_center: DerivedPoint,
    rack_mm: float = 0.0,
) -> UprightTwist:
    """Instantaneous motion of the upright per mm of wheel travel.

    ISO 8855. All points CHASSIS frame (a DWSolver result raised by its
    ``wheel_travel_mm``). ``rack_mm`` is the rack travel of the solve; it
    displaces the tie-rod inboard point along Y. Computed analytically from
    the positions (see the module docstring) -- no extra solves -- and
    normalised so the wheel centre's chassis-frame vertical velocity is +1.

    Raises:
        ValueError: if the positions are not in this corner's chassis frame,
            or the wheel centre does not move vertically.
        InstantKinematicsError: if the linkage is singular.
    """
    u, low, t = _vec(ubj), _vec(lbj), _vec(tro)
    null = _twist_null_vector(corner, u, low, t, rack_mm)
    th_u, th_l, w = float(null[0]), float(null[1]), null[2:5]

    p_l = _vec(corner.lca_inboard_front)
    a_l = _unit(_vec(corner.lca_inboard_rear) - p_l, f"{corner.corner_id} LCA pivot axis")
    v_l = th_l * np.cross(a_l, low - p_l)
    v_wc = v_l + np.cross(w, _vec(wheel_center) - low)
    v_wc_z = float(v_wc[2])
    scale = max(float(np.linalg.norm(v_wc)), float(np.linalg.norm(v_l)))
    if scale == 0.0 or abs(v_wc_z) <= _NO_TRAVEL_TOL * scale:
        raise ValueError(
            f"{corner.corner_id}: the wheel centre has no vertical velocity "
            "at this position -- the twist cannot be expressed per mm of travel."
        )
    k = 1.0 / v_wc_z
    w_n, v_n = w * k, v_l * k
    return UprightTwist(
        omega_rad_per_mm=(float(w_n[0]), float(w_n[1]), float(w_n[2])),
        lbj_mm=(float(low[0]), float(low[1]), float(low[2])),
        lbj_velocity=(float(v_n[0]), float(v_n[1]), float(v_n[2])),
        uca_rate_rad_per_mm=th_u * k,
        lca_rate_rad_per_mm=th_l * k,
    )


def instant_axis_fvic(
    corner: Corner,
    ubj: DerivedPoint,
    lbj: DerivedPoint,
    tro: DerivedPoint,
    contact_patch: DerivedPoint,
    rack_mm: float = 0.0,
) -> FVICResult:
    """Front-view instant centre from the upright's 3D instantaneous motion.

    ISO 8855 (Y+ left, Z+ up). All points CHASSIS frame. The FVIC is the
    point of the plane ``x = contact_patch.x_mm`` whose velocity relative to
    the chassis has no Y or Z component; the line from the contact patch
    through it is the force line of the force-based roll centre (Milliken
    RCVD Ch. 17, SAE J670). See the module docstring for the derivation.

    Args:
        corner: provides the chassis-fixed pivot axes, the tie-rod inboard
            point and ``corner_id``.
        ubj, lbj, tro: solved upper/lower ball joint and tie-rod (toe-link)
            outboard positions, chassis frame.
        contact_patch: solved contact patch, chassis frame. Sets the X
            station of the front view and the origin of the FVSA.
        rack_mm: rack travel of the solve (displaces the tie-rod inboard
            point along Y). Zero for a rear toe link.

    Returns:
        FVICResult. ``is_finite`` is False -- coordinates and FVSA ``inf`` --
        when the upright has no front-view rotation (parallel arms).

    Raises:
        ValueError: if the positions are not in this corner's chassis frame.
        InstantKinematicsError: if the linkage is singular.
    """
    u, low, t = _vec(ubj), _vec(lbj), _vec(tro)
    null = _twist_null_vector(corner, u, low, t, rack_mm)
    th_l, w = float(null[1]), null[2:5]
    if abs(float(w[0])) <= _PARALLEL_TOL:
        return _infinite(corner.corner_id)

    p_l = _vec(corner.lca_inboard_front)
    a_l = _unit(_vec(corner.lca_inboard_rear) - p_l, f"{corner.corner_id} LCA pivot axis")
    v_l = th_l * np.cross(a_l, low - p_l)

    # v(p) = v_L + w x (p - L) with p = (x_cp, y, z). Setting v_y = v_z = 0:
    #   v_y = v_Ly + w_z dx - w_x dz = 0,   v_z = v_Lz + w_x dy - w_y dx = 0,
    # where d = p - L and dx = x_cp - L_x is fixed by the station.
    dx = contact_patch.x_mm - float(low[0])
    wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
    ic_y = float(low[1]) + (wy * dx - float(v_l[2])) / wx
    ic_z = float(low[2]) + (float(v_l[1]) + wz * dx) / wx

    return FVICResult(
        fvic_y_mm=ic_y,
        fvic_z_mm=ic_z,
        is_finite=True,
        fvsa_mm=_fvsa_signed(corner.corner_id, contact_patch.y_mm, contact_patch.z_mm, ic_y, ic_z),
        corner_id=corner.corner_id,
    )


def _effective_pivot_at_x(
    inboard_front: Arr,
    inboard_rear: Arr,
    x_target: float,
) -> Arr:
    """Find the point on the pivot axis at a given X coordinate.

    ISO 8855: X+ forward, Y+ left, Z+ up. All coordinates in mm.
    Returns a 3D point on the pivot axis.

    The pivot axis is the line from inboard_front to inboard_rear.
    We parameterise as P(t) = front + t * (rear - front), find the t
    where P(t).x = x_target, and return P(t).

    If the axis is parallel to the Y-Z plane (delta_x ~ 0), the
    front-view projection of the axis is a point -- use the midpoint.
    """
    axis = inboard_rear - inboard_front
    dx = axis[0]

    if abs(dx) < 1e-10:
        mid: Arr = 0.5 * (inboard_front + inboard_rear)
        return mid

    t = (x_target - inboard_front[0]) / dx
    out: Arr = inboard_front + t * axis
    return out


def _line_intersect_yz(
    p1_y: float,
    p1_z: float,
    p2_y: float,
    p2_z: float,
    p3_y: float,
    p3_z: float,
    p4_y: float,
    p4_z: float,
) -> tuple[float, float, bool]:
    """Intersect two lines in the Y-Z plane.

    Line 1: through (p1_y, p1_z) and (p2_y, p2_z).
    Line 2: through (p3_y, p3_z) and (p4_y, p4_z).

    Returns (y, z, is_finite). If lines are parallel, returns
    (inf, inf, False).
    """
    d1_y = p2_y - p1_y
    d1_z = p2_z - p1_z
    d2_y = p4_y - p3_y
    d2_z = p4_z - p3_z

    denom = d1_y * d2_z - d1_z * d2_y

    if abs(denom) < 1e-10:
        return float("inf"), float("inf"), False

    t = ((p3_y - p1_y) * d2_z - (p3_z - p1_z) * d2_y) / denom

    y = p1_y + t * d1_y
    z = p1_z + t * d1_z
    return y, z, True


def projected_fvic(
    corner: Corner,
    ubj: DerivedPoint,
    lbj: DerivedPoint,
    contact_patch: DerivedPoint,
) -> FVICResult:
    """Front-view instant centre by intersecting the projected arm lines.

    ISO 8855: Y+ left, Z+ up. All points CHASSIS frame. A CROSS-CHECK: every
    vdcore roll centre uses :func:`instant_axis_fvic`.

    Each arm contributes the front-view line from its pivot axis, taken at
    the same X as its outboard ball joint, to that ball joint. That line is
    the normal to the ball joint's front-view velocity at the joint's own X
    station, and the front-view pole of a rigid upright moves with X by
    ``(w_y / w_x, w_z / w_x)`` mm per mm. The intersection is therefore the
    instant centre at the contact-patch station only when both ball joints
    and the patch share one X, or the upright neither pitches nor yaws
    (``w_y = w_z = 0``). With caster it is not: on the 2027 front it puts the
    static RC 0.13 mm high and the RC migration at -0.311 instead of
    -0.200 mm/mm.

    Args:
        corner: provides the chassis-fixed inboard pivot axes and corner_id.
        ubj: Solved upper ball joint position.
        lbj: Solved lower ball joint position.
        contact_patch: Solved contact patch position (origin of the FVSA).
    """
    uca_if = _vec(corner.uca_inboard_front)
    uca_ir = _vec(corner.uca_inboard_rear)
    lca_if = _vec(corner.lca_inboard_front)
    lca_ir = _vec(corner.lca_inboard_rear)

    ubj_arr = _vec(ubj)
    lbj_arr = _vec(lbj)

    uca_pivot = _effective_pivot_at_x(uca_if, uca_ir, float(ubj_arr[0]))
    lca_pivot = _effective_pivot_at_x(lca_if, lca_ir, float(lbj_arr[0]))

    fvic_y, fvic_z, is_finite = _line_intersect_yz(
        float(uca_pivot[1]),
        float(uca_pivot[2]),
        float(ubj_arr[1]),
        float(ubj_arr[2]),
        float(lca_pivot[1]),
        float(lca_pivot[2]),
        float(lbj_arr[1]),
        float(lbj_arr[2]),
    )
    if not is_finite:
        return _infinite(corner.corner_id)

    return FVICResult(
        fvic_y_mm=fvic_y,
        fvic_z_mm=fvic_z,
        is_finite=True,
        fvsa_mm=_fvsa_signed(
            corner.corner_id, contact_patch.y_mm, contact_patch.z_mm, fvic_y, fvic_z
        ),
        corner_id=corner.corner_id,
    )


def front_view_instant_centre(
    corner: Corner,
    ubj: DerivedPoint,
    lbj: DerivedPoint,
    contact_patch: DerivedPoint,
    tro: DerivedPoint | None = None,
    rack_mm: float = 0.0,
) -> FVICResult:
    """Front-view instant centre of one corner.

    ISO 8855: Y+ left, Z+ up. All points CHASSIS frame.

    With ``tro`` (the tie-rod / toe-link outboard point) this is
    :func:`instant_axis_fvic`, the definition every vdcore roll centre uses.
    Without it the sixth constraint of the upright is missing, its motion is
    undetermined, and the result is the :func:`projected_fvic` cross-check,
    which is exact only when both ball joints and the contact patch share
    one X station.

    Raises:
        ValueError: if ``rack_mm`` is given without ``tro``, or the
            positions do not fit this corner's chassis frame.
        InstantKinematicsError: if the linkage is singular.
    """
    if tro is None:
        if rack_mm != 0.0:
            raise ValueError(
                "rack_mm needs tro: the rack moves the tie rod, which only the "
                "instant-axis construction uses."
            )
        return projected_fvic(corner, ubj=ubj, lbj=lbj, contact_patch=contact_patch)
    return instant_axis_fvic(
        corner, ubj=ubj, lbj=lbj, tro=tro, contact_patch=contact_patch, rack_mm=rack_mm
    )


def _raised(p: DerivedPoint, dz_mm: float) -> DerivedPoint:
    """``p`` translated by ``dz_mm`` along Z (world -> chassis for DWSolver output)."""
    return DerivedPoint(x_mm=p.x_mm, y_mm=p.y_mm, z_mm=p.z_mm + dz_mm)


def roll_centre_height(
    axle: Axle,
    left_result: SolverResult,
    right_result: SolverResult,
    *,
    left_travel_mm: float = 0.0,
    right_travel_mm: float = 0.0,
    rack_mm: float = 0.0,
) -> RollCentreResult:
    """Force-based roll centre of an axle from two solved corners.

    ISO 8855: Y+ left, Z+ up. The RC is the intersection of the left and
    right CP-to-FVIC lines in the Y-Z plane (Milliken RCVD Ch. 17, SAE J670),
    each FVIC from :func:`instant_axis_fvic`.

    Args:
        axle: The Axle model (provides the chassis-side pickups).
        left_result, right_result: ``DWSolver`` results, as solved: world
            frame, chassis displaced by ``-wheel_travel_mm``, ``roll_deg = 0``.
        left_travel_mm, right_travel_mm: the wheel travel each result was
            solved at. Used to bring the points into the CHASSIS frame, so
            the returned Z is chassis-referenced. Zero (the default) for a
            static solve, where the two frames coincide.
        rack_mm: the rack travel both results were solved at.

    Raises:
        RuntimeError: if either FVIC is at infinity (parallel arms) or the
            two CP-to-FVIC lines are parallel.
        InstantKinematicsError: if either linkage is singular.
        ValueError: if a travel or the rack does not match the solve (the
            points then do not fit the chassis-frame pickups).
    """
    fvics: list[FVICResult] = []
    patches: list[DerivedPoint] = []
    for corner, res, travel in (
        (axle.left, left_result, left_travel_mm),
        (axle.right, right_result, right_travel_mm),
    ):
        cp = _raised(res.contact_patch, travel)
        fvic = instant_axis_fvic(
            corner,
            ubj=_raised(res.ubj, travel),
            lbj=_raised(res.lbj, travel),
            tro=_raised(res.tro, travel),
            contact_patch=cp,
            rack_mm=rack_mm,
        )
        if not fvic.is_finite:
            side = "Left" if corner is axle.left else "Right"
            raise RuntimeError(
                f"{side} corner {corner.corner_id}: FVIC is at infinity "
                "(the upright has no front-view rotation, e.g. parallel arms). "
                "Roll centre is undefined."
            )
        fvics.append(fvic)
        patches.append(cp)

    left_fvic, right_fvic = fvics
    left_cp, right_cp = patches
    rc_y, rc_z, rc_finite = _line_intersect_yz(
        left_cp.y_mm,
        left_cp.z_mm,
        left_fvic.fvic_y_mm,
        left_fvic.fvic_z_mm,
        right_cp.y_mm,
        right_cp.z_mm,
        right_fvic.fvic_y_mm,
        right_fvic.fvic_z_mm,
    )
    if not rc_finite:
        raise RuntimeError(
            "Left and right CP-to-FVIC lines are parallel -- roll centre is at infinity."
        )

    return RollCentreResult(
        rc_height_mm=rc_z,
        rc_y_mm=rc_y,
        left_fvic=left_fvic,
        right_fvic=right_fvic,
    )


def roll_centre_migration(
    axle: Axle,
    wheel_travel_min_mm: float = -25.0,
    wheel_travel_max_mm: float = 25.0,
    steps: int = 50,
) -> RollCentreMigrationResult:
    """RC height and lateral position vs symmetric wheel travel.

    ISO 8855. Both sides receive the same wheel travel (parallel bump/droop).
    Heights are CHASSIS-referenced, the frame of
    :func:`vdcore.analysis.axle.axle_rates`; ground-referenced height differs
    by exactly the travel. For roll-induced RC migration use
    :func:`vdcore.analysis.axle.axle_roll`.

    Args:
        axle: Front or rear axle.
        wheel_travel_min_mm: Start of sweep (negative = droop).
        wheel_travel_max_mm: End of sweep (positive = bump).
        steps: Number of evaluation points.

    Returns:
        RollCentreMigrationResult with RC height and Y position at each
        travel point. A point whose solve did not converge, or whose RC is
        undefined, is NaN with ``converged`` False.
    """
    travel_vals = np.linspace(wheel_travel_min_mm, wheel_travel_max_mm, steps)
    solver_l = DWSolver(axle.left)
    solver_r = DWSolver(axle.right)

    travel_list: list[float] = []
    rc_h_list: list[float] = []
    rc_y_list: list[float] = []
    conv_list: list[bool] = []

    for wt in travel_vals:
        wt_f = float(wt)
        rl = solver_l.solve(wheel_travel_mm=wt_f)
        rr = solver_r.solve(wheel_travel_mm=wt_f)

        travel_list.append(wt_f)

        if not (rl.converged and rr.converged):
            rc_h_list.append(float("nan"))
            rc_y_list.append(float("nan"))
            conv_list.append(False)
            continue

        try:
            rc = roll_centre_height(
                axle,
                left_result=rl,
                right_result=rr,
                left_travel_mm=wt_f,
                right_travel_mm=wt_f,
            )
            rc_h_list.append(rc.rc_height_mm)
            rc_y_list.append(rc.rc_y_mm)
            conv_list.append(True)
        except RuntimeError:
            rc_h_list.append(float("nan"))
            rc_y_list.append(float("nan"))
            conv_list.append(False)

    return RollCentreMigrationResult(
        wheel_travel_mm=travel_list,
        rc_height_mm=rc_h_list,
        rc_y_mm=rc_y_list,
        converged=conv_list,
    )


def _rc_height_from_cp_and_ic(
    cp_y: float,
    cp_z: float,
    ic_y: float,
    ic_z: float,
) -> float:
    """Z-intercept of the line from contact patch to instant centre at Y=0.

    ISO 8855: Y+ left, Z+ up. Returns the Z coordinate (mm) where the
    line crosses Y=0. Positive = above ground.

    If ic_y == cp_y the line is vertical and never crosses Y=0 (unless
    both are at Y=0). Raises ValueError in that case.
    """
    dy = ic_y - cp_y
    if abs(dy) < 1e-10:
        if abs(cp_y) < 1e-10:
            return ic_z
        raise ValueError(
            f"CP-to-IC line is vertical at Y={cp_y:.1f} mm -- no finite Z-intercept at Y=0."
        )

    slope = (ic_z - cp_z) / dy
    return cp_z + (0.0 - cp_y) * slope
