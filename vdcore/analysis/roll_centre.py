"""Roll centre analysis via front-view instant centre construction.

Coordinate system: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up.

Construction (Milliken RCVD Ch. 17):
  1. For each wishbone (UCA, LCA), find the effective inboard pivot
     in the front-view (Y-Z) plane: the point on the pivot axis at
     the same X as the outboard ball joint.
  2. In the Y-Z plane, draw lines from effective pivot to ball joint
     for both UCA and LCA. Their intersection is the front-view
     instant centre (FVIC).
  3. Draw a line from the tyre contact patch through the FVIC.
  4. The roll centre is where that line crosses the vehicle centreline
     (Y = 0).

The FVIC can be at infinity when the two front-view lines are
parallel (equal-length, parallel arms). This is detected and
reported.

All functions require solved joint positions (UBJ, LBJ, contact
patch). There is no silent fallback to static hardpoints -- a
call site that forgets to pass solved state gets a TypeError, not
a plausible wrong answer.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel

from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Axle, Corner, DerivedPoint


class FVICResult(BaseModel, frozen=True):
    """Front-view instant centre result for one corner.

    All coordinates in ISO 8855 (Y+ left, Z+ up).
    fvic_y_mm, fvic_z_mm: instant centre position in the Y-Z plane.
    is_finite: False when FVIC is at infinity (parallel arms).
    fvsa_mm: front-view swing arm length (distance from contact patch
      to FVIC, projected into Y-Z). Positive = IC on the same side as
      the wheel. Negative = IC on the opposite side (crossed arms).
      Infinite when arms are parallel.
    """

    fvic_y_mm: float
    fvic_z_mm: float
    is_finite: bool
    fvsa_mm: float
    corner_id: str


class RollCentreResult(BaseModel, frozen=True):
    """Roll centre height from front-view instant centre construction.

    rc_height_mm: Z coordinate where the CP-to-FVIC line crosses Y=0.
    Positive = above ground. Can be negative (below ground) for certain
    geometries -- this is physically valid and common in FSAE.

    rc_y_mm: Y coordinate of the roll centre. For a symmetric axle at
    static this is zero. Under asymmetric travel (one side in bump, the
    other in droop) the RC migrates laterally -- this lateral migration
    is what causes jacking.
    """

    rc_height_mm: float
    rc_y_mm: float
    left_fvic: FVICResult
    right_fvic: FVICResult


class RollCentreMigrationResult(BaseModel, frozen=True):
    """RC height and lateral position vs wheel travel."""

    wheel_travel_mm: list[float]
    rc_height_mm: list[float]
    rc_y_mm: list[float]
    converged: list[bool]


def _effective_pivot_at_x(
    inboard_front: np.ndarray,
    inboard_rear: np.ndarray,
    x_target: float,
) -> np.ndarray:
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
        return 0.5 * (inboard_front + inboard_rear)

    t = (x_target - inboard_front[0]) / dx
    return inboard_front + t * axis


def _line_intersect_yz(
    p1_y: float, p1_z: float,
    p2_y: float, p2_z: float,
    p3_y: float, p3_z: float,
    p4_y: float, p4_z: float,
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


def front_view_instant_centre(
    corner: Corner,
    ubj: DerivedPoint,
    lbj: DerivedPoint,
    contact_patch: DerivedPoint,
) -> FVICResult:
    """Compute the front-view instant centre for a suspension corner.

    ISO 8855: Y+ left, Z+ up.

    The FVIC is found by intersecting, in the Y-Z plane, the lines
    from each arm's effective inboard pivot to its outboard ball joint.

    The effective inboard pivot is the point on the pivot axis at the
    same X as the outboard ball joint. This correctly handles non-planar
    (3D) suspension geometry by projecting through the correct X-plane.

    Args:
        corner: The Corner model -- provides inboard pivot axes (these
            are chassis-fixed and do not move with wheel travel in the
            simple heave model) and corner_id.
        ubj: Solved upper ball joint position.
        lbj: Solved lower ball joint position.
        contact_patch: Solved contact patch position.
    """
    uca_if = np.array([corner.uca_inboard_front.x_mm,
                       corner.uca_inboard_front.y_mm,
                       corner.uca_inboard_front.z_mm])
    uca_ir = np.array([corner.uca_inboard_rear.x_mm,
                       corner.uca_inboard_rear.y_mm,
                       corner.uca_inboard_rear.z_mm])
    lca_if = np.array([corner.lca_inboard_front.x_mm,
                       corner.lca_inboard_front.y_mm,
                       corner.lca_inboard_front.z_mm])
    lca_ir = np.array([corner.lca_inboard_rear.x_mm,
                       corner.lca_inboard_rear.y_mm,
                       corner.lca_inboard_rear.z_mm])

    ubj_arr = np.array([ubj.x_mm, ubj.y_mm, ubj.z_mm])
    lbj_arr = np.array([lbj.x_mm, lbj.y_mm, lbj.z_mm])

    uca_pivot = _effective_pivot_at_x(uca_if, uca_ir, ubj_arr[0])
    lca_pivot = _effective_pivot_at_x(lca_if, lca_ir, lbj_arr[0])

    fvic_y, fvic_z, is_finite = _line_intersect_yz(
        uca_pivot[1], uca_pivot[2], ubj_arr[1], ubj_arr[2],
        lca_pivot[1], lca_pivot[2], lbj_arr[1], lbj_arr[2],
    )

    cp_y = contact_patch.y_mm

    if is_finite:
        fvsa = math.sqrt((fvic_y - cp_y) ** 2 + (fvic_z - 0.0) ** 2)
        is_left = corner.corner_id in ("FL", "RL")
        if is_left:
            if fvic_y < cp_y:
                fvsa = -fvsa
        else:
            if fvic_y > cp_y:
                fvsa = -fvsa
    else:
        fvsa = float("inf")

    return FVICResult(
        fvic_y_mm=fvic_y,
        fvic_z_mm=fvic_z,
        is_finite=is_finite,
        fvsa_mm=fvsa,
        corner_id=corner.corner_id,
    )


def roll_centre_height(
    axle: Axle,
    left_result: SolverResult,
    right_result: SolverResult,
) -> RollCentreResult:
    """Compute the roll centre for an axle.

    ISO 8855: Y+ left, Z+ up. Roll centre is the intersection of the
    left and right CP-to-FVIC lines in the Y-Z plane.

    Args:
        axle: The Axle model (provides inboard pivot axes).
        left_result: SolverResult for the left corner.
        right_result: SolverResult for the right corner.

    Construction (Milliken RCVD Ch. 17):
      For each side, draw a line from the ground-level contact patch
      through the FVIC. The roll centre is the geometric intersection
      of the left and right CP-to-FVIC lines in the Y-Z plane.

    Raises RuntimeError if either FVIC is at infinity (parallel arms)
    or if the two CP-to-FVIC lines are parallel.
    """
    left_fvic = front_view_instant_centre(
        axle.left,
        ubj=left_result.ubj,
        lbj=left_result.lbj,
        contact_patch=left_result.contact_patch,
    )
    right_fvic = front_view_instant_centre(
        axle.right,
        ubj=right_result.ubj,
        lbj=right_result.lbj,
        contact_patch=right_result.contact_patch,
    )

    if not left_fvic.is_finite:
        raise RuntimeError(
            f"Left corner {axle.left.corner_id}: FVIC is at infinity "
            "(UCA and LCA are parallel in front view). "
            "Roll centre is undefined."
        )
    if not right_fvic.is_finite:
        raise RuntimeError(
            f"Right corner {axle.right.corner_id}: FVIC is at infinity "
            "(UCA and LCA are parallel in front view). "
            "Roll centre is undefined."
        )

    left_cp_y = left_result.contact_patch.y_mm
    right_cp_y = right_result.contact_patch.y_mm

    rc_y, rc_z, rc_finite = _line_intersect_yz(
        left_cp_y, 0.0,
        left_fvic.fvic_y_mm, left_fvic.fvic_z_mm,
        right_cp_y, 0.0,
        right_fvic.fvic_y_mm, right_fvic.fvic_z_mm,
    )
    if not rc_finite:
        raise RuntimeError(
            "Left and right CP-to-FVIC lines are parallel -- "
            "roll centre is at infinity."
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

    Both sides receive the same wheel travel (parallel bump/droop).
    For roll-induced RC migration, use roll_centre_height() directly
    with asymmetric solver results.

    Args:
        axle: Front or rear axle.
        wheel_travel_min_mm: Start of sweep (negative = droop).
        wheel_travel_max_mm: End of sweep (positive = bump).
        steps: Number of evaluation points.

    Returns:
        RollCentreMigrationResult with RC height and Y position at
        each travel point.
    """
    import numpy as _np

    travel_vals = _np.linspace(wheel_travel_min_mm, wheel_travel_max_mm, steps)
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
            rc = roll_centre_height(axle, left_result=rl, right_result=rr)
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
    cp_y: float, cp_z: float,
    ic_y: float, ic_z: float,
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
            f"CP-to-IC line is vertical at Y={cp_y:.1f} mm -- "
            "no finite Z-intercept at Y=0."
        )

    slope = (ic_z - cp_z) / dy
    return cp_z + (0.0 - cp_y) * slope
