"""Anti-dive and anti-squat analysis via side-view instant centre construction.

Coordinate system: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up.

Construction (Milliken RCVD Ch. 17, eq. 17.21):
  1. For each wishbone (UCA, LCA), find the effective inboard pivot
     in the side-view (X-Z) plane: the point on the pivot axis at
     the same Y as the outboard ball joint.
  2. In the X-Z plane, draw lines from effective pivot to ball joint
     for both UCA and LCA. Their intersection is the side-view
     instant centre (SVIC).
  3. Draw a line from the tyre contact patch to the SVIC.
  4. The slope of that line, combined with wheelbase, CG height,
     and brake/drive bias, gives the anti-dive or anti-squat
     percentage.

The SVIC can be at infinity when the two side-view lines are
parallel (horizontal pivot axes -- the current 2027 geometry).
This is detected and reported as 0% anti.

Brake assumption: outboard brakes (the common FSAE case).

All functions require solved joint positions (UBJ, LBJ, contact
patch). There is no silent fallback to static hardpoints.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Corner, DerivedPoint


class SVICResult(BaseModel, frozen=True):
    """Side-view instant centre result for one corner.

    All coordinates in ISO 8855 (X+ forward, Z+ up).
    svic_x_mm, svic_z_mm: instant centre position in the X-Z plane.
    is_finite: False when SVIC is at infinity (parallel arms in side view).
    svsa_mm: side-view swing arm length (horizontal distance from the
      contact patch X to the SVIC X). Always positive. Infinite when
      arms are parallel.
    """

    svic_x_mm: float
    svic_z_mm: float
    is_finite: bool
    svsa_mm: float
    corner_id: str


class AntiResult(BaseModel, frozen=True):
    """Anti-dive or anti-squat percentage for one corner.

    anti_dive_pct: percentage for braking (front axle).
    anti_lift_pct: percentage for braking (rear axle).
    anti_squat_pct: percentage for traction (rear axle, RWD).
    svic: the underlying side-view instant centre result.
    """

    anti_dive_pct: float
    anti_lift_pct: float
    anti_squat_pct: float
    svic: SVICResult


class AntiSweepResult(BaseModel, frozen=True):
    """Anti-dive/squat vs wheel-travel sweep for one corner."""

    wheel_travel_mm: list[float]
    anti_dive_pct: list[float]
    anti_lift_pct: list[float]
    anti_squat_pct: list[float]
    converged: list[bool]
    corner_id: str


def _pivot_axis_xz(
    inboard_front: np.ndarray,
    inboard_rear: np.ndarray,
) -> tuple[float, float, float, float]:
    """Project a pivot axis onto the X-Z plane.

    Returns (x_front, z_front, x_rear, z_rear) -- the XZ coordinates
    of the two inboard pickup points. These define a line in the side
    view that represents the pivot axis direction.

    When both inboard pickups share the same Z the line is horizontal.
    """
    return (
        float(inboard_front[0]),
        float(inboard_front[2]),
        float(inboard_rear[0]),
        float(inboard_rear[2]),
    )


def _line_intersect_xz(
    p1_x: float, p1_z: float,
    p2_x: float, p2_z: float,
    p3_x: float, p3_z: float,
    p4_x: float, p4_z: float,
) -> tuple[float, float, bool]:
    """Intersect two lines in the X-Z plane.

    Line 1: through (p1_x, p1_z) and (p2_x, p2_z).
    Line 2: through (p3_x, p3_z) and (p4_x, p4_z).

    Returns (x, z, is_finite). If lines are parallel, returns
    (inf, inf, False).
    """
    d1_x = p2_x - p1_x
    d1_z = p2_z - p1_z
    d2_x = p4_x - p3_x
    d2_z = p4_z - p3_z

    denom = d1_x * d2_z - d1_z * d2_x

    if abs(denom) < 1e-10:
        return float("inf"), float("inf"), False

    t = ((p3_x - p1_x) * d2_z - (p3_z - p1_z) * d2_x) / denom

    x = p1_x + t * d1_x
    z = p1_z + t * d1_z
    return x, z, True


def side_view_instant_centre(
    corner: Corner,
    ubj: DerivedPoint,
    lbj: DerivedPoint,
    contact_patch: DerivedPoint,
) -> SVICResult:
    """Compute the side-view instant centre for a suspension corner.

    ISO 8855: X+ forward, Z+ up.

    The SVIC is found by intersecting the UCA and LCA *pivot axis*
    projections in the X-Z plane. Each pivot axis is the line from
    inboard_front to inboard_rear. When both axes are horizontal
    (front and rear inboard share the same Z), the projections are
    parallel and the SVIC is at infinity -- giving 0% anti.

    This is the correct construction per Milliken RCVD Ch. 17:
    project the pivot axes (not the arm lines) onto the side view.

    Args:
        corner: The Corner model -- provides inboard pivot axes and
            corner_id.
        ubj: Solved upper ball joint position (unused, kept for API
            consistency with front_view_instant_centre).
        lbj: Solved lower ball joint position (unused).
        contact_patch: Solved contact patch position (used for SVSA).
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

    uca_x1, uca_z1, uca_x2, uca_z2 = _pivot_axis_xz(uca_if, uca_ir)
    lca_x1, lca_z1, lca_x2, lca_z2 = _pivot_axis_xz(lca_if, lca_ir)

    svic_x, svic_z, is_finite = _line_intersect_xz(
        uca_x1, uca_z1, uca_x2, uca_z2,
        lca_x1, lca_z1, lca_x2, lca_z2,
    )

    svsa = abs(svic_x - contact_patch.x_mm) if is_finite else float("inf")

    return SVICResult(
        svic_x_mm=svic_x,
        svic_z_mm=svic_z,
        is_finite=is_finite,
        svsa_mm=svsa,
        corner_id=corner.corner_id,
    )


def anti_percent(
    corner: Corner,
    ubj: DerivedPoint,
    lbj: DerivedPoint,
    contact_patch: DerivedPoint,
    *,
    wheelbase_mm: float,
    cg_height_mm: float,
    brake_bias_front: float = 0.60,
    is_front_axle: bool,
) -> AntiResult:
    """Compute anti-dive, anti-lift, and anti-squat percentages.

    ISO 8855: X+ forward, Z+ up. Outboard brakes assumed.

    Reference: Milliken RCVD Ch. 17, eq. 17.21.

    For the front axle under braking:
        anti_dive_pct = brake_bias * tan(theta) * L / h * 100

    For the rear axle under braking:
        anti_lift_pct = (1 - brake_bias) * tan(theta) * L / h * 100

    For the rear axle under traction (RWD, diff on chassis):
        anti_squat_pct = tan(theta) * L / h * 100

    where theta is the angle from horizontal of the line from the
    contact patch to the SVIC.

    Args:
        corner: The Corner model with inboard pivot axes.
        ubj: Solved upper ball joint position.
        lbj: Solved lower ball joint position.
        contact_patch: Solved contact patch position.
        wheelbase_mm: Vehicle wheelbase in mm.
        cg_height_mm: CG height above ground in mm.
        brake_bias_front: Front brake bias as a fraction (0.0 to 1.0).
        is_front_axle: True for front axle corners, False for rear.
    """
    svic = side_view_instant_centre(corner, ubj, lbj, contact_patch)

    if not svic.is_finite or cg_height_mm <= 1e-6:
        return AntiResult(
            anti_dive_pct=0.0,
            anti_lift_pct=0.0,
            anti_squat_pct=0.0,
            svic=svic,
        )

    dx = svic.svic_x_mm - contact_patch.x_mm
    dz = svic.svic_z_mm - contact_patch.z_mm

    if abs(dx) < 1e-9:
        return AntiResult(
            anti_dive_pct=0.0,
            anti_lift_pct=0.0,
            anti_squat_pct=0.0,
            svic=svic,
        )

    # For front axle: SVIC is ahead of wheel (positive X direction).
    # For rear axle: SVIC is behind wheel (negative X direction).
    # tan(theta) uses the absolute slope from CP to SVIC.
    if is_front_axle:
        tan_theta = dz / dx if abs(dx) > 1e-9 else 0.0
    else:
        tan_theta = dz / (-dx) if abs(dx) > 1e-9 else 0.0

    l_over_h = wheelbase_mm / cg_height_mm

    anti_dive = 100.0 * brake_bias_front * tan_theta * l_over_h
    anti_lift = 100.0 * (1.0 - brake_bias_front) * tan_theta * l_over_h
    anti_squat = 100.0 * tan_theta * l_over_h

    return AntiResult(
        anti_dive_pct=anti_dive if is_front_axle else 0.0,
        anti_lift_pct=anti_lift if not is_front_axle else 0.0,
        anti_squat_pct=anti_squat if not is_front_axle else 0.0,
        svic=svic,
    )


def anti_sweep(
    corner: Corner,
    *,
    wheelbase_mm: float,
    cg_height_mm: float,
    brake_bias_front: float = 0.60,
    is_front_axle: bool,
    wheel_travel_min_mm: float = -25.0,
    wheel_travel_max_mm: float = 25.0,
    steps: int = 50,
) -> AntiSweepResult:
    """Anti-dive/squat vs wheel-travel sweep for one corner.

    ISO 8855: wheel_travel_mm positive = bump (wheel up relative to chassis).
    Non-converged points have anti values set to NaN.
    """
    travel_vals = np.linspace(wheel_travel_min_mm, wheel_travel_max_mm, steps)
    solver = DWSolver(corner)

    travel_list: list[float] = []
    dive_list: list[float] = []
    lift_list: list[float] = []
    squat_list: list[float] = []
    conv_list: list[bool] = []

    for wt in travel_vals:
        wt_f = float(wt)
        result = solver.solve(wheel_travel_mm=wt_f)
        travel_list.append(wt_f)

        if not result.converged:
            dive_list.append(float("nan"))
            lift_list.append(float("nan"))
            squat_list.append(float("nan"))
            conv_list.append(False)
            continue

        ar = anti_percent(
            corner,
            ubj=result.ubj,
            lbj=result.lbj,
            contact_patch=result.contact_patch,
            wheelbase_mm=wheelbase_mm,
            cg_height_mm=cg_height_mm,
            brake_bias_front=brake_bias_front,
            is_front_axle=is_front_axle,
        )
        dive_list.append(ar.anti_dive_pct)
        lift_list.append(ar.anti_lift_pct)
        squat_list.append(ar.anti_squat_pct)
        conv_list.append(True)

    return AntiSweepResult(
        wheel_travel_mm=travel_list,
        anti_dive_pct=dive_list,
        anti_lift_pct=lift_list,
        anti_squat_pct=squat_list,
        converged=conv_list,
        corner_id=corner.corner_id,
    )
