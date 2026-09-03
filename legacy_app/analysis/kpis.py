"""
analysis/kpis.py
================
Computation of additional KPIs for the suspension geometry.

Complements the basic KPIs (Caster, KPI, Camber, Scrub, Trail, RC Height)
computed in `geometry/model_3d.py` with:

    - Wheelbase, Track Width
    - Static Sum Toe
    - Steer Ratio, C-factor, Steer Arm Length

NOTE: some KPIs (wheel rate, motion ratio, damping) depend on external
parameters (spring stiffness, rocker geometry, damper data) and are not
computable from pure kinematics. Those remain USER INPUTS in the app, not
computed values.

Dynamic KPIs (camber gain, RC migration, Ackermann %, anti-dive/anti-squat)
are computed by `vdcore`/`DWSolver` via `analysis/vdcore_bridge.py`, and
static anti-dive/anti-squat by `model_3d.SuspensionCorner`. This module no
longer duplicates them: the old module-level implementations here modeled
each wishbone as a strut to the midpoint of its chassis pivots, which does
not close the real linkage — see CLAUDE.md.
"""

from __future__ import annotations

import math

import numpy as np

from geometry.model_3d import SuspensionCorner
from geometry.solver_3d import TieRod


# =============================================================================
# General vehicle dimensions
# =============================================================================

def wheelbase_mm(
    front_corner: SuspensionCorner,
    rear_corner:  SuspensionCorner,
) -> float:
    """
    Wheelbase: longitudinal (X) distance between the front WC and the rear WC
    on the SAME side.
    """
    return abs(front_corner.wheel_center.x - rear_corner.wheel_center.x)


def track_width_mm(
    left_corner:  SuspensionCorner,
    right_corner: SuspensionCorner,
) -> float:
    """
    Track width: lateral (Y) distance between the left WC and the right WC
    on the SAME axle (front or rear).
    """
    return abs(left_corner.wheel_center.y - right_corner.wheel_center.y)


# =============================================================================
# Static toe and Sum Toe
# =============================================================================

def static_toe_deg(
    corner: SuspensionCorner,
    tie_rod: TieRod,
) -> float:
    """
    Absolute static toe of this wheel, in degrees.

    CONVENTION:
        + = toe-in (wheel pointing toward the vehicle center)
        − = toe-out

    DEFINITION:
        Toe is the angle between the direction the wheel points (in the XY
        plane) and the vehicle's longitudinal X axis.

        Since the "point defining the front of the wheel" is not a hardpoint,
        we use the convention that the WC and CP are aligned in the wheel
        plane. For a perfectly neutral wheel (toe=0), the CP is at (X_wc,
        Y_wc, 0) — exactly below the WC.

        If there is a longitudinal offset between CP and WC (CP.x != WC.x),
        the wheel has toe.

    NOTE: for a symmetric assembly with the CP exactly below the WC in XY,
    this value is always 0. To introduce static toe, the user can offset the
    CP in X (or rotate the upright constructively).
    """
    wc = corner.wheel_center
    cp = corner.contact_patch

    # CP→WC vector projected onto XY: defines the wheel's longitudinal direction
    dx = wc.x - cp.x
    dy = wc.y - cp.y

    # For a neutral wheel, dx=0 and dy=0 (CP exactly below the WC) → toe = 0
    # If dx != 0 but dy = 0, it indicates a pure longitudinal offset: still toe = 0
    # Real toe = angle between the WHEEL AXIS (perpendicular to the hub axis)
    # and the X axis. Since the definition depends on the upright orientation,
    # we return 0 for symmetric geometries and the derived angle if the WC and
    # CP are rotated in XY.

    # Approximation: use the angle of the "WC forward" vector in XY.
    # If WC and CP coincide in XY, the toe is 0 (neutral geometry).
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return 0.0

    # If the user placed the CP offset, compute the toe relative to the YZ
    # plane (lateral axis). Positive toe (in) means the front of the wheel
    # points inward.
    # For the left (Y>0): front inward = +X (front) has smaller Y
    # For the right (Y<0): front inward = +X has larger Y
    angle = math.degrees(math.atan2(dx, abs(dy) + 1e-12))

    # Convention: small angle → toe ~0
    # If WC.y > 0 (left) and dx > 0 → front of the wheel pointing outward? no.
    # We keep the simple convention: small magnitude, sign per side.
    if abs(angle) > 45:
        # The CP was probably placed wrong, return 0
        return 0.0

    return angle if corner.wheel_center.y > 0 else -angle


def static_sum_toe_deg(
    left_corner:  SuspensionCorner, left_tie_rod:  TieRod,
    right_corner: SuspensionCorner, right_tie_rod: TieRod,
) -> float:
    """
    Static Sum Toe (degrees): sum of the static toe of both wheels on the same axle.

    CONVENTION (picture):
        + = total toe-in (both converging)
        − = total toe-out (both diverging)

    It is the value that appears on the car's setup sheet.
    """
    return (
        static_toe_deg(left_corner, left_tie_rod)
      + static_toe_deg(right_corner, right_tie_rod)
    )


# =============================================================================
# Steering arm length
# =============================================================================

def steering_arm_lengths(
    front_left_corner:  SuspensionCorner, fl_tie_rod: TieRod,
    front_right_corner: SuspensionCorner, fr_tie_rod: TieRod,
) -> dict[str, float]:
    """
    Steering-arm length (mm) at each front wheel: perpendicular distance from
    the tie-rod outboard point to the kingpin axis.

    Ackermann % is NOT computed here — use
    `steering_geometry._geometric_ackermann_pct` (the source of truth per
    CLAUDE.md) or the solved value from `vdcore_bridge`.
    """
    return {
        "steer_arm_length_left":  _steering_arm_length(front_left_corner, fl_tie_rod),
        "steer_arm_length_right": _steering_arm_length(front_right_corner, fr_tie_rod),
    }


def _steering_arm_length(corner: SuspensionCorner, tie_rod: TieRod) -> float:
    """
    Steering-arm length: perpendicular distance from the TRO to the kingpin
    axis (= effective steering radius).
    """
    ubj = corner.upper_arm.outboard.to_array()
    lbj = corner.lower_arm.outboard.to_array()
    tro = tie_rod.outboard.to_array()

    kp = ubj - lbj
    kp_norm = float(np.linalg.norm(kp))
    if kp_norm < 1e-12:
        return 0.0
    kp_unit = kp / kp_norm

    # Vector from LBJ to TRO, component perpendicular to the kingpin
    v = tro - lbj
    v_perp = v - np.dot(v, kp_unit) * kp_unit
    return float(np.linalg.norm(v_perp))


# =============================================================================
# Steer Ratio (rack)
# =============================================================================

def steer_ratio_from_pinion(
    rack_per_wheel_deg_mm_per_deg: float,
    c_factor_mm_per_rev: float,
) -> float:
    """
    Compute Steer Ratio (steering wheel:road wheel) from the rack C-factor.

    Steer Ratio = (steering-wheel degrees per road-wheel degree)
                = (mm_rack/wheel_deg) / (mm_rack/steering_wheel_deg)
                = rack_per_wheel_deg / (c_factor / 360)
    """
    if c_factor_mm_per_rev <= 0:
        return float("inf")
    rack_per_wheel_deg = abs(rack_per_wheel_deg_mm_per_deg)
    rack_per_steer_wheel_deg = c_factor_mm_per_rev / 360.0
    return rack_per_wheel_deg / rack_per_steer_wheel_deg

