"""
analysis/kpis.py
================
Computation of additional KPIs for the suspension geometry.

Complements the basic KPIs (Caster, KPI, Camber, Scrub, Trail, RC Height)
computed in `geometry/model_3d.py` with:

    - Wheelbase, Track Width
    - Ride Camber (°/m) and Roll Camber (°/°)
    - Static Sum Toe
    - Static Ackermann (%)
    - Steer Ratio, C-factor, Steer Arm Length
    - Roll Center under lateral load (1g approximation)
    - Anti-dive / Anti-squat (simplified version)

NOTE: some KPIs (wheel rate, motion ratio, damping) depend on external
parameters (spring stiffness, rocker geometry, damper data) and are not
computable from pure kinematics. Those remain USER INPUTS in the app, not
computed values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geometry.primitives import Point3D
from geometry.model_3d import SuspensionCorner, Vehicle
from geometry.solver_3d import TieRod, KinematicSolver3D


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
# Dynamic camber — Ride Camber and Roll Camber
# =============================================================================

def ride_camber_deg_per_m(
    corner: SuspensionCorner,
    tie_rod: TieRod,
    heave_range_mm: float = 25.0,
) -> float:
    """
    Ride Camber: rate of camber change with heave, in °/m.

    Same as "camber gain", but in the picture's unit (°/m instead of °/mm).

    Computed by linear regression of camber × heave over a small range.
    """
    solver = KinematicSolver3D(corner, tie_rod)
    heaves = np.linspace(-heave_range_mm, heave_range_mm, 11)
    cambers = []
    for h in heaves:
        solver.reset_seed()
        cambers.append(solver.solve(float(h), 0.0, 0.0).camber_deg)

    # Linear regression: slope in °/mm
    slope_mm = float(np.polyfit(heaves, cambers, 1)[0])
    return slope_mm * 1000.0   # convert to °/m


def roll_camber_deg_per_deg(
    corner: SuspensionCorner,
    tie_rod: TieRod,
    roll_range_deg: float = 2.0,
) -> float:
    """
    Roll Camber: rate of camber change with chassis roll, in °/°.

    Characterizes how much camber the OUTER wheel gains when the chassis rolls.
    Typical FSAE value: -0.5 to -1.5 (negative camber increases with positive roll).

    Computed by linear regression of camber × roll.
    """
    solver = KinematicSolver3D(corner, tie_rod)
    rolls = np.linspace(-roll_range_deg, roll_range_deg, 11)
    cambers = []
    for r in rolls:
        solver.reset_seed()
        cambers.append(solver.solve(0.0, float(r), 0.0).camber_deg)

    slope = float(np.polyfit(rolls, cambers, 1)[0])
    return slope


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
# Ackermann and steering geometry
# =============================================================================

def ackermann_geometry(
    front_left_corner:  SuspensionCorner, fl_tie_rod: TieRod,
    front_right_corner: SuspensionCorner, fr_tie_rod: TieRod,
    rear_corner: SuspensionCorner,
) -> dict[str, float]:
    """
    Compute the static Ackermann geometry.

    DEFINITION:
        Pure 100% Ackermann = when the extensions of the steering-arm lines
        (from the kingpin to the tie-rod outboard point, projected onto the
        horizontal plane) meet exactly on the rear axle.

        0% Ackermann = when those lines are parallel (inner and outer wheels
        steer by the same angle).

    COMPUTATION:
        1. For each front wheel, determine the point where the kingpin
           crosses the plane at the tie-rod height.
        2. Draw a line from that point to the TRO, projected onto XY.
        3. See where those two lines cross in X (longitudinal).
        4. Compare with the rear-axle position:
              x_inter = x_rear_axle  → 100% Ackermann
              x_inter = -∞           → 0% Ackermann
        5. Ackermann (%) = wheelbase / (x_kpi - x_intersect) × 100

    Returns a dict with:
        ackermann_percent : % of static Ackermann
        wheelbase_mm
        steer_arm_length_left, _right : steering-arm length (mm)
    """
    # --- Steering-arm length at each wheel ---
    # Steering arm = distance from the tie-rod outboard to the kingpin
    sa_l = _steering_arm_length(front_left_corner, fl_tie_rod)
    sa_r = _steering_arm_length(front_right_corner, fr_tie_rod)

    # --- Kingpin points in the horizontal plane (TRO height) ---
    # We approximate the "Ackermann line" as TRO → projection of the kingpin
    # onto XY at the TRO height.
    def kpi_at_height(corner: SuspensionCorner, z_target: float) -> tuple[float, float]:
        """Return (X, Y) of the kingpin axis at height z_target."""
        lbj = corner.lower_arm.outboard.to_array()
        ubj = corner.upper_arm.outboard.to_array()
        kp = ubj - lbj
        if abs(kp[2]) < 1e-12:
            return (float(lbj[0]), float(lbj[1]))
        t = (z_target - lbj[2]) / kp[2]
        p = lbj + t * kp
        return (float(p[0]), float(p[1]))

    fl_tro = fl_tie_rod.outboard
    fr_tro = fr_tie_rod.outboard
    kpi_l = kpi_at_height(front_left_corner,  fl_tro.z)
    kpi_r = kpi_at_height(front_right_corner, fr_tro.z)

    # --- Kingpin → TRO lines in the XY plane, extended rearward ---
    # Line equation: P(t) = KPI + t · (TRO - KPI)
    # For large t > 1, it goes beyond the TRO; we want to find where the two cross.
    def line_intersection_xy(p1, p2, p3, p4):
        """Intersection of lines (p1,p2) and (p3,p4) in the XY plane."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return None  # parallel lines → 0% Ackermann
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    intersect = line_intersection_xy(
        kpi_l, (fl_tro.x, fl_tro.y),
        kpi_r, (fr_tro.x, fr_tro.y),
    )

    wb = wheelbase_mm(front_left_corner, rear_corner)

    if intersect is None:
        ackermann_pct = 0.0
    else:
        # Take the average X of the front kingpins as the front-axle reference
        x_front = 0.5 * (kpi_l[0] + kpi_r[0])
        x_rear_target = front_left_corner.wheel_center.x - wb  # rear axle
        x_inter = intersect[0]

        # Ackermann% = (front_axle - x_inter) / (front_axle - rear_axle) × 100
        denom = x_front - x_rear_target
        if abs(denom) < 1e-9:
            ackermann_pct = 0.0
        else:
            ackermann_pct = float((x_front - x_inter) / denom * 100.0)

    return {
        "ackermann_percent":       ackermann_pct,
        "wheelbase_mm":            wb,
        "steer_arm_length_left":   sa_l,
        "steer_arm_length_right":  sa_r,
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
# Steer Ratio and C-factor (rack)
# =============================================================================

def steer_ratio_and_cfactor(
    corner: SuspensionCorner,
    tie_rod: TieRod,
    *,
    rack_test_mm: float = 5.0,
) -> dict[str, float]:
    """
    Compute Steer Ratio (steering wheel:road wheel) and C-factor (mm of rack
    per steering-wheel revolution).

    DEFINITIONS:
        C-factor (mm/rev) : rack displacement for 1 full revolution of the
                            pinion (360°). It DEPENDS ON THE RACK PINION,
                            which is a USER INPUT — here we return what can be
                            computed: the rack → wheel ratio in units of mm of
                            rack per degree of road wheel.

        Steer Ratio (x:1) : steering-wheel degrees per road-wheel degree.
                            Depends on the C-factor + what we compute here.

    INTERMEDIATE OUTPUT:
        rack_per_wheel_deg_mm_per_deg : how many mm of rack are needed
                                        for 1° of wheel steering.

    To obtain the final Steer Ratio, the user multiplies by:
        steer_ratio = c_factor / 360 / rack_per_wheel_deg_mm_per_deg
    """
    # Solve at rack=0 and rack=test, measure the toe difference
    solver = KinematicSolver3D(corner, tie_rod)
    s0 = solver.solve(0.0, 0.0, 0.0)
    s1 = solver.solve(0.0, 0.0, rack_test_mm)

    # delta toe in degrees
    d_toe = s1.toe_deg - s0.toe_deg

    if abs(d_toe) < 1e-9:
        return {
            "rack_per_wheel_deg_mm_per_deg": float("inf"),
            "wheel_deg_per_rack_mm":         0.0,
        }

    # How many mm of rack for 1° of wheel
    rack_per_deg = rack_test_mm / abs(d_toe)
    # Inverse: how many wheel degrees per mm of rack
    deg_per_rack_mm = 1.0 / rack_per_deg

    return {
        "rack_per_wheel_deg_mm_per_deg": rack_per_deg,
        "wheel_deg_per_rack_mm":         deg_per_rack_mm,
    }


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


# =============================================================================
# Roll Center under lateral load (1g)
# =============================================================================

def roll_center_at_1g_lat(
    left_corner:  SuspensionCorner, left_tie_rod:  TieRod,
    right_corner: SuspensionCorner, right_tie_rod: TieRod,
    *,
    cg_height_mm:           float = 280.0,
    track_width_mm_override: Optional[float] = None,
    roll_stiffness_deg_per_g: float = 1.5,
) -> dict[str, float]:
    """
    Estimate the Roll Center position (Y, Z) with the car under 1g of lateral
    acceleration, simulated as the equivalent roll.

    SIMPLIFICATION:
        Under 1g lateral, the chassis rolls an angle proportional to the roll
        stiffness. We apply that roll and compute where the RC ends up.

    Parameters:
        cg_height_mm           : CG height (needed only for the formal
                                  calculation — does not affect the RC directly here)
        roll_stiffness_deg_per_g: how many degrees the chassis rolls per g
                                  (typical FSAE value: 1.0–2.0 °/g)

    Returns a dict with:
        rc_y_mm, rc_z_mm : RC position under 1g
    """
    # Roll applied at 1g
    roll_1g = float(roll_stiffness_deg_per_g)

    # Run the solver with that roll on both corners; the RC is computed by the
    # 2D method on the YZ projection of the current points.
    solver_l = KinematicSolver3D(left_corner,  left_tie_rod)
    solver_r = KinematicSolver3D(right_corner, right_tie_rod)
    state_l = solver_l.solve(0.0, roll_1g, 0.0)
    state_r = solver_r.solve(0.0, roll_1g, 0.0)

    # LEFT RC (computed by the solver via 2D projection)
    from analysis.sweeps import SweepRunner
    runner_l = SweepRunner(solver=solver_l)
    rc_y_l, rc_z_l = runner_l._estimate_roll_center_yz(state_l)
    runner_r = SweepRunner(solver=solver_r)
    rc_y_r, rc_z_r = runner_r._estimate_roll_center_yz(state_r)

    # Average RC of the two sides (under roll they diverge; we take the average)
    return {
        "rc_y_mm": 0.5 * (rc_y_l + rc_y_r),
        "rc_z_mm": 0.5 * (rc_z_l + rc_z_r),
        "roll_applied_deg": roll_1g,
    }


# =============================================================================
# Anti-dive / Anti-squat (simplified version, side view)
# =============================================================================

def anti_dive_percent(
    corner: SuspensionCorner,
    *,
    brake_bias_pct: float = 60.0,
    cg_height_mm:   float = 280.0,
    wheelbase_mm_value: Optional[float] = None,
) -> float:
    """
    Anti-dive (%) — SIMPLIFIED VERSION.

    RIGOROUS DEFINITION:
        Anti-dive = tan(θ_SVIC) × wheelbase / cg_height × brake_bias_pct
        where θ_SVIC is the angle between the braking-force axis and the
        horizontal, measured from the CP to the "Side View Instant Center"
        (SVIC) — the intersection of the extended UCA and LCA arms in the
        SIDE view (X-Z).

    APPROXIMATION USED HERE:
        We compute the SVIC as the intersection of the extended arms projected
        onto XZ (using the effective inboard and outboard points).
        Then we apply the formula above.

    NOTE: for accurate results, the formal anti-dive calculation depends on
    whether the brake is INBOARD (attached to the chassis) or OUTBOARD
    (attached to the upright). Here we assume OUTBOARD (the common FSAE case).
    For an inboard brake, the geometrically possible anti-dive value is 0.
    """
    # XZ projection of the effective points
    uca_in = corner.upper_arm.effective_inboard
    uca_out = corner.upper_arm.outboard
    lca_in = corner.lower_arm.effective_inboard
    lca_out = corner.lower_arm.outboard

    # Intersection of the arm lines in the XZ plane
    def line_intersect_xz(p1, p2, p3, p4):
        x1, z1 = p1.x, p1.z
        x2, z2 = p2.x, p2.z
        x3, z3 = p3.x, p3.z
        x4, z4 = p4.x, p4.z
        denom = (x1 - x2) * (z3 - z4) - (z1 - z2) * (x3 - x4)
        if abs(denom) < 1e-9:
            return None
        t = ((x1 - x3) * (z3 - z4) - (z1 - z3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), z1 + t * (z2 - z1))

    svic = line_intersect_xz(lca_in, lca_out, uca_in, uca_out)
    if svic is None:
        return 0.0   # parallel arms in XZ → 0% anti-dive

    # Angle of the SVIC seen from the contact patch
    cp = corner.contact_patch
    dx = svic[0] - cp.x
    dz = svic[1] - cp.z
    if abs(dx) < 1e-9:
        return 0.0
    theta = math.atan2(dz, abs(dx))  # positive angle if the SVIC is above

    # Wheelbase: must be supplied externally (or uses a default)
    wb = wheelbase_mm_value if wheelbase_mm_value else 1550.0

    # Guard against an invalid cg_height (avoids ZeroDivision and negative values)
    if cg_height_mm <= 1e-6:
        return 0.0

    anti_dive = math.tan(theta) * wb / cg_height_mm * (brake_bias_pct / 100.0) * 100.0
    return float(anti_dive)


def anti_squat_percent(
    corner: SuspensionCorner,
    *,
    cg_height_mm:   float = 280.0,
    wheelbase_mm_value: Optional[float] = None,
    drive_type:     str = "RWD",
) -> float:
    """
    Anti-squat (%) — analogous to anti-dive for the rear.

    Works only for the rear corner (engine). For FWD, returns 0.
    """
    if drive_type.upper() not in ("RWD", "AWD"):
        return 0.0

    # Same logic as anti-dive, but for the rear corner
    return anti_dive_percent(
        corner,
        brake_bias_pct=100.0,   # all traction at the rear
        cg_height_mm=cg_height_mm,
        wheelbase_mm_value=wheelbase_mm_value,
    )


# =============================================================================
# KPI Bundle — gathers everything into a single dict
# =============================================================================

@dataclass
class FullKPIReport:
    """
    Complete KPI report for a vehicle, in the setup-sheet format.

    The None fields are those that cannot be computed without external data
    (spring stiffness, motion ratio, etc.) — they remain user inputs.
    """
    # Dimensions
    wheelbase_mm: float
    track_front_mm: float
    track_rear_mm: float

    # Per AXLE (row) — front and rear separated
    front: dict[str, float]
    rear:  dict[str, float]


def build_full_report(
    vehicle: Vehicle,
    tie_rods: dict[str, TieRod],
    *,
    cg_height_mm: float = 280.0,
    brake_bias_pct: float = 60.0,
    drive_type: str = "RWD",
    roll_stiffness_deg_per_g: float = 1.5,
) -> FullKPIReport:
    """
    Build the complete KPI report for the vehicle.

    External parameters (which do not come from the hardpoints):
        cg_height_mm              : CG height
        brake_bias_pct            : % of braking at the front
        drive_type                : "RWD", "FWD" or "AWD"
        roll_stiffness_deg_per_g  : roll per lateral g (tunable, depends on
                                     the springs and ARB; typical FSAE value)
    """
    fl, fr = vehicle.front_left,  vehicle.front_right
    rl, rr = vehicle.rear_left,   vehicle.rear_right
    tr_fl, tr_fr = tie_rods["FL"], tie_rods["FR"]
    tr_rl, tr_rr = tie_rods["RL"], tie_rods["RR"]

    wb     = wheelbase_mm(fl, rl)
    tr_f   = track_width_mm(fl, fr)
    tr_r   = track_width_mm(rl, rr)

    # --- Front ---
    ack_geom = ackermann_geometry(fl, tr_fl, fr, tr_fr, rl)
    steer_info = steer_ratio_and_cfactor(fl, tr_fl)
    rc_1g_front = roll_center_at_1g_lat(
        fl, tr_fl, fr, tr_fr,
        cg_height_mm=cg_height_mm,
        roll_stiffness_deg_per_g=roll_stiffness_deg_per_g,
    )

    front = {
        # Static
        "static_camber_left":   fl.static_camber_deg(),
        "static_camber_right":  fr.static_camber_deg(),
        "static_sum_toe":       static_sum_toe_deg(fl, tr_fl, fr, tr_fr),
        "caster_left":          fl.static_caster_deg(),
        "caster_right":         fr.static_caster_deg(),
        "kpi_left":             fl.static_kpi_deg(),
        "kpi_right":            fr.static_kpi_deg(),
        "scrub_left":           fl.static_scrub_radius_mm(),
        "scrub_right":          fr.static_scrub_radius_mm(),
        "trail_left":           fl.static_mechanical_trail_mm(),
        "trail_right":          fr.static_mechanical_trail_mm(),
        # RC
        "rc_height_static":     0.5*(fl.roll_center_height_mm() + fr.roll_center_height_mm()),
        "rc_y_at_1g":           rc_1g_front["rc_y_mm"],
        "rc_z_at_1g":           rc_1g_front["rc_z_mm"],
        # Dynamic
        "ride_camber_deg_per_m": ride_camber_deg_per_m(fl, tr_fl),
        "roll_camber":           roll_camber_deg_per_deg(fl, tr_fl),
        "anti_dive_pct":         anti_dive_percent(
                                    fl, brake_bias_pct=brake_bias_pct,
                                    cg_height_mm=cg_height_mm, wheelbase_mm_value=wb),
        # Steering
        "ackermann_pct":        ack_geom["ackermann_percent"],
        "steer_arm_length_l":   ack_geom["steer_arm_length_left"],
        "steer_arm_length_r":   ack_geom["steer_arm_length_right"],
        "rack_per_deg":         steer_info["rack_per_wheel_deg_mm_per_deg"],
        "wheel_deg_per_rack":   steer_info["wheel_deg_per_rack_mm"],
    }

    # --- Rear ---
    rc_1g_rear = roll_center_at_1g_lat(
        rl, tr_rl, rr, tr_rr,
        cg_height_mm=cg_height_mm,
        roll_stiffness_deg_per_g=roll_stiffness_deg_per_g,
    )

    rear = {
        "static_camber_left":   rl.static_camber_deg(),
        "static_camber_right":  rr.static_camber_deg(),
        "static_sum_toe":       static_sum_toe_deg(rl, tr_rl, rr, tr_rr),
        "kpi_left":             rl.static_kpi_deg(),
        "kpi_right":            rr.static_kpi_deg(),
        "scrub_left":           rl.static_scrub_radius_mm(),
        "scrub_right":          rr.static_scrub_radius_mm(),
        "rc_height_static":     0.5*(rl.roll_center_height_mm() + rr.roll_center_height_mm()),
        "rc_y_at_1g":           rc_1g_rear["rc_y_mm"],
        "rc_z_at_1g":           rc_1g_rear["rc_z_mm"],
        "ride_camber_deg_per_m": ride_camber_deg_per_m(rl, tr_rl),
        "roll_camber":           roll_camber_deg_per_deg(rl, tr_rl),
        "anti_squat_pct":        anti_squat_percent(
                                    rl, cg_height_mm=cg_height_mm,
                                    wheelbase_mm_value=wb,
                                    drive_type=drive_type),
    }

    return FullKPIReport(
        wheelbase_mm=wb,
        track_front_mm=tr_f,
        track_rear_mm=tr_r,
        front=front,
        rear=rear,
    )
