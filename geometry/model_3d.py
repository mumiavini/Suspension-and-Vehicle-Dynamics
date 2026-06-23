# Last commit without claude interfering
from __future__ import annotations
"""
geometry/model_3d.py
====================
Object-oriented model of the complete 3D suspension.

CLASS HIERARCHY:

    Point3D, Vector3D       (geometry/primitives.py)
            ↓
    ControlArm              control arm (A-arm with 2 inboards + 1 outboard)
            ↓
    KingpinGeometry         steering axis + computation methods
            ↓
    SuspensionCorner        one corner (UCA + LCA + WC + CP)
            ↓
    Vehicle                 4 corners (FL, FR, RL, RR) + general data

PURPOSE OF THIS MODULE:
    Compute STATIC 3D parameters from the hardpoint coordinates:
        - Caster
        - KPI (Kingpin Inclination)
        - Static camber
        - Scrub Radius
        - Mechanical Trail
        - Roll Center / Roll Axis

NOTE: this module works only with the STATIC POSITION. For motions
(bump, roll, steer), use `geometry/solver_3d.py`.
"""


import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from geometry.primitives import Point3D, Vector3D


# =============================================================================
# ControlArm — Control arm (UCA or LCA)
# =============================================================================

@dataclass
class ControlArm:
    """
    "A" or "L" shaped control arm.

    Has TWO anchor points on the chassis (inboard_front, inboard_rear)
    and ONE point on the upright (outboard).

    These two inboard points define the ARM's ROTATION AXIS — the arm
    pivots about that line when the suspension moves.

    Attributes:
        inboard_front : front inboard anchor (chassis)
        inboard_rear  : rear inboard anchor (chassis)
        outboard      : anchor on the upright
        name          : identifier (e.g. "UCA_FL")
    """
    inboard_front: Point3D
    inboard_rear:  Point3D
    outboard:      Point3D
    name:          str = "ControlArm"

    # -------------------------------------------------------------------------
    # Derived properties
    # -------------------------------------------------------------------------

    @property
    def effective_inboard(self) -> Point3D:
        """
        EFFECTIVE inboard point: midpoint between inboard_front and inboard_rear.

        For 2D calculations in the front view, we project the A-arm onto a
        single equivalent link from the midpoint to the outboard.
        """
        return self.inboard_front.midpoint(self.inboard_rear)

    def arm_vector(self) -> Vector3D:
        """Vector from the effective inboard to the outboard."""
        return Vector3D.from_points(self.effective_inboard, self.outboard)

    def arm_length(self) -> float:
        """Effective arm length (mm)."""
        return self.arm_vector().magnitude()

    def pivot_axis(self) -> Vector3D:
        """
        Axis about which the arm pivots (line inboard_front → inboard_rear).
        Not normalized.
        """
        return Vector3D.from_points(self.inboard_front, self.inboard_rear)

    def __repr__(self) -> str:
        return (
            f"ControlArm('{self.name}', "
            f"length={self.arm_length():.1f} mm)"
        )


# =============================================================================
# KingpinGeometry — Steering axis and derived angles
# =============================================================================

@dataclass
class KingpinGeometry:
    """
    Steering axis (kingpin axis) and associated metrics.

    The KINGPIN is the imaginary line joining the two ball joints (LBJ and UBJ).
    The wheel steers about this line. Its inclinations relative to the vertical
    axes produce four fundamental parameters:

        - Caster    : inclination in the side X-Z plane (affects self-centering)
        - KPI       : inclination in the front Y-Z plane (affects steer effects)
        - Scrub     : lateral distance between axis and contact (affects effort)
        - Trail     : longitudinal distance between axis and contact (affects return)

    Attributes:
        upper_ball_joint : center of the upper ball joint (UCA outboard)
        lower_ball_joint : center of the lower ball joint (LCA outboard)
        wheel_center     : wheel center (static)
        contact_patch    : tire-ground contact
    """
    upper_ball_joint: Point3D
    lower_ball_joint: Point3D
    wheel_center:     Point3D
    contact_patch:    Point3D

    # -------------------------------------------------------------------------
    # Kingpin axis (unit vector)
    # -------------------------------------------------------------------------

    def kingpin_axis(self) -> Vector3D:
        """
        UNIT vector along the kingpin, pointing from bottom to top
        (LBJ → UBJ).
        """
        return Vector3D.from_points(
            self.lower_ball_joint, self.upper_ball_joint
        ).normalize()

    # -------------------------------------------------------------------------
    # KPI (Kingpin Inclination)
    # -------------------------------------------------------------------------

    def kingpin_inclination_deg(self) -> float:
        """
        Kingpin inclination in the front Y-Z plane, in degrees.

        CONVENTION:
            POSITIVE KPI when the top of the kingpin is more INWARD
            (closer to the symmetry plane) than the base.

        TYPICAL FSAE: 5° to 10°
        """
        kp = self.kingpin_axis()

        # Projection onto the Y-Z plane (drops the X component)
        yz = np.array([0.0, kp.y, kp.z])
        norm = float(np.linalg.norm(yz))
        if norm < 1e-12:
            return 0.0
        yz_unit = yz / norm

        # Angle with the vertical Z axis (range 0°..180°)
        cos_theta = float(np.clip(yz_unit[2], -1.0, 1.0))
        angle = math.degrees(math.acos(cos_theta))

        # Sign: positive if UBJ is closer to the symmetry plane than LBJ
        # "Closer to the symmetry plane" = smaller |Y|
        ubj_inner = abs(self.upper_ball_joint.y) < abs(self.lower_ball_joint.y)
        return angle if ubj_inner else -angle

    # -------------------------------------------------------------------------
    # Caster
    # -------------------------------------------------------------------------

    def caster_deg(self) -> float:
        """
        Kingpin inclination in the side X-Z plane, in degrees.

        CONVENTION:
            POSITIVE caster when the top of the kingpin is offset REARWARD
            relative to the base (configuration that produces steering self-centering).

        TYPICAL FSAE: 3° to 7°
        """
        kp = self.kingpin_axis()

        # Projection onto the X-Z plane (drops the Y component)
        xz = np.array([kp.x, 0.0, kp.z])
        norm = float(np.linalg.norm(xz))
        if norm < 1e-12:
            return 0.0
        xz_unit = xz / norm

        cos_theta = float(np.clip(xz_unit[2], -1.0, 1.0))
        angle = math.degrees(math.acos(cos_theta))

        # Sign: positive if UBJ is BEHIND LBJ (UBJ.x < LBJ.x, i.e. kp.x < 0)
        return angle if kp.x < 0 else -angle

    # -------------------------------------------------------------------------
    # Scrub Radius
    # -------------------------------------------------------------------------

    def scrub_radius_mm(self) -> float:
        """
        LATERAL (Y) distance between the point where the kingpin intercepts the
        ground and the tire contact patch center.

        CONVENTION:
            POSITIVE: kingpin crosses the ground INWARD of the contact (typical)
            NEGATIVE: kingpin crosses the ground OUTWARD of the contact

        TYPICAL FSAE: −10 to +30 mm
        """
        intercept = self._kingpin_ground_intercept()
        if intercept is None:
            return float("inf")  # horizontal kingpin: does not intercept the ground
        return float(self.contact_patch.y - intercept[1])

    # -------------------------------------------------------------------------
    # Mechanical Trail
    # -------------------------------------------------------------------------

    def mechanical_trail_mm(self) -> float:
        """
        LONGITUDINAL (X) distance between the point where the kingpin intercepts
        the ground and the tire contact patch center.

        CONVENTION:
            POSITIVE: intercept ahead of the contact (conventional trail)

        TYPICAL FSAE: 5 to 25 mm (highly dependent on caster)
        """
        intercept = self._kingpin_ground_intercept()
        if intercept is None:
            return float("inf")
        return float(self.contact_patch.x - intercept[0])

    # -------------------------------------------------------------------------
    # Private helper: where does the kingpin cross the ground (Z=0)?
    # -------------------------------------------------------------------------

    def _kingpin_ground_intercept(self) -> Optional[NDArray[np.float64]]:
        """
        Find the point where the kingpin line crosses the Z=0 plane.

        Parametrization: P(t) = LBJ + t · kp_unit
        We want t such that P.z = 0:
            t = -LBJ.z / kp_unit.z

        Returns None if the kingpin is horizontal (kp.z ≈ 0).
        """
        kp = self.kingpin_axis().to_array()
        lbj = self.lower_ball_joint.to_array()

        if abs(kp[2]) < 1e-12:
            return None

        t = -lbj[2] / kp[2]
        return lbj + t * kp

    # -------------------------------------------------------------------------
    # Kingpin Offset @ Wheel Center (perpendicular distance from WC to the axis)
    # -------------------------------------------------------------------------

    def kingpin_offset_at_wheel_center_mm(self) -> float:
        """
        Kingpin-axis offset at the WHEEL CENTER level.

        DEFINITION: LATERAL (Y) distance between the point where the kingpin
        passes at the wheel-center height and the wheel center itself.

        DIFFERENCE from Scrub Radius:
            - Scrub Radius is the distance at GROUND LEVEL
            - Kingpin Offset (this) is at the WHEEL CENTER level
            These two values relate through the KPI:
                offset_wc - scrub_radius = WC.z · tan(KPI)

        TYPICAL FSAE: 30-80 mm (positive)
        """
        kp_unit = self.kingpin_axis().to_array()
        lbj = self.lower_ball_joint.to_array()

        if abs(kp_unit[2]) < 1e-12:
            return float("inf")

        # Parametrize the kingpin line and find the point at the WC height
        t_wc = (self.wheel_center.z - lbj[2]) / kp_unit[2]
        point_on_axis = lbj + t_wc * kp_unit

        # Lateral (Y) distance relative to the WC
        return float(self.wheel_center.y - point_on_axis[1])


# =============================================================================
# SuspensionCorner — One complete suspension corner
# =============================================================================

@dataclass
class SuspensionCorner:
    """
    One suspension corner (one wheel): UCA + LCA + upright + wheel.

    Required attributes:
        upper_arm     : upper arm (UCA)
        lower_arm     : lower arm (LCA)
        wheel_center  : wheel center (static)
        contact_patch : tire-ground contact (static)
        corner_id     : "FL" | "FR" | "RL" | "RR"

    Optional attributes (not used in this phase of the project):
        toe_link / pushrod / pullrod
    """
    upper_arm:     ControlArm
    lower_arm:     ControlArm
    wheel_center:  Point3D
    contact_patch: Point3D
    corner_id:     str = "FL"

    toe_link: Optional[ControlArm] = field(default=None)
    pushrod:  Optional[tuple[Point3D, Point3D]] = field(default=None)
    pullrod:  Optional[tuple[Point3D, Point3D]] = field(default=None)

    # -------------------------------------------------------------------------
    # Kingpin geometry (computed on demand)
    # -------------------------------------------------------------------------

    @property
    def kingpin(self) -> KingpinGeometry:
        """
        Kingpin axis of this corner.
        Built from the UCA outboard (UBJ) and LCA outboard (LBJ).
        """
        return KingpinGeometry(
            upper_ball_joint=self.upper_arm.outboard,
            lower_ball_joint=self.lower_arm.outboard,
            wheel_center=self.wheel_center,
            contact_patch=self.contact_patch,
        )

    # -------------------------------------------------------------------------
    # Static parameters (delegate to KingpinGeometry)
    # -------------------------------------------------------------------------

    def static_caster_deg(self)           -> float: return self.kingpin.caster_deg()
    def static_kpi_deg(self)              -> float: return self.kingpin.kingpin_inclination_deg()
    def static_scrub_radius_mm(self)      -> float: return self.kingpin.scrub_radius_mm()
    def static_mechanical_trail_mm(self)  -> float: return self.kingpin.mechanical_trail_mm()
    def static_kingpin_offset_mm(self)    -> float: return self.kingpin.kingpin_offset_at_wheel_center_mm()

    # -------------------------------------------------------------------------
    # Steer Arm Length — effective steering arm length
    # -------------------------------------------------------------------------

    def steer_arm_length_mm(self, tie_rod_outboard: Point3D) -> float:
        """
        Steering arm length (steer arm).

        DEFINITION: PERPENDICULAR distance from the tie-rod outboard point
        to the kingpin axis. This is the "lever arm" through which the
        tie-rod force generates steering torque on the wheel.

        FORMULA:
            steer_arm = |(TRO - LBJ) × kp_unit|

        TYPICAL FSAE: 50-100 mm
        """
        kp_unit = self.kingpin.kingpin_axis().to_array()
        v = tie_rod_outboard.to_array() - self.lower_arm.outboard.to_array()
        return float(np.linalg.norm(np.cross(v, kp_unit)))

    # -------------------------------------------------------------------------
    # Anti-dive / Anti-lift (side view X-Z)
    # -------------------------------------------------------------------------

    def anti_dive_percent(
        self,
        brake_bias: float = 0.6,
        wheelbase_mm: float = 1550.0,
        cg_height_mm: float = 280.0,
    ) -> float:
        """
        Anti-dive (%) — fraction of the braking force absorbed by the geometry.

        FORMULA (Milliken & Milliken, "Race Car Vehicle Dynamics" eq. 17.21):

            anti_dive_% = brake_bias × tan(θ) × (wheelbase / h_CG) × 100%

        where:
            θ      = angle between the horizontal and the line from CP to IC_side
            IC_side = intersection of the extended arm lines in the X-Z plane
            h_CG   = height of the vehicle's center of gravity (mm)

        Why wheelbase and h_CG are needed:
            The wheelbase/h_CG term comes from the torque balance in the
            longitudinal load transfer. Without these parameters, the isolated
            geometric formula cannot be interpreted as a "%".

        Default parameters are typical FSAE (wheelbase 1550, CG ~280mm).
        ADJUST for your vehicle to get a precise value.

        TYPICAL FSAE: 0% to 30% (anti-dive); negative = pro-dive
        """
        from geometry.primitives import line_intersection_2d

        uca_in_2d  = self.upper_arm.effective_inboard.project_xz()
        uca_out_2d = self.upper_arm.outboard.project_xz()
        lca_in_2d  = self.lower_arm.effective_inboard.project_xz()
        lca_out_2d = self.lower_arm.outboard.project_xz()

        try:
            ic_lat = line_intersection_2d(
                uca_in_2d, uca_out_2d, lca_in_2d, lca_out_2d,
            )
        except ValueError:
            return 0.0   # parallel arms: IC at infinity → 0%

        cp_2d = self.contact_patch.project_xz()
        dx = ic_lat.u - cp_2d.u
        dz = ic_lat.v - cp_2d.v

        if abs(dx) < 1e-6 or cg_height_mm < 1e-6:
            return 0.0

        # tan(θ) = dz / |dx|, with sign coming from dz
        tan_theta = dz / abs(dx)

        # Anti-dive % accounting for the longitudinal transfer
        anti_dive = brake_bias * tan_theta * (wheelbase_mm / cg_height_mm) * 100.0

        # Reasonable physical saturation
        return float(max(-200.0, min(200.0, anti_dive)))

    def anti_squat_percent(self, drive_fraction: float = 1.0) -> float:
        """
        Anti-squat (%) — equivalent to anti-dive but for acceleration
        (relevant only for the REAR suspension on a rear-wheel-drive car).

        For the front axle: anti_squat = 0 (receives no drive torque).
        For the rear axle (with locked diff): drive_fraction = 1.0

        ALGORITHM: identical to anti-dive, but counts the IC on the side opposite
        to braking. Here we simplify by returning the same geometry.
        """
        return self.anti_dive_percent(brake_bias=drive_fraction)

    # -------------------------------------------------------------------------
    # Static 3D camber — computed from the upright projection onto the Y-Z plane
    # -------------------------------------------------------------------------

    def static_camber_deg(self) -> float:
        """
        Constructive static camber in degrees.

        IMPORTANT: static camber CANNOT be inferred from the hardpoints alone.
        It depends on how the UPRIGHT was manufactured (relative position of
        the wheel-bearing holes with respect to the ball-joint holes).

        For most FSAE uprights, the constructive inclination is designed to
        give the desired static camber (e.g. -1.5°). That value is the
        OFFSET that needs to be added to the dynamic camber computed by the
        solver.

        This method returns the constructive offset stored in
        `self.static_camber_offset_deg`. If not set, returns 0.

        SAE CONVENTION:
            − = top of the wheel tilted INWARD
            + = top of the wheel tilted OUTWARD
        """
        return getattr(self, "static_camber_offset_deg", 0.0)

    # -------------------------------------------------------------------------
    # Static 3D Roll Center — uses the 2D solver on the Y-Z projection
    # -------------------------------------------------------------------------

    def roll_center_height_mm(self) -> float:
        """
        Roll Center height of this corner (mm), computed in the front view.

        Projects all points onto the Y-Z plane and uses the 2D solver to solve
        the static state (heave = 0). Returns the RC's Z coordinate.
        Returns NaN if undetermined.
        """
        # Local import to avoid an import cycle
        from geometry.solver_2d import SuspensionGeometry2D

        geom_2d = SuspensionGeometry2D(
            uca_inboard  = self.upper_arm.effective_inboard.project_yz(),
            uca_outboard = self.upper_arm.outboard.project_yz(),
            lca_inboard  = self.lower_arm.effective_inboard.project_yz(),
            lca_outboard = self.lower_arm.outboard.project_yz(),
            wheel_center = self.wheel_center.project_yz(),
            contact_patch= self.contact_patch.project_yz(),
        )
        state = geom_2d.solve_heave(0.0)
        h = state.roll_center_height
        return h if h is not None else float("nan")

    # -------------------------------------------------------------------------
    # Formatted summary
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """Return a formatted summary of the static parameters."""
        return "\n".join([
            f"═══ SuspensionCorner [{self.corner_id}] ═══",
            f"  Caster              : {self.static_caster_deg():+.3f}°",
            f"  KPI                 : {self.static_kpi_deg():+.3f}°",
            f"  Camber (static)     : {self.static_camber_deg():+.3f}°",
            f"  Scrub Radius        : {self.static_scrub_radius_mm():+.2f} mm",
            f"  Mechanical Trail    : {self.static_mechanical_trail_mm():+.2f} mm",
            f"  Roll Center Height  : {self.roll_center_height_mm():+.2f} mm",
        ])


# =============================================================================
# Vehicle — Complete car (4 corners + general dimensions)
# =============================================================================

@dataclass
class Vehicle:
    """
    Complete vehicle: four suspension corners + general parameters.

    Attributes:
        front_left   : FL corner
        front_right  : FR corner
        rear_left    : RL corner
        rear_right   : RR corner
        wheelbase_mm : wheelbase (mm)
        track_front_mm / track_rear_mm : track widths (mm)
    """
    front_left:  SuspensionCorner
    front_right: SuspensionCorner
    rear_left:   SuspensionCorner
    rear_right:  SuspensionCorner

    wheelbase_mm:   float = 1600.0
    track_front_mm: float = 1200.0
    track_rear_mm:  float = 1180.0

    # -------------------------------------------------------------------------
    # Roll Axis
    # -------------------------------------------------------------------------

    def roll_axis(self) -> tuple[float, float]:
        """
        Return (rc_front, rc_rear): average RC heights of the front and rear.

        The vehicle's ROLL AXIS is the line joining these two RCs.
        """
        rc_front = 0.5 * (self.front_left.roll_center_height_mm()
                        + self.front_right.roll_center_height_mm())
        rc_rear  = 0.5 * (self.rear_left.roll_center_height_mm()
                        + self.rear_right.roll_center_height_mm())
        return rc_front, rc_rear

    def roll_axis_inclination_deg(self) -> float:
        """
        Roll-axis inclination relative to the ground (degrees).
        Positive = rear end of the axis higher than the front.
        """
        rc_front, rc_rear = self.roll_axis()
        return math.degrees(math.atan2(rc_rear - rc_front, self.wheelbase_mm))

    # -------------------------------------------------------------------------
    # Static (geometric) Ackermann
    # -------------------------------------------------------------------------

    def static_ackermann_percent(
        self,
        tie_rod_fl_outboard: Point3D,
        tie_rod_fr_outboard: Point3D,
    ) -> float:
        """
        Static Ackermann (%) — how close the steering geometry COMES
        to perfect Ackermann (100%).

        ALGORITHM:
            Perfect Ackermann (100%) occurs when the lines starting from each
            kingpin axis, passing through the respective tie-rod outboard,
            converge to a single point on the rear axle.

            Practical approximation (Steer Arm method):
                tan(α_perfect) = (track / 2) / wheelbase
                α_real = atan(steer_arm_offset / wheel_center_to_pin_dist)

            Ratio (real / perfect) × 100% = Ackermann %

        SIMPLIFICATION USED:
            Measure the angle between the steer arm (vector from the kingpin
            projection to the TRO, in the XY plane) and the vehicle's Y axis.
            Compare it to the vehicle's perfect Ackermann angle.

        Returns a percentage (100% = perfect, 0% = parallel).
        """
        # Vehicle's perfect Ackermann angle (simple geometry)
        if self.wheelbase_mm < 1e-6 or self.track_front_mm < 1e-6:
            return 0.0
        perfect_angle_rad = math.atan2(self.track_front_mm / 2.0, self.wheelbase_mm)

        # Real steer-arm angle of the FL corner
        kp_fl = self.front_left.kingpin.kingpin_axis().to_array()
        wc_fl = self.front_left.wheel_center.to_array()
        tro_fl = tie_rod_fl_outboard.to_array()

        # Vector TRO → projection onto the XY plane (top view)
        steer_arm_vec_fl = tro_fl - wc_fl
        # Remove the component parallel to the kingpin axis
        steer_arm_perp_fl = steer_arm_vec_fl - np.dot(steer_arm_vec_fl, kp_fl) * kp_fl
        sa_xy_fl = np.array([steer_arm_perp_fl[0], steer_arm_perp_fl[1]])
        if float(np.linalg.norm(sa_xy_fl)) < 1e-9:
            return 0.0
        # The steer-arm angle with the Y axis (lateral)
        real_angle_rad = abs(math.atan2(sa_xy_fl[0], abs(sa_xy_fl[1]) + 1e-12))

        # Ackermann % = ratio of the real angle to the ideal one
        if perfect_angle_rad < 1e-9:
            return 0.0
        return float(100.0 * real_angle_rad / perfect_angle_rad)

    # -------------------------------------------------------------------------
    # Formatted summary
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        rc_f, rc_r = self.roll_axis()
        return "\n".join([
            "╔══════════════════════════════════════════╗",
            "║          VEHICLE SUSPENSION SUMMARY      ║",
            "╚══════════════════════════════════════════╝",
            "",
            self.front_left.summary(),
            "",
            self.front_right.summary(),
            "",
            self.rear_left.summary(),
            "",
            self.rear_right.summary(),
            "",
            "─── Roll Axis ───",
            f"  Front RC (average)    : {rc_f:+.2f} mm",
            f"  Rear RC  (average)    : {rc_r:+.2f} mm",
            f"  Roll Axis Inclination : {self.roll_axis_inclination_deg():+.4f}°",
        ])
