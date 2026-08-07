"""
geometry/solver_2d.py
=====================
2D kinematic solver — suspension analysis in the FRONT VIEW (Y-Z plane).

PHYSICAL CONCEPT
----------------
Seen from the front, the suspension is a FOUR-BAR MECHANISM:

       UCA_in ●──────────● UCA_out          (upper arm)
                          │
                          │ upright
                          │
       LCA_in ●──────────● LCA_out          (lower arm)

    Fixed link : chassis (segment UCA_in → LCA_in)
    Link 1     : upper arm (UCA_in → UCA_out)
    Link 2     : lower arm (LCA_in → LCA_out)
    Coupler    : upright (UCA_out → LCA_out)

During HEAVE motion (relative vertical displacement between chassis and wheel),
the inboard points rise/fall rigidly. The outboards (on the upright) must
simultaneously satisfy:
    |UCA_out − UCA_in| = L_UCA           (rigid UCA)
    |LCA_out − LCA_in| = L_LCA           (rigid LCA)
    |UCA_out − LCA_out| = L_upright      (rigid upright)

We solve this by TWO-CIRCLE INTERSECTION (once for UCA_out,
once for LCA_out), iterating until the three constraints are satisfied.

MAIN OUTPUTS
------------
    - Upright position in a given configuration
    - Camber
    - Roll Center
    - Camber Gain (°/mm of heave)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from geometry.primitives import (
    Point2D,
    circle_circle_intersection,
    line_intersection_2d,
)


# =============================================================================
# Kinematic state (result of one solve)
# =============================================================================

@dataclass
class KinematicState2D:
    """
    Result of solving the four-bar mechanism for a given heave.

    Attributes:
        heave_mm       : applied vertical displacement (mm); + = bump
        wheel_center   : wheel-center position in the Y-Z plane (mm)
        upright_upper  : position of the UCA outboard (mm)
        upright_lower  : position of the LCA outboard (mm)
        camber_deg     : camber angle (degrees); − = top inward
        roll_center    : Roll Center position (mm). None if undetermined.
    """
    heave_mm:      float
    wheel_center:  Point2D
    upright_upper: Point2D
    upright_lower: Point2D
    camber_deg:    float
    roll_center:   Optional[Point2D]

    @property
    def roll_center_height(self) -> Optional[float]:
        """Roll Center height (Z, in mm). None if undetermined."""
        return self.roll_center.v if self.roll_center else None


# =============================================================================
# 2D suspension geometry (one corner, in the front view)
# =============================================================================

@dataclass
class SuspensionGeometry2D:
    """
    Defines the 2D geometry (front view) of ONE suspension corner.

    Convention: for the LEFT SIDE use Y > 0; for the RIGHT, Y < 0.
    Since this class uses generic (u, v): u = Y, v = Z.

    Attributes:
        uca_inboard   : inboard anchor of the upper arm (on the chassis)
        uca_outboard  : outboard anchor of the upper arm (on the upright)
        lca_inboard   : inboard anchor of the lower arm (on the chassis)
        lca_outboard  : outboard anchor of the lower arm (on the upright)
        wheel_center  : wheel center (static position)
        contact_patch : tire-to-ground contact point
    """
    uca_inboard:   Point2D
    uca_outboard:  Point2D
    lca_inboard:   Point2D
    lca_outboard:  Point2D
    wheel_center:  Point2D
    contact_patch: Point2D

    def __post_init__(self) -> None:
        """
        Pre-computes invariant (rigid) lengths used by the solver.
        These values do not change during suspension motion.
        """
        # Arm lengths (measured in the static state and held fixed)
        self._L_uca:     float = self.uca_inboard.distance_to(self.uca_outboard)
        self._L_lca:     float = self.lca_inboard.distance_to(self.lca_outboard)
        self._L_upright: float = self.uca_outboard.distance_to(self.lca_outboard)

        # Offset (delta) of the wheel center relative to the LCA outboard, in
        # the GLOBAL STATIC frame. We use this to reconstruct the WC after the
        # upright moves — assuming a rigid upright.
        self._wc_offset_u: float = self.wheel_center.u - self.lca_outboard.u
        self._wc_offset_v: float = self.wheel_center.v - self.lca_outboard.v

    # -------------------------------------------------------------------------
    # Convenience properties
    # -------------------------------------------------------------------------

    @property
    def uca_length(self)     -> float: return self._L_uca
    @property
    def lca_length(self)     -> float: return self._L_lca
    @property
    def upright_length(self) -> float: return self._L_upright

    def static_camber_deg(self) -> float:
        """Static camber (with the suspension in the reference position)."""
        return self._compute_camber(self.uca_outboard, self.lca_outboard)

    # =========================================================================
    # MAIN METHOD: solve the suspension for a given heave
    # =========================================================================

    def solve_heave(self, heave_mm: float) -> KinematicState2D:
        """
        Solve the upright position for a heave displacement.

        HEAVE MODEL:
            When the wheel rises (bump) relative to the chassis, it is
            equivalent to moving the chassis UP while the wheel stays fixed.
            That is why we add +heave_mm to the Z coordinates of the inboard
            points.

        Parameters:
            heave_mm : vertical displacement (mm). + = bump, − = rebound.

        Returns:
            KinematicState2D with the solved positions and angles.
        """
        # ─── 1. Move the inboard (chassis) points ────────────────────────────
        # In pure heave, only Z changes (vertical rise/fall)
        uca_in_moved = Point2D(self.uca_inboard.u, self.uca_inboard.v + heave_mm)
        lca_in_moved = Point2D(self.lca_inboard.u, self.lca_inboard.v + heave_mm)

        # ─── 2. Solve the four-bar mechanism ─────────────────────────────────
        # Find the new outboard positions (on the upright)
        lca_out, uca_out = self._solve_four_bar(lca_in_moved, uca_in_moved)

        # ─── 3. Reconstruct the wheel center from the upright ────────────────
        # The WC is rigidly attached to the upright; when it rotates, the WC follows
        wc_new = self._reconstruct_wheel_center(lca_out, uca_out)

        # ─── 4. Compute derived angles ───────────────────────────────────────
        camber = self._compute_camber(uca_out, lca_out)
        rc     = self._compute_roll_center(lca_in_moved, lca_out,
                                            uca_in_moved, uca_out,
                                            wc_new)

        return KinematicState2D(
            heave_mm=heave_mm,
            wheel_center=wc_new,
            upright_upper=uca_out,
            upright_lower=lca_out,
            camber_deg=camber,
            roll_center=rc,
        )

    # =========================================================================
    # STEP 2: Four-bar mechanism solver (iterative)
    # =========================================================================

    def _solve_four_bar(
        self,
        lca_in_moved: Point2D,
        uca_in_moved: Point2D,
    ) -> tuple[Point2D, Point2D]:
        """
        Find (LCA_out, UCA_out) that satisfy the three rigid lengths.

        ALGORITHM (fixed-point iteration):
            1. Start with lca_out and uca_out at the static position (seed)
            2. Update lca_out = intersection of
                   circle(lca_in, r=L_lca) ∩ circle(uca_out, r=L_upright)
            3. Update uca_out = intersection of
                   circle(uca_in, r=L_uca) ∩ circle(lca_out, r=L_upright)
            4. Repeat until convergence (delta < tolerance)

        IMPORTANT — intersection solution choice:
            Each circle intersection has 2 solutions. To guarantee
            PHYSICAL CONTINUITY, we always choose the one closest to the
            previous position (tracking).
        """
        L_lca     = self._L_lca
        L_uca     = self._L_uca
        L_upright = self._L_upright

        # SEED: known static position (this IS the solution for heave=0,
        # and for small heave it is a good initial estimate)
        uca_out = self.uca_outboard
        lca_out = self.lca_outboard

        # Fixed-point iteration
        for _ in range(30):
            # Update LCA_out (keep distances to lca_in_moved and uca_out)
            lca_out_new = self._closest_intersection(
                c1=lca_in_moved, r1=L_lca,
                c2=uca_out,      r2=L_upright,
                reference=lca_out,
            )

            # Update UCA_out (keep distances to uca_in_moved and lca_out_new)
            uca_out_new = self._closest_intersection(
                c1=uca_in_moved, r1=L_uca,
                c2=lca_out_new,  r2=L_upright,
                reference=uca_out,
            )

            # Convergence criterion
            d_lca = lca_out_new.distance_to(lca_out)
            d_uca = uca_out_new.distance_to(uca_out)

            lca_out, uca_out = lca_out_new, uca_out_new

            if d_lca < 1e-8 and d_uca < 1e-8:
                break

        return lca_out, uca_out

    @staticmethod
    def _closest_intersection(
        c1: Point2D, r1: float,
        c2: Point2D, r2: float,
        reference: Point2D,
    ) -> Point2D:
        """
        Intersection of two circles, returning the solution closest to
        `reference`. Used for mechanism-continuity tracking.
        """
        p1 = c1.to_array()
        p2 = c2.to_array()
        ref = reference.to_array()

        d = float(np.linalg.norm(p2 - p1))
        if d < 1e-12:
            raise ValueError("Coincident centers.")
        if d > r1 + r2 + 1e-6:
            raise ValueError(
                f"Circles do not intersect: d={d:.2f}, r1+r2={r1+r2:.2f}"
            )
        if d < abs(r1 - r2) - 1e-6:
            raise ValueError("One circle contains the other.")

        # Standard circle-intersection geometry
        a   = (r1**2 - r2**2 + d**2) / (2.0 * d)
        h   = math.sqrt(max(r1**2 - a**2, 0.0))
        mid = p1 + a * (p2 - p1) / d
        perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]]) / d

        sol_a = mid + h * perp
        sol_b = mid - h * perp

        # Choose the solution closest to the reference (continuity)
        if np.linalg.norm(sol_a - ref) <= np.linalg.norm(sol_b - ref):
            return Point2D.from_array(sol_a)
        else:
            return Point2D.from_array(sol_b)

    # =========================================================================
    # STEP 3: Wheel-center reconstruction
    # =========================================================================

    def _reconstruct_wheel_center(
        self,
        lca_out: Point2D,
        uca_out: Point2D,
    ) -> Point2D:
        """
        Reconstruct the wheel center at the new upright position.

        The upright is a rigid body: the vector (LCA_out → WC) has constant
        length and orientation RELATIVE TO THE UPRIGHT. But the upright has
        rotated, so we need to rotate that vector by the same angle the
        upright rotated.

        ALGORITHM:
            1. Compute the static offset of the WC relative to LCA_out, in
               the upright's LOCAL frame (axial/perpendicular axes).
            2. Reconstruct the WC by applying that offset in the current frame
               (which rotates together with the upright).
        """
        # --- Upright's CURRENT LOCAL frame (axial and perpendicular axes) ---
        dx_now = uca_out.u - lca_out.u
        dz_now = uca_out.v - lca_out.v
        L_now  = math.hypot(dx_now, dz_now)
        if L_now < 1e-12:
            raise ValueError("Zero-length upright.")
        e_axial_now = np.array([dx_now / L_now, dz_now / L_now])  # along the upright
        e_perp_now  = np.array([-e_axial_now[1], e_axial_now[0]])  # 90° counter-clockwise

        # --- Upright's STATIC LOCAL frame (to recover the offset) ---
        dx_0 = self.uca_outboard.u - self.lca_outboard.u
        dz_0 = self.uca_outboard.v - self.lca_outboard.v
        L_0  = self._L_upright
        e_axial_0 = np.array([dx_0 / L_0, dz_0 / L_0])
        e_perp_0  = np.array([-e_axial_0[1], e_axial_0[0]])

        # Decompose the static offset onto the local axes
        offset = np.array([self._wc_offset_u, self._wc_offset_v])
        s_axial = float(np.dot(offset, e_axial_0))  # axial component
        s_perp  = float(np.dot(offset, e_perp_0))   # perpendicular component

        # Recompose using the CURRENT local axes
        wc_new_arr = (
            lca_out.to_array()
            + s_axial * e_axial_now
            + s_perp  * e_perp_now
        )
        return Point2D.from_array(wc_new_arr)

    # =========================================================================
    # STEP 4a: Camber computation
    # =========================================================================

    @staticmethod
    def _compute_camber(uca_out: Point2D, lca_out: Point2D) -> float:
        """
        Camber: angle of the UPRIGHT relative to the Z axis (vertical), in the Y-Z plane.

        SAE CONVENTION:
            - NEGATIVE camber: top of the wheel tilted INWARD
            - POSITIVE camber: top of the wheel tilted OUTWARD

        For the LEFT SIDE (Y > 0):
            If UCA_out.u < LCA_out.u → top inward → NEGATIVE camber
            (because "inward" = smaller Y for the left side)
        """
        dy = uca_out.u - lca_out.u   # lateral component
        dz = uca_out.v - lca_out.v   # vertical component

        # atan2(dy, dz) is the angle between the upright axis and the Z axis
        # With sign inverted to follow the SAE convention (− = inward)
        return -math.degrees(math.atan2(dy, dz))

    # =========================================================================
    # STEP 4b: Roll Center computation
    # =========================================================================

    @staticmethod
    def _compute_roll_center(
        lca_in:  Point2D,
        lca_out: Point2D,
        uca_in:  Point2D,
        uca_out: Point2D,
        wheel_center: Point2D,
    ) -> Optional[Point2D]:
        """
        Compute the Roll Center via the INSTANT CENTER (IC) method.

        REFERENCE: Milliken & Milliken, "Race Car Vehicle Dynamics", Ch. 17.

        ALGORITHM:
            1. IC = intersection of the extended arm lines
               (line LCA_in→LCA_out × line UCA_in→UCA_out)
            2. Connect IC to the tire contact patch (CP)
            3. RC = intersection of that line with the symmetry plane (u = 0)

        Degenerate cases:
            - Parallel arms: IC goes to infinity; RC stays at ground (Z=0)
            - Vertical IC→CP line: RC is the CP itself reflected onto the axis
        """
        # CP: contact point on the symmetry plane (u=Y of the WC, v=0)
        contact_patch = Point2D(wheel_center.u, 0.0)

        # STEP 1: Instant Center of the extended arm lines
        try:
            ic = line_intersection_2d(lca_in, lca_out, uca_in, uca_out)
        except ValueError:
            # Parallel arms: IC at infinity → RC at ground level
            return Point2D(0.0, 0.0)

        # STEP 2-3: line IC→CP, intersected with u=0 (symmetry plane)
        # Parametrization: P(t) = IC + t · (CP - IC)
        # We want t such that P.u = 0 → t = -IC.u / (CP.u - IC.u)
        delta_u = contact_patch.u - ic.u
        if abs(delta_u) < 1e-12:
            # Vertical line → RC at the IC height, projected onto u=0
            return Point2D(0.0, ic.v)

        t = -ic.u / delta_u
        v_rc = ic.v + t * (contact_patch.v - ic.v)
        return Point2D(0.0, v_rc)


# =============================================================================
# Utility function: camber-gain analysis via heave sweep
# =============================================================================

@dataclass
class CamberAnalysis:
    """
    Result of a parametric heave sweep.

    Attributes:
        heave_range_mm        : list of heave values tested
        camber_deg            : camber at each point
        roll_center_height_mm : Roll Center height at each point
    """
    heave_range_mm:        list[float]
    camber_deg:            list[float]
    roll_center_height_mm: list[Optional[float]]

    def camber_gain_deg_per_mm(self) -> float:
        """
        Rate of camber change with heave (°/mm), via linear regression.
        """
        if len(self.heave_range_mm) < 2:
            return 0.0
        # degree-1 polyfit: slope = camber gain
        coef = np.polyfit(self.heave_range_mm, self.camber_deg, 1)
        return float(coef[0])


def analyze_heave(
    geometry: SuspensionGeometry2D,
    heave_range_mm: float = 50.0,
    steps: int = 21,
) -> CamberAnalysis:
    """
    Run a symmetric heave sweep (from −range/2 to +range/2).

    Parameters:
        geometry       : 2D suspension geometry
        heave_range_mm : total sweep amplitude (mm)
        steps          : number of sampled points (prefer odd to include 0)

    Returns:
        CamberAnalysis with the result arrays.
    """
    half = heave_range_mm / 2.0
    heave_values = list(np.linspace(-half, half, steps))

    cambers: list[float] = []
    rc_heights: list[Optional[float]] = []

    for h in heave_values:
        state = geometry.solve_heave(h)
        cambers.append(state.camber_deg)
        rc_heights.append(state.roll_center_height)

    return CamberAnalysis(
        heave_range_mm=heave_values,
        camber_deg=cambers,
        roll_center_height_mm=rc_heights,
    )
