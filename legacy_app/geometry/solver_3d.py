"""
geometry/solver_3d.py
=====================
3D kinematic solver — motion of the upright as a RIGID BODY.

PHYSICAL CONCEPT
----------------
The upright has THREE anchor points (ball joints):
    - UBJ (Upper Ball Joint)  : UCA outboard
    - LBJ (Lower Ball Joint)  : LCA outboard
    - TRO (Tie-Rod Outboard)  : tie-rod outboard

Each one must keep a FIXED distance from its respective inboard point (3 spheres
in 3D space). Additionally, the three INTERNAL distances of the upright
(UBJ-LBJ, UBJ-TRO, LBJ-TRO) must also be preserved (rigid body).

NON-LINEAR SYSTEM TO SOLVE (9 unknowns, 6 equations):
    For each ball joint i:
        (X_i - x_i_in)² + (Y_i - y_i_in)² + (Z_i - z_i_in)² = L_i²

    Plus 3 rigid-body equations:
        |UBJ - LBJ| = const     (static dist.)
        |UBJ - TRO| = const
        |LBJ - TRO| = const

Since we have 9 DOF (3 points × 3 coords) and 6 constraints, 3 DOF remain.
We add SOFT REGULARIZATION (anchor near the previous position)
so the system is well-conditioned.

ALGORITHM: scipy.optimize.least_squares with Levenberg-Marquardt.

SOLVER INPUTS:
    - heave_mm : vertical displacement of the chassis
    - roll_deg : chassis roll angle (about the X axis)
    - rack_mm  : steering rack displacement (in Y)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from geometry.primitives import Point3D, Vector3D
from geometry.model_3d import SuspensionCorner, ControlArm


# =============================================================================
# Tie-Rod (steering link)
# =============================================================================

@dataclass
class TieRod:
    """
    Tie-rod: bar connecting the rack/pitman arm to the upright.

    Attributes:
        inboard  : fixed point on the rack (moves with the chassis + rack offset)
        outboard : point on the upright (rotates with it)
    """
    inboard:  Point3D
    outboard: Point3D
    name:     str = "TieRod"

    @property
    def length(self) -> float:
        """Tie-rod length (mm), invariant during motion."""
        return self.inboard.distance_to(self.outboard)

    def __repr__(self) -> str:
        return f"TieRod('{self.name}', length={self.length:.2f} mm)"


# =============================================================================
# 3D kinematic state (result of one solve)
# =============================================================================

@dataclass
class KinematicState3D:
    """
    Suspension state for a given configuration (heave, roll, rack).

    Inputs:
        heave_mm, roll_deg, rack_mm

    Solved positions:
        uca_outboard, lca_outboard, tie_rod_outboard, wheel_center, contact_patch

    Derived angles:
        camber_deg, toe_deg, caster_deg, kpi_deg

    Solver diagnostics:
        converged, residual_norm, iterations
    """
    heave_mm: float = 0.0
    roll_deg: float = 0.0
    rack_mm:  float = 0.0

    uca_outboard:     Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    lca_outboard:     Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    tie_rod_outboard: Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    wheel_center:     Point3D = field(default_factory=lambda: Point3D(0, 0, 0))
    contact_patch:    Point3D = field(default_factory=lambda: Point3D(0, 0, 0))

    camber_deg: float = 0.0
    toe_deg:    float = 0.0
    caster_deg: float = 0.0
    kpi_deg:    float = 0.0

    converged:     bool  = True
    residual_norm: float = 0.0
    iterations:    int   = 0


# =============================================================================
# 3D solver
# =============================================================================

class KinematicSolver3D:
    """
    Solve the 3D upright kinematics via least_squares.

    Typical use:
        solver = KinematicSolver3D(corner, tie_rod)
        state  = solver.solve(heave_mm=10.0, roll_deg=0.5, rack_mm=0.0)

    For sweeps, the solver keeps a cache of the last state as a SEED for the
    next one, ensuring physical continuity. Use `solver.reset_seed()` when
    starting a new sweep.
    """

    # Soft-regularization weight (used in _residuals and _residuals_jac)
    _REG_WEIGHT: float = 1e-4

    def __init__(
        self,
        corner: SuspensionCorner,
        tie_rod: TieRod,
        *,
        tolerance: float = 1e-9,
        max_iter:  int   = 100,
    ) -> None:
        """
        Initialize the solver by pre-computing all invariant distances.
        """
        self.corner: SuspensionCorner = corner
        self.tie_rod: TieRod           = tie_rod
        self.tolerance: float          = tolerance
        self.max_iter: int             = max_iter

        # ─── Link lengths (invariants) ───────────────────────────────────────
        self._L_uca: float = corner.upper_arm.arm_length()
        self._L_lca: float = corner.lower_arm.arm_length()
        self._L_tr:  float = tie_rod.length

        # ─── Internal upright distances (rigid body) ─────────────────────────
        ubj = corner.upper_arm.outboard.to_array()
        lbj = corner.lower_arm.outboard.to_array()
        tro = tie_rod.outboard.to_array()
        self._d_ubj_lbj: float = float(np.linalg.norm(ubj - lbj))
        self._d_ubj_tro: float = float(np.linalg.norm(ubj - tro))
        self._d_lbj_tro: float = float(np.linalg.norm(lbj - tro))

        # ─── Local offsets of WC and CP relative to the upright frame ────────
        # We need these to reconstruct the WC/CP positions after the upright
        # rotates. Computed ONCE here in init.
        self._wc_local_offset: NDArray[np.float64] = self._compute_local_offset(
            corner.wheel_center.to_array(), ubj, lbj, tro
        )
        self._cp_local_offset: NDArray[np.float64] = self._compute_local_offset(
            corner.contact_patch.to_array(), ubj, lbj, tro
        )

        # ─── REFERENCE toe (static state) ────────────────────────────────────
        # Pre-compute the toe at the static position so we always report the
        # DELTA relative to this zero. This removes the arbitrary offset of the
        # absolute toe.
        self._toe_static: float = self._compute_toe_absolute(
            corner.upper_arm.outboard, corner.lower_arm.outboard,
            tie_rod.outboard, corner.wheel_center,
        )

        # Cache of the last state (to use as the next seed)
        self._last_state: Optional[KinematicState3D] = None

    # =========================================================================
    # MAIN METHOD: solve
    # =========================================================================

    def solve(
        self,
        heave_mm: float = 0.0,
        roll_deg: float = 0.0,
        rack_mm:  float = 0.0,
    ) -> KinematicState3D:
        """
        Solve the 3D kinematics for a configuration (heave, roll, rack).

        Parameters:
            heave_mm : vertical chassis displacement (+ = chassis rises)
            roll_deg : chassis roll about the X axis
                       (+ = chassis rolls to the right; left side drops)
            rack_mm  : lateral rack displacement (+ = to the left)
        """
        # ─── 1. Move the inboard points (chassis + rack) ─────────────────────
        uca_in_eff = self.corner.upper_arm.effective_inboard
        lca_in_eff = self.corner.lower_arm.effective_inboard
        tr_in      = self.tie_rod.inboard

        uca_in_moved = self._move_chassis_point(uca_in_eff, heave_mm, roll_deg)
        lca_in_moved = self._move_chassis_point(lca_in_eff, heave_mm, roll_deg)
        tr_in_moved  = self._move_chassis_point(tr_in,      heave_mm, roll_deg)
        tr_in_moved[1] += rack_mm   # the rack moves laterally in Y

        # ─── 2. Build the initial seed (last state or static) ────────────────
        if self._last_state is not None:
            seed = np.concatenate([
                self._last_state.uca_outboard.to_array(),
                self._last_state.lca_outboard.to_array(),
                self._last_state.tie_rod_outboard.to_array(),
            ])
        else:
            seed = np.concatenate([
                self.corner.upper_arm.outboard.to_array(),
                self.corner.lower_arm.outboard.to_array(),
                self.tie_rod.outboard.to_array(),
            ])

        # ─── 3. Solve the non-linear system ──────────────────────────────────
        # Analytic jac: avoids ~10 numerical residual evaluations per LM step
        # (finite differences over 9 variables) → solver ~3-5× faster.
        result = least_squares(
            fun=self._residuals,
            jac=self._residuals_jac,
            x0=seed,
            args=(uca_in_moved, lca_in_moved, tr_in_moved, seed),
            method="lm",                   # Levenberg-Marquardt
            xtol=self.tolerance,
            ftol=self.tolerance,
            max_nfev=self.max_iter * 10,
        )

        # ─── 4. Extract the solved positions ─────────────────────────────────
        x = result.x
        ubj = x[0:3]; lbj = x[3:6]; tro = x[6:9]
        uca_out = Point3D.from_array(ubj)
        lca_out = Point3D.from_array(lbj)
        tr_out  = Point3D.from_array(tro)

        # ─── 5. Reconstruct WC and CP from the upright's local frame ─────────
        wc_arr = self._reconstruct_from_local(self._wc_local_offset, ubj, lbj, tro)
        cp_arr = self._reconstruct_from_local(self._cp_local_offset, ubj, lbj, tro)
        wheel_center  = Point3D.from_array(wc_arr)
        contact_patch = Point3D.from_array(cp_arr)

        # ─── 6. Compute the derived angles ───────────────────────────────────
        camber = self._compute_camber(uca_out, lca_out, wheel_center, contact_patch)
        caster = self._compute_caster(uca_out, lca_out)
        kpi    = self._compute_kpi(uca_out, lca_out)

        # Toe RELATIVE to the static state (= bump steer + steer angle)
        toe_abs = self._compute_toe_absolute(uca_out, lca_out, tr_out, wheel_center)
        toe = toe_abs - self._toe_static

        state = KinematicState3D(
            heave_mm=heave_mm, roll_deg=roll_deg, rack_mm=rack_mm,
            uca_outboard=uca_out,
            lca_outboard=lca_out,
            tie_rod_outboard=tr_out,
            wheel_center=wheel_center,
            contact_patch=contact_patch,
            camber_deg=camber,
            toe_deg=toe,
            caster_deg=caster,
            kpi_deg=kpi,
            converged=result.success,
            residual_norm=float(np.linalg.norm(result.fun)),
            iterations=int(result.nfev),
        )

        # Cache for the next call (continuity within the sweep)
        self._last_state = state
        return state

    def reset_seed(self) -> None:
        """Clear the cache. Use when starting a new sweep."""
        self._last_state = None

    # =========================================================================
    # Residual function (least_squares minimizes ||residuals||²)
    # =========================================================================

    def _residuals(
        self,
        x:            NDArray[np.float64],
        uca_in_moved: NDArray[np.float64],
        lca_in_moved: NDArray[np.float64],
        tr_in_moved:  NDArray[np.float64],
        seed:         NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Residual vector. The solver minimizes the sum of squares of these.

        Composition:
            r[0..2] : 3 distance constraints for the inboards (spheres)
            r[3..5] : 3 internal upright distance constraints (rigid body)
            r[6..14]: 9 soft-regularization terms (anchor near the seed)
        """
        ubj = x[0:3]; lbj = x[3:6]; tro = x[6:9]

        # 1. Distances to the inboards (spheres)
        r_ubj = np.linalg.norm(ubj - uca_in_moved) - self._L_uca
        r_lbj = np.linalg.norm(lbj - lca_in_moved) - self._L_lca
        r_tro = np.linalg.norm(tro - tr_in_moved)  - self._L_tr

        # 2. Internal upright distances (rigid body)
        r_d1 = np.linalg.norm(ubj - lbj) - self._d_ubj_lbj
        r_d2 = np.linalg.norm(ubj - tro) - self._d_ubj_tro
        r_d3 = np.linalg.norm(lbj - tro) - self._d_lbj_tro

        # 3. Soft regularization (very small weight so it does not dominate)
        reg = (x - seed) * self._REG_WEIGHT

        return np.concatenate([
            np.array([r_ubj, r_lbj, r_tro, r_d1, r_d2, r_d3]),
            reg,
        ])

    def _residuals_jac(
        self,
        x:            NDArray[np.float64],
        uca_in_moved: NDArray[np.float64],
        lca_in_moved: NDArray[np.float64],
        tr_in_moved:  NDArray[np.float64],
        seed:         NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        ANALYTIC Jacobian of `_residuals` (15 residuals × 9 variables).

        For a distance residual r = ||a − b|| − L:
            ∂r/∂a = (a − b) / ||a − b||      (and ∂r/∂b = −∂r/∂a)

        The regularization rows are simply I₉ × _REG_WEIGHT.
        """
        ubj = x[0:3]; lbj = x[3:6]; tro = x[6:9]

        def unit(d: NDArray[np.float64]) -> NDArray[np.float64]:
            n = float(np.linalg.norm(d))
            return d / n if n > 1e-12 else np.zeros(3)

        J = np.zeros((15, 9))

        # 1. Distances to the inboards (depend only on their own point)
        J[0, 0:3] = unit(ubj - uca_in_moved)
        J[1, 3:6] = unit(lbj - lca_in_moved)
        J[2, 6:9] = unit(tro - tr_in_moved)

        # 2. Internal upright distances (point pair, opposite signs)
        u = unit(ubj - lbj); J[3, 0:3] = u; J[3, 3:6] = -u
        u = unit(ubj - tro); J[4, 0:3] = u; J[4, 6:9] = -u
        u = unit(lbj - tro); J[5, 3:6] = u; J[5, 6:9] = -u

        # 3. Regularization
        J[6:15, :] = np.eye(9) * self._REG_WEIGHT
        return J

    # =========================================================================
    # Chassis point motion (heave + roll)
    # =========================================================================

    @staticmethod
    def _move_chassis_point(
        point:    Point3D,
        heave_mm: float,
        roll_deg: float,
    ) -> NDArray[np.float64]:
        """
        Apply heave (Z translation) and roll (X rotation) to a chassis point.

        ORDER OF TRANSFORMATIONS:
            1. Roll: rotation about the X (longitudinal) axis, origin at Y=Z=0
            2. Heave: translation in Z

        For the real roll axis (which does not pass through the origin), we
        should strictly translate to the RC, rotate, and undo. Here we use the
        simple approximation (rotation at the origin), which is valid for the
        small roll angles typical of FSAE (< 3°).
        """
        p = point.to_array().copy()

        # Roll about the X axis (Y and Z are rotated)
        if abs(roll_deg) > 1e-12:
            theta = math.radians(roll_deg)
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            y_new = p[1] * cos_t - p[2] * sin_t
            z_new = p[1] * sin_t + p[2] * cos_t
            p[1] = y_new
            p[2] = z_new

        # Heave
        p[2] += heave_mm
        return p

    # =========================================================================
    # Upright local frame (to reconstruct WC and CP)
    # =========================================================================

    @staticmethod
    def _build_local_frame(
        ubj: NDArray[np.float64],
        lbj: NDArray[np.float64],
        tro: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Build an ORTHONORMAL basis fixed on the upright, anchored at LBJ.

        Procedure (Gram-Schmidt):
            e1 = (UBJ - LBJ) / |UBJ - LBJ|         kingpin direction
            e2 = (TRO - LBJ) - (projected onto e1)  orthogonal to e1
            e3 = e1 × e2                            completes the right-handed basis

        This basis ROTATES together with the upright, keeping the local
        coordinates of any point RIGIDLY attached to the upright constant.
        """
        v1 = ubj - lbj
        e1 = v1 / np.linalg.norm(v1)

        v2 = tro - lbj
        v2_perp = v2 - np.dot(v2, e1) * e1
        e2 = v2_perp / np.linalg.norm(v2_perp)

        e3 = np.cross(e1, e2)
        return e1, e2, e3

    @classmethod
    def _compute_local_offset(
        cls,
        point: NDArray[np.float64],
        ubj:   NDArray[np.float64],
        lbj:   NDArray[np.float64],
        tro:   NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the coordinates of `point` in the upright's local frame.
        These values are INVARIANT during motion (rigid upright).
        """
        e1, e2, e3 = cls._build_local_frame(ubj, lbj, tro)
        delta = point - lbj
        return np.array([
            float(np.dot(delta, e1)),
            float(np.dot(delta, e2)),
            float(np.dot(delta, e3)),
        ])

    @classmethod
    def _reconstruct_from_local(
        cls,
        local_offset: NDArray[np.float64],
        ubj: NDArray[np.float64],
        lbj: NDArray[np.float64],
        tro: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Reconstruct the GLOBAL position of a point from its local offset.
        Uses the upright's local frame in the CURRENT configuration.
        """
        e1, e2, e3 = cls._build_local_frame(ubj, lbj, tro)
        return lbj + local_offset[0] * e1 + local_offset[1] * e2 + local_offset[2] * e3

    # =========================================================================
    # Computation of the derived angles
    # =========================================================================

    @staticmethod
    def _compute_camber(uca_out: Point3D, lca_out: Point3D,
                         wheel_center: Point3D, contact_patch: Point3D) -> float:
        """
        Dynamic camber: inclination of the wheel plane relative to vertical,
        in the front view (Y-Z plane).

        DEFINITION:
            Uses the CP→WC vector projected onto Y-Z. For a vertical wheel
            (camber=0), this vector is (0, 0, +R). If the upright rotates about
            X, the vector gains a Y component.

        SAE CONVENTION:
            − = top of the wheel tilted INWARD
            + = top of the wheel tilted OUTWARD
        """
        wc = wheel_center.to_array()
        cp = contact_patch.to_array()

        dy = wc[1] - cp[1]
        dz = wc[2] - cp[2]

        if abs(dz) < 1e-9:
            return 0.0

        # Angle between (CP→WC) and the vertical Z axis
        angle = math.degrees(math.atan2(dy, dz))

        # Sign: for the left (WC.y > 0), negative camber = WC more inward
        # than CP = dy < 0 → angle < 0 → camber = +angle (stays negative)
        # Wait: dy < 0 gives angle < 0; we want camber = -|angle| (negative)
        # → camber = angle for the left side
        # For the right (WC.y < 0), negative camber = WC more inward = dy > 0
        # → camber = -angle for the right side
        if wc[1] > 0:   # left
            return angle
        else:           # right
            return -angle

    @staticmethod
    def _compute_caster(uca_out: Point3D, lca_out: Point3D) -> float:
        """
        Caster: kingpin inclination in the X-Z plane.
        Positive = top of the kingpin behind the base.
        """
        ubj = uca_out.to_array()
        lbj = lca_out.to_array()
        kp = ubj - lbj
        kp_xz = np.array([kp[0], 0.0, kp[2]])
        n = float(np.linalg.norm(kp_xz))
        if n < 1e-12:
            return 0.0
        cos_t = float(np.clip(kp_xz[2] / n, -1.0, 1.0))
        angle = math.degrees(math.acos(cos_t))
        return angle if kp[0] < 0 else -angle

    @staticmethod
    def _compute_kpi(uca_out: Point3D, lca_out: Point3D) -> float:
        """
        KPI: kingpin inclination in the Y-Z plane.
        Positive = top of the kingpin inward (closer to the symmetry plane).
        """
        ubj = uca_out.to_array()
        lbj = lca_out.to_array()
        kp = ubj - lbj
        kp_yz = np.array([0.0, kp[1], kp[2]])
        n = float(np.linalg.norm(kp_yz))
        if n < 1e-12:
            return 0.0
        cos_t = float(np.clip(kp_yz[2] / n, -1.0, 1.0))
        angle = math.degrees(math.acos(cos_t))
        return angle if abs(ubj[1]) < abs(lbj[1]) else -angle

    @staticmethod
    def _compute_toe_absolute(
        uca_out:      Point3D,
        lca_out:      Point3D,
        tr_out:       Point3D,
        wheel_center: Point3D,
    ) -> float:
        """
        ABSOLUTE toe in degrees.

        ALGORITHM:
            1. Kingpin axis (LBJ → UBJ), normalized
            2. Steering arm = (TRO - WC), projected PERPENDICULARLY to the kingpin
            3. Toe = atan2(X component, Y component) of this projected vector

        IMPORTANT: this value alone has no direct meaning — it depends on an
        arbitrary choice of tie-rod orientation. What matters is the VARIATION
        relative to the static state (delta toe), which is what the solver
        returns in `state.toe_deg`.
        """
        ubj = uca_out.to_array()
        lbj = lca_out.to_array()
        tro = tr_out.to_array()
        wc  = wheel_center.to_array()

        # Kingpin unit axis
        kp = ubj - lbj
        kp_norm = float(np.linalg.norm(kp))
        if kp_norm < 1e-12:
            return 0.0
        kp_unit = kp / kp_norm

        # Steering arm: vector from WC to TRO, projected perpendicular to the kingpin
        steer_arm = tro - wc
        steer_perp = steer_arm - np.dot(steer_arm, kp_unit) * kp_unit

        # Projection onto the XY plane (top view)
        sa_xy = np.array([steer_perp[0], steer_perp[1]])
        if float(np.linalg.norm(sa_xy)) < 1e-9:
            return 0.0

        # Toe = angle of the steering arm relative to the Y axis, in the XY plane
        return math.degrees(math.atan2(sa_xy[0], abs(sa_xy[1]) + 1e-12))
