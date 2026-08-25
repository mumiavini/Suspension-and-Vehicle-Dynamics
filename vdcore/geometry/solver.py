"""3D kinematic solver for a double-wishbone suspension corner.

Treats the upright as a rigid body defined by three ball joints
(UBJ = UCA outboard, LBJ = LCA outboard, TRO = tie-rod outboard).
Each ball joint is constrained to lie on a sphere centred at the
corresponding chassis-side hardpoint, with radius equal to the
link length. The three inter-joint distances on the upright are
additional rigid-body constraints.

For a given wheel state (bump/droop, roll, rack), the six chassis-side
pickup points move, and we solve for the nine unknowns (3 × 3 ball-joint
positions) subject to nine constraints (3 sphere + 3 link-length +
3 rigid-body inter-distance).

Coordinate system: ISO 8855 — X+ forward, Y+ LEFT, Z+ up.

Solver: scipy.optimize.least_squares with default trust-region method.
Numerical Jacobian. Non-convergence returns converged=False.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from vdcore.geometry.primitives import Point3D, Vector3D, Z_HAT
from vdcore.models.hardpoint import Corner, DerivedPoint

Arr = npt.NDArray[np.float64]


class SolverResult(BaseModel, frozen=True):
    """Result of a kinematic solve for one corner."""

    ubj: DerivedPoint
    lbj: DerivedPoint
    tro: DerivedPoint
    wheel_center: DerivedPoint
    contact_patch: DerivedPoint
    camber_deg: float
    toe_deg_per_side: float
    caster_deg: float
    kpi_deg: float
    converged: bool
    residual_norm: float
    nfev: int
    njev: int


class DWSolver:
    """3D kinematic solver for a double-wishbone suspension corner.

    ISO 8855: X+ forward, Y+ LEFT, Z+ up.
    """

    def __init__(self, corner: Corner) -> None:
        self._corner = corner
        self._is_left = corner.corner_id in ("FL", "RL")

        # Static positions of the outboard joints (upright vertices)
        self._ubj_0 = np.array([
            corner.uca_outboard.x_mm,
            corner.uca_outboard.y_mm,
            corner.uca_outboard.z_mm,
        ])
        self._lbj_0 = np.array([
            corner.lca_outboard.x_mm,
            corner.lca_outboard.y_mm,
            corner.lca_outboard.z_mm,
        ])
        self._tro_0 = np.array([
            corner.tie_rod_outboard.x_mm,
            corner.tie_rod_outboard.y_mm,
            corner.tie_rod_outboard.z_mm,
        ])

        # Chassis-side pickup points (move with wheel travel/roll)
        self._uca_if_0 = self._to_vec(corner.uca_inboard_front)
        self._uca_ir_0 = self._to_vec(corner.uca_inboard_rear)
        self._lca_if_0 = self._to_vec(corner.lca_inboard_front)
        self._lca_ir_0 = self._to_vec(corner.lca_inboard_rear)
        self._tri_0 = self._to_vec(corner.tie_rod_inboard)
        self._wc_0 = self._to_vec(corner.wheel_center)

        # Arm lengths (UCA/LCA effective outboard-to-axis distances via plane)
        self._uca_len = self._arm_length(
            self._uca_if_0, self._uca_ir_0, self._ubj_0
        )
        self._lca_len = self._arm_length(
            self._lca_if_0, self._lca_ir_0, self._lbj_0
        )
        self._tr_len = float(np.linalg.norm(self._tri_0 - self._tro_0))

        # Rigid-body inter-distances on the upright
        self._d_ubj_lbj = float(np.linalg.norm(self._ubj_0 - self._lbj_0))
        self._d_ubj_tro = float(np.linalg.norm(self._ubj_0 - self._tro_0))
        self._d_lbj_tro = float(np.linalg.norm(self._lbj_0 - self._tro_0))

        # Upright body frame: WC and spin axis stored as coefficients
        # in the orthonormal frame built from UBJ-LBJ-TRO triangle
        self._wc_offset = self._decompose_in_body_frame(
            self._wc_0 - self._lbj_0
        )
        self._spin_axis_body = self._compute_static_spin_axis()

        self._tire_r = corner.tire.loaded_radius_mm

    @staticmethod
    def _to_vec(hp: object) -> Arr:
        return np.array([hp.x_mm, hp.y_mm, hp.z_mm])  # type: ignore[attr-defined]

    @staticmethod
    def _arm_length(
        inboard_front: Arr,
        inboard_rear: Arr,
        outboard: Arr,
    ) -> float:
        """Distance from outboard ball joint to the inboard pivot axis."""
        axis = inboard_rear - inboard_front
        axis_unit = axis / np.linalg.norm(axis)
        to_obp = outboard - inboard_front
        proj = np.dot(to_obp, axis_unit) * axis_unit
        perp = to_obp - proj
        return float(np.linalg.norm(perp))

    def _build_body_frame(
        self,
        ubj: Arr,
        lbj: Arr,
        tro: Arr,
    ) -> tuple[Arr, Arr, Arr]:
        """Build an orthonormal frame from the UBJ-LBJ-TRO triangle.

        e1: along kingpin (LBJ→UBJ)
        e3: normal to the UBJ-LBJ-TRO plane
        e2: completes the right-handed frame
        """
        e1 = ubj - lbj
        e1_unit: Arr = e1 / np.linalg.norm(e1)
        e_temp = tro - lbj
        e3 = np.cross(e1_unit, e_temp)
        e3_unit: Arr = e3 / np.linalg.norm(e3)
        e2_unit: Arr = np.cross(e3_unit, e1_unit)
        return e1_unit, e2_unit, e3_unit

    def _decompose_in_body_frame(self, vec: Arr) -> Arr:
        """Decompose a vector into the static upright body frame."""
        e1, e2, e3 = self._build_body_frame(
            self._ubj_0, self._lbj_0, self._tro_0
        )
        return np.array([
            float(np.dot(vec, e1)),
            float(np.dot(vec, e2)),
            float(np.dot(vec, e3)),
        ])

    def _compute_static_spin_axis(self) -> Arr:
        """Build the spin axis from static_camber_deg and static_toe_deg_per_side,
        then decompose into the upright body frame.

        The spin axis is the wheel axle direction, pointing outboard.
        At zero camber and zero toe:
          - left wheel: spin axis = +Y
          - right wheel: spin axis = -Y
        Camber rotates about X (negative camber = spin axis tilts up at outboard end).
        Toe rotates about Z (toe-in = front of wheel toward centreline).
        """
        camber_rad = math.radians(self._corner.static_camber_deg)
        toe_rad = math.radians(self._corner.static_toe_deg_per_side)

        if self._is_left:
            # Base direction: +Y. Camber rotates in Y-Z plane.
            # Negative camber: top inboard → spin axis tilts upward at outboard.
            # atan2 convention: camber_deg = -atan2(z, y) for left.
            # So z_component = -sin(camber_rad), y_component = cos(camber_rad).
            sy = math.cos(camber_rad)
            sz = -math.sin(camber_rad)
            # Toe rotates in X-Y plane. Toe-in yaws the front of the wheel
            # toward the centreline (-Y for a left wheel), which swings the
            # outboard-pointing spin axis toward +X. Both sides therefore have
            # x_component = +sin(toe_rad) * cos(camber_rad) for toe-in; only
            # the y_component differs in sign.
            # toe_deg_per_side = +atan2(x, y) for left.
            sx = math.sin(toe_rad) * math.cos(camber_rad)
            sy = math.cos(toe_rad) * math.cos(camber_rad)
        else:
            # Base direction: -Y. Camber rotates in Y-Z plane.
            # camber_deg = -atan2(z, -y) for right.
            sy = -math.cos(camber_rad)
            sz = -math.sin(camber_rad)
            sx = math.sin(toe_rad) * math.cos(camber_rad)
            sy = -math.cos(toe_rad) * math.cos(camber_rad)

        spin_global = np.array([sx, sy, sz])
        spin_global = spin_global / np.linalg.norm(spin_global)

        return self._decompose_in_body_frame(spin_global)

    def _reconstruct_from_body(
        self,
        coeffs: Arr,
        ubj: Arr,
        lbj: Arr,
        tro: Arr,
    ) -> Arr:
        """Reconstruct a vector from body-frame coefficients."""
        e1, e2, e3 = self._build_body_frame(ubj, lbj, tro)
        result: Arr = coeffs[0] * e1 + coeffs[1] * e2 + coeffs[2] * e3
        return result

    def _reconstruct_wc(
        self,
        ubj: Arr,
        lbj: Arr,
        tro: Arr,
    ) -> Arr:
        """Reconstruct wheel centre from solved upright positions."""
        result: Arr = lbj + self._reconstruct_from_body(
            self._wc_offset, ubj, lbj, tro
        )
        return result

    def _reconstruct_spin_axis(
        self,
        ubj: Arr,
        lbj: Arr,
        tro: Arr,
    ) -> Arr:
        """Reconstruct the spin axis direction in global frame."""
        spin: Arr = self._reconstruct_from_body(
            self._spin_axis_body, ubj, lbj, tro
        )
        return spin / np.linalg.norm(spin)

    def _move_chassis_points(
        self, chassis_dz_mm: float, roll_deg: float, rack_mm: float
    ) -> tuple[
        Arr,
        Arr,
        Arr,
        Arr,
        Arr,
    ]:
        """Apply chassis motion to all inboard pickup points.

        chassis_dz_mm: vertical chassis displacement (positive = chassis up).
        Roll: rotation about X-axis (right-hand rule).
        Rack: translation of tie-rod inboard point along Y-axis.
        """
        dz = np.array([0.0, 0.0, chassis_dz_mm])

        if abs(roll_deg) > 1e-12:
            R = Rotation.from_rotvec(math.radians(roll_deg) * np.array([1.0, 0.0, 0.0]))
            rot = R.as_matrix()
        else:
            rot = np.eye(3)

        uca_if = rot @ self._uca_if_0 + dz
        uca_ir = rot @ self._uca_ir_0 + dz
        lca_if = rot @ self._lca_if_0 + dz
        lca_ir = rot @ self._lca_ir_0 + dz
        tri = rot @ self._tri_0 + dz + np.array([0.0, rack_mm, 0.0])

        return uca_if, uca_ir, lca_if, lca_ir, tri

    def _residuals(
        self,
        x: Arr,
        uca_if: Arr,
        uca_ir: Arr,
        lca_if: Arr,
        lca_ir: Arr,
        tri: Arr,
    ) -> Arr:
        """Compute 9 residuals for the 9 unknowns.

        Unknowns: x = [ubj_x, ubj_y, ubj_z, lbj_x, lbj_y, lbj_z, tro_x, tro_y, tro_z]

        Constraints:
          0: UCA arm length (distance from UBJ to UCA pivot axis)
          1: UCA axial position (projection along pivot axis)
          2: LCA arm length (distance from LBJ to LCA pivot axis)
          3: LCA axial position (projection along pivot axis)
          4: Tie-rod length (TRO to TRI)
          5-7: Upright rigid-body inter-distances (UBJ-LBJ, UBJ-TRO, LBJ-TRO)
          8: WC-to-kingpin projection (anti-flip)
        """
        ubj = x[0:3]
        lbj = x[3:6]
        tro = x[6:9]

        r = np.empty(9)

        # UCA: UBJ must be at the correct distance from the UCA axis
        r[0] = self._arm_length(uca_if, uca_ir, ubj) - self._uca_len
        # Also constrain UBJ to lie in the plane swept by the UCA
        # by requiring UBJ projected onto the axis is within bounds
        uca_axis = uca_ir - uca_if
        uca_axis_unit = uca_axis / np.linalg.norm(uca_axis)
        ubj_proj = np.dot(ubj - uca_if, uca_axis_unit)
        ubj_proj_0 = np.dot(self._ubj_0 - self._uca_if_0, uca_axis_unit)
        r[1] = ubj_proj - ubj_proj_0

        # LCA: LBJ must be at the correct distance from the LCA axis
        r[2] = self._arm_length(lca_if, lca_ir, lbj) - self._lca_len
        lca_axis = lca_ir - lca_if
        lca_axis_unit = lca_axis / np.linalg.norm(lca_axis)
        lbj_proj = np.dot(lbj - lca_if, lca_axis_unit)
        lbj_proj_0 = np.dot(self._lbj_0 - self._lca_if_0, lca_axis_unit)
        r[3] = lbj_proj - lbj_proj_0

        # Tie rod length
        r[4] = float(np.linalg.norm(tro - tri)) - self._tr_len

        # Upright rigid-body constraints
        r[5] = float(np.linalg.norm(ubj - lbj)) - self._d_ubj_lbj
        r[6] = float(np.linalg.norm(ubj - tro)) - self._d_ubj_tro
        r[7] = float(np.linalg.norm(lbj - tro)) - self._d_lbj_tro

        # 9th constraint: wheel centre must maintain its Z relationship
        # to the upright (prevents the solution from flipping)
        wc = self._reconstruct_wc(ubj, lbj, tro)
        kingpin = ubj - lbj
        kingpin_unit = kingpin / np.linalg.norm(kingpin)
        wc_to_lbj = wc - lbj
        wc_along_kp = np.dot(wc_to_lbj, kingpin_unit)
        kp_ref = (self._ubj_0 - self._lbj_0) / np.linalg.norm(self._ubj_0 - self._lbj_0)
        wc_along_kp_ref = float(np.dot(self._wc_0 - self._lbj_0, kp_ref))
        r[8] = wc_along_kp - wc_along_kp_ref

        return r

    def _extract_angles(
        self,
        ubj: Arr,
        lbj: Arr,
        spin_unit: Arr,
    ) -> tuple[float, float, float, float]:
        """Extract camber, toe, caster, and KPI from solved positions.

        ISO 8855: X+ forward, Y+ LEFT, Z+ up.

        Returns (camber_deg, toe_deg_per_side, caster_deg, kpi_deg) — all in degrees.

        Sign conventions:
          Camber: negative = top inboard (both sides).
          Toe: positive = toe-in (front of wheel points toward centreline).
          Caster: positive = kingpin axis tilts rearward at top.
          KPI: positive = kingpin axis tilts inboard at top.

        The spin axis is the wheel axle direction, reconstructed from the
        body-frame representation stored at construction. It is independent
        of the kingpin axis — camber and KPI are separate design parameters.
        """
        kingpin = ubj - lbj
        kp_len = float(np.linalg.norm(kingpin))
        kingpin_unit = kingpin / kp_len

        # --- Camber ---
        spin_yz = np.array([0.0, spin_unit[1], spin_unit[2]])
        spin_yz_norm = float(np.linalg.norm(spin_yz))
        if spin_yz_norm < 1e-12:
            camber_deg = 0.0
        else:
            spin_yz_unit = spin_yz / spin_yz_norm
            if self._is_left:
                camber_deg = -math.degrees(math.atan2(spin_yz_unit[2], spin_yz_unit[1]))
            else:
                camber_deg = -math.degrees(math.atan2(spin_yz_unit[2], -spin_yz_unit[1]))

        # --- Toe (per side) ---
        spin_xy = np.array([spin_unit[0], spin_unit[1], 0.0])
        spin_xy_norm = float(np.linalg.norm(spin_xy))
        if spin_xy_norm < 1e-12:
            toe_deg_per_side = 0.0
        else:
            spin_xy_unit = spin_xy / spin_xy_norm
            # Toe-in swings the outboard-pointing spin axis toward +X on BOTH
            # sides, so both branches must return the same sign for the same
            # physical toe direction. Only the y_component sign differs.
            if self._is_left:
                toe_deg_per_side = math.degrees(math.atan2(spin_xy_unit[0], spin_xy_unit[1]))
            else:
                toe_deg_per_side = -math.degrees(math.atan2(-spin_xy_unit[0], -spin_xy_unit[1]))

        # --- Caster ---
        # Kingpin projected onto X-Z side view.
        # Positive caster = top of kingpin tilts rearward (X < 0 at top).
        kp_xz = np.array([kingpin_unit[0], 0.0, kingpin_unit[2]])
        kp_xz_norm = float(np.linalg.norm(kp_xz))
        if kp_xz_norm < 1e-12:
            caster_deg = 0.0
        else:
            kp_xz_unit = kp_xz / kp_xz_norm
            # atan2(-x, z): positive when kingpin top tilts rearward
            caster_deg = math.degrees(math.atan2(-kp_xz_unit[0], kp_xz_unit[2]))

        # --- KPI (kingpin inclination) ---
        # Kingpin projected onto Y-Z front view.
        # Positive KPI = top of kingpin tilts inboard.
        kp_yz = np.array([0.0, kingpin_unit[1], kingpin_unit[2]])
        kp_yz_norm = float(np.linalg.norm(kp_yz))
        if kp_yz_norm < 1e-12:
            kpi_deg = 0.0
        else:
            kp_yz_unit = kp_yz / kp_yz_norm
            if self._is_left:
                # Left: inboard = toward -Y. Positive KPI when Y < 0 at top.
                kpi_deg = math.degrees(math.atan2(-kp_yz_unit[1], kp_yz_unit[2]))
            else:
                # Right: inboard = toward +Y. Positive KPI when Y > 0 at top.
                kpi_deg = math.degrees(math.atan2(kp_yz_unit[1], kp_yz_unit[2]))

        return camber_deg, toe_deg_per_side, caster_deg, kpi_deg

    def solve(
        self,
        wheel_travel_mm: float = 0.0,
        roll_deg: float = 0.0,
        rack_mm: float = 0.0,
    ) -> SolverResult:
        """Solve for the displaced upright positions.

        Args:
            wheel_travel_mm: Wheel vertical travel relative to the chassis.
                Positive = bump (wheel moves up relative to chassis).
                Internally applied as chassis Z displacement of -wheel_travel_mm.
            roll_deg: Chassis roll angle (positive = roll right in ISO 8855).
            rack_mm: Rack travel (positive = rack moves in +Y direction).

        Returns:
            SolverResult with solved positions and angles.
        """
        uca_if, uca_ir, lca_if, lca_ir, tri = self._move_chassis_points(
            -wheel_travel_mm, roll_deg, rack_mm
        )

        x0 = np.concatenate([self._ubj_0, self._lbj_0, self._tro_0])

        result = least_squares(
            self._residuals,
            x0,
            args=(uca_if, uca_ir, lca_if, lca_ir, tri),
            method="trf",
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            max_nfev=500,
        )

        converged = bool(result.success) and result.cost < 1e-10
        ubj = result.x[0:3]
        lbj = result.x[3:6]
        tro = result.x[6:9]
        wc = self._reconstruct_wc(ubj, lbj, tro)
        spin = self._reconstruct_spin_axis(ubj, lbj, tro)

        camber_deg, toe_deg, caster_deg, kpi_deg = self._extract_angles(
            ubj, lbj, spin
        )

        # Contact patch with camber correction
        gamma = math.radians(camber_deg)
        r = self._tire_r
        lateral_shift = r * math.sin(gamma)
        vertical_corr = r * (1.0 - math.cos(gamma))
        cp_y = wc[1] + lateral_shift if self._is_left else wc[1] - lateral_shift
        cp_z = wc[2] - r + vertical_corr

        return SolverResult(
            ubj=DerivedPoint(x_mm=float(ubj[0]), y_mm=float(ubj[1]), z_mm=float(ubj[2])),
            lbj=DerivedPoint(x_mm=float(lbj[0]), y_mm=float(lbj[1]), z_mm=float(lbj[2])),
            tro=DerivedPoint(x_mm=float(tro[0]), y_mm=float(tro[1]), z_mm=float(tro[2])),
            wheel_center=DerivedPoint(x_mm=float(wc[0]), y_mm=float(wc[1]), z_mm=float(wc[2])),
            contact_patch=DerivedPoint(x_mm=float(wc[0]), y_mm=cp_y, z_mm=cp_z),
            camber_deg=camber_deg,
            toe_deg_per_side=toe_deg,
            caster_deg=caster_deg,
            kpi_deg=kpi_deg,
            converged=converged,
            residual_norm=float(np.sqrt(2.0 * result.cost)),
            nfev=int(result.nfev),
            njev=int(result.njev) if result.njev is not None else 0,
        )
