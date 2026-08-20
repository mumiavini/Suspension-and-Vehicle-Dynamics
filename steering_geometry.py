#!/usr/bin/env python3
"""
steering_geometry.py
====================

Tie-rod and rack synthesis for the PUCPR Racing FSAE 2027 front axle.

Sibling script to ``sla_geometry.py``: consumes its front-axle solution (the
four wishbone hardpoints in the front view), adds steering-specific inputs
(caster, steering arm, rack, effort), and outputs the tie-rod/rack hardpoints
plus steering behaviour over bump and steer sweeps.

Uses ``vdcore.geometry.solver.DWSolver`` for the 3D kinematic solve, so
bump steer and Ackermann are extracted from the full rigid-upright problem
rather than the front-view four-bar.  Camber-vs-bump from this script will
differ slightly from ``sla_geometry.py``'s 2D sweep — that is expected.

KNOWN LIMITATION: DWSolver translates the inner tie-rod joint purely along Y
(solver.py:264), so the rack axis is assumed lateral and horizontal.  This
covers essentially every FSAE rack.  An inclined rack would need a rack-axis
direction added to ``_move_chassis_points``.

COORDINATE SYSTEMS
------------------
* DESIGN frame (inherited from sla_geometry.py):
      y positive OUTBOARD, z positive UP, x positive REARWARD.
      This is the frame in which the designer edits SteeringInputs.

* MODEL frame = ISO 8855:
      X positive FORWARD, Y positive LEFT, Z positive UP.
      Conversion:  X_iso = -x_rearward,  Y_iso = ±y_outboard,  Z_iso = z.

Units: mm and degrees throughout I/O. Radians only inside the solvers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from sla_geometry import (
    _RULE,
    D2R,
    FRONT_2027,
    R2D,
    VEHICLE_2027,
    AxleGeometry,
    AxleInputs,
    Band,
    KinematicError,
    VehicleData,
    _band,
    _flag,
    build_corner,
    nz,
    solve_axle,
)
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage

W = 78

# --------------------------------------------------------------------------- #
# 1. INPUTS
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class SteeringLimits:
    """Band-check targets for steering KPIs."""

    bump_steer_deg_per_mm: Band = (0.0, 0.005)
    ackermann_pct: Band = (60.0, 120.0)
    steering_ratio: Band = (3.0, 7.0)
    tie_rod_length_mm: Band = (150.0, 350.0)
    rod_end_misalignment_deg: float = 12.0
    mechanical_trail_mm: Band = (10.0, 35.0)
    steering_wheel_torque_Nm: float = 10.0
    rack_x_window_mm: Band = (-80.0, 80.0)
    rack_z_window_mm: Band = (50.0, 180.0)


@dataclass(frozen=True, kw_only=True)
class SteeringInputs:
    """Everything the designer chooses for the steering system. Nothing here
    is derived."""

    # kingpin axis in side view (absent from sla_geometry.py)
    caster_deg: float = 5.0
    caster_offset_mm: float = 0.0

    # steering arm — cylindrical parameterisation about the kingpin axis
    tro_height_along_kingpin_mm: float
    steer_arm_length_mm: float
    steer_arm_angle_deg: float

    # rack — axis assumed lateral and horizontal (along Y)
    rack_x_mm: float
    rack_z_mm: float
    rack_half_length_mm: float

    # rack hardware
    pinion_radius_mm: float
    max_rack_travel_mm: float
    steering_wheel_diameter_mm: float = 260.0

    # static alignment
    static_toe_deg_per_side: float = 0.0

    # design intent targets (arguments to a query, never defaults the script chases)
    target_ackermann_pct: float = 100.0
    ackermann_at_steer_deg: float = 10.0
    target_bump_steer_deg_per_mm: float = 0.0

    # effort
    mu_parking: float = 1.0

    # sweeps
    steer_sweep_deg: float = 25.0
    n_sweep: int = 21
    hardpoint_tol_mm: float = 1.0

    limits: SteeringLimits = field(default_factory=SteeringLimits)


# --------------------------------------------------------------------------- #
# 2. CASTER
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CasterResult:
    """3D ball joints with caster applied, plus kingpin geometry."""

    lbj_3d: np.ndarray  # (3,)
    ubj_3d: np.ndarray  # (3,)
    kingpin_axis: np.ndarray  # unit (3,), LBJ→UBJ
    mechanical_trail_mm: float
    scrub_radius_mm: float
    kp_ground_x: float
    kp_ground_y: float


def apply_caster(
    geo: AxleGeometry,
    steer: SteeringInputs,
) -> CasterResult:
    """Lift sla_geometry.py's 2D ball joints to 3D with caster applied.

    Pivots about wheel-centre height so that ``caster_offset_mm`` translates
    the kingpin axis bodily and ``caster_deg`` tilts it, giving independent
    control of caster angle and mechanical trail.

    Design frame: x +rearward, y +outboard, z +up.
    Returns points in the design frame.
    """
    inp = geo.inputs
    tau = steer.caster_deg * D2R
    lr = inp.loaded_radius_mm

    lbj_x = (
        inp.axle_x_mm
        + steer.caster_offset_mm
        + (float(geo.lbj[1]) - lr) * np.tan(tau)
    )
    ubj_x = (
        inp.axle_x_mm
        + steer.caster_offset_mm
        + (float(geo.ubj[1]) - lr) * np.tan(tau)
    )

    lbj_3d = np.array([lbj_x, float(geo.lbj[0]), float(geo.lbj[1])])
    ubj_3d = np.array([ubj_x, float(geo.ubj[0]), float(geo.ubj[1])])

    kp = ubj_3d - lbj_3d
    kp_unit = kp / np.linalg.norm(kp)

    # kingpin ground intercept (z=0 plane)
    if abs(kp_unit[2]) < 1e-12:
        kp_gnd_x = lbj_3d[0]
        kp_gnd_y = lbj_3d[1]
    else:
        t = -lbj_3d[2] / kp_unit[2]
        kp_gnd_x = lbj_3d[0] + t * kp_unit[0]
        kp_gnd_y = lbj_3d[1] + t * kp_unit[1]

    cp_x = inp.axle_x_mm
    cp_y = inp.half_track_mm

    # mechanical trail: kingpin ground X vs contact patch X (positive = intercept forward of patch)
    # in design frame x is +rearward, so "forward" = smaller x
    mechanical_trail_mm = float(cp_x - kp_gnd_x)

    # scrub radius: contact patch y vs kingpin ground y (positive = intercept inboard)
    scrub_radius_mm = float(cp_y - kp_gnd_y)

    return CasterResult(
        lbj_3d=lbj_3d,
        ubj_3d=ubj_3d,
        kingpin_axis=kp_unit,
        mechanical_trail_mm=mechanical_trail_mm,
        scrub_radius_mm=scrub_radius_mm,
        kp_ground_x=float(kp_gnd_x),
        kp_ground_y=float(kp_gnd_y),
    )


# --------------------------------------------------------------------------- #
# 3. TRO AND TRI
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SteeringGeometry:
    """Solved steering geometry for one axle (both sides symmetric)."""

    tro_left: np.ndarray   # (3,) design frame
    tri_left: np.ndarray   # (3,) design frame
    tro_right: np.ndarray  # (3,) design frame
    tri_right: np.ndarray  # (3,) design frame
    tie_rod_length_mm: float
    steer_arm_length_mm: float
    kingpin_axis: np.ndarray   # unit (3,) design frame
    mechanical_trail_mm: float
    scrub_radius_mm: float
    geometric_ackermann_pct: float
    caster_result: CasterResult
    lbj_3d: np.ndarray   # (3,) caster-corrected, design frame
    ubj_3d: np.ndarray   # (3,) caster-corrected, design frame


def _tro_from_arm_params(
    caster: CasterResult,
    steer: SteeringInputs,
) -> np.ndarray:
    """Compute TRO position from the cylindrical parameterisation about the
    kingpin axis.

    Returns TRO in the design frame (x +rearward, y +outboard, z +up).
    """
    e_kp = caster.kingpin_axis

    # forward direction = -x in the design frame (forward = negative x)
    e_fwd_raw = np.array([-1.0, 0.0, 0.0])
    # remove component along kingpin
    e_fwd_raw = e_fwd_raw - np.dot(e_fwd_raw, e_kp) * e_kp
    e_fwd = e_fwd_raw / np.linalg.norm(e_fwd_raw)

    # lateral: kingpin × forward gives inboard direction
    e_lat = np.cross(e_kp, e_fwd)
    e_lat = e_lat / np.linalg.norm(e_lat)

    h = steer.tro_height_along_kingpin_mm
    L = steer.steer_arm_length_mm
    theta = steer.steer_arm_angle_deg * D2R

    tro: np.ndarray = (
        caster.lbj_3d
        + h * e_kp
        + L * (np.cos(theta) * e_fwd + np.sin(theta) * e_lat)
    )
    return tro


def _tri_from_rack_params(steer: SteeringInputs) -> tuple[np.ndarray, np.ndarray]:
    """Compute TRI left and right from rack parameters.

    Returns (tri_left, tri_right) in the design frame.
    """
    tri_left = np.array([
        steer.rack_x_mm,
        steer.rack_half_length_mm,
        steer.rack_z_mm,
    ])
    tri_right = np.array([
        steer.rack_x_mm,
        steer.rack_half_length_mm,
        steer.rack_z_mm,
    ])
    return tri_left, tri_right


def _geometric_ackermann_pct(
    tro: np.ndarray,
    lbj_3d: np.ndarray,
    ubj_3d: np.ndarray,
    veh: VehicleData,
    inp: AxleInputs,
) -> float:
    """Geometric Ackermann % from the kingpin→TRO line vs rear axle centre.

    Measured in the top-view (x-y plane in the design frame, where x is
    +rearward and y is +outboard). 100% Ackermann means the TRO–kingpin
    line, extended, crosses the rear axle centreline.
    """
    # project to top view (x, y) — design frame
    kp_mid = (lbj_3d + ubj_3d) / 2.0

    kp_xy = np.array([kp_mid[0], kp_mid[1]])
    tro_xy = np.array([tro[0], tro[1]])

    # direction from kp to tro
    d = tro_xy - kp_xy
    if abs(d[0]) < 1e-12:
        return 0.0

    # where does the line kp→tro cross y=0?
    t = (0.0 - kp_xy[1]) / d[1] if abs(d[1]) > 1e-12 else np.nan
    if np.isnan(t):
        return 0.0

    x_cross = kp_xy[0] + t * d[0]
    x_ideal = veh.wheelbase_mm

    # Ackermann % = 100 if the line crosses the rear axle centre
    if abs(x_ideal) < 1e-12:
        return 0.0
    return float(100.0 * x_cross / x_ideal)


def synthesize_steering(
    geo: AxleGeometry,
    steer: SteeringInputs,
    veh: VehicleData,
) -> SteeringGeometry:
    """Build the complete steering geometry from designer inputs."""
    caster = apply_caster(geo, steer)

    tro_left = _tro_from_arm_params(caster, steer)
    tri_left, tri_right = _tri_from_rack_params(steer)

    # right side: mirror the TRO across the car centreline (y=0 in design frame)
    tro_right = tro_left.copy()
    tro_right[1] = -tro_right[1]

    # right side TRI: mirror y
    tri_right[1] = -tri_right[1]

    tie_rod_length = float(np.linalg.norm(tri_left - tro_left))

    ack = _geometric_ackermann_pct(
        tro_left, caster.lbj_3d, caster.ubj_3d, veh, geo.inputs,
    )

    return SteeringGeometry(
        tro_left=tro_left,
        tri_left=tri_left,
        tro_right=tro_right,
        tri_right=tri_right,
        tie_rod_length_mm=tie_rod_length,
        steer_arm_length_mm=steer.steer_arm_length_mm,
        kingpin_axis=caster.kingpin_axis,
        mechanical_trail_mm=caster.mechanical_trail_mm,
        scrub_radius_mm=caster.scrub_radius_mm,
        geometric_ackermann_pct=ack,
        caster_result=caster,
        lbj_3d=caster.lbj_3d,
        ubj_3d=caster.ubj_3d,
    )


# --------------------------------------------------------------------------- #
# 4. VDCORE BRIDGE
# --------------------------------------------------------------------------- #


def _hp(
    name: str,
    x: float,
    y: float,
    z: float,
    tol: float,
) -> Hardpoint:
    return Hardpoint(
        name=name,
        x_mm=nz(x),
        y_mm=nz(y),
        z_mm=nz(z),
        source="design_intent",
        tol_mm=tol,
    )


def build_vdcore_corner(
    front_geo: AxleGeometry,
    sg: SteeringGeometry,
    steer: SteeringInputs,
    side: Literal["left", "right"],
    corner_id: Literal["FL", "FR"],
) -> Corner:
    """Build a vdcore Corner from the sla_geometry solution plus steering.

    Uses build_corner for the 4 inboard pickups and wheel centre, then
    substitutes the caster-corrected UCA_OUT / LCA_OUT and adds the tie rod.

    Applies design→ISO conversion: X_iso = -x_rearward, Y_iso = ±y_outboard.
    """
    inp = front_geo.inputs
    tol = steer.hardpoint_tol_mm
    sy = 1.0 if side == "left" else -1.0

    base = build_corner(front_geo, side, corner_id)

    # caster-corrected outboard points (design frame → ISO)
    lbj = sg.lbj_3d
    ubj = sg.ubj_3d

    uca_out = _hp(
        "UCA_OUT",
        nz(-ubj[0]),
        nz(sy * ubj[1]),
        nz(ubj[2]),
        tol,
    )
    lca_out = _hp(
        "LCA_OUT",
        nz(-lbj[0]),
        nz(sy * lbj[1]),
        nz(lbj[2]),
        tol,
    )

    # tie rod points (design frame → ISO)
    # tro_right/tri_right already have negative Y from the mirror in
    # synthesize_steering, so the ISO conversion is just X_iso = -x.
    if side == "left":
        tro_d, tri_d = sg.tro_left, sg.tri_left
    else:
        tro_d, tri_d = sg.tro_right, sg.tri_right

    tie_rod_out = _hp(
        "TIE_ROD_OUT",
        nz(-tro_d[0]),
        nz(tro_d[1]),
        nz(tro_d[2]),
        tol,
    )
    tie_rod_in = _hp(
        "TIE_ROD_IN",
        nz(-tri_d[0]),
        nz(tri_d[1]),
        nz(tri_d[2]),
        tol,
    )

    # repackage the base inboard pickups with provenance
    def _rehp(name: str, pt: tuple[float, float, float]) -> Hardpoint:
        return _hp(name, pt[0], pt[1], pt[2], tol)

    tire = TirePackage(
        loaded_radius_mm=inp.loaded_radius_mm,
        source="design_intent",
        tol_mm=tol,
    )

    wc_pt = base.wheel_centre
    return Corner(
        corner_id=corner_id,
        uca_inboard_front=_rehp("UCA_IN_FRONT", base.uca_in_front),
        uca_inboard_rear=_rehp("UCA_IN_REAR", base.uca_in_rear),
        uca_outboard=uca_out,
        lca_inboard_front=_rehp("LCA_IN_FRONT", base.lca_in_front),
        lca_inboard_rear=_rehp("LCA_IN_REAR", base.lca_in_rear),
        lca_outboard=lca_out,
        tie_rod_inboard=tie_rod_in,
        tie_rod_outboard=tie_rod_out,
        wheel_center=_rehp("WHEEL_CENTER", wc_pt),
        tire=tire,
        static_camber_deg=inp.static_camber_deg,
        static_toe_deg_per_side=steer.static_toe_deg_per_side,
    )


# --------------------------------------------------------------------------- #
# 5. KINEMATICS
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SteeringRates:
    """Kinematic rates derived from the 3D solver."""

    bump_steer_deg_per_mm_per_side: float
    bump_steer_total_toe_deg_per_mm: float
    toe_at_full_bump_deg_per_side: float
    toe_at_full_droop_deg_per_side: float
    c_factor_mm_per_deg: float
    steering_ratio: float
    max_steer_at_stroke_deg: float      # outer wheel (governs turning radius)
    max_steer_inner_at_stroke_deg: float  # inner wheel (larger due to Ackermann)
    ackermann_pct_at_target: float
    worst_rod_end_misalignment_deg: float


class SteeringKinematics:
    """Drives FL and FR DWSolvers with the same rack travel."""

    def __init__(
        self,
        front_geo: AxleGeometry,
        sg: SteeringGeometry,
        steer: SteeringInputs,
        veh: VehicleData,
    ) -> None:
        self.steer = steer
        self.veh = veh
        self.inp = front_geo.inputs

        corner_fl = build_vdcore_corner(front_geo, sg, steer, "left", "FL")
        corner_fr = build_vdcore_corner(front_geo, sg, steer, "right", "FR")

        self.solver_fl = DWSolver(corner_fl)
        self.solver_fr = DWSolver(corner_fr)
        self.sg = sg

    def solve_pair(
        self,
        bump_mm: float = 0.0,
        rack_mm: float = 0.0,
    ) -> tuple[SolverResult, SolverResult]:
        """Solve FL and FR for given bump and rack. Raises on non-convergence."""
        fl = self.solver_fl.solve(wheel_travel_mm=bump_mm, rack_mm=rack_mm)
        if not fl.converged:
            raise KinematicError(
                f"FL solver did not converge at bump={bump_mm:.2f}, rack={rack_mm:.2f}"
            )
        fr = self.solver_fr.solve(wheel_travel_mm=bump_mm, rack_mm=rack_mm)
        if not fr.converged:
            raise KinematicError(
                f"FR solver did not converge at bump={bump_mm:.2f}, rack={rack_mm:.2f}"
            )
        return fl, fr

    def compute_rates(self) -> SteeringRates:
        inp = self.inp
        steer = self.steer

        # --- bump steer (central difference about static, FL only) ---
        # FL and FR are symmetric: |dtoe_fl| = |dtoe_fr| with opposite sign.
        # Averaging them cancels to zero. Use FL as the reference side.
        step = inp.travel_bump_mm / 20.0
        fl_up, _ = self.solve_pair(bump_mm=+step, rack_mm=0.0)
        fl_dn, _ = self.solve_pair(bump_mm=-step, rack_mm=0.0)

        bump_steer_per_side = (fl_up.toe_deg_per_side - fl_dn.toe_deg_per_side) / (2.0 * step)
        bump_steer_total = 2.0 * abs(bump_steer_per_side)

        # toe at full bump/droop (FL, per-side)
        fl_fb, _ = self.solve_pair(bump_mm=inp.travel_bump_mm, rack_mm=0.0)
        fl_fd, _ = self.solve_pair(bump_mm=-inp.travel_droop_mm, rack_mm=0.0)
        toe_full_bump = fl_fb.toe_deg_per_side
        toe_full_droop = fl_fd.toe_deg_per_side

        # --- C-factor: numerical (rack_mm per deg steer) ---
        eps = 0.5
        fl_p, _ = self.solve_pair(bump_mm=0.0, rack_mm=+eps)
        fl_m, _ = self.solve_pair(bump_mm=0.0, rack_mm=-eps)
        delta_toe_deg = fl_p.toe_deg_per_side - fl_m.toe_deg_per_side
        c_factor = float("inf") if abs(delta_toe_deg) < 1e-12 else (2.0 * eps) / delta_toe_deg

        # --- steering ratio ---
        if steer.pinion_radius_mm > 0:
            steering_ratio = 360.0 * abs(c_factor) / (2.0 * np.pi * steer.pinion_radius_mm)
        else:
            steering_ratio = float("inf")

        # --- max steer at stroke (outer = FR for +rack, inner = FL) ---
        fl_max, fr_max = self.solve_pair(bump_mm=0.0, rack_mm=steer.max_rack_travel_mm)
        fl_static, fr_static = self.solve_pair(bump_mm=0.0, rack_mm=0.0)
        max_steer_outer = abs(fr_max.toe_deg_per_side - fr_static.toe_deg_per_side)
        max_steer_inner = abs(fl_max.toe_deg_per_side - fl_static.toe_deg_per_side)

        # --- Ackermann at target steer angle ---
        ack_pct = self._ackermann_at_steer(steer.ackermann_at_steer_deg)

        # --- worst rod-end misalignment over bump × steer envelope ---
        worst_misalign = self._worst_rod_end_misalignment()

        return SteeringRates(
            bump_steer_deg_per_mm_per_side=bump_steer_per_side,
            bump_steer_total_toe_deg_per_mm=bump_steer_total,
            toe_at_full_bump_deg_per_side=toe_full_bump,
            toe_at_full_droop_deg_per_side=toe_full_droop,
            c_factor_mm_per_deg=c_factor,
            steering_ratio=steering_ratio,
            max_steer_at_stroke_deg=max_steer_outer,
            max_steer_inner_at_stroke_deg=max_steer_inner,
            ackermann_pct_at_target=ack_pct,
            worst_rod_end_misalignment_deg=worst_misalign,
        )

    def _ackermann_at_steer(self, outer_steer_deg: float) -> float:
        """Compute actual Ackermann % at a given outer wheel steer angle.

        Uses a LEFT turn (positive rack): FL is inner, FR is outer.
        Finds the rack where FR (outer) reaches ``outer_steer_deg``, then
        compares the FL (inner) steer with the Ackermann-ideal inner angle.
        """
        T = self.inp.track_mm
        L = self.veh.wheelbase_mm

        fl_static, fr_static = self.solve_pair(bump_mm=0.0, rack_mm=0.0)

        def outer_err(rack: float) -> float:
            try:
                _, fr = self.solve_pair(bump_mm=0.0, rack_mm=rack)
            except KinematicError:
                return float("nan")
            return abs(fr.toe_deg_per_side - fr_static.toe_deg_per_side) - outer_steer_deg

        lo = 0.01
        hi = self.steer.max_rack_travel_mm * 0.95

        try:
            val_lo, val_hi = outer_err(lo), outer_err(hi)
            if np.isnan(val_lo) or np.isnan(val_hi):
                return float("nan")
            if val_lo * val_hi > 0:
                return float("nan")
            rack_target = brentq(outer_err, lo, hi, xtol=1e-6)
        except (ValueError, KinematicError):
            return float("nan")

        fl_sol, fr_sol = self.solve_pair(bump_mm=0.0, rack_mm=rack_target)

        delta_outer = abs(fr_sol.toe_deg_per_side - fr_static.toe_deg_per_side)
        delta_inner = abs(fl_sol.toe_deg_per_side - fl_static.toe_deg_per_side)

        if delta_outer < 1e-12:
            return 0.0

        cot_outer = 1.0 / np.tan(delta_outer * D2R)
        cot_inner_ideal = cot_outer - T / L
        if abs(cot_inner_ideal) < 1e-12:
            delta_inner_ideal_deg = 90.0
        else:
            delta_inner_ideal_deg = R2D * np.arctan(1.0 / cot_inner_ideal)

        denom = delta_inner_ideal_deg - delta_outer
        if abs(denom) < 1e-12:
            return 100.0

        return float(100.0 * (delta_inner - delta_outer) / denom)

    def _worst_rod_end_misalignment(self) -> float:
        """Screening check: max swing of tie-rod direction over bump × steer."""
        # static tie-rod direction
        fl_s, _ = self.solve_pair(bump_mm=0.0, rack_mm=0.0)
        tro_s = np.array([fl_s.tro.x_mm, fl_s.tro.y_mm, fl_s.tro.z_mm])
        tri_s = np.array([
            self.solver_fl._tri_0[0],
            self.solver_fl._tri_0[1],
            self.solver_fl._tri_0[2],
        ])
        dir_s = tro_s - tri_s
        dir_s = dir_s / np.linalg.norm(dir_s)

        worst = 0.0
        bumps = np.linspace(
            -self.inp.travel_droop_mm, self.inp.travel_bump_mm, 9,
        )
        racks = np.linspace(
            -self.steer.max_rack_travel_mm, self.steer.max_rack_travel_mm, 9,
        )
        for b in bumps:
            for r in racks:
                try:
                    fl, _ = self.solve_pair(bump_mm=b, rack_mm=r)
                except KinematicError:
                    continue
                tro = np.array([fl.tro.x_mm, fl.tro.y_mm, fl.tro.z_mm])
                tri = tri_s + np.array([0.0, r, 0.0])
                d = tro - tri
                d = d / np.linalg.norm(d)
                cos_a = float(np.clip(np.dot(d, dir_s), -1.0, 1.0))
                angle = np.degrees(np.arccos(cos_a))
                worst = max(worst, angle)
        return worst

    def ackermann_curve(
        self,
        n: int = 0,
    ) -> list[tuple[float, float, float]]:
        """Return (outer_deg, inner_actual_deg, inner_ideal_deg) over the steer sweep."""
        if n == 0:
            n = self.steer.n_sweep
        T = self.inp.track_mm
        L = self.veh.wheelbase_mm
        fl_static, fr_static = self.solve_pair(bump_mm=0.0, rack_mm=0.0)

        results: list[tuple[float, float, float]] = []
        racks = np.linspace(
            0.0, self.steer.max_rack_travel_mm, n,
        )
        for rack in racks:
            fl, fr = self.solve_pair(bump_mm=0.0, rack_mm=rack)
            delta_inner = fl.toe_deg_per_side - fl_static.toe_deg_per_side
            delta_outer = fr.toe_deg_per_side - fr_static.toe_deg_per_side

            outer_abs = abs(delta_outer)
            if outer_abs < 1e-6:
                results.append((0.0, 0.0, 0.0))
                continue
            cot_o = 1.0 / np.tan(outer_abs * D2R)
            cot_i_ideal = cot_o - T / L
            ideal = 90.0 if abs(cot_i_ideal) < 1e-12 else R2D * np.arctan(1.0 / cot_i_ideal)

            results.append((outer_abs, abs(delta_inner), ideal))
        return results

    def bump_steer_curve(
        self,
        n: int = 0,
    ) -> list[tuple[float, float, float]]:
        """Return (bump_mm, toe_fl_deg_per_side, toe_fr_deg_per_side) over bump."""
        if n == 0:
            n = self.steer.n_sweep
        bumps = np.linspace(
            -self.inp.travel_droop_mm, self.inp.travel_bump_mm, n,
        )
        results: list[tuple[float, float, float]] = []
        for b in bumps:
            fl, fr = self.solve_pair(bump_mm=b, rack_mm=0.0)
            results.append((float(b), fl.toe_deg_per_side, fr.toe_deg_per_side))
        return results

    def steer_curve(
        self,
        n: int = 0,
    ) -> list[tuple[float, float, float, float]]:
        """Return (rack_mm, toe_fl, toe_fr, camber_fl) over rack sweep."""
        if n == 0:
            n = self.steer.n_sweep
        racks = np.linspace(
            -self.steer.max_rack_travel_mm, self.steer.max_rack_travel_mm, n,
        )
        fl_static, fr_static = self.solve_pair(bump_mm=0.0, rack_mm=0.0)
        results: list[tuple[float, float, float, float]] = []
        for r in racks:
            try:
                fl, fr = self.solve_pair(bump_mm=0.0, rack_mm=r)
            except KinematicError:
                continue
            results.append((
                float(r),
                fl.toe_deg_per_side - fl_static.toe_deg_per_side,
                fr.toe_deg_per_side - fr_static.toe_deg_per_side,
                fl.camber_deg,
            ))
        return results


# --------------------------------------------------------------------------- #
# 6. BACK-SOLVERS (opt-in)
# --------------------------------------------------------------------------- #


def y_tri_for_zero_bump_steer(
    front_geo: AxleGeometry,
    steer: SteeringInputs,
    veh: VehicleData,
) -> float:
    """Sweep rack_half_length_mm to hit target_bump_steer_deg_per_mm.

    Returns the value that would hit the target, or nan if no sign change.
    """
    target = steer.target_bump_steer_deg_per_mm

    def err(rhl: float) -> float:
        s = _replace(steer, rack_half_length_mm=rhl)
        sg = synthesize_steering(front_geo, s, veh)
        kin = SteeringKinematics(front_geo, sg, s, veh)
        rates = kin.compute_rates()
        return rates.bump_steer_deg_per_mm_per_side - target

    lo = steer.rack_half_length_mm * 0.5
    hi = steer.rack_half_length_mm * 1.5
    try:
        if err(lo) * err(hi) < 0:
            return brentq(err, lo, hi, xtol=1e-4)
    except (ValueError, KinematicError):
        pass
    return float("nan")


def arm_angle_for_ackermann_target(
    front_geo: AxleGeometry,
    steer: SteeringInputs,
    veh: VehicleData,
) -> float:
    """Sweep steer_arm_angle_deg to hit target_ackermann_pct.

    Returns the value that would hit the target, or nan if no sign change.
    """
    target = steer.target_ackermann_pct

    def err(angle: float) -> float:
        s = _replace(steer, steer_arm_angle_deg=angle)
        sg = synthesize_steering(front_geo, s, veh)
        kin = SteeringKinematics(front_geo, sg, s, veh)
        rates = kin.compute_rates()
        return rates.ackermann_pct_at_target - target

    lo = -30.0
    hi = 60.0
    try:
        if err(lo) * err(hi) < 0:
            return brentq(err, lo, hi, xtol=1e-4)
    except (ValueError, KinematicError):
        pass
    return float("nan")


# --------------------------------------------------------------------------- #
# 7. EFFORT
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SteeringEffort:
    """Parking-effort budget."""

    Fz_per_wheel_N: float
    kingpin_moment_Nm: float
    rack_force_N: float
    steering_wheel_torque_Nm: float
    rim_force_N: float
    scrub_contrib_mm: float
    trail_contrib_mm: float


def rack_force_parking(
    sg: SteeringGeometry,
    rates: SteeringRates,
    steer: SteeringInputs,
    veh: VehicleData,
) -> SteeringEffort:
    """Parking effort via virtual work with the solver-derived C-factor."""
    Fz = (
        veh.total_mass_kg * 9.81 * veh.front_mass_fraction / 2.0
    )
    mu = steer.mu_parking
    rs = abs(sg.scrub_radius_mm)
    tm = abs(sg.mechanical_trail_mm)

    M_kp_Nmm = mu * Fz * np.sqrt(rs**2 + tm**2)

    C = abs(rates.c_factor_mm_per_deg)
    F_rack = float("inf") if C < 1e-6 else float(2.0 * M_kp_Nmm * (np.pi / 180.0) / C)

    T_sw_Nm = float(F_rack * steer.pinion_radius_mm / 1000.0)
    F_rim_N = float(T_sw_Nm / (steer.steering_wheel_diameter_mm / 2000.0))

    return SteeringEffort(
        Fz_per_wheel_N=Fz,
        kingpin_moment_Nm=M_kp_Nmm / 1000.0,
        rack_force_N=F_rack,
        steering_wheel_torque_Nm=T_sw_Nm,
        rim_force_N=F_rim_N,
        scrub_contrib_mm=rs,
        trail_contrib_mm=tm,
    )


# --------------------------------------------------------------------------- #
# 8. HARDPOINTS OUTPUT (ISO 8855)
# --------------------------------------------------------------------------- #


STEER_POINT_NAMES: tuple[str, ...] = (
    "TIE_ROD_IN", "TIE_ROD_OUT", "UCA_OUT", "LCA_OUT",
)

Point3D_T = tuple[float, float, float]


@dataclass(frozen=True)
class SteeringCornerHP:
    """One corner's steering hardpoints in ISO 8855."""

    corner: Literal["FL", "FR"]
    tie_rod_in: Point3D_T
    tie_rod_out: Point3D_T
    uca_out: Point3D_T
    lca_out: Point3D_T

    def all_points(self) -> Iterator[tuple[str, Point3D_T]]:
        yield "TIE_ROD_IN", self.tie_rod_in
        yield "TIE_ROD_OUT", self.tie_rod_out
        yield "UCA_OUT", self.uca_out
        yield "LCA_OUT", self.lca_out


@dataclass(frozen=True)
class SteeringHardpoints:
    """Steering hardpoints for FL and FR in ISO 8855."""

    frame: str = (
        "ISO 8855: X+ forward, Y+ left, Z+ up; "
        "origin front axle / ground / centreline"
    )
    corners: tuple[SteeringCornerHP, ...] = ()

    def rows(self) -> Iterator[tuple[str, str, float, float, float]]:
        for c in self.corners:
            for name, (x, y, z) in c.all_points():
                yield c.corner, name, x, y, z

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "units": "mm",
            "corners": {
                c.corner: {n: [round(v, 3) for v in pt] for n, pt in c.all_points()}
                for c in self.corners
            },
        }


def build_steering_hardpoints(
    sg: SteeringGeometry,
) -> SteeringHardpoints:
    """Build the ISO 8855 hardpoint table for the steering deliverable.

    Design frame → ISO 8855: X_iso = -x_rearward, Y_iso = ±y_outboard, Z_iso = z.

    For left side (FL): Y_iso = +y_outboard (left = positive Y in ISO).
    For right side (FR): Y_iso = -y_outboard (right = negative Y in ISO).

    The left-side arrays (tri_left, tro_left) have y > 0 (outboard).
    The right-side arrays (tri_right, tro_right) have y < 0 (already mirrored
    in synthesize_steering). So the ISO conversion just negates x for both,
    and for right-side tie rod points the y is already correct.

    The ball joints (lbj_3d, ubj_3d) are stored in the left-side design frame
    (y > 0 = outboard), so they need sy = ±1 applied.
    """
    def to_iso_left(d: np.ndarray) -> Point3D_T:
        return (nz(-d[0]), nz(d[1]), nz(d[2]))

    def to_iso_right_mirrored(d: np.ndarray) -> Point3D_T:
        """For arrays where y is already mirrored (negative for right side)."""
        return (nz(-d[0]), nz(d[1]), nz(d[2]))

    def to_iso_bj(d: np.ndarray, side: Literal["left", "right"]) -> Point3D_T:
        """For ball joints stored in the left outboard frame (y > 0)."""
        sy = 1.0 if side == "left" else -1.0
        return (nz(-d[0]), nz(sy * d[1]), nz(d[2]))

    fl = SteeringCornerHP(
        corner="FL",
        tie_rod_in=to_iso_left(sg.tri_left),
        tie_rod_out=to_iso_left(sg.tro_left),
        uca_out=to_iso_bj(sg.ubj_3d, "left"),
        lca_out=to_iso_bj(sg.lbj_3d, "left"),
    )
    fr = SteeringCornerHP(
        corner="FR",
        tie_rod_in=to_iso_right_mirrored(sg.tri_right),
        tie_rod_out=to_iso_right_mirrored(sg.tro_right),
        uca_out=to_iso_bj(sg.ubj_3d, "right"),
        lca_out=to_iso_bj(sg.lbj_3d, "right"),
    )
    return SteeringHardpoints(corners=(fl, fr))


# --------------------------------------------------------------------------- #
# 9. REPORT
# --------------------------------------------------------------------------- #


def steering_report(
    sg: SteeringGeometry,
    rates: SteeringRates,
    effort: SteeringEffort,
    steer: SteeringInputs,
    hp: SteeringHardpoints,
) -> str:
    lim = steer.limits
    L: list[str] = []

    L.append("")
    L.append(_RULE)
    L.append(" STEERING GEOMETRY — FSAE 2027")
    L.append(_RULE)

    L.append("")
    L.append(" CASTER AND KINGPIN")
    L.append(f"   Caster angle                   {steer.caster_deg:8.2f} deg")
    L.append(f"   Caster offset                  {steer.caster_offset_mm:8.2f} mm")
    L.append(f"   Mechanical trail               {sg.mechanical_trail_mm:8.2f} mm")
    L.append(f"   Scrub radius (3D)              {sg.scrub_radius_mm:8.2f} mm")
    L.append(_band(
        "Mechanical trail", sg.mechanical_trail_mm, lim.mechanical_trail_mm,
    ))

    L.append("")
    L.append(" STEERING ARM AND TIE ROD")
    L.append(f"   Arm height along kingpin       {steer.tro_height_along_kingpin_mm:8.2f} mm")
    L.append(f"   Arm length (C-factor arm)      {steer.steer_arm_length_mm:8.2f} mm")
    L.append(f"   Arm angle                      {steer.steer_arm_angle_deg:8.2f} deg")
    L.append(f"   Tie rod length                 {sg.tie_rod_length_mm:8.2f} mm")
    L.append(_band(
        "Tie rod length", sg.tie_rod_length_mm, lim.tie_rod_length_mm,
    ))

    L.append("")
    L.append(" RACK")
    L.append(f"   Rack x (rearward)              {steer.rack_x_mm:8.2f} mm")
    L.append(f"   Rack z                         {steer.rack_z_mm:8.2f} mm")
    L.append(f"   Rack half length               {steer.rack_half_length_mm:8.2f} mm")
    L.append(f"   Pinion radius                  {steer.pinion_radius_mm:8.2f} mm")
    L.append(f"   Max rack travel (half-stroke)  {steer.max_rack_travel_mm:8.2f} mm")
    L.append(_band("Rack x position", steer.rack_x_mm, lim.rack_x_window_mm))
    L.append(_band("Rack z position", steer.rack_z_mm, lim.rack_z_window_mm))

    L.append("")
    L.append(" KINEMATIC RATES")
    L.append(f"   Bump steer (per-side)          "
             f"{rates.bump_steer_deg_per_mm_per_side:8.5f} deg/mm")
    L.append(f"   Bump steer (total toe)         "
             f"{rates.bump_steer_total_toe_deg_per_mm:8.5f} deg/mm")
    L.append(f"   Toe at full bump (per-side)    "
             f"{rates.toe_at_full_bump_deg_per_side:8.3f} deg")
    L.append(f"   Toe at full droop (per-side)   "
             f"{rates.toe_at_full_droop_deg_per_side:8.3f} deg")
    L.append(f"   C-factor                       "
             f"{rates.c_factor_mm_per_deg:8.3f} mm/deg")
    L.append(f"   Steering ratio                 "
             f"{rates.steering_ratio:8.2f} :1")
    L.append(f"   Max steer at stroke (outer)    "
             f"{rates.max_steer_at_stroke_deg:8.2f} deg")
    L.append(f"   Max steer at stroke (inner)    "
             f"{rates.max_steer_inner_at_stroke_deg:8.2f} deg")
    L.append(f"   Geometric Ackermann            "
             f"{sg.geometric_ackermann_pct:8.1f} %")
    L.append(f"   Ackermann at {steer.ackermann_at_steer_deg:.0f} deg        "
             f"{rates.ackermann_pct_at_target:8.1f} %")

    L.append("")
    L.append(" CHECKS")
    L.append(_band(
        "Bump steer (per-side)",
        abs(rates.bump_steer_deg_per_mm_per_side),
        lim.bump_steer_deg_per_mm,
        "deg/mm",
        "8.5f",
    ))
    L.append(_band(
        "Ackermann %", rates.ackermann_pct_at_target, lim.ackermann_pct, "%",
    ))
    L.append(_band(
        "Steering ratio", rates.steering_ratio, lim.steering_ratio, ":1",
    ))
    ok_rod = rates.worst_rod_end_misalignment_deg <= lim.rod_end_misalignment_deg
    L.append(f"  {_flag(ok_rod)}{'Rod end misalignment':<32s}"
             f"{rates.worst_rod_end_misalignment_deg:8.2f} deg     "
             f"limit {lim.rod_end_misalignment_deg:g}")

    L.append("")
    L.append(" STEERING EFFORT (parking)")
    L.append(f"   Fz per front wheel             {effort.Fz_per_wheel_N:8.1f} N")
    L.append(f"   Kingpin moment (per wheel)     {effort.kingpin_moment_Nm:8.2f} N.m")
    L.append(f"     scrub contribution           {effort.scrub_contrib_mm:8.2f} mm")
    L.append(f"     trail contribution           {effort.trail_contrib_mm:8.2f} mm")
    L.append(f"   Rack force (both wheels)       {effort.rack_force_N:8.1f} N")
    L.append(f"   Steering wheel torque          {effort.steering_wheel_torque_Nm:8.2f} N.m")
    L.append(f"   Rim force                      {effort.rim_force_N:8.1f} N")
    ok_eff = effort.steering_wheel_torque_Nm <= lim.steering_wheel_torque_Nm
    L.append(f"  {_flag(ok_eff)}{'Steering wheel torque':<32s}"
             f"{effort.steering_wheel_torque_Nm:8.2f} N.m    "
             f"limit {lim.steering_wheel_torque_Nm:g}")

    L.append("")
    L.append(" HARDPOINTS  (ISO 8855, caster-corrected UCA_OUT / LCA_OUT)")
    L.append(f" {hp.frame}")
    L.append(f"   {'corner':<6s} {'point':<16s}{'X [mm]':>11s}{'Y [mm]':>11s}{'Z [mm]':>11s}")
    for corner, name, x, y, z in hp.rows():
        L.append(f"   {corner:<6s} {name:<16s}{x:11.2f}{y:11.2f}{z:11.2f}")
    L.append(" * UCA_OUT and LCA_OUT supersede sla_geometry.py's zero-caster values")

    return "\n".join(L)


def notes_report(steer: SteeringInputs) -> str:
    L = ["", _RULE, " NOTES", _RULE]
    L.append(" Values tagged design_intent (chosen, not measured or computed):")
    L.append("   caster angle, caster offset, steering arm geometry, rack position,")
    L.append("   rack hardware, static toe, Ackermann target, bump steer target.")
    L.append("")
    L.append(" The 3D solver (vdcore.geometry.solver.DWSolver) is more accurate")
    L.append(" than sla_geometry.py's front-view four-bar. Camber-vs-bump will")
    L.append(" differ slightly between the two scripts — that is expected.")
    L.append("")
    L.append(" Rack axis is assumed lateral and horizontal (DWSolver translates")
    L.append(" the inner tie-rod joint purely along Y). This covers standard FSAE")
    L.append(" racks; an inclined rack would need solver modification.")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 10. PLOTTING
# --------------------------------------------------------------------------- #


def plot_steering(
    kin: SteeringKinematics,
    sg: SteeringGeometry,
    rates: SteeringRates,
    path: str = "steering.png",
) -> str:
    """Generate a 4-panel steering chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Steering Geometry — FSAE 2027", fontsize=13)

    # panel 1: toe vs bump (bump steer)
    bs = kin.bump_steer_curve()
    bumps = [b for b, _, _ in bs]
    toe_fl = [t for _, t, _ in bs]
    toe_fr = [t for _, _, t in bs]
    ax[0, 0].plot(toe_fl, bumps, label="FL toe")
    ax[0, 0].plot(toe_fr, bumps, label="FR toe")
    ax[0, 0].set(title="Bump steer", xlabel="toe [deg/side]", ylabel="bump [mm]")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)

    # panel 2: Ackermann
    ack = kin.ackermann_curve()
    outer = [o for o, _, _ in ack]
    inner_act = [i for _, i, _ in ack]
    inner_idl = [i for _, _, i in ack]
    ax[0, 1].plot(outer, inner_act, label="actual inner")
    ax[0, 1].plot(outer, inner_idl, "--", label="ideal inner (100%)")
    ax[0, 1].plot(outer, outer, ":", alpha=0.5, label="parallel")
    ax[0, 1].set(title="Ackermann", xlabel="outer steer [deg]",
                 ylabel="inner steer [deg]")
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3)

    # panel 3: inner/outer steer vs rack travel
    sc = kin.steer_curve()
    rack_mm_arr = [r for r, _, _, _ in sc]
    toe_fl_s = [t for _, t, _, _ in sc]
    toe_fr_s = [t for _, _, t, _ in sc]
    ax[1, 0].plot(rack_mm_arr, toe_fl_s, label="FL (inner @ +rack)")
    ax[1, 0].plot(rack_mm_arr, toe_fr_s, label="FR (outer @ +rack)")
    ax[1, 0].set(title="Steer vs rack travel", xlabel="rack [mm]",
                 ylabel="steer angle [deg]")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    # panel 4: camber vs steer
    camber_fl = [c for _, _, _, c in sc]
    ax[1, 1].plot(toe_fl_s, camber_fl, label="FL camber")
    ax[1, 1].set(title="Camber vs steer", xlabel="FL steer [deg]",
                 ylabel="camber [deg]")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 11. CONFIG
# --------------------------------------------------------------------------- #

# Edit this block to change the steering design inputs.
STEERING_2027 = SteeringInputs(
    caster_deg=5.0,
    caster_offset_mm=0.0,
    tro_height_along_kingpin_mm=40.0,
    steer_arm_length_mm=80.0,          # was 90.0 — shorter arm for ratio ~4.6:1
    steer_arm_angle_deg=-12.0,         # was 15.0 — rotated for ~101% geometric Ackermann
    rack_x_mm=30.0,
    rack_z_mm=158.3,                   # was 100.0 — solved for zero bump steer
    rack_half_length_mm=270.0,         # was 230.0 — wider rack shortens tie rod below 350 mm
    pinion_radius_mm=16.0,             # was 20.0
    max_rack_travel_mm=38.0,           # was 25.0
    steering_wheel_diameter_mm=280.0,  # was 260.0
    static_toe_deg_per_side=0.0,
    target_ackermann_pct=100.0,
    ackermann_at_steer_deg=10.0,
    target_bump_steer_deg_per_mm=0.0,
    mu_parking=1.0,
    steer_sweep_deg=30.0,
    n_sweep=21,
    hardpoint_tol_mm=1.0,
)


# --------------------------------------------------------------------------- #
# 12. TOP LEVEL
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SteeringReport:
    geometry: SteeringGeometry
    rates: SteeringRates
    effort: SteeringEffort
    hardpoints: SteeringHardpoints
    text: str


def run(
    veh: VehicleData = VEHICLE_2027,
    front_in: AxleInputs = FRONT_2027,
    steer: SteeringInputs = STEERING_2027,
    *,
    with_sweep: bool = True,
) -> SteeringReport:
    front_geo = solve_axle(front_in, veh)
    sg = synthesize_steering(front_geo, steer, veh)

    kin = SteeringKinematics(front_geo, sg, steer, veh)
    rates = kin.compute_rates()
    effort = rack_force_parking(sg, rates, steer, veh)
    hp = build_steering_hardpoints(sg)

    text_parts: list[str] = []
    text_parts.append(steering_report(sg, rates, effort, steer, hp))
    text_parts.append(notes_report(steer))

    return SteeringReport(
        geometry=sg,
        rates=rates,
        effort=effort,
        hardpoints=hp,
        text="\n".join(text_parts) + "\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Steering geometry synthesis — FSAE 2027.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Edit STEERING_2027 near the bottom of the file to change inputs.",
    )
    ap.add_argument("--json", metavar="PATH",
                    help="write hardpoints to JSON")
    ap.add_argument("--csv", metavar="PATH",
                    help="write hardpoints to CSV")
    ap.add_argument("--plot", nargs="?", const="steering.png", metavar="PATH",
                    help="generate the 4-panel chart (default: steering.png)")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the text report")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the kinematic sweeps")
    ap.add_argument("--solve-bump-steer", action="store_true",
                    help="back-solve rack_half_length for target bump steer")
    ap.add_argument("--solve-ackermann", action="store_true",
                    help="back-solve steer_arm_angle for target Ackermann")
    args = ap.parse_args(argv)

    rep = run(with_sweep=not args.no_sweep)

    if not args.quiet:
        print(rep.text)

    if args.solve_bump_steer:
        front_geo = solve_axle(FRONT_2027, VEHICLE_2027)
        val = y_tri_for_zero_bump_steer(front_geo, STEERING_2027, VEHICLE_2027)
        print(f"\n  Back-solver: rack_half_length_mm for target bump steer "
              f"= {STEERING_2027.target_bump_steer_deg_per_mm} deg/mm:")
        if np.isnan(val):
            print("    No solution found in the search range.")
        else:
            print(f"    rack_half_length_mm = {val:.2f}")

    if args.solve_ackermann:
        front_geo = solve_axle(FRONT_2027, VEHICLE_2027)
        val = arm_angle_for_ackermann_target(front_geo, STEERING_2027, VEHICLE_2027)
        print(f"\n  Back-solver: steer_arm_angle_deg for target Ackermann "
              f"= {STEERING_2027.target_ackermann_pct:.0f} %:")
        if np.isnan(val):
            print("    No solution found in the search range.")
        else:
            print(f"    steer_arm_angle_deg = {val:.2f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep.hardpoints.to_dict(), fh, indent=2)
        print(f"hardpoints written to {args.json}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["corner", "point", "x_mm", "y_mm", "z_mm"])
            for corner, name, x, y, z in rep.hardpoints.rows():
                w.writerow([corner, name, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])
        print(f"hardpoints written to {args.csv}")

    if args.plot:
        front_geo = solve_axle(FRONT_2027, VEHICLE_2027)
        sg = synthesize_steering(front_geo, STEERING_2027, VEHICLE_2027)
        kin = SteeringKinematics(front_geo, sg, STEERING_2027, VEHICLE_2027)
        path = plot_steering(kin, sg, rep.rates, args.plot)
        print(f"chart saved to {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
