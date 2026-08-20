#!/usr/bin/env python3
"""
sla_geometry_complete.py
========================

Double A-arm (SLA) suspension hardpoint synthesis for the PUCPR Racing FSAE 2027
car -- front and rear axle, all four corners.

WHAT IT DOES
------------
You supply a small set of *design intent* numbers (track, tyre radius, roll
centre height, front-view swing-arm length, lower ball joint position, KPI,
kingpin length, chassis pickup plane, longitudinal base/sweep).  The script
returns the complete A-arm hardpoint set of every corner:

    UCA_IN_FRONT   UCA_IN_REAR   UCA_OUT        (upper wishbone: 2 in + 1 out)
    LCA_IN_FRONT   LCA_IN_REAR   LCA_OUT        (lower wishbone: 2 in + 1 out)
    WHEEL_CENTRE   CONTACT_PATCH                (reference points)

plus every static KPI, verification and rate table found in
"Hardpoints Suspensao 2027.pdf".  Running the file with no arguments prints the
full report and reproduces that document.

Stdlib only -- no numpy, no scipy.  Python 3.10+.


THE SYNTHESIS CHAIN (this is the part that was lost)
----------------------------------------------------
Everything follows from the front view.  The *outboard* points are chosen by
package (rim, upright, steering); the *inboard* chassis pickups are DERIVED:

  1. FVIC (front view instant centre) is placed by two targets:
         y_ic = half_track - fvsa_length
         z_ic = rc_height * fvsa_length / half_track
     i.e. the FVIC is the point that puts the roll centre at `rc_height` on the
     car centreline, on the line drawn from the tyre contact patch.

  2. LBJ (lower ball joint) is an input: y from package/scrub, z from rim clearance.

  3. UBJ (upper ball joint) sits on the kingpin axis, `kingpin_length` above the
     LBJ, leaning inboard by KPI:
         UBJ = LBJ + kingpin_length * (-sin(KPI), +cos(KPI))

  4. The two wishbone axes are the lines BALL-JOINT -> FVIC.  The inboard pickups
     are simply where those lines cross the chassis pickup plane y = inner_y.
     That is what makes both arms point at the same FVIC by construction.

  5. Longitudinally each wishbone is defined by a `base` (distance between its two
     inboard pickups) and a `sweep` (how far the outboard ball joint sits behind
     the midpoint of the pivot axis).  Both pickups of a wishbone share one z, so
     both pivot axes are parallel to X whatever the sweep is.

LIMITATION -- ANTI-GEOMETRY IS STRUCTURALLY ZERO
------------------------------------------------
Because both inboard pickups of a wishbone share a single z, every pivot axis is
horizontal, both side-view axis projections are parallel, the SVIC is always at
infinity and anti-dive / anti-squat therefore always report 0.000 %.  That is the
right answer for the 2027 geometry (the PDF says 0 % too) but it is NOT a
computed result -- the code cannot currently produce any other number, and the
non-zero branch of `_side_view_anti` is unreachable.

To make anti-geometry a live design variable you need one more input per
wishbone -- the pivot-axis rise (rear pickup z minus front pickup z) -- fed into
both `_side_view_anti` and `build_corner`.  Note that inclining the axes also
makes the front-view construction above an approximation, since the wishbone no
longer swings in a plane perpendicular to X.


COORDINATE SYSTEMS  (two are used -- do not mix them)
-----------------------------------------------------
* DESIGN frame (sections 2 and 3 of the PDF, and everything inside this module):
      y positive OUTBOARD, z positive UP, origin at ground / car centreline,
      x positive REARWARD from the front axle.
      This is a per-corner working frame: it is mirror-symmetric, so one set of
      numbers describes both the left and the right corner.

* MODEL frame = ISO 8855 (section 4 of the PDF, and the CLAUDE.md convention):
      X positive FORWARD, Y positive LEFT, Z positive UP,
      origin at the front axle centreline, on the ground, on the car centreline.
      Left corners (FL, RL) have Y > 0; right corners (FR, RR) have Y < 0.
      Conversion:  X_iso = -x_rearward,  Y_iso = +/- y_outboard,  Z_iso = z.

Units: mm and degrees throughout the I/O.  Radians only inside the solvers.
Camber sign: negative = top of the wheel inboard (both sides).
Scrub sign:  positive = kingpin axis meets the ground INBOARD of the contact patch.


NOTE ON THE RATE TABLE (read this before trusting the sweep numbers)
--------------------------------------------------------------------
The original script transformed the upright-attached points (wheel centre,
contact patch) with a rotation of the WRONG SIGN: it applied rot(d_psi) where
d_psi is measured from +Z toward +Y, so the correct transform is rot(-d_psi).
Static geometry is unaffected (it never uses the transform), so ALL hardpoints
and ALL static KPIs are correct.  The *rates* are not:

    front camber gain   legacy 0.0402 deg/mm   corrected 0.0382 deg/mm
    front half-track    legacy -0.1231 mm/mm   corrected +0.0566 mm/mm  (sign!)
    rear  camber gain   legacy 0.0435 deg/mm   corrected 0.0409 deg/mm

The corrected front camber gain is exactly 57.2958 / FVSA = 57.2958 / 1500, as
it must be for a wheel rotating about an instant centre 1500 mm away.  The
legacy value divides by 1424 mm instead, because the buggy rotation moves the
contact patch the wrong way and shortens the apparent travel.

Both are reported side by side.  `--corrected` makes the corrected column the
one used for the pass/fail checks.  The default reproduces the PDF.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from typing import Iterator, Literal, Sequence

# --------------------------------------------------------------------------- #
# Types and small geometric helpers
# --------------------------------------------------------------------------- #

Point2D = tuple[float, float]            # (y, z) in the design front view
Point3D = tuple[float, float, float]     # (X, Y, Z) in the ISO 8855 model frame
Band = tuple[float, float]               # inclusive (low, high) acceptance band

GRAVITY = 9.81                           # m/s^2
D2R = math.pi / 180.0
R2D = 180.0 / math.pi


class KinematicError(RuntimeError):
    """The mechanism cannot be assembled in the requested state.

    Raised instead of returning a plausible-looking number: a four-bar that does
    not close has no solution, and silently clamping it would poison every KPI
    downstream.
    """


def line_intersection(p1: Point2D, p2: Point2D,
                      p3: Point2D, p4: Point2D) -> Point2D | None:
    """Intersection of the infinite lines (p1,p2) and (p3,p4). None if parallel."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = p1, p2, p3, p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return ((a * (x3 - x4) - (x1 - x2) * b) / den,
            (a * (y3 - y4) - (y1 - y2) * b) / den)


def circle_intersection(c1: Point2D, r1: float, c2: Point2D, r2: float,
                        reference: Point2D) -> Point2D:
    """Intersect two circles; return the branch closest to `reference`.

    `reference` is the previous/static position of the joint, which is what keeps
    the four-bar on the assembly branch it was designed on instead of flipping.
    """
    (x1, z1), (x2, z2) = c1, c2
    dx, dz = x2 - x1, z2 - z1
    d = math.hypot(dx, dz)
    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-12:
        raise KinematicError(
            f"four-bar does not close: centre distance {d:.3f} mm, radii "
            f"{r1:.3f} / {r2:.3f} mm -- requested travel exceeds the kinematic limit"
        )
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h = math.sqrt(max(r1 * r1 - a * a, 0.0))
    bx, bz = x1 + a * dx / d, z1 + a * dz / d
    px, pz = -dz / d, dx / d
    s1 = (bx + h * px, bz + h * pz)
    s2 = (bx - h * px, bz - h * pz)
    return s1 if math.dist(s1, reference) < math.dist(s2, reference) else s2


def rotate(p: Point2D, angle_rad: float) -> Point2D:
    """Rotate a 2D vector counter-clockwise in the (y, z) plane."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return (c * p[0] - s * p[1], s * p[0] + c * p[1])


def nz(v: float) -> float:
    """Collapse negative zero, so a point on the front axle prints as 0.00."""
    return v + 0.0


def bisect(f, lo: float, hi: float, tol: float = 1e-12, max_iter: int = 200) -> float:
    """Plain bisection. Explicit failure if the bracket does not contain a root."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0.0:
        raise KinematicError(
            f"root not bracketed on [{lo:.6f}, {hi:.6f}] (f = {f_lo:.6g}, {f_hi:.6g})"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if hi - lo < tol:
            return mid
        if f_lo * f(mid) <= 0.0:
            hi = mid
        else:
            lo, f_lo = mid, f(mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# 1. VEHICLE AND STIFFNESS                                     (PDF section 1)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, kw_only=True)
class VehicleData:
    """Mass properties and stiffness targets. All of these are INPUTS."""

    name: str = "FSAE 2027 -- PUCPR Racing"
    total_mass_kg: float = 315.0            # car + driver
    unsprung_mass_kg: float = 45.0          # sum of the four corners
    cg_height_mm: float = 320.0             # with driver
    wheelbase_mm: float = 1540.0

    # CG longitudinal station, measured rearward from the FRONT axle.
    # 693 mm = 45 % of the wheelbase => 55 % static load on the front axle.
    cg_from_front_axle_mm: float = 693.0

    target_roll_gradient_deg_per_g: float = 1.00

    # Chassis torsional stiffness is specified as a multiple of the roll
    # stiffness it has to react. 3x is the usual "do not embarrass yourself"
    # floor, 5x the design target.
    chassis_factor_min: float = 3.0
    chassis_factor_target: float = 5.0

    tilt_test_angle_deg: float = 60.0       # FSAE rules tilt table angle

    # Only used by the anti-dive calculation. Not recoverable from the PDF, so it
    # is tagged as an assumption in the report.
    brake_bias_front: float = 0.65

    @property
    def sprung_mass_kg(self) -> float:
        return self.total_mass_kg - self.unsprung_mass_kg

    @property
    def front_mass_fraction(self) -> float:
        return 1.0 - self.cg_from_front_axle_mm / self.wheelbase_mm


@dataclass(frozen=True)
class VehicleResults:
    """Derived roll / stiffness / tilt numbers (PDF section 1 table)."""

    roll_axis_height_at_cg_mm: float
    roll_moment_arm_mm: float
    sprung_weight_N: float
    required_roll_stiffness_Nm_per_deg: float
    chassis_torsion_min_Nm_per_deg: float
    chassis_torsion_target_Nm_per_deg: float
    tilt_min_track_mm: float
    narrowest_track_mm: float

    @property
    def tilt_ok(self) -> bool:
        return self.narrowest_track_mm >= self.tilt_min_track_mm


def solve_vehicle(veh: VehicleData,
                  front_rc_mm: float, rear_rc_mm: float,
                  narrowest_track_mm: float) -> VehicleResults:
    """Roll stiffness required to hit the roll gradient target, plus tilt check.

    The roll axis is the straight line joining the front and rear roll centres;
    its height at the CG station sets the roll moment arm.
    """
    frac = veh.cg_from_front_axle_mm / veh.wheelbase_mm
    h_axis = front_rc_mm + (rear_rc_mm - front_rc_mm) * frac
    arm_mm = veh.cg_height_mm - h_axis

    w_sprung = veh.sprung_mass_kg * GRAVITY                      # N
    k_roll = w_sprung * (arm_mm / 1000.0) / veh.target_roll_gradient_deg_per_g

    tilt_min = 2.0 * veh.cg_height_mm * math.tan(veh.tilt_test_angle_deg * D2R)

    return VehicleResults(
        roll_axis_height_at_cg_mm=h_axis,
        roll_moment_arm_mm=arm_mm,
        sprung_weight_N=w_sprung,
        required_roll_stiffness_Nm_per_deg=k_roll,
        chassis_torsion_min_Nm_per_deg=veh.chassis_factor_min * k_roll,
        chassis_torsion_target_Nm_per_deg=veh.chassis_factor_target * k_roll,
        tilt_min_track_mm=tilt_min,
        narrowest_track_mm=narrowest_track_mm,
    )


# --------------------------------------------------------------------------- #
# 2. AXLE INPUTS                                          (PDF sections 2 / 3)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, kw_only=True)
class CheckLimits:
    """Acceptance bands for the verification table. Front and rear differ in KPI."""

    rc_height_mm: Band = (20.0, 70.0)
    fvsa_length_mm: Band = (1300.0, 1700.0)
    scrub_radius_mm: Band = (5.0, 25.0)
    kpi_deg: Band = (6.0, 14.0)
    kingpin_length_mm: Band = (200.0, 260.0)
    lca_length_mm: Band = (320.0, 430.0)
    uca_lca_ratio: Band = (0.55, 0.98)
    camber_gain_deg_per_mm: Band = (0.030, 0.050)
    ea_ratio_max: float = 1.5
    anti_percent: Band = (0.0, 30.0)
    outer_camber_in_roll_deg: Band = (-2.5, 0.0)
    ball_joint_clearance_mm: float = 15.0   # margin the joints must keep inside the rim


@dataclass(frozen=True, kw_only=True)
class AxleInputs:
    """Everything the designer chooses for one axle. Nothing here is derived."""

    name: str

    # --- wheel and tyre package -------------------------------------------- #
    track_mm: float
    loaded_radius_mm: float = 245.0         # = wheel centre height at static
    rim_diameter_in: float = 13.0
    static_camber_deg: float = -1.50

    # --- outboard: lower ball joint and kingpin axis ------------------------ #
    lbj_y_mm: float                         # outboard, from the car centreline
    lbj_z_mm: float = 130.0                 # above ground
    kpi_deg: float                          # kingpin inclination, top leaning inboard
    kingpin_length_mm: float                # LBJ -> UBJ distance along the kingpin axis
    #  NOTE: the PDF calls this row "comprimento de manga de eixo" (spindle
    #  length). It is numerically the LBJ->UBJ distance, not the wheel-centre
    #  offset from the steering axis. Kept under the honest name here.

    # --- inboard: chassis pickup plane -------------------------------------- #
    inner_pickup_y_mm: float = 175.0        # half width of the chassis rail

    # --- front-view targets that place the FVIC ----------------------------- #
    rc_height_mm: float                     # design roll centre height
    fvsa_length_mm: float                   # contact patch -> FVIC, horizontal

    # --- side view / longitudinal layout ------------------------------------ #
    axle_x_mm: float                        # axle station, positive REARWARD
    lca_base_mm: float                      # distance between the two LCA pickups
    lca_sweep_mm: float                     # ball joint behind the pivot-axis midpoint
    uca_base_mm: float
    uca_sweep_mm: float

    # --- sweep settings ------------------------------------------------------ #
    travel_bump_mm: float = 25.0
    travel_droop_mm: float = 25.0
    roll_reference_deg: float = 1.5

    limits: CheckLimits = field(default_factory=CheckLimits)

    @property
    def half_track_mm(self) -> float:
        return self.track_mm / 2.0

    @property
    def contact_patch(self) -> Point2D:
        return (self.half_track_mm, 0.0)

    @property
    def wheel_centre(self) -> Point2D:
        return (self.half_track_mm, self.loaded_radius_mm)

    @property
    def rim_z_band(self) -> Band:
        """Vertical window the ball joints must live inside to clear the rim."""
        r = self.rim_diameter_in * 25.4 / 2.0
        return (self.loaded_radius_mm - r, self.loaded_radius_mm + r)


# --------------------------------------------------------------------------- #
# 3. STATIC SYNTHESIS -- the four wishbone hardpoints in the front view
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AxleGeometry:
    """Solved static geometry of one axle, in the design front view."""

    inputs: AxleInputs

    # front-view points, (y outboard, z up)
    lbj: Point2D
    ubj: Point2D
    lca_in: Point2D
    uca_in: Point2D
    fvic: Point2D

    # wishbone metrics
    lca_length_mm: float
    uca_length_mm: float
    lca_inclination_deg: float              # positive = falls from wheel to chassis
    uca_inclination_deg: float
    outer_vertical_sep_mm: float
    inner_vertical_sep_mm: float

    # steering-axis metrics
    scrub_radius_mm: float
    rc_height_for_flat_lca_mm: float

    # side view
    lca_in_front_x_mm: float                # all four x are positive REARWARD
    lca_in_rear_x_mm: float
    uca_in_front_x_mm: float
    uca_in_rear_x_mm: float
    lca_ea_ratio: float
    uca_ea_ratio: float
    svic: Point2D | None                    # (x rearward, z), None = at infinity
    anti_percent: float
    anti_label: str                         # "Anti-dive" (front) or "Anti-squat" (rear)

    @property
    def uca_lca_ratio(self) -> float:
        return self.uca_length_mm / self.lca_length_mm


def solve_axle(inp: AxleInputs, veh: VehicleData) -> AxleGeometry:
    """Turn the design targets into hardpoints. This is the core of the script."""

    # -- 1. front view instant centre, from roll centre height + FVSA length --
    #    The FVIC is the point that makes the line contact_patch -> FVIC cross the
    #    car centreline at exactly rc_height.
    ic_y = inp.half_track_mm - inp.fvsa_length_mm
    ic_z = inp.rc_height_mm * inp.fvsa_length_mm / inp.half_track_mm
    fvic = (ic_y, ic_z)

    # -- 2 & 3. outboard ball joints ----------------------------------------
    lbj = (inp.lbj_y_mm, inp.lbj_z_mm)
    kpi = inp.kpi_deg * D2R
    ubj = (lbj[0] - inp.kingpin_length_mm * math.sin(kpi),
           lbj[1] + inp.kingpin_length_mm * math.cos(kpi))

    # -- 4. inboard pickups: where each wishbone axis crosses y = inner_y -----
    def on_line_at_y(outer: Point2D, ic: Point2D, y: float) -> Point2D:
        t = (y - outer[0]) / (ic[0] - outer[0])
        return (y, outer[1] + t * (ic[1] - outer[1]))

    lca_in = on_line_at_y(lbj, fvic, inp.inner_pickup_y_mm)
    uca_in = on_line_at_y(ubj, fvic, inp.inner_pickup_y_mm)

    lca_len = math.dist(lbj, lca_in)
    uca_len = math.dist(ubj, uca_in)

    # -- steering axis at ground level ---------------------------------------
    #    positive scrub = axis lands INBOARD of the contact patch
    y_axis_at_ground = lbj[0] + lbj[1] * math.tan(kpi)
    scrub = inp.half_track_mm - y_axis_at_ground

    # -- 5. longitudinal layout ----------------------------------------------
    def pickups_x(base: float, sweep: float) -> tuple[float, float]:
        mid = inp.axle_x_mm - sweep
        return (mid - base / 2.0, mid + base / 2.0)   # (front, rear), rearward +

    lca_xf, lca_xr = pickups_x(inp.lca_base_mm, inp.lca_sweep_mm)
    uca_xf, uca_xr = pickups_x(inp.uca_base_mm, inp.uca_sweep_mm)

    # e/a ratio: pivot-axis offset relative to half the base. Large values mean the
    # ball joint hangs a long way off the middle of its pivot axis, which loads the
    # rear leg in bending and is the usual reason a wishbone bends in a crash test.
    lca_ea = 2.0 * abs(inp.lca_sweep_mm) / inp.lca_base_mm
    uca_ea = 2.0 * abs(inp.uca_sweep_mm) / inp.uca_base_mm

    is_front = inp.axle_x_mm < veh.wheelbase_mm / 2.0
    svic, anti = _side_view_anti(inp, veh, is_front, lca_in[1], uca_in[1],
                                 (lca_xf, lca_xr), (uca_xf, uca_xr))

    return AxleGeometry(
        inputs=inp,
        lbj=lbj, ubj=ubj, lca_in=lca_in, uca_in=uca_in, fvic=fvic,
        lca_length_mm=lca_len,
        uca_length_mm=uca_len,
        lca_inclination_deg=math.degrees(
            math.atan2(lbj[1] - lca_in[1], lbj[0] - lca_in[0])),
        uca_inclination_deg=math.degrees(
            math.atan2(ubj[1] - uca_in[1], ubj[0] - uca_in[0])),
        outer_vertical_sep_mm=ubj[1] - lbj[1],
        inner_vertical_sep_mm=uca_in[1] - lca_in[1],
        scrub_radius_mm=scrub,
        rc_height_for_flat_lca_mm=inp.lbj_z_mm * inp.half_track_mm / inp.fvsa_length_mm,
        lca_in_front_x_mm=lca_xf, lca_in_rear_x_mm=lca_xr,
        uca_in_front_x_mm=uca_xf, uca_in_rear_x_mm=uca_xr,
        lca_ea_ratio=lca_ea, uca_ea_ratio=uca_ea,
        svic=svic, anti_percent=anti,
        anti_label="Anti-dive" if is_front else "Anti-squat",
    )


def _side_view_anti(inp: AxleInputs, veh: VehicleData, is_front: bool,
                    lca_in_z: float, uca_in_z: float,
                    lca_x: tuple[float, float],
                    uca_x: tuple[float, float]) -> tuple[Point2D | None, float]:
    """Side view instant centre and anti-dive / anti-squat percentage.

    Construction: project both wishbone pivot axes into the XZ plane; where those
    two lines meet is the SVIC.

    WARNING: with the current `AxleInputs` both pickups of a wishbone share one z,
    so each projected axis is horizontal, the two lines are always parallel, and
    this always returns (None, 0.0).  The branch below is kept because it is the
    correct construction, but it is unreachable until the input model gains a
    pivot-axis rise -- see the LIMITATION section of the module docstring.  Do not
    read the 0 % in the report as a computed result.
    """
    lca_line = ((lca_x[0], lca_in_z), (lca_x[1], lca_in_z))
    uca_line = ((uca_x[0], uca_in_z), (uca_x[1], uca_in_z))
    svic = line_intersection(lca_line[0], lca_line[1], uca_line[0], uca_line[1])

    if svic is None:
        return None, 0.0                    # parallel axes -> SVIC at infinity

    # angle of the line contact patch -> SVIC, seen from the side
    dx = svic[0] - inp.axle_x_mm
    if abs(dx) < 1e-9:
        return svic, 0.0
    tan_theta = svic[1] / dx

    h_over_l = veh.cg_height_mm / veh.wheelbase_mm
    if is_front:                                        # anti-dive
        return svic, 100.0 * tan_theta / h_over_l * veh.brake_bias_front
    # rear axle -> anti-squat (all drive torque reacted by this axle)
    return svic, 100.0 * (-tan_theta) / h_over_l


# --------------------------------------------------------------------------- #
# 4. KINEMATICS -- front view four-bar sweep
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CornerState:
    """One solved position of the corner, in the design front view."""

    lca_angle_rad: float
    lbj: Point2D
    ubj: Point2D
    contact_patch: Point2D
    wheel_centre: Point2D
    ic: Point2D
    camber_deg: float
    bump_mm: float                          # vertical travel of the contact patch
    half_track_mm: float
    rc_height_mm: float                     # measured in the CHASSIS frame, as the PDF does
    fvsa_length_mm: float


class AxleKinematics:
    """Front-view four-bar: chassis -- LCA -- upright -- UCA.

    The two inboard pickups are fixed; the LCA angle drives the mechanism; the UBJ
    comes from a circle/circle intersection; the upright is then a rigid body and
    carries the wheel centre and contact patch with it.

    `legacy_rotation_sign=True` reproduces the sign error present in the script
    that generated the 2027 PDF -- see the module docstring.  It changes the rates
    only; every static number is identical either way.
    """

    def __init__(self, geo: AxleGeometry, *, legacy_rotation_sign: bool = True):
        self.geo = geo
        self.inp = geo.inputs
        self.legacy = legacy_rotation_sign

        self.lca_in, self.uca_in = geo.lca_in, geo.uca_in
        self.l_lca, self.l_uca = geo.lca_length_mm, geo.uca_length_mm
        self.l_kingpin = self.inp.kingpin_length_mm

        self.lbj0, self.ubj0 = geo.lbj, geo.ubj
        self.theta0 = math.atan2(self.lbj0[1] - self.lca_in[1],
                                 self.lbj0[0] - self.lca_in[0])
        # psi is measured from +z toward +y, so a counter-clockwise rotation of the
        # upright DECREASES psi. Camber = static + d_psi follows directly.
        self.psi0 = math.atan2(self.ubj0[0] - self.lbj0[0],
                               self.ubj0[1] - self.lbj0[1])

        cp0, wc0 = self.inp.contact_patch, self.inp.wheel_centre
        self.r_cp = (cp0[0] - self.lbj0[0], cp0[1] - self.lbj0[1])
        self.r_wc = (wc0[0] - self.lbj0[0], wc0[1] - self.lbj0[1])

    # ------------------------------------------------------------------ #
    def state_at_angle(self, theta: float) -> CornerState:
        lbj = (self.lca_in[0] + self.l_lca * math.cos(theta),
               self.lca_in[1] + self.l_lca * math.sin(theta))
        ubj = circle_intersection(self.uca_in, self.l_uca,
                                  lbj, self.l_kingpin, reference=self.ubj0)

        psi = math.atan2(ubj[0] - lbj[0], ubj[1] - lbj[1])
        d_psi = psi - self.psi0
        rot = d_psi if self.legacy else -d_psi

        def carry(r: Point2D) -> Point2D:
            v = rotate(r, rot)
            return (lbj[0] + v[0], lbj[1] + v[1])

        cp, wc = carry(self.r_cp), carry(self.r_wc)

        ic = line_intersection(self.lca_in, lbj, self.uca_in, ubj)
        if ic is None:                       # parallel arms: IC at infinity
            ic = (math.copysign(1e7, self.geo.fvic[0]), lbj[1])

        rc = line_intersection(cp, ic, (0.0, 0.0), (0.0, 1.0))
        rc_z = rc[1] if rc is not None else math.nan

        return CornerState(
            lca_angle_rad=theta, lbj=lbj, ubj=ubj,
            contact_patch=cp, wheel_centre=wc, ic=ic,
            camber_deg=self.inp.static_camber_deg + math.degrees(d_psi),
            bump_mm=cp[1] - self.inp.contact_patch[1],
            half_track_mm=cp[0],
            rc_height_mm=rc_z,
            fvsa_length_mm=cp[0] - ic[0],
        )

    def state_at_bump(self, bump_mm: float) -> CornerState:
        """Solve for the LCA angle that puts the contact patch at `bump_mm`."""

        def residual(theta: float) -> float:
            return self.state_at_angle(theta).bump_mm - bump_mm

        span = 0.75                          # rad, shrunk until the four-bar closes
        for _ in range(80):
            lo, hi = self.theta0 - span, self.theta0 + span
            try:
                return self.state_at_angle(bisect(residual, lo, hi))
            except KinematicError:
                span *= 0.9
        raise KinematicError(
            f"{self.inp.name}: cannot reach {bump_mm:+.2f} mm of travel"
        )

    def sweep(self, n: int = 41) -> list[CornerState]:
        lo, hi = -self.inp.travel_droop_mm, self.inp.travel_bump_mm
        return [self.state_at_bump(lo + (hi - lo) * i / (n - 1)) for i in range(n)]


@dataclass(frozen=True)
class SweepRates:
    """Rates about the static position, plus the travel extremes."""

    camber_gain_deg_per_mm: float           # d(camber)/d(bump), negative in bump
    rc_migration_mm_per_mm: float
    half_track_change_mm_per_mm: float
    camber_full_bump_deg: float
    camber_full_droop_deg: float
    rc_min_mm: float
    rc_max_mm: float


def compute_rates(kin: AxleKinematics) -> SweepRates:
    """Central difference at static (small step) + the two travel extremes."""
    step = kin.inp.travel_bump_mm / 20.0     # matches the 41-point sweep of the original
    up, dn = kin.state_at_bump(+step), kin.state_at_bump(-step)
    two_h = 2.0 * step

    full_bump = kin.state_at_bump(kin.inp.travel_bump_mm)
    full_droop = kin.state_at_bump(-kin.inp.travel_droop_mm)
    rc_all = [s.rc_height_mm for s in kin.sweep()]

    return SweepRates(
        camber_gain_deg_per_mm=(up.camber_deg - dn.camber_deg) / two_h,
        rc_migration_mm_per_mm=(up.rc_height_mm - dn.rc_height_mm) / two_h,
        half_track_change_mm_per_mm=(up.half_track_mm - dn.half_track_mm) / two_h,
        camber_full_bump_deg=full_bump.camber_deg,
        camber_full_droop_deg=full_droop.camber_deg,
        rc_min_mm=min(rc_all), rc_max_mm=max(rc_all),
    )


@dataclass(frozen=True)
class RollState:
    """Both wheels on the ground, chassis rolled -- everything relative to the ROAD."""

    roll_deg: float
    wheel_travel_mm: float
    outer_camber_deg: float
    inner_camber_deg: float
    rc_height_mm: float
    rc_lateral_mm: float


def compute_roll(kin: AxleKinematics, roll_deg: float) -> RollState:
    """Roll the chassis and keep both contact patches on one plane.

    The travel is solved (not assumed to be half_track * tan(roll)) so that both
    contact patches end up at the same height in the road frame.  Roll centre is
    then the intersection of the two contact-patch -> IC lines, expressed with the
    ground plane as datum.
    """
    phi = roll_deg * D2R
    mirror = lambda p: (-p[0], p[1])         # design frame -> opposite side

    def patch_mismatch(b: float) -> float:
        outer, inner = kin.state_at_bump(+b), kin.state_at_bump(-b)
        cp_o = rotate(outer.contact_patch, -phi)
        cp_i = rotate(mirror(inner.contact_patch), -phi)
        return cp_o[1] - cp_i[1]

    guess = kin.inp.half_track_mm * math.tan(phi)
    travel = 0.0 if abs(roll_deg) < 1e-12 else bisect(
        patch_mismatch, 0.2 * guess, 2.5 * guess, tol=1e-10)

    outer, inner = kin.state_at_bump(+travel), kin.state_at_bump(-travel)

    # chassis frame -> road frame, then drop the datum onto the ground plane
    cp_o, ic_o = rotate(outer.contact_patch, -phi), rotate(outer.ic, -phi)
    cp_i = rotate(mirror(inner.contact_patch), -phi)
    ic_i = rotate(mirror(inner.ic), -phi)
    dz = -cp_o[1]
    cp_o, ic_o, cp_i, ic_i = [(p[0], p[1] + dz) for p in (cp_o, ic_o, cp_i, ic_i)]

    rc = line_intersection(cp_o, ic_o, cp_i, ic_i)
    rc_y, rc_z = (rc if rc is not None else (math.nan, math.nan))

    # Camber relative to the road: the body leans by phi, so the outer wheel gains
    # +phi against the track and the inner wheel loses phi.
    return RollState(
        roll_deg=roll_deg,
        wheel_travel_mm=travel,
        outer_camber_deg=outer.camber_deg + roll_deg,
        inner_camber_deg=inner.camber_deg - roll_deg,
        rc_height_mm=rc_z,
        rc_lateral_mm=rc_y,
    )


# --------------------------------------------------------------------------- #
# 5. 3D MODEL HARDPOINTS                                       (PDF section 4)
# --------------------------------------------------------------------------- #

CornerId = Literal["FL", "FR", "RL", "RR"]

#: The six wishbone hardpoints, in the order they are reported.
ARM_POINT_NAMES: tuple[str, ...] = (
    "UCA_IN_FRONT", "UCA_IN_REAR", "UCA_OUT",
    "LCA_IN_FRONT", "LCA_IN_REAR", "LCA_OUT",
)


@dataclass(frozen=True)
class CornerHardpoints:
    """One corner in the ISO 8855 model frame: X forward, Y left, Z up."""

    corner: CornerId
    axle: str
    side: Literal["left", "right"]

    uca_in_front: Point3D
    uca_in_rear: Point3D
    uca_out: Point3D                        # = UBJ
    lca_in_front: Point3D
    lca_in_rear: Point3D
    lca_out: Point3D                        # = LBJ
    wheel_centre: Point3D
    contact_patch: Point3D

    def arm_points(self) -> Iterator[tuple[str, Point3D]]:
        """The six wishbone points -- the deliverable of this script."""
        yield from zip(ARM_POINT_NAMES,
                       (self.uca_in_front, self.uca_in_rear, self.uca_out,
                        self.lca_in_front, self.lca_in_rear, self.lca_out))

    def all_points(self) -> Iterator[tuple[str, Point3D]]:
        yield from self.arm_points()
        yield "WHEEL_CENTER", self.wheel_centre
        yield "CONTACT_PATCH", self.contact_patch


def build_corner(geo: AxleGeometry, side: Literal["left", "right"],
                 corner: CornerId) -> CornerHardpoints:
    """Lift the front-view solution into 3D and convert to ISO 8855.

        X_iso = -x_rearward        (design x grows rearward, ISO X grows forward)
        Y_iso = +y_outboard on the left,  -y_outboard on the right
        Z_iso =  z                 (unchanged, ground datum)
    """
    inp = geo.inputs
    sy = 1.0 if side == "left" else -1.0

    def p(x_rearward: float, y_outboard: float, z: float) -> Point3D:
        return (nz(-x_rearward), nz(sy * y_outboard), nz(z))

    x_out = inp.axle_x_mm
    return CornerHardpoints(
        corner=corner, axle=inp.name, side=side,
        uca_in_front=p(geo.uca_in_front_x_mm, geo.uca_in[0], geo.uca_in[1]),
        uca_in_rear=p(geo.uca_in_rear_x_mm, geo.uca_in[0], geo.uca_in[1]),
        uca_out=p(x_out, geo.ubj[0], geo.ubj[1]),
        lca_in_front=p(geo.lca_in_front_x_mm, geo.lca_in[0], geo.lca_in[1]),
        lca_in_rear=p(geo.lca_in_rear_x_mm, geo.lca_in[0], geo.lca_in[1]),
        lca_out=p(x_out, geo.lbj[0], geo.lbj[1]),
        wheel_centre=p(x_out, inp.half_track_mm, inp.loaded_radius_mm),
        contact_patch=p(x_out, inp.half_track_mm, 0.0),
    )


@dataclass(frozen=True)
class ModelHardpoints:
    """The combined four-corner hardpoint set, ISO 8855."""

    frame: str = "ISO 8855: X+ forward, Y+ left, Z+ up; origin front axle / ground / centreline"
    corners: tuple[CornerHardpoints, ...] = ()

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


def build_model(front: AxleGeometry, rear: AxleGeometry) -> ModelHardpoints:
    return ModelHardpoints(corners=(
        build_corner(front, "left", "FL"),
        build_corner(front, "right", "FR"),
        build_corner(rear, "left", "RL"),
        build_corner(rear, "right", "RR"),
    ))


# --------------------------------------------------------------------------- #
# 6. REPORT
# --------------------------------------------------------------------------- #

W = 78
_RULE = "=" * W


def _flag(ok: bool) -> str:
    return "OK  " if ok else "!!  "


def _band(label: str, value: float, band: Band, unit: str = "mm",
          fmt: str = "8.2f") -> str:
    ok = band[0] <= value <= band[1]
    return (f"  {_flag(ok)}{label:<32s}{value:{fmt}} {unit:<7s}"
            f"target {band[0]:g} to {band[1]:g}")


def vehicle_report(veh: VehicleData, res: VehicleResults) -> str:
    L = ["", _RULE, " 1. VEHICLE AND STIFFNESS", _RULE]
    L.append(f"   Total / sprung mass            {veh.total_mass_kg:8.1f} / "
             f"{veh.sprung_mass_kg:.1f} kg")
    L.append(f"   Wheelbase                      {veh.wheelbase_mm:8.1f} mm")
    L.append(f"   CG height / station            {veh.cg_height_mm:8.1f} mm / "
             f"{veh.cg_from_front_axle_mm:.1f} mm behind the front axle")
    L.append(f"   Static front mass fraction     {100*veh.front_mass_fraction:8.1f} %")
    L.append(f"   Roll axis height at the CG     {res.roll_axis_height_at_cg_mm:8.1f} mm")
    L.append(f"   Roll moment arm                {res.roll_moment_arm_mm:8.1f} mm")
    L.append(f"   Target roll gradient           "
             f"{veh.target_roll_gradient_deg_per_g:8.2f} deg/g")
    L.append(f"   -> Required roll stiffness     "
             f"{res.required_roll_stiffness_Nm_per_deg:8.1f} N.m/deg")
    L.append(f"   -> Chassis torsional stiffness "
             f"{res.chassis_torsion_min_Nm_per_deg:8.0f} min / "
             f"{res.chassis_torsion_target_Nm_per_deg:.0f} target  N.m/deg")
    L.append(f"  {_flag(res.tilt_ok)}{veh.tilt_test_angle_deg:.0f} deg tilt test: "
             f"minimum track {res.tilt_min_track_mm:.0f} mm "
             f"(narrowest fitted {res.narrowest_track_mm:.0f} mm)")
    return "\n".join(L)


def axle_report(geo: AxleGeometry, kin: AxleKinematics,
                rates: SweepRates, alt_rates: SweepRates,
                roll: RollState, section: str) -> str:
    inp, lim = geo.inputs, geo.inputs.limits
    legacy_first = kin.legacy
    L = ["", _RULE, f" {section}. {inp.name.upper()} SUSPENSION", _RULE]

    L.append("")
    L.append(" FRONT-VIEW POINTS   (y outboard, z up, origin ground / centreline)")
    L.append(f"   {'point':<28s}{'y [mm]':>10s}{'z [mm]':>10s}")
    for label, pt in (("Lower ball joint (LBJ)", geo.lbj),
                      ("Upper ball joint (UBJ)", geo.ubj),
                      ("LCA inboard", geo.lca_in),
                      ("UCA inboard", geo.uca_in),
                      ("FVIC (construction)", geo.fvic)):
        L.append(f"   {label:<28s}{pt[0]:10.2f}{pt[1]:10.2f}")

    L.append("")
    L.append(" WISHBONES")
    L.append(f"   LCA / UCA length               {geo.lca_length_mm:8.2f} / "
             f"{geo.uca_length_mm:.2f} mm   (ratio {geo.uca_lca_ratio:.3f})")
    L.append(f"   Outboard vertical separation   {geo.outer_vertical_sep_mm:8.2f} mm")
    L.append(f"   Inboard vertical separation    {geo.inner_vertical_sep_mm:8.2f} mm")
    L.append(f"   LCA inclination                {geo.lca_inclination_deg:8.2f} deg   "
             f"({'falls' if geo.lca_inclination_deg > 0 else 'rises'} from wheel to chassis)")
    L.append(f"   UCA inclination                {geo.uca_inclination_deg:8.2f} deg   "
             f"({'falls' if geo.uca_inclination_deg > 0 else 'rises'} from wheel to chassis)")
    L.append(f"   RC that would flatten the LCA  "
             f"{geo.rc_height_for_flat_lca_mm:8.2f} mm")

    L.append("")
    L.append(" CHECKS")
    L.append(_band("Roll centre height", inp.rc_height_mm, lim.rc_height_mm))
    L.append(_band("FVSA length", inp.fvsa_length_mm, lim.fvsa_length_mm))
    L.append(_band("Scrub radius", geo.scrub_radius_mm, lim.scrub_radius_mm))
    L.append(_band("KPI", inp.kpi_deg, lim.kpi_deg, "deg"))
    L.append(_band("Kingpin length (LBJ-UBJ)", inp.kingpin_length_mm,
                   lim.kingpin_length_mm))
    L.append(_band("LCA length", geo.lca_length_mm, lim.lca_length_mm))
    L.append(_band("UCA / LCA ratio", geo.uca_lca_ratio, lim.uca_lca_ratio, "-",
                   "8.3f"))
    L.append(_band("Camber gain", abs(rates.camber_gain_deg_per_mm),
                   lim.camber_gain_deg_per_mm, "deg/mm", "8.4f"))
    L.append(f"  {_flag(geo.fvic[0] < 0)}{'FVIC on the far side of the car':<32s}"
             f"{geo.fvic[0]:8.2f} mm")
    rim_lo, rim_hi = inp.rim_z_band
    c = lim.ball_joint_clearance_mm
    inside = (rim_lo + c) <= geo.lbj[1] and geo.ubj[1] <= (rim_hi - c)
    L.append(f"  {_flag(inside)}{'Ball joints inside the rim':<32s}"
             f"{geo.lbj[1]:8.0f} / {geo.ubj[1]:.0f} mm  "
             f"window {rim_lo:.0f} to {rim_hi:.0f}")

    L.append("")
    L.append(" RATES ABOUT STATIC")
    lc, rc_ = ("legacy (PDF)", "corrected") if legacy_first else ("corrected", "legacy (PDF)")
    L.append(f"   {'quantity':<32s}{lc:>14s}{rc_:>14s}")
    for label, a, b, unit in (
        ("Camber gain [deg/mm]", rates.camber_gain_deg_per_mm,
         alt_rates.camber_gain_deg_per_mm, ""),
        ("Roll centre migration [mm/mm]", rates.rc_migration_mm_per_mm,
         alt_rates.rc_migration_mm_per_mm, ""),
        ("Half-track change [mm/mm]", rates.half_track_change_mm_per_mm,
         alt_rates.half_track_change_mm_per_mm, ""),
        ("Camber at full bump [deg]", rates.camber_full_bump_deg,
         alt_rates.camber_full_bump_deg, ""),
        ("Camber at full droop [deg]", rates.camber_full_droop_deg,
         alt_rates.camber_full_droop_deg, ""),
    ):
        L.append(f"   {label:<32s}{a:14.4f}{b:14.4f}{unit}")
    L.append(f"   {'RC over the travel range [mm]':<32s}"
             f"{rates.rc_min_mm:7.1f} to {rates.rc_max_mm:<4.1f}"
             f"{alt_rates.rc_min_mm:8.1f} to {alt_rates.rc_max_mm:.1f}")
    L.append(f"   (travel {inp.travel_bump_mm:.0f} mm bump / "
             f"{inp.travel_droop_mm:.0f} mm droop; camber gain x 25 mm = "
             f"{rates.camber_gain_deg_per_mm * inp.travel_bump_mm:+.2f} deg)")

    L.append("")
    L.append(f" AT {roll.roll_deg:.1f} DEG OF ROLL   (camber relative to the ROAD)")
    L.append(f"   Outer wheel                    {roll.outer_camber_deg:8.2f} deg   "
             f"(static {inp.static_camber_deg:+.2f})")
    L.append(f"   Inner wheel                    {roll.inner_camber_deg:8.2f} deg")
    L.append(f"   Roll centre height             {roll.rc_height_mm:8.1f} mm   "
             f"(design {inp.rc_height_mm:.1f})")
    L.append(f"   Roll centre lateral migration  {roll.rc_lateral_mm:8.1f} mm")
    L.append(f"   Wheel travel at that roll      {roll.wheel_travel_mm:8.2f} mm")
    band = lim.outer_camber_in_roll_deg
    ok = band[0] <= roll.outer_camber_deg <= band[1]
    L.append(f"  {_flag(ok)}Outer wheel camber in the useful window "
             f"({band[0]:.1f} to {band[1]:.1f} deg)")
    if roll.outer_camber_deg > band[1]:
        L.append("      -> outer wheel gone positive: add static negative camber "
                 "or shorten the FVSA")

    L.append("")
    L.append(" LONGITUDINAL LAYOUT AND ANTI-GEOMETRY   (x positive REARWARD)")
    L.append(f"   LCA pickups x                  {geo.lca_in_front_x_mm:8.1f} / "
             f"{geo.lca_in_rear_x_mm:.1f} mm   (base {inp.lca_base_mm:.0f}, "
             f"sweep {inp.lca_sweep_mm:.0f})")
    L.append(f"   UCA pickups x                  {geo.uca_in_front_x_mm:8.1f} / "
             f"{geo.uca_in_rear_x_mm:.1f} mm   (base {inp.uca_base_mm:.0f}, "
             f"sweep {inp.uca_sweep_mm:.0f})")
    L.append(f"  {_flag(geo.lca_ea_ratio <= lim.ea_ratio_max)}"
             f"{'LCA e/a ratio':<32s}{geo.lca_ea_ratio:8.2f}         "
             f"target <= {lim.ea_ratio_max:g}")
    L.append(f"  {_flag(geo.uca_ea_ratio <= lim.ea_ratio_max)}"
             f"{'UCA e/a ratio':<32s}{geo.uca_ea_ratio:8.2f}         "
             f"target <= {lim.ea_ratio_max:g}")
    L.append(_band(geo.anti_label, geo.anti_percent, lim.anti_percent, "%"))
    if geo.svic is None:
        L.append("      pivot axes are horizontal -> SVIC at infinity -> anti = 0 %")
        L.append("      NOT a computed result: the input model cannot express an")
        L.append("      inclined pivot axis, so this is always 0. See the docstring.")
    else:
        L.append(f"      SVIC at x = {geo.svic[0]:.0f} mm, z = {geo.svic[1]:.0f} mm")
    return "\n".join(L)


def hardpoints_report(model: ModelHardpoints) -> str:
    L = ["", _RULE, " 4. MODEL HARDPOINTS  --  THE DELIVERABLE", _RULE]
    L.append(f" {model.frame}")
    L.append(" Each corner: 4 inboard chassis pickups + 2 outboard ball joints.")
    for c in model.corners:
        L.append("")
        L.append(f" [{c.corner}]  {c.axle} {c.side}")
        L.append(f"   {'point':<16s}{'X [mm]':>11s}{'Y [mm]':>11s}{'Z [mm]':>11s}")
        for name, (x, y, z) in c.all_points():
            marker = " " if name in ARM_POINT_NAMES else "*"
            L.append(f"  {marker}{name:<16s}{x:11.2f}{y:11.2f}{z:11.2f}")
    L.append("")
    L.append(" (* reference points, not hardpoints)")
    return "\n".join(L)


def notes_report(front: AxleGeometry, legacy: bool) -> str:
    fvsa = front.inputs.fvsa_length_mm
    mode = "legacy (reproduces the 2027 PDF)" if legacy else "corrected"
    L = ["", _RULE, " 5. NOTES", _RULE]
    L.append(f" Sweep mode in use for the checks: {mode}.")
    L.append(" The legacy column carries a sign error in the upright rigid-body")
    L.append(" rotation of the original script. It changes the RATES only -- every")
    L.append(" hardpoint and every static KPI above is identical in both modes.")
    L.append(f" Corrected front camber gain = 57.2958 / FVSA = 57.2958 / {fvsa:.0f}")
    L.append(f" = {R2D / fvsa:.4f} deg/mm, the textbook value for a wheel turning")
    L.append(f" about an instant centre {fvsa:.0f} mm away. Run with --corrected to")
    L.append(" switch the checks over.")
    L.append("")
    L.append(" Values tagged design_intent (chosen, not measured or computed):")
    L.append("   static camber, roll centre heights, FVSA lengths, KPI, roll")
    L.append("   gradient target, chassis stiffness factors, brake bias.")
    L.append(" Leg loads (PDF sections 2 and 3) are NOT computed here: the lateral")
    L.append(" load transfer distribution behind those numbers is not recoverable")
    L.append(" from the document, and guessing it would produce confident nonsense.")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 7. FSAE 2027 CONFIGURATION  --  edit these numbers, that is the whole point
# --------------------------------------------------------------------------- #

VEHICLE_2027 = VehicleData(
    name="FSAE 2027 -- PUCPR Racing",
    total_mass_kg=315.0,
    unsprung_mass_kg=45.0,
    cg_height_mm=320.0,
    wheelbase_mm=1540.0,
    cg_from_front_axle_mm=693.0,
    target_roll_gradient_deg_per_g=1.00,
)

FRONT_2027 = AxleInputs(
    name="Front",
    track_mm=1240.0,                 # half track 620 mm
    loaded_radius_mm=245.0,
    rim_diameter_in=13.0,
    static_camber_deg=-1.50,
    lbj_y_mm=582.0,
    lbj_z_mm=130.0,
    kpi_deg=10.0,
    kingpin_length_mm=259.34,
    inner_pickup_y_mm=175.0,
    rc_height_mm=35.0,
    fvsa_length_mm=1500.0,
    axle_x_mm=0.0,
    lca_base_mm=260.0, lca_sweep_mm=0.0,
    uca_base_mm=240.0, uca_sweep_mm=0.0,
    limits=CheckLimits(kpi_deg=(6.0, 14.0)),
)

REAR_2027 = AxleInputs(
    name="Rear",
    track_mm=1200.0,                 # half track 600 mm
    loaded_radius_mm=245.0,
    rim_diameter_in=13.0,
    static_camber_deg=-1.50,
    lbj_y_mm=558.6,
    lbj_z_mm=130.0,
    kpi_deg=8.5,
    kingpin_length_mm=263.23,
    inner_pickup_y_mm=175.0,
    rc_height_mm=55.0,
    fvsa_length_mm=1400.0,
    axle_x_mm=1540.0,
    lca_base_mm=340.0, lca_sweep_mm=230.0,
    uca_base_mm=320.0, uca_sweep_mm=220.0,
    limits=CheckLimits(kpi_deg=(3.0, 10.0)),
)


# --------------------------------------------------------------------------- #
# 8. TOP LEVEL
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DesignReport:
    vehicle: VehicleData
    vehicle_results: VehicleResults
    front: AxleGeometry
    rear: AxleGeometry
    model: ModelHardpoints
    text: str


def run(veh: VehicleData = VEHICLE_2027,
        front_in: AxleInputs = FRONT_2027,
        rear_in: AxleInputs = REAR_2027,
        *, legacy: bool = True, with_sweep: bool = True) -> DesignReport:
    """Solve everything and build the text report."""

    front = solve_axle(front_in, veh)
    rear = solve_axle(rear_in, veh)
    model = build_model(front, rear)

    veh_res = solve_vehicle(veh, front.inputs.rc_height_mm, rear.inputs.rc_height_mm,
                            min(front_in.track_mm, rear_in.track_mm))

    chunks = [_RULE, " SLA SUSPENSION GEOMETRY -- FSAE 2027", f" {veh.name}",
              " Double A-arm, front and rear. Lengths in mm, angles in deg.",
              _RULE, vehicle_report(veh, veh_res)]

    if with_sweep:
        for geo, section in ((front, "2"), (rear, "3")):
            kin = AxleKinematics(geo, legacy_rotation_sign=legacy)
            alt = AxleKinematics(geo, legacy_rotation_sign=not legacy)
            chunks.append(axle_report(geo, kin, compute_rates(kin), compute_rates(alt),
                                      compute_roll(kin, geo.inputs.roll_reference_deg),
                                      section))
    else:
        chunks.append("\n(sweep skipped: --no-sweep)")

    chunks.append(hardpoints_report(model))
    chunks.append(notes_report(front, legacy))

    return DesignReport(vehicle=veh, vehicle_results=veh_res, front=front, rear=rear,
                        model=model, text="\n".join(chunks) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SLA suspension hardpoint synthesis -- FSAE 2027.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Edit VEHICLE_2027 / FRONT_2027 / REAR_2027 near the bottom of the "
               "file to change the design inputs.")
    ap.add_argument("--corrected", action="store_true",
                    help="use the corrected upright rotation for the checks "
                         "(default reproduces the 2027 PDF)")
    ap.add_argument("--no-sweep", action="store_true",
                    help="static synthesis only, skip the kinematic sweep")
    ap.add_argument("--json", metavar="PATH", help="write the hardpoints to JSON")
    ap.add_argument("--csv", metavar="PATH", help="write the hardpoints to CSV")
    ap.add_argument("--quiet", action="store_true", help="suppress the text report")
    args = ap.parse_args(argv)

    rep = run(legacy=not args.corrected, with_sweep=not args.no_sweep)

    if not args.quiet:
        print(rep.text)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rep.model.to_dict(), fh, indent=2)
        print(f"hardpoints written to {args.json}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["corner", "point", "x_mm", "y_mm", "z_mm"])
            for corner, name, x, y, z in rep.model.rows():
                w.writerow([corner, name, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])
        print(f"hardpoints written to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
