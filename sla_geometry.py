#!/usr/bin/env python3
"""
sla_geometry.py
===============

Double A-arm (SLA) suspension hardpoint synthesis for the PUCPR Racing FSAE 2027
car -- front and rear axle, all four corners.

Merged from ``sla_geometry_complete.py`` (architecture, ISO 8855 model, corrected
rotation, JSON/CSV export, CLI) and ``sla_geometry_with_x.py`` (pivot-axis rise
for real anti-geometry, force decomposition, anti-solver).

WHAT IT DOES
------------
You supply a small set of *design intent* numbers (track, tyre radius, roll
centre height, front-view swing-arm length, lower ball joint position, KPI,
kingpin length, chassis pickup plane, longitudinal base/sweep, and optionally
pivot-axis rise for anti-geometry).  The script returns the complete A-arm
hardpoint set of every corner:

    UCA_IN_FRONT   UCA_IN_REAR   UCA_OUT        (upper wishbone: 2 in + 1 out)
    LCA_IN_FRONT   LCA_IN_REAR   LCA_OUT        (lower wishbone: 2 in + 1 out)
    WHEEL_CENTRE   CONTACT_PATCH                (reference points)

plus static KPIs, kinematic rates, roll behaviour, anti-geometry, and
approximate leg forces under representative load cases.


THE SYNTHESIS CHAIN
-------------------
Everything follows from the front view.  The *outboard* points are chosen by
package (rim, upright, steering); the *inboard* chassis pickups are DERIVED:

  1. FVIC (front view instant centre) is placed by two targets:
         y_ic = half_track - fvsa_length
         z_ic = rc_height * fvsa_length / half_track

  2. LBJ (lower ball joint) is an input.

  3. UBJ (upper ball joint) sits on the kingpin axis, ``kingpin_length`` above
     the LBJ, leaning inboard by KPI.

  4. The two wishbone axes are the lines BALL-JOINT -> FVIC.  The inboard
     pickups are where those lines cross the chassis pickup plane y = inner_y.

  5. Longitudinally each wishbone is defined by a ``base`` and a ``sweep``.
     Optionally, ``dz`` inclines the pivot axis to produce anti-geometry.


COORDINATE SYSTEMS  (two are used -- do not mix them)
-----------------------------------------------------
* DESIGN frame (front-view working frame):
      y positive OUTBOARD, z positive UP, origin at ground / car centreline,
      x positive REARWARD from the front axle.

* MODEL frame = ISO 8855:
      X positive FORWARD, Y positive LEFT, Z positive UP,
      origin at the front axle centreline, on the ground, on the car centreline.
      Left corners (FL, RL) have Y > 0; right corners (FR, RR) have Y < 0.
      Conversion:  X_iso = -x_rearward,  Y_iso = +/- y_outboard,  Z_iso = z.

Units: mm and degrees throughout the I/O.  Radians only inside the solvers.
Camber sign: negative = top of the wheel inboard (both sides).
Scrub sign:  positive = kingpin axis meets the ground INBOARD of the contact
             patch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field, replace as _replace
from typing import Iterator, Literal, Sequence

import numpy as np
from scipy.optimize import brentq

# --------------------------------------------------------------------------- #
# Types and constants
# --------------------------------------------------------------------------- #

Vec2 = np.ndarray  # shape (2,), float64
Point3D = tuple[float, float, float]
Band = tuple[float, float]

GRAVITY = 9.81
D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class KinematicError(RuntimeError):
    """The mechanism cannot be assembled in the requested state."""


# --------------------------------------------------------------------------- #
# Geometric helpers
# --------------------------------------------------------------------------- #

def line_intersection(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2) -> Vec2 | None:
    """Intersection of the infinite lines (p1,p2) and (p3,p4). None if parallel."""
    p1, p2, p3, p4 = (np.asarray(v, float) for v in (p1, p2, p3, p4))
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    a = x1 * y2 - y1 * x2
    b = x3 * y4 - y3 * x4
    return np.array([(a * (x3 - x4) - (x1 - x2) * b) / den,
                     (a * (y3 - y4) - (y1 - y2) * b) / den])


def circle_intersection(c1: Vec2, r1: float, c2: Vec2, r2: float,
                        reference: Vec2) -> Vec2:
    """Intersect two circles; return the branch closest to ``reference``."""
    c1, c2, ref = (np.asarray(v, float) for v in (c1, c2, reference))
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)
    if d > r1 + r2 or d < abs(r1 - r2) or d < 1e-12:
        raise KinematicError(
            f"four-bar does not close: centre distance {d:.3f} mm, radii "
            f"{r1:.3f} / {r2:.3f} mm -- requested travel exceeds the kinematic limit"
        )
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h = np.sqrt(max(r1 * r1 - a * a, 0.0))
    base = c1 + a * d_vec / d
    perp = np.array([-d_vec[1], d_vec[0]]) / d
    s1 = base + h * perp
    s2 = base - h * perp
    return s1 if np.linalg.norm(s1 - ref) < np.linalg.norm(s2 - ref) else s2


def rot2(angle_rad: float) -> np.ndarray:
    """2x2 rotation matrix, counter-clockwise in the (y, z) plane."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def nz(v: float) -> float:
    """Collapse negative zero."""
    return v + 0.0


# --------------------------------------------------------------------------- #
# 1. VEHICLE AND STIFFNESS
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, kw_only=True)
class VehicleData:
    """Mass properties and stiffness targets. All of these are INPUTS."""

    name: str = "FSAE 2027 -- PUCPR Racing"
    total_mass_kg: float = 315.0
    unsprung_mass_kg: float = 45.0
    cg_height_mm: float = 320.0
    wheelbase_mm: float = 1540.0
    cg_from_front_axle_mm: float = 693.0
    target_roll_gradient_deg_per_g: float = 1.00
    chassis_factor_min: float = 3.0
    chassis_factor_target: float = 5.0
    tilt_test_angle_deg: float = 60.0
    brake_bias_front: float = 0.65

    @property
    def sprung_mass_kg(self) -> float:
        return self.total_mass_kg - self.unsprung_mass_kg

    @property
    def front_mass_fraction(self) -> float:
        return 1.0 - self.cg_from_front_axle_mm / self.wheelbase_mm


@dataclass(frozen=True)
class VehicleResults:
    """Derived roll / stiffness / tilt numbers."""

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
    frac = veh.cg_from_front_axle_mm / veh.wheelbase_mm
    h_axis = front_rc_mm + (rear_rc_mm - front_rc_mm) * frac
    arm_mm = veh.cg_height_mm - h_axis
    w_sprung = veh.sprung_mass_kg * GRAVITY
    k_roll = w_sprung * (arm_mm / 1000.0) / veh.target_roll_gradient_deg_per_g
    tilt_min = 2.0 * veh.cg_height_mm * np.tan(veh.tilt_test_angle_deg * D2R)
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
# 2. AXLE INPUTS
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, kw_only=True)
class CheckLimits:
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
    ball_joint_clearance_mm: float = 15.0


@dataclass(frozen=True, kw_only=True)
class AxleInputs:
    """Everything the designer chooses for one axle. Nothing here is derived."""

    name: str

    # wheel and tyre package
    track_mm: float
    loaded_radius_mm: float = 245.0
    rim_diameter_in: float = 13.0
    static_camber_deg: float = -1.50

    # outboard: lower ball joint and kingpin axis
    lbj_y_mm: float
    lbj_z_mm: float = 130.0
    kpi_deg: float
    kingpin_length_mm: float

    # inboard: chassis pickup plane
    inner_pickup_y_mm: float = 175.0

    # front-view targets
    rc_height_mm: float
    fvsa_length_mm: float

    # side view / longitudinal layout
    axle_x_mm: float
    lca_base_mm: float
    lca_sweep_mm: float
    uca_base_mm: float
    uca_sweep_mm: float

    # pivot-axis rise: dz = z(rear pickup) - z(front pickup)
    # non-zero values incline the pivot axis and produce anti-geometry
    dz_lca_mm: float = 0.0
    dz_uca_mm: float = 0.0

    # sweep settings
    travel_bump_mm: float = 25.0
    travel_droop_mm: float = 25.0
    roll_reference_deg: float = 1.5

    limits: CheckLimits = field(default_factory=CheckLimits)

    @property
    def half_track_mm(self) -> float:
        return self.track_mm / 2.0

    @property
    def contact_patch(self) -> Vec2:
        """Tyre contact centre -- the TRACK DATUM.

        ``track_mm`` is measured between the two contact centres, so this
        point sits at ``half_track_mm`` whatever the static camber is. Scrub
        radius, FVSA, the roll centre construction, the tilt test and lateral
        load transfer are all referenced to this point.
        """
        return np.array([self.half_track_mm, 0.0])

    @property
    def wheel_centre(self) -> Vec2:
        """Wheel centre, displaced from the contact patch by static camber.

        The static camber is built into the UPRIGHT (the hub face is machined
        at ``static_camber_deg``); the ball joints, KPI and kingpin length are
        unaffected, and the 2 mm plates at the upper arm trim around this
        nominal. Negative camber tips the top of the wheel inboard, so the
        wheel centre sits INBOARD of the contact patch by
        ``loaded_radius * tan(camber)``.

        ``loaded_radius_mm`` is the vertical distance from the wheel centre to
        the road, so the height is unaffected by camber.
        """
        return np.array([
            self.half_track_mm
            + self.loaded_radius_mm * np.tan(self.static_camber_deg * D2R),
            self.loaded_radius_mm,
        ])

    @property
    def wheel_centre_track_mm(self) -> float:
        """Track measured at the wheel centres, not at the ground.

        Differs from ``track_mm`` whenever static camber is non-zero. Reported
        so the two are never confused: the rules and the tilt test use
        ``track_mm``, wheel offset and upright width use this one.
        """
        return 2.0 * float(self.wheel_centre[0])

    @property
    def rim_z_band(self) -> Band:
        r = self.rim_diameter_in * 25.4 / 2.0
        return (self.loaded_radius_mm - r, self.loaded_radius_mm + r)


# --------------------------------------------------------------------------- #
# 3. STATIC SYNTHESIS -- the four wishbone hardpoints in the front view
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class AxleGeometry:
    """Solved static geometry of one axle, in the design front view."""

    inputs: AxleInputs

    lbj: Vec2
    ubj: Vec2
    lca_in: Vec2
    uca_in: Vec2
    fvic: Vec2

    lca_length_mm: float
    uca_length_mm: float
    lca_inclination_deg: float
    uca_inclination_deg: float
    outer_vertical_sep_mm: float
    inner_vertical_sep_mm: float

    scrub_radius_mm: float
    rc_height_for_flat_lca_mm: float

    # side view (x positive REARWARD)
    lca_in_front_x_mm: float
    lca_in_rear_x_mm: float
    uca_in_front_x_mm: float
    uca_in_rear_x_mm: float
    lca_in_front_z_mm: float
    lca_in_rear_z_mm: float
    uca_in_front_z_mm: float
    uca_in_rear_z_mm: float
    lca_ea_ratio: float
    uca_ea_ratio: float
    svic: Vec2 | None
    anti_percent: float
    anti_label: str

    @property
    def uca_lca_ratio(self) -> float:
        return self.uca_length_mm / self.lca_length_mm


def solve_axle(inp: AxleInputs, veh: VehicleData) -> AxleGeometry:
    """Turn the design targets into hardpoints."""

    # front view instant centre
    ic_y = inp.half_track_mm - inp.fvsa_length_mm
    ic_z = inp.rc_height_mm * inp.fvsa_length_mm / inp.half_track_mm
    fvic = np.array([ic_y, ic_z])

    # outboard ball joints
    lbj = np.array([inp.lbj_y_mm, inp.lbj_z_mm])
    kpi = inp.kpi_deg * D2R
    ubj = lbj + inp.kingpin_length_mm * np.array([-np.sin(kpi), np.cos(kpi)])

    # inboard pickups: where each wishbone axis crosses y = inner_y
    def on_line_at_y(outer: Vec2, ic: Vec2, y: float) -> Vec2:
        t = (y - outer[0]) / (ic[0] - outer[0])
        return outer + t * (ic - outer)

    lca_in = on_line_at_y(lbj, fvic, inp.inner_pickup_y_mm)
    uca_in = on_line_at_y(ubj, fvic, inp.inner_pickup_y_mm)

    lca_len = float(np.linalg.norm(lbj - lca_in))
    uca_len = float(np.linalg.norm(ubj - uca_in))

    # steering axis at ground level
    y_axis_at_ground = inp.lbj_y_mm + inp.lbj_z_mm * np.tan(kpi)
    scrub = inp.half_track_mm - y_axis_at_ground

    # longitudinal layout
    def pickups_x(base: float, sweep: float) -> tuple[float, float]:
        mid = inp.axle_x_mm - sweep
        return (mid - base / 2.0, mid + base / 2.0)

    lca_xf, lca_xr = pickups_x(inp.lca_base_mm, inp.lca_sweep_mm)
    uca_xf, uca_xr = pickups_x(inp.uca_base_mm, inp.uca_sweep_mm)

    lca_ea = 2.0 * abs(inp.lca_sweep_mm) / inp.lca_base_mm
    uca_ea = 2.0 * abs(inp.uca_sweep_mm) / inp.uca_base_mm

    # per-pickup z with pivot-axis rise
    lca_z_front = float(lca_in[1]) - inp.dz_lca_mm / 2.0
    lca_z_rear = float(lca_in[1]) + inp.dz_lca_mm / 2.0
    uca_z_front = float(uca_in[1]) - inp.dz_uca_mm / 2.0
    uca_z_rear = float(uca_in[1]) + inp.dz_uca_mm / 2.0

    is_front = inp.axle_x_mm < veh.wheelbase_mm / 2.0
    svic, anti = _side_view_anti(inp, veh, is_front,
                                 (lca_xf, lca_z_front), (lca_xr, lca_z_rear),
                                 (uca_xf, uca_z_front), (uca_xr, uca_z_rear),
                                 (inp.axle_x_mm, float(lbj[1])),
                                 (inp.axle_x_mm, float(ubj[1])))

    return AxleGeometry(
        inputs=inp,
        lbj=lbj, ubj=ubj, lca_in=lca_in, uca_in=uca_in, fvic=fvic,
        lca_length_mm=lca_len, uca_length_mm=uca_len,
        lca_inclination_deg=float(np.degrees(
            np.arctan2(lbj[1] - lca_in[1], lbj[0] - lca_in[0]))),
        uca_inclination_deg=float(np.degrees(
            np.arctan2(ubj[1] - uca_in[1], ubj[0] - uca_in[0]))),
        outer_vertical_sep_mm=float(ubj[1] - lbj[1]),
        inner_vertical_sep_mm=float(uca_in[1] - lca_in[1]),
        scrub_radius_mm=float(scrub),
        rc_height_for_flat_lca_mm=float(
            inp.lbj_z_mm * inp.half_track_mm / inp.fvsa_length_mm),
        lca_in_front_x_mm=lca_xf, lca_in_rear_x_mm=lca_xr,
        uca_in_front_x_mm=uca_xf, uca_in_rear_x_mm=uca_xr,
        lca_in_front_z_mm=lca_z_front, lca_in_rear_z_mm=lca_z_rear,
        uca_in_front_z_mm=uca_z_front, uca_in_rear_z_mm=uca_z_rear,
        lca_ea_ratio=lca_ea, uca_ea_ratio=uca_ea,
        svic=svic, anti_percent=anti,
        anti_label="Anti-dive" if is_front else "Anti-squat",
    )


def _side_view_anti(inp: AxleInputs, veh: VehicleData, is_front: bool,
                    lca_front_xz: tuple[float, float],
                    lca_rear_xz: tuple[float, float],
                    uca_front_xz: tuple[float, float],
                    uca_rear_xz: tuple[float, float],
                    lbj_xz: tuple[float, float],
                    ubj_xz: tuple[float, float],
                    ) -> tuple[Vec2 | None, float]:
    """Side view instant centre and anti-dive / anti-squat percentage.

    With non-zero dz the pivot axes are inclined, the SVIC is finite, and
    anti-geometry is a live design variable.

    CONSTRUCTION -- the lines pass through the BALL JOINTS, not the pickups.
        An arm rotating about an inboard axis of side-view slope ``m = dz/base``
        moves its ball joint with velocity slope ``dx/dz = -m`` (the arm length
        cancels: v = omega * n_hat x r, and both components scale with the same
        lateral offset). The instant centre must therefore lie on the line
        through the BALL JOINT perpendicular to that velocity -- i.e. slope
        ``m`` through the ball joint.

        The inboard pickups fix only the axis DIRECTION. Intersecting the two
        pickup lines instead puts the SVIC on lines parallel to the right ones
        but offset by the pickup-to-ball-joint height difference, which is
        wrong: with ``dz_lca = 0`` the LBJ moves purely vertically, so the SVIC
        must sit at the LBJ height (130 mm here), not at the LCA pickup height
        (117.4 mm). That error was worth 0.9 % to 50 % of the reported anti
        across a 20-geometry sweep. The corrected construction reproduces the
        instant centre measured from the full 3D linkage (``DWSolver``) exactly
        -- see tests/benchmarks/test_anti_geometry_and_ackermann.py.

    PARALLEL AXES are not zero anti. Equal slopes put the SVIC at infinity, but
    the swing arm is still inclined, and ``z/dx -> m`` in the limit, so
    ``tan(theta) = m``. Returning 0 % there (as this function used to, via
    ``line_intersection`` returning None) is a discontinuity: a hair off
    parallel already gives the full value.
    """
    def axis_slope(front_xz: tuple[float, float],
                   rear_xz: tuple[float, float]) -> float | None:
        run = rear_xz[0] - front_xz[0]
        if abs(run) < 1e-12:
            return None          # pickups coincident in side view
        return (rear_xz[1] - front_xz[1]) / run

    m_lca = axis_slope(lca_front_xz, lca_rear_xz)
    m_uca = axis_slope(uca_front_xz, uca_rear_xz)
    if m_lca is None or m_uca is None:
        return None, 0.0

    svic: Vec2 | None
    if abs(m_lca - m_uca) < 1e-15:
        # Parallel: swing arm at infinity, inclined at the shared slope.
        svic = None
        tan_theta = m_lca
    else:
        x = ((ubj_xz[1] - lbj_xz[1]) + m_lca * lbj_xz[0] - m_uca * ubj_xz[0]) \
            / (m_lca - m_uca)
        z = lbj_xz[1] + m_lca * (x - lbj_xz[0])
        svic = np.array([x, z])
        dx = x - inp.axle_x_mm
        if abs(dx) < 1e-9:
            return svic, 0.0
        tan_theta = z / dx

    # theta is measured at the contact patch. The rear axle looks the other way
    # along X, so its angle flips sign. `+ 0.0` clears the negative zero that
    # flip produces on a flat axle, which would otherwise print "-0.00 %".
    if not is_front:
        tan_theta = -tan_theta + 0.0

    h_over_l = veh.cg_height_mm / veh.wheelbase_mm
    if is_front:
        # Only the front share of the braking force reacts through this axle.
        return svic, float(100.0 * veh.brake_bias_front * tan_theta / h_over_l)
    # Anti-squat: all drive torque goes to the rear, so no bias factor.
    return svic, float(100.0 * tan_theta / h_over_l)


# --------------------------------------------------------------------------- #
# 4. KINEMATICS -- front view four-bar sweep
# --------------------------------------------------------------------------- #

def dz_uca_for_anti(inp: AxleInputs, veh: VehicleData,
                    target_pct: float, mode: str = "dive") -> float:
    """Back-solve the UCA pivot-axis rise (dz_uca) to hit a target anti-%."""
    def err(dz: float) -> float:
        inp2 = _replace(inp, dz_uca_mm=dz)
        geo = solve_axle(inp2, veh)
        return geo.anti_percent - target_pct

    if abs(target_pct) < 1e-9:
        return inp.dz_lca_mm

    lo, hi = inp.dz_lca_mm - 120.0, inp.dz_lca_mm + 120.0
    eps = 1e-3
    for a_, b_ in [(lo, inp.dz_lca_mm - eps), (inp.dz_lca_mm + eps, hi)]:
        try:
            if err(a_) * err(b_) < 0:
                return brentq(err, a_, b_, xtol=1e-9)
        except (ValueError, ZeroDivisionError, KinematicError):
            continue
    return np.nan


# --------------------------------------------------------------------------- #
# 6. FORCE DECOMPOSITION
# --------------------------------------------------------------------------- #

def leg_forces(bj: np.ndarray, p_front: np.ndarray, p_rear: np.ndarray,
               load_at_bj: np.ndarray) -> np.ndarray:
    """Divide the ball joint load between the two legs of a wishbone.

    Returns array([F_front_leg, F_rear_leg]). Positive = tension.
    """
    bj, p1, p2 = (np.asarray(v, float) for v in (bj, p_front, p_rear))
    u1, u2 = p1 - bj, p2 - bj
    n = np.cross(u1, u2)
    n = n / np.linalg.norm(n)

    F = np.asarray(load_at_bj, float)
    F_plane = F - np.dot(F, n) * n

    e1 = u1 / np.linalg.norm(u1)
    e2 = u2 / np.linalg.norm(u2)
    b2 = np.cross(n, e1)
    A = np.array([[1.0, np.dot(e2, e1)],
                  [0.0, np.dot(e2, b2)]])
    rhs = -np.array([np.dot(F_plane, e1), np.dot(F_plane, b2)])
    return np.linalg.solve(A, rhs)


def upright_reactions(geo: AxleGeometry) -> dict:
    """Return 3D positions of ball joints and UCA direction for force analysis."""
    inp = geo.inputs
    lbj_3d = np.array([inp.axle_x_mm, float(geo.lbj[0]), float(geo.lbj[1])])
    ubj_3d = np.array([inp.axle_x_mm, float(geo.ubj[0]), float(geo.ubj[1])])
    cp_3d = np.array([inp.axle_x_mm, inp.half_track_mm, 0.0])

    d = geo.uca_in - geo.ubj
    d_3d = np.array([0.0, float(d[0]), float(d[1])])
    d_3d = d_3d / np.linalg.norm(d_3d)

    return {"lbj": lbj_3d, "ubj": ubj_3d, "cp": cp_3d, "uca_dir": d_3d}


def solve_ball_joint_forces(geo: AxleGeometry, load_cp: np.ndarray,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Equilibrium of the upright -> forces at both ball joints.

    Returns (F_lbj, F_ubj) -- forces applied BY the upright on the wishbones.
    """
    r = upright_reactions(geo)
    F = np.asarray(load_cp, float)
    r_cp = r["cp"] - r["lbj"]
    r_u = r["ubj"] - r["lbj"]
    d = r["uca_dir"]

    M = np.cross(r_cp, F)[0]
    denom = np.cross(r_u, d)[0]
    T_u = -M / denom
    F_ubj = T_u * d
    F_lbj = -(F + F_ubj)
    return -F_lbj, -F_ubj


def load_cases(geo: AxleGeometry, veh: VehicleData,
               mu_y: float = 1.5, mu_x: float = 1.4,
               lltd: float = 0.55, ay: float = 1.5) -> dict:
    """Approximate tyre contact-patch loads for the outer loaded wheel."""
    inp = geo.inputs
    W = veh.total_mass_kg * GRAVITY
    is_front = inp.axle_x_mm < veh.wheelbase_mm / 2.0
    frac = veh.front_mass_fraction if is_front else 1 - veh.front_mass_fraction
    share = lltd if is_front else 1 - lltd

    Fz_static = W * frac / 2.0
    dW = share * (W * ay * veh.cg_height_mm / 1000.0) / (inp.track_mm / 1000.0)
    Fz = Fz_static + dW
    return {
        "Cornering": np.array([0.0, -mu_y * Fz, Fz]),
        "Braking": np.array([mu_x * Fz, 0.0, Fz]),
        "Combined": np.array([0.7 * mu_x * Fz, -0.7 * mu_y * Fz, Fz]),
        "Fz": Fz,
    }


# --------------------------------------------------------------------------- #
# 7. 3D MODEL HARDPOINTS (ISO 8855)
# --------------------------------------------------------------------------- #

CornerId = Literal["FL", "FR", "RL", "RR"]

ARM_POINT_NAMES: tuple[str, ...] = (
    "UCA_IN_FRONT", "UCA_IN_REAR", "UCA_OUT",
    "LCA_IN_FRONT", "LCA_IN_REAR", "LCA_OUT",
)


@dataclass(frozen=True)
class CornerHardpoints:
    """One corner in the ISO 8855 model frame."""

    corner: CornerId
    axle: str
    side: Literal["left", "right"]

    uca_in_front: Point3D
    uca_in_rear: Point3D
    uca_out: Point3D
    lca_in_front: Point3D
    lca_in_rear: Point3D
    lca_out: Point3D
    wheel_centre: Point3D
    contact_patch: Point3D

    def arm_points(self) -> Iterator[tuple[str, Point3D]]:
        yield from zip(ARM_POINT_NAMES,
                       (self.uca_in_front, self.uca_in_rear, self.uca_out,
                        self.lca_in_front, self.lca_in_rear, self.lca_out))

    def all_points(self) -> Iterator[tuple[str, Point3D]]:
        yield from self.arm_points()
        yield "WHEEL_CENTER", self.wheel_centre
        yield "CONTACT_PATCH", self.contact_patch


def build_corner(geo: AxleGeometry, side: Literal["left", "right"],
                 corner: CornerId) -> CornerHardpoints:
    """Lift the front-view solution into 3D and convert to ISO 8855."""
    inp = geo.inputs
    sy = 1.0 if side == "left" else -1.0

    def p(x_rearward: float, y_outboard: float, z: float) -> Point3D:
        return (nz(-x_rearward), nz(sy * y_outboard), nz(z))

    x_out = inp.axle_x_mm
    return CornerHardpoints(
        corner=corner, axle=inp.name, side=side,
        uca_in_front=p(geo.uca_in_front_x_mm, float(geo.uca_in[0]),
                       geo.uca_in_front_z_mm),
        uca_in_rear=p(geo.uca_in_rear_x_mm, float(geo.uca_in[0]),
                      geo.uca_in_rear_z_mm),
        uca_out=p(x_out, float(geo.ubj[0]), float(geo.ubj[1])),
        lca_in_front=p(geo.lca_in_front_x_mm, float(geo.lca_in[0]),
                       geo.lca_in_front_z_mm),
        lca_in_rear=p(geo.lca_in_rear_x_mm, float(geo.lca_in[0]),
                      geo.lca_in_rear_z_mm),
        lca_out=p(x_out, float(geo.lbj[0]), float(geo.lbj[1])),
        wheel_centre=p(x_out, float(inp.wheel_centre[0]), float(inp.wheel_centre[1])),
        contact_patch=p(x_out, float(inp.contact_patch[0]), float(inp.contact_patch[1])),
    )


@dataclass(frozen=True)
class ModelHardpoints:
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
# 8. REPORT
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


def member_legs_mm(geo: AxleGeometry) -> dict[str, float]:
    """True 3D length of each wishbone leg, off this script's own hardpoints.

    The front-view ``lca_length_mm`` is a PROJECTION -- the right quantity for
    the FVSA construction, but not the member anyone cuts, and not what decides
    whether a leg is slender enough to buckle. On a swept wishbone the front
    leg is longer than that projection (e.g. the rear LCA front leg is ~40 mm
    over its front-view length at the current e/a of 1.0).

    Caveat: these use the zero-caster outboard ball joints this script
    synthesises. steering_geometry.py supersedes them with caster-corrected
    points, so the merged summary's section 5 is the authority for the front
    axle. The rear has zero caster, so the two agree there.
    """
    corner = build_corner(geo, "left", "FL")
    lo = np.array(corner.lca_out)
    uo = np.array(corner.uca_out)
    return {
        "LCA front leg": float(np.linalg.norm(lo - np.array(corner.lca_in_front))),
        "LCA rear leg": float(np.linalg.norm(lo - np.array(corner.lca_in_rear))),
        "UCA front leg": float(np.linalg.norm(uo - np.array(corner.uca_in_front))),
        "UCA rear leg": float(np.linalg.norm(uo - np.array(corner.uca_in_rear))),
    }


def axle_report(geo: AxleGeometry, section: str, veh: VehicleData) -> str:
    inp, lim = geo.inputs, geo.inputs.limits
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
    L.append(_band("UCA / LCA ratio", geo.uca_lca_ratio, lim.uca_lca_ratio, "-",
                   "8.3f"))
    # Camber gain from the design FVSA: 57.2958 / FVSA is the exact rate for a
    # wheel turning about an instant centre that far away. The SOLVED rate,
    # which carries pivot-axis rake and 3D effects, is in section 3b of the
    # merged summary (vdcore.analysis.axle).
    L.append(_band("Camber gain (from FVSA)", R2D / inp.fvsa_length_mm,
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
    L.append(" MEMBER LENGTHS   (true 3D, not the front-view projection)")
    L.append(f"   front-view LCA projection is {geo.lca_length_mm:.2f} mm -- "
             f"the FVSA quantity, not a member")
    for leg_label, leg_len in member_legs_mm(geo).items():
        L.append(_band(leg_label, leg_len, lim.lca_length_mm))


    L.append("")
    L.append(" LONGITUDINAL LAYOUT AND ANTI-GEOMETRY   (x positive REARWARD)")
    L.append(f"   LCA pickups x                  {geo.lca_in_front_x_mm:8.1f} / "
             f"{geo.lca_in_rear_x_mm:.1f} mm   (base {inp.lca_base_mm:.0f}, "
             f"sweep {inp.lca_sweep_mm:.0f})")
    L.append(f"   UCA pickups x                  {geo.uca_in_front_x_mm:8.1f} / "
             f"{geo.uca_in_rear_x_mm:.1f} mm   (base {inp.uca_base_mm:.0f}, "
             f"sweep {inp.uca_sweep_mm:.0f})")
    if inp.dz_lca_mm != 0.0 or inp.dz_uca_mm != 0.0:
        L.append(f"   LCA pivot-axis rise (dz)       {inp.dz_lca_mm:8.1f} mm")
        L.append(f"   UCA pivot-axis rise (dz)       {inp.dz_uca_mm:8.1f} mm")
        L.append(f"   LCA front/rear z               {geo.lca_in_front_z_mm:8.1f} / "
                 f"{geo.lca_in_rear_z_mm:.1f} mm")
        L.append(f"   UCA front/rear z               {geo.uca_in_front_z_mm:8.1f} / "
                 f"{geo.uca_in_rear_z_mm:.1f} mm")
    L.append(f"  {_flag(geo.lca_ea_ratio <= lim.ea_ratio_max)}"
             f"{'LCA e/a ratio':<32s}{geo.lca_ea_ratio:8.2f}         "
             f"target <= {lim.ea_ratio_max:g}")
    L.append(f"  {_flag(geo.uca_ea_ratio <= lim.ea_ratio_max)}"
             f"{'UCA e/a ratio':<32s}{geo.uca_ea_ratio:8.2f}         "
             f"target <= {lim.ea_ratio_max:g}")
    L.append(_band(geo.anti_label, geo.anti_percent, lim.anti_percent, "%"))
    if geo.svic is None:
        L.append("      pivot axes are horizontal -> SVIC at infinity -> anti = 0 %")
        if inp.dz_lca_mm == 0.0 and inp.dz_uca_mm == 0.0:
            L.append("      set dz_lca_mm / dz_uca_mm to incline the pivot axes")
    else:
        L.append(f"      SVIC at x = {geo.svic[0]:.0f} mm, z = {geo.svic[1]:.0f} mm")

    # leg forces
    lc = load_cases(geo, veh)
    L.append("")
    L.append(f" LEG FORCES   (Fz outer estimated {lc['Fz']:.0f} N)")

    lbj_3d = np.array([inp.axle_x_mm, float(geo.lbj[0]), float(geo.lbj[1])])
    ubj_3d = np.array([inp.axle_x_mm, float(geo.ubj[0]), float(geo.ubj[1])])
    y_in = float(geo.lca_in[0])

    lca_front_3d = np.array([geo.lca_in_front_x_mm, y_in, geo.lca_in_front_z_mm])
    lca_rear_3d = np.array([geo.lca_in_rear_x_mm, y_in, geo.lca_in_rear_z_mm])
    uca_front_3d = np.array([geo.uca_in_front_x_mm, float(geo.uca_in[0]),
                             geo.uca_in_front_z_mm])
    uca_rear_3d = np.array([geo.uca_in_rear_x_mm, float(geo.uca_in[0]),
                            geo.uca_in_rear_z_mm])

    L.append(f"   {'case':<14s} {'LCA front':>11s} {'LCA rear':>11s} "
             f"{'UCA front':>11s} {'UCA rear':>11s}")
    for case_name in ("Cornering", "Braking", "Combined"):
        F_lbj, F_ubj = solve_ball_joint_forces(geo, lc[case_name])
        fl = leg_forces(lbj_3d, lca_front_3d, lca_rear_3d, F_lbj)
        fu = leg_forces(ubj_3d, uca_front_3d, uca_rear_3d, F_ubj)
        L.append(f"   {case_name:<14s} {fl[0]:11.0f} {fl[1]:11.0f} "
                 f"{fu[0]:11.0f} {fu[1]:11.0f}")
    L.append("   (positive = tension, negative = compression)")

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
    L.extend(_static_alignment_block(model))
    return "\n".join(L)


def _static_alignment_block(model: ModelHardpoints) -> list[str]:
    """State the alignment the exported points actually carry.

    Read back off the points themselves, not off the config: a consumer
    recovers static camber from CONTACT_PATCH -> WHEEL_CENTER, so this is the
    only place the deliverable can be checked against design intent. Until
    the wheel centre was displaced by the camber, this table silently encoded
    a zero-camber car while every rate table above assumed the design value.
    """
    L = ["", " STATIC ALIGNMENT CARRIED BY THIS TABLE",
         "   recovered from CONTACT_PATCH -> WHEEL_CENTER, not restated from the config"]

    for c in model.corners:
        (_, wy, wz), (_, cy, cz) = c.wheel_centre, c.contact_patch
        sy = 1.0 if c.side == "left" else -1.0
        off = sy * (wy - cy)
        cam = float(np.degrees(np.arctan2(off, wz - cz)))
        where = "inboard" if off < 0 else "outboard"
        L.append(f"   [{c.corner}]  camber {cam:+7.3f} deg    "
                 f"wheel centre {abs(off):5.2f} mm {where} of the contact patch")

    tracks: dict[str, tuple[float, float]] = {}
    for c in model.corners:
        if c.side != "left":
            continue
        tracks[c.axle] = (2.0 * abs(c.contact_patch[1]), 2.0 * abs(c.wheel_centre[1]))

    L.append("")
    L.append(f"   {'axle':<10s}{'ground track':>15s}{'wheel-centre track':>21s}")
    for axle, (gt, wt) in tracks.items():
        L.append(f"   {axle:<10s}{gt:12.1f} mm{wt:18.1f} mm")

    L.append("")
    L.append("   TRACK IS MEASURED AT THE CONTACT PATCHES. That is the number the")
    L.append("   rules, the tilt test and lateral load transfer all use. The wheel")
    L.append("   centres sit inboard of it by loaded_radius * tan(camber); wheel")
    L.append("   offset and upright width have to absorb the difference.")
    L.append("")
    L.append("   Static camber is built into the UPRIGHT, so the ball joints, KPI")
    L.append("   and kingpin length are the designed values and the 2 mm plates at")
    L.append("   the upper arm trim +/-0.44 deg around this nominal.")
    L.append("")
    L.append("   Static TOE is zero by design. Note that a contact-patch /")
    L.append("   wheel-centre pair CANNOT encode toe at all -- toe rotates the")
    L.append("   wheel about a vertical axis and leaves both points where they")
    L.append("   are. A consumer needing static toe must read it from the config.")
    return L


def notes_report(front: AxleGeometry) -> str:
    fvsa = front.inputs.fvsa_length_mm
    L = ["", _RULE, " 5. NOTES", _RULE]
    L.append(" THIS SCRIPT IS STATIC SYNTHESIS ONLY.")
    L.append(" Rates, roll behaviour and the kinematic plots are solved in 3D by")
    L.append(" vdcore.analysis.axle and printed in section 3b of the merged")
    L.append(" summary (scripts/geometry_summary.py). They live there because")
    L.append(" DWSolver needs the tie rod to close the sixth degree of freedom,")
    L.append(" and the tie rod comes from steering_geometry.py -- only the merged")
    L.append(" corner is complete.")
    L.append("")
    L.append(" The front-view four-bar that used to produce those tables here was")
    L.append(" blind to dz_lca_mm / dz_uca_mm, so inclining the pivot axes for")
    L.append(" anti-dive left every rate unchanged while the real geometry moved.")
    L.append(" The --legacy flag went with it: it reproduced a known sign error in")
    L.append(" the original script's upright rotation, and a deliberately wrong")
    L.append(" code path has no place in the reference.")
    L.append("")
    L.append(f" Design camber gain = 57.2958 / FVSA = 57.2958 / {fvsa:.0f}")
    L.append(f" = {R2D / fvsa:.4f} deg/mm, the textbook rate for a wheel turning")
    L.append(f" about an instant centre {fvsa:.0f} mm away. The SOLVED rate carries")
    L.append(" pivot rake and 3D effects and will differ slightly.")
    L.append("")
    L.append(" Values tagged design_intent (chosen, not measured or computed):")
    L.append("   static camber, roll centre heights, FVSA lengths, KPI, roll")
    L.append("   gradient target, chassis stiffness factors, brake bias.")
    return '\n'.join(L)


# --------------------------------------------------------------------------- #
# 10. FSAE 2027 CONFIGURATION
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
    track_mm=1240.0,
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
    track_mm=1200.0,
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
    # Rear pickup held on the rear-axle line (no pickup behind the axle):
    # sweep = base/2 places the rearmost pickup at x = axle_x exactly.
    # This forces e/a = 1.0; base is shrunk to 180/160 to keep both front
    # legs under the 430 mm limit. See docs/GEOMETRY_AUDIT_2026-08-26_rev3.md.
    lca_base_mm=180.0, lca_sweep_mm=90.0,
    uca_base_mm=160.0, uca_sweep_mm=80.0,
    limits=CheckLimits(kpi_deg=(3.0, 10.0)),
)


# --------------------------------------------------------------------------- #
# 11. TOP LEVEL
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
        rear_in: AxleInputs = REAR_2027) -> DesignReport:
    front = solve_axle(front_in, veh)
    rear = solve_axle(rear_in, veh)
    model = build_model(front, rear)

    veh_res = solve_vehicle(veh, front.inputs.rc_height_mm, rear.inputs.rc_height_mm,
                            min(front_in.track_mm, rear_in.track_mm))

    chunks = [_RULE, " SLA SUSPENSION GEOMETRY -- FSAE 2027", f" {veh.name}",
              " Double A-arm, front and rear. Lengths in mm, angles in deg.",
              _RULE, vehicle_report(veh, veh_res)]

    for geo, section in ((front, "2"), (rear, "3")):
        chunks.append(axle_report(geo, section, veh))

    chunks.append(hardpoints_report(model))
    chunks.append(notes_report(front))

    return DesignReport(vehicle=veh, vehicle_results=veh_res, front=front, rear=rear,
                        model=model, text="\n".join(chunks) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="SLA suspension hardpoint synthesis -- FSAE 2027.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Edit VEHICLE_2027 / FRONT_2027 / REAR_2027 near the bottom of the "
               "file to change the design inputs.")
    ap.add_argument("--json", metavar="PATH", help="write the hardpoints to JSON")
    ap.add_argument("--csv", metavar="PATH", help="write the hardpoints to CSV")
    ap.add_argument("--quiet", action="store_true", help="suppress the text report")
    args = ap.parse_args(argv)

    rep = run()

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
