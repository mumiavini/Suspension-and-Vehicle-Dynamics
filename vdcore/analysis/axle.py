"""Axle-level rate and roll analysis on the 3D double-wishbone solver.

This is the analysis that used to live in ``sla_geometry.AxleKinematics``, a
private front-view four-bar. That model was exact only while every pivot axis
ran parallel to X: it was built from the projected arm length and never read
``dz_lca_mm`` / ``dz_uca_mm``, so inclining the pivot axes to buy anti-dive
left every rate bit-identical while the real geometry moved (2.6% off camber
gain at 28.5% anti-dive, 5.1% at 57%). Running on :class:`DWSolver` removes
that failure mode: pivot rake is carried by the constraint set itself.

Frame: ISO 8855, X+ forward, Y+ LEFT, Z+ up. Origin at the front axle
centreline on the ground plane.

Roll convention: the chassis rolls by ``roll_deg`` and both wheels stay on the
road. Equivalently, and as implemented here, the road tilts under a fixed
chassis and the wheel travel per side is solved so both contact patches land on
the tilted plane. Camber is reported relative to the ROAD, which is the frame
the tyre works in.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.optimize import brentq

from vdcore.analysis.roll_centre import front_view_instant_centre
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Axle, Corner, DerivedPoint

Arr = NDArray[np.float64]

__all__ = [
    "AxleRates",
    "AxleRollState",
    "CornerSample",
    "axle_rates",
    "axle_roll",
    "sample_corner",
]


class ConvergenceError(RuntimeError):
    """A solve in the sweep did not converge.

    Raised rather than returned so a non-converged state can never reach a
    report as a plausible-looking number.
    """


class CornerSample(BaseModel, frozen=True):
    """One solved corner position, reduced to the front-view quantities.

    All lengths mm, angles deg. ``camber_deg`` is relative to the CHASSIS;
    :func:`axle_roll` adds the roll angle to convert to road-relative.
    """

    wheel_travel_mm: float
    camber_deg: float
    cp_y_mm: float
    cp_z_mm: float
    fvic_y_mm: float
    fvic_z_mm: float
    fvsa_mm: float
    rc_height_mm: float


class AxleRates(BaseModel, frozen=True):
    """Rates about the static position, and the extremes of travel."""

    camber_gain_deg_per_mm: float
    rc_migration_mm_per_mm: float
    half_track_change_mm_per_mm: float
    camber_full_bump_deg: float
    camber_full_droop_deg: float
    rc_min_mm: float
    rc_max_mm: float


class AxleRollState(BaseModel, frozen=True):
    """Axle state at a given chassis roll angle, both wheels on the road."""

    roll_deg: float
    wheel_travel_mm: float
    outer_camber_deg: float
    inner_camber_deg: float
    rc_height_mm: float
    rc_lateral_mm: float


def _rot2(theta_rad: float) -> Arr:
    """2D rotation in the (Y, Z) plane."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def _line_intersection(p1: Arr, p2: Arr, p3: Arr, p4: Arr) -> Arr | None:
    """Intersection of line (p1,p2) with line (p3,p4). None if parallel."""
    d = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    if abs(d) < 1e-12:
        return None
    t = ((p1[0] - p3[0]) * (p3[1] - p4[1]) - (p1[1] - p3[1]) * (p3[0] - p4[0])) / d
    return np.array([p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1])])


def sample_corner(
    corner: Corner,
    solver: DWSolver,
    wheel_travel_mm: float,
) -> CornerSample:
    """Solve one corner at a wheel travel and reduce it to the front view.

    ``wheel_travel_mm`` is positive in bump (wheel up relative to chassis).
    All returned Z values are CHASSIS-referenced, not ground-referenced.

    Raises:
        ConvergenceError: if the solve did not converge.
    """
    r: SolverResult = solver.solve(wheel_travel_mm=wheel_travel_mm)
    if not r.converged:
        raise ConvergenceError(
            f"{corner.corner_id} did not converge at "
            f"{wheel_travel_mm:+.3f} mm travel "
            f"(residual {r.residual_norm:.3e})"
        )
    # DWSolver drives travel by displacing the CHASSIS down by wheel_travel_mm,
    # so its results are world-frame. Convert the moving points to the CHASSIS
    # frame BEFORE the instant-centre construction: front_view_instant_centre
    # reads the inboard pivots off the Corner, and those are chassis-fixed, so
    # feeding it world-frame ball joints mixes two frames and puts the FVIC in
    # the wrong place at every travel except zero.
    #
    # Chassis-referenced is also the right frame for these quantities -- ride
    # height measured from the sprung mass, which is what the roll-couple
    # calculation needs. Ground-referenced RC migration differs by exactly
    # 1 mm per mm of travel.
    def to_chassis(p: DerivedPoint) -> DerivedPoint:
        return DerivedPoint(
            x_mm=p.x_mm, y_mm=p.y_mm, z_mm=p.z_mm + wheel_travel_mm
        )

    ubj_c = to_chassis(r.ubj)
    lbj_c = to_chassis(r.lbj)
    cp_c = to_chassis(r.contact_patch)

    fvic = front_view_instant_centre(
        corner, ubj=ubj_c, lbj=lbj_c, contact_patch=cp_c
    )
    cp = np.array([cp_c.y_mm, cp_c.z_mm])
    ic = np.array([fvic.fvic_y_mm, fvic.fvic_z_mm])
    centre = _line_intersection(
        cp, ic, np.array([0.0, 0.0]), np.array([0.0, 1.0])
    )
    rc_z = float(centre[1]) if centre is not None else math.nan
    return CornerSample(
        wheel_travel_mm=wheel_travel_mm,
        camber_deg=r.camber_deg,
        cp_y_mm=float(cp[0]),
        cp_z_mm=float(cp[1]),
        fvic_y_mm=float(ic[0]),
        fvic_z_mm=float(ic[1]),
        fvsa_mm=fvic.fvsa_mm,
        rc_height_mm=rc_z,
    )


def axle_rates(
    axle: Axle,
    *,
    travel_bump_mm: float = 25.0,
    travel_droop_mm: float = 25.0,
    sweep_steps: int = 41,
) -> AxleRates:
    """Rates about static, from a central difference on the left corner.

    The derivative step is ``travel_bump_mm / 20`` to match the historical
    front-view implementation, so the numbers are directly comparable.

    Sign convention: ``camber_gain_deg_per_mm`` is per mm of BUMP, and is
    negative for a geometry that gains negative camber in bump.

    Raises:
        ConvergenceError: if any solve in the sweep fails.
    """
    corner = axle.left
    solver = DWSolver(corner)
    step = travel_bump_mm / 20.0

    up = sample_corner(corner, solver, +step)
    dn = sample_corner(corner, solver, -step)
    two_h = 2.0 * step

    full_bump = sample_corner(corner, solver, +travel_bump_mm)
    full_droop = sample_corner(corner, solver, -travel_droop_mm)

    rc_all = [
        sample_corner(corner, solver, float(t)).rc_height_mm
        for t in np.linspace(-travel_droop_mm, travel_bump_mm, sweep_steps)
    ]

    return AxleRates(
        camber_gain_deg_per_mm=(up.camber_deg - dn.camber_deg) / two_h,
        rc_migration_mm_per_mm=(up.rc_height_mm - dn.rc_height_mm) / two_h,
        half_track_change_mm_per_mm=(up.cp_y_mm - dn.cp_y_mm) / two_h,
        camber_full_bump_deg=full_bump.camber_deg,
        camber_full_droop_deg=full_droop.camber_deg,
        rc_min_mm=min(rc_all),
        rc_max_mm=max(rc_all),
    )


def axle_roll(axle: Axle, roll_deg: float) -> AxleRollState:
    """Solve the axle at a chassis roll angle with both wheels on the road.

    The wheel travel per side is found by requiring both contact patches to
    lie on the tilted road plane. Camber is returned relative to the ROAD:
    the outer wheel loses the roll angle, the inner gains it.

    Raises:
        ConvergenceError: if any solve fails.
        ValueError: if the travel solve does not bracket a root.
    """
    phi = math.radians(roll_deg)
    # Outer = the wheel in bump. For a symmetric axle the choice of side is
    # arbitrary; left is taken as outer. Both corners are already in ISO 8855
    # (left +Y, right -Y), so no mirroring is applied -- unlike the design-frame
    # implementation this replaces, where both sides carried positive Y.
    outer_corner, inner_corner = axle.left, axle.right
    outer_solver = DWSolver(outer_corner)
    inner_solver = DWSolver(inner_corner)

    def patch_mismatch(travel: float) -> float:
        o = sample_corner(outer_corner, outer_solver, +travel)
        i = sample_corner(inner_corner, inner_solver, -travel)
        cp_o = _rot2(-phi) @ np.array([o.cp_y_mm, o.cp_z_mm])
        cp_i = _rot2(-phi) @ np.array([i.cp_y_mm, i.cp_z_mm])
        return float(cp_o[1] - cp_i[1])

    if abs(roll_deg) < 1e-12:
        travel = 0.0
    else:
        static = sample_corner(outer_corner, outer_solver, 0.0)
        half_track = abs(static.cp_y_mm)
        guess = half_track * math.tan(phi)
        lo, hi = 0.2 * guess, 2.5 * guess
        if patch_mismatch(lo) * patch_mismatch(hi) > 0.0:
            raise ValueError(
                f"wheel travel at {roll_deg:.3f} deg roll is not bracketed by "
                f"[{lo:.3f}, {hi:.3f}] mm"
            )
        travel = float(brentq(patch_mismatch, lo, hi, xtol=1e-10))

    outer = sample_corner(outer_corner, outer_solver, +travel)
    inner = sample_corner(inner_corner, inner_solver, -travel)

    rot = _rot2(-phi)
    cp_o = rot @ np.array([outer.cp_y_mm, outer.cp_z_mm])
    ic_o = rot @ np.array([outer.fvic_y_mm, outer.fvic_z_mm])
    cp_i = rot @ np.array([inner.cp_y_mm, inner.cp_z_mm])
    ic_i = rot @ np.array([inner.fvic_y_mm, inner.fvic_z_mm])

    # Put the outer patch back on the road so the RC is measured from ground.
    shift = np.array([0.0, -cp_o[1]])
    cp_o, ic_o, cp_i, ic_i = cp_o + shift, ic_o + shift, cp_i + shift, ic_i + shift

    rc = _line_intersection(cp_o, ic_o, cp_i, ic_i)
    rc_y = float(rc[0]) if rc is not None else math.nan
    rc_z = float(rc[1]) if rc is not None else math.nan

    return AxleRollState(
        roll_deg=roll_deg,
        wheel_travel_mm=travel,
        outer_camber_deg=outer.camber_deg + roll_deg,
        inner_camber_deg=inner.camber_deg - roll_deg,
        rc_height_mm=rc_z,
        rc_lateral_mm=rc_y,
    )
