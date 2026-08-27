"""Derived geometry quantities computed from hardpoint models.

These are free functions, not model methods, to keep models/ as
pure inputs and avoid a circular import (models cannot import geometry).

Coordinate system: ISO 8855 — X+ forward, Y+ LEFT, Z+ up.
"""

from __future__ import annotations

import math

from vdcore.geometry.solver import SolverResult
from vdcore.models.hardpoint import Axle, Corner, DerivedPoint, Vehicle


def contact_patch(corner: Corner, camber_deg: float = 0.0) -> DerivedPoint:
    """Compute the contact patch position from wheel centre and tire radius.

    ``loaded_radius_mm`` is the VERTICAL distance from the wheel centre to the
    road (the SAE definition, and what a corner-weight scale measures), so the
    contact patch always lies exactly on the ground plane, z = wc_z - r,
    whatever the camber. Scrub radius, mechanical trail and the roll-centre
    construction therefore all reference a true z=0 ground plane.

    Accounts for camber:
      - Lateral shift: -r * tan(gamma)  (negative gamma shifts patch OUTBOARD)
      - No vertical correction: the patch stays on the ground.

    ISO 8855: X+ forward, Y+ LEFT, Z+ up.

    The contact patch is at the BOTTOM of the wheel, so it moves opposite to
    the top. Negative camber tips the top inboard, therefore the bottom — and
    the contact patch — moves OUTBOARD. This is why building static camber
    into a design widens the track at the ground and increases scrub radius.

    For a left corner (Y+), outboard is +Y, so negative camber gives
    ``y_patch > wc.y_mm``. For a right corner (Y-), outboard is -Y, so
    negative camber gives ``y_patch < wc.y_mm``. Both move AWAY from Y=0.

    Args:
        corner: The suspension corner.
        camber_deg: Camber angle in degrees. Negative = top inboard.

    Returns:
        Contact patch position as a DerivedPoint.
    """
    wc = corner.wheel_center
    r = corner.tire.loaded_radius_mm
    gamma = math.radians(camber_deg)

    lateral_shift = -r * math.tan(gamma)

    # The wheel-plane normal mirrors between left and right corners, and the
    # patch sits at the bottom of the wheel, so the shift is -r*tan(γ) on the
    # left and +r*tan(γ) on the right: outboard on both sides for negative
    # camber.
    is_left = corner.corner_id in ("FL", "RL")
    y_patch = wc.y_mm + lateral_shift if is_left else wc.y_mm - lateral_shift

    z_patch = wc.z_mm - r

    return DerivedPoint(x_mm=wc.x_mm, y_mm=y_patch, z_mm=z_patch)


def contact_patch_uncambered(corner: Corner) -> DerivedPoint:
    """Contact patch ignoring camber (vertical projection of wheel centre).

    ISO 8855: X+ forward, Y+ LEFT, Z+ up.
    """
    return contact_patch(corner, camber_deg=0.0)


def track_mm(axle: Axle, camber_left_deg: float = 0.0, camber_right_deg: float = 0.0) -> float:
    """Track width from contact patches.

    ISO 8855: Y+ is LEFT, so left CP has positive Y, right CP has negative Y.
    Track = left_y - right_y (always positive).

    Args:
        axle: The axle (left + right corners).
        camber_left_deg: Camber of the left wheel in degrees.
        camber_right_deg: Camber of the right wheel in degrees.
    """
    cp_left = contact_patch(axle.left, camber_left_deg)
    cp_right = contact_patch(axle.right, camber_right_deg)
    return cp_left.y_mm - cp_right.y_mm


def wheelbase_mm(vehicle: Vehicle) -> float:
    """Wheelbase from front and rear axle contact patches (uncambered).

    ISO 8855: X+ is FORWARD, so front axle has more positive X.
    Wheelbase = front_x - rear_x (always positive).
    """
    cp_front = contact_patch_uncambered(vehicle.front.left)
    cp_rear = contact_patch_uncambered(vehicle.rear.left)
    return cp_front.x_mm - cp_rear.x_mm


def _kingpin_ground_intercept(result: SolverResult) -> tuple[float, float, float]:
    """Point where the kingpin axis pierces the ground plane (z = 0).

    The kingpin axis is the line through the two solved ball joints, from the
    lower (LBJ) toward the upper (UBJ). We walk along it from the LBJ by the
    parameter ``t`` that drives z to zero and return the (x, y, z) intercept.

    ISO 8855: X+ forward, Y+ LEFT, Z+ up; the contact patch sits at z = 0, so
    this intercept and the patch are compared in the same ground plane.

    Args:
        result: A solved corner (carries UBJ, LBJ, contact patch).

    Returns:
        (x_mm, y_mm, z_mm) of the ground intercept; z_mm is 0 by construction.

    Raises:
        ValueError: if the kingpin axis is (near) horizontal, so it never
            reaches the ground plane -- returning a number here would be a
            silent fiction, so we refuse.
    """
    ubj = result.ubj
    lbj = result.lbj
    kp_x = ubj.x_mm - lbj.x_mm
    kp_y = ubj.y_mm - lbj.y_mm
    kp_z = ubj.z_mm - lbj.z_mm
    if abs(kp_z) < 1e-10:
        raise ValueError(
            "Kingpin axis is horizontal (UBJ and LBJ at equal height); it "
            "never intercepts the ground plane, so scrub radius and mechanical "
            "trail are undefined."
        )
    t = -lbj.z_mm / kp_z
    return (lbj.x_mm + t * kp_x, lbj.y_mm + t * kp_y, 0.0)


def scrub_radius_mm(result: SolverResult) -> float:
    """Scrub radius: lateral offset from the kingpin ground intercept to the
    contact patch, in the ground plane.

    ISO 8855: Y+ is LEFT. Defined as ``contact_patch_y - kingpin_ground_y`` on
    the LEFT corner and its mirror on the RIGHT, so a positive value always
    means the contact patch sits OUTBOARD of the kingpin ground point (the
    common FSAE "positive scrub" convention). This matches the OptimumK-
    correlated derivation pinned at +15.08 mm on the 2027 front-left corner.

    The sign is corner-dependent because "outboard" is +Y on the left and -Y on
    the right; we fold that in here so both sides report positive for an
    outboard patch. Negative static camber moves the patch outboard (see
    :func:`contact_patch`), increasing scrub on both sides.

    Args:
        result: A solved corner (:class:`SolverResult`).

    Returns:
        Scrub radius in mm; positive = contact patch outboard of the kingpin.

    Raises:
        ValueError: if the kingpin axis never reaches the ground plane.
    """
    _, kp_ground_y, _ = _kingpin_ground_intercept(result)
    cp_y = result.contact_patch.y_mm
    # Left corner: outboard is +Y, so outboard patch => cp_y > kp_ground_y > 0.
    # Right corner: outboard is -Y, so outboard patch => cp_y < kp_ground_y < 0.
    # Flipping the right-side difference makes "outboard" positive on both sides.
    is_left = result.contact_patch.y_mm >= 0.0
    delta = cp_y - kp_ground_y
    return delta if is_left else -delta


def mechanical_trail_mm(result: SolverResult) -> float:
    """Mechanical trail: longitudinal offset from the contact patch to the
    kingpin ground intercept, in the ground plane.

    ISO 8855: X+ is FORWARD. Defined as ``kingpin_ground_x - contact_patch_x``
    so a POSITIVE value means the kingpin ground intercept lies AHEAD of the
    contact patch -- i.e. the tyre trails behind the steer axis, the
    self-aligning ("positive trail") case. This reproduces the +21.43 mm pinned
    on the 2027 front geometry.

    NOTE ON FRAME: the root ``steering_geometry.py`` computes trail in the
    design frame (X+ REARWARD) as ``cp_x - kp_gnd_x``. Here X+ is FORWARD, so
    the equivalent "intercept ahead of patch = positive" expression flips to
    ``kp_gnd_x - cp_x``. Do not copy the legacy app's
    ``model_3d.mechanical_trail_mm`` -- its sign is inverted for this frame.

    Args:
        result: A solved corner (:class:`SolverResult`).

    Returns:
        Mechanical trail in mm; positive = kingpin ground point ahead of patch.

    Raises:
        ValueError: if the kingpin axis never reaches the ground plane.
    """
    kp_ground_x, _, _ = _kingpin_ground_intercept(result)
    cp_x = result.contact_patch.x_mm
    return kp_ground_x - cp_x
