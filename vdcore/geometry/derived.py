"""Derived geometry quantities computed from hardpoint models.

These are free functions, not model methods, to keep models/ as
pure inputs and avoid a circular import (models cannot import geometry).

Coordinate system: ISO 8855 — X+ forward, Y+ LEFT, Z+ up.
"""

from __future__ import annotations

import math

from vdcore.models.hardpoint import Axle, Corner, DerivedPoint, Vehicle


def contact_patch(corner: Corner, camber_deg: float = 0.0) -> DerivedPoint:
    """Compute the contact patch position from wheel centre and tire radius.

    Accounts for camber:
      - Lateral shift: r * sin(gamma)  (positive gamma shifts patch outboard)
      - Vertical correction: r * (1 - cos(gamma))

    ISO 8855: X+ forward, Y+ LEFT, Z+ up.
    For left corners (positive Y), negative camber (top inboard) means the
    top of the wheel tilts toward Y=0, so the contact patch shifts in +Y.
    For right corners (negative Y), the shift is in -Y (toward Y=0).

    Args:
        corner: The suspension corner.
        camber_deg: Camber angle in degrees. Negative = top inboard.

    Returns:
        Contact patch position as a DerivedPoint.
    """
    wc = corner.wheel_center
    r = corner.tire.loaded_radius_mm
    gamma = math.radians(camber_deg)

    lateral_shift = r * math.sin(gamma)
    vertical_correction = r * (1.0 - math.cos(gamma))

    # The wheel-plane normal mirrors between left and right corners.
    # For a left corner (Y+), the contact patch lateral displacement is +r*sin(γ).
    # For a right corner (Y-), the wheel faces the other way, so the displacement
    # is -r*sin(γ) — toward Y=0 for negative camber on both sides.
    is_left = corner.corner_id in ("FL", "RL")
    y_patch = wc.y_mm + lateral_shift if is_left else wc.y_mm - lateral_shift

    z_patch = wc.z_mm - r + vertical_correction

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
