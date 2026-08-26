"""Tests for vdcore.geometry.derived — contact patch, track, wheelbase."""

from __future__ import annotations

import math

import pytest

from vdcore.geometry.derived import (
    contact_patch,
    contact_patch_uncambered,
    track_mm,
    wheelbase_mm,
)
from vdcore.models.hardpoint import (
    Axle,
    Corner,
    Hardpoint,
    TirePackage,
    Vehicle,
)


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire(r: float = 228.0) -> TirePackage:
    return TirePackage(loaded_radius_mm=r, source="cad", tol_mm=1.0)


def _corner(cid: str, y_sign: float, x_offset: float = 0.0) -> Corner:
    s = y_sign
    return Corner(
        corner_id=cid,
        uca_inboard_front=_hp("UCA_IF", 80 + x_offset, s * 150, 280),
        uca_inboard_rear=_hp("UCA_IR", -80 + x_offset, s * 150, 280),
        uca_outboard=_hp("UCA_O", 0 + x_offset, s * 530, 290),
        lca_inboard_front=_hp("LCA_IF", 100 + x_offset, s * 130, 80),
        lca_inboard_rear=_hp("LCA_IR", -100 + x_offset, s * 130, 80),
        lca_outboard=_hp("LCA_O", 0 + x_offset, s * 580, 75),
        tie_rod_inboard=_hp("TR_I", -60 + x_offset, s * 160, 120),
        tie_rod_outboard=_hp("TR_O", -50 + x_offset, s * 540, 110),
        wheel_center=_hp("WC", 0 + x_offset, s * 600, 200),
        tire=_tire(),
        static_camber_deg=-2.0,
        static_toe_deg_per_side=0.0,
    )


class TestContactPatch:
    def test_uncambered_z_equals_wc_minus_radius(self) -> None:
        """With zero camber, contact patch Z = wheel_center Z - loaded_radius."""
        fl = _corner("FL", 1.0)
        cp = contact_patch_uncambered(fl)
        assert cp.z_mm == pytest.approx(200.0 - 228.0)

    def test_uncambered_y_equals_wc(self) -> None:
        """With zero camber, contact patch Y = wheel_center Y."""
        fl = _corner("FL", 1.0)
        cp = contact_patch_uncambered(fl)
        assert cp.y_mm == pytest.approx(600.0)

    def test_negative_camber_shifts_patch_outboard(self) -> None:
        """Negative camber shifts the contact patch AWAY from Y=0 (outboard).

        The patch is at the bottom of the wheel. Negative camber tips the TOP
        inboard, so the bottom goes outboard. This is why building static
        camber into a design widens the track at the ground -- see the
        FSAE2027 geometry summary, where -1.50 deg moves the patch 6.42 mm
        outboard and takes front scrub from 15.08 to 21.49 mm.

        Before the sign fix this asserted the opposite and passed, which put
        vdcore 2*r*sin(gamma) = 12.8 mm out on any cambered corner.
        """
        fl = _corner("FL", 1.0)
        cp_zero = contact_patch(fl, camber_deg=0.0)
        cp_neg = contact_patch(fl, camber_deg=-2.0)
        # Left corner (Y+): outboard is +Y
        assert cp_neg.y_mm > cp_zero.y_mm

    def test_negative_camber_right_corner_shifts_outboard(self) -> None:
        """For the right corner, negative camber also shifts outboard (lower Y)."""
        fr = _corner("FR", -1.0)
        cp_zero = contact_patch(fr, camber_deg=0.0)
        cp_neg = contact_patch(fr, camber_deg=-2.0)
        # Right corner (Y-): outboard is -Y
        assert cp_neg.y_mm < cp_zero.y_mm

    def test_camber_correction_magnitude(self) -> None:
        """Lateral shift should be r * tan(gamma), directed outboard.

        tan, not sin: loaded_radius_mm is the VERTICAL drop from wheel centre
        to road, so the patch slides along the ground plane by r*tan(gamma).
        """
        fl = _corner("FL", 1.0)
        gamma = -2.0
        r = 228.0
        expected_shift = -r * math.tan(math.radians(gamma))
        cp_zero = contact_patch(fl, camber_deg=0.0)
        cp_camb = contact_patch(fl, camber_deg=gamma)
        actual_shift = cp_camb.y_mm - cp_zero.y_mm
        assert actual_shift == pytest.approx(expected_shift)
        assert actual_shift > 0.0  # outboard for a left corner

    def test_camber_does_not_lift_patch_off_the_ground(self) -> None:
        """Camber must not change the patch height.

        loaded_radius_mm is the vertical wheel-centre-to-road distance, so the
        contact patch sits on the ground plane at every camber angle. A
        non-zero vertical term here would mean scrub, trail and the roll-centre
        construction were all measured off a plane that is not the road.
        """
        fl = _corner("FL", 1.0)
        cp_zero = contact_patch(fl, camber_deg=0.0)
        cp_camb = contact_patch(fl, camber_deg=-2.0)
        assert cp_camb.z_mm - cp_zero.z_mm == pytest.approx(0.0, abs=1e-12)
        assert cp_camb.z_mm == pytest.approx(fl.wheel_center.z_mm - 228.0)


class TestTrack:
    def test_positive_track(self) -> None:
        """Track width must be positive (left_y - right_y > 0 in ISO 8855)."""
        axle = Axle(left=_corner("FL", 1.0), right=_corner("FR", -1.0))
        t = track_mm(axle)
        assert t > 0

    def test_track_value(self) -> None:
        """Track = left_wc_y - right_wc_y at zero camber."""
        axle = Axle(left=_corner("FL", 1.0), right=_corner("FR", -1.0))
        t = track_mm(axle)
        assert t == pytest.approx(1200.0)  # 600 - (-600)

    def test_camber_widens_track(self) -> None:
        """Negative camber on both sides WIDENS the track at the ground.

        The contact patch is at the bottom of the wheel, so tipping the tops
        inboard pushes the patches outboard by r*tan|gamma| each. This is the
        same fact the FSAE2027 summary records: building in -1.50 deg moves
        each patch 6.42 mm outboard.

        Before the sign fix this asserted the opposite and passed.
        """
        axle = Axle(left=_corner("FL", 1.0), right=_corner("FR", -1.0))
        t_zero = track_mm(axle)
        t_camb = track_mm(axle, camber_left_deg=-5.0, camber_right_deg=-5.0)
        assert t_camb > t_zero
        # exactly two patches moved out by r*tan(5 deg), r = 228 mm
        assert t_camb - t_zero == pytest.approx(2 * 228.0 * math.tan(math.radians(5.0)))


class TestWheelbase:
    def test_positive_wheelbase(self) -> None:
        """Wheelbase must be positive (front_x > rear_x in ISO 8855)."""
        fl = _corner("FL", 1.0, x_offset=0.0)
        fr = _corner("FR", -1.0, x_offset=0.0)
        rl = _corner("RL", 1.0, x_offset=-1550.0)
        rr = _corner("RR", -1.0, x_offset=-1550.0)
        v = Vehicle(front=Axle(left=fl, right=fr), rear=Axle(left=rl, right=rr))
        wb = wheelbase_mm(v)
        assert wb == pytest.approx(1550.0)
