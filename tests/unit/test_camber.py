"""Tests for vdcore.analysis.camber — static camber, camber gain, sweep."""

from __future__ import annotations

import math

import pytest

from vdcore.analysis.camber import (
    camber_gain_deg_per_mm,
    camber_sweep,
    static_camber_deg,
)
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire(r: float = 228.0) -> TirePackage:
    return TirePackage(loaded_radius_mm=r, source="cad", tol_mm=1.0)


def _corner(cid: str, y_sign: float) -> Corner:
    s = y_sign
    return Corner(
        corner_id=cid,
        uca_inboard_front=_hp("UCA_IF", 80, s * 150, 280),
        uca_inboard_rear=_hp("UCA_IR", -80, s * 150, 280),
        uca_outboard=_hp("UCA_O", 0, s * 530, 290),
        lca_inboard_front=_hp("LCA_IF", 100, s * 130, 80),
        lca_inboard_rear=_hp("LCA_IR", -100, s * 130, 80),
        lca_outboard=_hp("LCA_O", 0, s * 580, 75),
        tie_rod_inboard=_hp("TR_I", -60, s * 160, 120),
        tie_rod_outboard=_hp("TR_O", -50, s * 540, 110),
        wheel_center=_hp("WC", 0, s * 600, 200),
        tire=_tire(),
        static_camber_deg=-2.0,
        static_toe_deg_per_side=0.0,
    )


class TestStaticCamber:
    def test_returns_finite_value(self) -> None:
        c = static_camber_deg(_corner("FL", 1.0))
        assert math.isfinite(c)

    def test_left_right_equal(self) -> None:
        """Symmetric geometry must give identical camber on both sides."""
        cl = static_camber_deg(_corner("FL", 1.0))
        cr = static_camber_deg(_corner("FR", -1.0))
        assert cl == pytest.approx(cr, abs=0.01)

    def test_sign_negative_for_this_geometry(self) -> None:
        """With UCA_O inboard of LCA_O, the wheel tilts top-inboard = negative camber."""
        c = static_camber_deg(_corner("FL", 1.0))
        assert c < 0


class TestCamberGain:
    def test_returns_finite(self) -> None:
        gain = camber_gain_deg_per_mm(_corner("FL", 1.0))
        assert math.isfinite(gain)

    def test_left_right_equal_gain(self) -> None:
        """Symmetric geometry must give identical camber gain."""
        gl = camber_gain_deg_per_mm(_corner("FL", 1.0))
        gr = camber_gain_deg_per_mm(_corner("FR", -1.0))
        assert gl == pytest.approx(gr, abs=0.001)

    def test_gain_magnitude_reasonable(self) -> None:
        """Camber gain should be in a reasonable range (< 0.5 deg/mm typically)."""
        gain = camber_gain_deg_per_mm(_corner("FL", 1.0))
        assert abs(gain) < 0.5


class TestCamberSweep:
    def test_sweep_length(self) -> None:
        result = camber_sweep(_corner("FL", 1.0), steps=20)
        assert len(result.wheel_travel_mm) == 20
        assert len(result.camber_deg) == 20
        assert len(result.converged) == 20

    def test_sweep_all_converged(self) -> None:
        """All points in ±25mm range should converge."""
        result = camber_sweep(_corner("FL", 1.0), steps=20)
        assert all(result.converged)

    def test_sweep_corner_id(self) -> None:
        result = camber_sweep(_corner("FL", 1.0), steps=10)
        assert result.corner_id == "FL"

    def test_sweep_covers_range(self) -> None:
        result = camber_sweep(_corner("FL", 1.0), wheel_travel_min_mm=-20.0, wheel_travel_max_mm=20.0, steps=10)
        assert result.wheel_travel_mm[0] == pytest.approx(-20.0)
        assert result.wheel_travel_mm[-1] == pytest.approx(20.0)

    def test_sweep_camber_varies(self) -> None:
        """Camber should not be constant across the sweep."""
        result = camber_sweep(_corner("FL", 1.0), steps=20)
        camber_vals = [c for c, conv in zip(result.camber_deg, result.converged) if conv]
        assert max(camber_vals) > min(camber_vals)
