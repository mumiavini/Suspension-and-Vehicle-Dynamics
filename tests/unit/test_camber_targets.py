"""Tests for the camber trade surface.

All tests use synthetic BinMetrics/CamberSensitivity with known values.
"""

from __future__ import annotations

import pytest

from vdcore.analysis.camber_targets import camber_trade_surface
from vdcore.tire.metrics import BinMetrics, CamberSensitivity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bins() -> list[BinMetrics]:
    return [
        BinMetrics(
            fz_nominal_n=fz,
            ia_nominal_deg=0.0,
            p_nominal_kpa=83.0,
            n_points=100,
            peak_mu_lateral=1.5 - 0.0003 * (fz - 400.0),
            peak_mu_sa_deg=8.0,
            cornering_stiffness_n_per_deg=50.0,
            cs_regression_window_deg=2.0,
            cs_r_squared=0.99,
            peak_sharpness=0.95,
            pneumatic_trail_at_peak_mm=20.0,
        )
        for fz in [400.0, 800.0, 1200.0, 1600.0]
    ]


def _camber_sens() -> list[CamberSensitivity]:
    """Camber sensitivity: 30 N/deg at FZ=800."""
    return [
        CamberSensitivity(
            dfy_dia_n_per_deg=30.0,
            r_squared=0.95,
            at_sa_deg=8.0,
            fz_nominal_n=800.0,
            p_nominal_kpa=83.0,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCamberTradeSurface:
    def test_higher_roll_gradient_requires_more_camber_gain(self) -> None:
        """Higher roll gradient means more body roll → more camber change
        → more camber gain needed to compensate.  Monotonically.
        """
        df = camber_trade_surface(
            _bins(),
            _camber_sens(),
            roll_gradient_range_deg_per_g=(1.0, 5.0),
            roll_gradient_steps=5,
            static_camber_range_deg=(-2.0, -2.0),
            static_camber_steps=1,
            target_ay_g=1.5,
            front_track_mm=1220.0,
            tire_best_ia_deg=-2.0,
        )

        gains = df["required_camber_gain_deg_per_mm"].to_list()
        # Camber gain should be negative (bump produces more negative camber)
        # and its absolute value should increase with roll gradient.
        abs_gains = [abs(g) for g in gains]
        for i in range(len(abs_gains) - 1):
            assert abs_gains[i + 1] >= abs_gains[i], (
                f"Camber gain should increase with roll gradient, "
                f"but got {abs_gains[i]:.4f} then {abs_gains[i+1]:.4f}"
            )

    def test_zero_roll_no_gain_needed(self) -> None:
        """At zero roll gradient with static camber = tire's best IA,
        no camber gain is needed.
        """
        df = camber_trade_surface(
            _bins(),
            _camber_sens(),
            roll_gradient_range_deg_per_g=(0.0, 0.0),
            roll_gradient_steps=1,
            static_camber_range_deg=(-2.0, -2.0),
            static_camber_steps=1,
            target_ay_g=1.5,
            front_track_mm=1220.0,
            tire_best_ia_deg=-2.0,
        )

        gain = df["required_camber_gain_deg_per_mm"][0]
        assert gain == pytest.approx(0.0, abs=1e-6)

    def test_fy_penalty_no_gain_non_negative(self) -> None:
        """Penalty of doing nothing should be >= 0 (can't gain from wrong IA)."""
        df = camber_trade_surface(
            _bins(),
            _camber_sens(),
            roll_gradient_range_deg_per_g=(1.0, 8.0),
            roll_gradient_steps=8,
            static_camber_range_deg=(-3.0, 0.0),
            static_camber_steps=4,
            target_ay_g=1.5,
            front_track_mm=1220.0,
            tire_best_ia_deg=-2.0,
        )

        penalties = df["fy_penalty_no_gain_pct"].to_list()
        for p in penalties:
            assert p >= -1e-10, f"Penalty should be non-negative, got {p}"

    def test_correct_row_count(self) -> None:
        df = camber_trade_surface(
            _bins(),
            _camber_sens(),
            roll_gradient_range_deg_per_g=(1.0, 5.0),
            roll_gradient_steps=5,
            static_camber_range_deg=(-3.0, 0.0),
            static_camber_steps=4,
            target_ay_g=1.5,
            front_track_mm=1220.0,
            tire_best_ia_deg=-2.0,
        )
        assert df.height == 5 * 4

    def test_columns_present(self) -> None:
        df = camber_trade_surface(
            _bins(),
            _camber_sens(),
            roll_gradient_range_deg_per_g=(2.0, 4.0),
            roll_gradient_steps=3,
            static_camber_range_deg=(-2.0, -1.0),
            static_camber_steps=2,
            target_ay_g=1.5,
            front_track_mm=1220.0,
            tire_best_ia_deg=-2.0,
        )
        expected_cols = {
            "roll_gradient_deg_per_g",
            "static_camber_deg",
            "roll_angle_at_target_ay_deg",
            "outside_ia_no_gain_deg",
            "fy_penalty_no_gain_pct",
            "required_camber_gain_deg_per_mm",
            "implied_fvsa_mm",
            "implied_rc_height_mm",
            "rc_height_sensitivity_mm_per_mm_bump",
            "fy_at_best_ia_n",
        }
        assert expected_cols == set(df.columns)
