"""Tests for the balance trade table.

All tests use synthetic BinMetrics with known values.
"""

from __future__ import annotations

import pytest

from vdcore.analysis.balance_targets import balance_trade_table
from vdcore.models.mass import (
    MassProperties,
    ProvenanceFloat,
    UnsprungMass,
    UnsprungMassSet,
)
from vdcore.tire.metrics import BinMetrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pf(value: float, tol: float, source: str = "estimate") -> ProvenanceFloat:
    return ProvenanceFloat(value=value, source=source, tol=tol)  # type: ignore[arg-type]


def _symmetric_mass() -> MassProperties:
    """Symmetric car: fmf = 0.5."""
    return MassProperties(
        total_mass_kg=_pf(300.0, 2.0),
        driver_mass_kg=_pf(75.0, 1.0),
        cg_height_mm=_pf(300.0, 20.0),
        cg_x_mm=_pf(775.0, 10.0),
        front_mass_fraction=_pf(0.5, 0.01),
        yaw_inertia_kgm2=_pf(108.0, 30.0),
        roll_inertia_kgm2=_pf(17.0, 5.0),
    )


def _unsprung() -> UnsprungMassSet:
    corner = UnsprungMass(
        mass_kg=_pf(15.0, 1.0),
        cg_height_mm=_pf(254.0, 5.0),
    )
    return UnsprungMassSet(fl=corner, fr=corner, rl=corner, rr=corner)


def _symmetric_bins() -> list[BinMetrics]:
    """Bins with load sensitivity (same tire for all corners)."""
    fz_levels = [400.0, 800.0, 1200.0, 1600.0]
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
        for fz in fz_levels
    ]


_BALANCE_KW = dict(
    ay_g=1.0,
    front_rc_height_mm=50.0,
    rear_rc_height_mm=50.0,
    front_track_mm=1220.0,
    rear_track_mm=1220.0,
    wheelbase_mm=1550.0,
    total_roll_stiffness_nm_per_deg=850.0,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBalanceTradeTable:
    def test_symmetric_lltd_05_neutral(self) -> None:
        """Symmetric car + symmetric tires + LLTD = 0.5 → neutral balance.

        Equal LLTD with equal mass distribution means equal load transfer
        front and rear.  With the same tires, force capacity is equal.
        """
        df = balance_trade_table(
            _symmetric_mass(),
            _unsprung(),
            _symmetric_bins(),
            lltd_range=(0.5, 0.5),
            lltd_steps=1,
            **_BALANCE_KW,
        )

        row = df.row(0, named=True)
        assert row["balance"] == "neutral", (
            f"Expected neutral at LLTD=0.5 on symmetric car, "
            f"got {row['balance']} (front={row['front_fy_n']:.1f}, "
            f"rear={row['rear_fy_n']:.1f})"
        )

    def test_higher_front_lltd_makes_front_limited(self) -> None:
        """Increasing front LLTD beyond 0.5 degrades front grip more than rear.

        More load transfer at the front means the outside front tire is
        loaded more heavily (lower mu due to load sensitivity), while the
        inside front is unloaded more.  Net: front force capacity drops
        relative to rear.
        """
        df = balance_trade_table(
            _symmetric_mass(),
            _unsprung(),
            _symmetric_bins(),
            lltd_range=(0.6, 0.7),
            lltd_steps=3,
            **_BALANCE_KW,
        )

        for row_dict in df.iter_rows(named=True):
            assert row_dict["balance"] == "front_limited", (
                f"Expected front_limited at LLTD={row_dict['lltd']:.2f}, "
                f"got {row_dict['balance']}"
            )

    def test_lltd_maps_to_stiffness_split(self) -> None:
        """front_roll_stiffness_fraction should match the LLTD input."""
        df = balance_trade_table(
            _symmetric_mass(),
            _unsprung(),
            _symmetric_bins(),
            lltd_range=(0.3, 0.7),
            lltd_steps=5,
            **_BALANCE_KW,
        )

        for row_dict in df.iter_rows(named=True):
            assert row_dict["front_roll_stiffness_fraction"] == pytest.approx(
                row_dict["lltd"], abs=1e-10,
            )

    def test_correct_row_count(self) -> None:
        df = balance_trade_table(
            _symmetric_mass(),
            _unsprung(),
            _symmetric_bins(),
            lltd_range=(0.3, 0.7),
            lltd_steps=11,
            **_BALANCE_KW,
        )
        assert df.height == 11

    def test_columns_present(self) -> None:
        df = balance_trade_table(
            _symmetric_mass(),
            _unsprung(),
            _symmetric_bins(),
            lltd_range=(0.4, 0.6),
            lltd_steps=3,
            **_BALANCE_KW,
        )
        expected_cols = {
            "lltd",
            "front_roll_stiffness_nm_per_deg",
            "rear_roll_stiffness_nm_per_deg",
            "front_fy_n",
            "rear_fy_n",
            "balance",
            "margin_n",
            "margin_pct",
            "front_roll_stiffness_fraction",
            "fz_was_clamped",
        }
        assert expected_cols == set(df.columns)

    def test_margin_is_non_negative(self) -> None:
        df = balance_trade_table(
            _symmetric_mass(),
            _unsprung(),
            _symmetric_bins(),
            lltd_range=(0.3, 0.7),
            lltd_steps=11,
            **_BALANCE_KW,
        )
        for m in df["margin_n"].to_list():
            assert m >= 0.0
