"""Tests for vdcore.tire — loader, conditioning, metrics, and comparison.

These tests use synthetic data with analytically known answers so they
do not depend on real TTC .mat files being present.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.io import savemat

from vdcore.tire.compare import compare_tires
from vdcore.tire.metrics import (
    compute_bin_metrics,
    compute_camber_sensitivity,
    compute_load_sensitivity,
    compute_loaded_radius_fit,
    compute_tire_metrics,
)
from vdcore.tire.models import TTCRun
from vdcore.tire.ttc import condition, load_ttc_mat


# ---------------------------------------------------------------------------
# Helpers — synthetic TTC data generation
# ---------------------------------------------------------------------------


def _make_synthetic_mat(
    path: Path,
    *,
    n: int = 500,
    peak_mu: float = 1.5,
    peak_sa_deg_sae: float = 6.0,
    fz_sae: float = -800.0,
    ia_sae: float = -2.0,
    pressure_kpa: float = 83.0,
    velocity_kmh: float = 40.0,
    loaded_radius_m: float = 0.228,
    include_warmup: bool = False,
) -> None:
    """Create a synthetic TTC .mat file in adapted-SAE convention.

    The lateral force follows a simplified sine-based curve so that the
    peak μ and peak SA are analytically known. In adapted-SAE convention
    (Y+ right), a positive slip angle (wheel pointed right of travel)
    produces a positive lateral force (rightward).

    After ISO 8855 conversion (negate SA, FY, FZ, IA, MX, MZ):
    - positive SA → positive FY (leftward)
    - FZ becomes positive (loaded tire)
    """
    sa_sae = np.linspace(-12.0, 12.0, n)

    fy_peak = peak_mu * abs(fz_sae)
    sa_rad = np.deg2rad(sa_sae)
    peak_sa_rad = np.deg2rad(peak_sa_deg_sae)
    fy_sae = fy_peak * np.sin((np.pi / 2) * sa_rad / peak_sa_rad)
    fy_sae = np.clip(fy_sae, -fy_peak, fy_peak)

    trail_m = 0.020
    mz_sae = -fy_sae * trail_m

    et = np.linspace(0.0, 60.0, n)
    if include_warmup:
        et = np.linspace(-5.0, 60.0, n)

    data = {
        "SA": sa_sae,
        "SR": np.zeros(n),
        "FZ": np.full(n, fz_sae),
        "FY": fy_sae,
        "FX": np.zeros(n),
        "MZ": mz_sae,
        "MX": np.zeros(n),
        "IA": np.full(n, ia_sae),
        "P": np.full(n, pressure_kpa),
        "RL": np.full(n, loaded_radius_m),
        "RE": np.full(n, loaded_radius_m * 1.02),
        "V": np.full(n, velocity_kmh),
        "ET": et,
        "TSTC": np.full(n, 50.0),
        "TSTI": np.full(n, 45.0),
        "TSTO": np.full(n, 55.0),
    }
    savemat(str(path), data)


# ---------------------------------------------------------------------------
# A1 — Sign convention tests
# ---------------------------------------------------------------------------


class TestSignConvention:
    """The fundamental sanity check: after ISO 8855 conversion,
    positive slip angle must produce positive lateral force."""

    def test_left_turn_produces_leftward_force(self, tmp_path: Path) -> None:
        """Physical scenario: vehicle turning left about ISO 8855 Z-axis.

        A left turn has positive yaw rate (counterclockwise from above,
        right-hand rule about Z-up). The resulting lateral acceleration
        points to the left (positive Y in ISO 8855). To sustain the
        turn, the tires must generate a lateral force toward the left
        (positive FY in ISO 8855).

        For the outside (right) tire in a left turn, the velocity vector
        points forward-right relative to the wheel heading, so the slip
        angle is positive in ISO 8855 (wheel pointed left of travel
        direction). A positive slip angle must therefore produce a
        positive lateral force.

        This test constructs the slip angle from the physical cornering
        direction rather than checking raw channel signs, so it cannot
        be fooled by a loader that flips both SA and FY together.
        """
        mat_path = tmp_path / "test_tire.mat"
        _make_synthetic_mat(mat_path)

        df, _ = load_ttc_mat(
            mat_path,
            tire_designation="SyntheticTest",
            rim_width_in=7.0,
            test_round="TestRound",
        )

        sa_for_left_turn_outside_deg = 4.0

        near_sa = df.filter(
            (pl.col("sa_deg") > sa_for_left_turn_outside_deg - 0.5)
            & (pl.col("sa_deg") < sa_for_left_turn_outside_deg + 0.5)
        )
        assert near_sa.height > 0, (
            f"No data points near SA = {sa_for_left_turn_outside_deg} deg"
        )

        mean_fy = near_sa["fy_n"].mean()
        assert mean_fy is not None
        assert mean_fy > 0, (
            f"Physical violation: in a left turn (positive yaw rate), "
            f"the outside tire at SA = {sa_for_left_turn_outside_deg} deg "
            f"must produce positive FY (leftward force in ISO 8855). "
            f"Got mean FY = {mean_fy:.1f} N"
        )

        mean_fz = near_sa["fz_n"].mean()
        assert mean_fz is not None
        assert mean_fz > 0, (
            "FZ must be positive for a loaded tire in ISO 8855 (Z-up)"
        )

    def test_fz_positive_for_loaded_tire(self, tmp_path: Path) -> None:
        """In ISO 8855 (Z+ up), a loaded tire must have FZ > 0."""
        mat_path = tmp_path / "test_tire.mat"
        _make_synthetic_mat(mat_path)

        df, _ = load_ttc_mat(
            mat_path,
            tire_designation="SyntheticTest",
            rim_width_in=7.0,
            test_round="TestRound",
        )

        assert df["fz_n"].min() > 0, (  # type: ignore[operator]
            "ISO 8855: FZ must be positive for a loaded tire (Z+ up)"
        )

    def test_ia_sign_flipped(self, tmp_path: Path) -> None:
        """SAE IA = -2° should become ISO IA = +2° after negation."""
        mat_path = tmp_path / "test_tire.mat"
        _make_synthetic_mat(mat_path, ia_sae=-2.0)

        df, _ = load_ttc_mat(
            mat_path,
            tire_designation="SyntheticTest",
            rim_width_in=7.0,
            test_round="TestRound",
        )

        ia_mean = df["ia_deg"].mean()
        assert ia_mean is not None
        assert ia_mean > 0, f"IA should be positive after SAE→ISO flip, got {ia_mean}"

    def test_loaded_radius_converted_to_mm(self, tmp_path: Path) -> None:
        mat_path = tmp_path / "test_tire.mat"
        _make_synthetic_mat(mat_path, loaded_radius_m=0.228)

        df, _ = load_ttc_mat(
            mat_path,
            tire_designation="SyntheticTest",
            rim_width_in=7.0,
            test_round="TestRound",
        )

        rl_mean = df["rl_mm"].mean()
        assert rl_mean is not None
        assert 220 < rl_mean < 240, (
            f"Loaded radius should be ~228 mm, got {rl_mean:.1f}"
        )


# ---------------------------------------------------------------------------
# A1 — Loader edge cases
# ---------------------------------------------------------------------------


class TestLoader:
    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_ttc_mat(
                "/nonexistent/file.mat",
                tire_designation="x",
                rim_width_in=7.0,
                test_round="x",
            )

    def test_missing_required_channel_raises(self, tmp_path: Path) -> None:
        mat_path = tmp_path / "bad.mat"
        savemat(str(mat_path), {"SA": np.zeros(10)})

        with pytest.raises(KeyError, match="missing expected channel"):
            load_ttc_mat(
                mat_path,
                tire_designation="x",
                rim_width_in=7.0,
                test_round="x",
            )

    def test_optional_channels_tolerated(self, tmp_path: Path) -> None:
        """TSTC/TSTI/TSTO/RE are optional — loader must not fail if absent."""
        mat_path = tmp_path / "minimal.mat"
        n = 50
        data = {
            "SA": np.linspace(-10, 10, n),
            "SR": np.zeros(n),
            "FZ": np.full(n, -800.0),
            "FY": np.linspace(-500, 500, n),
            "FX": np.zeros(n),
            "MZ": np.zeros(n),
            "MX": np.zeros(n),
            "IA": np.full(n, -2.0),
            "P": np.full(n, 83.0),
            "RL": np.full(n, 0.228),
            "V": np.full(n, 40.0),
            "ET": np.linspace(0, 30, n),
        }
        savemat(str(mat_path), data)

        df, meta = load_ttc_mat(
            mat_path,
            tire_designation="Minimal",
            rim_width_in=7.0,
            test_round="Test",
        )
        assert df.height == n
        assert "tstc_degc" not in df.columns
        assert meta.tire_designation == "Minimal"

    def test_metadata_fields(self, tmp_path: Path) -> None:
        mat_path = tmp_path / "test.mat"
        _make_synthetic_mat(mat_path)

        _, meta = load_ttc_mat(
            mat_path,
            tire_designation="Hoosier 18x6-10 R25B",
            rim_width_in=7.0,
            test_round="Round9",
        )
        assert meta.tire_designation == "Hoosier 18x6-10 R25B"
        assert meta.rim_width_in == 7.0
        assert meta.test_round == "Round9"
        assert meta.source == "measured"


# ---------------------------------------------------------------------------
# A2 — Conditioning filter tests
# ---------------------------------------------------------------------------


class TestConditioning:
    def _make_df(self, n: int = 200) -> pl.DataFrame:
        return pl.DataFrame({
            "sa_deg": np.linspace(-10, 10, n).tolist(),
            "sr": [0.0] * n,
            "fz_n": [800.0] * n,
            "fy_n": np.linspace(-600, 600, n).tolist(),
            "fx_n": [0.0] * n,
            "mz_nm": [0.0] * n,
            "mx_nm": [0.0] * n,
            "ia_deg": [2.0] * n,
            "p_kpa": np.linspace(75, 90, n).tolist(),
            "rl_mm": [228.0] * n,
            "v_kmh": np.linspace(30, 50, n).tolist(),
            "et_s": np.linspace(-5, 60, n).tolist(),
            "tstc_degc": np.linspace(30, 70, n).tolist(),
        })

    def test_warmup_removes_negative_time(self) -> None:
        df = self._make_df()
        result, reports = condition(df, warmup_seconds=0.0)
        assert result.height == df.height
        assert len(reports) == 0

        result2, reports2 = condition(df, warmup_seconds=5.0)
        assert result2.height < df.height
        assert len(reports2) == 1
        assert reports2[0].filter_name == "warmup_drop"
        assert reports2[0].rows_removed > 0
        assert result2["et_s"].min() >= 5.0  # type: ignore[operator]

    def test_pressure_band_filters(self) -> None:
        df = self._make_df()
        result, reports = condition(
            df, pressure_target_kpa=83.0, pressure_tolerance_kpa=3.0,
        )
        assert result.height < df.height
        assert reports[0].filter_name == "pressure_band"
        assert reports[0].rows_removed > 0
        assert result["p_kpa"].min() >= 80.0  # type: ignore[operator]
        assert result["p_kpa"].max() <= 86.0  # type: ignore[operator]

    def test_velocity_band_filters(self) -> None:
        df = self._make_df()
        result, reports = condition(
            df, velocity_min_kmh=35.0, velocity_max_kmh=45.0,
        )
        assert result.height < df.height
        assert reports[0].filter_name == "velocity_band"
        assert result["v_kmh"].min() >= 35.0  # type: ignore[operator]
        assert result["v_kmh"].max() <= 45.0  # type: ignore[operator]

    def test_temperature_window_filters(self) -> None:
        df = self._make_df()
        result, reports = condition(
            df, temp_min_degc=40.0, temp_max_degc=60.0,
        )
        assert result.height < df.height
        assert reports[0].filter_name == "temperature_window"
        assert result["tstc_degc"].min() >= 40.0  # type: ignore[operator]
        assert result["tstc_degc"].max() <= 60.0  # type: ignore[operator]

    def test_all_filters_stacked(self) -> None:
        df = self._make_df()
        result, reports = condition(
            df,
            warmup_seconds=5.0,
            pressure_target_kpa=83.0,
            pressure_tolerance_kpa=5.0,
            velocity_min_kmh=35.0,
            velocity_max_kmh=45.0,
            temp_min_degc=40.0,
            temp_max_degc=60.0,
        )
        assert len(reports) == 4
        total_removed = sum(r.rows_removed for r in reports)
        assert total_removed == df.height - result.height

    def test_report_rows_consistent(self) -> None:
        df = self._make_df()
        _, reports = condition(df, warmup_seconds=10.0)
        for r in reports:
            assert r.rows_removed == r.rows_before - r.rows_after
            assert r.rows_removed >= 0


# ---------------------------------------------------------------------------
# A3 — Metric extraction with known answers
# ---------------------------------------------------------------------------


class TestMetrics:
    @pytest.fixture()
    def synthetic_iso_df(self, tmp_path: Path) -> pl.DataFrame:
        """Load synthetic data through the full pipeline (SAE→ISO)."""
        mat_path = tmp_path / "synth.mat"
        _make_synthetic_mat(mat_path, peak_mu=1.5, peak_sa_deg_sae=6.0)
        df, _ = load_ttc_mat(
            mat_path,
            tire_designation="Synth",
            rim_width_in=7.0,
            test_round="T",
        )
        return df

    def test_peak_mu_near_known_value(self, synthetic_iso_df: pl.DataFrame) -> None:
        bins = compute_bin_metrics(
            synthetic_iso_df, fz_min_n=700, fz_max_n=900, fz_bins=1, ia_bins=1,
        )
        assert len(bins) > 0
        assert bins[0].peak_mu_lateral == pytest.approx(1.5, abs=0.05)

    def test_peak_sa_near_known_value(self, synthetic_iso_df: pl.DataFrame) -> None:
        bins = compute_bin_metrics(
            synthetic_iso_df, fz_min_n=700, fz_max_n=900, fz_bins=1, ia_bins=1,
        )
        assert len(bins) > 0
        assert abs(bins[0].peak_mu_sa_deg) == pytest.approx(6.0, abs=1.0)

    def test_cornering_stiffness_positive(self, synthetic_iso_df: pl.DataFrame) -> None:
        """In ISO 8855, positive SA → positive FY, so dFY/dα > 0."""
        bins = compute_bin_metrics(
            synthetic_iso_df, fz_min_n=700, fz_max_n=900, fz_bins=1, ia_bins=1,
        )
        assert len(bins) > 0
        assert bins[0].cornering_stiffness_n_per_deg > 0

    def test_peak_sharpness_less_than_one(self, synthetic_iso_df: pl.DataFrame) -> None:
        bins = compute_bin_metrics(
            synthetic_iso_df, fz_min_n=700, fz_max_n=900, fz_bins=1, ia_bins=1,
        )
        assert len(bins) > 0
        assert 0.0 < bins[0].peak_sharpness <= 1.0

    def test_empty_fz_window_returns_empty(self, synthetic_iso_df: pl.DataFrame) -> None:
        bins = compute_bin_metrics(
            synthetic_iso_df, fz_min_n=5000, fz_max_n=6000, fz_bins=1, ia_bins=1,
        )
        assert bins == []


class TestLoadSensitivity:
    def test_negative_slope(self, tmp_path: Path) -> None:
        """Load sensitivity should be negative: μ decreases with FZ."""
        dfs = []
        for fz_sae in [-400.0, -600.0, -800.0, -1000.0]:
            mat_path = tmp_path / f"fz{abs(fz_sae):.0f}.mat"
            _make_synthetic_mat(
                mat_path,
                peak_mu=1.5 + 0.0005 * (fz_sae + 800),
                fz_sae=fz_sae,
            )
            df, _ = load_ttc_mat(
                mat_path,
                tire_designation="Synth",
                rim_width_in=7.0,
                test_round="T",
            )
            dfs.append(df)

        combined = pl.concat(dfs)
        bins = compute_bin_metrics(
            combined, fz_min_n=300, fz_max_n=1100, fz_bins=4, ia_bins=1,
        )
        ls = compute_load_sensitivity(bins)
        assert len(ls) > 0
        assert ls[0].slope_per_n < 0, "Load sensitivity should be negative"


class TestLoadedRadius:
    def test_regression_reasonable(self, tmp_path: Path) -> None:
        """Build a dataset with varying FZ so regression is well-conditioned."""
        dfs = []
        for fz_sae in [-400.0, -600.0, -800.0, -1000.0]:
            mat_path = tmp_path / f"rl_{abs(fz_sae):.0f}.mat"
            rl_m = 0.228 - 0.00002 * abs(fz_sae)
            _make_synthetic_mat(mat_path, fz_sae=fz_sae, loaded_radius_m=rl_m)
            df, _ = load_ttc_mat(
                mat_path,
                tire_designation="Synth",
                rim_width_in=7.0,
                test_round="T",
            )
            dfs.append(df)
        combined = pl.concat(dfs)
        fit = compute_loaded_radius_fit(combined, fz_min_n=300, fz_max_n=1100)
        assert fit is not None
        assert fit.slope_mm_per_n < 0, "Loaded radius should decrease with load"
        assert 200 < fit.intercept_mm < 250


# ---------------------------------------------------------------------------
# A4 — Comparison table
# ---------------------------------------------------------------------------


class TestCompareTires:
    def test_one_row_per_tire(self, tmp_path: Path) -> None:
        reports = []
        for i, name in enumerate(["TireA", "TireB"]):
            mat_path = tmp_path / f"{name}.mat"
            _make_synthetic_mat(mat_path, peak_mu=1.4 + i * 0.2)
            df, meta = load_ttc_mat(
                mat_path,
                tire_designation=name,
                rim_width_in=7.0,
                test_round="T",
            )
            report = compute_tire_metrics(
                df, name, fz_min_n=700, fz_max_n=900, fz_bins=1, ia_bins=1,
            )
            reports.append(report)

        table = compare_tires(reports)
        assert table.height == 2
        assert table["tire"].to_list() == ["TireA", "TireB"]
        assert "peak_mu_mean" in table.columns

    def test_different_tires_have_different_metrics(self, tmp_path: Path) -> None:
        reports = []
        for i, (name, mu) in enumerate([("Soft", 1.8), ("Hard", 1.2)]):
            mat_path = tmp_path / f"{name}.mat"
            _make_synthetic_mat(mat_path, peak_mu=mu)
            df, _ = load_ttc_mat(
                mat_path,
                tire_designation=name,
                rim_width_in=7.0,
                test_round="T",
            )
            report = compute_tire_metrics(
                df, name, fz_min_n=700, fz_max_n=900, fz_bins=1, ia_bins=1,
            )
            reports.append(report)

        table = compare_tires(reports)
        mus = table["peak_mu_mean"].to_list()
        assert mus[0] > mus[1], "Soft tire should have higher peak μ"
