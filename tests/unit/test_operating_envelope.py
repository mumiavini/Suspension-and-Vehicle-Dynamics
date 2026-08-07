"""Tests for operating envelope: mu_at_fz, corner_loads, achievable_ay.

All tests use synthetic BinMetrics fixtures with analytically known values.
No dependency on real .mat files.

Fixture values are UNMEASURED ESTIMATES — they exercise the code, not the car.
"""

from __future__ import annotations

import pytest

from vdcore.analysis.operating_envelope import (
    CornerLoadResult,
    VehicleCornerLoads,
    achievable_ay,
    corner_loads_at_ay,
    mu_at_fz,
)
from vdcore.explain import Explained
from vdcore.models.mass import (
    MassProperties,
    ProvenanceFloat,
    UnsprungMass,
    UnsprungMassSet,
)
from vdcore.tire.metrics import BinMetrics

_G = 9.81


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _pf(value: float, tol: float, source: str = "estimate") -> ProvenanceFloat:
    return ProvenanceFloat(value=value, source=source, tol=tol)  # type: ignore[arg-type]


def _mass_props(cg_h: float = 300.0) -> MassProperties:
    return MassProperties(
        total_mass_kg=_pf(300.0, 2.0),
        driver_mass_kg=_pf(75.0, 1.0),
        cg_height_mm=_pf(cg_h, 20.0),
        cg_x_mm=_pf(800.0, 10.0),
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


def _bins_with_load_sensitivity(base_mu: float = 1.5) -> list[BinMetrics]:
    """Create bins at 4 FZ levels with negative load sensitivity.

    mu decreases linearly from base_mu at FZ=400 to base_mu-0.4 at FZ=1600.
    Slope = -0.4/1200 ≈ -3.33e-4 per N.
    """
    fz_levels = [400.0, 800.0, 1200.0, 1600.0]
    slope = -0.4 / 1200.0
    return [
        BinMetrics(
            fz_nominal_n=fz,
            ia_nominal_deg=0.0,
            p_nominal_kpa=83.0,
            n_points=100,
            peak_mu_lateral=base_mu + slope * (fz - 400.0),
            peak_mu_sa_deg=8.0,
            cornering_stiffness_n_per_deg=50.0,
            cs_regression_window_deg=2.0,
            cs_r_squared=0.99,
            peak_sharpness=0.95,
            pneumatic_trail_at_peak_mm=20.0,
        )
        for fz in fz_levels
    ]


def _bins_constant_mu(mu: float = 1.5) -> list[BinMetrics]:
    """Create bins with zero load sensitivity (constant mu across all FZ)."""
    fz_levels = [400.0, 800.0, 1200.0, 1600.0]
    return [
        BinMetrics(
            fz_nominal_n=fz,
            ia_nominal_deg=0.0,
            p_nominal_kpa=83.0,
            n_points=100,
            peak_mu_lateral=mu,
            peak_mu_sa_deg=8.0,
            cornering_stiffness_n_per_deg=50.0,
            cs_regression_window_deg=2.0,
            cs_r_squared=0.99,
            peak_sharpness=0.95,
            pneumatic_trail_at_peak_mm=20.0,
        )
        for fz in fz_levels
    ]


_ENVELOPE_KW: dict[str, float] = dict(
    front_rc_height_mm=50.0,
    rear_rc_height_mm=60.0,
    front_track_mm=1220.0,
    rear_track_mm=1220.0,
    front_roll_stiffness_nm_per_deg=500.0,
    rear_roll_stiffness_nm_per_deg=350.0,
    wheelbase_mm=1550.0,
)


# ---------------------------------------------------------------------------
# mu_at_fz tests
# ---------------------------------------------------------------------------


class TestMuAtFz:
    def test_interpolation(self) -> None:
        """Midpoint between two bins should give linear average of peak_mu."""
        bins = _bins_with_load_sensitivity(base_mu=1.5)
        # At FZ=600 (midpoint of 400-800), mu should be average of bins[0] and bins[1].
        mu_400 = bins[0].peak_mu_lateral
        mu_800 = bins[1].peak_mu_lateral
        expected = (mu_400 + mu_800) / 2.0

        mu, clamped = mu_at_fz(600.0, bins)
        assert mu == pytest.approx(expected, rel=1e-6)
        assert not clamped

    def test_exact_bin_value(self) -> None:
        bins = _bins_with_load_sensitivity()
        mu, clamped = mu_at_fz(800.0, bins)
        assert mu == pytest.approx(bins[1].peak_mu_lateral)
        assert not clamped

    def test_clamp_below(self) -> None:
        bins = _bins_with_load_sensitivity()
        mu, clamped = mu_at_fz(100.0, bins)
        assert mu == bins[0].peak_mu_lateral
        assert clamped

    def test_clamp_above(self) -> None:
        bins = _bins_with_load_sensitivity()
        mu, clamped = mu_at_fz(2000.0, bins)
        assert mu == bins[-1].peak_mu_lateral
        assert clamped

    def test_zero_fz_returns_zero(self) -> None:
        bins = _bins_with_load_sensitivity()
        mu, clamped = mu_at_fz(0.0, bins)
        assert mu == 0.0
        assert clamped

    def test_negative_fz_returns_zero(self) -> None:
        bins = _bins_with_load_sensitivity()
        mu, clamped = mu_at_fz(-100.0, bins)
        assert mu == 0.0
        assert clamped

    def test_empty_bins_raises(self) -> None:
        with pytest.raises(ValueError, match="No bin metrics"):
            mu_at_fz(800.0, [])

    def test_filter_by_ia(self) -> None:
        bins = _bins_with_load_sensitivity()
        # All bins have ia=0.0 — filtering for ia=2.0 should raise.
        with pytest.raises(ValueError, match="No bin metrics"):
            mu_at_fz(800.0, bins, ia_nominal_deg=2.0)


# ---------------------------------------------------------------------------
# Corner loads tests
# ---------------------------------------------------------------------------


class TestCornerLoads:
    def test_sum_to_total_weight(self) -> None:
        """Four corner loads must sum to m * g (conservation)."""
        mass = _mass_props()
        loads = corner_loads_at_ay(
            mass, _unsprung(), _bins_with_load_sensitivity(),
            ay_g=1.5, **_ENVELOPE_KW,
        )
        total_fz = sum(c.total_fz_n for c in loads.corners())
        expected = mass.total_mass_kg.value * _G
        assert total_fz == pytest.approx(expected, rel=0.01)

    def test_zero_ay_symmetric(self) -> None:
        """At zero Ay, all four corners should have equal load (symmetric car)."""
        mass = _mass_props()  # fmf = 0.5
        loads = corner_loads_at_ay(
            mass, _unsprung(), _bins_with_load_sensitivity(),
            ay_g=0.0, **_ENVELOPE_KW,
        )
        fzs = [c.total_fz_n for c in loads.corners()]
        for fz in fzs:
            assert fz == pytest.approx(fzs[0], rel=1e-6)

    def test_positive_ay_loads_right(self) -> None:
        """Positive Ay (leftward) loads the right side."""
        loads = corner_loads_at_ay(
            _mass_props(), _unsprung(), _bins_with_load_sensitivity(),
            ay_g=1.0, **_ENVELOPE_KW,
        )
        assert loads.fr.total_fz_n > loads.fl.total_fz_n
        assert loads.rr.total_fz_n > loads.rl.total_fz_n


# ---------------------------------------------------------------------------
# Achievable Ay tests
# ---------------------------------------------------------------------------


class TestAchievableAy:
    def test_converges(self) -> None:
        result = achievable_ay(
            _mass_props(), _unsprung(), _bins_with_load_sensitivity(),
            **_ENVELOPE_KW,
        )
        assert isinstance(result, Explained)
        assert result.value > 0
        assert result.intermediates["iterations"] > 0
        assert result.intermediates["residual"] < 1e-4

    def test_increasing_mu_increases_ay(self) -> None:
        """Higher peak_mu must produce higher achievable Ay. Monotonicity."""
        ay_low = achievable_ay(
            _mass_props(), _unsprung(), _bins_with_load_sensitivity(base_mu=1.2),
            **_ENVELOPE_KW,
        )
        ay_high = achievable_ay(
            _mass_props(), _unsprung(), _bins_with_load_sensitivity(base_mu=1.8),
            **_ENVELOPE_KW,
        )
        assert ay_high.value > ay_low.value

    def test_zero_load_sensitivity_ay_independent_of_cg_height(self) -> None:
        """With constant mu across all FZ bins, CG height does not affect grip.

        This is the key test: load sensitivity is what makes CG height
        matter for total grip.  Without it, load transfer redistributes
        Fz but doesn't degrade the total lateral force (mu * Fz summed
        over all corners remains constant), provided no wheel fully
        unloads (Fz clamped to 0 breaks the sum).

        We use moderate CG heights (250/350 mm) so load transfer at the
        converged Ay does not unload any wheel.
        """
        bins = _bins_constant_mu(mu=1.5)

        ay_low_cg = achievable_ay(
            _mass_props(cg_h=250.0), _unsprung(), bins,
            **_ENVELOPE_KW,
        )
        ay_high_cg = achievable_ay(
            _mass_props(cg_h=350.0), _unsprung(), bins,
            **_ENVELOPE_KW,
        )

        assert ay_low_cg.value == pytest.approx(ay_high_cg.value, rel=0.02), (
            f"With zero load sensitivity, Ay should be independent of CG height. "
            f"Got {ay_low_cg.value:.4f} g at 250 mm vs {ay_high_cg.value:.4f} g at 350 mm."
        )

    def test_has_assumptions(self) -> None:
        result = achievable_ay(
            _mass_props(), _unsprung(), _bins_with_load_sensitivity(),
            **_ENVELOPE_KW,
        )
        assert len(result.assumptions) >= 1

    def test_flags_estimate_inputs(self) -> None:
        result = achievable_ay(
            _mass_props(), _unsprung(), _bins_with_load_sensitivity(),
            **_ENVELOPE_KW,
        )
        assert result.has_estimate_inputs()
        assert "cg_height_mm" in result.estimate_input_names()

    def test_formula_variables_in_inputs(self) -> None:
        import re

        result = achievable_ay(
            _mass_props(), _unsprung(), _bins_with_load_sensitivity(),
            **_ENVELOPE_KW,
        )
        all_tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", result.formula))
        non_vars = {"sum", "min", "max", "abs"}
        formula_vars = all_tokens - non_vars
        lhs = result.formula.split("=")[0].strip()
        formula_vars.discard(lhs)
        missing = formula_vars - set(result.inputs.keys())
        assert not missing, f"Formula vars not in inputs: {missing}"
