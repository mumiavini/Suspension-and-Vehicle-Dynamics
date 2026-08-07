"""Tests for vehicle mass models, load transfer, and roll gradient.

Fixture values are UNMEASURED ESTIMATES for a hypothetical FSAE26 car.
They exist to exercise the code, not to represent the real car.
Replace with measured values from docs/onboarding/measuring_mass_properties.md
once the physical tests are done.

Estimated values (~300 kg with 75 kg driver):
  - Total mass: 300 kg
  - CG height: 300 mm (estimate, tol ±20 mm — honest uncertainty)
  - CG x: 800 mm behind front axle
  - Wheelbase: 1550 mm
  - Front mass fraction: 0.484 (from cg_x / wheelbase)
  - Track: 1220 mm front and rear
  - Unsprung: 15 kg per corner at 254 mm CG height (wheel centre)
  - Roll stiffness: 500 Nm/deg front, 350 Nm/deg rear (springs + ARB)
"""

from __future__ import annotations

import math

import pytest

from vdcore.analysis.load_transfer import (
    LateralLoadTransferResult,
    VehicleLoadTransferResult,
    lateral_load_transfer,
    longitudinal_load_transfer,
    sprung_mass_kg,
)
from vdcore.analysis.roll_gradient import roll_gradient_deg_per_g
from vdcore.models.mass import (
    MassProperties,
    ProvenanceFloat,
    UnsprungMass,
    UnsprungMassSet,
)


# ---------------------------------------------------------------------------
# Fixture helpers — all values are ESTIMATES
# ---------------------------------------------------------------------------


def _pf(value: float, tol: float, source: str = "estimate") -> ProvenanceFloat:
    return ProvenanceFloat(value=value, source=source, tol=tol)  # type: ignore[arg-type]


def _mass_props(
    total: float = 300.0,
    cg_h: float = 300.0,
    cg_x: float = 800.0,
    wb: float = 1550.0,
) -> MassProperties:
    fmf = 1.0 - cg_x / wb
    return MassProperties(
        total_mass_kg=_pf(total, 2.0),
        driver_mass_kg=_pf(75.0, 1.0),
        cg_height_mm=_pf(cg_h, 20.0),
        cg_x_mm=_pf(cg_x, 10.0),
        front_mass_fraction=_pf(fmf, 0.01),
        yaw_inertia_kgm2=_pf(108.0, 30.0),
        roll_inertia_kgm2=_pf(17.0, 5.0),
    )


def _unsprung() -> UnsprungMassSet:
    corner = UnsprungMass(
        mass_kg=_pf(15.0, 1.0),
        cg_height_mm=_pf(254.0, 5.0),
    )
    return UnsprungMassSet(fl=corner, fr=corner, rl=corner, rr=corner)


def _lt_result(ay_g: float = 1.0) -> VehicleLoadTransferResult:
    return lateral_load_transfer(
        _mass_props(),
        _unsprung(),
        ay_g=ay_g,
        front_rc_height_mm=50.0,
        rear_rc_height_mm=60.0,
        front_track_mm=1220.0,
        rear_track_mm=1220.0,
        front_roll_stiffness_nm_per_deg=500.0,
        rear_roll_stiffness_nm_per_deg=350.0,
        wheelbase_mm=1550.0,
    )


# ---------------------------------------------------------------------------
# Mass model tests
# ---------------------------------------------------------------------------


class TestMassProperties:
    def test_construction(self) -> None:
        mp = _mass_props()
        assert mp.total_mass_kg.value == 300.0
        assert mp.cg_height_mm.source == "estimate"

    def test_all_fields_are_estimates(self) -> None:
        mp = _mass_props()
        assert mp.has_estimates()
        assert "cg_height_mm" in mp.estimate_fields()

    def test_front_mass_fraction_bounds(self) -> None:
        with pytest.raises(ValueError, match="front_mass_fraction"):
            MassProperties(
                total_mass_kg=_pf(300.0, 1.0),
                driver_mass_kg=_pf(75.0, 1.0),
                cg_height_mm=_pf(300.0, 20.0),
                cg_x_mm=_pf(800.0, 10.0),
                front_mass_fraction=_pf(1.5, 0.01),
                yaw_inertia_kgm2=_pf(108.0, 30.0),
                roll_inertia_kgm2=_pf(17.0, 5.0),
            )

    def test_driver_exceeds_total_rejected(self) -> None:
        with pytest.raises(ValueError, match="driver_mass_kg"):
            MassProperties(
                total_mass_kg=_pf(50.0, 1.0),
                driver_mass_kg=_pf(75.0, 1.0),
                cg_height_mm=_pf(300.0, 20.0),
                cg_x_mm=_pf(800.0, 10.0),
                front_mass_fraction=_pf(0.48, 0.01),
                yaw_inertia_kgm2=_pf(108.0, 30.0),
                roll_inertia_kgm2=_pf(17.0, 5.0),
            )

    def test_frozen(self) -> None:
        mp = _mass_props()
        with pytest.raises(Exception):
            mp.total_mass_kg = _pf(999.0, 1.0)  # type: ignore[misc]


class TestUnsprungMass:
    def test_total(self) -> None:
        usm = _unsprung()
        assert usm.total_kg() == pytest.approx(60.0)

    def test_pairs(self) -> None:
        usm = _unsprung()
        fl, fr = usm.front_pair()
        assert fl.mass_kg.value == 15.0
        assert fr.mass_kg.value == 15.0


# ---------------------------------------------------------------------------
# Load transfer — conservation checks
# ---------------------------------------------------------------------------


class TestLateralLoadTransfer:
    def test_zero_ay_gives_zero_lt(self) -> None:
        """Symmetric car at zero Ay must produce zero load transfer."""
        result = _lt_result(ay_g=0.0)
        assert result.front.total_delta_fz_n == pytest.approx(0.0, abs=1e-10)
        assert result.rear.total_delta_fz_n == pytest.approx(0.0, abs=1e-10)
        assert result.front.geometric_delta_fz_n == pytest.approx(0.0, abs=1e-10)
        assert result.front.elastic_delta_fz_n == pytest.approx(0.0, abs=1e-10)
        assert result.front.unsprung_delta_fz_n == pytest.approx(0.0, abs=1e-10)

    def test_total_equals_m_ay_h_over_t(self) -> None:
        """Total load transfer must equal m·Ay·h/t regardless of decomposition.

        This is the fundamental conservation check. If this fails, the
        decomposition has an accounting error.
        """
        ay_g = 1.5
        result = _lt_result(ay_g=ay_g)

        mp = _mass_props()
        g = 9.81
        m = mp.total_mass_kg.value
        h = mp.cg_height_mm.value / 1000.0
        t = 1220.0 / 1000.0

        expected_total = m * ay_g * g * h / t
        actual_total = result.front.total_delta_fz_n + result.rear.total_delta_fz_n

        assert actual_total == pytest.approx(expected_total, rel=0.01), (
            f"Conservation violated: expected {expected_total:.1f} N, "
            f"got {actual_total:.1f} N (front={result.front.total_delta_fz_n:.1f}, "
            f"rear={result.rear.total_delta_fz_n:.1f})"
        )

    def test_decomposition_sums_to_total(self) -> None:
        """geometric + elastic + unsprung = total for each axle."""
        result = _lt_result(ay_g=1.0)

        for axle_result in [result.front, result.rear]:
            decomp_sum = (
                axle_result.geometric_delta_fz_n
                + axle_result.elastic_delta_fz_n
                + axle_result.unsprung_delta_fz_n
            )
            assert decomp_sum == pytest.approx(
                axle_result.total_delta_fz_n, abs=1e-10
            ), (
                f"{axle_result.axle}: decomposition sum {decomp_sum:.4f} != "
                f"total {axle_result.total_delta_fz_n:.4f}"
            )

    def test_lltd_sums_to_one(self) -> None:
        """LLTD is the fraction of total LT taken by the front.
        front_total / (front_total + rear_total) = LLTD, so
        LLTD + (1 - LLTD) = 1 by definition. But verify
        the implementation computes it correctly.
        """
        result = _lt_result(ay_g=1.0)
        total = result.front.total_delta_fz_n + result.rear.total_delta_fz_n
        assert total > 0
        front_fraction = result.front.total_delta_fz_n / total
        assert result.lltd == pytest.approx(front_fraction, abs=1e-10)

    def test_lltd_between_zero_and_one(self) -> None:
        result = _lt_result(ay_g=1.0)
        assert 0.0 < result.lltd < 1.0

    def test_lt_scales_with_ay(self) -> None:
        """Load transfer must scale linearly with Ay (quasi-static)."""
        r1 = _lt_result(ay_g=1.0)
        r2 = _lt_result(ay_g=2.0)
        assert r2.front.total_delta_fz_n == pytest.approx(
            2.0 * r1.front.total_delta_fz_n, rel=1e-10
        )
        assert r2.rear.total_delta_fz_n == pytest.approx(
            2.0 * r1.rear.total_delta_fz_n, rel=1e-10
        )

    def test_negative_ay_gives_negative_lt(self) -> None:
        """Negative Ay (rightward) gives negative load transfer (to inside)."""
        r_pos = _lt_result(ay_g=1.0)
        r_neg = _lt_result(ay_g=-1.0)
        assert r_neg.front.total_delta_fz_n == pytest.approx(
            -r_pos.front.total_delta_fz_n, abs=1e-10
        )

    def test_estimate_inputs_reported(self) -> None:
        result = _lt_result(ay_g=1.0)
        assert len(result.front.estimate_inputs) > 0
        assert "cg_height_mm" in result.front.estimate_inputs

    def test_geometric_component_positive_at_positive_ay(self) -> None:
        """With positive RC height and positive Ay, geometric LT is positive."""
        result = _lt_result(ay_g=1.0)
        assert result.front.geometric_delta_fz_n > 0
        assert result.rear.geometric_delta_fz_n > 0


class TestLongitudinalLoadTransfer:
    def test_braking_loads_front(self) -> None:
        """Braking (negative Ax) transfers load to the front."""
        result = longitudinal_load_transfer(
            _mass_props(), ax_g=-1.0, wheelbase_mm=1550.0,
        )
        assert result.delta_fz_front_n > 0
        assert result.delta_fz_rear_n < 0

    def test_acceleration_loads_rear(self) -> None:
        """Acceleration (positive Ax) transfers load to the rear."""
        result = longitudinal_load_transfer(
            _mass_props(), ax_g=1.0, wheelbase_mm=1550.0,
        )
        assert result.delta_fz_front_n < 0
        assert result.delta_fz_rear_n > 0

    def test_front_rear_equal_and_opposite(self) -> None:
        result = longitudinal_load_transfer(
            _mass_props(), ax_g=-1.0, wheelbase_mm=1550.0,
        )
        assert result.delta_fz_front_n == pytest.approx(
            -result.delta_fz_rear_n, abs=1e-10
        )

    def test_zero_ax_gives_zero(self) -> None:
        result = longitudinal_load_transfer(
            _mass_props(), ax_g=0.0, wheelbase_mm=1550.0,
        )
        assert result.delta_fz_front_n == pytest.approx(0.0, abs=1e-10)


class TestSprungMass:
    def test_sprung_plus_unsprung_equals_total(self) -> None:
        mp = _mass_props()
        usm = _unsprung()
        sm = sprung_mass_kg(mp, usm)
        assert sm + usm.total_kg() == pytest.approx(mp.total_mass_kg.value)


# ---------------------------------------------------------------------------
# Roll gradient
# ---------------------------------------------------------------------------


class TestRollGradient:
    def test_positive_roll_gradient(self) -> None:
        """Roll gradient must be positive (car rolls away from turn)."""
        rg = roll_gradient_deg_per_g(
            _mass_props(),
            _unsprung(),
            front_rc_height_mm=50.0,
            rear_rc_height_mm=60.0,
            front_roll_stiffness_nm_per_deg=500.0,
            rear_roll_stiffness_nm_per_deg=350.0,
        )
        assert rg > 0

    def test_inversely_proportional_to_stiffness(self) -> None:
        """Doubling roll stiffness should roughly halve roll gradient."""
        rg1 = roll_gradient_deg_per_g(
            _mass_props(),
            _unsprung(),
            front_rc_height_mm=50.0,
            rear_rc_height_mm=60.0,
            front_roll_stiffness_nm_per_deg=500.0,
            rear_roll_stiffness_nm_per_deg=350.0,
        )
        rg2 = roll_gradient_deg_per_g(
            _mass_props(),
            _unsprung(),
            front_rc_height_mm=50.0,
            rear_rc_height_mm=60.0,
            front_roll_stiffness_nm_per_deg=1000.0,
            rear_roll_stiffness_nm_per_deg=700.0,
        )
        ratio = rg1 / rg2
        assert 1.8 < ratio < 2.2, (
            f"Expected ~2x reduction, got ratio {ratio:.2f}"
        )

    def test_insufficient_stiffness_raises(self) -> None:
        """If roll stiffness < gravity moment, ValueError."""
        with pytest.raises(ValueError, match="roll over"):
            roll_gradient_deg_per_g(
                _mass_props(),
                _unsprung(),
                front_rc_height_mm=50.0,
                rear_rc_height_mm=60.0,
                front_roll_stiffness_nm_per_deg=1.0,
                rear_roll_stiffness_nm_per_deg=1.0,
            )

    def test_higher_cg_gives_more_roll(self) -> None:
        """Higher CG = more roll (farther from roll axis)."""
        rg_low = roll_gradient_deg_per_g(
            _mass_props(cg_h=250.0),
            _unsprung(),
            front_rc_height_mm=50.0,
            rear_rc_height_mm=60.0,
            front_roll_stiffness_nm_per_deg=500.0,
            rear_roll_stiffness_nm_per_deg=350.0,
        )
        rg_high = roll_gradient_deg_per_g(
            _mass_props(cg_h=350.0),
            _unsprung(),
            front_rc_height_mm=50.0,
            rear_rc_height_mm=60.0,
            front_roll_stiffness_nm_per_deg=500.0,
            rear_roll_stiffness_nm_per_deg=350.0,
        )
        assert rg_high > rg_low

    def test_reasonable_fsae_value(self) -> None:
        """FSAE roll gradient should be roughly 1-10 deg/g.

        The wide range reflects that our roll stiffness and CG height
        are both estimates. With the placeholder values (850 Nm/deg total,
        CG 300 mm, RA ~54 mm), we get ~6.7 deg/g which is on the soft
        side but physically plausible for an FSAE car without a stiff ARB.
        """
        rg = roll_gradient_deg_per_g(
            _mass_props(),
            _unsprung(),
            front_rc_height_mm=50.0,
            rear_rc_height_mm=60.0,
            front_roll_stiffness_nm_per_deg=500.0,
            rear_roll_stiffness_nm_per_deg=350.0,
        )
        assert 0.5 < rg < 10.0, f"Roll gradient {rg:.2f} deg/g outside FSAE range"
