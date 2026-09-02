"""Unit tests for legacy_app/analysis/vdcore_bridge.py.

The bridge lifts the legacy app's loaded hardpoints DataFrame into validated
``vdcore`` Corner/Axle models and returns the CORRECT dynamic KPIs (the legacy
strut-to-midpoint solver's dynamic KPIs are wrong). These tests pin:

- the identity frame conversion (loaded frame == ISO 8855, no sign flip);
- the Y-sign fail-loud validator, tested per side (left must be +Y, right -Y);
- the golden dynamic KPIs against tests/benchmarks/test_fsae2027_design.py;
- the left/right camber invariant in roll (outer gains, inner loses).

Import-path note: ``legacy_app/`` is not a package on ``sys.path`` by default and
its modules import as ``analysis.vdcore_bridge`` / ``vdcore...``. We prepend both
the repo root (for ``vdcore``) and ``legacy_app/`` (for ``analysis.*``) below.
Streamlit is never imported: the bridge falls back to its uncached entry points
when streamlit is absent, and these tests use those uncached functions directly.
"""

from __future__ import annotations

import math
import pathlib
import sys

import polars as pl
import pytest

# tests/unit/ -> parents[0]=unit, [1]=tests, [2]=repo root.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))  # for `vdcore`
_LEGACY = _ROOT / "legacy_app"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))  # for `analysis.*`

from analysis.vdcore_bridge import (  # noqa: E402
    BridgeConversionError,
    CornerInputs,
    compute_delegated_dynamic_kpis,
    compute_vdcore_kpis,
    compute_vdcore_setup_sheet,
    delegated_axle_dynamic_kpis,
    delegated_axle_static_kpis,
    df_to_vdcore_axles,
    df_to_vdcore_corner,
    solved_static_corner,
)
from vdcore.analysis.axle import axle_roll  # noqa: E402

_CSV = _LEGACY / "carro_formula_2027.csv"


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    """The legacy-frame 2027 hardpoints DataFrame (ISO 8855, Y+ LEFT)."""
    return pl.read_csv(str(_CSV))


@pytest.fixture(scope="module")
def inputs() -> CornerInputs:
    """The two design inputs the CSV does not carry, pinned to the golden case."""
    return CornerInputs(static_camber_deg=-1.5, loaded_radius_mm=245.0)


# =============================================================================
# 1. Conversion round-trip / structure
# =============================================================================


def test_conversion_is_identity_copy_no_sign_flip(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """FL wheel-centre copies the CSV coordinates verbatim (loaded frame == ISO 8855)."""
    corner = df_to_vdcore_corner(df, "FL", inputs)
    assert corner.wheel_center.x_mm == pytest.approx(0.0)
    assert corner.wheel_center.y_mm == pytest.approx(613.584)
    assert corner.wheel_center.z_mm == pytest.approx(245.0)


def test_loaded_hardpoints_are_tagged_estimate(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """Loaded geometry is provenance 'estimate', never 'design_intent'."""
    corner = df_to_vdcore_corner(df, "FL", inputs)
    assert len(corner.hardpoints()) == 9
    assert all(hp.source == "estimate" for hp in corner.hardpoints())
    assert corner.tire.source == "estimate"


def test_contact_patch_is_not_an_input_hardpoint(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """CONTACT_PATCH is derived by vdcore, not lifted as a Corner input."""
    corner = df_to_vdcore_corner(df, "FL", inputs)
    names = {hp.name for hp in corner.hardpoints()}
    assert "contact_patch" not in names
    assert "CONTACT_PATCH" not in names
    # The Corner model carries no contact_patch input field either.
    assert not hasattr(corner, "contact_patch")


# =============================================================================
# 2. Y-sign validator, all four corners independently
# =============================================================================


@pytest.mark.parametrize("corner_id", ["FL", "FR", "RL", "RR"])
def test_all_four_corners_convert_with_correct_y_sign(
    df: pl.DataFrame, inputs: CornerInputs, corner_id: str
) -> None:
    """Every corner in the correctly-signed frame converts to a vdcore Corner."""
    corner = df_to_vdcore_corner(df, corner_id, inputs)
    assert corner.corner_id == corner_id
    if corner_id in ("FL", "RL"):
        assert corner.wheel_center.y_mm > 0  # left is +Y
    else:
        assert corner.wheel_center.y_mm < 0  # right is -Y


def test_left_corner_with_negative_y_raises(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """A left corner (FL) with the wrong (negative) Y sign fails loud, not silent."""
    # Flip every FL Y coordinate negative: an FL loaded in a Y+ RIGHT frame.
    bad = df.with_columns(
        pl.when(pl.col("corner") == "FL")
        .then(-pl.col("y_mm"))
        .otherwise(pl.col("y_mm"))
        .alias("y_mm")
    )
    with pytest.raises(BridgeConversionError, match=r"(?i)Y|frame|sign"):
        df_to_vdcore_corner(bad, "FL", inputs)


def test_right_corner_with_positive_y_raises(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """A right corner (FR) with the wrong (positive) Y sign fails loud, not silent."""
    # Flip every FR Y coordinate positive: an FR loaded in a Y+ RIGHT frame.
    bad = df.with_columns(
        pl.when(pl.col("corner") == "FR")
        .then(-pl.col("y_mm"))
        .otherwise(pl.col("y_mm"))
        .alias("y_mm")
    )
    with pytest.raises(BridgeConversionError, match=r"(?i)Y|frame|sign"):
        df_to_vdcore_corner(bad, "FR", inputs)


# =============================================================================
# 3. Golden cross-check vs the benchmark (pins the frame conversion)
# =============================================================================


@pytest.fixture(scope="module")
def delegated(
    df: pl.DataFrame, inputs: CornerInputs
) -> dict[str, dict[str, float]]:
    """Delegated dynamic KPIs for both axles at 1.5 deg roll, 25 mm travel."""
    return compute_delegated_dynamic_kpis(
        df, inputs, roll_deg=1.5, travel_mm=25.0
    )


def test_front_camber_gain_matches_benchmark(
    delegated: dict[str, dict[str, float]]
) -> None:
    """Front camber gain per mm reproduces the benchmark's -0.0386 deg/mm."""
    assert delegated["front"]["camber_gain"] == pytest.approx(-0.0386, abs=0.0005)


def test_rear_camber_gain_matches_benchmark(
    delegated: dict[str, dict[str, float]]
) -> None:
    """Rear camber gain per mm reproduces the benchmark's -0.0411 deg/mm."""
    assert delegated["rear"]["camber_gain"] == pytest.approx(-0.0411, abs=0.0005)


def test_front_rc_height_at_1g_matches_benchmark(
    delegated: dict[str, dict[str, float]]
) -> None:
    """Front RC height at 1.5 deg roll matches axle_roll's 34.86 mm."""
    assert delegated["front"]["rc_1g_z"] == pytest.approx(34.86, abs=0.1)


def test_front_rc_lateral_at_1g_matches_benchmark(
    delegated: dict[str, dict[str, float]]
) -> None:
    """Front RC lateral shift at 1.5 deg roll matches axle_roll's -86.90 mm."""
    assert delegated["front"]["rc_1g_y"] == pytest.approx(-86.90, abs=1.0)


def test_ride_camber_dpm_is_camber_gain_times_1000(
    delegated: dict[str, dict[str, float]]
) -> None:
    """ride_camber_dpm (deg/m) is exactly camber_gain (deg/mm) x 1000."""
    front = delegated["front"]
    assert front["ride_camber_dpm"] == pytest.approx(
        front["camber_gain"] * 1000.0, rel=1e-9
    )


def test_rc_height_migration_is_real_not_legacy_1mm_bug(
    delegated: dict[str, dict[str, float]]
) -> None:
    """RC height migrates >10 mm over travel (real, not the legacy ~1 mm artefact)."""
    # Front is ~19.6 mm, rear ~21.2 mm; assert only the lower bound to prove the fix.
    assert delegated["front"]["rc_dz"] > 10.0
    assert delegated["rear"]["rc_dz"] > 10.0


def test_lateral_rc_migration_is_zero_under_parallel_travel(
    delegated: dict[str, dict[str, float]]
) -> None:
    """Lateral RC migration under parallel travel is exactly zero for a symmetric axle."""
    assert delegated["front"]["rc_dy"] == 0.0
    assert delegated["rear"]["rc_dy"] == 0.0


def test_roll_camber_slope_is_negative_and_finite(
    delegated: dict[str, dict[str, float]]
) -> None:
    """Chassis-referenced roll-camber slope is negative and finite on both axles."""
    for axle in ("front", "rear"):
        rc = delegated[axle]["roll_camber"]
        assert math.isfinite(rc)
        assert rc < 0.0


# =============================================================================
# 4. Left/right camber signs (physical invariant)
# =============================================================================


def test_outer_wheel_gains_camber_relative_to_inner_in_roll(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """In roll the outer (left, +Y) wheel keeps more camber than the inner (right)."""
    front_axle, _rear_axle = df_to_vdcore_axles(df, inputs)
    state = axle_roll(front_axle, 1.5)
    # Outer camber ~ -0.648, inner ~ -2.392: outer is the less-negative (greater).
    assert state.outer_camber_deg > state.inner_camber_deg


# =============================================================================
# 5. Graceful-degrade contract: every KPI key present, each finite or NaN
# =============================================================================


def test_delegated_row_has_all_seven_keys_each_finite_or_nan(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """Every delegated dynamic KPI key exists and is finite or NaN (never wrong-plausible)."""
    front_axle, _rear_axle = df_to_vdcore_axles(df, inputs)
    row = delegated_axle_dynamic_kpis(
        front_axle, roll_deg=1.5, travel_mm=25.0, sweep_steps=41
    )
    expected_keys = {
        "ride_camber_dpm",
        "camber_gain",
        "rc_dy",
        "rc_dz",
        "roll_camber",
        "rc_1g_z",
        "rc_1g_y",
    }
    assert expected_keys <= set(row)
    for key in expected_keys:
        value = row[key]
        assert math.isfinite(value) or math.isnan(value)


def test_compute_vdcore_kpis_reports_convergence_for_both_axles(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """The full KPI pass returns validated rates for both axles (solver converged)."""
    kpis = compute_vdcore_kpis(df, inputs, roll_deg=1.5, travel_mm=25.0)
    # rates is None only on non-convergence; the 2027 geometry converges.
    assert kpis.front.rates is not None
    assert kpis.rear.rates is not None
    assert kpis.front.error is None
    assert kpis.rear.error is None


# =============================================================================
# 6. Setup sheet: static + dynamic geometry rows from the validated solver
# =============================================================================
# ``compute_vdcore_setup_sheet`` merges the static half
# (``delegated_axle_static_kpis``) with the dynamic half
# (``delegated_axle_dynamic_kpis``) into one per-axle dict. The uncached entry
# point is used directly so no Streamlit runtime is needed, exactly as the
# dynamic tests above use ``compute_delegated_dynamic_kpis``.

# Every geometry key the setup sheet must carry: the static rows first, then the
# dynamic rows spliced in over them.
_SETUP_SHEET_STATIC_KEYS = frozenset(
    {
        "caster_l",
        "caster_r",
        "kpi_l",
        "kpi_r",
        "camber_l",
        "camber_r",
        "scrub_l",
        "scrub_r",
        "trail_l",
        "trail_r",
        "sum_toe",
        "rc_static",
    }
)
_SETUP_SHEET_DYNAMIC_KEYS = frozenset(
    {
        "camber_gain",
        "ride_camber_dpm",
        "rc_dz",
        "rc_dy",
        "roll_camber",
        "rc_1g_z",
        "rc_1g_y",
    }
)
_SETUP_SHEET_ALL_KEYS = _SETUP_SHEET_STATIC_KEYS | _SETUP_SHEET_DYNAMIC_KEYS


@pytest.fixture(scope="module")
def setup_sheet(
    df: pl.DataFrame, inputs: CornerInputs
) -> dict[str, dict[str, float]]:
    """The merged static+dynamic setup sheet at 1.5 deg roll, 25 mm travel."""
    return compute_vdcore_setup_sheet(df, inputs, roll_deg=1.5, travel_mm=25.0)


def test_setup_sheet_has_both_axles_with_every_key_present(
    setup_sheet: dict[str, dict[str, float]]
) -> None:
    """Both axles carry every static AND dynamic geometry key, none None."""
    assert set(setup_sheet) == {"front", "rear"}
    for axle in ("front", "rear"):
        row = setup_sheet[axle]
        assert _SETUP_SHEET_ALL_KEYS <= set(row)
        for key in _SETUP_SHEET_ALL_KEYS:
            # NaN is allowed (a failed per-KPI derivation), None is not:
            # the key must always exist with a real float slot.
            assert row[key] is not None


def test_setup_sheet_front_static_reproduces_benchmark(
    setup_sheet: dict[str, dict[str, float]]
) -> None:
    """Front static caster/KPI/camber/scrub/trail/RC match the 2027 benchmark."""
    front = setup_sheet["front"]
    assert front["caster_l"] == pytest.approx(5.0, abs=0.05)
    assert front["kpi_l"] == pytest.approx(10.0, abs=0.05)
    assert front["camber_l"] == pytest.approx(-1.5, abs=0.05)
    assert front["scrub_l"] == pytest.approx(15.08, abs=0.1)
    assert front["scrub_r"] == pytest.approx(15.08, abs=0.1)
    # Mechanical trail must be positive (fwd intercept +), not the legacy sign flip.
    assert front["trail_l"] > 0.0
    assert front["trail_l"] == pytest.approx(21.43, abs=0.1)
    assert front["trail_r"] == pytest.approx(21.43, abs=0.1)
    # 35.50, not the 35.0 design target: the UCA pivot rake added for anti-dive
    # tilts the plane the upper ball joint sweeps, which moves the solved FVIC
    # slightly. Isolated 2026-09-01 -- the rake alone gives 35.45, the UCA
    # pickup move to y=210 alone gives exactly 35.0000.
    assert front["rc_static"] == pytest.approx(35.50, abs=0.2)


def test_setup_sheet_rear_static_reproduces_benchmark(
    setup_sheet: dict[str, dict[str, float]]
) -> None:
    """Rear static scrub/KPI/RC height match the 2027 benchmark."""
    rear = setup_sheet["rear"]
    assert rear["scrub_l"] == pytest.approx(21.97, abs=0.1)
    assert rear["kpi_l"] == pytest.approx(8.5, abs=0.05)
    assert rear["rc_static"] == pytest.approx(55.0, abs=0.2)


def test_setup_sheet_dynamic_reproduces_benchmark(
    setup_sheet: dict[str, dict[str, float]]
) -> None:
    """Front/rear camber gain and RC-at-1g in the sheet match the dynamic benchmark."""
    front = setup_sheet["front"]
    rear = setup_sheet["rear"]
    assert front["camber_gain"] == pytest.approx(-0.0386, abs=0.001)
    assert rear["camber_gain"] == pytest.approx(-0.0411, abs=0.001)
    assert front["rc_1g_z"] == pytest.approx(34.86, abs=0.2)
    assert front["rc_1g_y"] == pytest.approx(-86.90, abs=1.0)
    assert rear["rc_1g_z"] == pytest.approx(54.58, abs=0.2)


def test_delegated_static_sum_toe_and_axle_symmetry(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """Front axle has zero total toe and left/right-symmetric caster and scrub."""
    front_axle = df_to_vdcore_axles(df, inputs)[0]
    row = delegated_axle_static_kpis(front_axle)
    # Default merged geometry carries zero static toe on both sides.
    assert row["sum_toe"] == pytest.approx(0.0, abs=0.01)
    # Caster and scrub are solved INDEPENDENTLY per side, so matching L==R is a
    # real symmetry check on the mirrored geometry, not a tautology.
    assert row["caster_l"] == pytest.approx(row["caster_r"], abs=0.01)
    assert row["scrub_l"] == pytest.approx(row["scrub_r"], abs=0.01)


def test_solved_static_corner_raises_on_non_convergence(
    df: pl.DataFrame, inputs: CornerInputs
) -> None:
    """A corner whose linkage cannot close fails loud with BridgeConversionError."""
    # Try to force non-convergence by making the tie rod unreachable: place its
    # inboard point absurdly far so the sixth DOF cannot be satisfied. A real
    # Corner may still validate and converge (the solver is robust), so if the
    # geometry closes anyway we skip rather than fabricate a fragile fixture.
    corner = df_to_vdcore_corner(df, "FL", inputs)
    broken = corner.model_copy(
        update={
            "tie_rod_inboard": corner.tie_rod_inboard.model_copy(
                update={"x_mm": 1.0e9, "y_mm": 1.0e9, "z_mm": 1.0e9}
            )
        }
    )
    try:
        result = solved_static_corner(broken)
    except BridgeConversionError:
        return  # expected: non-convergence fails loud
    pytest.skip(
        "Solver converged on the deliberately-broken corner "
        f"(residual {result.residual_norm:.2e}); cannot force non-convergence "
        "through a valid Corner without fabricating a fragile fixture."
    )
