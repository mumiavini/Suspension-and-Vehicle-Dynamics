"""The Streamlit app and the published summary must quote the same numbers.

WHY THIS FILE EXISTS
    On 2026-09-01 the app showed 6.91 % anti-dive while the PDF handed to the
    chassis team said 7.50 %. Nothing was wrong with either solver: the app
    defaulted its brake-bias slider to 0.60 against a design value of 0.65, and
    the two constants had no link. A design document and the tool the team
    reads it next to disagreeing by 8 % is worse than either being slightly
    wrong, because it destroys trust in both.

    The same audit found the Synthesis tab's seed KPIs still running on the
    legacy strut-to-midpoint solver, which reported the rear bump steer with
    the SIGN INVERTED (+0.00155 deg/mm against a true -0.00015) -- as a
    reference for choosing optimisation targets, that is actively misleading.

WHAT THIS PINS
    Not values. Values belong in the benchmarks. This pins the AGREEMENT: the
    app's defaults track the design config, and the app's dynamic KPI path
    returns what geometry_summary publishes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[2]
for path in (REPO, REPO / "legacy_app", REPO / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sla_geometry as sla  # noqa: E402
from analysis.vdcore_bridge import (  # noqa: E402
    CornerInputs,
    compute_delegated_dynamic_kpis,
)
from ui.shared import DESIGN_BRAKE_BIAS_FRONT  # noqa: E402
from vdcore.analysis.toe import bump_steer  # noqa: E402

MERGED_CSV = REPO / "Geometry Summary" / "hardpoints_2027_merged.csv"


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return pl.read_csv(str(MERGED_CSV))


@pytest.fixture(scope="module")
def delegated(df: pl.DataFrame) -> dict[str, dict[str, float]]:
    return compute_delegated_dynamic_kpis(
        df, CornerInputs(static_camber_deg=-1.5, loaded_radius_mm=245.0)
    )


def test_app_brake_bias_default_matches_the_design_config() -> None:
    """The app's anti-dive default must track sla_geometry's brake bias.

    legacy_app/ cannot import sla_geometry (it is frozen and must not gain a
    dependency on the design scripts), so the constant is duplicated. This is
    the mechanism that stops the duplicate from drifting.
    """
    assert DESIGN_BRAKE_BIAS_FRONT == pytest.approx(
        sla.VEHICLE_2027.brake_bias_front
    ), (
        "legacy_app/ui/shared.py DESIGN_BRAKE_BIAS_FRONT has drifted from "
        "sla_geometry.VEHICLE_2027.brake_bias_front -- the app and the "
        "published summary would report different anti-dive"
    )


def test_sidebar_default_uses_the_shared_constant() -> None:
    """No second hard-coded copy of the bias may creep back into the sidebar."""
    src = (REPO / "legacy_app" / "ui" / "sidebar.py").read_text(encoding="utf-8")
    assert '"brake_bias": DESIGN_BRAKE_BIAS_FRONT' in src
    assert '"brake_bias": 0.6' not in src


def test_app_anti_dive_matches_the_published_value(df: pl.DataFrame) -> None:
    """At the design bias, the app's anti-dive equals the summary's.

    The app computes it through the LEGACY corner method, which builds the
    side-view IC from the pivot midpoint rather than the ball joints. That is
    still not a construction to trust in general -- it agrees here only because
    the front lower axis is horizontal -- so the tolerance is loose enough to
    admit the known 0.017 pp offset and tight enough to catch a bias mismatch
    (which would be worth 0.6 pp).
    """
    from analysis.io_hardpoints import build_vehicle_from_dataframe

    vehicle, _ = build_vehicle_from_dataframe(df)
    app_value = vehicle.front_left.anti_dive_percent(
        brake_bias=DESIGN_BRAKE_BIAS_FRONT,
        wheelbase_mm=vehicle.wheelbase_mm,
        cg_height_mm=sla.VEHICLE_2027.cg_height_mm,
    )
    published = sla.solve_axle(sla.FRONT_2027, sla.VEHICLE_2027).anti_percent

    assert app_value == pytest.approx(published, abs=0.05), (
        f"app shows {app_value:.3f} %, summary publishes {published:.3f} %"
    )


class TestBumpSteerIsDelegatedToVdcore:
    """The bridge must expose bump steer, and it must match the summary."""

    @pytest.mark.parametrize("axle", ["front", "rear"])
    def test_bridge_exposes_both_numbers(
        self, delegated: dict[str, dict[str, float]], axle: str
    ) -> None:
        row = delegated[axle]
        assert "bump_steer" in row, "linear rate missing from the delegated KPIs"
        assert "bump_steer_peak" in row, "peak missing -- the rate alone misleads"

    @pytest.mark.parametrize(
        ("axle", "corner"), [("front", "FL"), ("rear", "RL")]
    )
    def test_bridge_matches_vdcore_directly(
        self, delegated: dict[str, dict[str, float]],
        df: pl.DataFrame, axle: str, corner: str,
    ) -> None:
        """What the app displays is what vdcore.analysis.toe computes."""
        from analysis.vdcore_bridge import df_to_vdcore_corner

        direct = bump_steer(
            df_to_vdcore_corner(
                df, corner,
                CornerInputs(static_camber_deg=-1.5, loaded_radius_mm=245.0),
            ),
            wheel_travel_range_mm=25.0,
            steps=41,
        )
        assert delegated[axle]["bump_steer"] == pytest.approx(
            direct.linear_deg_per_mm_per_side, abs=1e-9
        )
        assert delegated[axle]["bump_steer_peak"] == pytest.approx(
            direct.peak_abs_deg_per_side, abs=1e-9
        )

    def test_rear_sign_is_not_the_legacy_one(
        self, delegated: dict[str, dict[str, float]]
    ) -> None:
        """Guard the specific regression this replaced.

        The legacy sweep gave the rear +0.00155 deg/mm; the truth is negative.
        A positive rear rate here means the legacy path came back.
        """
        assert delegated["rear"]["bump_steer"] < 0.0

    def test_front_peak_is_not_implied_by_the_linear_rate(
        self, delegated: dict[str, dict[str, float]]
    ) -> None:
        """The reason two numbers are reported instead of one."""
        front = delegated["front"]
        assert abs(front["bump_steer"]) < 0.0005, "linear rate is ~nulled"
        assert front["bump_steer_peak"] > 0.1, "yet the peak is large"


def test_synthesis_tab_no_longer_uses_the_legacy_sweep() -> None:
    """Seed KPIs drive target selection, so they must be the correct ones."""
    src = (REPO / "legacy_app" / "ui" / "tab_synthesis.py").read_text(
        encoding="utf-8"
    )
    assert "run_sweep_cached" not in src, (
        "tab_synthesis is back on the legacy strut-to-midpoint sweep"
    )
    assert "vdcore_sweep" in src
