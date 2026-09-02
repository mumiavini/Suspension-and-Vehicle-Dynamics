"""Unit tests for vdcore.analysis.toe.

The rear axle had no bump-steer number anywhere until 2026-09-01:
``steering_geometry.py`` owns bump steer but covers the front axle only, so the
rear toe link's effect on toe was simply uncomputed. This module closed that
gap; these tests pin its behaviour to physics rather than to stored output.

Physics that must hold for ANY geometry lives here. Values specific to the
FSAE2027 design are pinned in tests/benchmarks/test_fsae2027_design.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from vdcore.analysis.toe import bump_steer, toe_sweep  # noqa: E402
from vdcore.geometry.solver import DWSolver  # noqa: E402
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage  # noqa: E402


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _corner(tie_rod_in: tuple[float, float, float]) -> Corner:
    """An FSAE-scale front-left corner with a movable tie rod inboard point."""
    return Corner(
        corner_id="FL",
        uca_inboard_front=_hp("UCA_IF", 120, 175, 308),
        uca_inboard_rear=_hp("UCA_IR", -120, 175, 308),
        uca_outboard=_hp("UCA_O", 0, 537, 385),
        lca_inboard_front=_hp("LCA_IF", 130, 175, 117),
        lca_inboard_rear=_hp("LCA_IR", -130, 175, 117),
        lca_outboard=_hp("LCA_O", 0, 582, 130),
        tie_rod_inboard=_hp("TIE_ROD_IN", *tie_rod_in),
        tie_rod_outboard=_hp("TIE_ROD_OUT", 85, 590, 179),
        wheel_center=_hp("WC", 0, 613, 245),
        tire=TirePackage(loaded_radius_mm=245.0, source="cad", tol_mm=1.0),
        static_camber_deg=-1.5,
        static_toe_deg_per_side=0.0,
    )


class TestReferencedToRideHeight:
    """Bump steer is toe CHANGE, not toe."""

    def test_zero_travel_is_exactly_zero(self) -> None:
        """Whatever static toe the linkage carries, travel 0 must give 0."""
        sweep = toe_sweep(_corner((-30, 270, 158)), -25.0, 25.0, steps=51)
        i = sweep.wheel_travel_mm.index(0.0)
        assert sweep.toe_deg_per_side[i] == pytest.approx(0.0, abs=1e-12)

    def test_static_toe_does_not_leak_into_the_curve(self) -> None:
        """Half a degree of static toe must not appear as bump steer.

        Static toe is realised as a tie rod LENGTH change, so it does perturb
        the curve slightly -- a different length sweeps a slightly different
        arc, worth ~2 % of the rate here. The point is the ORDER: adding
        0.5 deg of static toe moves the peak by well under 0.01 deg, not by
        0.5. Were the module reporting raw toe rather than toe change, this
        would fail by two orders of magnitude.
        """
        base = _corner((-30, 270, 158))
        toed = base.model_copy(update={"static_toe_deg_per_side": 0.5})
        shift = abs(
            bump_steer(toed).peak_abs_deg_per_side
            - bump_steer(base).peak_abs_deg_per_side
        )
        assert shift < 0.01, f"static toe leaked {shift:.4f} deg into the curve"


class TestTieRodHeightIsTheKnob:
    """Bump steer responds to the tie rod inboard height, monotonically."""

    def test_rate_changes_sign_across_the_null(self) -> None:
        """A tie rod too low and one too high steer opposite ways.

        This is what makes a back-solver on Z well posed: the root is bracketed.
        """
        low = bump_steer(_corner((-30, 270, 120))).linear_deg_per_mm_per_side
        high = bump_steer(_corner((-30, 270, 200))).linear_deg_per_mm_per_side
        assert low * high < 0, f"no sign change: {low} and {high}"

    def test_rate_is_monotonic_in_tie_rod_height(self) -> None:
        """Raising the inboard end steers monotonically toward toe-out in bump.

        Direction is a property of this layout (tie rod ahead of the axle, so
        raising the inboard end is the same sense as shortening the effective
        arm); what the back-solver needs is only that it is monotonic, so the
        test asserts strict monotonicity rather than a hard-coded sign.
        """
        rates = [
            bump_steer(_corner((-30, 270, z))).linear_deg_per_mm_per_side
            for z in (130, 150, 170, 190)
        ]
        assert rates == sorted(rates, reverse=True), rates
        assert len(set(rates)) == len(rates), "must be strictly monotonic"

    def test_a_null_exists_and_makes_the_peak_small(self) -> None:
        """Nulling the linear term leaves only the quadratic, which is small."""
        from scipy.optimize import brentq

        z0 = brentq(
            lambda z: bump_steer(_corner((-30, 270, z))).linear_deg_per_mm_per_side,
            120.0, 200.0, xtol=1e-6,
        )
        result = bump_steer(_corner((-30, 270, z0)))
        assert result.linear_deg_per_mm_per_side == pytest.approx(0.0, abs=1e-6)
        assert result.peak_abs_deg_per_side < 0.5


class TestLinearIsNotThePeak:
    """The two numbers are different, and reporting only one is misleading."""

    def test_nulled_linear_rate_still_has_a_peak(self) -> None:
        """A 'zero bump steer' tie rod still toes at full travel.

        The curve is a parabola about ride height, so bump and droop steer the
        SAME way and the linear fit through them reads zero. This is exactly
        why BumpSteerResult carries both numbers -- the front 2027 axle sits at
        a linear rate of -0.00002 deg/mm and a peak of 0.16 deg.
        """
        from scipy.optimize import brentq

        z0 = brentq(
            lambda z: bump_steer(_corner((-30, 270, z))).linear_deg_per_mm_per_side,
            120.0, 200.0, xtol=1e-6,
        )
        result = bump_steer(_corner((-30, 270, z0)))
        assert abs(result.linear_deg_per_mm_per_side) < 1e-6
        assert result.peak_abs_deg_per_side > 1e-3, "peak must not be zero too"
        # Same sign at both ends is the signature of a pure quadratic.
        assert (result.toe_at_full_bump_deg_per_side
                * result.toe_at_full_droop_deg_per_side) > 0


class TestPerSideVsTotal:
    """Never leave a toe quantity ambiguous (CLAUDE.md)."""

    def test_total_toe_is_twice_per_side(self) -> None:
        result = bump_steer(_corner((-30, 270, 158)))
        assert result.peak_abs_total_toe_deg == pytest.approx(
            2.0 * result.peak_abs_deg_per_side, rel=1e-12
        )

    def test_sweep_total_toe_is_twice_per_side(self) -> None:
        sweep = toe_sweep(_corner((-30, 270, 158)), -10.0, 10.0, steps=5)
        for per_side, total in zip(
            sweep.toe_deg_per_side, sweep.total_toe_deg, strict=True
        ):
            assert total == pytest.approx(2.0 * per_side, rel=1e-12)


class TestContract:
    """Failure modes are explicit, never a plausible-looking number."""

    def test_too_few_steps_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            bump_steer(_corner((-30, 270, 158)), steps=2)

    def test_sweep_reports_convergence_per_point(self) -> None:
        sweep = toe_sweep(_corner((-30, 270, 158)), -25.0, 25.0, steps=11)
        assert len(sweep.converged) == 11
        assert all(sweep.converged), "this geometry should solve over full travel"

    def test_non_converged_points_are_nan_not_stale(self) -> None:
        """A NaN is a failure the caller can see; a stale value is not."""
        sweep = toe_sweep(_corner((-30, 270, 158)), -25.0, 25.0, steps=11)
        for value, converged in zip(
            sweep.toe_deg_per_side, sweep.converged, strict=True
        ):
            assert converged != math.isnan(value)


class TestAgreesWithTheSolver:
    """The module must not drift from a direct DWSolver reading."""

    def test_peak_matches_a_hand_rolled_sweep(self) -> None:
        corner = _corner((-30, 270, 158))
        solver = DWSolver(corner)
        reference = solver.solve(0.0).toe_deg_per_side
        by_hand = max(
            abs(solver.solve(float(t)).toe_deg_per_side - reference)
            for t in (-25, -12.5, 0, 12.5, 25)
        )
        result = bump_steer(corner, wheel_travel_range_mm=25.0, steps=5)
        assert result.peak_abs_deg_per_side == pytest.approx(by_hand, abs=1e-12)
