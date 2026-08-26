"""Golden values for the FSAE2027 design, pinned to the shipped config.

WHY THIS FILE EXISTS
--------------------
Until 2026-08-25 nothing under tests/ referenced sla_geometry.py or
steering_geometry.py. The two scripts the team quotes as the source of truth
were the only untested code in the pipeline, so every number in the geometry
summary could drift on the next edit with nothing to catch it.

WHAT BREAKS IT
--------------
This file pins the CURRENT DESIGN. Changing FRONT_2027 / REAR_2027 /
STEERING_2027 is meant to break it -- that is the alarm working. Update the
expected values deliberately, in the same commit as the design change, and say
in the message what moved and why.

Physics that must hold for ANY config belongs in
tests/property/test_fsae2027_invariants.py instead, which survives design churn.

Values verified 2026-08-25 against an independent reference solver and against
closed-form front-view geometry. Cross-checks that agreed to 4 decimals:
camber gain, half-track change, camber at full bump/droop, FVIC, roll-centre
height, outer/inner camber in roll.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402
from vdcore.analysis.axle import axle_rates, axle_roll  # noqa: E402
from vdcore.geometry.solver import DWSolver  # noqa: E402
from vdcore.models.hardpoint import Axle  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import geometry_summary as gs  # noqa: E402


@pytest.fixture(scope="module")
def design() -> sla.DesignReport:
    return sla.run()


@pytest.fixture(scope="module")
def merged(design: sla.DesignReport) -> gs.MergedHardpoints:
    return gs.build_merged(design, stg.run())


@pytest.fixture(scope="module")
def front_axle(merged: gs.MergedHardpoints, design: sla.DesignReport) -> Axle:
    inp = design.front.inputs
    return Axle(
        left=gs._vdcore_corner(merged, "FL", inp),
        right=gs._vdcore_corner(merged, "FR", inp),
    )


@pytest.fixture(scope="module")
def rear_axle(merged: gs.MergedHardpoints, design: sla.DesignReport) -> Axle:
    inp = design.rear.inputs
    return Axle(
        left=gs._vdcore_corner(merged, "RL", inp),
        right=gs._vdcore_corner(merged, "RR", inp),
    )


class TestStaticSynthesis:
    """sla_geometry.solve_axle -- pure front-view construction."""

    def test_front_roll_centre_height(self, design: sla.DesignReport) -> None:
        assert design.front.inputs.rc_height_mm == pytest.approx(35.0)

    def test_rear_roll_centre_height(self, design: sla.DesignReport) -> None:
        assert design.rear.inputs.rc_height_mm == pytest.approx(55.0)

    def test_front_fvic(self, design: sla.DesignReport) -> None:
        assert float(design.front.fvic[0]) == pytest.approx(-880.00, abs=0.01)
        assert float(design.front.fvic[1]) == pytest.approx(84.68, abs=0.01)

    def test_rear_fvic(self, design: sla.DesignReport) -> None:
        assert float(design.rear.fvic[0]) == pytest.approx(-800.00, abs=0.01)
        assert float(design.rear.fvic[1]) == pytest.approx(128.33, abs=0.01)

    def test_scrub_radius(self, design: sla.DesignReport) -> None:
        assert design.front.scrub_radius_mm == pytest.approx(15.08, abs=0.01)
        assert design.rear.scrub_radius_mm == pytest.approx(21.97, abs=0.01)

    def test_front_view_arm_projections(self, design: sla.DesignReport) -> None:
        assert design.front.lca_length_mm == pytest.approx(407.20, abs=0.01)
        assert design.front.uca_length_mm == pytest.approx(370.03, abs=0.01)
        assert design.rear.lca_length_mm == pytest.approx(383.60, abs=0.01)
        assert design.rear.uca_length_mm == pytest.approx(351.42, abs=0.01)

    def test_real_member_legs_rear_front_legs_are_over_length(
        self, design: sla.DesignReport
    ) -> None:
        """The rear front legs bust the 320-430 mm window and must keep saying so.

        This is the check that used to pass on a front-view PROJECTION of
        383.60 mm while the leg anyone cuts is 554.21 mm.
        """
        legs = sla.member_legs_mm(design.rear)
        assert legs["LCA front leg"] == pytest.approx(554.21, abs=0.05)
        assert legs["UCA front leg"] == pytest.approx(517.59, abs=0.05)
        assert legs["LCA rear leg"] == pytest.approx(388.26, abs=0.05)
        assert legs["UCA rear leg"] == pytest.approx(356.50, abs=0.05)
        lo, hi = design.rear.inputs.limits.lca_length_mm
        assert not (lo <= legs["LCA front leg"] <= hi)
        assert not (lo <= legs["UCA front leg"] <= hi)

    def test_anti_geometry_is_exactly_zero(self, design: sla.DesignReport) -> None:
        """Every pivot axis is horizontal, so the SVIC is at infinity.

        The legacy Streamlit app reported +200% anti-dive and +83.74%
        anti-squat here; both are artefacts of building the side-view instant
        centre from the pivot MIDPOINT instead of the pivot AXIS.
        """
        assert design.front.anti_percent == pytest.approx(0.0, abs=1e-9)
        assert design.rear.anti_percent == pytest.approx(0.0, abs=1e-9)


class TestExportedAlignment:
    """The deliverable must carry the design alignment, not a bare wheel."""

    @pytest.mark.parametrize("corner", ["FL", "FR", "RL", "RR"])
    def test_static_camber_is_encoded(
        self, merged: gs.MergedHardpoints, corner: str
    ) -> None:
        """Recovered from CONTACT_PATCH -> WHEEL_CENTER, not from the config.

        Before 2026-08-25 static_camber_deg was a reporting offset only: the
        exported points described a zero-camber car while every rate table
        assumed -1.50 deg.
        """
        camber, _toe = gs.static_alignment_encoded(merged, corner)
        assert camber == pytest.approx(-1.50, abs=0.001)

    def test_track_is_the_contact_patch_datum(
        self, merged: gs.MergedHardpoints, design: sla.DesignReport
    ) -> None:
        """Patches at the design track; wheel centres inboard by r*tan(camber)."""
        cp_l = merged.arr("FL", "CONTACT_PATCH")
        cp_r = merged.arr("FR", "CONTACT_PATCH")
        wc_l = merged.arr("FL", "WHEEL_CENTER")
        assert cp_l[1] - cp_r[1] == pytest.approx(design.front.inputs.track_mm, abs=0.01)
        assert cp_l[1] - wc_l[1] == pytest.approx(6.42, abs=0.01)

    def test_contact_patches_are_on_the_ground(
        self, merged: gs.MergedHardpoints
    ) -> None:
        for corner in ("FL", "FR", "RL", "RR"):
            assert merged.arr(corner, "CONTACT_PATCH")[2] == pytest.approx(0.0, abs=1e-9)


class TestAxleRates:
    """vdcore.analysis.axle on the complete merged corners, chassis-referenced."""

    def test_front_rates(self, front_axle: Axle) -> None:
        r = axle_rates(front_axle)
        assert r.camber_gain_deg_per_mm == pytest.approx(-0.0384, abs=0.0005)
        assert r.rc_migration_mm_per_mm == pytest.approx(-0.3914, abs=0.005)
        assert r.half_track_change_mm_per_mm == pytest.approx(0.0568, abs=0.0005)
        assert r.camber_full_bump_deg == pytest.approx(-2.4896, abs=0.01)
        assert r.camber_full_droop_deg == pytest.approx(-0.5692, abs=0.01)
        assert r.rc_min_mm == pytest.approx(25.30, abs=0.2)
        assert r.rc_max_mm == pytest.approx(44.90, abs=0.2)

    def test_rear_rates(self, rear_axle: Axle) -> None:
        r = axle_rates(rear_axle)
        assert r.camber_gain_deg_per_mm == pytest.approx(-0.0411, abs=0.0005)
        assert r.rc_migration_mm_per_mm == pytest.approx(-0.4239, abs=0.005)
        assert r.half_track_change_mm_per_mm == pytest.approx(0.0922, abs=0.0005)
        assert r.camber_full_bump_deg == pytest.approx(-2.5447, abs=0.01)
        assert r.camber_full_droop_deg == pytest.approx(-0.4856, abs=0.01)
        assert r.rc_min_mm == pytest.approx(44.52, abs=0.2)
        assert r.rc_max_mm == pytest.approx(65.74, abs=0.2)


class TestAxleRoll:
    """At 1.5 deg of roll, both wheels on the road."""

    def test_front_roll(self, front_axle: Axle) -> None:
        s = axle_roll(front_axle, 1.5)
        assert s.outer_camber_deg == pytest.approx(-0.635, abs=0.01)
        assert s.inner_camber_deg == pytest.approx(-2.390, abs=0.01)
        assert s.rc_height_mm == pytest.approx(33.89, abs=0.1)
        assert s.rc_lateral_mm == pytest.approx(-111.46, abs=1.0)
        assert s.wheel_travel_mm == pytest.approx(16.228, abs=0.01)

    def test_rear_roll(self, rear_axle: Axle) -> None:
        s = axle_roll(rear_axle, 1.5)
        assert s.outer_camber_deg == pytest.approx(-0.652, abs=0.01)
        assert s.inner_camber_deg == pytest.approx(-2.360, abs=0.01)
        assert s.rc_height_mm == pytest.approx(54.25, abs=0.1)
        assert s.rc_lateral_mm == pytest.approx(-71.09, abs=1.0)
        assert s.wheel_travel_mm == pytest.approx(15.704, abs=0.01)

    def test_outer_wheel_stays_in_the_useful_window(self, front_axle: Axle) -> None:
        """Static camber does most of the work; the geometry recovers ~42%."""
        s = axle_roll(front_axle, 1.5)
        assert -2.5 <= s.outer_camber_deg <= 0.0

    def test_roll_centre_migrates_far_sideways(self, front_axle: Axle) -> None:
        """~73 mm per degree of roll. The legacy app reported ~1 mm total,
        because it averaged the two sides and the lateral terms cancelled."""
        s = axle_roll(front_axle, 1.5)
        assert abs(s.rc_lateral_mm) > 90.0


@pytest.fixture(scope="module")
def steer() -> stg.SteeringReport:
    return stg.run()


class TestSteering:
    """steering_geometry -- front axle only."""

    def test_caster_and_trail(self, steer: stg.SteeringReport) -> None:
        assert steer.geometry.mechanical_trail_mm == pytest.approx(21.43, abs=0.05)
        assert steer.geometry.scrub_radius_mm == pytest.approx(15.08, abs=0.05)

    def test_steering_ratio(self, steer: stg.SteeringReport) -> None:
        assert steer.rates.steering_ratio == pytest.approx(4.58, abs=0.05)

    def test_parking_effort_within_limit(self, steer: stg.SteeringReport) -> None:
        """9.73 N.m against a 10 N.m limit -- only 3% of margin.

        Scrub radius drives this through M = mu*Fz*sqrt(rs^2 + tm^2), so any
        change that moves the contact patch outboard eats the margin.
        """
        assert steer.effort.steering_wheel_torque_Nm == pytest.approx(9.73, abs=0.1)
        assert steer.effort.steering_wheel_torque_Nm <= 10.0


class TestSolverHealth:
    """Guards on the solver itself, exercised through the real geometry."""

    def test_every_corner_converges_across_travel(self, front_axle: Axle) -> None:
        solver = DWSolver(front_axle.left)
        for travel in (-25.0, -12.5, 0.0, 12.5, 25.0):
            r = solver.solve(wheel_travel_mm=travel)
            assert r.converged, f"no convergence at {travel} mm"
            assert r.residual_norm < 1e-6
