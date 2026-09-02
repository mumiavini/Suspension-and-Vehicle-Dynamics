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
from vdcore.analysis.toe import bump_steer  # noqa: E402
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
        """The UCA projections shortened when its pickup moved outboard.

        Both arms shared one chassis plane at y = 175 until 2026-09-01; the
        upper arm now picks up at y = 210 so the chassis can carry it on a
        wider rail. The LCA is untouched, and the FVIC the two arms point at
        is unchanged -- see test_uca_pickup_y_is_kinematically_free.
        """
        assert design.front.lca_length_mm == pytest.approx(407.20, abs=0.01)
        assert design.front.uca_length_mm == pytest.approx(334.25, abs=0.01)
        assert design.rear.lca_length_mm == pytest.approx(383.60, abs=0.01)
        assert design.rear.uca_length_mm == pytest.approx(315.74, abs=0.01)

    def test_uca_pickup_is_outboard_of_the_lca_pickup(
        self, design: sla.DesignReport
    ) -> None:
        """The chassis constraint that drove the 2026-09-01 revision.

        The upper wishbone mounts to a wider rail than the lower one, so its
        inboard pickup must sit further from the car centreline.
        """
        for geo in (design.front, design.rear):
            assert geo.uca_in[0] > geo.lca_in[0], geo.inputs.name

    def test_uca_pickup_y_is_kinematically_free(
        self, design: sla.DesignReport
    ) -> None:
        """Sliding the inboard pickup along the arm line changes only length.

        The pickup lies on the BALL JOINT -> FVIC line, so moving it in y
        leaves the front-view construction -- FVIC, and therefore FVSA, roll
        centre height and the design camber gain -- bit-identical. This is the
        property that made the chassis request free to grant, and nothing
        tested it before. The full-travel values DO move (the arc curvature
        changes); those are pinned in TestAxleRates.
        """
        from dataclasses import replace

        for geo in (design.front, design.rear):
            moved = sla.solve_axle(
                replace(geo.inputs, uca_inner_y_mm=geo.inputs.uca_inner_y + 40.0),
                design.vehicle,
            )
            assert moved.fvic[0] == pytest.approx(float(geo.fvic[0]), abs=1e-12)
            assert moved.fvic[1] == pytest.approx(float(geo.fvic[1]), abs=1e-12)
            assert moved.uca_length_mm < geo.uca_length_mm

    def test_real_member_legs_rear_within_limits(
        self, design: sla.DesignReport
    ) -> None:
        """Rear legs are within the 320-490 mm band.

        2026-09-02: the LCA rear bracket alone moved to 100 mm clearance from
        the driveshaft plane (UCA stayed at 80 mm). Sweep = delta + base/2
        holds each arm's rearmost inboard pickup that many mm AHEAD of the
        rear-axle line. The LCA's clearance costs e/a = 1 + 2*delta/base, so
        base = 2*delta = 200 is the narrowest meeting the e/a <= 2.0 cap (it
        lands exactly on it) -- and therefore the shortest achievable front
        leg at that clearance. The binding member is now the LCA front leg
        at 487.0 mm, which is why the band went from 460 to 490. The UCA legs
        are untouched by this change.
        """
        legs = sla.member_legs_mm(design.rear)
        assert legs["LCA front leg"] == pytest.approx(486.98, abs=0.05)
        assert legs["UCA front leg"] == pytest.approx(396.60, abs=0.05)
        assert legs["LCA rear leg"] == pytest.approx(396.42, abs=0.05)
        assert legs["UCA rear leg"] == pytest.approx(325.71, abs=0.05)
        lo, hi = design.rear.inputs.limits.lca_length_mm
        assert lo <= legs["LCA front leg"] <= hi
        assert lo <= legs["UCA front leg"] <= hi
        assert lo <= legs["LCA rear leg"] <= hi
        assert lo <= legs["UCA rear leg"] <= hi

    def test_front_anti_dive_is_on_target(self, design: sla.DesignReport) -> None:
        """7.5 % anti-dive, the mid-band of the 5-10 % the team asked for.

        Carried entirely by the UCA pivot rake (dz_uca = -11.305 mm over a
        240 mm base). The lower axis stays horizontal, which is why the SVIC
        sits at the LOWER ball joint's height of 130 mm.

        WATCH THE SIGN. With the LCA axis horizontal a POSITIVE dz_uca gives
        PRO-dive; the rear UCA pickup has to sit BELOW the front one.
        """
        assert design.front.anti_percent == pytest.approx(7.5, abs=0.01)
        assert design.front.svic is not None
        assert design.front.svic[1] == pytest.approx(130.0, abs=0.01)

    def test_rear_anti_squat_is_exactly_zero(self, design: sla.DesignReport) -> None:
        """Both rear pivot axes are horizontal, so the SVIC is at infinity.

        The legacy Streamlit app reported +83.74% anti-squat here, an artefact
        of building the side-view instant centre from the pivot MIDPOINT
        instead of the pivot AXIS. Anti-squat was not requested and the rear
        rake stays at zero, so this stays an exact-zero assertion.
        """
        assert design.rear.anti_percent == pytest.approx(0.0, abs=1e-9)
        assert design.rear.inputs.dz_lca_mm == 0.0
        assert design.rear.inputs.dz_uca_mm == 0.0

    def test_no_chassis_point_sits_behind_the_wishbone(
        self, merged: gs.MergedHardpoints
    ) -> None:
        """The wishbone must be the rearmost thing bolted to the rear chassis.

        The toe link inboard used to sit at X = -1480, 20 mm BEHIND the
        rearmost wishbone pickup and only 60 mm ahead of the driveshaft plane.
        It now shares the LCA rear bracket at X = -1460.

        Outboard points are exempt: they are upright features that move with
        the wheel, not chassis connections.
        """
        for corner in ("RL", "RR"):
            wishbone_rearmost = min(
                merged.arr(corner, n)[0]
                for n in ("UCA_IN_FRONT", "UCA_IN_REAR",
                          "LCA_IN_FRONT", "LCA_IN_REAR")
            )
            toe_link_in = merged.arr(corner, "TIE_ROD_IN")[0]
            assert toe_link_in >= wishbone_rearmost - 1e-9, (
                f"{corner} toe link inboard at X={toe_link_in} is behind the "
                f"rearmost wishbone pickup at X={wishbone_rearmost}"
            )

    def test_rear_toe_link_shares_the_lca_rear_bracket(
        self, merged: gs.MergedHardpoints
    ) -> None:
        """One bracket, not two: same X and Y as the LCA rear pickup."""
        for corner in ("RL", "RR"):
            lca = merged.arr(corner, "LCA_IN_REAR")
            toe = merged.arr(corner, "TIE_ROD_IN")
            assert toe[0] == pytest.approx(lca[0], abs=1e-9)
            assert toe[1] == pytest.approx(lca[1], abs=1e-9)
            assert toe[2] - lca[2] == pytest.approx(39.47, abs=0.1)

    def test_rear_toe_link_comes_from_the_declared_config(
        self, merged: gs.MergedHardpoints
    ) -> None:
        """It is a declared design input now, not a hand-entered CSV row.

        Asserting against the config constant rather than against literals is
        what makes this test bite: if someone re-adds a CSV fallback, or the
        mirror onto the right side goes wrong, the merged set stops matching
        the declaration.
        """
        for corner, sy in (("RL", 1.0), ("RR", -1.0)):
            for name, declared in (
                ("TIE_ROD_IN", gs.REAR_TOE_LINK_INBOARD),
                ("TIE_ROD_OUT", gs.REAR_TOE_LINK_OUTBOARD),
            ):
                x, y, z = merged.arr(corner, name)
                assert (x, y, z) == pytest.approx(
                    (declared[0], sy * declared[1], declared[2]), abs=1e-9
                ), f"{corner}/{name}"

    def test_rear_pickups_clear_the_driveshaft_plane(
        self, design: sla.DesignReport
    ) -> None:
        """No rear inboard pickup may share the rear axle's X plane.

        The chassis team could not build a bracket around the driveshaft when
        the rearmost pickups sat exactly on the axle line. Since 2026-09-02
        the two arms carry different agreed clearances -- the LCA bracket
        needs 100 mm, the UCA stayed at 80 mm -- so each arm is checked
        against its own floor rather than one shared value.
        """
        rear = design.rear
        axle_x = rear.inputs.axle_x_mm
        for x in (rear.lca_in_rear_x_mm, rear.lca_in_front_x_mm):
            assert axle_x - x >= 100.0 - 1e-9, f"LCA pickup at x={x} is inside 100 mm"
        for x in (rear.uca_in_rear_x_mm, rear.uca_in_front_x_mm):
            assert axle_x - x >= 80.0 - 1e-9, f"UCA pickup at x={x} is inside 80 mm"


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
        """Re-anchored 2026-09-01 (UCA pickup to y=210, UCA raked for anti-dive).

        Camber gain barely moved (-0.0384 -> -0.0386): the pickup slid along
        the arm line, so the ride-height rate is set by the same FVIC. What
        moved is RC migration, -0.3914 -> -0.3106 mm/mm, a ~20 % improvement
        that falls out of the shorter upper arm. It was not a target.
        """
        r = axle_rates(front_axle)
        assert r.camber_gain_deg_per_mm == pytest.approx(-0.0386, abs=0.0005)
        assert r.rc_migration_mm_per_mm == pytest.approx(-0.3106, abs=0.005)
        assert r.half_track_change_mm_per_mm == pytest.approx(0.0574, abs=0.0005)
        assert r.camber_full_bump_deg == pytest.approx(-2.5155, abs=0.01)
        assert r.camber_full_droop_deg == pytest.approx(-0.5795, abs=0.01)
        assert r.rc_min_mm == pytest.approx(27.90, abs=0.2)
        assert r.rc_max_mm == pytest.approx(43.44, abs=0.2)

    def test_rear_rates(self, rear_axle: Axle) -> None:
        """Re-anchored 2026-09-01 (UCA pickup to y=210, pickups swept forward).

        Camber gain is unchanged to four decimals -- the rear rake is still
        zero and the pickup only slid along its arm line. RC migration
        improves -0.4239 -> -0.3380 mm/mm for the same reason as the front.
        """
        r = axle_rates(rear_axle)
        assert r.camber_gain_deg_per_mm == pytest.approx(-0.0411, abs=0.0005)
        assert r.rc_migration_mm_per_mm == pytest.approx(-0.3380, abs=0.005)
        assert r.half_track_change_mm_per_mm == pytest.approx(0.0922, abs=0.0005)
        assert r.camber_full_bump_deg == pytest.approx(-2.5649, abs=0.01)
        assert r.camber_full_droop_deg == pytest.approx(-0.5053, abs=0.01)
        assert r.rc_min_mm == pytest.approx(46.75, abs=0.2)
        assert r.rc_max_mm == pytest.approx(63.67, abs=0.2)


class TestBumpSteer:
    """Toe over travel on BOTH axles.

    The rear had no bump-steer number anywhere before 2026-09-01 --
    steering_geometry.py owns bump steer and covers the front only -- so
    moving the rear toe link was previously an unmeasured change.
    """

    def test_front_linear_rate_is_nulled(self, front_axle: Axle) -> None:
        """rack_z_mm is solved for this, so it should be ~0 by construction."""
        b = bump_steer(front_axle.left)
        assert b.linear_deg_per_mm_per_side == pytest.approx(0.0, abs=0.0005)

    def test_front_peak_is_the_quadratic_term(self, front_axle: Axle) -> None:
        """Nulling the linear rate does NOT make the peak zero.

        The front toes 0.16 deg per side at full travel, the same way in bump
        and droop. That is not a regression -- it measured 0.1581 deg before
        this revision too. It was invisible because only the linear rate was
        ever reported.
        """
        b = bump_steer(front_axle.left)
        assert b.peak_abs_deg_per_side == pytest.approx(0.1598, abs=0.005)
        assert (b.toe_at_full_bump_deg_per_side
                * b.toe_at_full_droop_deg_per_side) > 0

    def test_rear_improved_when_the_toe_link_moved(self, rear_axle: Axle) -> None:
        """Moving the inboard end to the LCA bracket more than halved the peak.

        It was 0.0313 deg per side with the toe link at X = -1480; it is
        0.0128 at X = -1460. The move was made for packaging, and the
        kinematics happened to improve -- worth pinning so a later change
        cannot quietly undo it.
        """
        b = bump_steer(rear_axle.left)
        assert b.peak_abs_deg_per_side == pytest.approx(0.0128, abs=0.002)
        assert b.peak_abs_deg_per_side < 0.02

    def test_both_axles_are_inside_the_reporting_bands(
        self, front_axle: Axle, rear_axle: Axle
    ) -> None:
        for axle in (front_axle, rear_axle):
            b = bump_steer(axle.left)
            assert abs(b.linear_deg_per_mm_per_side) <= gs.BUMP_STEER_LINEAR_LIMIT
            assert b.peak_abs_deg_per_side <= gs.BUMP_STEER_PEAK_LIMIT

    def test_left_and_right_are_mirrors(
        self, rear_axle: Axle
    ) -> None:
        """A symmetric axle must bump-steer identically on both sides."""
        left = bump_steer(rear_axle.left)
        right = bump_steer(rear_axle.right)
        assert left.peak_abs_deg_per_side == pytest.approx(
            right.peak_abs_deg_per_side, abs=1e-9
        )
        assert left.linear_deg_per_mm_per_side == pytest.approx(
            right.linear_deg_per_mm_per_side, abs=1e-9
        )


class TestAxleRoll:
    """At 1.5 deg of roll, both wheels on the road."""

    def test_front_roll(self, front_axle: Axle) -> None:
        s = axle_roll(front_axle, 1.5)
        assert s.outer_camber_deg == pytest.approx(-0.648, abs=0.01)
        assert s.inner_camber_deg == pytest.approx(-2.392, abs=0.01)
        assert s.rc_height_mm == pytest.approx(34.86, abs=0.1)
        assert s.rc_lateral_mm == pytest.approx(-86.90, abs=1.0)
        assert s.wheel_travel_mm == pytest.approx(16.229, abs=0.01)

    def test_rear_roll(self, rear_axle: Axle) -> None:
        s = axle_roll(rear_axle, 1.5)
        assert s.outer_camber_deg == pytest.approx(-0.660, abs=0.01)
        assert s.inner_camber_deg == pytest.approx(-2.368, abs=0.01)
        assert s.rc_height_mm == pytest.approx(54.58, abs=0.1)
        assert s.rc_lateral_mm == pytest.approx(-56.34, abs=1.0)
        assert s.wheel_travel_mm == pytest.approx(15.704, abs=0.01)

    def test_outer_wheel_stays_in_the_useful_window(self, front_axle: Axle) -> None:
        """Static camber does most of the work; the geometry recovers ~42%."""
        s = axle_roll(front_axle, 1.5)
        assert -2.5 <= s.outer_camber_deg <= 0.0

    def test_roll_centre_migrates_far_sideways(self, front_axle: Axle) -> None:
        """~58 mm per degree of roll. The legacy app reported ~1 mm TOTAL,
        because it averaged the two sides and the lateral terms cancelled.

        Threshold lowered from 90 to 70 mm on 2026-09-01: the shorter upper
        arm cut lateral migration from 111.5 to 86.9 mm at 1.5 deg. The guard
        is against the legacy app's order-of-magnitude error, not against a
        design target, so the bound tracks the geometry rather than pinning
        it -- the exact value is asserted in test_front_roll.
        """
        s = axle_roll(front_axle, 1.5)
        assert abs(s.rc_lateral_mm) > 70.0


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
