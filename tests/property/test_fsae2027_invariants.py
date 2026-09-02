"""Physics that must hold for ANY double-wishbone config, not just FSAE2027.

Companion to tests/benchmarks/test_fsae2027_design.py, which pins the current
design's numbers and is MEANT to break when the design changes. Nothing here
depends on the shipped values, so a design change must not break this file --
if it does, the change broke physics, not just a number.

Several of these are regression guards for bugs found in the 2026-08-25 audit.
Each one names the bug it exists to catch, because a guard whose purpose is
undocumented gets "fixed" by the next person who sees it fail.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402
from vdcore.analysis.axle import axle_rates, axle_roll, sample_corner  # noqa: E402
from vdcore.geometry.derived import contact_patch  # noqa: E402
from vdcore.geometry.solver import DWSolver  # noqa: E402
from vdcore.models.hardpoint import Axle  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import geometry_summary as gs  # noqa: E402

R2D = 57.29577951308232


@pytest.fixture(scope="module")
def design() -> sla.DesignReport:
    return sla.run()


@pytest.fixture(scope="module")
def merged(design: sla.DesignReport) -> gs.MergedHardpoints:
    return gs.build_merged(design, stg.run())


def _axle(merged: gs.MergedHardpoints, inp: sla.AxleInputs, left: str, right: str) -> Axle:
    return Axle(
        left=gs._vdcore_corner(merged, left, inp),
        right=gs._vdcore_corner(merged, right, inp),
    )


@pytest.fixture(scope="module")
def axles(merged: gs.MergedHardpoints, design: sla.DesignReport) -> dict[str, Axle]:
    return {
        "front": _axle(merged, design.front.inputs, "FL", "FR"),
        "rear": _axle(merged, design.rear.inputs, "RL", "RR"),
    }


class TestSolverDrivesTravel:
    """The solver must impose the travel it is asked for.

    GUARD FOR: DWSolver's 9th residual projected the wheel centre onto the
    kingpin axis, but the wheel centre is reconstructed rigidly from the ball
    joints and the kingpin is body-fixed, so the residual was identically zero.
    That left 8 constraints for 9 unknowns -- the travel DOF was free and
    least_squares resolved it by optimiser path. Asking for 25 mm of bump
    delivered 27.03 mm and put camber gain 7.3% out.
    """

    @pytest.mark.parametrize("travel", [-25.0, -10.0, 0.0, 10.0, 25.0])
    def test_requested_travel_is_delivered(
        self, axles: dict[str, Axle], travel: float
    ) -> None:
        corner = axles["front"].left
        solver = DWSolver(corner)
        static_wc_z = solver.solve(wheel_travel_mm=0.0).wheel_center.z_mm
        r = solver.solve(wheel_travel_mm=travel)
        assert r.converged
        # Chassis moves, wheel does not: relative travel is exactly what we asked.
        actual = (r.wheel_center.z_mm - static_wc_z) + travel
        assert actual == pytest.approx(travel, abs=1e-6)


class TestContactPatchSide:
    """Negative camber puts the contact patch OUTBOARD.

    GUARD FOR: both vdcore contact-patch implementations had the lateral shift
    sign inverted, putting the patch 2*r*tan(gamma) = 12.8 mm out of place at
    -1.5 deg, and two unit tests asserted the wrong direction and passed.
    """

    def test_left_corner_patch_goes_outboard(self, axles: dict[str, Axle]) -> None:
        corner = axles["front"].left
        cp_zero = contact_patch(corner, camber_deg=0.0)
        cp_neg = contact_patch(corner, camber_deg=-2.0)
        assert corner.wheel_center.y_mm > 0, "FL must have positive Y in ISO 8855"
        assert cp_neg.y_mm > cp_zero.y_mm

    def test_right_corner_patch_goes_outboard(self, axles: dict[str, Axle]) -> None:
        corner = axles["front"].right
        cp_zero = contact_patch(corner, camber_deg=0.0)
        cp_neg = contact_patch(corner, camber_deg=-2.0)
        assert corner.wheel_center.y_mm < 0, "FR must have negative Y in ISO 8855"
        assert cp_neg.y_mm < cp_zero.y_mm

    def test_negative_camber_widens_ground_track(self, axles: dict[str, Axle]) -> None:
        ax = axles["front"]
        zero = contact_patch(ax.left, 0.0).y_mm - contact_patch(ax.right, 0.0).y_mm
        camb = contact_patch(ax.left, -2.0).y_mm - contact_patch(ax.right, -2.0).y_mm
        assert camb > zero

    def test_patch_stays_on_the_ground_at_any_camber(
        self, axles: dict[str, Axle]
    ) -> None:
        """loaded_radius_mm is the VERTICAL drop, so the patch never lifts."""
        corner = axles["front"].left
        z0 = contact_patch(corner, 0.0).z_mm
        for gamma in (-5.0, -1.5, 0.0, 1.5, 5.0):
            assert contact_patch(corner, gamma).z_mm == pytest.approx(z0, abs=1e-9)


class TestStaticRecovery:
    """Zero travel must reproduce the design-intent alignment."""

    @pytest.mark.parametrize("axle_name", ["front", "rear"])
    def test_zero_travel_recovers_static_camber(
        self, axles: dict[str, Axle], axle_name: str
    ) -> None:
        corner = axles[axle_name].left
        r = DWSolver(corner).solve(wheel_travel_mm=0.0)
        assert r.converged
        assert r.camber_deg == pytest.approx(corner.static_camber_deg, abs=1e-3)

    @pytest.mark.parametrize("axle_name", ["front", "rear"])
    def test_exported_points_reproduce_static_camber(
        self, axles: dict[str, Axle], axle_name: str
    ) -> None:
        """CONTACT_PATCH -> WHEEL_CENTER must encode static_camber_deg.

        GUARD FOR: static_camber_deg was a reporting offset only, so the
        exported hardpoints described a zero-camber car while every rate table
        assumed the design value.
        """
        corner = axles[axle_name].left
        cp = contact_patch(corner, corner.static_camber_deg)
        dy = corner.wheel_center.y_mm - cp.y_mm
        dz = corner.wheel_center.z_mm - cp.z_mm
        recovered = math.degrees(math.atan2(dy, dz))
        assert recovered == pytest.approx(corner.static_camber_deg, abs=1e-6)


class TestLeftRightSymmetry:
    """A mirrored axle must behave identically on both sides."""

    @pytest.mark.parametrize("axle_name", ["front", "rear"])
    def test_symmetric_geometry_gives_symmetric_camber(
        self, axles: dict[str, Axle], axle_name: str
    ) -> None:
        ax = axles[axle_name]
        for travel in (-20.0, 0.0, 20.0):
            left = DWSolver(ax.left).solve(wheel_travel_mm=travel)
            right = DWSolver(ax.right).solve(wheel_travel_mm=travel)
            assert left.converged and right.converged
            assert left.camber_deg == pytest.approx(right.camber_deg, abs=1e-6)

    @pytest.mark.parametrize("axle_name", ["front", "rear"])
    def test_zero_roll_is_symmetric(
        self, axles: dict[str, Axle], axle_name: str
    ) -> None:
        s = axle_roll(axles[axle_name], 0.0)
        assert s.wheel_travel_mm == pytest.approx(0.0, abs=1e-9)
        assert s.outer_camber_deg == pytest.approx(s.inner_camber_deg, abs=1e-6)
        assert s.rc_lateral_mm == pytest.approx(0.0, abs=1e-6)


class TestCamberGainMatchesSwingArm:
    """Camber gain must agree with 57.2958 / FVSA.

    A wheel turning about an instant centre L mm away changes camber at
    57.2958/L deg per mm of travel. The 3D solve carries effects the planar
    construction does not, so allow a few percent -- but not more.
    """

    @pytest.mark.parametrize("axle_name", ["front", "rear"])
    def test_gain_matches_construction(
        self, axles: dict[str, Axle], axle_name: str, design: sla.DesignReport
    ) -> None:
        geo = design.front if axle_name == "front" else design.rear
        expected = -R2D / geo.inputs.fvsa_length_mm
        got = axle_rates(axles[axle_name]).camber_gain_deg_per_mm
        assert got == pytest.approx(expected, rel=0.05)

    @pytest.mark.parametrize("axle_name", ["front", "rear"])
    def test_bump_gives_negative_camber(
        self, axles: dict[str, Axle], axle_name: str
    ) -> None:
        """UCA shorter than LCA must gain negative camber in bump."""
        r = axle_rates(axles[axle_name])
        assert r.camber_gain_deg_per_mm < 0.0
        assert r.camber_full_bump_deg < r.camber_full_droop_deg


class TestRollKinematics:
    """Roll behaviour that must hold regardless of the numbers."""

    @pytest.mark.parametrize("axle_name", ["front", "rear"])
    def test_wheel_travel_tracks_half_track_times_tan_roll(
        self, axles: dict[str, Axle], axle_name: str
    ) -> None:
        ax = axles[axle_name]
        roll = 1.5
        s = axle_roll(ax, roll)
        half_track = abs(sample_corner(ax.left, DWSolver(ax.left), 0.0).cp_y_mm)
        assert s.wheel_travel_mm == pytest.approx(
            half_track * math.tan(math.radians(roll)), rel=0.05
        )

    @pytest.mark.parametrize("axle_name", ["front", "rear"])
    def test_outer_wheel_loses_camber_in_roll(
        self, axles: dict[str, Axle], axle_name: str
    ) -> None:
        """Relative to the road the outer wheel always gives some camber back;
        no double-wishbone recovers the full roll angle."""
        ax = axles[axle_name]
        static = ax.left.static_camber_deg
        s = axle_roll(ax, 1.5)
        assert s.outer_camber_deg > static, "outer must lose negative camber"
        assert s.inner_camber_deg < static, "inner must gain negative camber"

    @pytest.mark.parametrize("axle_name", ["front", "rear"])
    def test_roll_centre_migrates_toward_the_outer_wheel(
        self, axles: dict[str, Axle], axle_name: str
    ) -> None:
        """GUARD FOR: the legacy app averaged the two sides' roll centres, so
        the lateral terms cancelled and it always reported ~0 mm."""
        s = axle_roll(axles[axle_name], 1.5)
        assert abs(s.rc_lateral_mm) > 10.0


class TestAntiGeometryFollowsPivotRake:
    """Anti-geometry must respond to pivot-axis rake, and the rates with it.

    GUARD FOR: the front-view four-bar that used to produce the rate tables
    never read dz_lca_mm / dz_uca_mm, so sweeping anti-dive from 0% to 57%
    left every rate bit-identical while the real geometry moved.
    """

    def test_horizontal_pivot_axes_give_exactly_zero_anti(
        self, design: sla.DesignReport
    ) -> None:
        """Zero rake must give exactly zero anti, on whichever axle has it.

        The rear still runs both axes horizontal. The front no longer does
        (7.5 % anti-dive from UCA rake since 2026-09-01), so it is flattened
        here rather than dropped -- the invariant is about the construction,
        not about which axle happens to be flat this revision.
        """
        from dataclasses import replace

        for geo in (design.front, design.rear):
            flat = sla.solve_axle(
                replace(geo.inputs, dz_lca_mm=0.0, dz_uca_mm=0.0), design.vehicle
            )
            assert flat.svic is None
            assert flat.anti_percent == pytest.approx(0.0, abs=1e-9)

        assert design.rear.inputs.dz_lca_mm == 0.0
        assert design.rear.inputs.dz_uca_mm == 0.0
        assert design.rear.anti_percent == pytest.approx(0.0, abs=1e-9)

    def test_raking_the_pivots_changes_anti(self, design: sla.DesignReport) -> None:
        from dataclasses import replace

        veh = design.vehicle
        base = design.front.inputs
        anti = [
            sla.solve_axle(
                replace(base, dz_lca_mm=dz, dz_uca_mm=dz), veh
            ).anti_percent
            for dz in (0.0, 10.0, 25.0)
        ]
        assert anti[0] == pytest.approx(0.0, abs=1e-9)
        assert anti[1] > 5.0 and anti[2] > anti[1]

    def test_raking_the_pivots_also_changes_the_rates(
        self, merged: gs.MergedHardpoints, design: sla.DesignReport
    ) -> None:
        """The rate solve must SEE the rake. This is the assertion that would
        have failed for the whole life of the front-view implementation."""
        from dataclasses import replace

        veh = design.vehicle
        base = design.front.inputs
        gains = []
        for dz in (0.0, 40.0):
            geo = sla.solve_axle(replace(base, dz_lca_mm=dz, dz_uca_mm=dz), veh)
            rear = design.rear
            model_hp = gs.build_merged(
                sla.DesignReport(
                    vehicle=veh,
                    vehicle_results=design.vehicle_results,
                    front=geo,
                    rear=rear,
                    model=sla.build_model(geo, rear),
                    text="",
                ),
                stg.run(),
            )
            axle = _axle(model_hp, geo.inputs, "FL", "FR")
            gains.append(axle_rates(axle).camber_gain_deg_per_mm)
        assert gains[0] != pytest.approx(gains[1], abs=1e-5), (
            "camber gain did not respond to pivot-axis rake -- the rate solve "
            "is blind to dz_lca_mm / dz_uca_mm again"
        )


class TestConvergenceIsReported:
    """Non-convergence must never reach a report as a plausible number."""

    def test_solver_reports_convergence_flag(self, axles: dict[str, Axle]) -> None:
        r = DWSolver(axles["front"].left).solve(wheel_travel_mm=0.0)
        assert r.converged is True
        assert r.residual_norm < 1e-6
