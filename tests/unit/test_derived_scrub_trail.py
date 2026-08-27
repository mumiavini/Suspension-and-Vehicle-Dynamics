"""Golden scrub radius and mechanical trail from the vdcore derived functions.

Pins :func:`vdcore.geometry.derived.scrub_radius_mm`,
:func:`vdcore.geometry.derived.mechanical_trail_mm`, and the private
:func:`vdcore.geometry.derived._kingpin_ground_intercept` helper against the
shipped 2027 geometry.

ISO 8855: X+ forward, Y+ LEFT, Z+ up. Left corners (FL, RL) carry positive Y,
right corners (FR, RR) negative Y. Per the vdcore house rule, LEFT and RIGHT
signs are asserted INDEPENDENTLY -- each side is solved on its own so a Y-sign
fold bug cannot hide behind an assumed symmetry.

Values verified against sla_geometry.scrub_radius_mm and
steering_geometry.mechanical_trail_mm (the OptimumK-correlated derivations):
front scrub 15.08 mm, rear scrub 21.97 mm, front trail 21.43 mm.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402
from vdcore.geometry.derived import (  # noqa: E402
    _kingpin_ground_intercept,
    mechanical_trail_mm,
    scrub_radius_mm,
)
from vdcore.geometry.solver import DWSolver, SolverResult  # noqa: E402
from vdcore.models.hardpoint import Axle  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
import geometry_summary as gs  # noqa: E402


@pytest.fixture(scope="module")
def design() -> sla.DesignReport:
    """The static synthesis: the expensive sla.run() shared across tests."""
    return sla.run()


@pytest.fixture(scope="module")
def merged(design: sla.DesignReport) -> gs.MergedHardpoints:
    """The only complete corner: sla wishbones plus steering tie rod."""
    return gs.build_merged(design, stg.run())


@pytest.fixture(scope="module")
def front_axle(merged: gs.MergedHardpoints, design: sla.DesignReport) -> Axle:
    """FL/FR corners built from the merged 2027 front geometry."""
    inp = design.front.inputs
    return Axle(
        left=gs._vdcore_corner(merged, "FL", inp),
        right=gs._vdcore_corner(merged, "FR", inp),
    )


@pytest.fixture(scope="module")
def rear_axle(merged: gs.MergedHardpoints, design: sla.DesignReport) -> Axle:
    """RL/RR corners built from the merged 2027 rear geometry."""
    inp = design.rear.inputs
    return Axle(
        left=gs._vdcore_corner(merged, "RL", inp),
        right=gs._vdcore_corner(merged, "RR", inp),
    )


def _static(corner) -> SolverResult:  # type: ignore[no-untyped-def]
    """Solve a corner at static ride (zero heave/roll/rack)."""
    r = DWSolver(corner).solve(wheel_travel_mm=0.0, roll_deg=0.0, rack_mm=0.0)
    assert r.converged, "solver did not converge at static ride"
    return r


def test_front_scrub_fl_matches_golden(front_axle: Axle) -> None:
    """FL front scrub radius reproduces the pinned +15.08 mm from its own solve."""
    assert scrub_radius_mm(_static(front_axle.left)) == pytest.approx(15.08, abs=0.05)


def test_front_scrub_fr_matches_golden(front_axle: Axle) -> None:
    """FR front scrub radius reproduces +15.08 mm independently of FL (no symmetry assumed)."""
    assert scrub_radius_mm(_static(front_axle.right)) == pytest.approx(15.08, abs=0.05)


def test_rear_scrub_rl_matches_golden(rear_axle: Axle) -> None:
    """RL rear scrub radius reproduces the pinned +21.97 mm from its own solve."""
    assert scrub_radius_mm(_static(rear_axle.left)) == pytest.approx(21.97, abs=0.05)


def test_rear_scrub_rr_matches_golden(rear_axle: Axle) -> None:
    """RR rear scrub radius reproduces +21.97 mm independently of RL (no symmetry assumed)."""
    assert scrub_radius_mm(_static(rear_axle.right)) == pytest.approx(21.97, abs=0.05)


def test_front_trail_fl_matches_golden_and_is_positive(front_axle: Axle) -> None:
    """FL front mechanical trail is +21.43 mm and explicitly POSITIVE."""
    trail = mechanical_trail_mm(_static(front_axle.left))
    assert trail == pytest.approx(21.43, abs=0.05)
    assert trail > 0.0


def test_front_trail_fr_matches_golden_and_is_positive(front_axle: Axle) -> None:
    """FR front mechanical trail is +21.43 mm and positive, solved independently of FL."""
    trail = mechanical_trail_mm(_static(front_axle.right))
    assert trail == pytest.approx(21.43, abs=0.05)
    assert trail > 0.0


def test_scrub_positive_on_both_sides_all_corners(
    front_axle: Axle, rear_axle: Axle
) -> None:
    """Contact patch sits outboard of the kingpin on every corner: the left/right
    sign fold reports positive scrub for FL, FR, RL and RR independently."""
    assert scrub_radius_mm(_static(front_axle.left)) > 0.0
    assert scrub_radius_mm(_static(front_axle.right)) > 0.0
    assert scrub_radius_mm(_static(rear_axle.left)) > 0.0
    assert scrub_radius_mm(_static(rear_axle.right)) > 0.0


def test_front_trail_sign_follows_iso_frame(front_axle: Axle) -> None:
    """Front trail is positive: the kingpin ground intercept lies AHEAD of the
    contact patch in ISO X+ forward.

    This is the ISO-frame convention (kp_gnd_x - cp_x), the OPPOSITE raw formula
    to steering_geometry's design frame (X+ rearward, cp_x - kp_gnd_x), so the
    same physical trail reproduces +21.43 here rather than -21.43. Both sides are
    checked from their own solve to prove the frame sign is not accidental.
    """
    assert mechanical_trail_mm(_static(front_axle.left)) > 0.0
    assert mechanical_trail_mm(_static(front_axle.right)) > 0.0


def test_kingpin_ground_intercept_rejects_horizontal_axis() -> None:
    """A horizontal kingpin (UBJ.z == LBJ.z) never pierces the ground, so the
    helper refuses rather than fabricate an intercept."""
    horizontal = SimpleNamespace(
        ubj=SimpleNamespace(x_mm=100.0, y_mm=600.0, z_mm=250.0),
        lbj=SimpleNamespace(x_mm=110.0, y_mm=610.0, z_mm=250.0),
        contact_patch=SimpleNamespace(x_mm=105.0, y_mm=615.0, z_mm=0.0),
    )
    with pytest.raises(ValueError):
        _kingpin_ground_intercept(horizontal)  # type: ignore[arg-type]
