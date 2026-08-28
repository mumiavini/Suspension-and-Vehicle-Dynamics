"""Tests for the Altair MotionSolve KPI runner that do NOT need Altair.

The parts that need a local MotionSolve install are covered by
``altair_model/validate_kinematics.py`` (the gate) and by actually pressing the
button in the app. What is tested here is everything between: the travel-grid
arithmetic, the replay solver's contract, and -- most importantly -- that the
roll arithmetic duplicated out of ``axle_roll`` still agrees with it.

That last one is the real point. ``kpi_runner._roll_state_from_samples`` exists
because ``axle_roll`` finds its wheel travel by ``brentq``, and every probe of
that search would cost a MotionSolve subprocess. Duplicated physics is exactly
the kind of thing that drifts silently, so it is pinned here against the
original on DWSolver samples, where both must agree exactly.
"""

from __future__ import annotations

import math

import pytest

from altair_model.kpi_runner import (
    MotionSolveReplay,
    _patch_residual_mm,
    _roll_state_from_samples,
    roll_intervals,
    sweep_intervals,
)
from altair_model.msolve_driver import build_corner, read_csv_points
from vdcore.analysis.axle import axle_roll, sample_corner
from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Axle

CSV = "Geometry Summary/hardpoints_2027_merged.csv"


@pytest.fixture(scope="module")
def front_axle() -> Axle:
    points = read_csv_points(CSV)
    return Axle(
        left=build_corner("FL", points["FL"], 0.0,
                          static_camber_deg=-1.5, loaded_radius_mm=245.0),
        right=build_corner("FR", points["FR"], 0.0,
                           static_camber_deg=-1.5, loaded_radius_mm=245.0),
    )


# --------------------------------------------------------------------------- #
# travel grid
# --------------------------------------------------------------------------- #

def test_sweep_grid_contains_every_travel_axle_rates_asks_for() -> None:
    """The whole point of the grid: no interpolation is ever needed.

    ``axle_rates`` samples the linspace AND a central difference at
    +/- travel/20. Both must land exactly on MotionSolve's output grid.
    """
    travel, steps = 25.0, 41
    n = sweep_intervals(travel, steps)
    spacing = 2.0 * travel / n

    wanted = [travel / 20.0, -travel / 20.0, travel, -travel, 0.0]
    wanted += [-travel + i * (2.0 * travel) / (steps - 1) for i in range(steps)]

    for t in wanted:
        offset = (t + travel) / spacing
        assert abs(offset - round(offset)) < 1e-9, f"{t} is off-grid"


@pytest.mark.parametrize("steps", [2, 11, 21, 41, 81])
def test_sweep_intervals_divides_both_requirements(steps: int) -> None:
    n = sweep_intervals(25.0, steps)
    assert n % (steps - 1) == 0
    assert n % 40 == 0


def test_sweep_intervals_rejects_degenerate_sweep() -> None:
    with pytest.raises(ValueError):
        sweep_intervals(25.0, 1)


def test_roll_intervals_is_even_and_fine_enough() -> None:
    """Even keeps -t, 0, +t all exact; the spacing bound keeps MotionSolve accurate."""
    for roll_travel in (0.5, 5.0, 16.228150, 21.634, 40.0):
        n = roll_intervals(roll_travel, travel_mm=25.0)
        assert n % 2 == 0, "an odd count would miss travel 0"
        assert n >= 2
        spacing = 2.0 * roll_travel / n
        assert spacing <= 25.0 / 20.0 + 1e-12, "coarser than the measured safe spacing"


# --------------------------------------------------------------------------- #
# replay contract
# --------------------------------------------------------------------------- #

def test_replay_returns_the_state_it_was_given(front_axle: Axle) -> None:
    corner = front_axle.left
    truth = DWSolver(corner)
    replay = MotionSolveReplay(corner_id=corner.corner_id)
    for t in (-5.0, 0.0, 5.0):
        replay.add(t, truth.solve(wheel_travel_mm=t))

    for t in (-5.0, 0.0, 5.0):
        assert replay.solve(wheel_travel_mm=t).camber_deg == pytest.approx(
            truth.solve(wheel_travel_mm=t).camber_deg
        )


def test_replay_refuses_an_unsolved_travel(front_axle: Axle) -> None:
    """Never interpolate. A missing travel is a grid bug, not a rounding issue."""
    corner = front_axle.left
    replay = MotionSolveReplay(corner_id=corner.corner_id)
    replay.add(0.0, DWSolver(corner).solve(wheel_travel_mm=0.0))

    with pytest.raises(KeyError, match="no solved state"):
        replay.solve(wheel_travel_mm=3.7)


@pytest.mark.parametrize("kwargs", [{"roll_deg": 1.0}, {"rack_mm": 5.0}])
def test_replay_refuses_states_motionsolve_does_not_model(
    front_axle: Axle, kwargs: dict[str, float]
) -> None:
    """The corner model grounds the chassis and the inner tie rod.

    Silently returning the unrolled, unsteered answer would be a plausible
    wrong number, which is exactly what the project forbids.
    """
    corner = front_axle.left
    replay = MotionSolveReplay(corner_id=corner.corner_id)
    replay.add(0.0, DWSolver(corner).solve(wheel_travel_mm=0.0))

    with pytest.raises(NotImplementedError):
        replay.solve(wheel_travel_mm=0.0, **kwargs)


# --------------------------------------------------------------------------- #
# the duplicated roll arithmetic
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("roll_deg", [1.5, 2.0, -2.0, 0.75])
def test_roll_state_matches_axle_roll_on_dwsolver_samples(
    front_axle: Axle, roll_deg: float
) -> None:
    """The pin: fed the same samples, the copy must equal the original.

    ``_roll_state_from_samples`` is the post-root-find half of ``axle_roll``,
    lifted out so the Altair path can reuse the travel instead of paying for a
    ``brentq`` search in MotionSolve subprocesses. If someone edits either copy,
    this fails.
    """
    reference = axle_roll(front_axle, roll_deg)
    travel = reference.wheel_travel_mm

    left_solver = DWSolver(front_axle.left)
    right_solver = DWSolver(front_axle.right)
    outer = sample_corner(front_axle.left, left_solver, +travel)
    inner = sample_corner(front_axle.right, right_solver, -travel)

    o_cam, i_cam, rc_z, rc_y = _roll_state_from_samples(outer, inner, roll_deg)

    assert o_cam == pytest.approx(reference.outer_camber_deg, abs=1e-12)
    assert i_cam == pytest.approx(reference.inner_camber_deg, abs=1e-12)
    assert rc_z == pytest.approx(reference.rc_height_mm, abs=1e-9)
    assert rc_y == pytest.approx(reference.rc_lateral_mm, abs=1e-9)


def test_patch_residual_is_zero_at_the_solved_roll_travel(front_axle: Axle) -> None:
    """Sanity check on the residual the app reports.

    At the travel ``axle_roll`` converged to, the two patches meet by
    construction. A non-trivial residual there would mean the residual formula
    disagrees with the root the solver found -- which would make the number the
    app shows meaningless.
    """
    reference = axle_roll(front_axle, 1.5)
    travel = reference.wheel_travel_mm
    outer = sample_corner(front_axle.left, DWSolver(front_axle.left), +travel)
    inner = sample_corner(front_axle.right, DWSolver(front_axle.right), -travel)

    assert abs(_patch_residual_mm(outer, inner, 1.5)) < 1e-6


# --------------------------------------------------------------------------- #
# corner construction
# --------------------------------------------------------------------------- #

def test_static_camber_override_beats_the_csv_rounding() -> None:
    """The CSV records camber only through a rounded contact patch.

    On the 2027 file that reads back as -1.499938 deg, not -1.500000. The
    override exists so the Altair column is built from the same design input as
    the vdcore column instead of showing a 6e-5 deg input difference as if it
    were a solver disagreement.
    """
    points = read_csv_points(CSV)

    from_file = build_corner("FL", points["FL"], 0.0)
    assert from_file.static_camber_deg == pytest.approx(-1.4999, abs=1e-3)
    assert from_file.static_camber_deg != -1.5

    overridden = build_corner("FL", points["FL"], 0.0, static_camber_deg=-1.5)
    assert overridden.static_camber_deg == -1.5


def test_left_and_right_corners_get_mirrored_camber_sign() -> None:
    """Negative camber means top-inboard on BOTH sides, so Y sign must not leak."""
    points = read_csv_points(CSV)
    left = build_corner("FL", points["FL"], 0.0)
    right = build_corner("FR", points["FR"], 0.0)

    assert left.static_camber_deg < 0
    assert right.static_camber_deg < 0
    assert left.static_camber_deg == pytest.approx(right.static_camber_deg)
    assert left.wheel_center.y_mm > 0
    assert right.wheel_center.y_mm < 0
