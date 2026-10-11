"""Rod-end swing screening in ``steering_geometry.py`` must keep both tie-rod ends
in one frame.

``DWSolver`` applies wheel travel by displacing the CHASSIS by
``-wheel_travel_mm``, so a solved outer tie-rod joint (TRO) is in a frame where
the inner joint (TRI) sits at ``tri_0 + (0, rack, -bump)``. The screening check
used to take TRI at ``tri_0 + (0, rack, 0)``: the "tie rod" it measured changed
length with bump, and the worst swing on the 2027 front read 5.51 deg instead
of 7.82 deg. These tests pin the frame and the corrected value.

Frame: ISO 8855 (X+ forward, Y+ LEFT, Z+ up); bump positive.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402

BUMPS = np.linspace(-25.0, 25.0, 5)


@pytest.fixture(scope="module")
def kin() -> stg.SteeringKinematics:
    front = sla.solve_axle(sla.FRONT_2027, sla.VEHICLE_2027)
    sg = stg.synthesize_steering(front, stg.STEERING_2027, sla.VEHICLE_2027)
    return stg.SteeringKinematics(front, sg, stg.STEERING_2027, sla.VEHICLE_2027)


def _tro(res) -> np.ndarray:
    return np.array([res.tro.x_mm, res.tro.y_mm, res.tro.z_mm])


def test_tie_rod_is_rigid_when_tri_moves_with_the_chassis(kin) -> None:
    """With TRI displaced like every chassis point, the rod length never changes."""
    tri0 = np.array(kin.solver_fl._tri_0)
    length0 = float(np.linalg.norm(_tro(kin.solver_fl.solve()) - tri0))
    rack_max = stg.STEERING_2027.max_rack_travel_mm
    for b in BUMPS:
        for r in np.linspace(-rack_max, rack_max, 5):
            fl, _ = kin.solve_pair(bump_mm=float(b), rack_mm=float(r))
            tri = tri0 + np.array([0.0, r, -b])
            assert np.linalg.norm(_tro(fl) - tri) == pytest.approx(length0, abs=1e-6)


def test_undisplaced_tri_is_not_a_rigid_rod(kin) -> None:
    """The old construction: a rod that stretches with bump, i.e. a frame mix."""
    tri0 = np.array(kin.solver_fl._tri_0)
    fl0, _ = kin.solve_pair(bump_mm=0.0)
    fl25, _ = kin.solve_pair(bump_mm=25.0)
    l0 = np.linalg.norm(_tro(fl0) - tri0)
    l25_wrong = np.linalg.norm(_tro(fl25) - tri0)
    assert abs(l25_wrong - l0) > 1.0


def test_worst_swing_matches_a_chassis_frame_computation(kin) -> None:
    """Independent reference: move TRO into the CHASSIS frame (z + bump)."""
    tri0 = np.array(kin.solver_fl._tri_0)
    d0 = _tro(kin.solver_fl.solve()) - tri0
    d0 /= np.linalg.norm(d0)
    worst = 0.0
    inp, st = sla.FRONT_2027, stg.STEERING_2027
    for b in np.linspace(-inp.travel_droop_mm, inp.travel_bump_mm, 9):
        for r in np.linspace(-st.max_rack_travel_mm, st.max_rack_travel_mm, 9):
            fl, _ = kin.solve_pair(bump_mm=float(b), rack_mm=float(r))
            d = (_tro(fl) + np.array([0.0, 0.0, b])) - (tri0 + np.array([0.0, r, 0.0]))
            d /= np.linalg.norm(d)
            worst = max(worst, math.degrees(math.acos(float(np.clip(d @ d0, -1, 1)))))
    assert kin._worst_rod_end_misalignment() == pytest.approx(worst, abs=1e-9)


def test_worst_swing_2027_value(kin) -> None:
    """Pinned on the shipped 2027 front (was 5.51 deg before the frame fix)."""
    assert kin._worst_rod_end_misalignment() == pytest.approx(7.82, abs=0.01)
