"""VD Studio engine: the sweep / what-if engine fills tables and never decides.

Target queries are checked against the back-solvers the scripts already ship
(rack height for zero bump steer, UCA rise for an anti-dive target).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from apps.vdstudio.engine import whatif as W
from apps.vdstudio.engine.geometry import freeze_to_hardpoints
from apps.vdstudio.engine.project import Project


def test_param_registry_paths_resolve() -> None:
    p = Project()
    for par in W.params_for(p):
        assert math.isfinite(par.get(p)), par.path
    p.hardpoints = freeze_to_hardpoints(p)
    p.geometry_mode = "hardpoints"
    hp = W.params_for(p)
    assert any(x.mirror for x in hp)
    assert not any(x.path.startswith("front.rc_height") for x in hp)


def test_mirrored_hardpoint_param_keeps_the_car_symmetric() -> None:
    p = Project()
    p.hardpoints = freeze_to_hardpoints(p)
    p.geometry_mode = "hardpoints"
    q = W.param_by_path("hardpoints.points.FL.TIE_ROD_IN.1").apply(p, 280.0)
    assert q.hardpoints.points["FL"]["TIE_ROD_IN"][1] == 280.0
    assert q.hardpoints.points["FR"]["TIE_ROD_IN"][1] == -280.0
    q = W.param_by_path("hardpoints.points.FL.TIE_ROD_IN.2").apply(p, 160.0)
    assert q.hardpoints.points["FR"]["TIE_ROD_IN"][2] == 160.0
    assert p.hardpoints.points["FL"]["TIE_ROD_IN"][2] != 160.0   # original untouched


def test_grid_is_full_factorial_and_unranked() -> None:
    g = W.run_grid(Project(), ["front.rc_height_mm", "front.fvsa_length_mm"],
                   [[30.0, 40.0], [1400.0, 1600.0]], ["front.rc_static", "front.fvsa"])
    assert len(g.points) == 4
    assert [pt.assignment for pt in g.points] == [(30, 1400), (30, 1600), (40, 1400), (40, 1600)]
    arr = g.array("front.rc_static")
    assert arr.shape == (2, 2)
    assert arr[1, 0] > arr[0, 0]        # higher RC target -> higher solved RC
    assert np.all(np.isfinite(arr))


def test_grid_reports_unbuildable_points_as_failures() -> None:
    g = W.run_grid(Project(), ["front.travel_bump_mm"], [[25.0, 400.0]], ["front.camber_gain"])
    assert math.isfinite(g.points[0].values["front.camber_gain"])
    assert not math.isfinite(g.points[1].values["front.camber_gain"])
    assert g.points[1].status["front.camber_gain"] == "fail"


def test_sensitivity_sign_and_magnitude() -> None:
    s = W.sensitivity(Project(), ["front.rc_height_mm"], ["front.rc_static"])
    # moving the RC target by +5 mm moves the solved (instant-axis) RC by +6.46 mm, not
    # ~+5 as with the projected arm lines: the rack height stays where it nulls bump steer
    # at 35 mm, and the bump steer the edit creates tilts the force line (2026-10-12)
    assert s.per_step("front.rc_static", "front.rc_height_mm") == pytest.approx(6.46, abs=0.05)


def test_target_query_matches_the_bump_steer_back_solver() -> None:
    p = Project()
    r = W.target_query(p, "steering.rack_z_mm", "front.bump_steer", 0.0, 140.0, 175.0)
    assert len(r.roots) == 1
    # STEERING_2027.rack_z_mm was back-solved to null the (central-difference) rate;
    # vdcore's least-squares rate nulls a hair lower. Same knob, same answer to 0.05 mm.
    assert r.roots[0] == pytest.approx(p.steering.rack_z_mm, abs=0.05)
    assert "Nothing has been applied" in r.message


def test_target_query_matches_the_anti_dive_back_solver() -> None:
    import sla_geometry as sla

    p = Project()
    r = W.target_query(p, "front.dz_uca_mm", "front.anti", 7.5, -30.0, 10.0)
    assert len(r.roots) == 1
    ref = sla.dz_uca_for_anti(sla.FRONT_2027, sla.VEHICLE_2027, 7.5)
    # merged corner carries caster; sla's construction uses zero-caster ball joints
    assert r.roots[0] == pytest.approx(ref, abs=0.1)


def test_target_query_without_a_crossing_says_so() -> None:
    r = W.target_query(Project(), "front.rc_height_mm", "front.rc_static", 500.0, 20.0, 40.0,
                       n_scan=3)
    assert r.roots == []
    # states what was searched, never that the target is unreachable
    assert "No sign change" in r.message
    assert "not reachable" not in r.message


def test_parallel_runner_gives_the_serial_answer() -> None:
    p = Project()
    args = (["front.kpi_deg"], [[8.0, 10.0, 12.0, 14.0]], ["front.scrub_L", "front.rc_static"])
    serial = W.run_grid(p, *args, runner=W.Runner(parallel=False))
    runner = W.Runner(parallel=True)
    try:
        par = W.run_grid(p, *args, runner=runner)
    finally:
        runner.close()
    for a, b in zip(serial.points, par.points, strict=True):
        assert a.values == pytest.approx(b.values)
