"""scripts/rocker_concepts.py: the push/pull rod construction and its packaging checks."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from vdcore.analysis.rocker import rocker_sweep

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rc():
    spec = importlib.util.spec_from_file_location("rocker_concepts",
                                                  REPO / "scripts" / "rocker_concepts.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rocker_concepts"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ctx(rc):
    return rc.load_ctx()


PULL_FRONT = dict(rod_type="pullrod", mount_body="uca", rod_mount=(-10.0, 500.0, 375.0),
                  pivot=(150.0, 200.0, 175.0), damper_toward=(450.0, 170.0, 160.0),
                  rod_arm_mm=50.0)


def test_pullrod_reaches_the_target_mr_and_compresses_in_bump(rc, ctx) -> None:
    """The old builder only placed the damper arm for a pushrod: a pullrod came out with
    the damper extending in bump (negative MR), so no positive target was reachable."""
    c = rc.AxleConcept(**PULL_FRONT)
    spec = rc.make_spec(ctx.geo, ctx.project, "front", c)
    corner = ctx.geo.corner("FL")
    sw = rocker_sweep(corner, rc.S.layout_for(spec, "FL"), [-5.0, 0.0, 5.0])
    assert sw.motion_ratio[1] == pytest.approx(0.80, abs=1e-6)
    assert sw.damper_length_mm[2] < sw.damper_length_mm[1] < sw.damper_length_mm[0]
    # the rod is in tension: in bump the UCA mount moves away from the low rocker
    st0, st1 = sw.states[1], sw.states[2]
    d_static = np.linalg.norm(st0.rocker_rod - st0.rod_mount)
    assert np.linalg.norm(st1.rocker_rod - st1.rod_mount) == pytest.approx(d_static, abs=1e-6)
    pivot = np.asarray(spec.rocker_pivot)
    assert np.linalg.norm(st1.rod_mount - pivot) > np.linalg.norm(st0.rod_mount - pivot)


def test_pushrod_matches_the_engine_builder(rc, ctx) -> None:
    """For a pushrod the construction is the engine's build_rocker, unchanged."""
    args = ((9.0, 542.0, 145.0), (20.0, 180.0, 535.0), (560.0, 175.0, 500.0), 50.0, 60.0, 195.0,
            "lca", "pushrod")
    assert rc.build(*args, flip=False) == rc.S.build_rocker(*args)


def test_equal_damper_margins_and_seed_metrics(rc, ctx) -> None:
    c = rc.AxleConcept(**PULL_FRONT)
    specs = {"front": rc.make_spec(ctx.geo, ctx.project, "front", c),
             "rear": ctx.project.rocker_rear}
    res = rc.evaluate(ctx, specs, rc.reference_z(ctx))
    f = res["front"]
    assert f["droop_margin_mm"] == pytest.approx(f["bump_margin_mm"], abs=0.05)
    assert f["rod_type"] == "pullrod"
    assert f["rod_in_tension"] is True
    assert np.isnan(f["wheel_load_to_buckle_N"])            # tension member
    assert abs(f["rocker_moment_check_Nmm"]) < 1e-6 * f["rod_force_ride_N"] * 50.0
    assert res["cg"]["delta_cg_mm"] < 0.0                   # a low front rocker lowers the CG
    # the physical range reaches past the +-25 mm design travel to the damper end stops
    z_ext, z_comp = f["end_stops_mm"]
    assert z_ext < -25.0 < 25.0 < z_comp
    # the seed's pushrod is in compression
    assert res["rear"]["rod_in_tension"] is False


def test_frame_model_flags_sketch_only_lines(rc, ctx) -> None:
    """The front box's top side tubes are 3D-sketch lines with no weldment body in Xasis
    33.3 (kept, they may be planned tubes); 2D layout sketches with no body (templates,
    ground lines) are references, not tubes; the front bulkhead is bodied."""
    top = [m for m in ctx.members if m.name == "Tri_Frente"
           and round(float(m.a[0]), 1) == 559.6 and abs(float(m.a[1])) == 175.0
           and float(m.b[0]) < -80.0]
    assert len(top) == 2 and not any(m.bodied or m.reference for m in top)
    fbh = [m for m in ctx.members if m.name == "front bulkhead"]
    assert fbh and all(m.bodied and abs(float(m.a[0]) - 559.6) < 1.0 for m in fbh)
    refs = [m for m in ctx.members if m.name in ("Esboço8", "Esboço10", "head restraint")]
    assert refs and all(m.reference for m in refs)


def test_engine_sits_on_its_chassis_tabs(rc, ctx) -> None:
    """MOTOR_FIXADO placed by its four mounts: the fit closes, the engine's +x runs to the
    car's right (-y), and its own mount sketches land on the chassis tabs."""
    R, t, worst = rc.engine_placement(ctx.project)
    assert worst < 0.5
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0
    assert R[:, 0] @ np.array([0.0, -1.0, 0.0]) > 0.999
    names = [h.name for h in ctx.engine]
    assert names[0].startswith("engine") and "intake plenum" in names
    # the front-left mount arc (Esboço3) lands on the left 3DSketch8 tab end
    assert np.allclose(R @ np.array([66.0, 387.5, 159.5]) + t, [-1063.3, 164.0, 399.4],
                       atol=1.0)


def test_rack_is_at_its_tie_rod_inner_x(rc, ctx) -> None:
    """steering.rack_x_mm is rearward-positive: the rack sits at ISO x = -rack_x_mm."""
    assert -ctx.project.steering.rack_x_mm == pytest.approx(
        float(ctx.geo.p("FL", "TIE_ROD_IN")[0]), abs=0.5)


def test_clearance_helpers(rc) -> None:
    boxes = (((0.0, 10.0), 0.0, 5.0, (0.0, 10.0)), ((100.0, 110.0), 50.0, 5.0, (0.0, 10.0)))
    assert rc.seg_box((5, 0, 5), (5, 1, 5), boxes) == 0.0
    assert rc.seg_box((20, 0, 5), (30, 0, 5), boxes) == pytest.approx(10.0)
    assert rc.seg_box((105, 60, 5), (105, 70, 5), boxes) == pytest.approx(5.0)
    assert rc.pt_seg((0, 1, 0), (-1, 0, 0), (1, 0, 0)) == pytest.approx(1.0)
    a, b = rc.trim((0, 0, 0), (100, 0, 0), 10.0, 35.0)
    assert a[0] == pytest.approx(10.0) and b[0] == pytest.approx(65.0)


def test_frame_nodes_are_the_welded_joints(rc, ctx) -> None:
    """Nodes are joints of 3+ bodied tube ends; the front hoop's foot is the busiest."""
    assert all(n.valence >= 3 for n in ctx.nodes)
    d, n = rc.nearest_node(ctx, (-120.4, 205.0, 117.4))
    assert d < 1.0 and n.valence == 9 and "FRONTO HO" in n.tubes
    # the floor X brace crosses at the centre line: a node with no tube end in it
    d, n = rc.nearest_node(ctx, (0.0, 0.0, 117.4))
    assert d < 1.0 and n.valence == 4
    # a mid-span point is far from every node
    assert rc.nearest_node(ctx, (330.0, 215.0, 415.0))[0] > 150.0


def test_damper_end_aims_the_coilover_through_a_point(rc, ctx) -> None:
    end = (-105.9, 251.0, 319.5)
    c = rc.AxleConcept("pushrod", "upright", (5.5, 572.75, 182.49), (67.2, 274.3, 414.8),
                       (461.9, 311.7, 362.1), 62.8, damper_end=end)
    sp = rc.make_spec(ctx.geo, ctx.project, "front", c)
    rd, dc = np.asarray(sp.rocker_damper), np.asarray(sp.damper_chassis)
    u = (dc - rd) / np.linalg.norm(dc - rd)
    assert np.linalg.norm(np.cross(np.asarray(end) - rd, u)) < 0.01
    sw = rocker_sweep(ctx.geo.corner("FL"), rc.S.layout_for(sp, "FL"), [0.0])
    assert sw.motion_ratio[0] == pytest.approx(0.80, abs=1e-6)
