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
    assert np.isnan(f["wheel_load_to_buckle_N"])            # tension member
    assert res["cg"]["delta_cg_mm"] < 0.0                   # a low front rocker lowers the CG


def test_clearance_helpers(rc) -> None:
    box = ((0.0, 10.0), 5.0, (0.0, 10.0))
    assert rc.seg_box((5, 0, 5), (5, 1, 5), box) == 0.0
    assert rc.seg_box((20, 0, 5), (30, 0, 5), box) == pytest.approx(10.0)
    assert rc.pt_seg((0, 1, 0), (-1, 0, 0), (1, 0, 0)) == pytest.approx(1.0)
    a, b = rc.trim((0, 0, 0), (100, 0, 0), 10.0, 35.0)
    assert a[0] == pytest.approx(10.0) and b[0] == pytest.approx(65.0)
