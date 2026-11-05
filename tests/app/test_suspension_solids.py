"""scripts/export_suspension_solids.py: the SolidWorks primitive-solids macro."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ss():
    sys.path.insert(0, str(REPO / "scripts"))
    return _load("export_suspension_solids")


@pytest.fixture(scope="module")
def project(ss):
    return ss.load_project(REPO / "projects" / "FSAE_2027.vdproj.json")


@pytest.fixture(scope="module")
def parts(ss, project):
    return ss.all_parts(project)


def test_every_member_is_a_named_finite_cylinder(ss, parts) -> None:
    names = [p.name for p in parts]
    assert len(names) == len(set(names))
    assert all(n.startswith(ss.PREFIX) for n in names)
    for p in parts:
        a, b = np.array(p.a), np.array(p.b)
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        assert np.linalg.norm(b - a) > 1.0 and p.od_mm > 0
        assert p.family in ss.COLOURS
    for cid in ss.CORNERS:
        for part in ("UCA_front_leg", "LCA_rear_leg", "upright_kingpin", "tyre", "rocker_boss",
                     "coilover_body", "coilover_spring"):
            assert f"{ss.PREFIX}{cid}_{part}" in names


def test_left_and_right_are_mirrors(ss, parts) -> None:
    by = {p.name: p for p in parts}
    for p in parts:
        for lc, rc in (("FL", "FR"), ("RL", "RR")):
            if f"_{lc}_" in p.name:
                q = by[p.name.replace(f"_{lc}_", f"_{rc}_")]
                m = np.array([1.0, -1.0, 1.0])
                if "bracket" in p.name:          # the frame is not symmetric everywhere
                    continue
                pa, pb = np.array(p.a) * m, np.array(p.b) * m
                # the same cylinder; an axis mirrored as an axial vector swaps its ends
                assert (np.allclose(pa, q.a, atol=1e-6) and np.allclose(pb, q.b, atol=1e-6)) \
                    or (np.allclose(pa, q.b, atol=1e-6) and np.allclose(pb, q.a, atol=1e-6))


def test_wishbone_legs_end_on_the_hardpoints(ss, project, parts) -> None:
    geo = ss.G.resolve(project)
    by = {p.name: p for p in parts}
    for cid in ss.CORNERS:
        P = geo.points[cid]
        leg = by[f"{ss.PREFIX}{cid}_LCA_front_leg"]
        assert np.allclose(leg.a, P["LCA_IN_FRONT"]) and np.allclose(leg.b, P["LCA_OUT"])


def test_chassis_frame_puts_the_pickups_on_the_cad_nodes(ss, project, parts) -> None:
    """In the chassis frame, the LCA's inboard pick-ups land on the CAD frame nodes."""
    frame = ss.X.cad_frame(project)
    raw = json.loads((REPO / "projects" / "cad" / "Xasis 33.3_survey.json").read_text("utf-8"))
    nodes = np.array([pt for sk in raw["sketches"] if sk["name"] == "PSD"
                      for ln in sk["lines"] for pt in (ln["a"], ln["b"])])
    by = {p.name: p for p in parts}
    a_m = ss._m(by[f"{ss.PREFIX}FL_LCA_front_leg"].a, frame)
    d = np.min(np.linalg.norm(nodes - np.array(a_m) * 1000.0, axis=1))
    assert d < 0.1


def test_macro_text(ss, project, parts) -> None:
    sk = ss.skeleton_lines(project)
    text = ss.render_macro(parts, sk, ss.X.cad_frame(project), "test")
    assert "\r\n" in text and text.isascii()
    assert len(re.findall(r'^    Cyl "', text, flags=re.M)) == len(parts)
    assert "Calibrate" in text and "CreateFeatureFromBody3" in text
    # every Sub/Function closes, and no procedure is near VBA's 64 KB limit
    assert text.count("\nSub ") + text.count("\nPrivate Sub ") == text.count("\nEnd Sub")
    for proc in re.split(r"\r\nEnd (?:Sub|Function)", text):
        assert len(proc) < 60_000


def test_reservoir_is_drawn_where_the_evaluator_checks_it(ss, project) -> None:
    """``--reservoir``: the canister beside the chassis-end body, the same capsule axis as
    ``rocker_concepts.Linkage.reservoir`` on the left, its mirror image on the right."""
    import rocker_concepts as RC

    from apps.vdstudio.engine import suspension as S

    assert ss.parse_reservoir(["front=+1,120", "rear=-1,0"]) == {"front": (1.0, 120.0),
                                                                  "rear": (-1.0, 0.0)}
    with pytest.raises(SystemExit):
        ss.parse_reservoir(["front=2,0"])
    res = {"front": (1.0, 120.0), "rear": (-1.0, 30.0)}
    parts = {p.name: p for p in ss.all_parts(project, None, res)}
    assert not any("reservoir" in n for n in (p.name for p in ss.all_parts(project)))
    d = project.damper
    for cid, axle in (("FL", "front"), ("RL", "rear")):
        spec = project.rocker_front if axle == "front" else project.rocker_rear
        lay = S.layout_for(spec, cid)
        lk = RC.Linkage(0.0, np.array(lay.rod_mount), np.array(lay.rocker_rod),
                        np.array(lay.rocker_damper), np.array(lay.rocker_pivot),
                        np.array(lay.damper_chassis), RC._unit(lay.rocker_axis))
        right = "FR" if cid == "FL" else "RR"
        for c, link in ((cid, lk), (right, lk.mirrored())):
            a, b, _r = link.reservoir(d, res[axle][0], False, res[axle][1])
            q = parts[f"{ss.PREFIX}{c}_coilover_reservoir"]
            assert q.od_mm == pytest.approx(d.reservoir_od_mm)
            assert (np.array(q.a) + np.array(q.b)) / 2 == pytest.approx((a + b) / 2, abs=1e-9)
