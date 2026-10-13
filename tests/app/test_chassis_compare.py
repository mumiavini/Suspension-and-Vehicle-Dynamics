"""Chassis CAD survey vs project pick-ups (engine/chassis.py, scripts/compare_chassis.py)."""

from __future__ import annotations

import importlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from apps.vdstudio.engine import chassis as C
from apps.vdstudio.engine import exports as X
from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine.project import CORNERS, Project, save_project

swx = X.swx
FRAME = swx.axis_frame("-Y", "-X", (0.0, -120.4, 75.0))     # the "Xasis 33.3" placement


@pytest.fixture(scope="module")
def geo():
    return G.resolve(Project())


def _survey(geo, moved=None, schema=2, frame=FRAME) -> dict:
    """A fake survey: one 3D frame sketch whose nodes ARE the project's pick-ups, placed
    in CAD with ``frame``; ``moved`` = (corner, point, iso offset) displaces one node."""
    lines = []
    for c in CORNERS:
        pts = {n: np.asarray(geo.points[c][n], float) for n in C.CHASSIS_POINTS}
        if moved and moved[0] == c:
            pts[moved[1]] = pts[moved[1]] + np.asarray(moved[2], float)
        cad = {n: list(X.to_cad(tuple(p), frame)) for n, p in pts.items()}
        for a, b in (("UCA_IN_FRONT", "UCA_IN_REAR"), ("LCA_IN_FRONT", "LCA_IN_REAR"),
                     ("UCA_IN_FRONT", "LCA_IN_FRONT"), ("UCA_IN_REAR", "LCA_IN_REAR")):
            lines.append({"a": cad[a], "b": cad[b], "construction": False})
    lines.append({"a": [0, 0, 0], "b": [0, 5000, 0], "construction": True})   # ignored
    return {"schema": schema, "title": "fake", "path": "C:\\fake.SLDPRT", "sketches": [
        {"name": "Frame", "type": "3d", "lines": lines},
        {"name": "Hoop", "type": "2d", "lines": [{"a": [0, 0, 0], "b": [1, 1, 1],
                                                  "construction": False}]}]}


def test_identical_points_match_and_one_moved_point_is_found(geo, tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(_survey(geo, moved=("RL", "LCA_IN_FRONT", (-32.0, 40.0, -1.0)))))
    rows = C.compare(geo, C.load_survey(path), FRAME)
    assert len(rows) == 16
    by = {(r.corner, r.point): r for r in rows}
    moved = by[("RL", "LCA_IN_FRONT")]
    assert moved.delta_mm == pytest.approx((-32.0, 40.0, -1.0), abs=1e-9)
    assert moved.distance_mm == pytest.approx(np.linalg.norm([-32, 40, -1]))
    assert all(r.distance_mm < 1e-9 for k, r in by.items() if k != ("RL", "LCA_IN_FRONT"))
    assert by[("FL", "UCA_IN_FRONT")].tube_mm < 1e-9


def test_wrong_placement_reads_as_large_differences(geo, tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(_survey(geo)))
    right = C.compare(geo, C.load_survey(path), FRAME)
    wrong = C.compare(geo, C.load_survey(path), swx.axis_frame("-Y", "-X", (0.0, 0.0, 75.0)))
    assert max(r.distance_mm for r in right) < 1e-9
    # front axle mislocated by 120.4 mm: no pick-up finds its own node any more
    assert float(np.median([min(r.distance_mm, 1e3) for r in wrong])) > 30.0


def test_schema_1_skips_2d_sketches(geo, tmp_path) -> None:
    for schema, skipped in ((1, 1), (2, 0)):
        path = tmp_path / f"s{schema}.json"
        path.write_text(json.dumps(_survey(geo, schema=schema)))
        s = C.load_survey(path)
        assert s.skipped_2d == skipped and len(s.lines) == 16 + (1 - skipped)


def test_cad_to_iso_inverts_the_placement() -> None:
    p = (123.0, -45.0, 300.0)
    assert C.cad_to_iso(X.to_cad(p, FRAME), FRAME) == pytest.approx(p)
    # the real chassis: front LCA mount node in CAD -> ISO, 117.4 mm above the ground
    assert C.cad_to_iso((-205.0, -240.8, -42.4), FRAME) == pytest.approx((120.4, 205.0, 117.4))


def test_reports(geo, tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(_survey(geo, moved=("FL", "UCA_IN_FRONT", (0.0, 45.0, 0.0)))))
    survey = C.load_survey(path)
    rows = C.compare(geo, survey, FRAME)
    md = C.report_markdown(rows, survey, FRAME, "proj", "x.svg")
    assert "| FL | UCA_IN_FRONT |" in md and "**45.0**" in md and "forward -Y, left -X" in md
    assert C.report_csv(rows).splitlines()[0].startswith("corner,point,project_x_mm")
    root = ET.fromstring(C.views_svg(geo, survey, FRAME, rows))       # well-formed SVG
    assert root.tag.endswith("svg") and len(root) > 50


def test_cli_writes_next_to_a_dotted_survey_name(geo, tmp_path, capsys) -> None:
    cli = importlib.import_module("compare_chassis")
    survey = tmp_path / "Xasis 33.3_survey.json"
    survey.write_text(json.dumps(_survey(geo)))
    proj = tmp_path / "p.vdproj.json"
    from dataclasses import replace

    from apps.vdstudio.engine.project import CadFrame
    save_project(replace(Project(), cad_frame=CadFrame("-Y", "-X", 0.0, -120.4, 75.0)), proj)
    assert cli.main([str(survey), "--project", str(proj)]) == 0
    for ext in ("md", "csv", "svg"):
        assert (tmp_path / f"Xasis 33.3_comparison.{ext}").is_file()
    assert "| FL | LCA_IN_FRONT |" in capsys.readouterr().out


def _proj():
    from dataclasses import replace

    from apps.vdstudio.engine.project import CadFrame
    return replace(Project(), cad_frame=CadFrame("-Y", "-X", 0.0, -120.4, 75.0))


def test_cad_hardpoints_take_only_the_chassis_pickups(geo, tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(_survey(geo, moved=("FL", "UCA_IN_FRONT", (0.0, 45.0, 8.0)))))
    hs, rows = C.cad_hardpoints(_proj(), C.load_survey(path), FRAME, ("Frame",))
    assert hs.points["FL"]["UCA_IN_FRONT"] == pytest.approx(
        np.add(geo.points["FL"]["UCA_IN_FRONT"], (0.0, 45.0, 8.0)))
    assert hs.sources["FL"]["UCA_IN_FRONT"] == "cad" and hs.tols["FL"]["UCA_IN_FRONT"] == 0.5
    # the CAD has no uprights or wheels: those keep the project's value and provenance
    assert hs.points["FL"]["UCA_OUT"] == pytest.approx(geo.points["FL"]["UCA_OUT"])
    assert hs.sources["FL"]["UCA_OUT"] == "design_intent"
    assert not hs.missing() and "fake" in hs.origin
    # the variant solves
    from copy import copy

    v = copy(_proj())
    v.hardpoints, v.geometry_mode = hs, "hardpoints"
    assert G.resolve(v).points["FL"]["UCA_IN_FRONT"] == pytest.approx(
        hs.points["FL"]["UCA_IN_FRONT"])


def test_cad_hardpoints_refuse_to_guess(geo, tmp_path) -> None:
    path = tmp_path / "s.json"
    path.write_text(json.dumps(_survey(geo)))
    with pytest.raises(ValueError, match="no frame lines in the sketches"):
        C.cad_hardpoints(_proj(), C.load_survey(path), FRAME, ("PSD",))
    with pytest.raises(ValueError, match="no CAD node within"):
        C.cad_hardpoints(_proj(), C.load_survey(path),
                         swx.axis_frame("-Y", "-X", (0.0, -120.4, 500.0)), None, 50.0)
    # one node for two pick-ups: refused
    d = _survey(geo)
    lines = d["sketches"][0]["lines"]
    lines[:] = [ln for ln in lines if not ln.get("construction")][:1]
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        C.cad_hardpoints(_proj(), C.load_survey(path), FRAME, None, 1e6)


def test_import_cli_writes_a_variant_and_leaves_the_project(geo, tmp_path, capsys) -> None:
    cli = importlib.import_module("import_chassis_pickups")
    survey = tmp_path / "Xasis 33.3_survey.json"
    survey.write_text(json.dumps(_survey(geo, moved=("RL", "LCA_IN_FRONT", (-32.0, 40.0, 0.0)))))
    proj = tmp_path / "team.vdproj.json"
    save_project(_proj(), proj)
    before = proj.read_bytes()
    out = tmp_path / "variant.vdproj.json"
    assert cli.main([str(survey), "--project", str(proj), "--out", str(out)]) == 0
    assert proj.read_bytes() == before
    from apps.vdstudio.engine.project import load_project

    v = load_project(out)
    assert v.geometry_mode == "hardpoints" and v.hardpoints.sources["RL"]["LCA_IN_FRONT"] == "cad"
    text = capsys.readouterr().out
    assert "RL LCA_IN_FRONT" in text and "Front RC height" in text
