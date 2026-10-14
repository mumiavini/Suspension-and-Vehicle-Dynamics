"""VD Studio engine: hardpoint I/O in every frame, report wording, setup sheet."""

from __future__ import annotations

import pytest

from apps.vdstudio.engine import exports as X
from apps.vdstudio.engine import geometry as G
from apps.vdstudio.engine import kpis as K
from apps.vdstudio.engine.concept import concept_kpis
from apps.vdstudio.engine.project import CORNERS, INPUT_POINTS, Project


@pytest.fixture(scope="module")
def geo() -> G.CarGeometry:
    return G.resolve(Project())


def _inputs(geo):
    return {c: {n: geo.points[c][n] for n in INPUT_POINTS} for c in CORNERS}


@pytest.mark.parametrize("frame", ["iso", "sw_fsae", "optimumk"])
def test_csv_round_trip_in_every_frame(geo, tmp_path, frame) -> None:
    path = tmp_path / f"hp_{frame}.csv"
    path.write_text(X.hardpoints_csv(geo, frame, extended=True), encoding="utf-8")
    pts, src, tol = X.read_hardpoints(path, frame)
    assert X.diff_points(_inputs(geo), pts, 1e-3) == []
    assert src["FL"]["UCA_OUT"] == "design_intent"
    assert tol["FL"]["UCA_OUT"] == 0.5


def test_reading_in_the_wrong_frame_gives_a_mirrored_car(geo, tmp_path) -> None:
    path = tmp_path / "hp.csv"
    path.write_text(X.hardpoints_csv(geo, "sw_fsae"), encoding="utf-8")
    pts, _, _ = X.read_hardpoints(path, "iso")
    assert X.diff_points(_inputs(geo), pts, 1e-3)    # loudly different, not silently equal


def test_csv_matches_the_repository_merged_file(geo, tmp_path) -> None:
    """The ISO export is the format of Geometry Summary/hardpoints_2027_merged.csv."""
    from apps.vdstudio._paths import REPO

    ref = REPO / "legacy_app" / "carro_formula_2027.csv"
    pts, _, _ = X.read_hardpoints(ref, "iso")
    assert X.diff_points(_inputs(geo), pts, 1e-3) == []


def test_optimumk_text_negates_y(geo) -> None:
    txt = X.optimumk_text(geo)
    assert "UCA_CHASSIS_FRONT" in txt and "corner FR" in txt
    line = next(ln for ln in txt.splitlines() if "UCA_CHASSIS_FRONT" in ln)
    assert float(line.split()[2]) == pytest.approx(-geo.points["FL"]["UCA_IN_FRONT"][1], abs=0.01)


def test_solidworks_macro_contains_every_corner(geo) -> None:
    m = X.solidworks_macro(geo, "iso", "corner")
    for c in CORNERS:
        assert f"SUSP_{c}" in m
    assert m.count("CreateLine") == 4 * 10


def test_report_states_and_never_recommends(geo) -> None:
    p = Project()
    ks = K.compute_static(geo)
    md = X.markdown_report(p, geo, ks, concept_kpis(p))
    low = md.lower()
    for word in ("recommend", "should use", "optimal", "best value"):
        assert word not in low
    assert "design_intent" in md and "ISO 8855" in md
    assert "Kingpin length (true 3D)" in md          # the known out-of-band item is listed


def test_setup_sheet_derived_rows_need_inputs(geo) -> None:
    p = Project()
    ks = K.compute_static(geo)
    rows = X.setup_sheet_rows(p, geo, ks)
    wheel_rate = next(r for r in rows if r[1] == "Wheel rate")
    assert wheel_rate[3] == "—"                       # no spring / MR given -> no number
    p.setup_sheet.update({"spring_f": "40", "mr_f": "0.8"})
    rows = X.setup_sheet_rows(p, geo, ks)
    wheel_rate = next(r for r in rows if r[1] == "Wheel rate")
    assert float(wheel_rate[3]) == pytest.approx(40 * 0.8 ** 2)
