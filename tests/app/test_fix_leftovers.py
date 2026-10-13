"""Audit leftovers fixed after the group merges (import validation, SolidWorks frame,
provenance of the non-point inputs, the steering-page query dialog)."""

from __future__ import annotations

import pytest

from apps.vdstudio.engine import exports as X
from apps.vdstudio.engine.concept import apply_cg_estimate, cg_estimate
from apps.vdstudio.engine.geometry import freeze_to_hardpoints, resolve
from apps.vdstudio.engine.project import Project, project_from_dict, project_to_dict

HEADER = "corner,point,x_mm,y_mm,z_mm,source,tol_mm\n"


def _write(tmp_path, rows: str):
    p = tmp_path / "hp.csv"
    p.write_text(HEADER + rows, encoding="utf-8")
    return p


def test_import_normalises_and_validates_the_source(tmp_path) -> None:
    pts, src, _ = X.read_hardpoints(_write(tmp_path, "FL,UCA_OUT,0,500,380,CAD,0.5\n"))
    assert src["FL"]["UCA_OUT"] == "cad"                     # upper case accepted, stored lower
    with pytest.raises(ValueError, match="unknown source 'scanned'"):
        X.read_hardpoints(_write(tmp_path, "FL,UCA_OUT,0,500,380,scanned,0.5\n"))


def test_import_refuses_a_duplicated_point(tmp_path) -> None:
    rows = "FL,UCA_OUT,0,500,380,cad,0.5\nFL,UCA_OUT,999,500,380,cad,0.5\n"
    with pytest.raises(ValueError, match="FL UCA_OUT already given on line 2"):
        X.read_hardpoints(_write(tmp_path, rows))


def test_solidworks_export_names_the_supported_frames() -> None:
    geo = resolve(Project())
    with pytest.raises(ValueError, match="SolidWorks export supports the frames"):
        X.solidworks_macro(geo, "optimumk")
    assert "CreateLine" in X.solidworks_macro(geo, "iso")


def test_tyre_radius_is_an_estimate_until_ttc_data() -> None:
    geo = resolve(Project())
    assert {a.tire_source for a in geo.axles.values()} == {"estimate"}
    assert geo.estimate_count() == 2                       # one tyre radius per axle
    assert freeze_to_hardpoints(Project()).front.tire_source == "estimate"


def test_applying_the_cg_estimate_marks_the_mass_as_estimate() -> None:
    p = Project()
    q = apply_cg_estimate(p, cg_estimate(p.mass_items, p.vehicle.wheelbase_mm))
    assert q.mass_source == "estimate" and p.mass_source == "design_intent"
    assert resolve(q).estimate_count() == 3
    back = project_from_dict(project_to_dict(q))
    assert (back.mass_source, back.tire_source) == ("estimate", "estimate")


def test_old_project_files_get_the_provenance_defaults() -> None:
    d = project_to_dict(Project())
    del d["mass_source"], d["tire_source"]
    p = project_from_dict(d)
    assert (p.mass_source, p.tire_source) == ("design_intent", "estimate")
