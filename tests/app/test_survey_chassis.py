"""scripts/survey_chassis.py against fake SolidWorks COM objects (no SolidWorks here)."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace as NS

import pytest

import apps.vdstudio  # noqa: F401  (puts scripts/ on sys.path)

sc = importlib.import_module("survey_chassis")


class Pt:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.X, self.Y, self.Z = x, y, z


class BrokenFeature:
    """A sketch whose COM member fails, like a missing interface on a late-bound object."""

    Name = "Broken sketch"
    GetTypeName2 = "ProfileFeature"

    @property
    def GetSpecificFeature2(self):  # noqa: N802
        raise RuntimeError("member not found")


def _feature(name, type_name, specific):
    return NS(Name=name, GetTypeName2=type_name, GetSpecificFeature2=specific)


def _model():
    # A 3D sketch moved +100 mm along X: sketch = model + T, so model = sketch - T.
    moved = NS(ArrayData=[1, 0, 0, 0, 1, 0, 0, 0, 1, 0.1, 0.0, 0.0, 1.0])
    line = NS(GetType=0, GetStartPoint2=Pt(0.1, 0.0, 0.0), GetEndPoint2=Pt(0.6, 0.2, 0.3),
              ConstructionGeometry=False)
    arc = NS(GetType=1, GetCenterPoint2=[0.1, 0.0, 0.5], GetStartPoint2=[0.15, 0.0, 0.5],
             GetEndPoint2=[0.1, 0.05, 0.5], GetRadius=0.05)
    spline = NS(GetType=3)
    layout = NS(ModelToSketchTransform=moved, GetSketchSegments=[line, arc, spline],
                GetSketchPoints2=[Pt(0.1, 0.0, 0.0), Pt(0.6, 0.2, 0.3)])
    features = [
        NS(),                                                     # a folder: no Name
        _feature("Layout", "3DProfileFeature", layout),
        _feature("Point1", "RefPoint", NS(GetRefPoint=NS(ArrayData=[0.2, -0.3, 0.05]))),
        _feature("Axis1", "RefAxis", NS(GetRefAxisParams=[0, 0, 0, 0, 0, 1.5])),
        _feature("Coordinate System1", "CoordSys", None),
        BrokenFeature(),
        _feature("Weld", "WeldMemberFeat", None),
    ]

    def cs(name):
        assert name == "Coordinate System1"
        return NS(ArrayData=[0, 1, 0, -1, 0, 0, 0, 0, 1, 0.5, 0.25, 0.0, 1.0])

    return NS(
        GetTitle="Xasis 33.3.SLDPRT", GetPathName="C:\\x\\Xasis 33.3.SLDPRT",
        GetPartBox=lambda _vis: [-0.3, -1.2, 0.0, 0.3, 1.2, 1.1],
        GetBodies2=lambda _t, _v: [NS(Name="Tube<1>", GetBodyBox=[0, 0, 0, 0.1, 0.2, 0.3])],
        Extension=NS(GetCoordinateSystemTransformByName=cs),
        FeatureManager=NS(GetFeatures=lambda _top: features),
    )


def test_survey_reads_everything_in_model_space_mm() -> None:
    data = sc.Survey().run(_model())
    assert data["part_box_mm"] == [-300.0, -1200.0, 0.0, 300.0, 1200.0, 1100.0]
    assert data["bodies"] == [{"name": "Tube<1>", "box_mm": [0.0, 0.0, 0.0, 100.0, 200.0, 300.0]}]
    (layout,) = data["sketches"]
    assert layout["name"] == "Layout" and layout["type"] == "3d"
    # the sketch's transform is undone: sketch (0.1, 0, 0) m is the model origin
    assert layout["lines"] == [{"a": [0.0, 0.0, 0.0], "b": [500.0, 200.0, 300.0],
                                "construction": False}]
    assert layout["arcs"][0] == {"center": [0.0, 0.0, 500.0], "start": [50.0, 0.0, 500.0],
                                 "end": [0.0, 50.0, 500.0], "radius_mm": 50.0}
    assert layout["points"] == [[0.0, 0.0, 0.0], [500.0, 200.0, 300.0]]
    assert layout["other"] == {"spline": 1}
    assert data["ref_points"] == [{"name": "Point1", "xyz": [200.0, -300.0, 50.0]}]
    assert data["ref_axes"] == [{"name": "Axis1", "a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 1500.0]}]
    (c,) = data["coord_systems"]
    assert c["name"] == "Coordinate System1" and c["origin_mm"] == [500.0, 250.0, 0.0]
    assert ["Weld", "WeldMemberFeat"] in data["features"]
    # a feature that cannot be read is reported, not fatal and not silently dropped
    assert any("Broken sketch" in e and "member not found" in e for e in data["errors"])
    assert "2 sketches" not in sc.summary(data) and "1 sketches (1 lines, 2 points)" in \
        sc.summary(data)


def test_main_writes_json_and_closes_only_what_it_opened(tmp_path, monkeypatch, capsys) -> None:
    model = _model()
    closed = []
    app = NS(CloseDoc=closed.append, GetOpenDocumentByName=lambda _p: None)
    monkeypatch.setattr(sc, "_connect", lambda: app)
    monkeypatch.setattr(sc, "_open_document", lambda a, p: (model, True))
    out = tmp_path / "chassis.json"
    assert sc.main(["--file", str(tmp_path / "Xasis.SLDPRT"), "--out", str(out),
                    "--no-image"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == 2 and data["title"] == "Xasis 33.3.SLDPRT"
    assert closed == ["Xasis 33.3.SLDPRT"]
    assert "Wrote" in capsys.readouterr().out

    # already open in SolidWorks: read in place, never closed under the user
    closed.clear()
    app.GetOpenDocumentByName = lambda _p: model
    monkeypatch.setattr(sc, "_open_document",
                        lambda a, p: pytest.fail("must not reopen an open document"))
    assert sc.main(["--file", str(tmp_path / "Xasis.SLDPRT"), "--out", str(out),
                    "--no-image"]) == 0
    assert closed == []


def test_main_reports_a_failure(monkeypatch, capsys) -> None:
    def boom():
        raise RuntimeError("pywin32 is required to talk to SolidWorks.")

    monkeypatch.setattr(sc, "_connect", boom)
    assert sc.main(["--no-image"]) == 1
    assert "pywin32 is required" in capsys.readouterr().err


def test_drag_and_drop_batch() -> None:
    raw = (sc.REPO / "Survey chassis CAD.bat").read_bytes()
    assert b"\n" not in raw.replace(b"\r\n", b"")                       # CRLF only
    text = raw.decode("ascii")
    assert 'scripts\\survey_chassis.py --file "%MODEL%"' in text
    assert "--inexact" in text and 'cd /d "%~dp0"' in text


def test_2d_sketch_on_a_rotated_plane_lands_in_model_space() -> None:
    """The front-bulkhead case from the real chassis: a plane parallel to the Top plane.

    IMathTransform uses row vectors, p_sketch = p_model @ R + T. With
    R = [[1,0,0],[0,0,1],[0,-1,0]] and T = (0, 0, 0.68) m the plane is model
    Y = -680 mm and the model corner (175, -680, -425) mm is sketch (175, 425, 0).
    Reading it with R transposed gave (175, +680, +425): the mirror seen in the
    survey of "Xasis 33.3".
    """
    t = NS(ArrayData=[1, 0, 0, 0, 0, 1, 0, -1, 0, 0.0, 0.0, 0.68, 1.0])
    sk = NS(ModelToSketchTransform=t, GetSketchSegments=[], GetSketchPoints2=[
        Pt(0.175, 0.425, 0.0), Pt(-0.175, 0.075, 0.0)])
    out = sc.Survey().sketch("front bulkhead", "ProfileFeature", NS(GetSpecificFeature2=sk))
    assert out["points"] == [[175.0, -680.0, -425.0], [-175.0, -680.0, -75.0]]
