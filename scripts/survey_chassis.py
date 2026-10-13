#!/usr/bin/env python3
r"""Dump a SolidWorks chassis model's geometry to JSON, to check it against the hardpoints.

The app cannot read ``.SLDPRT`` files: only SolidWorks can. This script asks
SolidWorks for everything needed to place the suspension skeleton on the
chassis and to compare pick-up points, then writes it to one JSON file that
can be analysed anywhere (no SolidWorks needed afterwards):

* every sketch (2D and 3D): its lines, arcs and points, in **model space, mm**.
  A weldment's layout sketch is the tube centrelines, so this is the frame;
* reference points, reference axes and coordinate systems, by name;
* the part's and each body's bounding box;
* the feature tree (name and type), to see how the model is built;
* an isometric screenshot next to the JSON.

Usage (Windows, SolidWorks installed)::

    & .venv\Scripts\python.exe scripts\survey_chassis.py --file "C:\path\Chassis.SLDPRT"
    & .venv\Scripts\python.exe scripts\survey_chassis.py          # the open document

The file is opened read-only and closed again; a document that was already
open in SolidWorks is read in place and left open. Nothing is modified or
saved. Output: ``projects/cad/<file name>_survey.json`` (and ``.png``).

Coordinates are the CAD model's own frame: nothing here assumes ISO 8855.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from export_solidworks_skeleton import MM_PER_M, REPO, _prop  # noqa: E402
from import_solidworks_hardpoints import (  # noqa: E402
    _connect,
    _features,
    _open_document,
    _sketch_to_model,
)

OUT_DIR = REPO / "projects" / "cad"
SCHEMA = 1

# swSketchSegments_e
_SEG_LINE, _SEG_ARC = 0, 1
_SEG_NAMES = {0: "line", 1: "arc", 2: "ellipse", 3: "spline", 4: "text", 5: "parabola"}
# swBodyType_e.swSolidBody
_SW_SOLID_BODY = 0
SKETCH_TYPES = ("ProfileFeature", "3DProfileFeature")


def _xyz(v: Any) -> tuple[float, float, float]:
    """A point from the API: an ISketchPoint/IMathPoint object or a 3-number array."""
    if hasattr(v, "X"):
        return (float(_prop(v, "X")), float(_prop(v, "Y")), float(_prop(v, "Z")))
    if hasattr(v, "ArrayData"):
        v = _prop(v, "ArrayData")
    a = list(v)
    return (float(a[0]), float(a[1]), float(a[2]))


def _mm(p: tuple[float, float, float]) -> list[float]:
    return [round(c * MM_PER_M, 4) for c in p]


def _box_mm(raw: Any) -> list[float] | None:
    if raw is None:
        return None
    a = [float(v) for v in raw]
    return [round(v * MM_PER_M, 3) for v in a[:6]] if len(a) >= 6 else None


class Survey:
    """Collects the model's geometry; every failure is recorded, never fatal."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def guard(self, what: str, fn: Callable[[], Any], default: Any = None) -> Any:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- COM raises anything; record and go on
            self.errors.append(f"{what}: {type(exc).__name__}: {exc}")
            return default

    # -- sketches ---------------------------------------------------------------------
    def sketch(self, name: str, type_name: str, feature: Any) -> dict[str, Any]:
        sk = _prop(feature, "GetSpecificFeature2")
        to_model = _sketch_to_model(sk)
        out: dict[str, Any] = {"name": name, "type": "3d" if type_name.startswith("3D")
                               else "2d", "lines": [], "arcs": [], "points": [], "other": {}}
        for i, seg in enumerate(_prop(sk, "GetSketchSegments") or ()):
            kind = int(_prop(seg, "GetType"))
            if kind == _SEG_LINE:
                line = self.guard(f"{name} line {i}", lambda seg=seg: (
                    _mm(to_model(_xyz(_prop(seg, "GetStartPoint2")))),
                    _mm(to_model(_xyz(_prop(seg, "GetEndPoint2"))))))
                if line is not None:
                    construction = bool(self.guard(
                        f"{name} line {i} construction",
                        lambda seg=seg: _prop(seg, "ConstructionGeometry"), False))
                    out["lines"].append({"a": line[0], "b": line[1],
                                         "construction": construction})
            elif kind == _SEG_ARC:
                arc = self.guard(f"{name} arc {i}", lambda seg=seg: {
                    "center": _mm(to_model(_xyz(_prop(seg, "GetCenterPoint2")))),
                    "start": _mm(to_model(_xyz(_prop(seg, "GetStartPoint2")))),
                    "end": _mm(to_model(_xyz(_prop(seg, "GetEndPoint2")))),
                    "radius_mm": round(float(_prop(seg, "GetRadius")) * MM_PER_M, 4),
                })
                if arc is not None:
                    out["arcs"].append(arc)
            else:
                key = _SEG_NAMES.get(kind, str(kind))
                out["other"][key] = out["other"].get(key, 0) + 1
        for j, pt in enumerate(_prop(sk, "GetSketchPoints2") or ()):
            p = self.guard(f"{name} point {j}", lambda pt=pt: _mm(to_model(_xyz(pt))))
            if p is not None:
                out["points"].append(p)
        return out

    # -- reference geometry -----------------------------------------------------------
    def ref_point(self, feature: Any) -> list[float]:
        rp = _prop(feature, "GetSpecificFeature2")
        return _mm(_xyz(_prop(rp, "GetRefPoint")))

    def ref_axis(self, feature: Any) -> list[list[float]]:
        ax = _prop(feature, "GetSpecificFeature2")
        a = [float(v) for v in _prop(ax, "GetRefAxisParams")]
        return [_mm((a[0], a[1], a[2])), _mm((a[3], a[4], a[5]))]

    def coord_sys(self, model: Any, name: str) -> dict[str, Any]:
        t = _prop(model.Extension, "GetCoordinateSystemTransformByName", name)
        a = [float(v) for v in _prop(t, "ArrayData")]
        return {"rotation_rows": [a[0:3], a[3:6], a[6:9]],
                "origin_mm": _mm((a[9], a[10], a[11])), "scale": a[12],
                "note": "SolidWorks MathTransform ArrayData, translation in mm"}

    # -- bodies -----------------------------------------------------------------------
    def bodies(self, model: Any) -> list[dict[str, Any]]:
        out = []
        for i, body in enumerate(_prop(model, "GetBodies2", _SW_SOLID_BODY, False) or ()):
            out.append({
                "name": self.guard(f"body {i} name", lambda body=body: str(_prop(body, "Name")),
                                   f"body {i}"),
                "box_mm": self.guard(f"body {i} box",
                                     lambda body=body: _box_mm(_prop(body, "GetBodyBox"))),
            })
        return out

    def run(self, model: Any) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema": SCHEMA,
            "tool": "scripts/survey_chassis.py",
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "units": "mm, the CAD model's own frame (nothing assumes ISO 8855)",
            "title": self.guard("title", lambda: str(_prop(model, "GetTitle")), ""),
            "path": self.guard("path", lambda: str(_prop(model, "GetPathName")), ""),
            "part_box_mm": self.guard("part box",
                                      lambda: _box_mm(_prop(model, "GetPartBox", True))),
            "bodies": self.guard("bodies", lambda: self.bodies(model), []),
            "features": [], "sketches": [], "ref_points": [], "ref_axes": [],
            "coord_systems": [],
        }
        for feature in _features(model):
            try:
                name = str(feature.Name)
                type_name = str(_prop(feature, "GetTypeName2"))
            except Exception:
                continue                       # tree folders have no usable name
            result["features"].append([name, type_name])
            if type_name in SKETCH_TYPES:
                s = self.guard(f"sketch {name}", lambda f=feature, n=name, t=type_name:
                               self.sketch(n, t, f))
                if s is not None:
                    result["sketches"].append(s)
            elif type_name == "RefPoint":
                p = self.guard(f"ref point {name}", lambda f=feature: self.ref_point(f))
                if p is not None:
                    result["ref_points"].append({"name": name, "xyz": p})
            elif type_name == "RefAxis":
                a = self.guard(f"ref axis {name}", lambda f=feature: self.ref_axis(f))
                if a is not None:
                    result["ref_axes"].append({"name": name, "a": a[0], "b": a[1]})
            elif type_name == "CoordSys":
                c = self.guard(f"coordinate system {name}",
                               lambda n=name: self.coord_sys(model, n))
                if c is not None:
                    result["coord_systems"].append({"name": name, **c})
        result["errors"] = self.errors
        return result


def summary(data: dict[str, Any]) -> str:
    n_lines = sum(len(s["lines"]) for s in data["sketches"])
    n_pts = sum(len(s["points"]) for s in data["sketches"])
    box = data.get("part_box_mm")
    size = (" x ".join(f"{box[i + 3] - box[i]:.0f}" for i in range(3)) + " mm (X x Y x Z)"
            if box else "unknown")
    return (f"{data['title']}: {len(data['features'])} features, "
            f"{len(data['sketches'])} sketches ({n_lines} lines, {n_pts} points), "
            f"{len(data['ref_points'])} reference points, {len(data['coord_systems'])} "
            f"coordinate systems, {len(data['bodies'])} solid bodies; size {size}; "
            f"{len(data['errors'])} read error(s).")


def _already_open(app: Any, path: Path) -> Any:
    try:
        return _prop(app, "GetOpenDocumentByName", str(path))
    except Exception:
        return None


def _screenshot(model: Any, png: Path) -> bool:
    try:
        model.ShowNamedView2("*Isometric", 7)
        _prop(model, "ViewZoomtofit2")
        return model.SaveAs3(str(png), 0, 0) == 0
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Dump a SolidWorks chassis model's geometry "
                                             "to JSON (model frame, mm).")
    ap.add_argument("--file", type=Path, default=None, metavar="PATH",
                    help="the .SLDPRT/.SLDASM to read (default: the open document)")
    ap.add_argument("--out", type=Path, default=None, metavar="JSON",
                    help="output file (default: projects/cad/<name>_survey.json)")
    ap.add_argument("--no-image", action="store_true", help="skip the screenshot")
    args = ap.parse_args(argv)

    model = app = None
    opened_here = False
    try:
        app = _connect()
        if args.file is not None:
            model = _already_open(app, args.file.resolve())
        if model is None:
            model, opened_here = _open_document(app, args.file)
        print(f"Reading {_prop(model, 'GetTitle')} ...")
        data = Survey().run(model)
        stem = Path(data["path"] or data["title"] or "model").stem
        out = args.out or OUT_DIR / f"{stem}_survey.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=1), encoding="utf-8")
        png = out.with_suffix(".png")
        shot = False if args.no_image else _screenshot(model, png)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if opened_here and model is not None and app is not None:
            with contextlib.suppress(Exception):
                app.CloseDoc(_prop(model, "GetTitle"))
    print(summary(data))
    print(f"Wrote {out}" + (f"\nWrote {png}" if shot else ""))
    for e in data["errors"][:10]:
        print(f"  read error: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
