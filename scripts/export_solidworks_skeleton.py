#!/usr/bin/env python3
r"""Draw the FSAE26 suspension skeleton as 3D sketches in SolidWorks.

The SolidWorks counterpart of ``altair_model/suspension.py``: same hardpoint
CSV, same member list, same four corners -- but instead of MotionView bodies
and cylinder graphics it creates **3D sketch lines**, one per suspension
member, so the wishbones, tie rods, kingpin axes and wheels are visible as a
wireframe you can measure, dimension and build parts against.

Two ways to get it into SolidWorks:

1. **Live (default).** SolidWorks is driven over COM, so the lines appear in
   the document that is already open, exactly like running the Altair script
   from MotionView's console::

       & .venv\Scripts\python.exe scripts\export_solidworks_skeleton.py

   That writes into whatever part or assembly is active -- add ``--new-part``
   to get an empty part instead. Nothing is ever saved: an unwanted run is
   undone by closing the document without saving.

2. **Macro.** Emit a VBA module that any teammate can run without Python::

       & .venv\Scripts\python.exe scripts\export_solidworks_skeleton.py --macro

   Then in SolidWorks: Tools > Macro > New (save a blank .swp anywhere), and
   in the VBA editor File > Import File... > pick the generated .bas, F5.

Re-running is safe. Sketches are matched by name and deleted before being
rebuilt, so pushing a revised CSV into an open model replaces the skeleton
instead of stacking a second one on top of it.

Scope: wireframe only. No solid bodies, no mates, no relations between the
sketch entities -- ``AddToDB`` is on precisely so SolidWorks does *not* infer
relations and nudge a point off its CSV coordinate. Nothing here is
parametric: edit the CSV and re-run, do not drag the endpoints.

Units: the CSV is mm; the SolidWorks API is **always metres** regardless of
the document's display units, so every coordinate is divided by 1000 on the
way in. Document units are therefore irrelevant and are not checked.

Frame: the CSV is ISO 8855 (X+ forward, Y+ LEFT, Z+ up, origin at the front
axle centreline on the ground plane). ``--frame iso`` (the default) puts those
numbers into SolidWorks untransformed; ``--frame sw_fsae`` applies the
"typical FSAE SolidWorks" convention from
``.claude/skills/vd-conventions/references/frames.md`` (X+ rearward, Y+ right,
Z+ up). If the team's CAD uses neither, add its matrix to ``FRAMES`` below --
do not eyeball a rotation afterwards.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

try:
    # Only needed to talk to SolidWorks; the --macro path works without it.
    from win32com.client.dynamic import CDispatch as _CDispatch
except ImportError:  # pragma: no cover - environment dependent
    _CDispatch = ()  # type: ignore[assignment, misc]

# Same source of truth as altair_model/suspension.py: the merged set written by
# scripts/geometry_summary.py from sla_geometry.py + steering_geometry.py.
DEFAULT_CSV = REPO / "Geometry Summary" / "hardpoints_2027_merged.csv"
DEFAULT_MACRO = REPO / "Geometry Summary" / "solidworks_skeleton.bas"

CORNERS = ("FL", "FR", "RL", "RR")

POINT_NAMES = (
    "UCA_IN_FRONT",
    "UCA_IN_REAR",
    "UCA_OUT",
    "LCA_IN_FRONT",
    "LCA_IN_REAR",
    "LCA_OUT",
    "TIE_ROD_IN",
    "TIE_ROD_OUT",
    "WHEEL_CENTER",
    "CONTACT_PATCH",
)

# Deliberately identical to LINKS in altair_model/suspension.py, so the
# SolidWorks wireframe and the MotionView model show the same skeleton. A
# wishbone is two legs meeting at its outer ball joint, hence two segments.
MEMBERS = (
    ("uca", "UCA", (
        ("UCA_IN_FRONT", "UCA_OUT", "front leg"),
        ("UCA_IN_REAR", "UCA_OUT", "rear leg"),
    )),
    ("lca", "LCA", (
        ("LCA_IN_FRONT", "LCA_OUT", "front leg"),
        ("LCA_IN_REAR", "LCA_OUT", "rear leg"),
    )),
    ("tierod", "Tie Rod", (
        ("TIE_ROD_IN", "TIE_ROD_OUT", "rod"),
    )),
    ("upright", "Upright", (
        ("LCA_OUT", "UCA_OUT", "kingpin axis"),
        ("LCA_OUT", "WHEEL_CENTER", "lower spindle arm"),
        ("UCA_OUT", "WHEEL_CENTER", "upper spindle arm"),
        ("LCA_OUT", "TIE_ROD_OUT", "steering arm"),
    )),
    ("wheel", "Wheel", (
        # Wheel centre to contact patch. Drawn from the CSV's own two wheel
        # points, so this line's lean *is* the static camber -- it is a
        # visual check, not a wheel model.
        ("WHEEL_CENTER", "CONTACT_PATCH", "loaded radius"),
    )),
)

# Row-major 3x3. Both entries are diagonal today, but the full matrix is kept
# so a rotated CAD frame can be added without changing the transform code.
FRAMES = {
    "iso": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "sw_fsae": ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
}

SKETCH_PREFIX = "SUSP_"

# swDocPART / swDocASSEMBLY. A 3D sketch cannot live in a drawing.
_SW_DOC_PART = 1
_SW_DOC_ASSEMBLY = 2
# swUserPreferenceStringValue_e.swDefaultTemplatePart
_SW_DEFAULT_TEMPLATE_PART = 8

MM_PER_M = 1000.0

# One sketch line: (label, start, end), endpoints in metres in the target frame.
Segment = tuple[str, tuple[float, float, float], tuple[float, float, float]]
# One 3D sketch: (feature name, its lines).
Group = tuple[str, list[Segment]]


def load_hardpoints(csv_path: Path) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Read the hardpoint CSV. Return ``(corner, point) -> (x, y, z)`` in mm.

    Raises on a malformed file or an incomplete corner, so a half-drawn
    skeleton is never pushed into someone's CAD model.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"hardpoint CSV not found: {csv_path}")

    hardpoints: dict[tuple[str, str], tuple[float, float, float]] = {}
    # utf-8-sig: Excel writes a BOM, which would otherwise corrupt the first
    # column name and make every row lookup fail.
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        wanted = ("corner", "point", "x_mm", "y_mm", "z_mm")
        absent = [c for c in wanted if c not in (reader.fieldnames or ())]
        if absent:
            raise ValueError(f"{csv_path.name}: missing column(s) {', '.join(absent)}")

        for line, row in enumerate(reader, start=2):
            key = (row["corner"].strip().upper(), row["point"].strip().upper())
            try:
                hardpoints[key] = (
                    float(row["x_mm"]),
                    float(row["y_mm"]),
                    float(row["z_mm"]),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{csv_path.name} line {line}: bad coordinate ({exc})") from exc

    missing = [
        f"{corner}/{name}"
        for corner in CORNERS
        for name in POINT_NAMES
        if (corner, name) not in hardpoints
    ]
    if missing:
        raise ValueError(f"{csv_path.name}: missing hardpoint(s): {', '.join(missing)}")

    return hardpoints


def _transform(xyz: tuple[float, float, float], matrix) -> tuple[float, float, float]:
    """Rotate/mirror a mm point into the target CAD frame."""
    x, y, z = xyz
    return tuple(row[0] * x + row[1] * y + row[2] * z for row in matrix)  # type: ignore[return-value]


def build_groups(
    hardpoints: dict[tuple[str, str], tuple[float, float, float]],
    frame: str,
    grouping: str,
) -> list[Group]:
    """Turn hardpoints into ``(sketch_name, [(label, start_m, end_m), ...])``.

    Coordinates come back in **metres**, already in the target frame, ready to
    hand to ``ISketchManager::CreateLine``. Grouping decides how many 3D
    sketches are created: one per corner (default -- hide a whole corner with
    one click), one per part (drives a part model from its own sketch), or one
    for the whole car.
    """
    matrix = FRAMES[frame]
    groups: dict[str, list[Segment]] = {}
    order: list[str] = []

    for corner in CORNERS:
        for key, part_label, segments in MEMBERS:
            if grouping == "corner":
                name = f"{SKETCH_PREFIX}{corner}"
            elif grouping == "part":
                name = f"{SKETCH_PREFIX}{corner}_{key.upper()}"
            else:
                name = f"{SKETCH_PREFIX}ALL"

            if name not in groups:
                groups[name] = []
                order.append(name)

            for start_pt, end_pt, segment_label in segments:
                start_mm = _transform(hardpoints[(corner, start_pt)], matrix)
                end_mm = _transform(hardpoints[(corner, end_pt)], matrix)
                if start_mm == end_mm:
                    # SolidWorks rejects a zero-length line; a duplicated
                    # hardpoint is a data bug worth naming, not swallowing.
                    raise ValueError(
                        f"{corner} {part_label} {segment_label}: "
                        f"{start_pt} and {end_pt} are the same point"
                    )
                groups[name].append(
                    (
                        f"{corner} {part_label} - {segment_label}",
                        tuple(v / MM_PER_M for v in start_mm),  # type: ignore[arg-type]
                        tuple(v / MM_PER_M for v in end_mm),  # type: ignore[arg-type]
                    )
                )

    return [(name, groups[name]) for name in order]


# --------------------------------------------------------------------------
# Live COM
# --------------------------------------------------------------------------


def _prop(obj, name, *args):
    """Read a SolidWorks API member that may be exposed as a property or a method.

    Late-bound pywin32 already invokes most no-argument SolidWorks members on
    attribute access and hands back the value (``model.GetTitle`` is a string,
    ``model.EditRebuild3`` performs the rebuild). Members that take arguments
    come back as callables instead.

    The trap is a member that returns a COM object, such as
    ``feature.GetSpecificFeature2``: it has *already* been invoked, but every
    ``CDispatch`` is itself callable, so a bare ``callable(attr)`` test calls
    the returned object's default member -- which does not exist, and raises a
    "member not found" that looks exactly like an unsupported API call.
    """
    attr = getattr(obj, name)
    if args:
        return attr(*args)
    if isinstance(attr, _CDispatch):
        return attr
    return attr() if callable(attr) else attr


def _features(model) -> list:
    """Return every feature in the tree.

    ``IModelDoc2::FirstFeature`` was removed by SolidWorks 2025 (rev 33.x), so
    the classic FirstFeature/GetNextFeature walk raises "member not found".
    ``IFeatureManager::GetFeatures`` is the supported replacement; the
    positional walk is the fallback for older builds that lack it.
    """
    try:
        features = model.FeatureManager.GetFeatures(False)
        if features:
            return list(features)
    except Exception:
        pass

    count = int(_prop(model, "GetFeatureCount") or 0)
    return [model.FeatureByPositionReverse(i) for i in range(count)]


def _connect():
    """Attach to a running SolidWorks, or start one."""
    try:
        import win32com.client as win32
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pywin32 is required for --live. Install it, or use --macro instead:\n"
            "    & .venv\\Scripts\\python.exe -m pip install pywin32"
        ) from exc

    try:
        app = win32.GetActiveObject("SldWorks.Application")
    except Exception:
        # Not running yet. Dispatch launches it, which takes a while.
        print("No running SolidWorks found -- starting one (this can take a minute)...")
        app = win32.Dispatch("SldWorks.Application")
    app.Visible = True
    return app


def _new_part(app):
    """Create an empty part from the default template."""
    template = app.GetUserPreferenceStringValue(_SW_DEFAULT_TEMPLATE_PART)
    if not template:
        raise RuntimeError(
            "No default part template is configured in SolidWorks, so a new "
            "part cannot be created. Open a part or assembly and drop --new-part."
        )
    model = app.NewDocument(template, 0, 0, 0)
    if model is None:
        raise RuntimeError(f"SolidWorks refused to create a part from {template!r}")
    return model


def _active_document(app, new_part: bool):
    """Return the target part/assembly.

    Defaults to whatever is already open, so the skeleton lands in the
    document you are working in. ``--new-part`` forces an empty part instead
    -- worth using the first time, rather than discovering how the wireframe
    looks inside the real chassis model.
    """
    if new_part:
        print("Creating a new part (--new-part).")
        return _new_part(app)

    model = app.ActiveDoc
    if model is None:
        print("No document open -- creating a new part from the default template.")
        return _new_part(app)

    doc_type = _prop(model, "GetType")
    if doc_type not in (_SW_DOC_PART, _SW_DOC_ASSEMBLY):
        raise RuntimeError(
            f"Active document is type {doc_type}; a 3D sketch needs a part or an assembly."
        )
    return model


def _delete_sketch_named(model, name: str) -> bool:
    """Delete a feature by exact name. Returns True if one was removed.

    Selecting the feature object directly avoids ``SelectByID2``, whose
    ``Callout`` argument rejects a bare Python ``None`` under late binding
    ("type mismatch", argument 8).
    """
    for feature in _features(model):
        try:
            feature_name = feature.Name
        except Exception:
            continue  # tree folders (Sensors, Annotations, ...) have no usable Name
        if feature_name == name:
            model.ClearSelection2(True)
            feature.Select2(False, 0)
            _prop(model, "EditDelete")
            model.ClearSelection2(True)
            return True
    return False


def build_live(
    groups: list[Group],
    csv_path: Path,
    frame: str,
    new_part: bool = False,
    screenshot: Path | None = None,
) -> None:
    """Create (or replace) the skeleton sketches in a SolidWorks document.

    Nothing is saved. The document is left dirty on purpose, so an unwanted
    run is undone by closing without saving.
    """
    app = _connect()
    model = _active_document(app, new_part)
    sketch_mgr = model.SketchManager

    print(f"Target document: {_prop(model, 'GetTitle')}")

    replaced = 0
    for name, _ in groups:
        if _delete_sketch_named(model, name):
            replaced += 1

    # AddToDB stops SolidWorks inferring relations and snapping endpoints onto
    # nearby geometry -- without it the hardpoints land *near* the CSV values.
    # DisplayWhenAdded off is purely speed.
    sketch_mgr.AddToDB = True
    sketch_mgr.DisplayWhenAdded = False

    lines = 0
    try:
        for name, segments in groups:
            model.ClearSelection2(True)
            sketch_mgr.Insert3DSketch(True)
            for label, start, end in segments:
                segment = sketch_mgr.CreateLine(
                    start[0], start[1], start[2], end[0], end[1], end[2]
                )
                if segment is None:
                    raise RuntimeError(f"SolidWorks refused to create line: {label}")
                lines += 1
            sketch_mgr.Insert3DSketch(True)  # toggles: closes the sketch

            feature = model.FeatureByPositionReverse(0)
            if feature is not None:
                feature.Name = name
    finally:
        # Leave the session usable even if a line failed halfway through.
        sketch_mgr.AddToDB = False
        sketch_mgr.DisplayWhenAdded = True

    # These take no arguments, so late-bound pywin32 exposes them as
    # properties -- reading the property is what performs the action.
    _prop(model, "EditRebuild3")
    model.ShowNamedView2("*Isometric", 7)
    _prop(model, "ViewZoomtofit2")

    print(f"Hardpoints read from {csv_path}")
    print(f"Frame: {frame}")
    print(f"Created {len(groups)} 3D sketch(es), {lines} line(s); replaced {replaced} existing.")
    print("Wireframe only -- no bodies, no mates, no sketch relations.")
    print("Nothing saved: close without saving to undo this run.")

    if screenshot is not None:
        screenshot.parent.mkdir(parents=True, exist_ok=True)
        if screenshot.suffix.lower() == ".bmp":
            # SaveAs3 rejects .bmp (returns 256); SaveBMP is its own call and
            # returns a plain success flag.
            ok = bool(model.SaveBMP(str(screenshot), 0, 0))
        else:
            # SaveAs3 returns swFileSaveError_e: 0 means no error. Truth-testing
            # the return value gets this exactly backwards.
            ok = model.SaveAs3(str(screenshot), 0, 0) == 0
        if ok and screenshot.is_file():
            print(f"Viewport image: {screenshot}")
        else:
            print(f"warning: SolidWorks would not write {screenshot}", file=sys.stderr)


# --------------------------------------------------------------------------
# VBA macro
# --------------------------------------------------------------------------


def _vba_identifier(name: str) -> str:
    return "Build_" + "".join(c if c.isalnum() else "_" for c in name)


def render_macro(groups: list[Group], csv_path: Path, frame: str) -> str:
    """Render a VBA module that builds the same sketches without Python.

    VBA number literals are locale-independent, so the '.' decimal separator
    below is correct even on a pt-BR SolidWorks install.
    """
    out: list[str] = []
    add = out.append

    add("' FSAE26 suspension skeleton -- 3D sketch wireframe")
    add(f"' Generated by scripts/export_solidworks_skeleton.py from {csv_path.name}")
    add(f"' Frame: {frame}. Coordinates in metres (the SolidWorks API is always metres).")
    add("' Do not hand-edit: change the hardpoint CSV and regenerate.")
    add("'")
    add("' Tools > Macro > New (save a blank .swp), then in the VBA editor")
    add("' File > Import File... > this .bas, then run main() with F5.")
    add("")
    add("Option Explicit")
    add("")
    add("Dim swApp As Object")
    add("Dim swModel As Object")
    add("Dim swSketchMgr As Object")
    add("")
    add("Sub main()")
    add("    Set swApp = Application.SldWorks")
    add("    Set swModel = swApp.ActiveDoc")
    add("")
    add("    If swModel Is Nothing Then")
    add('        MsgBox "Open a part or an assembly first, then run this macro."')
    add("        Exit Sub")
    add("    End If")
    add(f"    If swModel.GetType <> {_SW_DOC_PART} And swModel.GetType <> {_SW_DOC_ASSEMBLY} Then")
    add('        MsgBox "A 3D sketch needs a part or an assembly."')
    add("        Exit Sub")
    add("    End If")
    add("")
    add("    Set swSketchMgr = swModel.SketchManager")
    add("")
    add("    ' Replace any previous run rather than stacking a second skeleton.")
    for name, _ in groups:
        add(f'    DeleteSketch "{name}"')
    add("")
    add("    ' AddToDB keeps SolidWorks from inferring relations and snapping")
    add("    ' the endpoints off their CSV coordinates.")
    add("    swSketchMgr.AddToDB = True")
    add("    swSketchMgr.DisplayWhenAdded = False")
    add("")
    for name, _ in groups:
        add(f"    {_vba_identifier(name)}")
    add("")
    add("    swSketchMgr.AddToDB = False")
    add("    swSketchMgr.DisplayWhenAdded = True")
    add("    swModel.EditRebuild3")
    add("    swModel.ViewZoomtofit2")
    total = sum(len(segments) for _, segments in groups)
    add(f'    MsgBox "Suspension skeleton: {len(groups)} sketch(es), {total} line(s)."')
    add("End Sub")
    add("")
    add("Private Sub DeleteSketch(sketchName As String)")
    add("    swModel.ClearSelection2 True")
    add(
        '    If swModel.Extension.SelectByID2(sketchName, "SKETCH", '
        "0, 0, 0, False, 0, Nothing, 0) Then"
    )
    add("        swModel.EditDelete")
    add("    End If")
    add("    swModel.ClearSelection2 True")
    add("End Sub")

    for name, segments in groups:
        add("")
        add(f"Private Sub {_vba_identifier(name)}()")
        add("    Dim swFeat As Object")
        add("    swModel.ClearSelection2 True")
        add("    swSketchMgr.Insert3DSketch True")
        for label, start, end in segments:
            add(f"    ' {label}")
            coords = ", ".join(f"{v:.9g}" for v in (*start, *end))
            add(f"    swSketchMgr.CreateLine {coords}")
        add("    swSketchMgr.Insert3DSketch True")
        add("    Set swFeat = swModel.FeatureByPositionReverse(0)")
        add(f'    If Not swFeat Is Nothing Then swFeat.Name = "{name}"')
        add("End Sub")

    add("")
    return "\n".join(out)


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw the suspension hardpoint skeleton as 3D sketches in SolidWorks.",
    )
    parser.add_argument(
        "--csv", type=Path, default=DEFAULT_CSV,
        help=f"hardpoint CSV (default: {DEFAULT_CSV.name})",
    )
    parser.add_argument(
        "--frame", choices=sorted(FRAMES), default="iso",
        help="target CAD frame: 'iso' passes the CSV through untouched (default); "
             "'sw_fsae' is X+ rearward, Y+ right, Z+ up",
    )
    parser.add_argument(
        "--group", choices=("corner", "part", "single"), default="corner",
        help="one 3D sketch per corner (default), per part, or one for the whole car",
    )
    parser.add_argument(
        "--macro", nargs="?", type=Path, const=DEFAULT_MACRO, default=None,
        metavar="PATH",
        help=f"write a VBA .bas macro instead of driving SolidWorks "
             f"(default path: {DEFAULT_MACRO.name})",
    )
    parser.add_argument(
        "--new-part", action="store_true",
        help="build in a new empty part instead of the document already open",
    )
    parser.add_argument(
        "--screenshot", type=Path, default=None, metavar="PATH",
        help="also save the viewport to an image file (.png/.jpg/.bmp/.tif)",
    )
    args = parser.parse_args()

    try:
        hardpoints = load_hardpoints(args.csv)
        groups = build_groups(hardpoints, args.frame, args.group)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.macro is not None:
        args.macro.parent.mkdir(parents=True, exist_ok=True)
        args.macro.write_text(render_macro(groups, args.csv, args.frame), encoding="utf-8")
        total = sum(len(segments) for _, segments in groups)
        print(f"Wrote {args.macro}")
        print(f"{len(groups)} sketch(es), {total} line(s), frame {args.frame}.")
        print("In SolidWorks: Tools > Macro > New (save a blank .swp), then in the")
        print("VBA editor File > Import File... > this .bas, then F5.")
        return 0

    try:
        build_live(groups, args.csv, args.frame, args.new_part, args.screenshot)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
