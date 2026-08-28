#!/usr/bin/env python3
r"""Read suspension hardpoints back out of SolidWorks into a hardpoint CSV.

The inverse of ``scripts/export_solidworks_skeleton.py``. It reads the 3D
sketches whose feature names start with ``SUSP_``, works out which hardpoint
each sketch point is, and writes ``hardpoints_solidworks.csv`` in exactly the
format of ``Geometry Summary/hardpoints_2027_merged.csv``::

    corner,point,x_mm,y_mm,z_mm
    FL,UCA_IN_FRONT,120.000,175.000,308.580

Usage::

    # from the document already open in SolidWorks
    & .venv\Scripts\python.exe scripts\import_solidworks_hardpoints.py

    # from a file on disk (opened read-only, closed again afterwards)
    & .venv\Scripts\python.exe scripts\import_solidworks_hardpoints.py --file "part.SLDPRT"

    # what sketches does this model actually contain?
    & .venv\Scripts\python.exe scripts\import_solidworks_hardpoints.py --survey

The point of this direction is the ``--compare`` diff. Once someone drags a
hardpoint in CAD, the CSV and the CAD model disagree and nothing says so.
Reading CAD back and diffing names the hardpoint and the delta::

    & .venv\Scripts\python.exe scripts\import_solidworks_hardpoints.py --compare

**Naming is reconstructed from topology, not from sketch order.** A corner is
a connected component of the line graph, and inside it every hardpoint has a
unique vertex degree: the lower ball joint has 5 lines meeting on it, the
upper 4, the wheel centre 3, the outer tie rod end 2. The four remaining
single-line ends are named by which of those they hang off. So the CSV comes
back correct even if the sketch was rebuilt, reordered, or partly redrawn by
hand -- and if the topology does *not* match a double wishbone, the script
raises instead of emitting a plausible-looking CSV with swapped names.

What it cannot do: pull hardpoints out of an arbitrary CAD model. It needs the
skeleton this project's exporter draws (or a hand-built 3D sketch with the
same ten lines and a ``SUSP_`` name). ``--survey`` lists what a model holds so
you can see whether that is there.

Frame: ``--frame`` states the frame the *CAD* is in, and the coordinates are
transformed back to ISO 8855 for the CSV -- so a model built with
``--frame sw_fsae`` must be read back with ``--frame sw_fsae``. Reading it as
``iso`` yields a mirrored car with no error, which is exactly the kind of
silent sign bug this repo keeps getting bitten by.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Shared with the exporter on purpose: the member topology, the frame
# matrices and the corner/point vocabulary must never drift between the two
# directions, or a round trip silently renames hardpoints.
from export_solidworks_skeleton import (  # noqa: E402
    CORNERS,
    FRAMES,
    MEMBERS,
    MM_PER_M,
    POINT_NAMES,
    REPO,
    SKETCH_PREFIX,
    _prop,
)

DEFAULT_OUT = REPO / "Geometry Summary" / "hardpoints_solidworks.csv"
DEFAULT_REFERENCE = REPO / "Geometry Summary" / "hardpoints_2027_merged.csv"

# Two sketch endpoints closer than this are the same hardpoint. Generous
# enough to survive CAD round-off, far tighter than any real design change.
WELD_TOL_MM = 1e-4

# swSketchSegments_e.swSketchLINE
_SW_SKETCH_LINE = 0
# swOpenDocOptions_e: read-only + silent, so opening a teammate's part cannot
# grab a write lock or pop a dialog on a headless run.
_SW_OPEN_READ_ONLY = 1
_SW_OPEN_SILENT = 8
# swFileLoadWarning_ReadOnly -- expected, since read-only is what we asked for.
_SW_WARN_READ_ONLY = 256
# swDocPART / swDocASSEMBLY, for ISldWorks::OpenDoc6
_SW_DOC_PART = 1
_SW_DOC_ASSEMBLY = 2

Vec = tuple[float, float, float]
# One CAD line as read back: (sketch name, start, end), millimetres, CAD frame.
RawSegment = tuple[str, Vec, Vec]
# An edge of the welded point graph, as a pair of node indices.
Edge = tuple[int, int]


def expected_topology() -> tuple[dict[str, int], list[tuple[str, str]]]:
    """Vertex degree per hardpoint, and the edge list, derived from MEMBERS.

    Computed rather than hard-coded so that editing MEMBERS in the exporter
    cannot leave the importer decoding a shape that is no longer drawn.
    """
    edges: list[tuple[str, str]] = []
    for _key, _label, segments in MEMBERS:
        for start, end, _seg_label in segments:
            edges.append((start, end))

    degree = dict.fromkeys(POINT_NAMES, 0)
    for start, end in edges:
        degree[start] += 1
        degree[end] += 1
    return degree, edges


def _inverse_frame(matrix) -> tuple[tuple[float, ...], ...]:
    """Transpose. Every entry in FRAMES is orthonormal, so transpose inverts."""
    return tuple(tuple(matrix[r][c] for r in range(3)) for c in range(3))


def _apply(matrix, xyz: Vec) -> Vec:
    x, y, z = xyz
    return tuple(row[0] * x + row[1] * y + row[2] * z for row in matrix)  # type: ignore[return-value]


# --------------------------------------------------------------------------
# SolidWorks reading
# --------------------------------------------------------------------------


def _connect():
    try:
        import win32com.client as win32
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pywin32 is required to talk to SolidWorks.") from exc

    try:
        return win32.GetActiveObject("SldWorks.Application")
    except Exception:
        print("No running SolidWorks found -- starting one (this can take a minute)...")
        app = win32.Dispatch("SldWorks.Application")
        app.Visible = True
        return app


def _open_document(app, path: Path | None):
    """Return (model, opened_here). Opens `path` read-only if given."""
    if path is None:
        model = app.ActiveDoc
        if model is None:
            raise RuntimeError(
                "No document is open in SolidWorks. Open one, or pass --file."
            )
        return model, False

    import pythoncom
    from win32com.client import VARIANT

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SolidWorks file not found: {path}")
    doc_type = _SW_DOC_ASSEMBLY if path.suffix.lower() == ".sldasm" else _SW_DOC_PART

    # OpenDoc6's last two arguments are ByRef out-parameters (swFileLoadError_e
    # and swFileLoadWarning_e). Passing plain ints raises "type mismatch".
    errors = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warnings = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    model = app.OpenDoc6(
        str(path), doc_type, _SW_OPEN_READ_ONLY | _SW_OPEN_SILENT, "", errors, warnings
    )
    if model is None:
        raise RuntimeError(
            f"SolidWorks could not open {path} "
            f"(swFileLoadError_e={errors.value}, warning={warnings.value})"
        )
    if warnings.value and warnings.value != _SW_WARN_READ_ONLY:
        print(f"note: SolidWorks reported load warning {warnings.value}", file=sys.stderr)
    return model, True


def _features(model) -> list:
    """Every feature in the tree (IModelDoc2::FirstFeature is gone in SW 2025)."""
    try:
        features = model.FeatureManager.GetFeatures(False)
        if features:
            return list(features)
    except Exception:
        pass
    count = int(_prop(model, "GetFeatureCount") or 0)
    return [model.FeatureByPositionReverse(i) for i in range(count)]


def _sketch_to_model(sketch):
    """Return a function mapping sketch-space points into model space.

    ``ISketchPoint`` coordinates live in the sketch's own frame. A 3D sketch
    created at the origin has an identity transform, but one that was moved
    does not -- and reading those coordinates as model space would place every
    hardpoint wrong by a constant offset, with no error anywhere.
    """
    try:
        transform = _prop(sketch, "ModelToSketchTransform")
        data = list(_prop(transform, "ArrayData"))
    except Exception:
        return lambda p: p  # no transform available: assume model space

    # ArrayData: [0:9] rotation (row-major), [9:12] translation, [12] scale.
    rotation = [data[0:3], data[3:6], data[6:9]]
    translation = data[9:12]
    scale = data[12] or 1.0

    rotation_is_identity = all(
        abs(rotation[i][j] - (1.0 if i == j else 0.0)) < 1e-12
        for i in range(3)
        for j in range(3)
    )
    identity = (
        rotation_is_identity
        and all(abs(t) < 1e-12 for t in translation)
        and abs(scale - 1.0) < 1e-12
    )
    if identity:
        return lambda p: p

    def to_model(point: Vec) -> Vec:
        # Inverse of p_sketch = scale * R @ p_model + T, with R orthonormal.
        shifted = [(point[i] - translation[i]) / scale for i in range(3)]
        return tuple(  # type: ignore[return-value]
            sum(rotation[r][c] * shifted[r] for r in range(3)) for c in range(3)
        )

    return to_model


def read_segments(model, prefix: str) -> list[RawSegment]:
    """Return (sketch_name, start_mm, end_mm) for every line in a SUSP_ sketch.

    Coordinates come back in model space, in millimetres, still in the CAD's
    own frame.
    """
    segments: list[RawSegment] = []
    for feature in _features(model):
        try:
            name = feature.Name
        except Exception:
            continue  # tree folders have no usable Name
        if not name.startswith(prefix):
            continue

        sketch = _prop(feature, "GetSpecificFeature2")
        if sketch is None:
            continue
        raw = _prop(sketch, "GetSketchSegments")
        if not raw:
            continue
        to_model = _sketch_to_model(sketch)

        for segment in raw:
            if int(_prop(segment, "GetType")) != _SW_SKETCH_LINE:
                continue  # arcs/splines are not suspension members
            start = _prop(segment, "GetStartPoint2")
            end = _prop(segment, "GetEndPoint2")
            a = to_model((_prop(start, "X"), _prop(start, "Y"), _prop(start, "Z")))
            b = to_model((_prop(end, "X"), _prop(end, "Y"), _prop(end, "Z")))
            segments.append(
                (
                    name,
                    tuple(v * MM_PER_M for v in a),  # type: ignore[arg-type]
                    tuple(v * MM_PER_M for v in b),  # type: ignore[arg-type]
                )
            )
    return segments


def survey(model, prefix: str) -> None:
    """Print every sketch in the model and how many line segments it holds."""
    print(f"{'feature':<40s} {'type':<20s} {'lines':>6s} {'other':>6s}")
    print("-" * 76)
    found = 0
    for feature in _features(model):
        try:
            name = feature.Name
            type_name = _prop(feature, "GetTypeName2")
        except Exception:
            continue
        if type_name not in ("ProfileFeature", "3DProfileFeature"):
            continue
        found += 1
        sketch = _prop(feature, "GetSpecificFeature2")
        raw = _prop(sketch, "GetSketchSegments") if sketch is not None else None
        lines = other = 0
        for segment in raw or ():
            if int(_prop(segment, "GetType")) == _SW_SKETCH_LINE:
                lines += 1
            else:
                other += 1
        marker = "  <-- read by this script" if name.startswith(prefix) else ""
        print(f"{name:<40s} {type_name:<20s} {lines:>6d} {other:>6d}{marker}")
    if not found:
        print("(no sketches in this model)")


# --------------------------------------------------------------------------
# Topology decode
# --------------------------------------------------------------------------


def _weld(segments: list[RawSegment], tol: float) -> tuple[list[Vec], list[Edge]]:
    """Collapse coincident endpoints into a node list plus an edge list."""
    nodes: list[Vec] = []

    def node_index(point: Vec) -> int:
        for index, existing in enumerate(nodes):
            if all(abs(point[i] - existing[i]) <= tol for i in range(3)):
                return index
        nodes.append(point)
        return len(nodes) - 1

    edges: list[tuple[int, int]] = []
    for _name, start, end in segments:
        a, b = node_index(start), node_index(end)
        if a == b:
            raise ValueError(f"zero-length line at {tuple(round(v, 3) for v in start)}")
        edge = (min(a, b), max(a, b))
        if edge not in edges:
            edges.append(edge)
    return nodes, edges


def _components(node_count: int, edges: list[Edge]) -> list[list[int]]:
    """Connected components. Each one is a suspension corner."""
    adjacency: dict[int, set[int]] = {i: set() for i in range(node_count)}
    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    seen: set[int] = set()
    groups: list[list[int]] = []
    for start in range(node_count):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            group.append(current)
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        groups.append(sorted(group))
    return groups


def _name_corner(nodes: list[Vec], group: list[int], edges: list[Edge]) -> dict[str, Vec]:
    """Map one connected component's nodes onto the ten hardpoint names."""
    degree_of, _ = expected_topology()

    adjacency: dict[int, set[int]] = {i: set() for i in group}
    for a, b in edges:
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)

    degrees = {i: len(adjacency[i]) for i in group}
    want = sorted(degree_of.values())
    got = sorted(degrees.values())
    if got != want:
        raise ValueError(
            f"corner near {tuple(round(v, 1) for v in nodes[group[0]])} is not a "
            f"double wishbone: vertex degrees {got}, expected {want}. "
            "The sketch has extra or missing lines -- refusing to guess names."
        )

    # Each of these degrees occurs exactly once in a double wishbone, so the
    # four hubs are identified outright; see expected_topology().
    by_degree = {degrees[i]: i for i in group}
    lca_out = by_degree[degree_of["LCA_OUT"]]
    uca_out = by_degree[degree_of["UCA_OUT"]]
    wheel_center = by_degree[degree_of["WHEEL_CENTER"]]
    tie_rod_out = by_degree[degree_of["TIE_ROD_OUT"]]

    def leaves(hub: int) -> list[int]:
        return [n for n in adjacency[hub] if degrees[n] == 1]

    def split_front_rear(hub: int, label: str) -> tuple[int, int]:
        pair = leaves(hub)
        if len(pair) != 2:
            raise ValueError(f"{label}: expected 2 inboard pivots, found {len(pair)}")
        first, second = pair
        if nodes[first][0] == nodes[second][0]:
            raise ValueError(
                f"{label}: both inboard pivots are at x={nodes[first][0]:.3f}; "
                "front and rear cannot be told apart."
            )
        # ISO 8855 X+ is forward, so the front pivot is the one with greater X.
        return (first, second) if nodes[first][0] > nodes[second][0] else (second, first)

    uca_front, uca_rear = split_front_rear(uca_out, "UCA")
    lca_front, lca_rear = split_front_rear(lca_out, "LCA")

    tie_rod_in = leaves(tie_rod_out)
    contact_patch = leaves(wheel_center)
    if len(tie_rod_in) != 1 or len(contact_patch) != 1:
        raise ValueError("tie rod / wheel centre do not each have one free end")

    return {
        "UCA_IN_FRONT": nodes[uca_front],
        "UCA_IN_REAR": nodes[uca_rear],
        "UCA_OUT": nodes[uca_out],
        "LCA_IN_FRONT": nodes[lca_front],
        "LCA_IN_REAR": nodes[lca_rear],
        "LCA_OUT": nodes[lca_out],
        "TIE_ROD_IN": nodes[tie_rod_in[0]],
        "TIE_ROD_OUT": nodes[tie_rod_out],
        "WHEEL_CENTER": nodes[wheel_center],
        "CONTACT_PATCH": nodes[contact_patch[0]],
    }


def decode(segments: list[RawSegment], frame: str) -> dict[tuple[str, str], Vec]:
    """Turn raw CAD line segments into a named hardpoint table in ISO 8855."""
    if not segments:
        raise ValueError(
            f"no line segments found in any '{SKETCH_PREFIX}*' sketch. "
            "Run --survey to see what this model contains."
        )

    inverse = _inverse_frame(FRAMES[frame])
    iso = [(name, _apply(inverse, a), _apply(inverse, b)) for name, a, b in segments]

    nodes, edges = _weld(iso, WELD_TOL_MM)
    groups = _components(len(nodes), edges)
    if len(groups) != len(CORNERS):
        raise ValueError(
            f"found {len(groups)} disconnected group(s), expected {len(CORNERS)} corners. "
            "Either a corner is missing or two corners share a point."
        )

    decoded = [_name_corner(nodes, group, edges) for group in groups]

    # Front corners are the pair with the greater wheel-centre X; left is Y>0
    # (ISO 8855 Y+ is LEFT). Both come from the decoded wheel centre rather
    # than a component average, so an unusual inboard layout cannot flip them.
    order = sorted(range(len(decoded)), key=lambda i: -decoded[i]["WHEEL_CENTER"][0])
    axle_of = {i: ("F" if position < 2 else "R") for position, i in enumerate(order)}

    table: dict[tuple[str, str], Vec] = {}
    seen: set[str] = set()
    for index, corner_points in enumerate(decoded):
        side = "L" if corner_points["WHEEL_CENTER"][1] > 0 else "R"
        corner = f"{axle_of[index]}{side}"
        if corner in seen:
            raise ValueError(
                f"two corners both resolved to {corner}. Check the model is a "
                "whole car and not a mirrored pair on one side."
            )
        seen.add(corner)
        for point_name, xyz in corner_points.items():
            table[(corner, point_name)] = xyz
    return table


def summarise(table: dict[tuple[str, str], Vec]) -> None:
    """Print the car the CSV describes, and flag frame/origin smells.

    Reading a model back in the wrong frame decodes perfectly well -- the
    topology is unchanged by a mirror -- and yields a mirrored car with no
    error anywhere. These checks lean on this project's origin convention
    (front axle centreline, ground plane) to make that visible.
    """
    front_track = abs(table[("FL", "CONTACT_PATCH")][1] - table[("FR", "CONTACT_PATCH")][1])
    rear_track = abs(table[("RL", "CONTACT_PATCH")][1] - table[("RR", "CONTACT_PATCH")][1])
    wheelbase = abs(table[("FL", "WHEEL_CENTER")][0] - table[("RL", "WHEEL_CENTER")][0])
    print(
        f"Decoded: wheelbase {wheelbase:.1f} mm, "
        f"front track {front_track:.1f} mm, rear track {rear_track:.1f} mm"
    )

    patch_z = max(abs(table[(c, "CONTACT_PATCH")][2]) for c in CORNERS)
    if patch_z > 1.0:
        print(
            f"warning: contact patches sit up to {patch_z:.1f} mm off z=0. The CAD "
            "origin is not on the ground plane, or Z is not up.",
            file=sys.stderr,
        )

    front_x = max(abs(table[(c, "WHEEL_CENTER")][0]) for c in ("FL", "FR"))
    if front_x > 1.0:
        print(
            f"warning: the front wheel centres are at x={front_x:.1f} mm, not 0. "
            "Either the CAD origin is not the front axle, or --frame is wrong "
            "and this car is mirrored front-to-rear.",
            file=sys.stderr,
        )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def write_csv(table: dict[tuple[str, str], Vec], out_path: Path) -> None:
    """Write in the exact column order and precision of the merged CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corner", "point", "x_mm", "y_mm", "z_mm"])
        for corner in CORNERS:
            for point_name in POINT_NAMES:
                x, y, z = table[(corner, point_name)]
                writer.writerow([corner, point_name, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])


def compare(table: dict[tuple[str, str], Vec], reference: Path, tol_mm: float) -> int:
    """Print a per-hardpoint diff against a reference CSV. Returns rows over tol."""
    from export_solidworks_skeleton import load_hardpoints

    expected = load_hardpoints(reference)
    print(f"\nDiff against {reference.name} (tolerance {tol_mm} mm)")
    print(f"{'corner/point':<28s} {'dx':>10s} {'dy':>10s} {'dz':>10s} {'|d|':>10s}")
    print("-" * 72)

    worst = 0.0
    over = 0
    for corner in CORNERS:
        for point_name in POINT_NAMES:
            got = table[(corner, point_name)]
            want = expected[(corner, point_name)]
            delta = [got[i] - want[i] for i in range(3)]
            magnitude = sum(d * d for d in delta) ** 0.5
            worst = max(worst, magnitude)
            if magnitude > tol_mm:
                over += 1
                label = f"{corner}/{point_name}"
                print(
                    f"{label:<28s} {delta[0]:>10.3f} {delta[1]:>10.3f} "
                    f"{delta[2]:>10.3f} {magnitude:>10.3f}"
                )

    if over == 0:
        print(f"(all 40 hardpoints agree; largest deviation {worst:.2e} mm)")
    else:
        print(f"\n{over} hardpoint(s) differ; largest deviation {worst:.3f} mm")
    return over


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read suspension hardpoints out of SolidWorks into a CSV.",
    )
    parser.add_argument(
        "--file", type=Path, default=None, metavar="PATH",
        help="read this .SLDPRT/.SLDASM (default: the document already open)",
    )
    parser.add_argument(
        "--frame", choices=sorted(FRAMES), default="iso",
        help="the frame the CAD model is in; output is always ISO 8855 (default: iso)",
    )
    parser.add_argument(
        "--prefix", default=SKETCH_PREFIX,
        help=f"sketch name prefix to read (default: {SKETCH_PREFIX})",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT, metavar="PATH",
        help=f"output CSV (default: {DEFAULT_OUT.name})",
    )
    parser.add_argument(
        "--compare", nargs="?", type=Path, const=DEFAULT_REFERENCE, default=None,
        metavar="CSV",
        help=f"diff the result against a reference CSV (default: {DEFAULT_REFERENCE.name})",
    )
    parser.add_argument(
        "--tol", type=float, default=0.001, metavar="MM",
        help="deviation above which --compare reports a hardpoint (default: 0.001)",
    )
    parser.add_argument(
        "--survey", action="store_true",
        help="list the model's sketches and exit, without decoding anything",
    )
    args = parser.parse_args()

    opened_here = False
    model = None
    app = None
    try:
        app = _connect()
        model, opened_here = _open_document(app, args.file)
        print(f"Source document: {_prop(model, 'GetTitle')}")

        if args.survey:
            survey(model, args.prefix)
            return 0

        segments = read_segments(model, args.prefix)
        print(f"Read {len(segments)} line(s) from '{args.prefix}*' sketches.")
        table = decode(segments, args.frame)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if opened_here and model is not None and app is not None:
            app.CloseDoc(_prop(model, "GetTitle"))

    summarise(table)
    write_csv(table, args.out)
    print(f"Wrote {args.out}")
    print(f"Frame: CAD is {args.frame}; CSV is ISO 8855.")

    if args.compare is not None:
        try:
            if compare(table, args.compare, args.tol):
                return 2
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
