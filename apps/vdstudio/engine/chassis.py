"""Compare the suspension's chassis pick-ups with a chassis CAD survey.

Input: the JSON written by ``scripts/survey_chassis.py`` (the CAD model's
sketches in its own frame, mm) and the project's CAD placement
(``project.cad_frame``: which CAD axes point forward and left, and where the
ISO origin sits in CAD). Output, per inboard pick-up: the nearest frame node
(an end of a frame centreline in the CAD sketches), the vector from the
project's point to it, and the distance to the nearest tube centreline.

What it is: a geometric cross-check of two sources of the same points.
What it is not: a statement that either source is right. The frame nodes are
taken as the CAD's intent for the pick-ups; a model that puts the pick-up on
a tab away from the node reads as a difference here, and says so.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path

import export_solidworks_skeleton as swx
import numpy as np

from .geometry import CarGeometry
from .project import CORNERS

CHASSIS_POINTS = ("UCA_IN_FRONT", "UCA_IN_REAR", "LCA_IN_FRONT", "LCA_IN_REAR")
NO_NODE_MM = 100.0      # beyond this the nearest node is reported as "none nearby"


@dataclass(frozen=True)
class Line:
    sketch: str
    a: tuple[float, float, float]      # CAD frame, mm
    b: tuple[float, float, float]


@dataclass(frozen=True)
class ChassisSurvey:
    title: str
    path: str
    schema: int
    lines: tuple[Line, ...]            # frame centrelines (no construction geometry)
    skipped_2d: int                    # 2D sketch lines left out (schema 1 mirrored them)

    @property
    def nodes(self) -> dict[tuple[float, float, float], str]:
        out: dict[tuple[float, float, float], str] = {}
        for ln in self.lines:
            for p in (ln.a, ln.b):
                out.setdefault(p, ln.sketch)
        return out


def load_survey(path: str | Path) -> ChassisSurvey:
    """Read a survey. Schema 1 mirrored 2D sketches on rotated planes: only its 3D
    sketches are used (a weldment's layout is usually 3D sketches anyway)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    schema = int(d.get("schema", 1))
    lines, skipped = [], 0
    for s in d.get("sketches", []):
        for ln in s.get("lines", []):
            if ln.get("construction"):
                continue
            if s.get("type") != "3d" and schema < 2:
                skipped += 1
                continue
            lines.append(Line(s["name"], tuple(float(v) for v in ln["a"]),  # type: ignore[arg-type]
                              tuple(float(v) for v in ln["b"])))  # type: ignore[arg-type]
    if not lines:
        raise ValueError(f"{path}: no frame lines in the survey's sketches")
    return ChassisSurvey(str(d.get("title", "")), str(d.get("path", "")), schema,
                         tuple(lines), skipped)


def cad_to_iso(p, frame: swx.Frame) -> np.ndarray:
    """Inverse of the skeleton placement: ``iso = R.T @ (cad - origin)``."""
    r = np.array(frame.matrix, float)
    return r.T @ (np.asarray(p, float) - np.array(frame.origin_mm, float))


@dataclass(frozen=True)
class PickupCheck:
    corner: str
    point: str
    project_mm: tuple[float, float, float]           # ISO 8855
    node_mm: tuple[float, float, float] | None       # ISO 8855, nearest frame node
    node_sketch: str
    tube_mm: float                                   # distance to the nearest centreline
    tube_sketch: str

    @property
    def delta_mm(self) -> tuple[float, float, float] | None:
        if self.node_mm is None:
            return None
        return tuple(n - p for n, p in zip(self.node_mm, self.project_mm, strict=True))  # type: ignore[return-value]

    @property
    def distance_mm(self) -> float:
        d = self.delta_mm
        return math.inf if d is None else math.hypot(*d)


def _seg_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)) if ab @ ab > 0 else 0.0
    return float(np.linalg.norm(a + t * ab - p))


def compare(geo: CarGeometry, survey: ChassisSurvey, frame: swx.Frame,
            points: tuple[str, ...] = CHASSIS_POINTS) -> list[PickupCheck]:
    nodes = survey.nodes
    keys = list(nodes)
    node_iso = np.array([cad_to_iso(k, frame) for k in keys])
    segs = [(ln.sketch, cad_to_iso(ln.a, frame), cad_to_iso(ln.b, frame)) for ln in survey.lines]
    out = []
    for c in CORNERS:
        for n in points:
            p = np.asarray(geo.points[c][n], float)
            d = np.linalg.norm(node_iso - p, axis=1)
            i = int(np.argmin(d))
            near = d[i] <= NO_NODE_MM
            tube, tube_sk = min((_seg_distance(p, a, b), sk) for sk, a, b in segs)
            out.append(PickupCheck(
                c, n, tuple(float(v) for v in p),  # type: ignore[arg-type]
                tuple(float(v) for v in node_iso[i]) if near else None,  # type: ignore[arg-type]
                nodes[keys[i]] if near else "", tube, tube_sk))
    return out


def placement_text(frame: swx.Frame) -> str:
    return frame.label


def report_csv(rows: list[PickupCheck]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["corner", "point", "project_x_mm", "project_y_mm", "project_z_mm",
                "cad_node_x_mm", "cad_node_y_mm", "cad_node_z_mm", "dx_mm", "dy_mm", "dz_mm",
                "distance_mm", "node_sketch", "tube_distance_mm", "tube_sketch"])
    for r in rows:
        node = r.node_mm or ("", "", "")
        dl = r.delta_mm or ("", "", "")
        w.writerow([r.corner, r.point, *(f"{v:.2f}" for v in r.project_mm),
                    *(f"{v:.2f}" if v != "" else "" for v in node),
                    *(f"{v:.2f}" if v != "" else "" for v in dl),
                    f"{r.distance_mm:.2f}" if r.node_mm else "", r.node_sketch,
                    f"{r.tube_mm:.2f}", r.tube_sketch])
    return buf.getvalue()


def report_markdown(rows: list[PickupCheck], survey: ChassisSurvey, frame: swx.Frame,
                    project_name: str, svg_name: str | None = None) -> str:
    L = [f"# Chassis CAD vs suspension pick-ups — {survey.title}", "",
         f"- Project: {project_name}",
         f"- CAD survey: `{survey.path or survey.title}` (schema {survey.schema}), "
         f"{len(survey.lines)} frame centrelines"
         + (f"; {survey.skipped_2d} lines from 2D sketches left out (schema 1 mirrored "
            "them)" if survey.skipped_2d else ""),
         f"- Placement (CAD → ISO 8855): {frame.label}",
         "- Coordinates: ISO 8855 (X forward, Y LEFT, Z up), mm. Δ = CAD node − project.",
         "- The CAD node is the nearest end of a frame centreline: taken as the CAD's "
         "intent for the pick-up. A pick-up on a tab away from its node shows up as a "
         "difference. *Tube* is the distance from the project's point to the nearest tube "
         "centreline (a tab has to bridge it).", ""]
    if svg_name:
        L += [f"![Top, side and front views]({svg_name})", ""]
    L += ["| corner | point | project (x, y, z) | CAD node (x, y, z) | Δx | Δy | Δz | "
          "\\|Δ\\| | tube | CAD sketch |", "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        pp = ", ".join(f"{v:.1f}" for v in r.project_mm)
        if r.node_mm is None:
            L.append(f"| {r.corner} | {r.point} | {pp} | none within {NO_NODE_MM:.0f} mm | | | "
                     f"| | {r.tube_mm:.1f} | {r.tube_sketch} |")
            continue
        nn = ", ".join(f"{v:.1f}" for v in r.node_mm)
        dx, dy, dz = r.delta_mm  # type: ignore[misc]
        L.append(f"| {r.corner} | {r.point} | {pp} | {nn} | {dx:+.1f} | {dy:+.1f} | {dz:+.1f} | "
                 f"**{r.distance_mm:.1f}** | {r.tube_mm:.1f} | {r.node_sketch} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Three orthographic views as SVG (no plotting dependency)
# --------------------------------------------------------------------------- #

# (title, horizontal axis, vertical axis) as (index, sign) in ISO; see module docs
VIEWS = (("Top (front up)", (1, -1), (0, 1)),
         ("Side, from the left (front to the left)", (0, -1), (2, 1)),
         ("Front, from ahead (car's left on the right)", (1, 1), (2, 1)))


def views_svg(geo: CarGeometry, survey: ChassisSurvey, frame: swx.Frame,
              rows: list[PickupCheck], scale: float = 0.22) -> str:
    frame_iso = [(cad_to_iso(ln.a, frame), cad_to_iso(ln.b, frame)) for ln in survey.lines]
    links = (("UCA_IN_FRONT", "UCA_OUT"), ("UCA_IN_REAR", "UCA_OUT"),
             ("LCA_IN_FRONT", "LCA_OUT"), ("LCA_IN_REAR", "LCA_OUT"),
             ("TIE_ROD_IN", "TIE_ROD_OUT"), ("UCA_OUT", "LCA_OUT"),
             ("WHEEL_CENTER", "CONTACT_PATCH"))
    susp = [(np.asarray(geo.points[c][a], float), np.asarray(geo.points[c][b], float))
            for c in CORNERS for a, b in links]
    pad, gap, title_h = 30.0, 40.0, 26.0
    panels, x0, height = [], pad, 0.0
    for title, (hi, hs), (vi, vs) in VIEWS:
        pts = [p for seg in frame_iso + susp for p in seg]
        hs_v = [hs * p[hi] for p in pts]
        vs_v = [vs * p[vi] for p in pts]
        hmin, hmax, vmin, vmax = min(hs_v), max(hs_v), min(vs_v), max(vs_v)
        w, h = (hmax - hmin) * scale, (vmax - vmin) * scale
        panels.append((title, hi, hs, vi, vs, hmin, vmax, x0, w, h))
        x0 += w + gap
        height = max(height, h)
    width = x0 - gap + pad
    total_h = height + title_h + 2 * pad + 40
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
           f'height="{total_h:.0f}" viewBox="0 0 {width:.1f} {total_h:.1f}" '
           'font-family="Segoe UI, Arial, sans-serif" font-size="12">',
           f'<rect width="{width:.1f}" height="{total_h:.1f}" fill="#ffffff"/>']
    for title, hi, hs, vi, vs, hmin, vmax, px, _w, _h in panels:
        oy = pad + title_h

        def xy(p, hi=hi, hs=hs, vi=vi, vs=vs, hmin=hmin, vmax=vmax, px=px, oy=oy):
            return (px + (hs * p[hi] - hmin) * scale, oy + (vmax - vs * p[vi]) * scale)

        out.append(f'<text x="{px:.1f}" y="{pad + 12:.1f}" font-weight="600">{title}</text>')
        for a, b in frame_iso:
            (x1, y1), (x2, y2) = xy(a), xy(b)
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                       'stroke="#9aa4b2" stroke-width="1.2"/>')
        for a, b in susp:
            (x1, y1), (x2, y2) = xy(a), xy(b)
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                       'stroke="#2563eb" stroke-width="1.6"/>')
        for r in rows:
            x1, y1 = xy(r.project_mm)
            out.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="3" fill="#2563eb"/>')
            if r.node_mm is not None:
                x2, y2 = xy(r.node_mm)
                out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                           'stroke="#dc2626" stroke-width="1.6"/>')
                out.append(f'<circle cx="{x2:.1f}" cy="{y2:.1f}" r="3" fill="#dc2626"/>')
    ly = total_h - 18
    out.append(f'<text x="{pad:.1f}" y="{ly:.1f}"><tspan fill="#6b7280">grey: CAD frame '
               'centrelines</tspan>   <tspan fill="#2563eb">blue: project suspension and '
               'its chassis pick-ups</tspan>   <tspan fill="#dc2626">red: nearest CAD frame '
               'node, joined to its pick-up</tspan></text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"
