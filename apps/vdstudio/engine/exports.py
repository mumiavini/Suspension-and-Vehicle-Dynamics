"""Import / export: hardpoint tables, OptimumK, SolidWorks, reports, setup sheet.

The SolidWorks and OptimumK formats reuse the repository scripts
(``scripts/export_solidworks_skeleton.py``, ``scripts/import_solidworks_hardpoints.py``,
``scripts/export_hardpoints_for_optimumk.py``) so the app writes exactly what
the command-line tools write.

Frames: tables are ISO 8855 unless a frame is named. ``sw_fsae`` is the
"typical FSAE SolidWorks" frame (X+ rearward, Y+ right, Z+ up); ``optimumk``
is X+ forward, Y+ right, Z+ up. Reading a file in the wrong frame yields a
mirrored car without any error, so the frame is always an explicit argument.
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import math
import re
import sys
from pathlib import Path

import export_hardpoints_for_optimumk as optk
import export_solidworks_skeleton as swx
import numpy as np

from .. import APP_NAME, __version__
from . import mass
from .geometry import POINT_ORDER, CarGeometry
from .kpis import GRAVITY, KPI, KpiSet
from .project import CORNERS, INPUT_POINTS, SOURCES, STAGES, Project, Pt

FRAMES: dict[str, np.ndarray] = {
    "iso": np.eye(3),
    "sw_fsae": np.diag([-1.0, -1.0, 1.0]),
    "optimumk": np.diag([1.0, -1.0, 1.0]),
}
FRAME_LABELS = {
    "iso": "ISO 8855 (X fwd, Y left, Z up)",
    "sw_fsae": "SolidWorks FSAE (X rear, Y right, Z up)",
    "optimumk": "OptimumK (X fwd, Y right, Z up)",
}


def to_frame(p: Pt, frame: str) -> Pt:
    v = FRAMES[frame] @ np.asarray(p, float)
    return (float(v[0]) + 0.0, float(v[1]) + 0.0, float(v[2]) + 0.0)


def from_frame(p: Pt, frame: str) -> Pt:
    # every matrix here is diagonal +/-1, hence self-inverse
    return to_frame(p, frame)


# --------------------------------------------------------------------------- #
# Hardpoint tables
# --------------------------------------------------------------------------- #

def hardpoints_csv(geo: CarGeometry, frame: str = "iso", extended: bool = False) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    head = ["corner", "point", "x_mm", "y_mm", "z_mm"]
    if extended:
        head += ["source", "tol_mm"]
    w.writerow(head)
    for c, n, x, y, z, src, tol in geo.rows():
        X, Y, Z = to_frame((x, y, z), frame)
        row = [c, n, f"{X:.3f}", f"{Y:.3f}", f"{Z:.3f}"]
        if extended:
            row += [src, "" if math.isnan(tol) else f"{tol:g}"]
        w.writerow(row)
    return buf.getvalue()


def read_hardpoints(path: str | Path, frame: str = "iso"
                    ) -> tuple[dict[str, dict[str, Pt]], dict[str, dict[str, str]],
                               dict[str, dict[str, float]]]:
    """Read CSV (``corner,point,x_mm,y_mm,z_mm[,source,tol_mm]``) or the JSON the
    scripts write (``{"corners": {FL: {NAME: [x, y, z]}}}``). Returns ISO points.

    CONTACT_PATCH rows are ignored: the patch is derived from the wheel centre,
    tyre radius and camber, never an input.
    """
    p = Path(path)
    pts: dict[str, dict[str, Pt]] = {}
    src: dict[str, dict[str, str]] = {}
    tol: dict[str, dict[str, float]] = {}
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        corners = data.get("corners", data)
        for c, d in corners.items():
            for n, xyz in d.items():
                if n.upper() in INPUT_POINTS:
                    pts.setdefault(c.upper(), {})[n.upper()] = from_frame(tuple(xyz), frame)
    else:
        with open(p, newline="", encoding="utf-8-sig") as fh:
            rd = csv.DictReader(fh)
            need = {"corner", "point", "x_mm", "y_mm", "z_mm"}
            if not need <= set(rd.fieldnames or ()):
                raise ValueError(f"{p.name}: needs columns {sorted(need)}")
            seen: dict[tuple[str, str], int] = {}
            for line, row in enumerate(rd, start=2):
                c, n = row["corner"].strip().upper(), row["point"].strip().upper()
                if n not in INPUT_POINTS:
                    continue
                if (c, n) in seen:
                    raise ValueError(f"{p.name} line {line}: {c} {n} already given on line "
                                     f"{seen[(c, n)]}")
                seen[(c, n)] = line
                try:
                    xyz = (float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"]))
                except ValueError as exc:
                    raise ValueError(f"{p.name} line {line}: bad coordinate ({exc})") from exc
                pts.setdefault(c, {})[n] = from_frame(xyz, frame)
                if row.get("source"):
                    s = row["source"].strip().lower()
                    if s not in SOURCES:
                        raise ValueError(f"{p.name} line {line}: unknown source {s!r} "
                                         f"(one of {', '.join(SOURCES)})")
                    src.setdefault(c, {})[n] = s
                if row.get("tol_mm"):
                    tol.setdefault(c, {})[n] = float(row["tol_mm"])
    unknown = set(pts) - set(CORNERS)
    if unknown:
        raise ValueError(f"{p.name}: unknown corner(s) {sorted(unknown)}")
    return pts, src, tol


def diff_points(a: dict[str, dict[str, Pt]], b: dict[str, dict[str, Pt]], tol_mm: float = 0.001
                ) -> list[tuple[str, str, float, Pt | None, Pt | None]]:
    """Points that differ by more than ``tol_mm`` (or exist on one side only)."""
    out = []
    for c in CORNERS:
        for n in INPUT_POINTS:
            pa, pb = a.get(c, {}).get(n), b.get(c, {}).get(n)
            if pa is None or pb is None:
                if pa is not None or pb is not None:
                    out.append((c, n, math.inf, pa, pb))
                continue
            d = float(np.linalg.norm(np.subtract(pa, pb)))
            if d > tol_mm:
                out.append((c, n, d, pa, pb))
    return out


# --------------------------------------------------------------------------- #
# OptimumK and SolidWorks (the repository scripts' formats)
# --------------------------------------------------------------------------- #

def optimumk_text(geo: CarGeometry) -> str:
    parts = [optk.export_for_optimumk(geo.corner(c)) for c in CORNERS]
    return "\n\n".join(parts) + "\n"


def _sw_table(geo: CarGeometry) -> dict[tuple[str, str], Pt]:
    return {(c, n): geo.points[c][n] for c in CORNERS for n in POINT_ORDER}


def cad_frame(project: Project) -> swx.Frame:
    """The project's CAD placement (axes + origin) as the SolidWorks scripts' frame."""
    f = project.cad_frame
    return swx.axis_frame(f.forward, f.left, (f.origin_x_mm, f.origin_y_mm, f.origin_z_mm))


def to_cad(p: Pt, frame: str | swx.Frame) -> Pt:
    """An ISO 8855 point (mm) where the SolidWorks skeleton puts it in the CAD model."""
    return swx._transform(tuple(float(v) for v in p), swx.resolve_frame(frame))


def solidworks_macro(geo: CarGeometry, frame: str | swx.Frame = "iso",
                     grouping: str = "corner", name: str = "vdstudio_hardpoints.csv") -> str:
    groups = swx.build_groups(_sw_table(geo), frame, grouping)
    return swx.render_macro(groups, Path(name), frame)


def solidworks_available() -> tuple[bool, str]:
    if sys.platform != "win32":
        return False, "SolidWorks automation needs Windows with SolidWorks and pywin32."
    try:
        import win32com.client  # noqa: F401
    except ImportError:
        return False, "pywin32 is not installed (pip install pywin32)."
    return True, ""


def solidworks_push(geo: CarGeometry, frame: str | swx.Frame = "iso",
                    grouping: str = "corner", new_part: bool = False) -> None:
    """Draw the skeleton in the running SolidWorks (scripts/export_solidworks_skeleton)."""
    groups = swx.build_groups(_sw_table(geo), frame, grouping)
    swx.build_live(groups, Path("VD Studio"), frame, new_part=new_part)


def solidworks_pull(frame: str | swx.Frame = "iso", path: str | None = None
                    ) -> dict[str, dict[str, Pt]]:
    """Read the ``SUSP_`` skeleton back (scripts/import_solidworks_hardpoints)."""
    import import_solidworks_hardpoints as swi

    app = swi._connect()
    model, opened = swi._open_document(app, Path(path) if path else None)
    try:
        table = swi.decode(swi.read_segments(model, swi.SKETCH_PREFIX), frame)
    finally:
        if opened:
            app.CloseDoc(swi._prop(model, "GetTitle"))
    out: dict[str, dict[str, Pt]] = {}
    for (c, n), xyz in table.items():
        if n in INPUT_POINTS:
            out.setdefault(c, {})[n] = tuple(float(v) for v in xyz)  # type: ignore[assignment]
    return out


# --------------------------------------------------------------------------- #
# KPI tables and the Markdown report
# --------------------------------------------------------------------------- #

def kpi_csv(kpis: KpiSet) -> str:
    """One row per KPI. ``value`` is the display text (the KPI's own format);
    ``value_raw`` next to it is the full-precision number (empty when failed),
    so the exported evidence file does not inherit the display rounding."""
    buf = io.StringIO()
    rows = []
    for row, k in zip(kpis.rows(), kpis.items.values(), strict=True):
        out: dict[str, str] = {}
        for col, val in row.items():
            out[col] = val
            if col == "value":
                out["value_raw"] = repr(float(k.value)) if k.ok else ""
        rows.append(out)
    if not rows:
        return ""
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _md_table(head: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "/") for c in r) + " |")
    return out


_STATUS_MD = {"in": "inside band", "out": "**outside band**", "info": "", "fail": "**failed**"}


def _pair_band(f: KPI | None, r: KPI | None) -> str:
    """Band cell of a front/rear row: one band when both axles share it,
    ``F … / R …`` (``—`` for an axle without a band) when they differ, so a
    rear value is never printed beside the front band."""
    bf = f.band_text() if f is not None else ""
    br = r.band_text() if r is not None else ""
    if f is None or r is None or bf == br:
        return bf or br
    return f"F {bf or '—'} / R {br or '—'}"


def _pair_status(f: KPI | None, r: KPI | None) -> str:
    if f is not None and r is not None and f.band_text() != r.band_text():
        # per-axle bands: say which status belongs to which axle
        return " / ".join(f"{ax} {_STATUS_MD[k.status] or 'no band'}"
                          for ax, k in (("F", f), ("R", r)))
    return " / ".join(x for x in (_STATUS_MD[f.status] if f else "",
                                  _STATUS_MD[r.status] if r else "") if x)


def markdown_report(project: Project, geo: CarGeometry, kpis: KpiSet,
                    concept: KpiSet | None = None) -> str:
    """Design report for the team and the Design Event. States, never recommends."""
    L: list[str] = []
    today = _dt.date.today().isoformat()
    L += [f"# {project.name} -- suspension & steering report", "",
          f"{project.team} · generated {today} by {APP_NAME} {__version__} · "
          f"geometry source: **{geo.mode}**", "",
          "> Every number below was computed from the hardpoints listed in the table below by "
          "`vdcore.geometry.solver.DWSolver` (or the closed form named in its row). Bands "
          "are design-intent windows recorded in the configuration, not verdicts. Values "
          "tagged `design_intent` were chosen, not measured.", ""]
    if geo.warnings:
        L += ["**Warnings**", ""] + [f"- {w}" for w in geo.warnings] + [""]

    counts = kpis.counts()
    L += ["## 0. Status", "",
          f"- KPIs inside their band: {counts['in']}; outside: {counts['out']}; "
          f"no band: {counts['info']}; failed: {counts['fail']}",
          f"- Inputs tagged `estimate`: {geo.estimate_count()}", ""]
    out_rows = [[k.scope, k.label, k.text(), k.unit, k.band_text(), k.band_source]
                for k in kpis.items.values() if k.status == "out"]
    if out_rows:
        L += ["**Outside their design-intent band** (information for the designer):", ""]
        L += _md_table(["scope", "KPI", "value", "unit", "band", "band source"], out_rows) + [""]
    fail_rows = [[k.scope, k.label, k.error] for k in kpis.items.values() if k.status == "fail"]
    if fail_rows:
        L += ["**Could not be computed** (no value is shown rather than a wrong one):", ""]
        L += _md_table(["scope", "KPI", "reason"], fail_rows) + [""]

    v = geo.vehicle_data
    L += ["## 1. Vehicle inputs (design_intent unless stated)", ""]
    L += _md_table(["input", "value"], [
        ["Total mass", f"{v.total_mass_kg:g} kg"], ["Unsprung mass", f"{v.unsprung_mass_kg:g} kg"],
        ["CG height", f"{v.cg_height_mm:g} mm"],
        ["Wheelbase (hardpoints)", f"{geo.wheelbase_mm:.1f} mm"],
        ["CG behind front axle", f"{v.cg_from_front_axle_mm:.1f} mm "
                                 f"({100 * v.front_mass_fraction:.1f} % front)"],
        ["Target roll gradient", f"{v.target_roll_gradient_deg_per_g:g} deg/g"],
        ["Brake bias front", f"{v.brake_bias_front:g}"],
        ["Load-case μ lat / long, ay, LLTD", f"{v.mu_lateral:g} / {v.mu_longitudinal:g}, "
                                            f"{v.design_ay_g:g} g, {v.lltd_front:g} "
                                            "(tyre assumptions: no TTC data)"],
    ]) + [""]

    sec = 2
    for group in kpis.groups():
        items = kpis.by_group(group)
        L += [f"## {sec}. {group}", ""]
        sec += 1
        axle_items = [k for k in items if k.scope in ("front", "rear")]
        other = [k for k in items if k.scope not in ("front", "rear")]
        if axle_items:
            labels = list(dict.fromkeys(k.label for k in axle_items))
            rows = []
            for lab in labels:
                f = next((k for k in axle_items if k.label == lab and k.scope == "front"), None)
                r = next((k for k in axle_items if k.label == lab and k.scope == "rear"), None)
                ref = f or r
                rows.append([lab, f.text() if f else "", r.text() if r else "",
                             ref.unit if ref else "", _pair_band(f, r), _pair_status(f, r),
                             ref.source if ref else ""])
            L += _md_table(["KPI", "front", "rear", "unit", "band", "status", "source"], rows)
            L += [""]
        if other:
            rows = [[k.label, k.text(), k.unit, k.band_text(), _STATUS_MD[k.status],
                     k.source + (f" -- {k.note}" if k.note else "")] for k in other]
            L += _md_table(["KPI", "value", "unit", "band", "status", "source / note"], rows)
            L += [""]

    if concept is not None:
        L += [f"## {sec}. Concept-stage numbers", ""]
        sec += 1
        rows = [[k.group, k.label, k.text(), k.unit, k.source] for k in concept.items.values()]
        L += _md_table(["group", "quantity", "value", "unit", "source"], rows) + [""]

    L += [f"## {sec}. Hardpoints (ISO 8855: X fwd, Y LEFT, Z up; mm)", ""]
    sec += 1
    rows = [[c, n, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", s] for c, n, x, y, z, s, _ in geo.rows()]
    L += _md_table(["corner", "point", "X", "Y", "Z", "source"], rows) + [""]
    L += ["CONTACT_PATCH is derived (wheel centre, loaded radius, static camber). Static toe "
          "cannot be encoded by points and is a configuration input.", ""]

    if project.targets:
        L += [f"## {sec}. Design targets and their rationale", ""]
        sec += 1
        rows = [[t.name, f"{t.value:g} ± {t.tolerance:g} {t.unit}", t.event, t.rationale,
                 t.reference, t.source] for t in project.targets]
        L += _md_table(["target", "value", "event", "rationale", "reference", "source"], rows)
        L += [""]

    if project.design_log:
        L += [f"## {sec}. Design decisions", ""]
        sec += 1
        for e in project.design_log:
            L += [f"### {e.date} -- {e.title}  ({e.stage})", ""]
            for lab, val in (("Decision", e.decision), ("Rationale", e.rationale),
                             ("Alternatives considered", e.alternatives),
                             ("Author", e.author)):
                if val:
                    L += [f"**{lab}:** {val}", ""]

    L += [f"## {sec}. FSAE workflow status", ""]
    rows = [[label, project.stages[k].status, project.stages[k].notes]
            for k, label, _ in STAGES if k in project.stages]
    L += _md_table(["stage", "status", "notes"], rows) + [""]
    L += ["---", "Out of scope for this tool and deliberately not computed: wheel rates from "
          "the linkage, damping, frequency response, stress/buckling, and anything that "
          "needs tyre data (PUCPR Racing has no TTC data yet)."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# Setup sheet (legacy app's layout: user rows + validated geometry rows)
# --------------------------------------------------------------------------- #

SETUP_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    # key, label, unit, category
    ("tire", "Tyre size, compound, make", "", "Tyres & wheels"),
    ("wheel_size", "Wheel (diameter × width)", "inch", "Tyres & wheels"),
    ("wheel_material", "Wheel material / construction", "", "Tyres & wheels"),
    ("tyre_rate_f", "Tyre vertical rate front (estimate)", "N/mm", "Tyres & wheels"),
    ("tyre_rate_r", "Tyre vertical rate rear (estimate)", "N/mm", "Tyres & wheels"),
    ("susp_type", "Suspension type", "", "Suspension"),
    ("spring_f", "Spring rate front", "N/mm", "Suspension"),
    ("spring_r", "Spring rate rear", "N/mm", "Suspension"),
    ("mr_f", "Motion ratio front (spring/wheel)", "-", "Suspension"),
    ("mr_r", "Motion ratio rear (spring/wheel)", "-", "Suspension"),
    ("arb", "Anti-roll bar / adjustment methods", "", "Suspension"),
    ("jounce_f", "Jounce damping front", "% crit", "Damper"),
    ("rebound_f", "Rebound damping front", "% crit", "Damper"),
    ("jounce_r", "Jounce damping rear", "% crit", "Damper"),
    ("rebound_r", "Rebound damping rear", "% crit", "Damper"),
    ("camber_adj", "Static camber adjustment method", "", "Alignment"),
    ("ackermann_adj", "Adjustable Ackermann?", "", "Steering"),
)


# source column of input rows that are not plain "input"
_SETUP_SOURCE = {"tyre_rate_f": "input (estimate: no TTC data; optional)"}
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
MR_PLAUSIBLE = (0.3, 1.5)


def _f(x: object) -> float:
    """A typed setup-sheet number: a decimal comma is accepted and trailing unit
    text ignored (``'35,0 N/mm'`` -> 35.0). NaN when no leading number is found."""
    if isinstance(x, (int, float)):
        return float(x)
    m = _NUMBER.match(str(x if x is not None else "").strip().replace(",", "."))
    return float(m.group()) if m else math.nan


def _read(sheet: dict, key: str, what: str, notes: list[str]) -> float:
    """``_f`` of a setup field; an entry that cannot be used is reported in ``notes``."""
    raw = sheet.get(key, "")
    v = _f(raw)
    if str(raw if raw is not None else "").strip():
        if not math.isfinite(v):
            notes.append(f"could not read {what} '{raw}'")
        elif v <= 0:
            notes.append(f"{what} {v:g} is not positive")
    return v


def setup_sheet_rows(project: Project, geo: CarGeometry, kpis: KpiSet) -> list[list[str]]:
    """[category, parameter, unit, front, rear, source]."""
    s = project.setup_sheet
    rows: list[list[str]] = []
    for key, label, unit, cat in SETUP_FIELDS:
        val = str(s.get(key, "") or "—")
        if key.endswith("_f"):
            rows.append([cat, label.replace(" front", ""), unit, val,
                         str(s.get(key[:-2] + "_r", "") or "—"),
                         _SETUP_SOURCE.get(key, "input")])
        elif key.endswith("_r"):
            continue
        else:
            rows.append([cat, label, unit, val, val, "input"])
    v = geo.vehicle_data
    nan = math.nan
    # sprung corner mass: the sprung mass on each axle (engine.mass), not the
    # sprung mass times the whole-car axle share
    mass_src = f"vehicle data; {mass.UNSPRUNG_ASSUMPTION}"
    try:
        sm: mass.SprungMass | None = mass.sprung_mass(
            v.total_mass_kg, v.unsprung_mass_kg, v.cg_height_mm, v.cg_from_front_axle_mm,
            v.wheelbase_mm, geo.axles["front"].loaded_radius_mm,
            geo.axles["rear"].loaded_radius_mm)
    except ValueError as exc:
        sm = None
        mass_src = f"not computed: {exc}"
    sprung_corner = ({"front": sm.mass_kg * sm.front_share / 2.0,
                      "rear": sm.mass_kg * (1.0 - sm.front_share) / 2.0} if sm is not None
                     else {"front": nan, "rear": nan})
    notes: list[str] = []          # unusable spring / MR entries
    tnotes: list[str] = []         # unusable tyre-rate entries
    wr, rr, fn, ride, fn_ride = {}, {}, {}, {}, {}
    for axle, sk, mk, tk in (("front", "spring_f", "mr_f", "tyre_rate_f"),
                             ("rear", "spring_r", "mr_r", "tyre_rate_r")):
        k = _read(s, sk, f"{axle} spring rate", notes)
        mr = _read(s, mk, f"{axle} motion ratio", notes)
        kt = _read(s, tk, f"{axle} tyre rate", tnotes)
        if mr > 0 and not MR_PLAUSIBLE[0] <= mr <= MR_PLAUSIBLE[1]:
            # flagged, never rejected: the designer owns the number
            notes.append(f"{axle} MR {mr:g} is outside {MR_PLAUSIBLE[0]:g}-"
                         f"{MR_PLAUSIBLE[1]:g}: check MR definition (spring/wheel)")
        wr[axle] = k * mr * mr if k > 0 and mr > 0 else nan
        t = geo.track_mm(axle) / 1000.0
        rr[axle] = 0.5 * wr[axle] * 1000.0 * t * t * math.pi / 180.0
        m_c = sprung_corner[axle]
        fn[axle] = math.sqrt(wr[axle] * 1000.0 / m_c) / (2 * math.pi) \
            if m_c > 0 and math.isfinite(wr[axle]) else nan
        # ride rate: wheel rate in series with the tyre
        ride[axle] = wr[axle] * kt / (wr[axle] + kt) if kt > 0 and math.isfinite(wr[axle]) \
            else nan
        fn_ride[axle] = math.sqrt(ride[axle] * 1000.0 / m_c) / (2 * math.pi) \
            if m_c > 0 and math.isfinite(ride[axle]) else nan

    def fmt(x: float, d: int = 2) -> str:
        return "—" if not math.isfinite(x) else f"{x:.{d}f}"

    src = "derived from the inputs above (spring × MR²), not from the linkage"
    if notes:
        src += "; " + "; ".join(notes)
    rows.append(["Suspension", "Wheel rate", "N/mm", fmt(wr["front"]), fmt(wr["rear"]), src])
    rows.append(["Suspension", "Roll rate from springs", "N·m/deg", fmt(rr["front"], 1),
                 fmt(rr["rear"], 1), src])
    rows.append(["Suspension", "Sprung natural frequency", "Hz", fmt(fn["front"]),
                 fmt(fn["rear"]), src + "; sprung corner mass from vehicle data; tyre "
                 "compliance NOT included (wheel-rate frequency, an upper bound on the ride "
                 "frequency)"])
    if any(str(s.get(key, "") or "").strip() for key in ("tyre_rate_f", "tyre_rate_r")):
        tsrc = "wheel rate in series with the tyre rate (estimate: no TTC data)"
        if tnotes:
            tsrc += "; " + "; ".join(tnotes)
        rows.append(["Suspension", "Ride rate (wheel rate + tyre in series)", "N/mm",
                     fmt(ride["front"]), fmt(ride["rear"]), tsrc + "; kw·kt / (kw + kt)"])
        rows.append(["Suspension", "Ride frequency (with tyre)", "Hz", fmt(fn_ride["front"]),
                     fmt(fn_ride["rear"]), tsrc + "; sprung corner mass from vehicle data"])

    # what the typed springs imply on their own -- information, not a target
    k_tot = rr["front"] + rr["rear"]
    share = rr["front"] / k_tot if math.isfinite(k_tot) and k_tot > 0 else nan
    grad, gsrc = nan, ""
    rc_f, rc_r = kpis.value("front.rc_static"), kpis.value("rear.rc_static")
    if not math.isfinite(k_tot):
        gsrc = "needs the spring rate and motion ratio of both axles"
    elif sm is None:
        gsrc = mass_src
    elif not (math.isfinite(rc_f) and math.isfinite(rc_r)):
        gsrc = "not computed: static roll centre unavailable"
    else:
        h_axis = rc_f + (rc_r - rc_f) * sm.cg_from_front_axle_mm / v.wheelbase_mm
        wsd = sm.mass_kg * GRAVITY * (sm.cg_height_mm - h_axis) / 1000.0   # N·m per g
        denom = k_tot - wsd * math.pi / 180.0
        if wsd <= 0:
            gsrc = "not computed: roll axis at or above the sprung CG"
        elif denom <= 0:
            gsrc = "not computed: the springs alone do not hold the pendulum moment"
        else:
            grad = wsd / denom
    rows.append(["Suspension", "Roll stiffness front share (springs only)", "-", fmt(share, 3),
                 "N/A", "roll rates above; information: ARB and tyre not included"])
    rows.append(["Suspension", "Roll gradient (springs only; no ARB, no tyre)", "deg/g",
                 fmt(grad, 3), "N/A",
                 gsrc or "W_s·d / (K − W_s·d·π/180) with the pendulum term; sprung CG, solved "
                         "static RCs; information: ARB and tyre not included"])

    def pair(axle: str, name: str, d: str = ".3f") -> str:
        a, b = kpis.get(f"{axle}.{name}_L"), kpis.get(f"{axle}.{name}_R")
        return f"{a.text() if a else '—'} / {b.text() if b else '—'}"

    def one(key: str) -> str:
        k = kpis.get(key)
        return k.text() if k else "—"

    dw = "vdcore DWSolver"
    rows += [
        ["Kinematics", "Ride camber", "deg/m", one("front.ride_camber"),
         one("rear.ride_camber"), dw],
        ["Kinematics", "Roll camber (chassis-ref.)", "deg/deg", one("front.roll_camber"),
         one("rear.roll_camber"), dw],
        ["Kinematics", "Anti-dive / anti-squat", "%", one("front.anti"), one("rear.anti"),
         "pivot-axis rake construction"],
        ["Kinematics", "Bump steer (linear, per side)", "deg/mm", one("front.bump_steer"),
         one("rear.bump_steer"), dw],
        ["Kinematics", "Bump steer (peak, per side)", "deg", one("front.bump_steer_peak"),
         one("rear.bump_steer_peak"), dw],
        ["Kinematics", "Roll centre height, static", "mm", one("front.rc_static"),
         one("rear.rc_static"), dw],
        ["Kinematics", "Roll centre height in roll", "mm", one("front.roll_rc_height"),
         one("rear.roll_rc_height"), dw],
        ["Alignment", "KPI (L / R)", "deg", pair("front", "kpi"), pair("rear", "kpi"), dw],
        ["Alignment", "Caster (L / R)", "deg", pair("front", "caster"), pair("rear", "caster"),
         dw],
        ["Alignment", "Scrub radius (L / R)", "mm", pair("front", "scrub"), pair("rear", "scrub"),
         dw],
        ["Alignment", "Mechanical trail (L / R)", "mm", pair("front", "trail"),
         pair("rear", "trail"), dw],
        ["Alignment", "Static camber (L / R)", "deg", pair("front", "camber"),
         pair("rear", "camber"), dw],
        ["Alignment", "Static total toe (+ in)", "deg", one("front.total_toe"),
         one("rear.total_toe"), dw],
        ["Steering", "Ackermann (solved)", "%", one("steering.ackermann"), "N/A", dw],
        ["Steering", "Steering ratio", ":1", one("steering.ratio"), "N/A", dw],
        ["Steering", "Rack travel per pinion rev", "mm/rev", one("steering.rack_per_rev"), "N/A",
         "2π·r_pinion"],
        ["Steering", "Steering arm length (L / R)", "mm",
         f"{one('steering.arm_L')} / {one('steering.arm_R')}", "N/A", "hardpoints"],
        ["Masses", "Total mass with driver", "kg", f"{v.total_mass_kg:g}", f"{v.total_mass_kg:g}",
         "vehicle data"],
        ["Masses", "Weight distribution", "%", f"{100 * v.front_mass_fraction:.1f}",
         f"{100 * (1 - v.front_mass_fraction):.1f}", "vehicle data"],
        ["Masses", "Sprung mass per corner", "kg", fmt(sprung_corner["front"], 1),
         fmt(sprung_corner["rear"], 1), mass_src],
    ]
    return rows


def rows_csv(head: list[str], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(head)
    w.writerows(rows)
    return buf.getvalue()
