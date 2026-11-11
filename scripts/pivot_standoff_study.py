"""Inboard wishbone pivots moved off the frame nodes: what changes, and by how much.

The chassis CAD uses the inboard wishbone pivots as tube nodes, so a pivot bracket
needs the pivot some 30-35 mm away from the node. This script moves the inboard
pivots of a hardpoint-mode project (ball joints, tie-rod ends and wheel centres
stay) and compares every variant with the project as it is, on ``DWSolver``:

* ``leg D``      each pivot D mm along its own leg, toward the ball joint;
* ``lateral D``  each pivot D mm outboard in Y only;
* ``leg D + toe link``  ``leg D`` plus the rear toe link's inboard end D mm along
  the toe link.

Sliding a pivot along its leg keeps it in the wishbone's plane (front pivot, rear
pivot, ball joint). The ball joint's velocity is perpendicular to that plane
whatever axis in the plane the arm turns about, so the upright's first-order
motion at ride is the same: only the curvature (the arm is shorter) changes.
The report shows both groups. It presents consequences; it does not pick a value.

usage: uv run python scripts/pivot_standoff_study.py [--project P] [--out FILE]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from apps.vdstudio.engine import curves as C  # noqa: E402
from apps.vdstudio.engine import geometry as G  # noqa: E402
from apps.vdstudio.engine import kpis as K  # noqa: E402
from apps.vdstudio.engine import suspension as S  # noqa: E402
from apps.vdstudio.engine.project import Project, load_project  # noqa: E402
from vdcore.analysis.rocker import rocker_sweep  # noqa: E402

ARMS = (("UCA_IN_FRONT", "UCA_IN_REAR", "UCA_OUT"), ("LCA_IN_FRONT", "LCA_IN_REAR", "LCA_OUT"))
TRAVEL_MM = 36.0            # the packaging range of the 550 lb/in layouts (design_intent)
ROLL_DEG = 1.5


def _unit(v: np.ndarray) -> np.ndarray:
    return v / float(np.linalg.norm(v))


def moved(project: Project, kind: str, d_mm: float, toe_link: bool = False) -> Project:
    """``project`` with its inboard pivots moved (left corners; right ones mirrored)."""
    if project.geometry_mode != "hardpoints":
        raise SystemExit("the project is not in hardpoint mode")
    p = copy.deepcopy(project)
    pts = p.hardpoints.points
    for cid in ("FL", "RL"):
        for f, r, o in ARMS:
            pf, pr, po = (np.array(pts[cid][k], float) for k in (f, r, o))
            if kind == "leg":
                nf, nr = pf + d_mm * _unit(po - pf), pr + d_mm * _unit(po - pr)
            elif kind == "lateral":
                nf, nr = pf + np.array([0.0, d_mm, 0.0]), pr + np.array([0.0, d_mm, 0.0])
            else:
                raise ValueError(kind)
            pts[cid][f], pts[cid][r] = tuple(nf), tuple(nr)
        if toe_link and cid == "RL":
            ti, to = (np.array(pts[cid][k], float) for k in ("TIE_ROD_IN", "TIE_ROD_OUT"))
            pts[cid]["TIE_ROD_IN"] = tuple(ti + d_mm * _unit(to - ti))
    for left, right in (("FL", "FR"), ("RL", "RR")):
        pts[right] = {n: (x, -y + 0.0, z) for n, (x, y, z) in pts[left].items()}
    return p


def measure(p: Project) -> dict:
    geo = G.resolve(p)
    ks = K.compute_all(geo)
    out: dict = {"kpi": {k.key: k.value for k in ks.items.values() if k.ok}}
    hv = C.heave_curves(geo, -TRAVEL_MM, TRAVEL_MM, n=13)
    rl = C.roll_curves(geo, ROLL_DEG, n=4)
    for axle, cid in (("front", "FL"), ("rear", "RL")):
        q = hv.corners[cid]
        out[axle] = {
            "camber_lo": q["camber"][0], "camber_hi": q["camber"][-1],
            "toe_lo": q["toe"][0], "toe_hi": q["toe"][-1],
            "toe_peak": float(np.nanmax(np.abs(q["toe"]))),
            "rc_lo": q["rc"][0], "rc_hi": q["rc"][-1],
            "half_track_lo": q["half_track"][0], "half_track_hi": q["half_track"][-1],
            "roll_rc_lat": rl.axles[axle]["rc_lateral"][-1],
            "roll_rc_h": rl.axles[axle]["rc_height"][-1],
            "roll_outer_camber": rl.axles[axle]["outer_camber"][-1],
        }
        spec = p.rocker_front if axle == "front" else p.rocker_rear
        if spec.enabled:
            h = 0.5
            zs = np.array([-TRAVEL_MM, 0.0, TRAVEL_MM])
            L = rocker_sweep(geo.corner(cid), S.layout_for(spec, cid),
                             np.concatenate([zs - h, zs + h])).damper_length_mm
            out[axle]["mr"] = (-(L[3:] - L[:3]) / (2 * h)).tolist()
    return out


# (label, unit, format, getter)
def _k(key: str):
    return lambda m, ax: m["kpi"].get(f"{ax}.{key}", float("nan"))


def _a(key: str):
    return lambda m, ax: m[ax][key]


FIRST_ORDER = (
    ("Roll centre height, static", "mm", ".2f", _k("rc_static")),
    ("FVSA length", "mm", ".1f", _k("fvsa")),
    ("Camber gain", "deg/mm", ".4f", _k("camber_gain")),
    ("Roll camber (chassis-ref.)", "deg/deg", ".3f", _k("roll_camber")),
    ("Half-track change", "mm/mm", ".3f", _k("half_track_change")),
    ("Bump steer, linear (per side)", "deg/mm", ".4f", _k("bump_steer")),
    ("Anti-dive (front) / anti-squat (rear), 3D virtual work", "%", ".2f", _k("anti_3d")),
)
CURVATURE = (
    ("RC migration (chassis-ref.)", "mm/mm", ".3f", _k("rc_migration")),
    (f"RC height at -{TRAVEL_MM:g} / +{TRAVEL_MM:g} mm (chassis-ref.)", "mm", ".1f",
     lambda m, ax: (m[ax]["rc_lo"], m[ax]["rc_hi"])),
    (f"RC lateral at {ROLL_DEG:g} deg of roll", "mm", ".1f", _a("roll_rc_lat")),
    (f"RC height at {ROLL_DEG:g} deg of roll", "mm", ".2f", _a("roll_rc_h")),
    (f"Outer wheel camber at {ROLL_DEG:g} deg of roll (road)", "deg", ".3f",
     _a("roll_outer_camber")),
    (f"Camber at -{TRAVEL_MM:g} / +{TRAVEL_MM:g} mm", "deg", ".3f",
     lambda m, ax: (m[ax]["camber_lo"], m[ax]["camber_hi"])),
    (f"Toe change at -{TRAVEL_MM:g} / +{TRAVEL_MM:g} mm (per side)", "deg", ".3f",
     lambda m, ax: (m[ax]["toe_lo"], m[ax]["toe_hi"])),
    (f"Peak |toe change| over +-{TRAVEL_MM:g} mm (per side)", "deg", ".3f", _a("toe_peak")),
    (f"Half-track change at -{TRAVEL_MM:g} / +{TRAVEL_MM:g} mm", "mm", ".2f",
     lambda m, ax: (m[ax]["half_track_lo"], m[ax]["half_track_hi"])),
    (f"Motion ratio at -{TRAVEL_MM:g} / 0 / +{TRAVEL_MM:g} mm (the project's rocker)", "-",
     ".3f", lambda m, ax: tuple(m[ax].get("mr", (float("nan"),) * 3))),
)


def _fmt(v, fmt: str) -> str:
    if isinstance(v, tuple):
        return " / ".join(format(float(x), fmt) for x in v)
    return format(float(v), fmt)


def _table(rows, cases: list[tuple[str, dict]], axle: str) -> list[str]:
    out = ["| " + " | ".join(["Quantity"] + [n for n, _ in cases]) + " |",
           "|" + "---|" * (len(cases) + 1)]
    for label, unit, fmt, get in rows:
        out.append(f"| {label} [{unit}] | "
                   + " | ".join(_fmt(get(m, axle), fmt) for _, m in cases) + " |")
    return out


def report(project: Project, source: str, d_mm: tuple[float, ...]) -> str:
    cases = [("as is", measure(project))]
    for d in d_mm:
        cases.append((f"leg {d:g}", measure(moved(project, "leg", d))))
    cases.append((f"lateral {d_mm[-1]:g}", measure(moved(project, "lateral", d_mm[-1]))))
    cases.append((f"leg {d_mm[-1]:g} + toe link",
                  measure(moved(project, "leg", d_mm[-1], toe_link=True))))
    L = [
        "# Inboard pivots moved off the frame nodes",
        "",
        f"Source: `{source}`, by `scripts/pivot_standoff_study.py`. Every number is solved on "
        "`DWSolver` with the ball joints, tie-rod ends and wheel centres where they are, so "
        "the static alignment (camber, toe, caster, KPI, scrub, trail, track) is the same in "
        "every column. Roll-centre heights are chassis-referenced.",
        "",
        "- **leg D**: each inboard wishbone pivot D mm along its own leg toward the ball "
        "joint (the leg is D mm shorter, on the same line).",
        "- **lateral D**: each inboard pivot D mm outboard in Y only.",
        "- **leg D + toe link**: leg D, plus the rear toe link's inboard end D mm along the "
        "toe link. It changes nothing at the front.",
        "",
        "A pivot slid along its leg stays in the wishbone's plane, and a ball joint moves "
        "perpendicular to that plane about any axis lying in it: the first-order motion at "
        "ride is the same and only the curvature changes. A lateral shift tilts the plane.",
        "",
    ]
    for axle in ("front", "rear"):
        L += [f"## {axle.title()} axle", "", "First order, at ride:", ""]
        L += _table(FIRST_ORDER, cases, axle)
        L += ["", "Curvature, over the travel:", ""]
        L += _table(CURVATURE, cases, axle)
        L.append("")
    L += [
        "## Not covered",
        "",
        "- The pivot brackets themselves (tabs, bolts, bearing housings) and their fit next to "
        "the rocker and coil-over brackets that sit near the same nodes.",
        "- Loads: a shorter pivot base raises the leg forces for the same wheel load; no "
        "structural number is computed here.",
        "- The motion-ratio rows use the project's rocker as it is: a rocker is re-tuned to "
        "its target MR after the hardpoints move (`scripts/rocker_concepts.py`).",
        "- The side-view construction KPI (`anti`) uses the pivot-axis rake and moves a little "
        "with a leg slide; the 3D virtual-work figure (`anti_3d`) is the one tabulated.",
        "",
    ]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", type=Path, default=REPO / "projects" / "FSAE_2027.vdproj.json")
    ap.add_argument("--out", type=Path, default=REPO / "projects" / "pivot_standoff_study.md")
    ap.add_argument("--d", type=float, nargs="+", default=[30.0, 35.0],
                    help="stand-off distances [mm]")
    args = ap.parse_args()
    text = report(load_project(args.project), args.project.name, tuple(args.d))
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
