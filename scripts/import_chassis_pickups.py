#!/usr/bin/env python3
r"""Make a project variant whose chassis pick-ups come from the chassis CAD.

    & .venv\Scripts\python.exe scripts\import_chassis_pickups.py ^
        "projects\cad\Xasis 33.3_survey.json" --sketches PSD PST

The 16 inboard wishbone pick-ups become the nearest frame nodes of the CAD
survey (source ``cad``), mapped with the project's CAD placement; every other
point -- uprights, tie rods, wheels, which the chassis model does not have --
keeps its value from the project. The result is a NEW project in hardpoint
mode, ``projects/<project>_<model>_pickups.vdproj.json``: the team project is
never overwritten. The script prints the KPIs of both side by side; it does not
say which geometry to keep.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from apps.vdstudio.engine import chassis as C  # noqa: E402
from apps.vdstudio.engine import exports as X  # noqa: E402
from apps.vdstudio.engine import geometry as G  # noqa: E402
from apps.vdstudio.engine import kpis as K  # noqa: E402
from apps.vdstudio.engine.project import load_project, save_project  # noqa: E402
from apps.vdstudio.engine.recent import PROJECTS_DIR, TEAM_PROJECT  # noqa: E402

# KPIs printed side by side (key, label); all exist in engine/kpis.py
SHOWN = (
    ("front.rc_static", "Front RC height"), ("rear.rc_static", "Rear RC height"),
    ("front.kpi_L", "Front KPI"), ("front.caster_L", "Front caster"),
    ("front.scrub_L", "Front scrub radius"), ("front.trail_L", "Front mech. trail"),
    ("front.camber_gain", "Front camber gain"), ("rear.camber_gain", "Rear camber gain"),
    ("front.roll_camber", "Front roll camber"), ("rear.roll_camber", "Rear roll camber"),
    ("front.rc_migration", "Front RC migration"), ("rear.rc_migration", "Rear RC migration"),
    ("front.bump_steer", "Front bump steer"), ("rear.bump_steer", "Rear bump steer"),
    ("front.anti", "Front anti (dive)"), ("rear.anti", "Rear anti (squat)"),
    ("front.uca_lca_ratio", "Front UCA/LCA ratio"), ("rear.uca_lca_ratio", "Rear UCA/LCA ratio"),
    ("front.fvsa", "Front FVSA"), ("rear.fvsa", "Rear FVSA"),
)


def _fmt(ks: K.KpiSet, key: str) -> tuple[str, float | None]:
    k = ks.get(key)
    if k is None:
        return "—", None
    if not k.ok:
        return "FAILED", None
    return f"{k.value:{k.fmt}} {k.unit}".strip(), float(k.value)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Project variant with the chassis pick-ups "
                                             "from the CAD.")
    ap.add_argument("survey", type=Path, help="<model>_survey.json from survey_chassis.py")
    ap.add_argument("--project", type=Path, default=TEAM_PROJECT)
    ap.add_argument("--sketches", nargs="+", default=None, metavar="NAME",
                    help="take pick-up nodes only from these CAD sketches (e.g. PSD PST)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        base = load_project(args.project)
        frame = X.cad_frame(base)
        survey = C.load_survey(args.survey)
        hs, rows = C.cad_hardpoints(base, survey, frame,
                                    tuple(args.sketches) if args.sketches else None)
        variant = copy.copy(base)
        variant.hardpoints = hs
        variant.geometry_mode = "hardpoints"
        variant.name = f"{base.name.split('--')[0].strip()} -- pick-ups from {survey.title}"
        variant.notes = (base.notes + "\n\n" if base.notes else "") + hs.origin
        geo_b, geo_v = G.resolve(base), G.resolve(variant)
        kb, kv = K.compute_all(geo_b), K.compute_all(geo_v)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    stem = args.survey.name.removesuffix(".json").removesuffix("_survey").replace(" ", "_")
    name = f"{args.project.name.split('.')[0]}_{stem}_pickups.vdproj.json"
    out = args.out or PROJECTS_DIR / name
    save_project(variant, out)

    print(f"Pick-ups taken from the CAD ({len(rows)}), ISO 8855 mm:")
    for r in rows:
        dx, dy, dz = r.delta_mm  # type: ignore[misc]
        print(f"  {r.corner} {r.point:<13} {r.node_sketch:<10} moved {r.distance_mm:5.1f} mm "
              f"({dx:+.1f}, {dy:+.1f}, {dz:+.1f})")
    print(f"\n{'KPI':<22}{'project':>16}{'with CAD pick-ups':>20}{'change':>12}")
    for key, label in SHOWN:
        (tb, vb), (tv, vv) = _fmt(kb, key), _fmt(kv, key)
        d = f"{vv - vb:+.3g}" if vb is not None and vv is not None else ""
        print(f"{label:<22}{tb:>16}{tv:>20}{d:>12}")
    print(f"\nWrote {out}\n(the team project {args.project.name} is unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
