#!/usr/bin/env python3
r"""Compare a chassis CAD survey with the project's suspension pick-ups.

    & .venv\Scripts\python.exe scripts\compare_chassis.py "projects\cad\Xasis 33.3_survey.json"

Reads the survey (``scripts/survey_chassis.py``) and a VD Studio project
(default: the team project, ``projects/FSAE_2027.vdproj.json``). The CAD
placement -- which CAD axes point forward and left, and where the ISO origin
sits in CAD -- comes from the project (CAD & cross-checks > SolidWorks) or
from ``--forward/--left/--origin``. Writes next to the survey:
``<model>_comparison.md`` (table), ``.csv`` and ``.svg`` (top, side and front
views of the frame with the suspension and every pick-up joined to its nearest
frame node).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from apps.vdstudio.engine import chassis as C  # noqa: E402
from apps.vdstudio.engine import exports as X  # noqa: E402
from apps.vdstudio.engine import geometry as G  # noqa: E402
from apps.vdstudio.engine.project import load_project  # noqa: E402
from apps.vdstudio.engine.recent import TEAM_PROJECT  # noqa: E402

swx = X.swx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare a chassis CAD survey with the "
                                             "project's suspension pick-ups.")
    ap.add_argument("survey", type=Path, help="<model>_survey.json from survey_chassis.py")
    ap.add_argument("--project", type=Path, default=TEAM_PROJECT,
                    help=f"VD Studio project (default: {TEAM_PROJECT.name})")
    swx.add_placement_args(ap)
    ap.add_argument("--frame", default="iso", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        project = load_project(args.project)
        frame = (swx.frame_from_args(args) if args.forward or args.origin
                 else X.cad_frame(project))
        survey = C.load_survey(args.survey)
        geo = G.resolve(project)
        rows = C.compare(geo, survey, frame)
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    stem = args.survey.name.removesuffix(".json").removesuffix("_survey")
    # with_name, not with_suffix: "Xasis 33.3" has a dot in the stem
    out = {ext: args.survey.with_name(f"{stem}_comparison.{ext}") for ext in ("md", "csv", "svg")}
    out["svg"].write_text(C.views_svg(geo, survey, frame, rows), encoding="utf-8")
    out["csv"].write_text(C.report_csv(rows), encoding="utf-8")
    md = C.report_markdown(rows, survey, frame, project.name, out["svg"].name)
    out["md"].write_text(md, encoding="utf-8")
    print(md)
    print("Wrote " + ", ".join(str(p) for p in out.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
