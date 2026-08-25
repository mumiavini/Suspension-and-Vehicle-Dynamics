#!/usr/bin/env python3
"""
geometry_summary.py
===================

Single source of truth for the FSAE 2027 geometry summary document.

``sla_geometry.py`` owns the wishbones. ``steering_geometry.py`` owns caster and
the front tie rod, and supersedes the wishbone script's zero-caster outboard ball
joints. Neither script alone can produce the summary the team hands to a judge,
and the previous summary ("Hardpoints Suspensao 2027.pdf") was assembled by hand
from stale runs of both. This script composes them instead, adds the checks that
fell between the two, and emits the document.

WHAT THIS FIXES relative to that PDF
------------------------------------
Errata found by re-deriving every published number from the exported hardpoints:

  1. Rate tables were generated before the upright-rotation sign fix
     (commit 0e7e524, 2026-08-20). Half-track change was wrong in SIGN.
     -> fixed at the source: this script always uses the corrected sweep.
  2. Section 1 assumed 55 % front mass; the leg-force tables assumed 45 %.
     -> fixed: every number here comes from one VehicleData instance, and the
        load-case assumptions are printed with the table (section 7).
  3. The 3D point tables were captioned "x positivo para tras" while carrying
     ISO 8855 data (x positive FORWARD). Read literally that inverts caster.
     -> fixed: every table states its own frame, and the two frames are named
        differently everywhere (DESIGN vs ISO 8855).
  4. Front sweep / e-a rows described the pre-caster geometry (0 / 0 mm).
     -> fixed: sweep is measured off the MERGED hardpoints (section 5).
  5. The rear upright length fail was softened to "OK quase", and the front was
     checked in front view only, hiding that it is also over once caster counts.
     -> fixed: both the front-view and the true 3D kingpin length are checked.
  6. "bitola atual 1200 mm" was the rear track; the front is 1240 mm and neither
     was stated. -> fixed: section 1 names both and says which one the tilt test
     uses and why.
  7. Front-view points were rounded to integers, so the table could not
     reproduce its own FVIC/FVSA/RC. -> fixed: two decimals throughout.

And the KPIs that were absent entirely: caster, mechanical trail, bump steer,
Ackermann, steering ratio, rack travel, effort (section 4); true 3D member
lengths (section 5); wheelbase and track rules compliance (section 1); the
static camber / toe convention actually encoded in the export (section 6).

USAGE
-----
    python scripts/geometry_summary.py                 # print the report
    python scripts/geometry_summary.py --md OUT.md     # write Markdown
    python scripts/geometry_summary.py --csv OUT.csv   # write merged hardpoints
    python scripts/geometry_summary.py --verify        # check against the CSV
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402

W = 78
RULE = "=" * W
CSV_PATH = REPO / "legacy_app" / "carro_formula_2027.csv"

# FSAE rules the geometry alone can be checked against.
FSAE_MIN_WHEELBASE_MM = 1525.0
FSAE_MIN_TRACK_RATIO = 0.75
ARM_LENGTH_WINDOW_MM = (320.0, 430.0)
KINGPIN_WINDOW_MM = (200.0, 260.0)

POINT_ORDER = (
    "UCA_IN_FRONT", "UCA_IN_REAR", "UCA_OUT",
    "LCA_IN_FRONT", "LCA_IN_REAR", "LCA_OUT",
    "TIE_ROD_IN", "TIE_ROD_OUT",
    "WHEEL_CENTER", "CONTACT_PATCH",
)
CORNERS = ("FL", "FR", "RL", "RR")


def flag(ok: bool) -> str:
    return "OK  " if ok else "!!  "


def band(label: str, value: float, lo: float, hi: float,
         unit: str = "mm", fmt: str = "9.2f") -> str:
    return (f"  {flag(lo <= value <= hi)}{label:<34s}{value:{fmt}} {unit:<7s}"
            f"target {lo:g} to {hi:g}")


# --------------------------------------------------------------------------- #
# 1. MERGED HARDPOINT SET
# --------------------------------------------------------------------------- #

Pt = tuple[float, float, float]


@dataclass(frozen=True)
class MergedHardpoints:
    """All four corners in ISO 8855, with provenance per point group."""

    points: dict[str, dict[str, Pt]]
    rear_tie_rod_from_csv: bool

    def rows(self) -> Iterator[tuple[str, str, float, float, float]]:
        for corner in CORNERS:
            for name in POINT_ORDER:
                pt = self.points[corner].get(name)
                if pt is not None:
                    yield corner, name, pt[0], pt[1], pt[2]

    def arr(self, corner: str, name: str) -> np.ndarray:
        return np.array(self.points[corner][name], dtype=float)


def _read_csv_points() -> dict[str, dict[str, Pt]]:
    out: dict[str, dict[str, Pt]] = {}
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["corner"], {})[row["point"]] = (
                float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"])
            )
    return out


def build_merged(design: sla.DesignReport, steer: stg.SteeringReport) -> MergedHardpoints:
    """sla_geometry wishbones + steering_geometry caster/tie rod + rear toe link.

    The rear tie rod is not synthesised by any script -- ``steering_geometry.py``
    covers the front axle only. It is read from the exported CSV and flagged, so
    the report can say plainly that it is a hand-entered point.
    """
    pts: dict[str, dict[str, Pt]] = {c: {} for c in CORNERS}

    for corner, name, x, y, z in design.model.rows():
        pts[corner][name] = (x, y, z)

    # steering_geometry supersedes the zero-caster outboard ball joints
    for corner, name, x, y, z in steer.hardpoints.rows():
        pts[corner][name] = (x, y, z)

    from_csv = False
    csv_pts = _read_csv_points()
    for corner in ("RL", "RR"):
        for name in ("TIE_ROD_IN", "TIE_ROD_OUT"):
            if name not in pts[corner] and name in csv_pts.get(corner, {}):
                pts[corner][name] = csv_pts[corner][name]
                from_csv = True

    return MergedHardpoints(points=pts, rear_tie_rod_from_csv=from_csv)


def verify_against_csv(hp: MergedHardpoints, tol_mm: float = 5e-4) -> list[str]:
    """Return a list of mismatches between the merged set and the exported CSV."""
    csv_pts = _read_csv_points()
    problems: list[str] = []

    for corner in CORNERS:
        mine = hp.points.get(corner, {})
        theirs = csv_pts.get(corner, {})
        for name in sorted(set(mine) | set(theirs)):
            a, b = mine.get(name), theirs.get(name)
            if a is None:
                problems.append(f"{corner}/{name}: in CSV, not generated")
            elif b is None:
                problems.append(f"{corner}/{name}: generated, missing from CSV")
            else:
                d = max(abs(u - v) for u, v in zip(a, b, strict=True))
                if d > tol_mm:
                    problems.append(
                        f"{corner}/{name}: generated {a} vs CSV {b}  (max |d| {d:.4f} mm)")
    return problems


# --------------------------------------------------------------------------- #
# 2. CHECKS THAT NEEDED THE MERGED SET
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MemberLengths:
    """True 3D lengths -- what gets cut, as opposed to the front-view projection."""

    corner: str
    lca_front: float
    lca_rear: float
    uca_front: float
    uca_rear: float
    tie_rod: float
    kingpin_3d: float
    kingpin_2d: float
    lca_sweep: float
    uca_sweep: float
    lca_base: float
    uca_base: float

    @property
    def legs(self) -> tuple[tuple[str, float], ...]:
        return (("LCA front leg", self.lca_front), ("LCA rear leg", self.lca_rear),
                ("UCA front leg", self.uca_front), ("UCA rear leg", self.uca_rear))


def member_lengths(hp: MergedHardpoints, corner: str) -> MemberLengths:
    p = lambda n: hp.arr(corner, n)  # noqa: E731
    lbj, ubj = p("LCA_OUT"), p("UCA_OUT")
    d = ubj - lbj

    lca_mid_x = 0.5 * (p("LCA_IN_FRONT")[0] + p("LCA_IN_REAR")[0])
    uca_mid_x = 0.5 * (p("UCA_IN_FRONT")[0] + p("UCA_IN_REAR")[0])

    tie = float("nan")
    if "TIE_ROD_IN" in hp.points[corner]:
        tie = float(np.linalg.norm(p("TIE_ROD_OUT") - p("TIE_ROD_IN")))

    return MemberLengths(
        corner=corner,
        lca_front=float(np.linalg.norm(p("LCA_OUT") - p("LCA_IN_FRONT"))),
        lca_rear=float(np.linalg.norm(p("LCA_OUT") - p("LCA_IN_REAR"))),
        uca_front=float(np.linalg.norm(p("UCA_OUT") - p("UCA_IN_FRONT"))),
        uca_rear=float(np.linalg.norm(p("UCA_OUT") - p("UCA_IN_REAR"))),
        tie_rod=tie,
        kingpin_3d=float(np.linalg.norm(d)),
        kingpin_2d=float(math.hypot(d[1], d[2])),
        lca_sweep=abs(float(p("LCA_OUT")[0] - lca_mid_x)),
        uca_sweep=abs(float(p("UCA_OUT")[0] - uca_mid_x)),
        lca_base=abs(float(p("LCA_IN_FRONT")[0] - p("LCA_IN_REAR")[0])),
        uca_base=abs(float(p("UCA_IN_FRONT")[0] - p("UCA_IN_REAR")[0])),
    )


def static_alignment_encoded(hp: MergedHardpoints, corner: str) -> tuple[float, float]:
    """Static camber and toe the exported CONTACT_PATCH actually encodes.

    The contact patch is exported directly below the wheel centre, which is a
    zero-camber, zero-toe convention. The rate tables assume the design-intent
    camber instead. Whoever loads these points needs to know which they get.
    """
    wc, cp = hp.arr(corner, "WHEEL_CENTER"), hp.arr(corner, "CONTACT_PATCH")
    dz = wc[2] - cp[2]
    if abs(dz) < 1e-9:
        return 0.0, 0.0
    camber = -math.degrees(math.atan2(cp[1] - wc[1], dz)) * (1 if wc[1] >= 0 else -1)
    toe = math.degrees(math.atan2(cp[0] - wc[0], dz))
    return camber + 0.0, toe + 0.0  # normalise -0.0


# --------------------------------------------------------------------------- #
# 3. REPORT
# --------------------------------------------------------------------------- #

def provenance_report(hp: MergedHardpoints) -> str:
    L = ["", RULE, " 0. PROVENANCE", RULE]
    L.append("   Wishbone hardpoints, static KPIs, rates, roll, anti-geometry,")
    L.append("     leg forces                     sla_geometry.py")
    L.append("   Caster, outboard ball joints, tie rod, rack, bump steer,")
    L.append("     Ackermann, steering effort     steering_geometry.py")
    L.append("   3D kinematic solve behind the steering numbers")
    L.append("                                    vdcore.geometry.solver.DWSolver")
    if hp.rear_tie_rod_from_csv:
        L.append("")
        L.append("  !!  REAR TIE ROD IS NOT SYNTHESISED BY ANY SCRIPT.")
        L.append("      steering_geometry.py covers the FRONT axle only. RL/RR")
        L.append("      TIE_ROD_IN and TIE_ROD_OUT are hand-entered points read")
        L.append("      from legacy_app/carro_formula_2027.csv. They are not")
        L.append("      regenerated, not bounds-checked, and not covered by any")
        L.append("      test. Treat them as measured-once inputs until a rear")
        L.append("      steering synthesis exists.")
    L.append("")
    L.append("   TWO FRAMES ARE IN USE. Every table below names the one it uses.")
    L.append("     DESIGN frame  y OUTBOARD, z UP, x REARWARD   (front-view work)")
    L.append("     ISO 8855      X FORWARD,  Y LEFT, Z UP       (model export)")
    L.append("     Conversion:   X_iso = -x_rearward,  Y_iso = +/- y_outboard")
    return "\n".join(L)


def rules_report(veh: sla.VehicleData, front: sla.AxleGeometry,
                 rear: sla.AxleGeometry, res: sla.VehicleResults) -> str:
    tf, tr = front.inputs.track_mm, rear.inputs.track_mm
    narrow, wide = min(tf, tr), max(tf, tr)
    ratio = narrow / wide

    L = ["", RULE, " 1b. RULES COMPLIANCE (geometry only)", RULE]
    L.append(f"   Front track                      {tf:9.1f} mm")
    L.append(f"   Rear track                       {tr:9.1f} mm")
    L.append(f"   Wheelbase                        {veh.wheelbase_mm:9.1f} mm")
    L.append("")
    L.append(band("Wheelbase", veh.wheelbase_mm, FSAE_MIN_WHEELBASE_MM, 1e4))
    L.append(f"      margin over the minimum       "
             f"{veh.wheelbase_mm - FSAE_MIN_WHEELBASE_MM:9.1f} mm")
    L.append(band("Narrow / wide track ratio", 100 * ratio,
                  100 * FSAE_MIN_TRACK_RATIO, 100.0, unit="%"))
    L.append(f"  {flag(narrow >= res.tilt_min_track_mm)}"
             f"{'60 deg tilt test':<34s}{narrow:9.1f} mm     "
             f"needs {res.tilt_min_track_mm:.1f}")
    L.append(f"      the tilt test uses the NARROWER track, which is the "
             f"{'front' if tf < tr else 'rear'} ({narrow:.0f} mm).")
    L.append(f"      Quoting it as 'the track' hides the {wide:.0f} mm "
             f"{'rear' if tf < tr else 'front'}.")
    return "\n".join(L)


def steering_section(sr: stg.SteeringReport) -> str:
    L = ["", RULE, " 4. STEERING  (front axle)", RULE]
    L.append("   Absent from the previous summary in its entirety. Caster and")
    L.append("   mechanical trail set steering feel; bump steer is the tie rod's")
    L.append("   primary KPI.")
    L.append("")
    body = sr.text.strip("\n").splitlines()
    skip = False
    for line in body:
        if line.strip().startswith("HARDPOINTS"):
            skip = True
        if line.strip() == "NOTES" or set(line.strip()) == {"="}:
            continue
        if not skip:
            L.append(line)
    return "\n".join(L)


def members_section(hp: MergedHardpoints) -> str:
    lo, hi = ARM_LENGTH_WINDOW_MM
    klo, khi = KINGPIN_WINDOW_MM

    L = ["", RULE, " 5. MEMBER LENGTHS AND SWEEP  (true 3D, ISO 8855)", RULE]
    L.append("   The front-view 'LCA length' in section 2/3 is a PROJECTION -- the")
    L.append("   right quantity for FVSA, but not the length anyone cuts. These are")
    L.append("   the real members, measured off the merged hardpoints.")
    for corner, label in (("FL", "Front"), ("RL", "Rear")):
        m = member_lengths(hp, corner)
        L.append("")
        L.append(f" {label.upper()} AXLE  [{corner}]")
        for name, val in m.legs:
            L.append(band(name, val, lo, hi))
        if not math.isnan(m.tie_rod):
            L.append(f"      {'Tie rod':<34s}{m.tie_rod:9.2f} mm")
        L.append("")
        L.append(band("Kingpin length, front view", m.kingpin_2d, klo, khi))
        L.append(band("Kingpin length, TRUE 3D", m.kingpin_3d, klo, khi))
        if abs(m.kingpin_3d - m.kingpin_2d) > 0.05:
            L.append("      the 3D length is the one that matters for the upright;")
            L.append("      checking the front-view value alone hides caster.")
        L.append("")
        L.append(f"      {'LCA base / sweep':<34s}{m.lca_base:9.2f} / {m.lca_sweep:.2f} mm"
                 f"   e/a {m.lca_sweep / (m.lca_base / 2):.3f}")
        L.append(f"      {'UCA base / sweep':<34s}{m.uca_base:9.2f} / {m.uca_sweep:.2f} mm"
                 f"   e/a {m.uca_sweep / (m.uca_base / 2):.3f}")
        L.append("      e/a is measured against the HALF base, and sweep is measured")
        L.append("      off the caster-corrected ball joint, not the input config.")
    return "\n".join(L)


def hardpoints_section(hp: MergedHardpoints) -> str:
    L = ["", RULE, " 6. MERGED HARDPOINTS  (ISO 8855)", RULE]
    L.append("   X POSITIVE FORWARD. Y POSITIVE LEFT. Z POSITIVE UP.")
    L.append("   Origin: front axle centreline, ground plane, vehicle centreline.")
    L.append("   The rear axle is therefore at NEGATIVE X. Right-hand corners are at")
    L.append("   NEGATIVE Y. This is not the design frame used in sections 2 and 3.")
    for corner in CORNERS:
        L.append("")
        L.append(f" [{corner}]")
        L.append(f"   {'point':<18s}{'X [mm]':>10s}{'Y [mm]':>11s}{'Z [mm]':>11s}")
        for name in POINT_ORDER:
            pt = hp.points[corner].get(name)
            if pt is None:
                continue
            star = "*" if name in ("WHEEL_CENTER", "CONTACT_PATCH") else " "
            L.append(f" {star} {name:<18s}{pt[0]:10.2f}{pt[1]:11.2f}{pt[2]:11.2f}")

    cam, toe = static_alignment_encoded(hp, "FL")
    L.append("")
    L.append(" (* reference points, not hardpoints)")
    L.append("")
    L.append("  !!  STATIC ALIGNMENT ENCODED IN THIS TABLE:")
    L.append(f"      camber {cam:+.2f} deg, toe {toe:+.2f} deg per side.")
    L.append("      CONTACT_PATCH sits directly below WHEEL_CENTER on all four")
    L.append("      corners, which is a ZERO-camber, ZERO-toe convention. The rate")
    L.append("      and roll tables assume the design-intent static camber instead.")
    L.append("      Build the design camber in for real and the contact patch moves")
    L.append("      outboard, which moves scrub radius -- see section 8.")
    return "\n".join(L)


def load_case_note(veh: sla.VehicleData) -> str:
    L = ["", RULE, " 7. LOAD CASE BEHIND THE LEG FORCES", RULE]
    L.append("   The previous summary printed leg forces with no stated load case,")
    L.append("   so they could not be checked. These are the assumptions:")
    L.append("")
    L.append(f"   Total mass                       {veh.total_mass_kg:9.1f} kg")
    L.append(f"   Front mass fraction              {100*veh.front_mass_fraction:9.1f} %"
             f"   <- also sets the roll axis")
    L.append(f"   CG height                        {veh.cg_height_mm:9.1f} mm")
    L.append("   Lateral friction coefficient          1.50        design_intent")
    L.append("   Longitudinal friction coefficient     1.40        design_intent")
    L.append("   Lateral acceleration                  1.50 g      design_intent")
    L.append("   Lateral load transfer distribution    0.55        design_intent")
    L.append("")
    L.append("   TWO LIMITS TO STATE WHEREVER THIS TABLE APPEARS:")
    L.append("")
    L.append("   1. The friction coefficients are a TYRE assumption. PUCPR Racing")
    L.append("      has no TTC data. Every force below inherits that assumption and")
    L.append("      none of it is measured.")
    L.append("   2. There is no pushrod in this model. An upright on two A-arms and")
    L.append("      a tie rod has five constraints against six degrees of freedom;")
    L.append("      the sixth member is the pushrod. The solve closes the system by")
    L.append("      treating the UCA as a two-force member with no X component and")
    L.append("      enforcing only the moment about the roll axis, and it places the")
    L.append("      front ball joints at zero caster, so braking generates no")
    L.append("      kingpin moment and the tie rod carries nothing.")
    L.append("")
    L.append("   These are screening numbers. They are NOT a load set to size tubes")
    L.append("   from, and they do not cover buckling of the long swept rear legs")
    L.append("   flagged in section 5.")
    return "\n".join(L)


def open_items(hp: MergedHardpoints, front: sla.AxleGeometry,
               rear: sla.AxleGeometry) -> str:
    L = ["", RULE, " 8. OPEN ITEMS", RULE]

    r = front.inputs.loaded_radius_mm
    for label, geo in (("Front", front), ("Rear", rear)):
        cam = abs(geo.inputs.static_camber_deg)
        shift = r * math.tan(math.radians(cam))
        moved = geo.scrub_radius_mm + shift
        L.append(f"   {label}: building the design {cam:.2f} deg of static camber into")
        L.append(f"     the contact patch moves it {shift:.2f} mm outboard and takes")
        L.append(f"     scrub radius from {geo.scrub_radius_mm:.2f} to {moved:.2f} mm."
                 f"{'  <-- leaves the 5 to 25 window' if moved > 25 else ''}")
    L.append("")
    L.append("   The static camber / toe convention has to be decided once and")
    L.append("   applied everywhere: either the export carries it and scrub moves,")
    L.append("   or it does not and the rate tables stop claiming it.")
    L.append("")
    if hp.rear_tie_rod_from_csv:
        L.append("   The rear tie rod has no synthesis script and no test.")
        L.append("")
    L.append("   Not in scope for this tool, and deliberately not computed here:")
    L.append("     wheel rates, motion ratio, ride frequency, damping")
    L.append("     stress, deflection, fatigue, buckling")
    L.append("     anything requiring tyre data, until TTC data is acquired")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 4. TOP LEVEL
# --------------------------------------------------------------------------- #

def build_report() -> tuple[str, MergedHardpoints]:
    design = sla.run()
    steer = stg.run()
    hp = build_merged(design, steer)

    veh = design.vehicle
    front, rear = design.front, design.rear

    parts = [
        RULE,
        " FSAE 2027 GEOMETRY SUMMARY -- PUCPR Racing",
        " Generated by scripts/geometry_summary.py. Do not hand-edit the output;",
        " fix the config in sla_geometry.py / steering_geometry.py and re-run.",
        RULE,
        provenance_report(hp),
        design.text.rstrip("\n"),
        rules_report(veh, front, rear, design.vehicle_results),
        steering_section(steer),
        members_section(hp),
        hardpoints_section(hp),
        load_case_note(veh),
        open_items(hp, front, rear),
        "",
    ]
    return "\n".join(parts) + "\n", hp


def to_markdown(text: str) -> str:
    """Wrap the fixed-width report so it survives a paste into Word or a PR."""
    head = [
        "# FSAE 2027 Geometry Summary",
        "",
        "PUCPR Racing (team #27) -- FSAE26 car. Generated by "
        "`scripts/geometry_summary.py`.",
        "",
        "> Do not hand-edit. Change the config in `sla_geometry.py` or",
        "> `steering_geometry.py` and re-run. The previous hand-assembled",
        "> summary carried stale rate tables and two different weight",
        "> distributions; that is what this script exists to prevent.",
        "",
        "```text",
    ]
    return "\n".join(head) + "\n" + text + "```\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate the FSAE 2027 geometry summary from the current scripts.")
    ap.add_argument("--md", metavar="PATH", help="write the report as Markdown")
    ap.add_argument("--csv", metavar="PATH", help="write the merged hardpoints as CSV")
    ap.add_argument("--verify", action="store_true",
                    help="check the merged hardpoints against the exported CSV")
    ap.add_argument("--quiet", action="store_true", help="suppress the report on stdout")
    args = ap.parse_args(argv)

    text, hp = build_report()

    if not args.quiet:
        print(text)

    if args.md:
        Path(args.md).write_text(to_markdown(text), encoding="utf-8")
        print(f"Markdown written to {args.md}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["corner", "point", "x_mm", "y_mm", "z_mm"])
            for corner, name, x, y, z in hp.rows():
                w.writerow([corner, name, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])
        print(f"Merged hardpoints written to {args.csv}")

    if args.verify:
        problems = verify_against_csv(hp)
        if problems:
            print(f"\n!!  {len(problems)} mismatch(es) vs {CSV_PATH.name}:")
            for p in problems:
                print(f"      {p}")
            return 1
        print(f"\nOK  merged hardpoints match {CSV_PATH.name} exactly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
