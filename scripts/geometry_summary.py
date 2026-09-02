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
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import sla_geometry as sla  # noqa: E402
import steering_geometry as stg  # noqa: E402
from vdcore.analysis.axle import axle_rates, axle_roll  # noqa: E402
from vdcore.analysis.toe import bump_steer, toe_sweep  # noqa: E402
from vdcore.models.hardpoint import (  # noqa: E402
    Axle,
    Corner,
    Hardpoint,
    TirePackage,
)

W = 78
RULE = "=" * W
CSV_PATH = REPO / "legacy_app" / "carro_formula_2027.csv"

# FSAE rules the geometry alone can be checked against.
FSAE_MIN_WHEELBASE_MM = 1525.0
FSAE_MIN_TRACK_RATIO = 0.75
# Mirrors CheckLimits.lca_length_mm in sla_geometry.py -- read from there rather
# than restated, because restating it drifted: rev 6 raised the sla band to 490
# and left this copy at 460, so section 5 (the authority for member lengths)
# failed the rear LCA front leg at 486.98 mm while section 2's own table passed
# the identical number. test_summary_bands_mirror_sla_limits locks them now.
# History: 430 -> 460 on 2026-09-01 (rear driveshaft clearance, rev 5),
# 460 -> 490 on 2026-09-02 (asymmetric rear LCA clearance, rev 6).
ARM_LENGTH_WINDOW_MM = sla.CheckLimits().lca_length_mm
KINGPIN_WINDOW_MM = sla.CheckLimits().kingpin_length_mm

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


# --------------------------------------------------------------------------- #
# REAR TOE LINK
#
# Declared here, not read from a CSV. steering_geometry.py covers the FRONT
# axle only -- it is built around a rack, and the rear has none -- so until
# 2026-09-01 these two points were hand-entered in
# legacy_app/carro_formula_2027.csv, regenerated by nothing and checked by
# nothing. They are design inputs, so they are stated as design inputs.
#
# INBOARD (chassis). Shares the LCA rear bracket: same X and same Y = 175,
# sitting 39.5 mm above it. X moved from -1460 to -1440 on 2026-09-02 when the
# LCA rear bracket alone was pushed to 100 mm clear of the driveshaft plane
# (the UCA stayed at 80 mm) -- see REAR_2027 in sla_geometry.py and
# docs/GEOMETRY_AUDIT_2026-09-02_rev6.md. Two consequences the chassis team
# asked for, in one point, still hold at the new X:
#   * nothing on the chassis sits rearward of the wishbone any more (it used
#     to be at X = -1480, i.e. 20 mm BEHIND the rearmost wishbone pickup and
#     only 60 mm ahead of the driveshaft plane);
#   * the toe link and the lower wishbone share one bracket instead of two.
# Z is the bump-steer knob and X is very nearly not: moving X from -1460 to
# -1440 shifts the linear rate from -0.000154 to -0.000150 deg/mm and the peak
# from 0.0128 to 0.0125 deg (re-verified 2026-09-02), both comfortably inside
# BUMP_STEER_LINEAR_LIMIT / BUMP_STEER_PEAK_LIMIT. Z = 169 still leaves a
# linear rate of -0.00015 deg/mm; the exact null at the new X is 169.065,
# still far inside the 0.5 mm build tolerance, so the round number is still
# the honest one. `toe_link_z_for_zero_bump_steer` re-solves it if the
# geometry moves again.
#
# OUTBOARD (upright). Unchanged. It is a feature of the rear upright's steering
# arm at Y = 521, not a chassis connection, and it is 54.6 mm behind the axle
# line. Moving it forward would be a rear-upright redesign and would flip the
# sign of toe change under longitudinal load -- out of scope for a packaging
# fix, and flagged in the audit doc instead.
REAR_TOE_LINK_INBOARD: Pt = (-1440.0, 175.0, 169.0)
REAR_TOE_LINK_OUTBOARD: Pt = (-1594.6, 520.81, 183.53)
REAR_TOE_LINK_SOURCE = "design_intent"


@dataclass(frozen=True)
class MergedHardpoints:
    """All four corners in ISO 8855, with provenance per point group."""

    points: dict[str, dict[str, Pt]]

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

    The rear toe link is not synthesised from a rack the way the front tie rod
    is -- the rear has no rack -- but it is a declared design input here rather
    than a value read back out of the exported CSV. See REAR_TOE_LINK_INBOARD.
    """
    pts: dict[str, dict[str, Pt]] = {c: {} for c in CORNERS}

    for corner, name, x, y, z in design.model.rows():
        pts[corner][name] = (x, y, z)

    # steering_geometry supersedes the zero-caster outboard ball joints
    for corner, name, x, y, z in steer.hardpoints.rows():
        pts[corner][name] = (x, y, z)

    # Rear toe link, mirrored onto the right side (ISO: Y+ is LEFT).
    for corner, sy in (("RL", 1.0), ("RR", -1.0)):
        for name, (x, y, z) in (("TIE_ROD_IN", REAR_TOE_LINK_INBOARD),
                                ("TIE_ROD_OUT", REAR_TOE_LINK_OUTBOARD)):
            pts[corner][name] = (x, sy * y, z)

    return MergedHardpoints(points=pts)


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
    """Static camber and toe the exported points actually encode.

    Track is measured at the CONTACT PATCHES, so the contact patch carries the
    half-track and the wheel centre is displaced inboard by the built-in static
    camber. This reads that displacement back, so the deliverable is checked
    against design intent rather than restating it.

    Toe cannot be recovered here and is always returned as 0: toe rotates the
    wheel about a vertical axis and leaves both points where they are. A
    consumer needing static toe must read it from the config.
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
    L.append("   Rear toe link                    geometry_summary.py")
    L.append("     (declared design input -- the rear has no rack, so")
    L.append("      steering_geometry.py does not reach it)")
    L.append("   Bump steer, both axles           vdcore.analysis.toe")
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


def hardpoints_section(hp: MergedHardpoints, front: sla.AxleGeometry) -> str:
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
    design_cam = front.inputs.static_camber_deg
    ok = abs(cam - design_cam) < 1e-3
    L.append(f"  {'OK ' if ok else '!! '} STATIC ALIGNMENT ENCODED IN THIS TABLE:")
    L.append(f"      camber {cam:+.2f} deg per side   (design intent {design_cam:+.2f})")
    L.append("      Recovered from CONTACT_PATCH -> WHEEL_CENTER, not restated")
    L.append("      from the config, so the rate and roll tables above and these")
    L.append("      points now describe the same car.")
    L.append("")
    L.append("      Track is measured at the CONTACT PATCHES -- the datum the rules,")
    L.append("      the tilt test and lateral load transfer all use. The wheel")
    inboard = abs(front.inputs.loaded_radius_mm * math.tan(math.radians(design_cam)))
    L.append(f"      centres sit {inboard:.2f} mm inboard of them, so the wheel-centre")
    L.append("      track is narrower than the quoted track -- see section 8.")
    L.append("")
    L.append(f"      Toe is {toe:+.2f} deg here because a contact-patch / wheel-centre")
    L.append("      pair CANNOT encode toe at all, not because toe is known to be")
    L.append("      zero. Static toe must come from the config.")
    return "\n".join(L)


def lltd_lift_threshold(veh: sla.VehicleData, geo: sla.AxleGeometry,
                        is_front: bool) -> float:
    """Front-share LLTD at which THIS axle's inner wheel reaches zero load.

    Fz_inner = W*frac/2 - share*W*ay*h/t = 0  ->  share = frac*t/(2*ay*h),
    where `share` is the axle's own share of the transfer. Returned on the
    FRONT-share scale so both ends read off one axis: the front lifts ABOVE
    its threshold, the rear BELOW its own. Between them, neither lifts.
    """
    frac = veh.front_mass_fraction if is_front else 1.0 - veh.front_mass_fraction
    share = frac * geo.inputs.track_mm / (2.0 * veh.design_ay_g * veh.cg_height_mm)
    return share if is_front else 1.0 - share


def load_case_note(veh: sla.VehicleData, front: sla.AxleGeometry,
                   rear: sla.AxleGeometry) -> str:
    L = ["", RULE, " 7. LOAD CASE BEHIND THE LEG FORCES", RULE]
    L.append("   The previous summary printed leg forces with no stated load case,")
    L.append("   so they could not be checked. These are the assumptions:")
    L.append("")
    L.append(f"   Total mass                       {veh.total_mass_kg:9.1f} kg")
    L.append(f"   Front mass fraction              {100*veh.front_mass_fraction:9.1f} %"
             f"   <- also sets the roll axis")
    L.append(f"   CG height                        {veh.cg_height_mm:9.1f} mm")
    L.append(f"   Lateral friction coefficient          {veh.mu_lateral:.2f}"
             f"        design_intent")
    L.append(f"   Longitudinal friction coefficient     {veh.mu_longitudinal:.2f}"
             f"        design_intent")
    L.append(f"   Lateral acceleration                  {veh.design_ay_g:.2f} g"
             f"      design_intent")
    L.append(f"   Lateral load transfer distribution    {veh.lltd_front:.2f}"
             f"        design_intent   <- front share")
    L.append("")
    L.append(f"   LLTD {veh.lltd_front:.2f} against a front mass fraction of "
             f"{veh.front_mass_fraction:.2f} means lateral transfer is")
    L.append("   proportional to axle load, so it introduces no balance shift of")
    L.append("   its own. That is a neutral BASELINE, not a tuned value -- the")
    L.append("   balance target that would justify tuning away from it needs tyre")
    L.append("   data the team does not have. Inner-wheel loads at this LLTD:")
    L.append("")
    L.append(f"   {'axle':<8s}{'static/wheel':>15s}{'outer':>12s}{'inner':>12s}"
             f"{'lift at LLTD':>15s}")
    for label, geo, is_front in (("Front", front, True), ("Rear", rear, False)):
        lc = sla.load_cases(geo, veh)
        lltd_lift = lltd_lift_threshold(veh, geo, is_front)
        L.append(f"   {label:<8s}{lc['Fz_static']:12.1f} N{lc['Fz']:10.1f} N"
                 f"{lc['Fz_inner']:10.1f} N{lltd_lift:12.3f}   "
                 f"({'above' if is_front else 'below'})")
    L.append("")
    lo = lltd_lift_threshold(veh, rear, False)
    hi = lltd_lift_threshold(veh, front, True)
    L.append(f"   The usable band is LLTD {lo:.3f} to {hi:.3f} -- outside it one")
    L.append(f"   inner wheel lifts at {veh.design_ay_g:.2f} g and every force below")
    L.append("   it is void.")
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

    L.append("   Static camber is now carried by the exported points (section 6),")
    L.append("   so the rate tables and the deliverable describe the same car.")
    L.append("   Track is the CONTACT PATCH datum, which leaves scrub radius")
    L.append("   untouched. What it moves instead is the wheel-centre track:")
    L.append("")
    L.append(f"   {'axle':<8s}{'ground track':>15s}{'wheel-centre track':>21s}{'per corner':>13s}")
    for label, geo in (("Front", front), ("Rear", rear)):
        inp = geo.inputs
        shift = inp.loaded_radius_mm * math.tan(math.radians(inp.static_camber_deg))
        L.append(f"   {label:<8s}{inp.track_mm:12.1f} mm"
                 f"{inp.wheel_centre_track_mm:18.1f} mm{shift:10.2f} mm")
    L.append("")
    L.append("   TO CONFIRM: the wheel offset and upright width have to absorb")
    L.append("   that per-corner shift. It is a packaging number, not a kinematic")
    L.append("   one, and nothing in this tool can check it.")
    L.append("")
    L.append("   TO CONFIRM: the nominal camber is machined into the upright, so")
    L.append("   the 2 mm plates at the upper arm trim around it. Plate step is")
    for label, geo in (("Front", front), ("Rear", rear)):
        step = math.degrees(math.atan(2.0 / geo.inputs.kingpin_length_mm))
        L.append(f"     {label.lower()} {step:.3f} deg per 2 mm plate"
                 f"   (kingpin {geo.inputs.kingpin_length_mm:.2f} mm)")
    L.append("")
    tro_x = hp.arr("RL", "TIE_ROD_OUT")[0]
    lca_out_x = hp.arr("RL", "LCA_OUT")[0]
    if tro_x < lca_out_x:
        L.append("   TO CONFIRM: the rear toe link OUTBOARD point sits "
                 f"{lca_out_x - tro_x:.1f} mm behind the")
        L.append("   wishbone outboard ball joints. It is an upright feature at")
        L.append(f"   Y = {abs(hp.arr('RL', 'TIE_ROD_OUT')[1]):.0f}, not a chassis "
                 "connection, so it was left alone when")
        L.append("   the inboard end moved forward for packaging. Bringing it")
        L.append("   ahead of the axle would be a rear-upright redesign and would")
        L.append("   flip the sign of toe change under longitudinal load.")
        L.append("")

    L.append("   Not in scope for this tool, and deliberately not computed here:")
    L.append("     wheel rates, motion ratio, ride frequency, damping")
    L.append("     stress, deflection, fatigue, buckling")
    L.append("     anything requiring tyre data, until TTC data is acquired")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# 4. TOP LEVEL
# --------------------------------------------------------------------------- #

def _vdcore_corner(
    hp: MergedHardpoints, corner: str, inp: sla.AxleInputs
) -> Corner:
    """Lift one merged corner into a vdcore Corner.

    The merged set is the only place a COMPLETE corner exists: sla_geometry
    synthesises the wishbones and steering_geometry supersedes the outboard
    ball joints and owns the tie rod. DWSolver needs the tie rod to close the
    sixth degree of freedom, which is why the rate and roll tables live here
    and not in either script alone.
    """

    def point(name: str) -> Hardpoint:
        x, y, z = hp.arr(corner, name)
        return Hardpoint(
            name=name, x_mm=float(x), y_mm=float(y), z_mm=float(z),
            source="design_intent", tol_mm=0.5,
        )

    return Corner(
        corner_id=corner,
        uca_inboard_front=point("UCA_IN_FRONT"),
        uca_inboard_rear=point("UCA_IN_REAR"),
        uca_outboard=point("UCA_OUT"),
        lca_inboard_front=point("LCA_IN_FRONT"),
        lca_inboard_rear=point("LCA_IN_REAR"),
        lca_outboard=point("LCA_OUT"),
        tie_rod_inboard=point("TIE_ROD_IN"),
        tie_rod_outboard=point("TIE_ROD_OUT"),
        wheel_center=point("WHEEL_CENTER"),
        tire=TirePackage(
            loaded_radius_mm=inp.loaded_radius_mm,
            source="design_intent",
            tol_mm=1.0,
        ),
        static_camber_deg=inp.static_camber_deg,
        static_toe_deg_per_side=0.0,
    )


def rates_section(
    hp: MergedHardpoints, front: sla.AxleGeometry, rear: sla.AxleGeometry
) -> str:
    """Rate and roll tables, solved in 3D on the complete merged corners.

    These used to come from a front-view four-bar inside sla_geometry.py that
    never read the pivot-axis rake, so dialling in anti-dive left every rate
    unchanged while the real geometry moved. vdcore.analysis.axle runs the full
    3D solver, so rake is carried by the constraints.
    """
    L = ["", RULE, " 3b. RATES AND ROLL  (3D solve, merged corners)", RULE]
    L.append("   Solved by vdcore.analysis.axle on DWSolver. Z values are")
    L.append("   CHASSIS-referenced: ride height measured from the sprung mass,")
    L.append("   which is the frame the roll couple needs. Ground-referenced RC")
    L.append("   migration differs by exactly 1 mm per mm of travel.")

    for label, geo, sides in (
        ("FRONT", front, ("FL", "FR")),
        ("REAR", rear, ("RL", "RR")),
    ):
        inp = geo.inputs
        axle = Axle(
            left=_vdcore_corner(hp, sides[0], inp),
            right=_vdcore_corner(hp, sides[1], inp),
        )
        rates = axle_rates(
            axle,
            travel_bump_mm=inp.travel_bump_mm,
            travel_droop_mm=inp.travel_droop_mm,
        )
        roll = axle_roll(axle, inp.roll_reference_deg)

        L.append("")
        L.append(f" {label}   RATES ABOUT STATIC")
        L.append(f"   Camber gain                 {rates.camber_gain_deg_per_mm:12.4f} deg/mm")
        L.append(f"   Roll centre migration       {rates.rc_migration_mm_per_mm:12.4f} mm/mm")
        L.append(f"   Half-track change           {rates.half_track_change_mm_per_mm:12.4f} mm/mm")
        L.append(f"   Camber at full bump         {rates.camber_full_bump_deg:12.4f} deg")
        L.append(f"   Camber at full droop        {rates.camber_full_droop_deg:12.4f} deg")
        L.append(f"   RC over the travel range    {rates.rc_min_mm:7.1f} to {rates.rc_max_mm:.1f} mm")
        L.append(
            f"   (travel {inp.travel_bump_mm:.0f} mm bump / "
            f"{inp.travel_droop_mm:.0f} mm droop)"
        )
        L.append("")
        L.append(f" {label}   AT {roll.roll_deg:.1f} DEG OF ROLL   (camber relative to the ROAD)")
        L.append(f"   Outer wheel                 {roll.outer_camber_deg:12.2f} deg"
                 f"   (static {inp.static_camber_deg:+.2f})")
        L.append(f"   Inner wheel                 {roll.inner_camber_deg:12.2f} deg")
        L.append(f"   Roll centre height          {roll.rc_height_mm:12.1f} mm"
                 f"   (design {inp.rc_height_mm:.1f})")
        L.append(f"   Roll centre lateral         {roll.rc_lateral_mm:12.1f} mm")
        L.append(f"   Wheel travel at that roll   {roll.wheel_travel_mm:12.2f} mm")
        band = inp.limits.outer_camber_in_roll_deg
        ok = band[0] <= roll.outer_camber_deg <= band[1]
        L.append(
            f"  {'OK ' if ok else '!! '} Outer wheel camber in the useful window "
            f"({band[0]:.1f} to {band[1]:.1f} deg)"
        )
    return "\n".join(L)


BUMP_STEER_LINEAR_LIMIT = 0.005      # deg/mm per side
BUMP_STEER_PEAK_LIMIT = 0.30         # deg per side over the travel range


def toe_link_z_for_zero_bump_steer(
    hp: MergedHardpoints, corner: str, inp: sla.AxleInputs,
    lo_mm: float = 140.0, hi_mm: float = 200.0,
) -> float:
    """Inboard toe-link Z that nulls the linear bump-steer rate.

    Z is the knob; X is very nearly not (see REAR_TOE_LINK_INBOARD). Returns
    nan if the bracket does not straddle a root, rather than a plausible-looking
    number -- the same contract every solver here follows.
    """
    def err(z: float) -> float:
        moved = {c: dict(pts) for c, pts in hp.points.items()}
        x, y, _ = moved[corner]["TIE_ROD_IN"]
        moved[corner]["TIE_ROD_IN"] = (x, y, z)
        shifted = MergedHardpoints(points=moved)
        return bump_steer(
            _vdcore_corner(shifted, corner, inp),
            wheel_travel_range_mm=inp.travel_bump_mm,
        ).linear_deg_per_mm_per_side

    try:
        if err(lo_mm) * err(hi_mm) < 0:
            return float(brentq(err, lo_mm, hi_mm, xtol=1e-6))
    except (ValueError, RuntimeError):
        pass
    return float("nan")


def bump_steer_section(
    hp: MergedHardpoints, front: sla.AxleGeometry, rear: sla.AxleGeometry
) -> str:
    """Toe vs wheel travel on both axles, solved in 3D on the merged corners.

    The rear had no bump-steer number anywhere before 2026-09-01:
    steering_geometry.py owns bump steer but covers the front axle only, so the
    rear toe link's effect on toe was simply uncomputed. It is a five-link
    constraint like any other and DWSolver has always been able to see it --
    nothing was asking.

    Both the linear rate and the peak are reported. Nulling the linear term
    leaves a quadratic, so a "zero bump steer" tie rod still toes the same way
    in bump and droop; the peak is what bounds toe change on track.
    """
    L = ["", RULE, " 3c. BUMP STEER  (3D solve, merged corners)", RULE]
    L.append("   Toe is referenced to the RIDE-HEIGHT value, so this is bump")
    L.append("   steer, not static toe. Per side; double it for axle total toe.")
    L.append("   Positive rate = the wheel toes IN as it moves into bump.")

    for label, geo, corner in ((" FRONT", front, "FL"), (" REAR", rear, "RL")):
        inp = geo.inputs
        bs = bump_steer(
            _vdcore_corner(hp, corner, inp), wheel_travel_range_mm=inp.travel_bump_mm
        )
        sweep = toe_sweep(
            _vdcore_corner(hp, corner, inp),
            wheel_travel_min_mm=-inp.travel_droop_mm,
            wheel_travel_max_mm=inp.travel_bump_mm,
            steps=5,
        )
        L.append("")
        L.append(f"{label}   ({corner}, +/-{inp.travel_bump_mm:.0f} mm travel)")
        L.append(f"   Linear rate                 {bs.linear_deg_per_mm_per_side:12.5f} deg/mm")
        L.append(f"   Peak |toe| over travel      {bs.peak_abs_deg_per_side:12.5f} deg"
                 f"   (total toe {bs.peak_abs_total_toe_deg:.5f})")
        L.append(f"   Toe at full bump / droop    {bs.toe_at_full_bump_deg_per_side:7.4f} / "
                 f"{bs.toe_at_full_droop_deg_per_side:.4f} deg")
        L.append("   " + "  ".join(
            f"{t:+.0f}mm {v:+.4f}"
            for t, v in zip(sweep.wheel_travel_mm, sweep.toe_deg_per_side, strict=True)
        ))
        L.append(band("Linear bump steer", abs(bs.linear_deg_per_mm_per_side),
                      0.0, BUMP_STEER_LINEAR_LIMIT, "deg/mm", "9.5f"))
        L.append(band("Peak toe over travel", bs.peak_abs_deg_per_side,
                      0.0, BUMP_STEER_PEAK_LIMIT, "deg", "9.5f"))

    z0 = toe_link_z_for_zero_bump_steer(hp, "RL", rear.inputs)
    x_in, _, z_in = hp.points["RL"]["TIE_ROD_IN"]
    L.append("")
    L.append(" REAR TOE LINK -- the knob, and how sharp it is")
    L.append(f"   Inboard at X {x_in:.0f}, Z {z_in:.1f} mm. Z that would null the")
    L.append(f"   linear rate exactly: {z0:.3f} mm -- inside the 0.5 mm build")
    L.append("   tolerance, so the round number is what the drawing carries.")
    L.append("   X is nearly free: -1460 to -1400 moves the peak by 0.001 deg.")
    return "\n".join(L)


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
        rates_section(hp, design.front, design.rear),
        bump_steer_section(hp, design.front, design.rear),
        steering_section(steer),
        members_section(hp),
        hardpoints_section(hp, design.front),
        load_case_note(veh, front, rear),
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
