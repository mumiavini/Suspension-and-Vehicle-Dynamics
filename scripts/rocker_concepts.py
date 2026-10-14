"""Rocker / push-pull rod layout concepts, checked against the chassis CAD.

Each concept is a set of construction inputs per axle (rod type, the member the
rod is mounted on, the rod's outer end, the rocker pivot, the direction the
coil-over lies in, the rod arm and the motion ratio to tune to). From them:

1. the rocker is built with both forces tangential at ride height (optionally
   clocked off tangent) and the damper arm tuned (bisection) to the asked
   motion ratio, the coil-over's ride length set for equal droop/bump margins
   -- the construction of ``engine.suspension.build_rocker`` / ``tune_rocker``,
   extended to a pullrod (the damper arm goes on the side that compresses the
   damper in bump);
2. the linkage is solved on ``DWSolver`` by ``vdcore.analysis.rocker`` through
   ``engine.suspension`` over the axle's travel (MR, damper usage, rates, roll
   budget), and again out to the damper's physical end stops (extended and
   compressed), where the packaging, transmission angles and rocker rotation
   are measured;
3. packaging is measured as AIR GAPS (centreline distance minus both estimated
   radii) against the chassis CAD survey "Xasis 33.3" as built -- both
   corners, because the frame is not symmetric at the engine mounts: the rod,
   the rocker (plates and pickup bolts), the coil-over body and its eyes
   against every frame member; the rod against the wishbone legs and the tie
   rod / toe link over travel x full rack travel; the rod against the rim
   (13", steered); left against right in heave and in roll; the linkage
   against the steering rack and column (front), a drive shaft and an engine
   envelope (rear), and the solid bodies of the CAD that have no centreline;
4. loads at ride: rod force and whether it is tension or compression, the
   rocker pivot reaction and the bending couples it puts into its bracket.

What it is: consequences of layouts the designer proposes. What it is not: a
choice. The sizes, the engine envelope, the drive shaft, the steering column,
the moving masses and the rod tube are ESTIMATES (``ESTIMATES``) -- the CAD has
no engine, differential, dampers or rockers. The motion ratio target is a
query argument (the same for every concept so they compare like for like).

CAD frame model: the survey's 3D-sketch centrelines, each flagged "bodied" when
a weldment body of matching shape exists (a sketch line with no body may not
be a real tube); weldment bodies with no sketch line are added as centrelines
when they are slim and axis-aligned, and kept as bounding boxes otherwise.
Bracket lengths are measured to bodied members only.

Usage::

    uv run python scripts/rocker_concepts.py eval CONCEPT.json      # metrics as JSON
    uv run python scripts/rocker_concepts.py seed                   # the project's layout
    uv run python scripts/rocker_concepts.py frame                  # the CAD frame model
    uv run python scripts/rocker_concepts.py build                  # variants, report, views

Frame: ISO 8855 (X forward, Y LEFT, Z up), chassis-fixed, mm; specs are the
LEFT corner, the right corner's linkage is the Y-mirror.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402

from apps.vdstudio.engine import chassis as C  # noqa: E402
from apps.vdstudio.engine import exports as X  # noqa: E402
from apps.vdstudio.engine import geometry as G  # noqa: E402
from apps.vdstudio.engine import kpis as K  # noqa: E402
from apps.vdstudio.engine import suspension as S  # noqa: E402
from apps.vdstudio.engine import tilt as T  # noqa: E402
from apps.vdstudio.engine.project import (  # noqa: E402
    Project,
    RockerSpec,
    load_project,
    save_project,
)
from vdcore.analysis.rocker import (  # noqa: E402
    RockerSolveError,
    _Rocker,
    body_transform,
    rocker_sweep,
)
from vdcore.geometry.solver import DWSolver  # noqa: E402

PROJECT = REPO / "projects" / "FSAE_2027.vdproj.json"
SURVEY = REPO / "projects" / "cad" / "Xasis 33.3_survey.json"
OUT_DIR = REPO / "projects" / "rocker_concepts"

# --------------------------------------------------------------------------- #
# Estimates (no CAD for these parts). Every number here is source="estimate".
# --------------------------------------------------------------------------- #
ESTIMATES = {
    "frame_tube_od_mm": 25.4,          # most of the frame (SIZE_B/C 25.4 mm)
    "rod_od_mm": 16.0,                 # push/pull rod tube
    "rod_wall_mm": 1.0,
    "rod_E_N_mm2": 205e3,              # steel
    "coilover_od_mm": 60.0,            # damper body with the spring over it
    "coilover_body_from_eye_mm": 30.0,  # the full OD starts this far from each eye
    "rocker_half_mm": 10.0,            # rocker plate / pickup hardware half-envelope
    "pickup_bolt_half_mm": 20.0,       # pickup bolt and clevis, each side of the plate
    "wishbone_od_mm": 20.0,
    "tie_rod_od_mm": 16.0,
    "rack_housing_od_mm": 40.0,
    "column_od_mm": 40.0,              # steering column with its U-joints
    # column from the pinion (rack centre) up and back to the wheel behind the front hoop
    "column_mm": ((-30.0, 0.0, 157.0), (-300.0, 0.0, 560.0)),
    "driveshaft_od_mm": 60.0,          # CV joint / tripod housing envelope
    "diff_output_y_mm": 120.0,         # drive shaft inboard joint, |y|, on the rear axle
    # engine envelope from the CAD engine-mount tabs, which are centred about y = -27 (the
    # frame is not symmetric there): the front tabs (3DSketch8, x ~ -1070) end at y +164 /
    # -218, the rear ones (3DSketch6/7, x ~ -1340..-1385) at y +53 / -107, so the case is
    # narrower at the back. Two boxes: (x range, y centre, half width, z range)
    "engine_boxes_mm": (((-1250.0, -1080.0), -27.0, 190.0, (115.0, 490.0)),
                        ((-1400.0, -1250.0), -27.0, 130.0, (115.0, 490.0))),
    "coilover_kg": 1.3,                # moving masses per corner (CG effect only)
    "rocker_kg": 0.3,
}

# Air-gap thresholds (mm) and limits. Information for the flags, not requirements.
# (key, axle or None, "min"/"max", limit, label)
THRESHOLDS = (
    ("gap_rod_frame", None, "min", 5.0, "rod vs frame"),
    ("gap_rocker_frame", None, "min", 5.0, "rocker (plates, pickup bolts) vs frame"),
    ("gap_coilover_frame", None, "min", 5.0, "coil-over body vs frame"),
    ("gap_eyes_frame", None, "min", 5.0, "coil-over eyes vs frame"),
    ("gap_rod_members", None, "min", 5.0, "rod vs wishbones / tie rod / toe link (with steer)"),
    ("gap_rod_rim", None, "min", 15.0, "rod vs rim (13\", steered)"),
    ("gap_left_right", None, "min", 5.0, "left vs right linkage (heave and roll)"),
    ("gap_rack", "front", "min", 5.0, "linkage vs steering rack"),
    ("gap_column", "front", "min", 5.0, "linkage vs steering column (estimate)"),
    ("gap_driveshaft", "rear", "min", 10.0, "linkage vs drive shaft (estimate)"),
    ("gap_engine", "rear", "min", 10.0, "linkage vs engine envelope (estimate)"),
    ("pivot_bracket_mm", None, "max", 60.0, "rocker pivot to nearest bodied tube"),
    ("damper_bracket_mm", None, "max", 60.0, "coil-over chassis end to nearest bodied tube"),
    ("rod_angle_max_deg", None, "max", 30.0, "rod transmission angle (to the end stops)"),
    ("damper_angle_max_deg", None, "max", 30.0,
     "coil-over transmission angle (to the end stops)"),
    ("psi_abs_max_deg", None, "max", 35.0, "rocker rotation (to the end stops)"),
)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _t4(v) -> tuple[float, float, float]:
    return tuple(round(float(x), 4) for x in v)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Construction (a designer's helper, not a pick)
# --------------------------------------------------------------------------- #

def build(mount, pivot, damper_toward, rod_arm_mm: float, damper_arm_mm: float,
          damper_ride_mm: float, mount_body: str, rod_type: str, flip: bool,
          damper_clock_deg: float = 0.0) -> RockerSpec:
    """``engine.suspension.build_rocker``, with two options.

    ``flip`` puts the damper pickup on the opposite side of the pivot: for a pullrod
    it is the side that compresses the damper in bump. ``damper_clock_deg`` turns the
    damper arm about the rocker axis away from tangent (the coil-over keeps its
    direction); off-tangent arms change the MR progression.
    """
    sp = S.build_rocker(mount, pivot, damper_toward, rod_arm_mm, damper_arm_mm,
                        damper_ride_mm, mount_body, rod_type)
    if not flip and damper_clock_deg == 0.0:
        return sp
    o = np.asarray(sp.rocker_pivot, float)
    a = _unit(sp.rocker_axis)
    ud = _unit(np.asarray(damper_toward, float) - o)
    arm = np.asarray(sp.rocker_damper, float) - o
    if flip:
        arm = -arm
    arm = _rot(a, math.radians(damper_clock_deg)) @ arm
    rd = o + arm
    return replace(sp, rocker_damper=_t4(rd), damper_chassis=_t4(rd + damper_ride_mm * ud))


@dataclass(frozen=True)
class AxleConcept:
    rod_type: str                      # "pushrod" | "pullrod"
    mount_body: str                    # "lca" | "uca" | "upright"
    rod_mount: tuple[float, float, float]
    pivot: tuple[float, float, float]
    damper_toward: tuple[float, float, float]
    rod_arm_mm: float
    target_mr: float = 0.80
    damper_clock_deg: float = 0.0


def make_spec(geo, project: Project, axle: str, c: AxleConcept) -> RockerSpec:
    """Damper arm for the target MR (bisection on 10-200 mm), then equal damper margins."""
    cid = S.AXLE_CORNER[axle]
    corner = geo.corner(cid)
    droop, bump = S.travel_range(geo, axle)

    def mr(arm: float, flip: bool) -> float:
        sp = build(c.rod_mount, c.pivot, c.damper_toward, c.rod_arm_mm, arm, 200.0,
                   c.mount_body, c.rod_type, flip, c.damper_clock_deg)
        return float(rocker_sweep(corner, S.layout_for(sp, cid), [0.0]).motion_ratio[0])

    flip = mr(60.0, False) < 0.0          # the side that compresses the damper in bump
    lo, hi = 10.0, 200.0
    flo, fhi = mr(lo, flip) - c.target_mr, mr(hi, flip) - c.target_mr
    if not flo * fhi < 0:
        raise ValueError(f"{axle}: MR {c.target_mr:g} not reachable with damper arms "
                         f"10-200 mm (MR {flo + c.target_mr:.3f} .. {fhi + c.target_mr:.3f}); "
                         "change the rod arm")
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = mr(mid, flip) - c.target_mr
        if flo * fm <= 0:
            hi = mid
        else:
            lo, flo = mid, fm
    arm = 0.5 * (lo + hi)
    d = project.damper
    compressed = d.eye_to_eye_extended_mm - d.stroke_mm + d.bump_rubber_mm
    ride = 0.5 * (d.eye_to_eye_extended_mm + compressed)
    sp = None
    for _ in range(8):
        sp = build(c.rod_mount, c.pivot, c.damper_toward, c.rod_arm_mm, arm, ride,
                   c.mount_body, c.rod_type, flip, c.damper_clock_deg)
        sw = rocker_sweep(corner, S.layout_for(sp, cid), [droop, bump])
        m_droop = d.eye_to_eye_extended_mm - float(sw.damper_length_mm.max())
        m_bump = float(sw.damper_length_mm.min()) - compressed
        ride += 0.5 * (m_droop - m_bump)
    assert sp is not None
    return replace(sp, source="design_intent")


# --------------------------------------------------------------------------- #
# The CAD frame model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Member:
    name: str
    a: np.ndarray
    b: np.ndarray
    bodied: bool          # a weldment body of this shape exists in the CAD


@dataclass(frozen=True)
class Box:
    name: str
    lo: np.ndarray
    hi: np.ndarray


def frame_model(project: Project) -> tuple[list[Member], list[Box]]:
    sv = C.load_survey(SURVEY)
    fr = X.cad_frame(project)
    raw = json.loads(SURVEY.read_text(encoding="utf-8"))
    bodies = []
    for b in raw.get("bodies", []):
        bx = b["box_mm"]
        p, q = C.cad_to_iso(bx[:3], fr), C.cad_to_iso(bx[3:], fr)
        bodies.append((str(b["name"]), np.minimum(p, q), np.maximum(p, q)))
    r = ESTIMATES["frame_tube_od_mm"] / 2.0
    members, matched = [], set()
    for ln in sv.lines:
        a, b = C.cad_to_iso(ln.a, fr), C.cad_to_iso(ln.b, fr)
        lb, hb = np.minimum(a, b) - r, np.maximum(a, b) + r
        vol = float(np.prod(np.maximum(hb - lb, 1.0)))
        tight = [i for i, (_n, lo, hi) in enumerate(bodies)
                 if np.all(lo >= lb - 30.0) and np.all(hi <= hb + 30.0)
                 and float(np.prod(np.maximum(hi - lo, 1.0))) > 0.2 * vol]
        matched.update(tight)
        members.append(Member(ln.sketch, a, b, bool(tight)))
    # a body piece lying inside a bodied member's own envelope is part of it (a member
    # split at a crossing, an end trim)
    for i, (_n, lo, hi) in enumerate(bodies):
        if i in matched:
            continue
        for m in members:
            lb, hb = np.minimum(m.a, m.b) - r - 30.0, np.maximum(m.a, m.b) + r + 30.0
            if m.bodied and np.all(lo >= lb) and np.all(hi <= hb) and \
                    pt_seg(0.5 * (lo + hi), m.a, m.b) <= r + 15.0:
                matched.add(i)
                break
    boxes = []
    for i, (name, lo, hi) in enumerate(bodies):
        if i in matched:
            continue
        ext = hi - lo
        small = ext <= 2 * r + 15.0
        if int(small.sum()) >= 2:           # slim and axis-aligned: a tube centreline
            c = 0.5 * (lo + hi)
            k = int(np.argmax(ext))
            a, b = c.copy(), c.copy()
            a[k], b[k] = lo[k] + r, hi[k] - r
            if b[k] > a[k]:
                members.append(Member(f"body {name}", a, b, True))
                continue
        boxes.append(Box(name, lo, hi))
    return members, boxes


# --------------------------------------------------------------------------- #
# Context and geometry helpers
# --------------------------------------------------------------------------- #

@dataclass
class Ctx:
    project: Project
    geo: object
    rc: dict[str, float]
    members: list[Member]
    boxes: list[Box]

    @property
    def tubes(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        return [(m.name, m.a, m.b) for m in self.members]


def load_ctx() -> Ctx:
    p = load_project(PROJECT)
    g = G.resolve(p)
    st = K.compute_static(g)
    rc = {"front": st.value("front.rc_static"), "rear": st.value("rear.rc_static")}
    members, boxes = frame_model(p)
    return Ctx(p, g, rc, members, boxes)


def seg_seg(p1, q1, p2, q2) -> float:
    return S._seg_dist(np.asarray(p1, float), np.asarray(q1, float),
                       np.asarray(p2, float), np.asarray(q2, float))


def pt_seg(p, a, b) -> float:
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0)) if ab @ ab > 0 else 0.0
    return float(np.linalg.norm(a + t * ab - p))


def trim(a, b, at_a: float, at_b: float) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = _unit(b - a)
    L = float(np.linalg.norm(b - a))
    at_a, at_b = min(at_a, 0.45 * L), min(at_b, 0.45 * L)
    return a + at_a * u, b - at_b * u


def seg_aabb(a, b, lo, hi) -> float:
    """Distance from a segment to an axis-aligned box (sampled; 0 = touching or inside)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    best = math.inf
    for s in np.linspace(0.0, 1.0, 61):
        p = a + s * (b - a)
        best = min(best, float(np.linalg.norm(p - np.clip(p, lo, hi))))
    return best


def seg_box(a, b, boxes) -> float:
    """Distance to the nearest ``engine_boxes_mm`` box (x range, y centre, half width, z)."""
    return min(seg_aabb(a, b, np.array([x0, yc - hw, z0]), np.array([x1, yc + hw, z1]))
               for (x0, x1), yc, hw, (z0, z1) in boxes)


def mirror(p) -> np.ndarray:
    p = np.asarray(p, float)
    return np.array([p[0], -p[1], p[2]])


def _foot(p, a, b) -> tuple[float, float]:
    """(distance, distance of the foot point to the member's nearer end)."""
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    L = float(np.linalg.norm(ab))
    t = float(np.clip((p - a) @ ab / (L * L), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - p)), min(t, 1.0 - t) * L


def bracket(ctx: Ctx, p) -> dict:
    """Nearest bodied member (the bracket's tube) and, if closer, a sketch-only line."""
    best = min(((*_foot(p, m.a, m.b), m) for m in ctx.members if m.bodied),
               key=lambda x: x[0])
    sk = min(((*_foot(p, m.a, m.b), m) for m in ctx.members if not m.bodied),
             key=lambda x: x[0], default=None)
    out = {"mm": best[0], "from_tube_end_mm": best[1],
           "tube": f"{best[2].name} {_pt(best[2].a, '.0f')}->{_pt(best[2].b, '.0f')}"}
    if sk is not None and sk[0] < best[0]:
        out["closer_sketch_only"] = f"{sk[2].name} at {sk[0]:.1f} mm (no body in the CAD)"
    return out


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class Linkage:
    """One corner's linkage at one wheel travel, chassis frame (both sides possible)."""

    z: float
    mount: np.ndarray
    rr: np.ndarray
    rd: np.ndarray
    pivot: np.ndarray
    dc: np.ndarray
    axis: np.ndarray

    def mirrored(self) -> Linkage:
        a = self.axis
        return Linkage(self.z, mirror(self.mount), mirror(self.rr), mirror(self.rd),
                       mirror(self.pivot), mirror(self.dc), np.array([-a[0], a[1], -a[2]]))

    def elements(self) -> list[tuple[str, np.ndarray, np.ndarray, float]]:
        """(name, a, b, radius): the parts that take space, with estimated radii."""
        E = ESTIMATES
        rr_ = E["rocker_half_mm"]
        bolt = E["pickup_bolt_half_mm"] * self.axis
        c0, c1 = trim(self.rd, self.dc, E["coilover_body_from_eye_mm"],
                      E["coilover_body_from_eye_mm"])
        return [("rod", self.mount, self.rr, E["rod_od_mm"] / 2.0),
                ("rocker", self.pivot, self.rr, rr_), ("rocker", self.pivot, self.rd, rr_),
                ("rocker", self.rr, self.rd, rr_),
                ("rocker", self.rr - bolt, self.rr + bolt, rr_),
                ("rocker", self.rd - bolt, self.rd + bolt, rr_),
                ("coil-over", c0, c1, E["coilover_od_mm"] / 2.0),
                ("eyes", self.rd, self.rd, 0.0), ("eyes", self.dc, self.dc, 0.0)]


def _eye_r(ctx: Ctx) -> float:
    return ctx.project.damper.bearing_od_mm / 2.0


def end_stops(corner, lay, project: Project) -> tuple[float, float]:
    """Wheel travel at the damper's full extension (droop) and full compression (bump)."""
    d = project.damper
    ext, comp = d.eye_to_eye_extended_mm, d.eye_to_eye_extended_mm - d.stroke_mm
    zs = np.arange(-60.0, 60.01, 2.0)
    try:
        sw = rocker_sweep(corner, lay, zs)
    except RockerSolveError:
        zs = np.arange(-40.0, 40.01, 2.0)
        sw = rocker_sweep(corner, lay, zs)
    L = sw.damper_length_mm
    z_ext = float(np.interp(-ext, -L, zs)) if L.max() >= ext else float(zs[0])
    z_comp = float(np.interp(comp, L[::-1], zs[::-1])) if L.min() <= comp else float(zs[-1])
    return z_ext, z_comp


def evaluate_axle(ctx: Ctx, axle: str, a: S.AxleSuspension) -> dict:
    """Packaging, end stops and loads on top of the engine's axle report."""
    E = ESTIMATES
    spec, sw0 = a.spec, a.sweep
    if not a.ok or sw0 is None:
        return {"error": a.error}
    cid = S.AXLE_CORNER[axle]
    corner = ctx.geo.corner(cid)
    lay = S.layout_for(spec, cid)
    solver = DWSolver(corner)
    rk = _Rocker(lay)
    axis = _unit(lay.rocker_axis)
    pivot, dc = np.asarray(lay.rocker_pivot, float), np.asarray(lay.damper_chassis, float)
    eye_r, tube_r = _eye_r(ctx), E["frame_tube_od_mm"] / 2.0

    z_ext, z_comp = end_stops(corner, lay, ctx.project)
    zs = np.linspace(z_ext, z_comp, 15)
    sw = rocker_sweep(corner, lay, zs)
    left = [Linkage(float(z), st.rod_mount, st.rocker_rod, st.rocker_damper, pivot, dc, axis)
            for z, st in zip(zs, sw.states, strict=True)]
    right = [s.mirrored() for s in left]
    out: dict = {"end_stops_mm": (z_ext, z_comp)}

    # ---- frame (both corners, the CAD as built) and solid bodies without centreline
    worst: dict[str, tuple[float, str]] = {}

    def keep(key: str, gap: float, where: str) -> None:
        if gap < worst.get(key, (math.inf, ""))[0]:
            worst[key] = (gap, where)

    for side, states in (("L", left), ("R", right)):
        for st in states:
            for name, p, q, rad in st.elements():
                key = {"rod": "gap_rod_frame", "rocker": "gap_rocker_frame",
                       "coil-over": "gap_coilover_frame", "eyes": "gap_eyes_frame"}[name]
                rad = eye_r if name == "eyes" else rad
                for m in ctx.members:
                    g = seg_seg(p, q, m.a, m.b) - rad - tube_r
                    if g < worst.get(key, (math.inf, ""))[0]:
                        keep(key, g, f"{m.name} {_pt(m.a, '.0f')}->{_pt(m.b, '.0f')}"
                                     f"{'' if m.bodied else ' (sketch only)'}, {side} at "
                                     f"{st.z:+.0f} mm")
                for bx in ctx.boxes:
                    g = seg_aabb(p, q, bx.lo, bx.hi) - rad
                    keep("gap_solid_bodies", g, f"{name} vs '{bx.name}' box, {side} at "
                                                f"{st.z:+.0f} mm")
                if axle == "rear":
                    keep("gap_engine", seg_box(p, q, E["engine_boxes_mm"]) - rad,
                         f"{name}, {side} at {st.z:+.0f} mm")
    # ---- left vs right, heave and roll pairs
    for sl in left:
        for sr in right:
            for n1, p1, q1, r1 in sl.elements():
                for n2, p2, q2, r2 in sr.elements():
                    if "eyes" in (n1, n2):
                        continue
                    keep("gap_left_right", seg_seg(p1, q1, p2, q2) - r1 - r2,
                         f"{n1} L {sl.z + 0.0:+.0f} / {n2} R {sr.z + 0.0:+.0f} mm")
    # ---- members, rim (steered), rack, column, drive shaft
    rack_max = float(ctx.project.steering.max_rack_travel_mm) if axle == "front" else 0.0
    racks = np.linspace(-rack_max, rack_max, 5) if rack_max > 0 else np.array([0.0])
    r_rim = ctx.geo.axle_inputs(axle).rim_diameter_in * 25.4 / 2.0
    half_w = 0.5 * float(ctx.project.turning_envelope.tyre_section_width_mm)
    legs = [("UCA front leg", corner.uca_inboard_front, "ubj"),
            ("UCA rear leg", corner.uca_inboard_rear, "ubj"),
            ("LCA front leg", corner.lca_inboard_front, "lbj"),
            ("LCA rear leg", corner.lca_inboard_rear, "lbj")]
    rod_r = E["rod_od_mm"] / 2.0
    for z, st0 in zip(zs[::2], sw.states[::2], strict=True):
        guess = math.radians(st0.psi_deg)
        for rack in racks:
            res = solver.solve(wheel_travel_mm=float(z), rack_mm=float(rack))
            if not res.converged:
                keep("gap_rod_members", -math.inf, f"no solve at {z:+.0f} mm, rack {rack:+.0f}")
                continue
            rot, t = body_transform(corner, res, float(z), lay.mount_body)
            mount = rot @ np.asarray(lay.rod_mount, float) + t
            try:
                psi = rk.solve_psi(mount, guess)
            except RockerSolveError as exc:
                keep("gap_rod_members", -math.inf, f"{exc} (rack {rack:+.0f})")
                continue
            rr = rk.point(rk.r, psi)
            lift = np.array([0.0, 0.0, float(z)])
            r0, r1 = trim(mount, rr, 30.0, 0.0)          # the mount's own bracket excluded
            for name, p_in, end in legs:
                pin = np.array([p_in.x_mm, p_in.y_mm, p_in.z_mm])
                e = getattr(res, end)
                pout = np.array([e.x_mm, e.y_mm, e.z_mm]) + lift
                keep("gap_rod_members", seg_seg(r0, r1, pin, pout) - rod_r
                     - E["wishbone_od_mm"] / 2.0, f"{name} at {z:+.0f} mm, rack {rack:+.0f}")
            tin = np.array([corner.tie_rod_inboard.x_mm, corner.tie_rod_inboard.y_mm + rack,
                            corner.tie_rod_inboard.z_mm])
            tout = np.array([res.tro.x_mm, res.tro.y_mm, res.tro.z_mm]) + lift
            keep("gap_rod_members", seg_seg(r0, r1, tin, tout) - rod_r
                 - E["tie_rod_od_mm"] / 2.0,
                 f"{'tie rod' if axle == 'front' else 'toe link'} at {z:+.0f} mm, "
                 f"rack {rack:+.0f}")
            # rim: rod points inside the wheel's width must stay inside the rim
            ubj = np.array([res.ubj.x_mm, res.ubj.y_mm, res.ubj.z_mm])
            lbj = np.array([res.lbj.x_mm, res.lbj.y_mm, res.lbj.z_mm])
            tro = np.array([res.tro.x_mm, res.tro.y_mm, res.tro.z_mm])
            spin = solver._reconstruct_spin_axis(ubj, lbj, tro)
            wc = np.array([res.wheel_center.x_mm, res.wheel_center.y_mm,
                           res.wheel_center.z_mm]) + lift
            for s in np.linspace(0.0, 1.0, 41):
                p = mount + s * (rr - mount)
                ax_ = float((p - wc) @ spin)
                if abs(ax_) <= half_w:
                    radial = float(np.linalg.norm(p - wc - ax_ * spin))
                    keep("gap_rod_rim", r_rim - radial - rod_r,
                         f"at {z:+.0f} mm, rack {rack:+.0f}")
        lk = left[list(zs).index(z)]
        if axle == "front":
            stg = ctx.project.steering
            x_r = -float(stg.rack_x_mm)            # rack_x_mm is rearward-positive
            ra = np.array([x_r, -stg.rack_half_length_mm, stg.rack_z_mm])
            rb = np.array([x_r, stg.rack_half_length_mm, stg.rack_z_mm])
            ca, cb = (np.array(v) for v in E["column_mm"])
            for st in (lk, lk.mirrored()):
                for name, p, q, rad in st.elements():
                    rad = eye_r if name == "eyes" else rad
                    keep("gap_rack", seg_seg(p, q, ra, rb) - rad - E["rack_housing_od_mm"] / 2,
                         f"{name} at {z:+.0f} mm")
                    keep("gap_column", seg_seg(p, q, ca, cb) - rad - E["column_od_mm"] / 2,
                         f"{name} at {z:+.0f} mm")
        else:
            res = solver.solve(wheel_travel_mm=float(z))
            wc = np.array([res.wheel_center.x_mm, res.wheel_center.y_mm,
                           res.wheel_center.z_mm + float(z)])
            inner = np.array([corner.wheel_center.x_mm, E["diff_output_y_mm"],
                              corner.wheel_center.z_mm])
            for name, p, q, rad in lk.elements():
                rad = eye_r if name == "eyes" else rad
                keep("gap_driveshaft", seg_seg(p, q, wc, inner) - rad
                     - E["driveshaft_od_mm"] / 2.0, f"{name} at {z:+.0f} mm")
    for k, (g, w) in worst.items():
        out[k], out[k + "_where"] = g, w

    # ---- brackets (both corners; the worse one)
    for key, p in (("pivot_bracket", pivot), ("damper_bracket", dc)):
        bl, br = bracket(ctx, p), bracket(ctx, mirror(p))
        b, side = (bl, "L") if bl["mm"] >= br["mm"] else (br, "R")
        out[key + "_mm"] = b["mm"]
        out[key + "_where"] = (f"{b['tube']} ({side}), foot {b['from_tube_end_mm']:.0f} mm "
                               f"from its nearer end"
                               + (f"; {b['closer_sketch_only']}" if "closer_sketch_only" in b
                                  else ""))

    # ---- angles over the physical range
    out["rod_angle_max_deg"] = float(sw.rod_angle_deg.max())
    out["damper_angle_max_deg"] = float(sw.damper_angle_deg.max())
    out["psi_range_deg"] = (float(sw.psi_deg.min()), float(sw.psi_deg.max()))
    out["psi_abs_max_deg"] = float(np.abs(sw.psi_deg).max())

    # ---- loads at ride (static corner weight through the fitted spring)
    st0 = sw0.states[int(np.argmin(np.abs(sw0.travel_mm)))]
    u_rod = _unit(st0.rod_mount - st0.rocker_rod)        # rocker pickup -> mount
    i_up = int(np.argmin(np.abs(sw0.travel_mm - 5.0)))
    m_up = sw0.states[i_up].rod_mount
    tension = (float(np.linalg.norm(m_up - st0.rocker_rod))
               > float(np.linalg.norm(st0.rod_mount - st0.rocker_rod)))
    f_rod = a.rod_force_ride_N * (u_rod if tension else -u_rod)
    f_dmp = a.spring_force_ride_N * _unit(st0.rocker_damper - dc)
    reaction = -(f_rod + f_dmp)
    moment = float(axis @ (np.cross(st0.rocker_rod - pivot, f_rod)
                           + np.cross(st0.rocker_damper - pivot, f_dmp)))
    out.update(rod_in_tension=tension,
               pivot_reaction_ride_N=float(np.linalg.norm(reaction)),
               pivot_bracket_couple_Nm=float(np.linalg.norm(reaction)) * out["pivot_bracket_mm"]
               / 1000.0,
               damper_bracket_couple_Nm=a.spring_force_ride_N * out["damper_bracket_mm"] / 1000.0,
               rocker_moment_check_Nmm=moment)
    if not tension:
        d_o, d_i = E["rod_od_mm"], E["rod_od_mm"] - 2 * E["rod_wall_mm"]
        inertia = math.pi / 64.0 * (d_o ** 4 - d_i ** 4)
        p_cr = math.pi ** 2 * E["rod_E_N_mm2"] * inertia / a.rod_length_mm ** 2
        out.update(rod_buckling_N=p_cr, wheel_load_to_buckle_N=p_cr / a.rod_per_wheel)
    else:
        out.update(rod_buckling_N=math.nan, wheel_load_to_buckle_N=math.nan)
    # spring seated: the coil-over extends past the spring's static compression in droop
    out["spring_unseats_at_mm"] = -a.spring_compression_ride_mm / a.mr_ride

    # ---- geometry descriptors
    arm_r = float(np.linalg.norm(np.asarray(lay.rocker_rod) - pivot))
    arm_d = float(np.linalg.norm(np.asarray(lay.rocker_damper) - pivot))
    rod_dir = _unit(np.asarray(lay.rocker_rod) - np.asarray(lay.rod_mount))
    pts = [p for s in left for p in (s.rr, s.rd, s.pivot, s.dc)]
    out.update(rocker_arm_rod_mm=arm_r, rocker_arm_damper_mm=arm_d,
               rod_incl_from_horizontal_deg=math.degrees(math.asin(abs(float(rod_dir[2])))),
               rod_plan_angle_deg=math.degrees(math.atan2(abs(float(rod_dir[0])),
                                                          abs(float(rod_dir[1])))),
               rocker_z_range_mm=(min(float(p[2]) for p in pts), max(float(p[2]) for p in pts)),
               moving_mass_z_mm=_moving_mass_z(spec))
    return out


def _moving_mass_z(spec: RockerSpec) -> float:
    E = ESTIMATES
    zc = 0.5 * (spec.rocker_damper[2] + spec.damper_chassis[2])
    return (E["coilover_kg"] * zc + E["rocker_kg"] * spec.rocker_pivot[2]) / \
        (E["coilover_kg"] + E["rocker_kg"])


def axle_summary(a: S.AxleSuspension) -> dict:
    if not a.ok:
        return {"error": a.error}
    fitted = [o for o in a.springs if o.fitted][0]
    return dict(
        rod_type=a.spec.rod_type, mount_body=a.spec.mount_body,
        mr_droop=a.mr_droop, mr_ride=a.mr_ride, mr_bump=a.mr_bump,
        progression_pct=a.progression_pct,
        damper_droop_mm=a.damper_droop_mm, damper_ride_mm=a.damper_ride_mm,
        damper_bump_mm=a.damper_bump_mm, droop_margin_mm=a.droop_margin_mm,
        bump_margin_mm=a.bump_margin_mm, stroke_used_mm=a.stroke_used_mm,
        rod_length_mm=a.rod_length_mm, rod_per_wheel=a.rod_per_wheel,
        rod_force_ride_N=a.rod_force_ride_N, spring_force_ride_N=a.spring_force_ride_N,
        wheel_rate_fitted_n_mm=fitted.wheel_rate_n_mm, freq_fitted_hz=fitted.frequency_hz,
        spring_roll_nm_deg=fitted.roll_rate_nm_deg,
    )


def evaluate(ctx: Ctx, specs: dict[str, RockerSpec], ref_z: dict[str, float] | None = None
             ) -> dict:
    p = replace(ctx.project, rocker_front=specs["front"], rocker_rear=specs["rear"])
    rep = S.suspension_report(ctx.geo, p, ctx.rc)
    out: dict = {"front": {}, "rear": {}}
    for ax, a in (("front", rep.front), ("rear", rep.rear)):
        out[ax].update(axle_summary(a))
        try:
            out[ax].update(evaluate_axle(ctx, ax, a))
        except (RockerSolveError, ValueError) as exc:
            out[ax]["error"] = f"packaging sweep failed: {exc}"
        out[ax]["fails"] = [f"{lab}: {v:.1f} (limit {op} {lim:g})"
                            for lab, v, op, lim, ok in check(ax, out[ax]) if not ok]
    if rep.budget is not None:
        b = rep.budget
        out["roll"] = dict(required_nm_deg=b.required_nm_deg, springs_nm_deg=b.springs_nm_deg,
                           arb_needed_nm_deg=b.arb_needed_nm_deg,
                           gradient_springs_only=b.gradient_springs_only)
    # CG effect of where the coil-overs and rockers sit (4 corners), against a reference
    if ref_z is not None and all("moving_mass_z_mm" in out[ax] for ax in ("front", "rear")):
        E = ESTIMATES
        m = 2 * (E["coilover_kg"] + E["rocker_kg"])
        dz = sum(m * (out[ax]["moving_mass_z_mm"] - ref_z[ax]) for ax in ("front", "rear"))
        veh = ctx.geo.vehicle_data
        dh = dz / veh.total_mass_kg
        tilt0 = T.tilt_table(ctx.geo, ctx.rc, T.TiltInputs()).tip_angle_deg
        tilt1 = T.tilt_table(ctx.geo, ctx.rc,
                             T.TiltInputs(cg_height_mm=veh.cg_height_mm + dh)).tip_angle_deg
        out["cg"] = dict(delta_cg_mm=dh, tilt_ref_deg=tilt0, tilt_deg=tilt1)
    return out


def check(axle: str, m: dict) -> list[tuple[str, float, str, float, bool]]:
    out = []
    for key, ax, kind, lim, label in THRESHOLDS:
        if ax not in (None, axle) or m.get(key) is None:
            continue
        v = float(m[key])
        ok = v >= lim if kind == "min" else v <= lim
        out.append((label, v, "≥" if kind == "min" else "≤", lim, ok))
    return out


def specs_for(ctx: Ctx, concept: dict) -> dict[str, RockerSpec]:
    out = {}
    for ax in ("front", "rear"):
        c = concept[ax]
        if "spec" in c:                       # a fully specified rocker
            out[ax] = RockerSpec(**{**c["spec"], "enabled": True})
            continue
        ac = AxleConcept(c["rod_type"], c["mount_body"], tuple(c["rod_mount"]),
                         tuple(c["pivot"]), tuple(c["damper_toward"]),
                         float(c["rod_arm_mm"]), float(c.get("target_mr", 0.80)),
                         float(c.get("damper_clock_deg", 0.0)))
        out[ax] = make_spec(ctx.geo, ctx.project, ax, ac)
    return out


def reference_z(ctx: Ctx) -> dict[str, float]:
    """Coil-over/rocker height of the project's current (seed) layout."""
    return {"front": _moving_mass_z(ctx.project.rocker_front),
            "rear": _moving_mass_z(ctx.project.rocker_rear)}


# --------------------------------------------------------------------------- #
# Views (SVG, no plotting dependency) -- same projections as engine.chassis
# --------------------------------------------------------------------------- #

def views_svg(ctx: Ctx, specs: dict[str, RockerSpec], title: str, scale: float = 0.2) -> str:
    g = ctx.geo
    links = (("UCA_IN_FRONT", "UCA_OUT"), ("UCA_IN_REAR", "UCA_OUT"),
             ("LCA_IN_FRONT", "LCA_OUT"), ("LCA_IN_REAR", "LCA_OUT"),
             ("TIE_ROD_IN", "TIE_ROD_OUT"), ("UCA_OUT", "LCA_OUT"),
             ("WHEEL_CENTER", "CONTACT_PATCH"))
    susp = [(np.asarray(g.points[c][a], float), np.asarray(g.points[c][b], float))
            for c in ("FL", "FR", "RL", "RR") for a, b in links]
    rods, rockers, dampers = [], [], []
    for ax in ("front", "rear"):
        for cid in (("FL", "FR") if ax == "front" else ("RL", "RR")):
            lay = S.layout_for(specs[ax], cid)
            m = np.asarray(lay.rod_mount)
            o, rr = np.asarray(lay.rocker_pivot), np.asarray(lay.rocker_rod)
            rd, dc = np.asarray(lay.rocker_damper), np.asarray(lay.damper_chassis)
            rods.append((m, rr))
            rockers += [(o, rr), (o, rd), (rr, rd)]
            dampers.append((rd, dc))
    box_edges = []
    for (ex0, ex1), eyc, ehw, (ez0, ez1) in ESTIMATES["engine_boxes_mm"]:
        box = [np.array(v, float) for v in
               ((ex0, eyc - ehw, ez0), (ex1, eyc - ehw, ez0), (ex1, eyc + ehw, ez0),
                (ex0, eyc + ehw, ez0), (ex0, eyc - ehw, ez1), (ex1, eyc - ehw, ez1),
                (ex1, eyc + ehw, ez1), (ex0, eyc + ehw, ez1))]
        box_edges += [(box[i], box[j]) for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                                    (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                                    (2, 6), (3, 7))]
    stg = ctx.project.steering
    x_r = -float(stg.rack_x_mm)
    rack = [(np.array([x_r, -stg.rack_half_length_mm, stg.rack_z_mm]),
             np.array([x_r, stg.rack_half_length_mm, stg.rack_z_mm])),
            tuple(np.array(v) for v in ESTIMATES["column_mm"])]
    frame = [(m.a, m.b) for m in ctx.members if m.bodied]
    sketch = [(m.a, m.b) for m in ctx.members if not m.bodied]
    tyres = []                      # outline only: loaded radius, tyre width
    half_w = 0.5 * float(ctx.project.turning_envelope.tyre_section_width_mm)
    for cid in ("FL", "FR", "RL", "RR"):
        wc = np.asarray(g.points[cid]["WHEEL_CENTER"], float)
        r = float(wc[2])
        for dy in (-half_w, half_w):
            ring = [wc + np.array([r * math.cos(t), dy, r * math.sin(t)])
                    for t in np.linspace(0.0, 2 * math.pi, 25)]
            tyres += list(zip(ring[:-1], ring[1:], strict=True))
        for t in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            e = wc + np.array([r * math.cos(t), 0.0, r * math.sin(t)])
            tyres.append((e - np.array([0, half_w, 0]), e + np.array([0, half_w, 0])))
    layers = (("#d6dae1", 1.0, tyres, ""), ("#b8c0cc", 1.2, frame, ""),
              ("#c9ced6", 1.0, sketch, ' stroke-dasharray="2,3"'),
              ("#9aa4b2", 1.0, box_edges, ' stroke-dasharray="5,4"'),
              ("#2563eb", 1.5, susp, ""), ("#0f766e", 2.0, rack, ""),
              ("#ea580c", 2.6, rods, ""), ("#7c3aed", 2.2, rockers, ""),
              ("#16a34a", 6.0, dampers, ' stroke-opacity="0.75" stroke-linecap="round"'))
    pad, gap, title_h = 24.0, 36.0, 44.0
    pts = [p for _c, _w, segs, _x in layers for s in segs for p in s]
    panels, x0, height = [], pad, 0.0
    for vt, (hi, hs), (vi, vs) in C.VIEWS:
        hv = [hs * p[hi] for p in pts]
        vv = [vs * p[vi] for p in pts]
        hmin, hmax, vmin, vmax = min(hv), max(hv), min(vv), max(vv)
        w, h = (hmax - hmin) * scale, (vmax - vmin) * scale
        panels.append((vt, hi, hs, vi, vs, hmin, vmax, x0))
        x0 += w + gap
        height = max(height, h)
    width = x0 - gap + pad
    total_h = height + title_h + 2 * pad + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{total_h:.0f}" '
           f'viewBox="0 0 {width:.1f} {total_h:.1f}" font-family="Segoe UI, Arial, sans-serif" '
           'font-size="12">', f'<rect width="{width:.1f}" height="{total_h:.1f}" fill="#ffffff"/>',
           f'<text x="{pad:.1f}" y="{pad:.1f}" font-size="15" font-weight="700">{title}</text>']
    for vt, hi, hs, vi, vs, hmin, vmax, px in panels:
        oy = pad + title_h

        def xy(p, hi=hi, hs=hs, vi=vi, vs=vs, hmin=hmin, vmax=vmax, px=px, oy=oy):
            return (px + (hs * p[hi] - hmin) * scale, oy + (vmax - vs * p[vi]) * scale)

        out.append(f'<text x="{px:.1f}" y="{pad + 22:.1f}" font-weight="600">{vt}</text>')
        for color, wdt, segs, extra in layers:
            for a, b in segs:
                (x1, y1), (x2, y2) = xy(a), xy(b)
                out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                           f'stroke="{color}" stroke-width="{wdt}"{extra}/>')
    ly = total_h - 12
    out.append(f'<text x="{pad:.1f}" y="{ly:.1f}"><tspan fill="#6b7280">grey: CAD frame (dotted: '
               'sketch line with no body); dashed box: engine envelope (estimate)</tspan>  '
               '<tspan fill="#2563eb">blue: wishbones, tie rods</tspan>  <tspan fill="#0f766e">'
               'teal: rack, column (estimate)</tspan>  <tspan fill="#ea580c">orange: rods</tspan>'
               '  <tspan fill="#7c3aed">purple: rockers</tspan>  <tspan fill="#16a34a">green: '
               'coil-overs</tspan></text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

ROWS = (  # (label, key, format)
    ("Rod type / mounted on", None, ""),
    ("Rod in", "rod_in_tension", ""),
    ("MR droop / ride / bump (±25 mm)", ("mr_droop", "mr_ride", "mr_bump"), ".3f"),
    ("MR progression, droop to bump [%]", "progression_pct", "+.1f"),
    ("Damper stroke used in ±25 mm [mm] (of 57, estimate)", "stroke_used_mm", ".1f"),
    ("Damper end stops, wheel travel [mm]", "end_stops_mm", "+.1f"),
    ("Spring unseats at wheel travel [mm] (220 lb/in, no preload)", "spring_unseats_at_mm",
     "+.1f"),
    ("Rocker rotation to the end stops [deg]", "psi_range_deg", "+.1f"),
    ("Rocker arms rod / damper [mm]", ("rocker_arm_rod_mm", "rocker_arm_damper_mm"), ".0f"),
    ("Transmission angle max, rod / damper [deg]",
     ("rod_angle_max_deg", "damper_angle_max_deg"), ".1f"),
    ("Rod length [mm]", "rod_length_mm", ".0f"),
    ("Rod inclination from horizontal / plan sweep [deg]",
     ("rod_incl_from_horizontal_deg", "rod_plan_angle_deg"), ".0f"),
    ("Rod force / wheel load", "rod_per_wheel", ".2f"),
    ("Rod force at ride [N]", "rod_force_ride_N", ".0f"),
    ("Wheel load to buckle the rod [N] (16×1 steel)", "wheel_load_to_buckle_N", ".0f"),
    ("Rocker pivot reaction at ride [N]", "pivot_reaction_ride_N", ".0f"),
    ("Bracket couple at ride, pivot / coil-over end [N·m] (upper bound: force × stand-off)",
     ("pivot_bracket_couple_Nm", "damper_bracket_couple_Nm"), ".0f"),
    ("Gap: rod vs frame [mm]", "gap_rod_frame", ".1f"),
    ("Gap: rocker vs frame [mm]", "gap_rocker_frame", ".1f"),
    ("Gap: coil-over body vs frame [mm]", "gap_coilover_frame", ".1f"),
    ("Gap: coil-over eyes vs frame [mm]", "gap_eyes_frame", ".1f"),
    ("Gap: rod vs wishbones / tie rod / toe link, with steer [mm]", "gap_rod_members", ".1f"),
    ("Gap: rod vs rim, with steer [mm] (to the 13\" bead-seat radius)", "gap_rod_rim", ".1f"),
    ("Gap: left vs right, heave and roll [mm]", "gap_left_right", ".1f"),
    ("Gap: linkage vs rack [mm]", "gap_rack", ".1f"),
    ("Gap: linkage vs steering column [mm] (estimate)", "gap_column", ".1f"),
    ("Gap: linkage vs drive shaft [mm] (estimate)", "gap_driveshaft", ".1f"),
    ("Gap: linkage vs engine envelope [mm] (estimate)", "gap_engine", ".1f"),
    ("Gap: linkage vs solid bodies' bounding boxes [mm]", "gap_solid_bodies", ".1f"),
    ("Rocker pivot to nearest bodied tube [mm]", "pivot_bracket_mm", ".1f"),
    ("Coil-over chassis end to nearest bodied tube [mm]", "damper_bracket_mm", ".1f"),
    ("Rocker / coil-over centreline height range z [mm]", "rocker_z_range_mm", ".0f"),
    ("Coil-over + rocker mass height z [mm]", "moving_mass_z_mm", ".0f"),
)


def _cell(m: dict, key, fmt: str, axle: str) -> str:
    if key is None:
        return f"{m.get('rod_type', '?')} / {str(m.get('mount_body', '?')).upper()}"
    if key == "rod_in_tension":
        v = m.get(key)
        return "—" if v is None else ("tension" if v else "compression")
    keys = key if isinstance(key, tuple) else (key,)
    vals = []
    for k in keys:
        v = m.get(k)
        if v is None:
            return "—"
        if isinstance(v, (list, tuple)):
            vals.append(" .. ".join(format(float(x), fmt) for x in v))
        elif isinstance(v, float) and not math.isfinite(v):
            vals.append("n/a")
        else:
            vals.append(format(float(v), fmt))
    s = " / ".join(vals)
    for k in keys:
        for key_, ax_, kind, lim, _label in THRESHOLDS:
            if key_ == k and m.get(k) is not None and ax_ in (None, axle):
                v = float(m[k])
                if not (v >= lim if kind == "min" else v <= lim):
                    s = f"**{s}** ✗"
    if key == "gap_solid_bodies" and m.get(key) is not None and float(m[key]) <= 0.0:
        s = f"**{s}** ⚠"
    return s


def _pt(p, fmt: str = ".1f") -> str:
    return "(" + ", ".join(format(float(v), fmt) for v in p) + ")"


def report(ctx: Ctx, cols) -> str:
    E = ESTIMATES
    n_sk = sum(not m.bodied for m in ctx.members)
    L = ["# Rocker and push/pull rod concepts (FSAE 2027)", "",
         "Generated by `scripts/rocker_concepts.py build` from the `concept_*.json` files in "
         "this folder. Each concept is also saved as a project variant, "
         "`projects/FSAE_2027_rocker_concept_<id>.vdproj.json`, to open in VD Studio "
         "(Rocker page). The team project is unchanged.", "",
         "**What this is:** the consequences of three layouts, side by side. **What it is "
         "not:** a choice -- no concept is recommended. The axles are independent: a front "
         "from one concept and a rear from another combine without changing either.", "",
         "- Frame: ISO 8855 (X forward, Y LEFT, Z up), mm, chassis-fixed; LEFT corner points "
         "(the right is the Y-mirror). Kinematics: `DWSolver` + `vdcore.analysis.rocker`.",
         "- Motion ratio 0.80 at ride on every axle is a **query argument** so the concepts "
         "compare like for like (the MR target is still an open decision). The coil-over is "
         "placed for equal droop/bump margins at ±25 mm; the rocker arms are tangential at "
         "ride unless the concept clocks them.",
         "- MR and damper usage are over the ±25 mm design travel. Packaging, transmission "
         "angles and rocker rotation are over the **physical** range, to the damper's end "
         "stops (no bump rubber in the project), and for the front over the full rack travel "
         f"(±{ctx.project.steering.max_rack_travel_mm:g} mm).",
         "- **Gaps are air gaps**: centreline distance minus both estimated radii. Both "
         "corners are checked against the CAD as built (the frame is not symmetric at the "
         "engine mounts).",
         f"- Chassis: CAD survey `Xasis 33.3`. {len(ctx.members)} members, of which {n_sk} "
         "are sketch lines with no weldment body (dotted in the views; they may not be real "
         f"tubes); {len(ctx.boxes)} solid bodies without a centreline are kept as bounding "
         "boxes (a box overstates a diagonal tube, so ⚠ means 'look in CAD', not a clash). "
         "Bracket lengths are to bodied members only.",
         "- **Estimates** (no CAD): coil-over OD "
         f"{E['coilover_od_mm']:.0f} mm (full OD from {E['coilover_body_from_eye_mm']:.0f} mm "
         f"off each eye), eyes OD {ctx.project.damper.bearing_od_mm:g} mm; rod "
         f"{E['rod_od_mm']:.0f}×{E['rod_wall_mm']:.0f} mm steel; rocker envelope "
         f"±{E['rocker_half_mm']:.0f} mm, pickup bolts ±{E['pickup_bolt_half_mm']:.0f} mm; "
         f"wishbone OD {E['wishbone_od_mm']:.0f}, tie rod OD {E['tie_rod_od_mm']:.0f}, rack "
         f"housing OD {E['rack_housing_od_mm']:.0f} mm; steering column "
         f"{_pt(E['column_mm'][0], '.0f')} → {_pt(E['column_mm'][1], '.0f')}, OD "
         f"{E['column_od_mm']:.0f}; damper stroke {ctx.project.damper.stroke_mm:g} mm "
         f"({ctx.project.damper.stroke_source}); drive shaft from the wheel centre to a joint "
         f"at |y| = {E['diff_output_y_mm']:.0f} mm on the axle line, OD "
         f"{E['driveshaft_od_mm']:.0f}; engine envelope "
         + " and ".join(f"x {b[0][0]:.0f}..{b[0][1]:.0f}, y {b[1] - b[2]:.0f}..{b[1] + b[2]:.0f}"
                        for b in E["engine_boxes_mm"])
         + f", z {E['engine_boxes_mm'][0][3][0]:.0f}..{E['engine_boxes_mm'][0][3][1]:.0f} mm "
         "(from the CAD engine-mount tabs, centred about y -27); coil-over "
         f"{E['coilover_kg']} kg and rocker {E['rocker_kg']} kg per corner (CG effect only).",
         "", "## Thresholds behind the ✗ flags", "",
         "Information, not requirements.", "", "| check | limit |", "|---|---|"]
    for _k, ax, kind, lim, label in THRESHOLDS:
        unit = "deg" if ("angle" in label or "rotation" in label) else "mm"
        L.append(f"| {label}{'' if ax is None else f' ({ax})'} | {'≥' if kind == 'min' else '≤'}"
                 f" {lim:g} {unit} |")
    for axle in ("front", "rear"):
        L += ["", f"## {axle.capitalize()} axle", "",
              "| | " + " | ".join("Seed" if cid == "seed" else f"Concept {cid}"
                                   for cid, *_ in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for label, key, fmt in ROWS:
            if axle == "front" and key in ("gap_driveshaft", "gap_engine"):
                continue
            if axle == "rear" and key in ("gap_rack", "gap_column"):
                continue
            cells = []
            for _cid, _n, res, _s, _c in cols:
                m = res[axle]
                cells.append(f"error: {m['error']}" if m.get("error") else
                             _cell(m, key, fmt, axle))
            L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "## Car level", "",
          "| | " + " | ".join("Seed" if cid == "seed" else cid for cid, *_ in cols) + " |",
          "|---|" + "---|" * len(cols)]
    for label, f in (("Springs-only roll gradient [deg/g] (220 lb/in both)",
                      lambda r: f"{r['roll']['gradient_springs_only']:.2f}"),
                     ("ARB needed for 1.0 deg/g at 50/50, F / R [N·m/deg]",
                      lambda r: f"{r['roll']['arb_needed_nm_deg']['front']:.0f} / "
                                f"{r['roll']['arb_needed_nm_deg']['rear']:.0f}"),
                     ("CG height change vs seed [mm] (estimate)",
                      lambda r: f"{r['cg']['delta_cg_mm']:+.1f}"),
                     ("Tilt-table tip angle [deg] (seed "
                      f"{cols[0][2]['cg']['tilt_ref_deg']:.2f})",
                      lambda r: f"{r['cg']['tilt_deg']:.2f}")):
        cells = []
        for _cid, _n, res, _s, _c in cols:
            try:
                cells.append(f(res))
            except (KeyError, TypeError):
                cells.append("—")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "The MR is the same at ride on every axle, so wheel rates, frequencies and the "
          "roll budget only move with the MR progression; they are listed to show that, not "
          "as a difference between the layouts. The spring unseating row is the same story: "
          "with 220 lb/in and no preload the spring comes off its seat before the damper "
          "tops out, whatever the layout."]
    for cid, name, res, specs, c in cols:
        L += ["", f"## {'Seed' if cid == 'seed' else 'Concept ' + cid}: {name}", "",
              f"![views]({'seed' if cid == 'seed' else 'concept_' + cid}_views.svg)", ""]
        if c.get("summary"):
            L += [c["summary"], ""]
        L += ["| point (LEFT corner) | front | rear |", "|---|---|---|"]
        for lab, attr in (("rod outer end (on the mount body)", "rod_mount"),
                          ("rocker pivot", "rocker_pivot"), ("rocker axis (unit)", "rocker_axis"),
                          ("rocker rod pickup", "rocker_rod"),
                          ("rocker damper pickup", "rocker_damper"),
                          ("coil-over chassis end", "damper_chassis")):
            fmt = ".4f" if attr == "rocker_axis" else ".1f"
            L.append(f"| {lab} | {_pt(getattr(specs['front'], attr), fmt)} | "
                     f"{_pt(getattr(specs['rear'], attr), fmt)} |")
        L.append(f"| rod type / mount body | {specs['front'].rod_type} / "
                 f"{specs['front'].mount_body} | {specs['rear'].rod_type} / "
                 f"{specs['rear'].mount_body} |")
        L += ["", "**Where the tightest numbers are:**", ""]
        for ax in ("front", "rear"):
            m = res[ax]
            for key in ("gap_rod_frame", "gap_rocker_frame", "gap_coilover_frame",
                        "gap_rod_members", "gap_rod_rim", "gap_engine", "gap_solid_bodies",
                        "pivot_bracket", "damper_bracket"):
                w = m.get(key + "_where")
                if w:
                    v = m.get(key, m.get(key + "_mm"))
                    L.append(f"- {ax} {key.replace('_', ' ')}: {float(v):.1f} mm — {w}")
        fails = [(ax, *row) for ax in ("front", "rear") for row in check(ax, res[ax])
                 if not row[-1]]
        L += ["", "**Outside the thresholds:** " + ("none" if not fails else "; ".join(
            f"{ax} {lab} {v:.1f} (limit {op} {lim:g})" for ax, lab, v, op, lim, _ok in fails))]
        for head, key in (("Pros", "pros"), ("Cons", "cons"), ("Not checked", "not_checked")):
            if c.get(key):
                L += ["", f"**{head}**", ""] + [f"- {s}" for s in c[key]]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def cmd_eval(path: Path) -> None:
    ctx = load_ctx()
    concept = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = specs_for(ctx, concept)
    res = evaluate(ctx, specs, reference_z(ctx))
    res["specs"] = {ax: asdict(s) for ax, s in specs.items()}
    print(json.dumps(_jsonable(res), indent=1))


def cmd_seed() -> None:
    ctx = load_ctx()
    specs = {"front": ctx.project.rocker_front, "rear": ctx.project.rocker_rear}
    print(json.dumps(_jsonable(evaluate(ctx, specs, reference_z(ctx))), indent=1))


def cmd_frame() -> None:
    ctx = load_ctx()
    for m in ctx.members:
        print(f"{'bodied ' if m.bodied else 'SKETCH '} {m.name:40s} {_pt(m.a)} -> {_pt(m.b)}")
    for b in ctx.boxes:
        print(f"box     {b.name:40s} {_pt(b.lo, '.0f')} .. {_pt(b.hi, '.0f')}")


def cmd_build() -> None:
    ctx = load_ctx()
    ref = reference_z(ctx)
    files = sorted(OUT_DIR.glob("concept_*.json"))
    if not files:
        raise SystemExit(f"no concept_*.json in {OUT_DIR}")
    seed_specs = {"front": ctx.project.rocker_front, "rear": ctx.project.rocker_rear}
    cols: list[tuple[str, str, dict, dict, dict]] = [
        ("seed", "the current team project", evaluate(ctx, seed_specs, ref), seed_specs, {})]
    for f in files:
        c = json.loads(f.read_text(encoding="utf-8"))
        cid = str(c.get("concept_id", f.stem.split("_")[-1]))
        specs = specs_for(ctx, c)
        cols.append((cid, str(c.get("name", cid)), evaluate(ctx, specs, ref), specs, c))
        variant = replace(
            ctx.project, rocker_front=specs["front"], rocker_rear=specs["rear"],
            name=f"{ctx.project.name} -- rocker concept {cid}",
            notes=(f"Rocker/rod concept {cid}: {c.get('name', '')}. Built by "
                   "scripts/rocker_concepts.py from projects/rocker_concepts/"
                   f"{f.name} (MR 0.80 is a comparison query, not a choice). See "
                   "projects/rocker_concepts/README.md."))
        save_project(variant, REPO / "projects" / f"FSAE_2027_rocker_concept_{cid}.vdproj.json")
        (OUT_DIR / f"concept_{cid}_views.svg").write_text(
            views_svg(ctx, specs, f"Concept {cid}: {c.get('name', '')}"), encoding="utf-8")
    (OUT_DIR / "seed_views.svg").write_text(
        views_svg(ctx, seed_specs, "Seed (current project)"), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(report(ctx, cols), encoding="utf-8")
    print(f"wrote {len(cols) - 1} variants, views and {OUT_DIR / 'README.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="metrics of one concept JSON")
    e.add_argument("concept", type=Path)
    sub.add_parser("seed", help="metrics of the project's current layout")
    sub.add_parser("frame", help="the CAD frame model (bodied / sketch-only members, boxes)")
    sub.add_parser("build", help="variants, report and views for every concept")
    args = ap.parse_args()
    {"eval": lambda: cmd_eval(args.concept), "seed": cmd_seed, "frame": cmd_frame,
     "build": cmd_build}[args.cmd]()


if __name__ == "__main__":
    main()
