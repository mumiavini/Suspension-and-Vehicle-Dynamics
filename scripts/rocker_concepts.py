"""Rocker / push-pull rod layout concepts, checked against the chassis CAD.

Each concept is a set of construction inputs per axle (rod type, the member the
rod is mounted on, the rod's outer end, the rocker pivot, the direction the
damper lies in, the rod arm and the motion ratio to tune to). From them:

1. the rocker is built with both forces tangential at ride height and the
   damper arm tuned (bisection) to the asked motion ratio, the damper's ride
   length set for equal droop/bump margins -- the same construction as
   ``engine.suspension.build_rocker`` / ``tune_rocker``, extended to a pullrod
   (the damper arm is put on the side that compresses the damper in bump);
2. the linkage is solved over the axle's travel on ``DWSolver`` by
   ``vdcore.analysis.rocker`` through ``engine.suspension`` (MR, damper usage,
   rates, roll budget, rod clearance to the wishbones and tie rod);
3. packaging is measured against the chassis CAD survey ("Xasis 33.3"): the
   rod and the damper against every frame centreline over the travel, the
   rocker pivot and the damper's chassis end against the nearest tube (bracket
   length), left against right, the rod against the steering rack (front) and
   a drive shaft (rear), the rocker and damper against an engine envelope.

What it is: consequences of layouts the designer proposes. What it is not: a
choice. The engine envelope, the drive-shaft inboard joint, the damper and
coil-over sizes, the moving masses and the rod tube are ESTIMATES (named in
``ESTIMATES``) -- the CAD has no engine, differential, dampers or rockers. The
motion ratio target is a query argument (the same for every concept so they
compare like for like), not a recommendation.

Usage::

    uv run python scripts/rocker_concepts.py eval CONCEPT.json      # metrics as JSON
    uv run python scripts/rocker_concepts.py build                  # variants, report, views

Frame: ISO 8855 (X forward, Y LEFT, Z up), chassis-fixed, mm; specs are the
LEFT corner, the right is the Y-mirror.
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
from vdcore.analysis.rocker import rocker_sweep  # noqa: E402
from vdcore.geometry.solver import DWSolver  # noqa: E402

PROJECT = REPO / "projects" / "FSAE_2027.vdproj.json"
SURVEY = REPO / "projects" / "cad" / "Xasis 33.3_survey.json"
OUT_DIR = REPO / "projects" / "rocker_concepts"
CONCEPTS = OUT_DIR / "concepts.json"

# --------------------------------------------------------------------------- #
# Estimates (no CAD for these parts). Every number here is source="estimate".
# --------------------------------------------------------------------------- #
ESTIMATES = {
    "frame_tube_od_mm": 25.4,          # most of the frame (SIZE_B/C 25.4 mm)
    "rod_od_mm": 16.0,                 # push/pull rod tube
    "rod_wall_mm": 1.0,
    "rod_E_N_mm2": 205e3,              # steel
    "coilover_od_mm": 60.0,            # damper body with the spring over it
    "damper_eye_clear_mm": 30.0,       # each end of the damper sits on its bracket
    "rod_end_clear_mm": 35.0,          # the rod's rocker end sits on the rocker
    "rack_housing_od_mm": 40.0,
    "driveshaft_od_mm": 60.0,          # CV joint / tripod housing envelope
    "diff_output_y_mm": 120.0,         # drive shaft inboard joint, |y|, at the rear axle
    # engine envelope from the CAD's engine-mount tabs (3DSketch6/7/8): x, |y|, z ranges
    "engine_box_mm": ((-1400.0, -1080.0), 190.0, (115.0, 490.0)),
    # moving masses per corner (for the CG effect only)
    "coilover_kg": 1.3,
    "rocker_kg": 0.3,
}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


# --------------------------------------------------------------------------- #
# Construction (a designer's helper, not a pick)
# --------------------------------------------------------------------------- #

def build(mount, pivot, damper_toward, rod_arm_mm: float, damper_arm_mm: float,
          damper_ride_mm: float, mount_body: str, rod_type: str, flip: bool) -> RockerSpec:
    """``engine.suspension.build_rocker`` with the damper arm optionally on the other side.

    ``flip`` puts the damper pickup on the opposite side of the pivot: for a pullrod
    it is the side that compresses the damper in bump.
    """
    sp = S.build_rocker(mount, pivot, damper_toward, rod_arm_mm, damper_arm_mm,
                        damper_ride_mm, mount_body, rod_type)
    if not flip:
        return sp
    o = np.asarray(sp.rocker_pivot, float)
    ud = _unit(np.asarray(damper_toward, float) - o)
    rd = o - (np.asarray(sp.rocker_damper, float) - o)
    dc = rd + damper_ride_mm * ud

    def t(v):
        return tuple(round(float(x), 4) for x in v)

    return replace(sp, rocker_damper=t(rd), damper_chassis=t(dc))


@dataclass(frozen=True)
class AxleConcept:
    rod_type: str                      # "pushrod" | "pullrod"
    mount_body: str                    # "lca" | "uca" | "upright"
    rod_mount: tuple[float, float, float]
    pivot: tuple[float, float, float]
    damper_toward: tuple[float, float, float]
    rod_arm_mm: float
    target_mr: float = 0.80


def make_spec(geo, project: Project, axle: str, c: AxleConcept) -> RockerSpec:
    """Damper arm for the target MR (bisection on 10-200 mm), then equal damper margins."""
    cid = S.AXLE_CORNER[axle]
    corner = geo.corner(cid)
    droop, bump = S.travel_range(geo, axle)

    def mr(arm: float, flip: bool) -> float:
        sp = build(c.rod_mount, c.pivot, c.damper_toward, c.rod_arm_mm, arm, 200.0,
                   c.mount_body, c.rod_type, flip)
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
                   c.mount_body, c.rod_type, flip)
        sw = rocker_sweep(corner, S.layout_for(sp, cid), [droop, bump])
        m_droop = d.eye_to_eye_extended_mm - float(sw.damper_length_mm.max())
        m_bump = float(sw.damper_length_mm.min()) - compressed
        ride += 0.5 * (m_droop - m_bump)
    assert sp is not None
    return replace(sp, source="design_intent")


# --------------------------------------------------------------------------- #
# Context and geometry helpers
# --------------------------------------------------------------------------- #

@dataclass
class Ctx:
    project: Project
    geo: object
    rc: dict[str, float]
    tubes: list[tuple[str, np.ndarray, np.ndarray]]      # ISO, both sides
    nodes: np.ndarray


def load_ctx() -> Ctx:
    p = load_project(PROJECT)
    g = G.resolve(p)
    st = K.compute_static(g)
    rc = {"front": st.value("front.rc_static"), "rear": st.value("rear.rc_static")}
    sv = C.load_survey(SURVEY)
    fr = X.cad_frame(p)
    tubes = [(ln.sketch, C.cad_to_iso(ln.a, fr), C.cad_to_iso(ln.b, fr)) for ln in sv.lines]
    nodes = np.array([C.cad_to_iso(k, fr) for k in sv.nodes])
    return Ctx(p, g, rc, tubes, nodes)


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


def seg_box(a, b, box) -> float:
    """Distance from a segment to an axis-aligned box (sampled; 0 = inside)."""
    lo = np.array([box[0][0], -box[1], box[2][0]])
    hi = np.array([box[0][1], box[1], box[2][1]])
    best = math.inf
    for s in np.linspace(0.0, 1.0, 81):
        p = np.asarray(a, float) + s * (np.asarray(b, float) - np.asarray(a, float))
        q = np.clip(p, lo, hi)
        best = min(best, float(np.linalg.norm(p - q)))
    return best


def mirror(p) -> np.ndarray:
    p = np.asarray(p, float)
    return np.array([p[0], -p[1], p[2]])


def nearest_tube(ctx: Ctx, p) -> tuple[float, str]:
    return min((pt_seg(p, a, b), sk) for sk, a, b in ctx.tubes)


def nearest_node(ctx: Ctx, p) -> float:
    return float(np.min(np.linalg.norm(ctx.nodes - np.asarray(p, float), axis=1)))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate_axle(ctx: Ctx, axle: str, a: S.AxleSuspension) -> dict:
    """Packaging and load numbers on top of the engine's axle report."""
    E = ESTIMATES
    spec, sw = a.spec, a.sweep
    out: dict = {}
    if not a.ok or sw is None:
        return {"error": a.error}
    cid = S.AXLE_CORNER[axle]
    corner = ctx.geo.corner(cid)
    solver = DWSolver(corner)
    dc = np.asarray(spec.damper_chassis, float)
    pivot = np.asarray(spec.rocker_pivot, float)

    rod_tube = (math.inf, "")
    damper_tube = (math.inf, "")
    lr = math.inf
    shaft = math.inf
    rack = math.inf
    engine = math.inf
    z_lo, z_hi = math.inf, -math.inf
    for z, st in zip(sw.travel_mm, sw.states, strict=True):
        rm, rr, rd = st.rod_mount, st.rocker_rod, st.rocker_damper
        r0, r1 = trim(rm, rr, 0.0, E["rod_end_clear_mm"])
        d0, d1 = trim(rd, dc, E["damper_eye_clear_mm"], E["damper_eye_clear_mm"])
        for sk, ta, tb in ctx.tubes:
            v = seg_seg(r0, r1, ta, tb)
            if v < rod_tube[0]:
                rod_tube = (v, f"{sk} at {z:+.0f} mm")
            v = seg_seg(d0, d1, ta, tb)
            if v < damper_tube[0]:
                damper_tube = (v, f"{sk} at {z:+.0f} mm")
        # left vs right: this side's rod, rocker arms and damper against the mirror's
        mine = [(rm, rr), (pivot, rr), (pivot, rd), (rd, dc)]
        other = [(mirror(p), mirror(q)) for p, q in mine]
        lr = min(lr, min(seg_seg(p, q, mp, mq) for p, q in mine for mp, mq in other))
        if axle == "rear":
            res = solver.solve(wheel_travel_mm=float(z))
            wc = np.array([res.wheel_center.x_mm, res.wheel_center.y_mm,
                           res.wheel_center.z_mm + float(z)])
            inner = np.array([corner.wheel_center.x_mm, E["diff_output_y_mm"],
                              corner.wheel_center.z_mm])
            shaft = min(shaft, seg_seg(rm, rr, wc, inner))
        else:
            stg = ctx.project.steering
            ra = np.array([stg.rack_x_mm, -stg.rack_half_length_mm, stg.rack_z_mm])
            rb = np.array([stg.rack_x_mm, stg.rack_half_length_mm, stg.rack_z_mm])
            rack = min(rack, seg_seg(rm, rr, ra, rb), seg_seg(d0, d1, ra, rb),
                       pt_seg(pivot, ra, rb))
        for p, q in ((pivot, rr), (pivot, rd), (rd, dc), (r0, r1)):
            engine = min(engine, seg_box(p, q, E["engine_box_mm"]))
        for p in (rr, rd, dc, pivot):
            z_lo, z_hi = min(z_lo, float(p[2])), max(z_hi, float(p[2]))

    piv_tube = nearest_tube(ctx, pivot)
    dc_tube = nearest_tube(ctx, dc)
    rod_len = a.rod_length_mm
    if spec.rod_type == "pushrod":
        d_o, d_i = E["rod_od_mm"], E["rod_od_mm"] - 2 * E["rod_wall_mm"]
        inertia = math.pi / 64.0 * (d_o ** 4 - d_i ** 4)
        p_cr = math.pi ** 2 * E["rod_E_N_mm2"] * inertia / rod_len ** 2
        buckle_wheel_N = p_cr / a.rod_per_wheel
    else:
        p_cr, buckle_wheel_N = math.nan, math.nan       # tension member

    arm_rod = float(np.linalg.norm(np.asarray(spec.rocker_rod) - pivot))
    arm_dmp = float(np.linalg.norm(np.asarray(spec.rocker_damper) - pivot))
    u1 = _unit(np.asarray(spec.rocker_rod) - pivot)
    u2 = _unit(np.asarray(spec.rocker_damper) - pivot)
    arms_angle = math.degrees(math.acos(float(np.clip(u1 @ u2, -1, 1))))
    damper_dir = _unit(dc - np.asarray(spec.rocker_damper, float))
    rod_dir = _unit(np.asarray(spec.rocker_rod) - np.asarray(spec.rod_mount))
    out.update(
        rod_tube_clear_mm=rod_tube[0], rod_tube_with=rod_tube[1],
        damper_tube_clear_mm=damper_tube[0], damper_tube_with=damper_tube[1],
        left_right_clear_mm=lr,
        driveshaft_clear_mm=shaft if axle == "rear" else None,
        rack_clear_mm=rack if axle == "front" else None,
        engine_clear_mm=engine,
        pivot_to_tube_mm=piv_tube[0], pivot_tube=piv_tube[1],
        pivot_to_node_mm=nearest_node(ctx, pivot),
        damper_end_to_tube_mm=dc_tube[0], damper_end_tube=dc_tube[1],
        damper_end_to_node_mm=nearest_node(ctx, dc),
        rocker_arm_rod_mm=arm_rod, rocker_arm_damper_mm=arm_dmp,
        rocker_arms_angle_deg=arms_angle,
        rocker_z_range_mm=(z_lo, z_hi),
        damper_direction=tuple(round(float(v), 3) for v in damper_dir),
        rod_direction=tuple(round(float(v), 3) for v in rod_dir),
        rod_incl_from_horizontal_deg=math.degrees(math.asin(abs(float(rod_dir[2])))),
        rod_plan_angle_deg=math.degrees(math.atan2(abs(float(rod_dir[0])),
                                                   abs(float(rod_dir[1])))),
        rod_buckling_N=p_cr, wheel_load_to_buckle_N=buckle_wheel_N,
        moving_mass_z_mm=_moving_mass_z(spec),
    )
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
        psi_range_deg=a.psi_range_deg, rod_angle_max_deg=a.rod_angle_max_deg,
        damper_angle_max_deg=a.damper_angle_max_deg, rod_length_mm=a.rod_length_mm,
        rod_per_wheel=a.rod_per_wheel, rod_force_ride_N=a.rod_force_ride_N,
        member_clear_mm=a.clearance_mm, member_clear_with=a.clearance_with,
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
        out[ax].update(evaluate_axle(ctx, ax, a))
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


def specs_for(ctx: Ctx, concept: dict) -> dict[str, RockerSpec]:
    out = {}
    for ax in ("front", "rear"):
        c = concept[ax]
        if "spec" in c:                       # a fully specified rocker
            out[ax] = RockerSpec(**{**c["spec"], "enabled": True})
            continue
        ac = AxleConcept(c["rod_type"], c["mount_body"], tuple(c["rod_mount"]),
                         tuple(c["pivot"]), tuple(c["damper_toward"]),
                         float(c["rod_arm_mm"]), float(c.get("target_mr", 0.80)))
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
    (ex0, ex1), ey, (ez0, ez1) = ESTIMATES["engine_box_mm"]
    box = [np.array(v, float) for v in
           ((ex0, -ey, ez0), (ex1, -ey, ez0), (ex1, ey, ez0), (ex0, ey, ez0),
            (ex0, -ey, ez1), (ex1, -ey, ez1), (ex1, ey, ez1), (ex0, ey, ez1))]
    box_edges = [(box[i], box[j]) for i, j in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                                               (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))]
    stg = ctx.project.steering
    rack = [(np.array([stg.rack_x_mm, -stg.rack_half_length_mm, stg.rack_z_mm]),
             np.array([stg.rack_x_mm, stg.rack_half_length_mm, stg.rack_z_mm]))]
    frame = [(a, b) for _sk, a, b in ctx.tubes]
    tyres = []                      # outline only: loaded radius, ~200 mm wide
    for cid in ("FL", "FR", "RL", "RR"):
        wc = np.asarray(g.points[cid]["WHEEL_CENTER"], float)
        r = float(wc[2])
        for dy in (-100.0, 100.0):
            ring = [wc + np.array([r * math.cos(t), dy, r * math.sin(t)])
                    for t in np.linspace(0.0, 2 * math.pi, 25)]
            tyres += list(zip(ring[:-1], ring[1:], strict=True))
        for t in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            e = wc + np.array([r * math.cos(t), 0.0, r * math.sin(t)])
            tyres.append((e - np.array([0, 100.0, 0]), e + np.array([0, 100.0, 0])))
    layers = (("#d6dae1", 1.0, tyres, ""), ("#b8c0cc", 1.2, frame, ""),
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
    out.append(f'<text x="{pad:.1f}" y="{ly:.1f}"><tspan fill="#6b7280">grey: CAD frame; dashed: '
               'engine envelope (estimate)</tspan>  <tspan fill="#2563eb">blue: wishbones, tie '
               'rods</tspan>  <tspan fill="#0f766e">teal: rack</tspan>  <tspan fill="#ea580c">'
               'orange: push/pull rods</tspan>  <tspan fill="#7c3aed">purple: rockers</tspan>  '
               '<tspan fill="#16a34a">green: coil-overs</tspan></text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
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
    res = evaluate(ctx, specs, reference_z(ctx))
    print(json.dumps(_jsonable(res), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="metrics of one concept JSON")
    e.add_argument("concept", type=Path)
    sub.add_parser("seed", help="metrics of the project's current layout")
    sub.add_parser("build", help="variants, report and views for every concept")
    args = ap.parse_args()
    if args.cmd == "eval":
        cmd_eval(args.concept)
    elif args.cmd == "seed":
        cmd_seed()
    else:
        cmd_build()


# Feasibility thresholds (centreline to centreline, mm, from the ESTIMATES sizes):
# (key, axle or None for both, "min"/"max", limit, label)
THRESHOLDS = (
    ("rod_tube_clear_mm", None, "min", 26.0, "rod vs frame"),
    ("damper_tube_clear_mm", None, "min", 48.0, "coil-over vs frame"),
    ("member_clear_mm", None, "min", 25.0, "rod vs wishbone legs / tie rod / toe link"),
    ("left_right_clear_mm", None, "min", 65.0, "left vs right linkage"),
    ("rack_clear_mm", "front", "min", 33.0, "rod / coil-over vs steering rack"),
    ("driveshaft_clear_mm", "rear", "min", 48.0, "rod vs drive shaft (estimate)"),
    ("engine_clear_mm", "rear", "min", 20.0, "linkage vs engine envelope (estimate)"),
    ("pivot_to_tube_mm", None, "max", 60.0, "rocker pivot to nearest tube (bracket)"),
    ("damper_end_to_tube_mm", None, "max", 60.0, "coil-over chassis end to nearest tube"),
    ("rod_angle_max_deg", None, "max", 30.0, "rod transmission angle"),
    ("damper_angle_max_deg", None, "max", 30.0, "coil-over transmission angle"),
)

ROWS = (  # (label, key, format)
    ("Rod type / mounted on", None, ""),
    ("MR droop / ride / bump", ("mr_droop", "mr_ride", "mr_bump"), ".3f"),
    ("MR progression, droop to bump [%]", "progression_pct", "+.1f"),
    ("Damper stroke used [mm] (of 57, estimate)", "stroke_used_mm", ".1f"),
    ("Damper margin droop / bump [mm]", ("droop_margin_mm", "bump_margin_mm"), ".1f"),
    ("Rocker rotation over travel [deg]", "psi_range_deg", ".1f"),
    ("Rocker arms rod / damper [mm]", ("rocker_arm_rod_mm", "rocker_arm_damper_mm"), ".0f"),
    ("Transmission angle max, rod / damper [deg]",
     ("rod_angle_max_deg", "damper_angle_max_deg"), ".1f"),
    ("Rod length [mm]", "rod_length_mm", ".0f"),
    ("Rod inclination from horizontal [deg]", "rod_incl_from_horizontal_deg", ".0f"),
    ("Rod force / wheel load", "rod_per_wheel", ".2f"),
    ("Rod force at ride [N]", "rod_force_ride_N", ".0f"),
    ("Wheel load to buckle the rod [N] (16x1 steel)", "wheel_load_to_buckle_N", ".0f"),
    ("Rod vs frame [mm]", "rod_tube_clear_mm", ".1f"),
    ("Coil-over vs frame [mm]", "damper_tube_clear_mm", ".1f"),
    ("Rod vs wishbones / tie rod / toe link [mm]", "member_clear_mm", ".1f"),
    ("Left vs right [mm]", "left_right_clear_mm", ".1f"),
    ("Rod / coil-over vs rack [mm]", "rack_clear_mm", ".1f"),
    ("Rod vs drive shaft [mm] (estimate)", "driveshaft_clear_mm", ".1f"),
    ("Linkage vs engine envelope [mm] (estimate)", "engine_clear_mm", ".1f"),
    ("Rocker pivot to nearest tube [mm]", "pivot_to_tube_mm", ".1f"),
    ("Coil-over chassis end to nearest tube [mm]", "damper_end_to_tube_mm", ".1f"),
    ("Rocker / coil-over height range z [mm]", "rocker_z_range_mm", ".0f"),
    ("Coil-over + rocker mass height z [mm]", "moving_mass_z_mm", ".0f"),
)


def check(axle: str, m: dict) -> list[tuple[str, float, str, float, bool]]:
    out = []
    for key, ax, kind, lim, label in THRESHOLDS:
        if ax not in (None, axle) or m.get(key) is None:
            continue
        v = float(m[key])
        ok = v >= lim if kind == "min" else v <= lim
        out.append((label, v, "≥" if kind == "min" else "≤", lim, ok))
    return out


def _cell(m: dict, key, fmt: str) -> str:
    if key is None:
        return f"{m.get('rod_type', '?')} / {m.get('mount_body', '?').upper()}"
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
            if key_ == k and m.get(k) is not None and ax_ in (None, m.get("_axle")):
                v = float(m[k])
                if not (v >= lim if kind == "min" else v <= lim):
                    s = f"**{s}** ✗"
    return s


def _pt(p, fmt: str = ".1f") -> str:
    return "(" + ", ".join(format(float(v), fmt) for v in p) + ")"


def cmd_build() -> None:
    ctx = load_ctx()
    ref = reference_z(ctx)
    files = sorted(OUT_DIR.glob("concept_*.json"))
    if not files:
        raise SystemExit(f"no concept_*.json in {OUT_DIR}")
    seed_specs = {"front": ctx.project.rocker_front, "rear": ctx.project.rocker_rear}
    seed = evaluate(ctx, seed_specs, ref)
    cols: list[tuple[str, str, dict, dict, dict]] = [("seed", "Seed (current project)",
                                                      seed, seed_specs, {})]
    for f in files:
        c = json.loads(f.read_text(encoding="utf-8"))
        cid = str(c.get("concept_id", f.stem.split("_")[-1]))
        specs = specs_for(ctx, c)
        res = evaluate(ctx, specs, ref)
        cols.append((cid, str(c.get("name", cid)), res, specs, c))
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


def report(ctx: Ctx, cols) -> str:
    E = ESTIMATES
    L = ["# Rocker and push/pull rod concepts (FSAE 2027)", "",
         "Generated by `scripts/rocker_concepts.py build` from the `concept_*.json` files in "
         "this folder. Each concept is also saved as a project variant, "
         "`projects/FSAE_2027_rocker_concept_<id>.vdproj.json`, to open in VD Studio "
         "(Rocker page). The team project is unchanged.", "",
         "**What this is:** the consequences of three layouts, side by side. **What it is "
         "not:** a choice -- no concept is recommended. The axles are independent: a front "
         "from one concept and a rear from another combine without changing either.", "",
         "- Frame: ISO 8855 (X forward, Y LEFT, Z up), mm, chassis-fixed; LEFT corner (the "
         "right is the Y-mirror). Kinematics: `DWSolver` + `vdcore.analysis.rocker`, "
         "±25 mm wheel travel (chassis-referenced).",
         "- Motion ratio 0.80 at ride on every axle is a **query argument** so the concepts "
         "compare like for like (the MR target is still an open decision). The coil-over is "
         "placed for equal droop/bump margins; the rocker arms are tangential at ride.",
         f"- Chassis: CAD survey `Xasis 33.3` (3D sketch centrelines, {len(ctx.tubes)} lines). "
         "Clearances are centreline to centreline (subtract the radii).",
         "- **Estimates** (no CAD): coil-over OD "
         f"{E['coilover_od_mm']:.0f} mm; rod {E['rod_od_mm']:.0f}×{E['rod_wall_mm']:.0f} mm "
         f"steel; damper stroke {ctx.project.damper.stroke_mm:.0f} mm "
         f"({ctx.project.damper.stroke_source}); drive shaft from the wheel centre to a joint "
         f"at |y| = {E['diff_output_y_mm']:.0f} mm on the axle line, OD "
         f"{E['driveshaft_od_mm']:.0f} mm; engine envelope x "
         f"{E['engine_box_mm'][0][0]:.0f}..{E['engine_box_mm'][0][1]:.0f}, |y| ≤ "
         f"{E['engine_box_mm'][1]:.0f}, z {E['engine_box_mm'][2][0]:.0f}.."
         f"{E['engine_box_mm'][2][1]:.0f} mm (from the CAD engine-mount tabs); coil-over "
         f"{E['coilover_kg']} kg and rocker {E['rocker_kg']} kg per corner (CG effect only).",
         "", "## Thresholds used for the ✗ flags", "",
         "Centreline distances derived from the estimated sizes above; information, not "
         "requirements.", "", "| check | limit |", "|---|---|"]
    for _k, ax, kind, lim, label in THRESHOLDS:
        unit = "deg" if "angle" in label else "mm"
        L.append(f"| {label}{'' if ax is None else f' ({ax})'} | {'≥' if kind == 'min' else '≤'}"
                 f" {lim:g} {unit} |")
    for axle in ("front", "rear"):
        L += ["", f"## {axle.capitalize()} axle", "",
              "| | " + " | ".join(f"{cid if cid == 'seed' else 'Concept ' + cid}"
                                   for cid, *_ in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for label, key, fmt in ROWS:
            if axle == "front" and key == "driveshaft_clear_mm":
                continue
            if axle == "rear" and key == "rack_clear_mm":
                continue
            cells = []
            for _cid, _n, res, _s, _c in cols:
                m = dict(res[axle])
                m["_axle"] = axle
                cells.append(_cell(m, key, fmt) if "error" not in m or not m["error"]
                             else f"error: {m['error']}")
            L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "## Car level", "",
          "| | " + " | ".join(cid for cid, *_ in cols) + " |", "|---|" + "---|" * len(cols)]
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
    L += ["", "MR is the same on every axle, so wheel rates, frequencies and the roll budget "
          "only move with the MR progression; they are listed to show that, not as a "
          "difference between the layouts."]
    for cid, name, res, specs, c in cols:
        L += ["", f"## {'Seed' if cid == 'seed' else 'Concept ' + cid}: {name}", "",
              f"![views]({'seed' if cid == 'seed' else 'concept_' + cid}_views.svg)", ""]
        if c.get("summary"):
            L += [c["summary"], ""]
        L += ["| point (LEFT corner) | front | rear |", "|---|---|---|"]
        for lab, attr in (("rod outer end (on the " + "mount body)", "rod_mount"),
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
        fails = [(ax, *row) for ax in ("front", "rear") for row in check(ax, res[ax])
                 if not row[-1]]
        L += ["", "**Outside the thresholds:** " + ("none" if not fails else "; ".join(
            f"{ax} {lab} {v:.1f} (limit {op} {lim:g})" for ax, lab, v, op, lim, _ok in fails))]
        for head, key in (("Pros", "pros"), ("Cons", "cons"), ("Not checked", "not_checked")):
            if c.get(key):
                L += ["", f"**{head}**", ""] + [f"- {s}" for s in c[key]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
