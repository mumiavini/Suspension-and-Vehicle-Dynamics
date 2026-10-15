#!/usr/bin/env python3
r"""Primitive solids of the whole suspension, as a SolidWorks VBA macro (.bas).

From a VD Studio project (hardpoints + the rocker / rod / coil-over of each axle)
and, optionally, an anti-roll-bar provision, this writes a macro that builds one
solid body per member, all of them cylinders:

* wishbones: the four legs, the inboard pivot bushes, the outboard ball-joint
  housings;
* uprights (simplified): kingpin body, spindle, steering / toe arm, hub; the
  wheel as a hollow tyre (or a plain cylinder if the boolean fails);
* tie rods and rear toe links, the steering rack housing;
* push/pull rods with their rod ends and the bracket on the member they sit on;
* rockers: pivot boss, the plate's three edges, pickup bolts, and the chassis
  stand-offs of a double-shear pivot bracket (one to the nearest bodied tube
  from each side of the rocker);
* coil-overs: damper body, shaft, spring, both eyes, the chassis stand-off;
* the anti-roll bar provision when given: bar, bearings, arms, drop links and
  the rocker's third arm.

The macro also draws each corner's centreline skeleton as a 3D sketch.

It is a modelling aid: placeholder diameters (``SIZES``) on the project's
points. Nothing here is a detailed part, a structural size or a choice.

Two frames, two macros (``--frame``):

* ``chassis`` -- the coordinates of the chassis part (``project.cad_frame``,
  Xasis 33.3). Run it in a new, empty part, then insert that part into the
  assembly with its origin and planes coincident with the chassis part's.
* ``iso`` -- ISO 8855 numbers, like the team's ``susp.SLDPRT`` (whose sketches
  are in ISO): run it inside that part, which is already placed in the assembly.

How to run: SolidWorks > Tools > Macro > New (save a blank .swp), then in the
VBA editor File > Import File... > the .bas, then run ``main`` (F5). Re-running
replaces the bodies it made before (features named ``SUSP_...``).

The SolidWorks API works in metres whatever the document units; the macro
converts. ``IModeler.CreateBodyFromCyl`` is calibrated at run time (whether its
point is the base or the centre of the cylinder), so a reading of the API docs
cannot put every part half a length off.

Usage::

    uv run python scripts/export_suspension_solids.py                # team project
    uv run python scripts/export_suspension_solids.py --project P --arb ARB.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import export_solidworks_skeleton as swx  # noqa: E402
import numpy as np  # noqa: E402

from apps.vdstudio.engine import exports as X  # noqa: E402
from apps.vdstudio.engine import geometry as G  # noqa: E402
from apps.vdstudio.engine import suspension as S  # noqa: E402
from apps.vdstudio.engine.project import Project, load_project  # noqa: E402
from vdcore.geometry.solver import DWSolver  # noqa: E402

PREFIX = "SUSP_"
CORNERS = ("FL", "FR", "RL", "RR")

# Placeholder diameters / lengths (mm). Modelling sizes, not structural ones.
SIZES = {
    "wishbone_od": 19.05, "bush_od": 28.0, "bush_len": 30.0,
    "ball_joint_od": 34.0, "ball_joint_len": 26.0,
    "kingpin_od": 34.0, "spindle_od": 40.0, "arm_od": 18.0,
    "hub_od": 110.0, "hub_len": 40.0,
    "tie_rod_od": 16.0, "rod_end_od": 22.0, "rod_end_len": 14.0,
    "rack_od": 40.0,
    "rod_od": 16.0, "mount_bracket_od": 16.0,
    "rocker_boss_od": 32.0, "rocker_boss_len": 44.0, "rocker_edge_od": 16.0,
    "pickup_bolt_od": 10.0, "pickup_bolt_len": 40.0,
    "standoff_od": 14.0,
    "damper_body_od": 46.0, "damper_shaft_od": 14.0, "spring_od": 60.0,
    "eye_od": 40.0, "eye_len": 20.0,
    "tyre_od": 520.0, "tyre_width": 190.0, "rim_od": 330.0,
    "arb_bar_od": 18.0, "arb_bearing_od": 40.0, "arb_bearing_len": 30.0,
    "arb_arm_od": 14.0, "arb_link_od": 10.0,
}

# Colours (R, G, B in 0..1) per family of parts.
COLOURS = {
    "wishbone": (0.15, 0.39, 0.92), "upright": (0.45, 0.45, 0.50),
    "wheel": (0.12, 0.12, 0.12), "steering": (0.06, 0.46, 0.43),
    "rod": (0.92, 0.35, 0.05), "rocker": (0.49, 0.23, 0.93),
    "coilover": (0.09, 0.64, 0.29), "spring": (0.95, 0.75, 0.10),
    "bracket": (0.60, 0.60, 0.60), "arb": (0.86, 0.15, 0.47),
}


@dataclass(frozen=True)
class Cyl:
    """A solid cylinder from ``a`` to ``b`` (ISO 8855, mm)."""

    name: str
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    od_mm: float
    family: str
    hollow_id_mm: float = 0.0      # > 0: a tube (the tyre)


def _v(p) -> np.ndarray:
    return np.asarray(p, float)


def _unit(v) -> np.ndarray:
    v = _v(v)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("zero-length direction")
    return v / n


def _t(p) -> tuple[float, float, float]:
    return tuple(float(x) for x in p)  # type: ignore[return-value]


def _centred(name: str, c, axis, length: float, od: float, family: str,
             hollow: float = 0.0) -> Cyl:
    u = _unit(axis)
    return Cyl(name, _t(_v(c) - 0.5 * length * u), _t(_v(c) + 0.5 * length * u), od, family,
               hollow)


def _foot(p, a, b) -> np.ndarray:
    p, a, b = _v(p), _v(a), _v(b)
    ab = b - a
    t = float(np.clip((p - a) @ ab / (ab @ ab), 0.0, 1.0))
    return a + t * ab


def nearest_bodied(members, p) -> tuple[np.ndarray, str]:
    """Foot point on the nearest bodied frame member (centreline)."""
    best = None
    for m in members:
        if not m.bodied:
            continue
        f = _foot(p, m.a, m.b)
        d = float(np.linalg.norm(f - _v(p)))
        if best is None or d < best[0]:
            best = (d, f, m.name)
    if best is None:
        raise ValueError("no bodied frame member")
    return best[1], best[2]


def corner_parts(geo, cid: str) -> list[Cyl]:
    """Wishbones, upright, wheel and tie rod / toe link of one corner (static)."""
    P = {k: _v(v) for k, v in geo.points[cid].items()}
    Z = SIZES
    out: list[Cyl] = []
    n = f"{PREFIX}{cid}_"
    for wb in ("UCA", "LCA"):
        f, r, o = P[f"{wb}_IN_FRONT"], P[f"{wb}_IN_REAR"], P[f"{wb}_OUT"]
        out.append(Cyl(f"{n}{wb}_front_leg", _t(f), _t(o), Z["wishbone_od"], "wishbone"))
        out.append(Cyl(f"{n}{wb}_rear_leg", _t(r), _t(o), Z["wishbone_od"], "wishbone"))
        pivot_axis = r - f
        out.append(_centred(f"{n}{wb}_bush_front", f, pivot_axis, Z["bush_len"], Z["bush_od"],
                            "wishbone"))
        out.append(_centred(f"{n}{wb}_bush_rear", r, pivot_axis, Z["bush_len"], Z["bush_od"],
                            "wishbone"))
    kp = P["UCA_OUT"] - P["LCA_OUT"]
    for bj in ("UCA_OUT", "LCA_OUT"):
        out.append(_centred(f"{n}{bj.split('_')[0]}_ball_joint", P[bj], kp, Z["ball_joint_len"],
                            Z["ball_joint_od"], "upright"))
    # upright: kingpin body, spindle (kingpin axis -> wheel centre), steering / toe arm
    out.append(Cyl(f"{n}upright_kingpin", _t(P["LCA_OUT"]), _t(P["UCA_OUT"]), Z["kingpin_od"],
                   "upright"))
    wc, tro = P["WHEEL_CENTER"], P["TIE_ROD_OUT"]
    out.append(Cyl(f"{n}upright_spindle", _t(_foot(wc, P["LCA_OUT"], P["UCA_OUT"])), _t(wc),
                   Z["spindle_od"], "upright"))
    out.append(Cyl(f"{n}upright_arm", _t(_foot(tro, P["LCA_OUT"], P["UCA_OUT"])), _t(tro),
                   Z["arm_od"], "upright"))
    # wheel: spin axis from the static solve
    corner = geo.corner(cid)
    sv = DWSolver(corner)
    res = sv.solve(wheel_travel_mm=0.0)
    if not res.converged:
        raise RuntimeError(f"{cid}: static solve did not converge")
    spin = sv._reconstruct_spin_axis(_v([res.ubj.x_mm, res.ubj.y_mm, res.ubj.z_mm]),
                                     _v([res.lbj.x_mm, res.lbj.y_mm, res.lbj.z_mm]),
                                     _v([res.tro.x_mm, res.tro.y_mm, res.tro.z_mm]))
    out.append(_centred(f"{n}hub", wc, spin, Z["hub_len"], Z["hub_od"], "upright"))
    out.append(_centred(f"{n}tyre", wc, spin, Z["tyre_width"], Z["tyre_od"], "wheel",
                        hollow=Z["rim_od"]))
    # tie rod / toe link with its rod ends
    tri = P["TIE_ROD_IN"]
    kind = "tie_rod" if cid in ("FL", "FR") else "toe_link"
    out.append(Cyl(f"{n}{kind}", _t(tri), _t(tro), Z["tie_rod_od"], "steering"))
    for end, p in (("in", tri), ("out", tro)):
        out.append(_centred(f"{n}{kind}_rod_end_{end}", p, kp, Z["rod_end_len"],
                            Z["rod_end_od"], "steering"))
    return out


def linkage_parts(geo, project: Project, members, cid: str) -> list[Cyl]:
    """Rod, rocker, coil-over and their brackets of one corner (static, ride height)."""
    axle = "front" if cid in ("FL", "FR") else "rear"
    spec = project.rocker_front if axle == "front" else project.rocker_rear
    if not spec.enabled:
        return []
    lay = S.layout_for(spec, cid)
    Z = SIZES
    n = f"{PREFIX}{cid}_"
    m, o, a = _v(lay.rod_mount), _v(lay.rocker_pivot), _unit(lay.rocker_axis)
    rr, rd, dc = _v(lay.rocker_rod), _v(lay.rocker_damper), _v(lay.damper_chassis)
    P = {k: _v(v) for k, v in geo.points[cid].items()}
    out: list[Cyl] = []
    rod = spec.rod_type
    out.append(Cyl(f"{n}{rod}", _t(m), _t(rr), Z["rod_od"], "rod"))
    for end, p in (("mount", m), ("rocker", rr)):
        out.append(_centred(f"{n}{rod}_rod_end_{end}", p, a, Z["rod_end_len"], Z["rod_end_od"],
                            "rod"))
    # the bracket on the member the rod sits on
    body = spec.mount_body
    if body in ("lca", "uca"):
        w = body.upper()
        legs = [(P[f"{w}_IN_FRONT"], P[f"{w}_OUT"]), (P[f"{w}_IN_REAR"], P[f"{w}_OUT"])]
        foot = min((_foot(m, *leg) for leg in legs), key=lambda f: float(np.linalg.norm(f - m)))
    else:
        foot = _foot(m, P["LCA_OUT"], P["UCA_OUT"])
    if float(np.linalg.norm(foot - m)) > 1.0:
        out.append(Cyl(f"{n}{rod}_mount_bracket", _t(foot), _t(m), Z["mount_bracket_od"],
                       "bracket"))
    # rocker: boss, plate edges, pickup bolts
    out.append(_centred(f"{n}rocker_boss", o, a, Z["rocker_boss_len"], Z["rocker_boss_od"],
                        "rocker"))
    for nm, p, q in (("arm_rod", o, rr), ("arm_damper", o, rd), ("edge", rr, rd)):
        out.append(Cyl(f"{n}rocker_{nm}", _t(p), _t(q), Z["rocker_edge_od"], "rocker"))
    for nm, p in (("rod", rr), ("damper", rd)):
        out.append(_centred(f"{n}rocker_bolt_{nm}", p, a, Z["pickup_bolt_len"],
                            Z["pickup_bolt_od"], "rocker"))
    # double-shear pivot bracket: a stand-off from each side of the rocker
    half = 0.5 * Z["rocker_boss_len"]
    for side, sgn in (("a", 1.0), ("b", -1.0)):
        end = o + sgn * half * a
        f, _tube = nearest_bodied(members, end)
        out.append(Cyl(f"{n}rocker_bracket_{side}", _t(f), _t(end), Z["standoff_od"],
                       "bracket"))
    # coil-over: eyes on the rocker axis direction, body at the chassis end
    u = _unit(dc - rd)
    L = float(np.linalg.norm(dc - rd))
    for nm, p in (("eye_rocker", rd), ("eye_chassis", dc)):
        out.append(_centred(f"{n}coilover_{nm}", p, a, Z["eye_len"], Z["eye_od"], "coilover"))
    body_len = 0.55 * L
    out.append(Cyl(f"{n}coilover_body", _t(dc - 0.5 * Z["eye_od"] * u),
                   _t(dc - body_len * u), Z["damper_body_od"], "coilover"))
    out.append(Cyl(f"{n}coilover_shaft", _t(dc - body_len * u),
                   _t(rd + 0.5 * Z["eye_od"] * u), Z["damper_shaft_od"], "coilover"))
    out.append(Cyl(f"{n}coilover_spring", _t(dc - 0.25 * L * u), _t(rd + 0.25 * L * u),
                   Z["spring_od"], "spring", hollow_id_mm=Z["spring_od"] - 12.0))
    for side, sgn in (("a", 1.0), ("b", -1.0)):
        end = dc + sgn * 0.5 * Z["eye_len"] * a
        f, _tube = nearest_bodied(members, end)
        out.append(Cyl(f"{n}coilover_bracket_{side}", _t(f), _t(end), Z["standoff_od"],
                       "bracket"))
    return out


def _mirror(p) -> np.ndarray:
    p = _v(p)
    return np.array([p[0], -p[1], p[2]])


def arb_parts(arb: dict, project: Project) -> list[Cyl]:
    """The anti-roll-bar provision: bar, bearings, arms, drop links, rocker third arms."""
    Z = SIZES
    out: list[Cyl] = []
    for axle, (lc, rc) in (("front", ("FL", "FR")), ("rear", ("RL", "RR"))):
        a = arb.get(axle)
        if not a:
            continue
        spec = project.rocker_front if axle == "front" else project.rocker_rear
        bl = _v(a["bar_bearing"])
        br = _mirror(bl)
        bar_od = float(a.get("bar_od_mm", Z["arb_bar_od"]))
        link_od = float(a.get("link_od_mm", Z["arb_link_od"]))
        tag = "F" if axle == "front" else "R"
        out.append(Cyl(f"{PREFIX}ARB_{tag}_bar", _t(bl), _t(br), bar_od, "arb"))
        for cid, mir in ((lc, False), (rc, True)):
            f = _mirror if mir else _v
            n = f"{PREFIX}{cid}_arb_"
            out.append(_centred(f"{n}bearing", f(bl), br - bl, Z["arb_bearing_len"],
                                Z["arb_bearing_od"], "arb"))
            out.append(Cyl(f"{n}arm", _t(f(a["arm_root"])), _t(f(a["link_bar_end"])),
                           Z["arb_arm_od"], "arb"))
            out.append(Cyl(f"{n}drop_link", _t(f(a["link_bar_end"])), _t(f(a["rocker_arb"])),
                           link_od, "arb"))
            lay = S.layout_for(spec, cid)
            out.append(Cyl(f"{n}rocker_arm", _t(lay.rocker_pivot), _t(f(a["rocker_arb"])),
                           Z["rocker_edge_od"], "rocker"))
    return out


def rack_part(geo) -> list[Cyl]:
    a, b = _v(geo.points["FL"]["TIE_ROD_IN"]), _v(geo.points["FR"]["TIE_ROD_IN"])
    u = _unit(b - a)
    return [Cyl(f"{PREFIX}rack_housing", _t(a + 30.0 * u), _t(b - 30.0 * u), SIZES["rack_od"],
                "steering")]


def all_parts(project: Project, arb: dict | None = None) -> list[Cyl]:
    import rocker_concepts as RC

    geo = G.resolve(project)
    members, _boxes = RC.frame_model(project)
    parts: list[Cyl] = []
    for cid in CORNERS:
        parts += corner_parts(geo, cid)
        parts += linkage_parts(geo, project, members, cid)
    parts += rack_part(geo)
    if arb:
        parts += arb_parts(arb, project)
    names = [p.name for p in parts]
    dup = {x for x in names if names.count(x) > 1}
    if dup:
        raise ValueError(f"duplicate part names: {sorted(dup)}")
    return parts


SKELETON = (("UCA_IN_FRONT", "UCA_OUT"), ("UCA_IN_REAR", "UCA_OUT"),
            ("LCA_IN_FRONT", "LCA_OUT"), ("LCA_IN_REAR", "LCA_OUT"),
            ("TIE_ROD_IN", "TIE_ROD_OUT"), ("LCA_OUT", "UCA_OUT"),
            ("WHEEL_CENTER", "CONTACT_PATCH"))


def skeleton_lines(project: Project) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    geo = G.resolve(project)
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for cid in CORNERS:
        P = geo.points[cid]
        lines = [(_v(P[a]), _v(P[b])) for a, b in SKELETON]
        axle = "front" if cid in ("FL", "FR") else "rear"
        spec = project.rocker_front if axle == "front" else project.rocker_rear
        if spec.enabled:
            lay = S.layout_for(spec, cid)
            o = _v(lay.rocker_pivot)
            lines += [(_v(lay.rod_mount), _v(lay.rocker_rod)), (o, _v(lay.rocker_rod)),
                      (o, _v(lay.rocker_damper)), (_v(lay.rocker_damper), _v(lay.damper_chassis))]
        out[f"{PREFIX}SKELETON_{cid}"] = lines
    return out


# --------------------------------------------------------------------------- #
# VBA
# --------------------------------------------------------------------------- #

def _m(p, frame) -> tuple[float, float, float]:
    """ISO mm -> target frame, metres."""
    return tuple(v / 1000.0 for v in swx._transform(_t(p), frame))  # type: ignore[return-value]


def _num(v: float) -> str:
    s = f"{v:.9g}"
    return s if s not in ("-0",) else "0"


def render_macro(parts: list[Cyl], sketches: dict, frame, source: str) -> str:
    """The VBA module. Number literals use '.', which VBA reads whatever the locale."""
    fam = sorted({p.family for p in parts})
    L: list[str] = []
    add = L.append
    add("' FSAE 2027 suspension -- primitive solids (cylinders) and centreline skeleton")
    add(f"' Generated by scripts/export_suspension_solids.py from {source}")
    add(f"' Frame: {frame.label}. API units are metres.")
    add("' Placeholder diameters on the project's points: a modelling aid, not a detailed")
    add("' or structural design. Do not hand-edit: change the project and regenerate.")
    add("'")
    add("' Tools > Macro > New (save a blank .swp), then in the VBA editor")
    add("' File > Import File... > this .bas, then run main (F5).")
    add("")
    add("Option Explicit")
    add("")
    add("Dim swApp As Object")
    add("Dim swModel As Object")
    add("Dim swModeler As Object")
    add("Dim gCentred As Boolean")
    add("Dim gMade As Long")
    add("Dim gFailed As Long")
    add("Dim gFailList As String")
    add("")
    add("Sub main()")
    add("    Set swApp = Application.SldWorks")
    add("    Set swModel = swApp.ActiveDoc")
    add("    Dim useActive As Boolean")
    add("    useActive = False")
    add("    If Not swModel Is Nothing Then")
    add("        If swModel.GetType = 1 Then")
    add('            If InStr(1, LCase(swModel.GetTitle), "susp", vbTextCompare) > 0 Then '
        "useActive = True")
    add("        End If")
    add("    End If")
    add("    If Not useActive Then")
    add("        ' a new, empty part: never write into the chassis or another part")
    add("        Dim tmpl As String")
    add("        tmpl = swApp.GetUserPreferenceStringValue(8)")
    add("        Set swModel = swApp.NewDocument(tmpl, 0, 0, 0)")
    add("        If swModel Is Nothing Then")
    add('            MsgBox "Could not create a new part (default part template)."')
    add("            Exit Sub")
    add("        End If")
    add("    End If")
    add("    Set swModeler = swApp.GetModeler")
    add("    DeletePrevious")
    add("    Calibrate")
    add("    swModel.FeatureManager.EnableFeatureTree = False")
    add("    gMade = 0: gFailed = 0: gFailList = \"\"")
    for f in fam:
        add(f"    Build_{f}")
    add("    swModel.FeatureManager.EnableFeatureTree = True")
    add("    BuildSketches")
    add("    swModel.EditRebuild3")
    add("    swModel.ViewZoomtofit2")
    add('    Dim msg As String')
    add('    msg = "Suspension solids: " & gMade & " made, " & gFailed & " failed."')
    add('    If gFailed > 0 Then msg = msg & vbCrLf & gFailList')
    add("    MsgBox msg")
    add("End Sub")
    add("")
    add("' CreateBodyFromCyl takes a point, an axis, a radius and a length. Whether the point")
    add("' is the base or the middle of the cylinder is measured here on a probe body.")
    add("Private Sub Calibrate()")
    add("    Dim p(7) As Double")
    add("    p(0) = 0: p(1) = 0: p(2) = 0: p(3) = 0: p(4) = 0: p(5) = 1")
    add("    p(6) = 0.01: p(7) = 0.1")
    add("    Dim v As Variant")
    add("    v = p")
    add("    Dim b As Object")
    add("    Set b = swModeler.CreateBodyFromCyl((v))")
    add("    gCentred = False")
    add("    If Not b Is Nothing Then")
    add("        Dim bx As Variant")
    add("        bx = b.GetBodyBox")
    add("        If Abs(bx(2) + 0.05) < 0.005 Then gCentred = True")
    add("    End If")
    add("End Sub")
    add("")
    add("Private Sub DeletePrevious()")
    add("    Dim f As Object")
    add("    Dim n As Long")
    add("    swModel.ClearSelection2 True")
    add("    Set f = swModel.FirstFeature")
    add("    Do While Not f Is Nothing")
    add(f'        If Left(f.Name, {len(PREFIX)}) = "{PREFIX}" Then')
    add("            If f.Select2(True, 0) Then n = n + 1")
    add("        End If")
    add("        Set f = f.GetNextFeature")
    add("    Loop")
    add("    ' delete the selection with its children, without a confirmation dialog")
    add("    If n > 0 Then")
    add("        If Not swModel.Extension.DeleteSelection2(3) Then swModel.EditDelete")
    add("    End If")
    add("    swModel.ClearSelection2 True")
    add("End Sub")
    add("")
    add("Private Function MakeCylBody(ByVal x1 As Double, ByVal y1 As Double, _")
    add("                             ByVal z1 As Double, ByVal x2 As Double, _")
    add("                             ByVal y2 As Double, ByVal z2 As Double, _")
    add("                             ByVal r As Double) As Object")
    add("    Dim dx As Double, dy As Double, dz As Double, L As Double")
    add("    dx = x2 - x1: dy = y2 - y1: dz = z2 - z1")
    add("    L = Sqr(dx * dx + dy * dy + dz * dz)")
    add("    If L < 0.000001 Or r <= 0 Then Exit Function")
    add("    Dim p(7) As Double")
    add("    If gCentred Then")
    add("        p(0) = (x1 + x2) / 2: p(1) = (y1 + y2) / 2: p(2) = (z1 + z2) / 2")
    add("    Else")
    add("        p(0) = x1: p(1) = y1: p(2) = z1")
    add("    End If")
    add("    p(3) = dx / L: p(4) = dy / L: p(5) = dz / L: p(6) = r: p(7) = L")
    add("    Dim v As Variant")
    add("    v = p")
    add("    Set MakeCylBody = swModeler.CreateBodyFromCyl((v))")
    add("End Function")
    add("")
    add("Private Sub Cyl(ByVal nm As String, ByVal x1 As Double, ByVal y1 As Double, _")
    add("                ByVal z1 As Double, ByVal x2 As Double, ByVal y2 As Double, _")
    add("                ByVal z2 As Double, ByVal od As Double, ByVal idia As Double, _")
    add("                ByVal cr As Double, ByVal cg As Double, ByVal cb As Double)")
    add("    Dim b As Object, inner As Object, f As Object")
    add("    Set b = MakeCylBody(x1, y1, z1, x2, y2, z2, od / 2)")
    add("    If b Is Nothing Then GoTo failed")
    add("    If idia > 0 Then")
    add("        ' a tube: cut a longer inner cylinder; keep the solid one if the cut fails")
    add("        Dim ex As Double, ey As Double, ez As Double, L As Double")
    add("        L = Sqr((x2 - x1) ^ 2 + (y2 - y1) ^ 2 + (z2 - z1) ^ 2)")
    add("        ex = (x2 - x1) / L * 0.001: ey = (y2 - y1) / L * 0.001")
    add("        ez = (z2 - z1) / L * 0.001")
    add("        Set inner = MakeCylBody(x1 - ex, y1 - ey, z1 - ez, x2 + ex, y2 + ey, z2 + ez, "
        "idia / 2)")
    add("        If Not inner Is Nothing Then")
    add("            Dim res As Variant, opErr As Long")
    add("            On Error Resume Next")
    add("            res = b.Operations2(15902, inner, opErr)")
    add("            If opErr = 0 And Not IsEmpty(res) Then")
    add("                If UBound(res) >= 0 Then Set b = res(0)")
    add("            End If")
    add("            On Error GoTo 0")
    add("        End If")
    add("    End If")
    add("    Set f = swModel.CreateFeatureFromBody3(b, False, 0)")
    add("    If f Is Nothing Then GoTo failed")
    add("    f.Name = nm")
    add("    Dim c(8) As Double")
    add("    c(0) = cr: c(1) = cg: c(2) = cb: c(3) = 0.5: c(4) = 0.7: c(5) = 0.3")
    add("    c(6) = 0.3: c(7) = 0: c(8) = 0")
    add("    Dim cv As Variant")
    add("    cv = c")
    add("    On Error Resume Next")
    add("    f.SetMaterialPropertyValues2 cv, 1, Nothing")
    add("    On Error GoTo 0")
    add("    gMade = gMade + 1")
    add("    Exit Sub")
    add("failed:")
    add("    gFailed = gFailed + 1")
    add("    If gFailed <= 20 Then gFailList = gFailList & nm & vbCrLf")
    add("End Sub")
    for f in fam:
        col = COLOURS.get(f, (0.6, 0.6, 0.6))
        add("")
        add(f"Private Sub Build_{f}()")
        for p in parts:
            if p.family != f:
                continue
            a, b = _m(p.a, frame), _m(p.b, frame)
            args = ", ".join(_num(v) for v in (*a, *b, p.od_mm / 1000.0, p.hollow_id_mm / 1000.0,
                                                *col))
            add(f'    Cyl "{p.name}", {args}')
        add("End Sub")
    add("")
    add("Private Sub BuildSketches()")
    add("    Dim sm As Object, f As Object")
    add("    Set sm = swModel.SketchManager")
    add("    sm.AddToDB = True")
    add("    sm.DisplayWhenAdded = False")
    for name, lines in sketches.items():
        add("    swModel.ClearSelection2 True")
        add("    sm.Insert3DSketch True")
        for p, q in lines:
            add("    sm.CreateLine " + ", ".join(_num(v) for v in (*_m(p, frame), *_m(q, frame))))
        add("    sm.Insert3DSketch True")
        add("    Set f = swModel.FeatureByPositionReverse(0)")
        add(f'    If Not f Is Nothing Then f.Name = "{name}"')
    add("    sm.AddToDB = False")
    add("    sm.DisplayWhenAdded = True")
    add("End Sub")
    add("")
    return "\r\n".join(L)


def preview_svg(project: Project, parts: list[Cyl], title: str, scale: float = 0.2) -> str:
    """Top, side and front views of the solids (stroke width = diameter) over the CAD frame
    and the placed engine, for a look before SolidWorks."""
    import rocker_concepts as RC
    from scipy.spatial import ConvexHull

    members, _boxes = RC.frame_model(project)
    hulls, _fit = RC.engine_model(project)
    frame = [(m.a, m.b) for m in members if m.bodied]
    eng = []
    for h in hulls:
        for sx in ConvexHull(h.points).simplices:
            for i in range(3):
                eng.append((h.points[sx[i]], h.points[sx[(i + 1) % 3]]))
    views = (("Top (front up)", (1, -1), (0, 1)),
             ("Side, from the left (front to the left)", (0, -1), (2, 1)),
             ("Front, from ahead (car's left on the right)", (1, 1), (2, 1)))
    pts = [_v(q) for c in parts for q in (c.a, c.b)] + [q for seg in frame for q in seg]
    r_max = 0.5 * max(c.od_mm for c in parts)       # room for the widest solid (the tyre)
    pts += [q + d for q in pts[:2 * len(parts)] for d in
            (np.array([r_max, 0, 0]), np.array([-r_max, 0, 0]), np.array([0, 0, r_max]),
             np.array([0, 0, -r_max]))]
    pad, gap, title_h = 24.0, 36.0, 44.0
    panels, x0, height = [], pad, 0.0
    for vt, (hi, hs), (vi, vs) in views:
        hv = [hs * q[hi] for q in pts]
        vv = [vs * q[vi] for q in pts]
        hmin, hmax, vmin, vmax = min(hv), max(hv), min(vv), max(vv)
        panels.append((vt, hi, hs, vi, vs, hmin, vmax, x0))
        x0 += (hmax - hmin) * scale + gap
        height = max(height, (vmax - vmin) * scale)
    width, total_h = x0 - gap + pad, height + title_h + 2 * pad + 30
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{total_h:.0f}" '
           f'viewBox="0 0 {width:.1f} {total_h:.1f}" font-family="Segoe UI, Arial, sans-serif" '
           'font-size="12">', f'<rect width="{width:.1f}" height="{total_h:.1f}" fill="#fff"/>',
           f'<text x="{pad}" y="{pad}" font-size="15" font-weight="700">{title}</text>']
    order = ("wheel", "wishbone", "upright", "steering", "bracket", "arb", "rod", "rocker",
             "coilover", "spring")
    for vt, hi, hs, vi, vs, hmin, vmax, px in panels:
        oy = pad + title_h

        def xy(q, hi=hi, hs=hs, vi=vi, vs=vs, hmin=hmin, vmax=vmax, px=px, oy=oy):
            return (px + (hs * q[hi] - hmin) * scale, oy + (vmax - vs * q[vi]) * scale)

        out.append(f'<text x="{px:.1f}" y="{pad + 22:.1f}" font-weight="600">{vt}</text>')
        for col, segs, w in (("#b8c0cc", frame, 1.2), ("#e2b4a6", eng, 0.6)):
            for a, b in segs:
                (x1, y1), (x2, y2) = xy(a), xy(b)
                out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                           f'stroke="{col}" stroke-width="{w}"/>')
        for fam in order:
            r, g, b_ = (int(255 * c) for c in COLOURS[fam])
            op = 0.25 if fam == "wheel" else 0.85
            for c in parts:
                if c.family != fam:
                    continue
                (x1, y1), (x2, y2) = xy(c.a), xy(c.b)
                out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                           f'stroke="rgb({r},{g},{b_})" stroke-opacity="{op}" '
                           f'stroke-width="{max(1.0, c.od_mm * scale):.1f}" '
                           'stroke-linecap="round"/>')
    out.append(f'<text x="{pad}" y="{total_h - 12:.1f}" fill="#6b7280">grey: CAD frame '
               '(bodied members); light red: engine features, plenum, chain (estimate); '
               'solids drawn with their diameters</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", type=Path, default=REPO / "projects" / "FSAE_2027.vdproj.json")
    ap.add_argument("--arb", type=Path, default=None,
                    help="anti-roll-bar provision JSON (rocker_concepts format)")
    ap.add_argument("--frame", choices=("chassis", "iso", "both"), default="both")
    ap.add_argument("--out", type=Path, default=REPO / "projects" / "cad",
                    help="output folder")
    ap.add_argument("--name", default="suspension_solids")
    args = ap.parse_args()
    project = load_project(args.project)
    arb = json.loads(args.arb.read_text(encoding="utf-8")) if args.arb else None
    parts = all_parts(project, arb)
    sk = skeleton_lines(project)
    frames = {"chassis": X.cad_frame(project), "iso": swx.resolve_frame("iso")}
    todo = ("chassis", "iso") if args.frame == "both" else (args.frame,)
    src = args.project.name + (f" + {args.arb.name}" if args.arb else "")
    for f in todo:
        text = render_macro(parts, sk, frames[f], src)
        path = args.out / f"{args.name}_{'Xasis_33.3' if f == 'chassis' else 'iso'}.bas"
        path.write_bytes(text.encode("cp1252", errors="replace"))
        print(f"wrote {path} ({len(parts)} solids)")
    svg = args.out / f"{args.name}_preview.svg"
    svg.write_text(preview_svg(project, parts, f"Suspension solids: {project.name}"),
                   encoding="utf-8")
    print(f"wrote {svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
