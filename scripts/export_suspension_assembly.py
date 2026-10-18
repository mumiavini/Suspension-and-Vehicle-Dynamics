#!/usr/bin/env python3
r"""The suspension as a SolidWorks assembly: one part per component, mated to move.

Same project and placeholder solids as ``export_suspension_solids.py`` (which
puts every body in one part), split by component. The macro (.bas) builds, in a
folder chosen when it runs:

* one part per component of each corner -- ``SUSP_FL_UCA``, ``SUSP_FL_LCA``,
  ``SUSP_FL_upright``, ``SUSP_FL_wheel``, ``SUSP_FL_tie_rod`` (rear:
  ``toe_link``), ``SUSP_FL_pushrod``, ``SUSP_FL_rocker``,
  ``SUSP_FL_damper_body`` and ``SUSP_FL_damper_shaft`` -- with its solids and
  named reference points (``P_...``) and axes (``A_...``) at its joints;
* ``SUSP_FL_chassis_mounts``: the chassis side of the corner (the rocker and
  coil-over stand-offs, and the reference geometry of every chassis pick-up),
  fixed. The stand-offs are frame tabs for the chassis team;
* a sub-assembly per corner (``SUSP_FL``) whose mates make it a mechanism:
  the wishbones and the rocker revolve on the chassis axes, ball joints and rod
  ends are spherical, the damper is two parts sliding on one axis and the wheel
  spins on the upright. Dragging the wheel moves the whole corner;
* the top assembly (``SUSP_2027``): the four corners (flexible), the rack
  housing and the centreline skeleton, fixed.

Every part is modelled in the assembly's coordinates (the chassis part's or ISO
8855), so each component goes in at the identity transform and each mate is
already satisfied when it is added: nothing moves while the macro runs.

The mate scheme is checked here before SolidWorks sees it (``mobility``): the
mates hold at the static geometry, none is redundant, and each corner keeps one
degree of freedom that moves its joints (heave) plus the idle spins of the
parts held by two points on their own axis. Steering is held straight ahead:
the tie rods' inner joints sit on the chassis part.

The macro checks what it builds and writes a log next to the files. Re-running
it rebuilds every file it made (close the assemblies that use them first):
edits made inside those files are lost. Placeholder diameters on the project's
points: a modelling aid, not a detailed part, a structural size or a choice.

Usage::

    uv run python scripts/export_suspension_assembly.py          # team project, chassis frame
    uv run python scripts/export_suspension_assembly.py --frame iso
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import export_solidworks_skeleton as swx  # noqa: E402
import export_suspension_solids as SS  # noqa: E402
import numpy as np  # noqa: E402

from apps.vdstudio.engine import exports as X  # noqa: E402
from apps.vdstudio.engine import geometry as G  # noqa: E402
from apps.vdstudio.engine import suspension as S  # noqa: E402
from apps.vdstudio.engine.project import Project, load_project  # noqa: E402

PREFIX = SS.PREFIX
CORNERS = SS.CORNERS
FRONT = ("FL", "FR")

# Half length of the construction lines that cross at a reference point, and the
# offsets of the second point / axis ends (mm). Construction only.
REF_CROSS_MM = 5.0
# The two crossing lines run along these unit directions of the CAD frame, skewed so that
# neither lies along a frame axis, where pivot axes often do (an overlap is ambiguous).
REF_DIRS = ((0.8, 0.48, 0.36), (-0.48, 0.6, 0.64))
PIVOT_B_MM = 20.0
AXIS_HALF_MM = 30.0
WHEEL_B_MM = 40.0

# Solids of one corner by component (name suffix after "SUSP_FL_"). The links,
# the rod and the rod's mount bracket depend on the corner and the rocker spec.
_GROUPS = {
    "UCA": ("UCA_front_leg", "UCA_rear_leg", "UCA_bush_front", "UCA_bush_rear"),
    "LCA": ("LCA_front_leg", "LCA_rear_leg", "LCA_bush_front", "LCA_bush_rear"),
    "upright": ("UCA_ball_joint", "LCA_ball_joint", "upright_kingpin", "upright_spindle",
                "upright_arm", "hub"),
    "wheel": ("tyre",),
    "rocker": ("rocker_boss", "rocker_arm_rod", "rocker_arm_damper", "rocker_edge",
               "rocker_bolt_rod", "rocker_bolt_damper"),
    "chassis_mounts": ("rocker_bracket_a", "rocker_bracket_b", "coilover_bracket_a",
                       "coilover_bracket_b"),
    "damper_body": ("coilover_eye_chassis", "coilover_body", "coilover_spring"),
    "damper_shaft": ("coilover_eye_rocker", "coilover_shaft"),
}
_MOUNT_GROUP = {"lca": "LCA", "uca": "UCA", "upright": "upright"}


@dataclass(frozen=True)
class Ref:
    """A named reference in a part: a point at ``a``, or an axis from ``a`` to ``b``."""

    name: str
    a: tuple[float, float, float]
    b: tuple[float, float, float] | None = None

    @property
    def is_axis(self) -> bool:
        return self.b is not None


@dataclass
class Component:
    """One part file: its solids, its references, whether it is fixed."""

    name: str
    corner: str | None
    bodies: list[SS.Cyl] = field(default_factory=list)
    refs: list[Ref] = field(default_factory=list)
    ground: bool = False
    sketches: dict = field(default_factory=dict)

    def ref(self, name: str) -> Ref:
        for r in self.refs:
            if r.name == name:
                return r
        raise KeyError(f"{self.name} has no reference {name!r}")


@dataclass(frozen=True)
class Mate:
    """Coincident ``child_ref`` (on ``child``) with ``parent_ref`` (on ``parent``)."""

    child: str
    child_ref: str
    parent: str
    parent_ref: str
    label: str


@dataclass(frozen=True)
class Model:
    components: dict[str, Component]
    mates: dict[str, list[Mate]]          # by corner assembly name
    corner_parts: dict[str, list[str]]    # corner assembly name -> its part names
    top_parts: list[str]                  # fixed parts of the top assembly
    asm_name: str


def _spec(project: Project, cid: str):
    return project.rocker_front if cid in FRONT else project.rocker_rear


def _link(cid: str) -> str:
    return "tie_rod" if cid in FRONT else "toe_link"


def group_of(name: str, cid: str, rod: str, mount_body: str) -> str:
    """The component (suffix after ``SUSP_<cid>_``) a corner solid belongs to."""
    suffix = name[len(f"{PREFIX}{cid}_"):]
    link = _link(cid)
    if suffix in (link, f"{link}_rod_end_in", f"{link}_rod_end_out"):
        return link
    if suffix == f"{rod}_mount_bracket":
        return _MOUNT_GROUP[mount_body]
    if suffix in (rod, f"{rod}_rod_end_mount", f"{rod}_rod_end_rocker"):
        return rod
    for g, members in _GROUPS.items():
        if suffix in members:
            return g
    raise ValueError(f"no component for solid {name!r}")


def _pt(name: str, p) -> Ref:
    return Ref(name, SS._t(p))


def _ax(name: str, a, b) -> Ref:
    return Ref(name, SS._t(a), SS._t(b))


def corner_refs(geo, project: Project, cid: str) -> dict[str, list[Ref]]:
    """Reference points and axes of each component of a corner (ISO, mm)."""
    P = {k: SS._v(v) for k, v in geo.points[cid].items()}
    spec = _spec(project, cid)
    lay = S.layout_for(spec, cid)
    a = SS._unit(lay.rocker_axis)
    o, m = SS._v(lay.rocker_pivot), SS._v(lay.rod_mount)
    rr, rd, dc = SS._v(lay.rocker_rod), SS._v(lay.rocker_damper), SS._v(lay.damper_chassis)
    wc, spin = P["WHEEL_CENTER"], SS.spin_axis(geo, cid)
    refs: dict[str, list[Ref]] = {
        "chassis_mounts": [
            _pt("P_UCA_IN_FRONT", P["UCA_IN_FRONT"]),
            _ax("A_UCA_PIVOT", P["UCA_IN_FRONT"], P["UCA_IN_REAR"]),
            _pt("P_LCA_IN_FRONT", P["LCA_IN_FRONT"]),
            _ax("A_LCA_PIVOT", P["LCA_IN_FRONT"], P["LCA_IN_REAR"]),
            _pt("P_TIE_ROD_IN", P["TIE_ROD_IN"]),
            _pt("P_ROCKER_PIVOT", o),
            _ax("A_ROCKER", o - AXIS_HALF_MM * a, o + AXIS_HALF_MM * a),
            _pt("P_DAMPER_CHASSIS", dc),
        ],
        "UCA": [_pt("P_IN_FRONT", P["UCA_IN_FRONT"]), _pt("P_IN_REAR", P["UCA_IN_REAR"]),
                _pt("P_OUT", P["UCA_OUT"])],
        "LCA": [_pt("P_IN_FRONT", P["LCA_IN_FRONT"]), _pt("P_IN_REAR", P["LCA_IN_REAR"]),
                _pt("P_OUT", P["LCA_OUT"])],
        "upright": [_pt("P_UBJ", P["UCA_OUT"]), _pt("P_LBJ", P["LCA_OUT"]),
                    _pt("P_TIE_ROD_OUT", P["TIE_ROD_OUT"]), _pt("P_WHEEL_CENTER", wc),
                    _ax("A_SPIN", wc - AXIS_HALF_MM * spin, wc + AXIS_HALF_MM * spin)],
        _link(cid): [_pt("P_IN", P["TIE_ROD_IN"]), _pt("P_OUT", P["TIE_ROD_OUT"])],
        spec.rod_type: [_pt("P_MOUNT", m), _pt("P_ROCKER", rr)],
        "rocker": [_pt("P_PIVOT", o), _pt("P_PIVOT_B", o + PIVOT_B_MM * a),
                   _pt("P_ROD", rr), _pt("P_DAMPER", rd)],
        "damper_body": [_pt("P_EYE", dc), _ax("A_AXIS", dc, rd)],
        "damper_shaft": [_pt("P_EYE", rd), _ax("A_AXIS", rd, dc)],
        "wheel": [_pt("P_CENTER", wc), _pt("P_CENTER_B", wc + WHEEL_B_MM * spin)],
    }
    refs[_MOUNT_GROUP[spec.mount_body]].append(_pt("P_ROD_MOUNT", m))
    return refs


def corner_mates(cid: str, rod: str, mount_body: str) -> list[Mate]:
    """The corner's joints as coincident mates (child first)."""
    def n(g: str) -> str:
        return f"{PREFIX}{cid}_{g}"

    ch, link, mount = n("chassis_mounts"), n(_link(cid)), n(_MOUNT_GROUP[mount_body])
    out = []
    for w in ("UCA", "LCA"):
        out += [Mate(n(w), "P_IN_FRONT", ch, f"P_{w}_IN_FRONT", f"{cid} {w} front pivot"),
                Mate(n(w), "P_IN_REAR", ch, f"A_{w}_PIVOT", f"{cid} {w} pivot axis")]
    return out + [
        Mate(n("upright"), "P_UBJ", n("UCA"), "P_OUT", f"{cid} upper ball joint"),
        Mate(n("upright"), "P_LBJ", n("LCA"), "P_OUT", f"{cid} lower ball joint"),
        Mate(link, "P_OUT", n("upright"), "P_TIE_ROD_OUT", f"{cid} {_link(cid)} outer"),
        Mate(link, "P_IN", ch, "P_TIE_ROD_IN", f"{cid} {_link(cid)} inner"),
        Mate(n("rocker"), "P_PIVOT", ch, "P_ROCKER_PIVOT", f"{cid} rocker pivot"),
        Mate(n("rocker"), "P_PIVOT_B", ch, "A_ROCKER", f"{cid} rocker axis"),
        Mate(n(rod), "P_MOUNT", mount, "P_ROD_MOUNT", f"{cid} {rod} lower end"),
        Mate(n(rod), "P_ROCKER", n("rocker"), "P_ROD", f"{cid} {rod} rocker end"),
        Mate(n("damper_body"), "P_EYE", ch, "P_DAMPER_CHASSIS", f"{cid} damper chassis eye"),
        Mate(n("damper_shaft"), "P_EYE", n("rocker"), "P_DAMPER", f"{cid} damper rocker eye"),
        Mate(n("damper_shaft"), "A_AXIS", n("damper_body"), "A_AXIS", f"{cid} damper slide"),
        Mate(n("wheel"), "P_CENTER", n("upright"), "P_WHEEL_CENTER", f"{cid} wheel centre"),
        Mate(n("wheel"), "P_CENTER_B", n("upright"), "A_SPIN", f"{cid} wheel spin axis"),
    ]


def build_model(project: Project, arb: dict | None = None,
                asm_name: str = "SUSP_2027") -> Model:
    if arb:
        # the arms and drop links would sit in one fixed part, torn off the moving rockers
        raise ValueError("an anti-roll bar is not jointed in the assembly export yet; "
                         "use export_suspension_solids.py --arb for the single part")
    geo = G.resolve(project)
    parts = SS.all_parts(project, arb)
    comps: dict[str, Component] = {}

    def comp(name: str, corner: str | None) -> Component:
        if name not in comps:
            comps[name] = Component(name, corner)
        return comps[name]

    mates: dict[str, list[Mate]] = {}
    corner_parts: dict[str, list[str]] = {}
    for cid in CORNERS:
        spec = _spec(project, cid)
        if not spec.enabled:
            raise ValueError(f"{cid}: the assembly needs the rocker layout (spec disabled)")
        for g, refs in corner_refs(geo, project, cid).items():
            comp(f"{PREFIX}{cid}_{g}", cid).refs.extend(refs)
        comps[f"{PREFIX}{cid}_chassis_mounts"].ground = True
        mates[f"{PREFIX}{cid}"] = corner_mates(cid, spec.rod_type, spec.mount_body)
        corner_parts[f"{PREFIX}{cid}"] = [c.name for c in comps.values() if c.corner == cid]
    for p in parts:
        cid = next((c for c in CORNERS if p.name.startswith(f"{PREFIX}{c}_")), None)
        if cid is not None:
            spec = _spec(project, cid)
            g = group_of(p.name, cid, spec.rod_type, spec.mount_body)
            comps[f"{PREFIX}{cid}_{g}"].bodies.append(p)
        elif p.name == f"{PREFIX}rack_housing":
            comp(f"{PREFIX}rack", None).bodies.append(p)
        else:
            raise ValueError(f"no component for solid {p.name!r}")
    skel = comp(f"{PREFIX}skeleton", None)
    skel.sketches = SS.skeleton_lines(project)
    top = [c.name for c in comps.values() if c.corner is None]
    for c in top:
        comps[c].ground = True
    return Model(comps, mates, corner_parts, top, asm_name)


# --------------------------------------------------------------------------- #
# The mate scheme, checked
# --------------------------------------------------------------------------- #

def ref_segments(comp: Component, frame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The construction lines the macro draws in a part's REF_SKETCH (CAD frame, mm)."""
    out = []
    h = REF_CROSS_MM
    for r in comp.refs:
        a = np.array(SS._m(r.a, frame)) * 1000.0
        if r.is_axis:
            out.append((r.name, a, np.array(SS._m(r.b, frame)) * 1000.0))
            continue
        for k, d in enumerate(REF_DIRS):
            dv = np.asarray(d)
            out.append((f"{r.name}#{k}", a - h * dv, a + h * dv))
    return out


def _rot(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _perp_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n1 = np.cross(d, t)
    n1 /= np.linalg.norm(n1)
    return n1, np.cross(d, n1)


@dataclass(frozen=True)
class Mobility:
    max_residual_mm: float   # largest mate violation at the static geometry
    rows: int                # scalar constraints the mates impose
    redundant: int           # constraints that repeat others (over-defined if > 0)
    dof: int                 # free motions, idle spins included
    joint_dof: int           # free motions that move any reference (heave: 1)
    heave_moves_wheel: bool
    motion_ratio: float      # -dL_damper / dz_wheel_centre of that motion, at ride


def mobility(model: Model, asm: str) -> Mobility:
    """Linearised mobility of one corner assembly under its mates (positions in m)."""
    mates = model.mates[asm]
    names = sorted({m.child for m in mates} | {m.parent for m in mates})
    unmated = set(model.corner_parts.get(asm, ())) - set(names)
    if unmated:
        raise ValueError(f"{asm}: parts with no mate would float: {sorted(unmated)}")
    moving = [n for n in names if not model.components[n].ground]
    col = {n: 6 * i for i, n in enumerate(moving)}
    nx = 6 * len(moving)

    def pose(x: np.ndarray, name: str):
        if name not in col:
            return np.eye(3), np.zeros(3)
        i = col[name]
        return _rot(x[i + 3:i + 6]), x[i:i + 3]

    def place(x, name, p):
        R, t = pose(x, name)
        return R @ (np.asarray(p, float) / 1000.0) + t

    def residual(x: np.ndarray) -> np.ndarray:
        out: list[float] = []
        for m in mates:
            ra = model.components[m.child].ref(m.child_ref)
            rb = model.components[m.parent].ref(m.parent_ref)
            pa = place(x, m.child, ra.a)
            if not rb.is_axis:
                if ra.is_axis:
                    raise ValueError(f"{m.label}: an axis on a point is not a joint here")
                out += list(pa - place(x, m.parent, rb.a))
                continue
            qb, qb2 = place(x, m.parent, rb.a), place(x, m.parent, rb.b)
            d0 = SS._unit(np.asarray(rb.b, float) - np.asarray(rb.a, float))
            n1, n2 = _perp_basis(d0)
            db = (qb2 - qb) / np.linalg.norm(qb2 - qb)
            e = np.cross(pa - qb, db)                      # point on the parent's line
            out += [float(e @ n1), float(e @ n2)]
            if ra.is_axis:                                 # and parallel to it
                pa2 = place(x, m.child, ra.b)
                da = (pa2 - pa) / np.linalg.norm(pa2 - pa)
                e = np.cross(da, db)
                out += [float(e @ n1), float(e @ n2)]
        return np.asarray(out)

    x0 = np.zeros(nx)
    r0 = residual(x0)
    eps = 1e-7
    J = np.empty((r0.size, nx))
    for j in range(nx):
        dx = np.zeros(nx)
        dx[j] = eps
        J[:, j] = (residual(x0 + dx) - residual(x0 - dx)) / (2.0 * eps)
    sv = np.linalg.svd(J, compute_uv=False)
    tol = 1e-7 * max(1.0, float(sv[0]))
    rank = int(np.sum(sv > tol))
    _, _, vt = np.linalg.svd(J)
    null = vt[rank:].T                                    # free motions (columns)
    # velocity of every reference point (and axis end) under each free motion
    pts = []
    for n in moving:
        for r in model.components[n].refs:
            pts += [(n, r.a)] + ([(n, r.b)] if r.is_axis else [])
    V = np.zeros((3 * len(pts), null.shape[1]))
    for k in range(null.shape[1]):
        v = null[:, k]
        for i, (n, p) in enumerate(pts):
            c = col[n]
            V[3 * i:3 * i + 3, k] = v[c:c + 3] + np.cross(v[c + 3:c + 6],
                                                          np.asarray(p, float) / 1000.0)
    sv_v = np.linalg.svd(V, compute_uv=False) if V.size else np.zeros(0)
    joint_dof = int(np.sum(sv_v > 1e-7 * max(1.0, float(sv_v[0]) if sv_v.size else 1.0)))
    def vel(name: str, ref: str) -> np.ndarray:
        r = model.components[name].ref(ref)
        if name not in col:
            return np.zeros((3, V.shape[1]))
        i = pts.index((name, r.a))
        return V[3 * i:3 * i + 3, :]

    wheel = next((n for n in moving if n.endswith("_wheel")), None)
    vz = vel(wheel, "P_CENTER")[2] if wheel else np.zeros(V.shape[1])
    moves = bool(np.any(np.abs(vz) > 1e-6))
    mr = float("nan")
    body = next((n for n in names if n.endswith("_damper_body")), None)
    shaft = next((n for n in names if n.endswith("_damper_shaft")), None)
    if moves and body and shaft:
        k = int(np.argmax(np.abs(vz)))                 # spins leave the references still
        dc = np.asarray(model.components[body].ref("P_EYE").a, float)
        rd = np.asarray(model.components[shaft].ref("P_EYE").a, float)
        u = (rd - dc) / np.linalg.norm(rd - dc)
        dl = u @ (vel(shaft, "P_EYE")[:, k] - vel(body, "P_EYE")[:, k])
        mr = float(-dl / vz[k])
    return Mobility(max_residual_mm=float(np.max(np.abs(r0))) * 1000.0 if r0.size else 0.0,
                    rows=int(r0.size), redundant=int(r0.size - rank), dof=nx - rank,
                    joint_dof=joint_dof, heave_moves_wheel=moves, motion_ratio=mr)


# --------------------------------------------------------------------------- #
# VBA
# --------------------------------------------------------------------------- #

def _vb(*vals: float) -> str:
    return ", ".join(SS._num(v) for v in vals)


def _sub_name(name: str) -> str:
    return name.replace(" ", "_")


def render_macro(model: Model, frame, source: str) -> str:
    """The VBA module. Number literals use '.', which VBA reads whatever the locale."""
    comps = model.components
    corner_asms = list(model.mates)
    files = ([f"{model.asm_name}.SLDASM"] + [f"{a}.SLDASM" for a in corner_asms]
             + [f"{c}.SLDPRT" for c in comps])
    n_bodies = sum(len(c.bodies) for c in comps.values())
    n_refs = sum(len(c.refs) for c in comps.values())
    n_mates = sum(len(v) for v in model.mates.values())
    L: list[str] = []
    add = L.append
    add(f"Attribute VB_Name = \"{model.asm_name}_build\"")
    add("' FSAE 2027 suspension -- SolidWorks assembly: one part per component, mated to move")
    add(f"' Generated by scripts/export_suspension_assembly.py from {source}")
    add(f"' Frame: {frame.label}. API units are metres.")
    add("' Placeholder diameters on the project's points: a modelling aid, not a detailed")
    add("' or structural design. Do not hand-edit: change the project and regenerate.")
    add("'")
    add("' Tools > Macro > New (save a blank .swp), then in the VBA editor")
    add(f"' File > Import File... > this .bas. Open the module {model.asm_name}_build, click "
        "inside")
    add("' Sub main and press F5 (the .swp's own module has an empty main). It asks for a "
        "folder,")
    add(f"' writes {len(files)} files there ({len(comps)} parts, {len(corner_asms)} corner "
        "assemblies and the top one)")
    add(f"' and a log, {model.asm_name}_log.txt. Re-running replaces those files: close the")
    add("' assemblies that use them first.")
    add("")
    add("Option Explicit")
    add("")
    add("Const ASM_NAME As String = \"" + model.asm_name + "\"")
    add(f"Const N_BODIES As Long = {n_bodies}")
    add(f"Const N_REFS As Long = {n_refs}")
    add(f"Const N_MATES As Long = {n_mates}")
    add("")
    add("Dim swApp As Object")
    add("Dim swModel As Object")
    add("Dim swModeler As Object")
    add("Dim gCentred As Boolean")
    add("Dim gMade As Long")
    add("Dim gFailed As Long")
    add("Dim gFailList As String")
    add("Dim gDir As String")
    add("Dim gCur As String")
    add("Dim gLog As String")
    add("Dim gNames As String")
    add("Dim gAsmTitle As String")
    add("Dim gComps As Collection")
    add("Dim gRefMade As Long")
    add("Dim gRefFailed As Long")
    add("Dim gMateMade As Long")
    add("Dim gMateFailed As Long")
    add("Dim gOtherFailed As Long")
    add("Dim gSketch As Object")
    add("' reference geometry queued while a part's REF_SKETCH is open")
    add("Dim gRefN As Long")
    add("Dim gRefName(1 To 64) As String")
    add("Dim gRefAxis(1 To 64) As Boolean")
    add("Dim gRefX(1 To 64) As Double")
    add("Dim gRefY(1 To 64) As Double")
    add("Dim gRefZ(1 To 64) As Double")
    add("Dim gSegA(1 To 64) As Object")
    add("Dim gSegB(1 To 64) As Object")
    add("Dim gSegNameA(1 To 64) As String")
    add("Dim gSegNameB(1 To 64) As String")
    add("")
    add("Private Sub LogLine(ByVal s As String)")
    add("    gLog = gLog & s & vbCrLf")
    add("End Sub")
    add("")
    add("Private Sub WriteLog()")
    add("    Dim fnum As Integer")
    add("    On Error Resume Next")
    add("    fnum = FreeFile")
    add("    Open gDir & ASM_NAME & \"_log.txt\" For Output As #fnum")
    add("    Print #fnum, gLog")
    add("    Close #fnum")
    add("End Sub")
    add("")
    add("' the files this macro writes, as |NAME.EXT| in upper case")
    add("Private Sub InitNames()")
    add("    gNames = \"|\"")
    for f in files:
        add(f"    gNames = gNames & \"{f.upper()}|\"")
    add("End Sub")
    add("")
    add("Private Function FileOf(ByVal p As String) As String")
    add("    FileOf = Mid(p, InStrRev(p, \"\\\") + 1)")
    add("End Function")
    add("")
    add("' Close every open document with a name this macro writes (assemblies first, so")
    add("' that their parts are no longer in use). Returns the title of one still open.")
    add("Private Function CloseOurs() As String")
    add("    Dim doc As Object, asms As Collection, prts As Collection, t As Variant")
    add("    Dim fn As String, pass As Long")
    add("    For pass = 1 To 3")
    add("        Set asms = New Collection")
    add("        Set prts = New Collection")
    add("        Set doc = swApp.GetFirstDocument")
    add("        Do While Not doc Is Nothing")
    add("            fn = FileOf(doc.GetPathName)")
    add("            If Len(fn) > 0 Then")
    add("                If InStr(1, gNames, \"|\" & UCase(fn) & \"|\") > 0 Then")
    add("                    If doc.GetType = 2 Then")
    add("                        asms.Add doc.GetTitle")
    add("                    Else")
    add("                        prts.Add doc.GetTitle")
    add("                    End If")
    add("                End If")
    add("            End If")
    add("            Set doc = doc.GetNext")
    add("        Loop")
    add("        CloseOurs = \"\"")
    add("        If asms.Count + prts.Count = 0 Then Exit Function")
    add("        If asms.Count > 0 Then CloseOurs = asms(1) Else CloseOurs = prts(1)")
    add("        If pass < 3 Then")
    add("            For Each t In asms")
    add("                swApp.CloseDoc t")
    add("            Next")
    add("            For Each t In prts")
    add("                swApp.CloseDoc t")
    add("            Next")
    add("        End If")
    add("    Next")
    add("End Function")
    add("")
    add("' Close this macro's documents left open by a previous run, delete its old files.")
    add("Private Function ClearOld() As Boolean")
    add("    Dim still As String")
    add("    still = CloseOurs()")
    add("    If Len(still) > 0 Then")
    add("        MsgBox \"Still open (an open assembly uses it): \" & still & vbCrLf & _")
    add("               \"Close the assemblies that use the suspension files and run again.\"")
    add("        Exit Function")
    add("    End If")
    add("    Dim fileList As Variant, i As Long, p As String")
    add("    fileList = Split(Mid(gNames, 2, Len(gNames) - 2), \"|\")")
    add("    For i = LBound(fileList) To UBound(fileList)")
    add("        p = gDir & fileList(i)")
    add("        If Len(Dir(p)) > 0 Then")
    add("            On Error Resume Next")
    add("            SetAttr p, vbNormal")
    add("            Kill p")
    add("            On Error GoTo 0")
    add("            If Len(Dir(p)) > 0 Then")
    add("                MsgBox \"Could not replace \" & p & \" (open in another program?)\"")
    add("                Exit Function")
    add("            End If")
    add("        End If")
    add("    Next")
    add("    ClearOld = True")
    add("End Function")
    add("")
    L.extend(SS.vba_calibrate())
    add("")
    L.extend(SS.vba_cylinder())
    add("")
    L.extend(_vba_part_helpers())
    add("")
    L.extend(_vba_assembly_helpers())
    # ------------------------------------------------------------------ parts
    add("")
    add("Private Sub BuildParts()")
    for c in comps:
        add(f"    Part_{_sub_name(c)}")
    add("End Sub")
    for c in comps.values():
        col_refs = c.refs
        add("")
        add(f"Private Sub Part_{_sub_name(c.name)}()")
        add(f"    If Not BeginPart(\"{c.name}\") Then Exit Sub")
        for p in c.bodies:
            col = SS.COLOURS.get(p.family, (0.6, 0.6, 0.6))
            a, b = SS._m(p.a, frame), SS._m(p.b, frame)
            add(f"    Cyl \"{p.name}\", "
                + _vb(*a, *b, p.od_mm / 1000.0, p.hollow_id_mm / 1000.0, *col))
        for sk_name, lines in c.sketches.items():
            add("    SketchOpen")
            for p, q in lines:
                add("    SketchLine " + _vb(*SS._m(p, frame), *SS._m(q, frame)))
            add(f"    SketchClose \"{sk_name}\"")
        if col_refs:
            add("    RefOpen")
            for r in col_refs:
                if r.is_axis:
                    add(f"    RefAxis \"{r.name}\", "
                        + _vb(*SS._m(r.a, frame), *SS._m(r.b, frame)))
                else:
                    add(f"    RefPoint \"{r.name}\", " + _vb(*SS._m(r.a, frame)))
        add(f"    EndPart \"{c.name}\"")
        add("End Sub")
    # ------------------------------------------------------- corner assemblies
    add("")
    add("Private Sub BuildCorners()")
    for a in corner_asms:
        add(f"    Asm_{_sub_name(a)}")
    add("End Sub")
    for a in corner_asms:
        add("")
        add(f"Private Sub Asm_{_sub_name(a)}()")
        add(f"    If Not BeginAsm(\"{a}\") Then Exit Sub")
        for c in model.corner_parts[a]:
            add(f"    AddComp \"{c}\", \".SLDPRT\", {'True' if comps[c].ground else 'False'}")
        for m in model.mates[a]:
            add(f"    MateRef \"{m.child}\", \"{m.child_ref}\", \"{m.parent}\", "
                f"\"{m.parent_ref}\", \"{m.label}\"")
        add("    EndAsm")
        add("End Sub")
    # ------------------------------------------------------------ top assembly
    add("")
    add("Private Sub BuildTop()")
    add("    If Not BeginAsm(ASM_NAME) Then Exit Sub")
    for c in model.top_parts:
        add(f"    AddComp \"{c}\", \".SLDPRT\", True")
    for a in corner_asms:
        add(f"    AddComp \"{a}\", \".SLDASM\", True")
        add(f"    MakeFlexible \"{a}\"")
    add("    EndAsm")
    add("End Sub")
    add("")
    # main comes last: Tools > Macro > Run starts the last argument-less Sub of the most
    # recently created module (the imported one)
    add("Sub main()")
    add("    Set swApp = Application.SldWorks")
    add("    Set swModeler = swApp.GetModeler")
    add("    Dim d As String")
    add("    On Error Resume Next")
    add("    d = swApp.GetCurrentMacroPathFolder")
    add("    On Error GoTo 0")
    add("    If Len(d) > 0 And Right(d, 1) <> \"\\\" Then d = d & \"\\\"")
    add("    d = InputBox(\"Folder for the suspension files (created if missing). \" & _")
    add("                 \"Files with the names this macro writes are replaced.\", _")
    add("                 ASM_NAME, d & ASM_NAME & \"\\\")")
    add("    If Len(d) = 0 Then Exit Sub")
    add("    If Right(d, 1) <> \"\\\" Then d = d & \"\\\"")
    add("    gDir = d")
    add("    If Len(Dir(gDir, vbDirectory)) = 0 Then")
    add("        On Error Resume Next")
    add("        MkDir gDir")
    add("        On Error GoTo 0")
    add("    End If")
    add("    If Len(Dir(gDir, vbDirectory)) = 0 Then")
    add("        MsgBox \"Could not create the folder \" & gDir")
    add("        Exit Sub")
    add("    End If")
    add("    InitNames")
    add("    If Not ClearOld() Then Exit Sub")
    add("    gLog = ASM_NAME & \" build log\" & vbCrLf & \"folder: \" & gDir & vbCrLf")
    add("    gMade = 0: gFailed = 0: gFailList = \"\"")
    add("    gRefMade = 0: gRefFailed = 0: gMateMade = 0: gMateFailed = 0: gOtherFailed = 0")
    add("    Calibrate")
    add("    LogLine \"CreateBodyFromCyl point is the centre: \" & gCentred")
    add("    ' parts are built without windows. Components go in with both flags on, as in")
    add("    ' the top assembly that worked: built hidden, the corners came out empty")
    add("    swApp.DocumentVisible False, 1")
    add("    swApp.DocumentVisible True, 2")
    add("    On Error GoTo stopped")
    add("    BuildParts")
    add("    swApp.DocumentVisible True, 1")
    add("    BuildCorners")
    add("    BuildTop")
    add("    ReopenTop")
    add("    On Error GoTo 0")
    add("    Dim msg As String")
    add("    msg = ASM_NAME & \": \" & gMade & \" of \" & N_BODIES & \" solids, \" & _")
    add("          gRefMade & \" of \" & N_REFS & \" reference points / axes, \" & _")
    add("          gMateMade & \" of \" & N_MATES & \" mates.\"")
    add("    LogLine msg")
    add("    WriteLog")
    add("    If gOtherFailed > 0 Then msg = msg & vbCrLf & gOtherFailed & _")
    add("        \" add / placement / flexible / save step(s) failed.\"")
    add("    If gFailed + gRefFailed + gMateFailed + gOtherFailed > 0 Then")
    add("        msg = msg & vbCrLf & \"Some steps failed: see \" & gDir & ASM_NAME & _")
    add("              \"_log.txt\"")
    add("    End If")
    add("    MsgBox msg")
    add("    Exit Sub")
    add("stopped:")
    add("    Dim why As String")
    add("    why = Err.Description")
    add("    On Error Resume Next")
    add("    swApp.DocumentVisible True, 1")
    add("    swApp.DocumentVisible True, 2")
    add("    LogLine \"STOPPED in \" & gCur & \": \" & why")
    add("    WriteLog")
    add("    MsgBox \"Stopped in \" & gCur & \": \" & why & vbCrLf & \"Log: \" & gDir & ASM_NAME "
        "& \"_log.txt\"")
    add("End Sub")
    add("")
    return "\r\n".join(L)


def _vba_part_helpers() -> list[str]:
    return [
        "Private Function NewDoc(ByVal kind As Long) As Object",
        "    ' kind 8: default part template, 9: default assembly template",
        "    Dim tmpl As String",
        "    tmpl = swApp.GetUserPreferenceStringValue(kind)",
        "    Set NewDoc = swApp.NewDocument(tmpl, 0, 0, 0)",
        "    If NewDoc Is Nothing Then",
        "        LogLine \"FAIL new document from template '\" & tmpl & \"'\"",
        "    End If",
        "End Function",
        "",
        "Private Function BeginPart(ByVal nm As String) As Boolean",
        "    gCur = nm",
        "    Set swModel = NewDoc(8)",
        "    If swModel Is Nothing Then Exit Function",
        "    gRefN = 0",
        "    BeginPart = True",
        "End Function",
        "",
        "Private Sub EndPart(ByVal nm As String)",
        "    BuildRefs",
        "    swModel.ForceRebuild3 False",
        "    SaveAsFile nm & \".SLDPRT\"",
        "End Sub",
        "",
        "Private Function SaveAsFile(ByVal fn As String) As Boolean",
        "    Dim e As Long, w As Long",
        "    On Error Resume Next",
        "    SaveAsFile = swModel.Extension.SaveAs(gDir & fn, 0, 1, Nothing, e, w)",
        "    On Error GoTo 0",
        "    If SaveAsFile Then",
        "        LogLine \"saved \" & fn",
        "    Else",
        "        gOtherFailed = gOtherFailed + 1",
        "        LogLine \"FAIL save \" & fn & \" (error \" & e & \")\"",
        "    End If",
        "End Function",
        "",
        "Private Function NewestFeature(ByVal n0 As Long) As Object",
        "    If swModel.GetFeatureCount > n0 Then Set NewestFeature = "
        "swModel.FeatureByPositionReverse(0)",
        "End Function",
        "",
        "' a centreline 3D sketch (the skeleton part)",
        "Private Sub SketchOpen()",
        "    swModel.ClearSelection2 True",
        "    swModel.SketchManager.AddToDB = True",
        "    swModel.SketchManager.DisplayWhenAdded = False",
        "    swModel.SketchManager.Insert3DSketch True",
        "End Sub",
        "",
        "Private Sub SketchLine(ByVal x1 As Double, ByVal y1 As Double, ByVal z1 As Double, _",
        "                      ByVal x2 As Double, ByVal y2 As Double, ByVal z2 As Double)",
        "    swModel.SketchManager.CreateLine x1, y1, z1, x2, y2, z2",
        "End Sub",
        "",
        "Private Sub SketchClose(ByVal nm As String)",
        "    Dim f As Object",
        "    swModel.SketchManager.Insert3DSketch True",
        "    swModel.SketchManager.AddToDB = False",
        "    swModel.SketchManager.DisplayWhenAdded = True",
        "    Set f = swModel.FeatureByPositionReverse(0)",
        "    If Not f Is Nothing Then f.Name = nm",
        "End Sub",
        "",
        "' Reference geometry: construction lines in one 3D sketch (REF_SKETCH), then a",
        "' reference axis on a line, or a reference point where two short lines cross.",
        "Private Sub RefOpen()",
        "    SketchOpen",
        "    gRefN = 0",
        "End Sub",
        "",
        "Private Function SegName(seg As Object) As String",
        "    On Error Resume Next",
        "    SegName = seg.GetName",
        "End Function",
        "",
        "Private Sub RefPoint(ByVal nm As String, ByVal x As Double, ByVal y As Double, _",
        "                    ByVal z As Double)",
        f"    Const h As Double = {SS._num(REF_CROSS_MM / 1000.0)}",
        "    gRefN = gRefN + 1",
        "    gRefName(gRefN) = nm: gRefAxis(gRefN) = False",
        "    gRefX(gRefN) = x: gRefY(gRefN) = y: gRefZ(gRefN) = z",
        "    Set gSegA(gRefN) = swModel.SketchManager.CreateLine( _",
        f"        x - h * {REF_DIRS[0][0]}, y - h * {REF_DIRS[0][1]}, z - h * {REF_DIRS[0][2]}, _",
        f"        x + h * {REF_DIRS[0][0]}, y + h * {REF_DIRS[0][1]}, z + h * {REF_DIRS[0][2]})",
        "    Set gSegB(gRefN) = swModel.SketchManager.CreateLine( _",
        f"        x - h * {REF_DIRS[1][0]}, y - h * {REF_DIRS[1][1]}, z - h * {REF_DIRS[1][2]}, _",
        f"        x + h * {REF_DIRS[1][0]}, y + h * {REF_DIRS[1][1]}, z + h * {REF_DIRS[1][2]})",
        "    gSegNameA(gRefN) = SegName(gSegA(gRefN))",
        "    gSegNameB(gRefN) = SegName(gSegB(gRefN))",
        "End Sub",
        "",
        "Private Sub RefAxis(ByVal nm As String, ByVal x1 As Double, ByVal y1 As Double, _",
        "                   ByVal z1 As Double, ByVal x2 As Double, ByVal y2 As Double, _",
        "                   ByVal z2 As Double)",
        "    gRefN = gRefN + 1",
        "    gRefName(gRefN) = nm: gRefAxis(gRefN) = True",
        "    gRefX(gRefN) = x1: gRefY(gRefN) = y1: gRefZ(gRefN) = z1",
        "    Set gSegA(gRefN) = swModel.SketchManager.CreateLine(x1, y1, z1, x2, y2, z2)",
        "    Set gSegB(gRefN) = Nothing",
        "    gSegNameA(gRefN) = SegName(gSegA(gRefN))",
        "    gSegNameB(gRefN) = \"\"",
        "End Sub",
        "",
        "Private Sub BuildRefs()",
        "    If gRefN = 0 Then Exit Sub",
        "    Dim i As Long, sk As String",
        "    swModel.SketchManager.Insert3DSketch True",
        "    swModel.SketchManager.AddToDB = False",
        "    swModel.SketchManager.DisplayWhenAdded = True",
        "    Set gSketch = swModel.FeatureByPositionReverse(0)",
        "    If Not gSketch Is Nothing Then",
        "        gSketch.Name = \"REF_SKETCH\"",
        "        sk = gSketch.Name",
        "    End If",
        "    For i = 1 To gRefN",
        "        If gRefAxis(i) Then",
        "            MakeAxis i, sk",
        "        Else",
        "            MakePoint i, sk",
        "        End If",
        "    Next",
        "    ' hide the construction sketch",
        "    swModel.ClearSelection2 True",
        "    If Not gSketch Is Nothing Then",
        "        If gSketch.Select2(False, 0) Then swModel.BlankSketch",
        "    End If",
        "    swModel.ClearSelection2 True",
        "    gRefN = 0",
        "End Sub",
        "",
        "' Select a construction line: the object first, its name in the sketch if that fails.",
        "Private Function SelSeg(seg As Object, ByVal segName As String, ByVal sk As String, _",
        "                        ByVal app As Boolean) As Boolean",
        "    Dim want As Long, ok As Boolean",
        "    want = 1",
        "    If app Then want = swModel.SelectionManager.GetSelectedObjectCount2(-1) + 1",
        "    On Error Resume Next",
        "    If Not seg Is Nothing Then ok = seg.Select4(app, Nothing)",
        "    If Err.Number <> 0 Then ok = False",
        "    Err.Clear",
        "    If ok Then ok = (swModel.SelectionManager.GetSelectedObjectCount2(-1) = want)",
        "    If Not ok And Len(segName) > 0 And Len(sk) > 0 Then",
        "        ok = swModel.Extension.SelectByID2(segName & \"@\" & sk, \"EXTSKETCHSEGMENT\", _",
        "                                           0, 0, 0, app, 0, Nothing, 0)",
        "    End If",
        "    On Error GoTo 0",
        "    SelSeg = ok",
        "End Function",
        "",
        "Private Sub RefFail(ByVal i As Long, ByVal why As String)",
        "    gRefFailed = gRefFailed + 1",
        "    LogLine \"FAIL reference \" & gRefName(i) & \" in \" & gCur & \": \" & why",
        "    swModel.ClearSelection2 True",
        "End Sub",
        "",
        "' 1: the point is where it should be; 0: it is not; -1: it could not be read",
        "Private Function PointCheck(f As Object, ByVal x As Double, ByVal y As Double, _",
        "                            ByVal z As Double) As Long",
        "    Dim rp As Object, mp As Object, d As Variant",
        "    PointCheck = -1",
        "    On Error Resume Next",
        "    Set rp = f.GetSpecificFeature2",
        "    If rp Is Nothing Then Exit Function",
        "    Set mp = rp.GetRefPoint",
        "    If mp Is Nothing Then Exit Function",
        "    d = mp.ArrayData",
        "    If Not IsArray(d) Then Exit Function",
        "    If Abs(d(0) - x) + Abs(d(1) - y) + Abs(d(2) - z) < 0.00001 Then",
        "        PointCheck = 1",
        "    Else",
        "        PointCheck = 0",
        "    End If",
        "End Function",
        "",
        "Private Sub MakePoint(ByVal i As Long, ByVal sk As String)",
        "    Dim n0 As Long, f As Object, v As Variant",
        "    swModel.ClearSelection2 True",
        "    If Not SelSeg(gSegA(i), gSegNameA(i), sk, False) Then",
        "        RefFail i, \"could not select its construction lines\"",
        "        Exit Sub",
        "    End If",
        "    If Not SelSeg(gSegB(i), gSegNameB(i), sk, True) Then",
        "        RefFail i, \"could not select its construction lines\"",
        "        Exit Sub",
        "    End If",
        "    n0 = swModel.GetFeatureCount",
        "    On Error Resume Next",
        "    ' 6 = swRefPointIntersection: where the two selected lines cross",
        "    v = swModel.FeatureManager.InsertReferencePoint(6, 0, 0#, 1)",
        "    On Error GoTo 0",
        "    Set f = NewestFeature(n0)",
        "    If f Is Nothing Then",
        "        RefFail i, \"InsertReferencePoint made no feature\"",
        "        Exit Sub",
        "    End If",
        "    f.Name = gRefName(i)",
        "    Select Case PointCheck(f, gRefX(i), gRefY(i), gRefZ(i))",
        "        Case 1",
        "            gRefMade = gRefMade + 1",
        "        Case 0",
        "            swModel.ClearSelection2 True",
        "            If f.Select2(False, 0) Then swModel.EditDelete",
        "            RefFail i, \"made in the wrong place (deleted)\"",
        "        Case Else",
        "            gRefMade = gRefMade + 1",
        "            LogLine \"unverified reference point \" & gRefName(i) & \" in \" & gCur",
        "    End Select",
        "    swModel.ClearSelection2 True",
        "End Sub",
        "",
        "Private Sub MakeAxis(ByVal i As Long, ByVal sk As String)",
        "    Dim n0 As Long, f As Object, ok As Boolean",
        "    swModel.ClearSelection2 True",
        "    If Not SelSeg(gSegA(i), gSegNameA(i), sk, False) Then",
        "        RefFail i, \"could not select its construction line\"",
        "        Exit Sub",
        "    End If",
        "    n0 = swModel.GetFeatureCount",
        "    On Error Resume Next",
        "    ok = swModel.InsertAxis2(True)",
        "    On Error GoTo 0",
        "    Set f = NewestFeature(n0)",
        "    If f Is Nothing Then",
        "        RefFail i, \"InsertAxis2 made no feature\"",
        "        Exit Sub",
        "    End If",
        "    f.Name = gRefName(i)",
        "    gRefMade = gRefMade + 1",
        "    swModel.ClearSelection2 True",
        "End Sub",
    ]


def _vba_assembly_helpers() -> list[str]:
    return [
        "Private Function BeginAsm(ByVal nm As String) As Boolean",
        "    gCur = nm",
        "    Set swModel = NewDoc(9)",
        "    If swModel Is Nothing Then Exit Function",
        "    Set gComps = New Collection",
        "    ' saved first: the fallback selection names end in @<assembly name>",
        "    If Not SaveAsFile(nm & \".SLDASM\") Then Exit Function",
        "    gAsmTitle = nm",
        "    ActivateAsm",
        "    BeginAsm = True",
        "End Function",
        "",
        "' AddComponent works on the active, visible assembly.",
        "Private Sub ActivateAsm()",
        "    Dim e As Long, d As Object",
        "    On Error Resume Next",
        "    Set d = swApp.ActivateDoc3(swModel.GetTitle, False, 1, e)",
        "    If Not d Is Nothing Then Set swModel = d",
        "End Sub",
        "",
        "Private Function ActiveTitle() As String",
        "    On Error Resume Next",
        "    ActiveTitle = swApp.ActiveDoc.GetTitle",
        "End Function",
        "",
        "' AddComponent needs the file loaded in memory. The parts are (built in this run);",
        "' a part that is not is opened without a window, then the assembly is active again.",
        "Private Sub LoadDoc(ByVal p As String, ByVal ext As String)",
        "    Dim e As Long, w As Long, t As Long, d As Object",
        "    On Error Resume Next",
        "    Set d = swApp.GetOpenDocumentByName(p)",
        "    If Not d Is Nothing Then Exit Sub",
        "    t = 1",
        "    If ext = \".SLDASM\" Then t = 2",
        "    Set d = swApp.OpenDoc6(p, t, 1, \"\", e, w)",
        "    On Error GoTo 0",
        "    LogLine \"loaded \" & FileOf(p) & \" from disk (error \" & e & \")\"",
        "    ActivateAsm",
        "End Sub",
        "",
        "' The same insertion through the list method, at the identity.",
        "Private Function AddByList(ByVal p As String) As Object",
        "    Dim nms(0) As String, tr(15) As Double, cs(0) As String",
        "    Dim vn As Variant, vt As Variant, vc As Variant, res As Variant",
        "    nms(0) = p",
        "    tr(0) = 1: tr(4) = 1: tr(8) = 1: tr(12) = 1",
        "    cs(0) = \"\"",
        "    vn = nms: vt = tr: vc = cs",
        "    On Error Resume Next",
        "    res = swModel.AddComponents3((vn), (vt), (vc))",
        "    If IsArray(res) Then Set AddByList = res(LBound(res))",
        "End Function",
        "",
        "' Every part was modelled in place and every mate holds there: a component that is",
        "' not at the identity after the mates was placed or moved wrongly.",
        "Private Sub CheckPlaced()",
        "    Dim c As Variant, r As Variant",
        "    For Each c In gComps",
        "        r = Empty",
        "        On Error Resume Next",
        "        r = c.Transform2.ArrayData",
        "        On Error GoTo 0",
        "        If IsArray(r) Then",
        "            If Abs(r(9)) + Abs(r(10)) + Abs(r(11)) > 0.000001 Or _",
        "               Abs(r(0) - 1) + Abs(r(4) - 1) + Abs(r(8) - 1) > 0.000001 Then",
        "                gOtherFailed = gOtherFailed + 1",
        "                LogLine \"FAIL \" & c.Name2 & \" is not where it was modelled in \" & _",
        "                    gCur",
        "            End If",
        "        End If",
        "    Next",
        "End Sub",
        "",
        "Private Sub EndAsm()",
        "    Dim e As Long, w As Long, ok As Boolean",
        "    swModel.ClearSelection2 True",
        "    swModel.ForceRebuild3 False",
        "    CheckPlaced",
        "    On Error Resume Next",
        "    ok = swModel.Save3(1, e, w)",
        "    On Error GoTo 0",
        "    If Not ok Then",
        "        gOtherFailed = gOtherFailed + 1",
        "        LogLine \"FAIL save \" & gCur & \" (error \" & e & \")\"",
        "    End If",
        "End Sub",
        "",
        "' Every part is modelled in the assembly's coordinates: it goes in at the identity.",
        "Private Sub AddComp(ByVal nm As String, ByVal ext As String, ByVal fixIt As Boolean)",
        "    Dim c As Object, p As String, why As String",
        "    p = gDir & nm & ext",
        "    LoadDoc p, ext",
        "    On Error Resume Next",
        "    Set c = swModel.AddComponent5(p, 0, \"\", False, \"\", 0, 0, 0)",
        "    If c Is Nothing Then",
        "        If Err.Number <> 0 Then why = \"; AddComponent5: \" & Err.Description",
        "        Err.Clear",
        "        Set c = swModel.AddComponent4(p, \"\", 0, 0, 0)",
        "    End If",
        "    If c Is Nothing Then",
        "        If Err.Number <> 0 Then why = why & \"; AddComponent4: \" & Err.Description",
        "        Err.Clear",
        "        Set c = AddByList(p)",
        "    End If",
        "    On Error GoTo 0",
        "    If c Is Nothing Then",
        "        gOtherFailed = gOtherFailed + 1",
        "        LogLine \"FAIL add component \" & nm & \" to \" & gCur & \" (file found: \" & _",
        "            (Len(Dir(p)) > 0) & \"; active document: \" & ActiveTitle() & why & \")\"",
        "        Exit Sub",
        "    End If",
        "    PlaceAtOrigin c, nm",
        "    gComps.Add c, nm",
        "    If fixIt Then",
        "        swModel.ClearSelection2 True",
        "        If c.Select4(False, Nothing, False) Then swModel.FixComponent",
        "        swModel.ClearSelection2 True",
        "    End If",
        "End Sub",
        "",
        "' A method first (no Let / Set question on a late-bound property), then the",
        "' property both ways; read back and counted when it is not the identity.",
        "Private Function PlaceAtOrigin(c As Object, ByVal nm As String) As Boolean",
        "    Dim d(15) As Double, v As Variant, t As Object, r As Variant, ok As Boolean",
        "    d(0) = 1: d(4) = 1: d(8) = 1: d(12) = 1",
        "    v = d",
        "    On Error Resume Next",
        "    Set t = swApp.GetMathUtility.CreateTransform((v))",
        "    ok = c.SetTransformAndSolve3(t, False)",
        "    If Err.Number <> 0 Or Not ok Then",
        "        Err.Clear",
        "        ok = c.SetTransformAndSolve2(t)",
        "    End If",
        "    If Err.Number <> 0 Or Not ok Then",
        "        Err.Clear",
        "        c.Transform2 = t",
        "        If Err.Number <> 0 Then",
        "            Err.Clear",
        "            Set c.Transform2 = t",
        "        End If",
        "    End If",
        "    Err.Clear",
        "    r = c.Transform2.ArrayData",
        "    If Err.Number = 0 And IsArray(r) Then",
        "        PlaceAtOrigin = (Abs(r(9)) + Abs(r(10)) + Abs(r(11)) < 0.000001) And _",
        "                        (Abs(r(0) - 1) + Abs(r(4) - 1) + Abs(r(8) - 1) < 0.000001)",
        "    End If",
        "    On Error GoTo 0",
        "    If Not PlaceAtOrigin Then",
        "        gOtherFailed = gOtherFailed + 1",
        "        LogLine \"FAIL place \" & nm & \" at the origin of \" & gCur",
        "    End If",
        "End Function",
        "",
        "' Component2.Solving only reads the state: set it on the selected component.",
        "Private Sub MakeFlexible(ByVal nm As String)",
        "    Dim c As Object, ok As Boolean, st As Long",
        "    On Error Resume Next",
        "    Set c = gComps(nm)",
        "    If c Is Nothing Then Exit Sub",
        "    swModel.ClearSelection2 True",
        "    If c.Select4(False, Nothing, False) Then",
        "        ' suppression 2 = fully resolved, solving 1 = flexible",
        "        ok = swModel.CompConfigProperties5(2, 1, True, False, \"\", False, False)",
        "        If Err.Number <> 0 Then",
        "            Err.Clear",
        "            ok = swModel.CompConfigProperties4(2, 1, True, False, \"\", False)",
        "        End If",
        "    End If",
        "    Err.Clear",
        "    st = -1",
        "    st = c.Solving",
        "    swModel.ClearSelection2 True",
        "    On Error GoTo 0",
        "    If st <> 1 Then",
        "        gOtherFailed = gOtherFailed + 1",
        "        LogLine \"FAIL make \" & nm & \" flexible (right-click it > \" & _",
        "            \"Make Subassembly Flexible)\"",
        "    End If",
        "End Sub",
        "",
        "' The parts were made without windows (such documents cannot be shown later):",
        "' close everything this macro made and open the top assembly again.",
        "Private Sub ReopenTop()",
        "    Dim e As Long, w As Long, doc As Object, still As String",
        "    gCur = ASM_NAME",
        "    still = CloseOurs()",
        "    If Len(still) > 0 Then LogLine \"WARN still open after closing: \" & still",
        "    On Error Resume Next",
        "    Set doc = swApp.OpenDoc6(gDir & ASM_NAME & \".SLDASM\", 2, 1, \"\", e, w)",
        "    On Error GoTo 0",
        "    If doc Is Nothing Then",
        "        gOtherFailed = gOtherFailed + 1",
        "        LogLine \"FAIL reopen \" & ASM_NAME & \" (error \" & e & \")\"",
        "        Exit Sub",
        "    End If",
        "    Set swModel = doc",
        "    On Error Resume Next",
        "    swModel.ShowNamedView2 \"*Isometric\", 7",
        "    swModel.ViewZoomtofit2",
        "End Sub",
        "",
        "' Select a component's reference: its feature in the assembly, or its name path.",
        "Private Function SelRef(ByVal cn As String, ByVal rn As String, _",
        "                        ByVal app As Boolean) As Boolean",
        "    Dim c As Object, f As Object, want As Long, ok As Boolean, typ As String",
        "    On Error Resume Next",
        "    Set c = gComps(cn)",
        "    On Error GoTo 0",
        "    If c Is Nothing Then Exit Function",
        "    want = 1",
        "    If app Then want = swModel.SelectionManager.GetSelectedObjectCount2(-1) + 1",
        "    On Error Resume Next",
        "    Set f = c.FeatureByName(rn)",
        "    If Not f Is Nothing Then ok = f.Select2(app, 1)",
        "    If Err.Number <> 0 Then ok = False",
        "    Err.Clear",
        "    If ok Then ok = (swModel.SelectionManager.GetSelectedObjectCount2(-1) = want)",
        "    If Not ok Then",
        "        typ = \"DATUMPOINT\"",
        "        If Left(rn, 2) = \"A_\" Then typ = \"AXIS\"",
        "        ok = swModel.Extension.SelectByID2(rn & \"@\" & c.Name2 & \"@\" & gAsmTitle, _",
        "                                           typ, _",
        "                                           0, 0, 0, app, 1, Nothing, 0)",
        "    End If",
        "    On Error GoTo 0",
        "    SelRef = ok",
        "End Function",
        "",
        "Private Function TryMate(ByVal ca As String, ByVal ra As String, ByVal cb As String, _",
        "                         ByVal rb As String, ByVal kind As Long, e As Long) As Object",
        "    swModel.ClearSelection2 True",
        "    e = -1",
        "    If Not SelRef(ca, ra, False) Then Exit Function",
        "    e = -2",
        "    If Not SelRef(cb, rb, True) Then Exit Function",
        "    e = 0",
        "    On Error Resume Next",
        "    ' kind 0 coincident, 1 concentric; align 2 = closest",
        "    Set TryMate = swModel.AddMate5(kind, 2, False, 0, 0, 0, 1, 1, 0, 0, 0, False, _",
        "                                   False, 0, e)",
        "    If Err.Number <> 0 Then",
        "        Err.Clear",
        "        Set TryMate = swModel.AddMate3(kind, 2, False, 0, 0, 0, 1, 1, 0, 0, 0, False, e)",
        "    End If",
        "    On Error GoTo 0",
        "    swModel.ClearSelection2 True",
        "End Function",
        "",
        "Private Sub MateRef(ByVal ca As String, ByVal ra As String, ByVal cb As String, _",
        "                    ByVal rb As String, ByVal label As String)",
        "    Dim m As Object, e As Long",
        "    Set m = TryMate(ca, ra, cb, rb, 0, e)",
        "    If m Is Nothing And Left(ra, 2) = \"A_\" And Left(rb, 2) = \"A_\" Then",
        "        Set m = TryMate(ca, ra, cb, rb, 1, e)",
        "    End If",
        "    If m Is Nothing Then",
        "        gMateFailed = gMateFailed + 1",
        "        LogLine \"FAIL mate \" & label & \": \" & ra & \"@\" & ca & \" to \" & rb & _",
        "            \"@\" & cb & _",
        "            \" (\" & MateWhy(e) & \")\"",
        "    Else",
        "        gMateMade = gMateMade + 1",
        "    End If",
        "End Sub",
        "",
        "Private Function MateWhy(ByVal e As Long) As String",
        "    Select Case e",
        "        Case -1: MateWhy = \"first reference not selected\"",
        "        Case -2: MateWhy = \"second reference not selected\"",
        "        Case Else: MateWhy = \"swAddMateError \" & e",
        "    End Select",
        "End Function",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--project", type=Path, default=REPO / "projects" / "FSAE_2027.vdproj.json")
    ap.add_argument("--frame", choices=("chassis", "iso"), default="chassis",
                    help="chassis: the chassis part's coordinates (Xasis 33.3); iso: ISO 8855")
    ap.add_argument("--out", type=Path, default=REPO / "projects" / "cad", help="output folder")
    ap.add_argument("--name", default="suspension_assembly")
    ap.add_argument("--asm-name", default="SUSP_2027", help="top assembly file name")
    args = ap.parse_args()
    project = load_project(args.project)
    model = build_model(project, None, args.asm_name)
    for asm in model.mates:
        mob = mobility(model, asm)
        print(f"{asm}: {len(model.corner_parts[asm])} parts, {mob.rows} mate constraints, "
              f"max violation {mob.max_residual_mm:.2e} mm, redundant {mob.redundant}, "
              f"free motions {mob.dof} ({mob.joint_dof} moving the joints), "
              f"damper/wheel ratio of that motion {mob.motion_ratio:.3f}")
        if mob.max_residual_mm > 1e-6 or mob.redundant or mob.joint_dof != 1:
            raise SystemExit(f"{asm}: the mate scheme is not a one-DOF mechanism")
    frame = X.cad_frame(project) if args.frame == "chassis" else swx.resolve_frame("iso")
    text = render_macro(model, frame, args.project.name)
    tag = "Xasis_33.3" if args.frame == "chassis" else "iso"
    path = args.out / f"{args.name}_{tag}.bas"
    path.write_bytes(text.encode("cp1252", errors="replace"))
    print(f"wrote {path} ({len(model.components)} parts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
