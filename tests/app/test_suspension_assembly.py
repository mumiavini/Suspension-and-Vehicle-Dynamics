"""scripts/export_suspension_assembly.py: one part per component, mated to move."""

from __future__ import annotations

import dataclasses
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from vdcore.analysis.rocker import rocker_sweep

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def sa():
    sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "export_suspension_assembly", REPO / "scripts" / "export_suspension_assembly.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["export_suspension_assembly"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def project(sa):
    return sa.load_project(REPO / "projects" / "FSAE_2027.vdproj.json")


@pytest.fixture(scope="module")
def model(sa, project):
    return sa.build_model(project)


def test_every_solid_is_in_exactly_one_part(sa, project, model) -> None:
    solids = [p.name for p in sa.SS.all_parts(project)]
    placed = [b.name for c in model.components.values() for b in c.bodies]
    assert sorted(placed) == sorted(solids)
    for cid in sa.CORNERS:
        asm = f"{sa.PREFIX}{cid}"
        link = "tie_rod" if cid in ("FL", "FR") else "toe_link"
        want = {f"{asm}_{g}" for g in ("chassis_mounts", "UCA", "LCA", "upright", link,
                                       "pushrod", "rocker", "damper_body", "damper_shaft",
                                       "wheel")}
        assert set(model.corner_parts[asm]) == want
        # the rear rod sits on the LCA: its bracket goes with it (the front rod is on the
        # kingpin axis, no bracket)
        if cid in ("RL", "RR"):
            names = [b.name for b in model.components[f"{asm}_LCA"].bodies]
            assert f"{asm}_pushrod_mount_bracket" in names
    assert all(model.components[c].ground for c in model.top_parts)
    assert [c for c in model.components.values() if c.ground and c.corner] == \
        [model.components[f"{sa.PREFIX}{cid}_chassis_mounts"] for cid in sa.CORNERS]


def test_mates_name_existing_references(model) -> None:
    for asm, mates in model.mates.items():
        assert len(mates) == 17
        for m in mates:
            assert m.child in model.corner_parts[asm] and m.parent in model.corner_parts[asm]
            model.components[m.child].ref(m.child_ref)
            model.components[m.parent].ref(m.parent_ref)
        for c in model.corner_parts[asm]:
            names = [r.name for r in model.components[c].refs]
            assert len(names) == len(set(names))
            assert all(re.fullmatch(r"[PA]_[A-Z_]+", n) for n in names)


def test_each_corner_is_a_one_dof_mechanism_with_the_project_motion_ratio(sa, project,
                                                                           model) -> None:
    """The mates hold at the static geometry, none is redundant (SolidWorks would call
    the corner over-defined), and only heave moves the joints. Its damper / wheel ratio
    is the one vdcore computes for the project's rocker."""
    geo = sa.G.resolve(project)
    for cid in sa.CORNERS:
        mob = sa.mobility(model, f"{sa.PREFIX}{cid}")
        assert mob.max_residual_mm < 1e-9
        assert mob.rows == 48 and mob.redundant == 0
        # heave, plus the idle spins of the tie rod, the rod, the damper (two) and the wheel
        assert mob.dof == 6 and mob.joint_dof == 1 and mob.heave_moves_wheel
        spec = project.rocker_front if cid in ("FL", "FR") else project.rocker_rear
        sw = rocker_sweep(geo.corner(cid), sa.S.layout_for(spec, cid), [-0.5, 0.0, 0.5])
        assert mob.motion_ratio == pytest.approx(float(sw.motion_ratio[1]), abs=2e-4)


def test_mobility_sees_a_missing_and_a_repeated_mate(sa, model) -> None:
    asm = f"{sa.PREFIX}FL"
    mates = model.mates[asm]
    no_tie_rod = [m for m in mates if "tie_rod" not in m.label]
    mob = sa.mobility(dataclasses.replace(model, mates={asm: no_tie_rod}), asm)
    assert mob.redundant == 0 and mob.joint_dof == 2       # heave, and the wheel steers
    axis = next(m for m in mates if m.label.endswith("UCA pivot axis"))
    doubled = sa.mobility(dataclasses.replace(model, mates={asm: mates + [axis]}), asm)
    assert doubled.redundant == 2 and doubled.max_residual_mm < 1e-9
    wrong = sa.Mate(f"{asm}_UCA", "P_IN_REAR", f"{asm}_chassis_mounts", "P_UCA_IN_FRONT", "x")
    bad = sa.mobility(dataclasses.replace(model, mates={asm: mates + [wrong]}), asm)
    assert bad.max_residual_mm > 1.0               # a mate that does not hold shows


def _statements(text: str) -> list[str]:
    """VBA logical lines: continuations joined, comments and string bodies blanked."""
    out, buf = [], ""
    for raw in text.split("\r\n"):
        line = re.sub(r'"[^"]*"', '""', raw)
        line = line.split("'", 1)[0].rstrip()
        if line.endswith(" _"):
            buf += line[:-2] + " "
            continue
        out.append((buf + line).strip())
        buf = ""
    return [s for s in out if s]


_KEYWORDS = {"If", "Else", "ElseIf", "End", "For", "Next", "Do", "Loop", "Select", "Case",
             "Dim", "Set", "Exit", "On", "Private", "Sub", "Function", "Const", "Option",
             "Open", "Print", "Close", "Kill", "MkDir", "SetAttr", "MsgBox", "GoTo"}


def test_macro_text_is_balanced_and_calls_only_defined_procedures(sa, project, model) -> None:
    frame = sa.X.cad_frame(project)
    text = sa.render_macro(model, frame, "test")
    assert "\r\n" in text and text.isascii()
    st = _statements(text)
    defined = {m.group(2) for s in st
               if (m := re.match(r"(Private )?(?:Sub|Function) (\w+)", s))}
    stack: list[str] = []
    pairs = {"End Sub": "Sub", "End Function": "Function", "End If": "If",
             "Next": "For", "Loop": "Do", "End Select": "Select"}
    for s in st:
        head = s.split()[0]
        if re.match(r"(Private )?(Sub|Function) ", s):
            assert not stack, f"procedure opened inside {stack}: {s}"
            stack.append("Sub" if " Sub " in f" {s} " else "Function")
        elif re.match(r"If .* Then$", s):
            stack.append("If")
        elif re.match(r"For (Each )?\w", s):
            stack.append("For")
        elif re.match(r"Do( |$)", s):
            stack.append("Do")
        elif s.startswith("Select Case"):
            stack.append("Select")
        else:
            for close, opener in pairs.items():
                if s == close or (close == "Next" and re.fullmatch(r"Next( \w+)?", s)) \
                        or (close == "Loop" and s.startswith("Loop")):
                    assert stack and stack[-1] == opener, f"{s!r} closes {stack}"
                    stack.pop()
                    break
            else:
                # a bare call statement names a procedure of this module
                if (re.fullmatch(r"[A-Z]\w*", head) and head not in _KEYWORDS
                        and not re.match(r"\w+ *=", s) and ":" not in head):
                    assert head in defined, f"call to undefined {head!r}: {s}"
    assert not stack
    for c in model.components:
        assert f"Part_{c}" in defined
    for asm in model.mates:
        assert f"Asm_{asm}" in defined
    assert len(re.findall(r"^    MateRef ", text, flags=re.M)) == 4 * 17
    assert len(re.findall(r'^    Cyl "', text, flags=re.M)) == len(sa.SS.all_parts(project))
    n_refs = sum(len(c.refs) for c in model.components.values())
    assert len(re.findall(r"^    Ref(Point|Axis) ", text, flags=re.M)) == n_refs
    assert f"Const N_REFS As Long = {n_refs}" in text
    for proc in re.split(r"\r\nEnd (?:Sub|Function)", text):
        assert len(proc) < 60_000
    assert all(len(line) < 1000 for line in text.split("\r\n"))


def test_parts_sit_in_the_chassis_frame(sa, project, model) -> None:
    """In the chassis frame the LCA's front pivot reference is on the CAD frame node,
    like the single-part macro's solids."""
    import json
    frame = sa.X.cad_frame(project)
    raw = json.loads((REPO / "projects" / "cad" / "Xasis 33.3_survey.json").read_text("utf-8"))
    nodes = np.array([pt for sk in raw["sketches"] if sk["name"] == "PSD"
                      for ln in sk["lines"] for pt in (ln["a"], ln["b"])])
    r = model.components[f"{sa.PREFIX}FL_chassis_mounts"].ref("P_LCA_IN_FRONT")
    d = np.min(np.linalg.norm(nodes - np.array(sa.SS._m(r.a, frame)) * 1000.0, axis=1))
    assert d < 0.1
