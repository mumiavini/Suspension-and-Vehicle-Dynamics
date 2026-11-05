r"""
Build a suspension assembly in SolidWorks from one of the generated
`suspension_?_assembly_*.bas` macros, without the VBA editor.

The .bas is read as data (it is generated: every part is a list of Cyl / RefPoint /
RefAxis / SketchLine calls, every corner a list of AddComp / MateRef calls) and replayed
here through COM, with two deliberate differences from the macro:

1. Every cylinder is a NORMAL, editable feature instead of a dumb imported body:
   a 3D sketch with the centre line (<name>_eixo) and a circular-profile sweep (<name>)
   whose diameter is an editable dimension. Tubes get a second circular sweep cut
   (<name>_furo). Move a sketch end point or change a diameter and rebuild.
2. The generated damper (body + shaft cylinders) is replaced by the real damper,
   Amortecedor.SLDPRT, split in two copies that slide on the damper axis:
   AMORTECEDOR_CORPO (body, reservoir, top eye, spring) and AMORTECEDOR_HASTE
   (bottom eye, lower spring seat, rod). The original file is not modified.

3. --clock turns a corner's real damper (body and shaft together) about its own axis at
   placement: angle 0 has the eye pins parallel to the rocker axis and the reservoir toward
   the part's -X; the angle is the turn travel_check.Car.clock / clock_sweep.py apply. The
   damper's spin about its axis is free in the mates (a spherical chassis eye), so the
   placement is what holds the clock.

Usage:
    python -I build_susp.py <macro.bas> <output folder> <Amortecedor.SLDPRT> [--clock FL=deg FR=deg ...]
"""
import math
import os
import re
import shutil
import sys
import time

import pythoncom
import win32com.client as wc
from win32com.client import VARIANT

NOTHING = VARIANT(pythoncom.VT_DISPATCH, None)


def C(o, name, *a):
    """Call a method on a late-bound COM object (pywin32 would read a no-arg method as a property)."""
    o._FlagAsMethod(name)
    return getattr(o, name)(*a)


def byref_long():
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)


def dbl_array(vals):
    return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, [float(v) for v in vals])


# ----------------------------------------------------------------------------- parse
def parse_bas(path):
    txt = open(path, encoding="latin-1").read().replace(" _\r\n", " ").replace(" _\n", " ")
    subs = {}
    for m in re.finditer(r"^(?:Private )?Sub (\w+)\(\)\s*$(.*?)^End Sub", txt, re.S | re.M):
        subs[m.group(1)] = [ln.strip() for ln in m.group(2).splitlines() if ln.strip()
                            and not ln.strip().startswith("'")]
    consts = dict(re.findall(r'^Const (\w+) As \w+ = "?([^"\r\n]+)"?', txt, re.M))
    return subs, consts


TOKEN = re.compile(r'\s*(?:"([^"]*)"|([^,]+))')


def parse_args(s):
    out = []
    for q, raw in TOKEN.findall(s):
        if q or raw.strip() == '""':
            out.append(q)
        else:
            raw = raw.strip()
            try:
                out.append(float(raw))
            except ValueError:
                out.append({"True": True, "False": False}.get(raw, raw))
    return out


# ----------------------------------------------------------------------------- builder
class Builder:
    def __init__(self, bas, out_dir, damper_src, clocks=None):
        self.clocks = clocks or {}
        self.subs, self.consts = parse_bas(bas)
        self.asm_name = self.consts["ASM_NAME"]
        self.dir = os.path.abspath(out_dir) + "\\"
        self.damper_src = damper_src
        self.sw = wc.GetActiveObject("SldWorks.Application")
        self.math = C(self.sw, "GetMathUtility")
        self.log = [f"{self.asm_name} build log (build_susp.py from {os.path.basename(bas)})",
                    f"folder: {self.dir}"]
        self.n = dict(solids=0, solids_fail=0, refs=0, refs_fail=0, mates=0, mates_fail=0, other_fail=0)
        self.part_refs = {}      # part name -> {ref name: (kind, coords)}
        self.written = []        # file names this run writes
        self.model = None
        self.cur = ""

    # ------------------------------------------------------------------ helpers
    def L(self, s):
        self.log.append(s)
        print(s, flush=True)

    def new_doc(self, kind):
        tmpl = C(self.sw, "GetUserPreferenceStringValue", kind)
        d = C(self.sw, "NewDocument", tmpl, 0, 0, 0)
        if d is None:
            self.L(f"FAIL new document from template '{tmpl}'")
        return d

    def save_as(self, fn):
        e, w = byref_long(), byref_long()
        ok = C(self.model.Extension, "SaveAs", self.dir + fn, 0, 1, NOTHING, e, w)
        if ok:
            self.written.append(fn.upper())
            self.L(f"saved {fn}")
        else:
            self.n["other_fail"] += 1
            self.L(f"FAIL save {fn} (error {e.value})")
        return ok

    def newest(self, n0):
        if C(self.model, "GetFeatureCount") > n0:
            return C(self.model, "FeatureByPositionReverse", 0)
        return None

    def clear(self):
        C(self.model, "ClearSelection2", True)

    def hide_sketch(self, feat):
        self.clear()
        if feat is not None and C(feat, "Select2", False, 0):
            C(self.model, "BlankSketch")
        self.clear()

    def color(self, feat, rgb):
        vals = [rgb[0], rgb[1], rgb[2], 0.5, 0.7, 0.3, 0.3, 0, 0]
        try:
            C(feat, "SetMaterialPropertyValues2", dbl_array(vals), 1, NOTHING)
        except pythoncom.com_error:
            pass

    # ------------------------------------------------------------------ parts
    def line_sketch(self, p1, p2, name):
        sm = self.model.SketchManager
        self.clear()
        sm.AddToDB = True
        sm.DisplayWhenAdded = False
        C(sm, "Insert3DSketch", True)
        seg = C(sm, "CreateLine", *p1, *p2)
        C(sm, "Insert3DSketch", True)
        sm.AddToDB = False
        sm.DisplayWhenAdded = True
        f = C(self.model, "FeatureByPositionReverse", 0)
        if seg is None or f is None:
            return None
        f.Name = name
        return f

    def circ_sweep(self, path_feat, dia, cut, name):
        self.clear()
        fm = self.model.FeatureManager
        data = C(fm, "CreateDefinition", 18 if cut else 17)  # swFmSweepCut / swFmSweep
        data.CircularProfile = True
        data.CircularProfileDiameter = dia
        if not C(path_feat, "Select2", False, 4):
            return None
        f = C(fm, "CreateFeature", data)
        self.clear()
        if f is not None:
            f.Name = name
        return f

    def cyl(self, nm, x1, y1, z1, x2, y2, z2, od, idia, cr, cg, cb):
        """Editable cylinder (tube when idia > 0) between two points."""
        p1, p2 = (x1, y1, z1), (x2, y2, z2)
        Ln = math.dist(p1, p2)
        ok = False
        if Ln > 1e-6 and od > 0:
            path = self.line_sketch(p1, p2, nm + "_eixo")
            f = self.circ_sweep(path, od, False, nm) if path is not None else None
            if f is not None:
                ok = True
                self.color(f, (cr, cg, cb))
                self.hide_sketch(path)
                if idia > 0:
                    # the bore runs 1 mm past both ends, as in the macro
                    u = [(b - a) / Ln * 0.001 for a, b in zip(p1, p2)]
                    q1 = tuple(a - d for a, d in zip(p1, u))
                    q2 = tuple(b + d for b, d in zip(p2, u))
                    hp = self.line_sketch(q1, q2, nm + "_eixo_furo")
                    hc = self.circ_sweep(hp, idia, True, nm + "_furo") if hp is not None else None
                    if hc is None:
                        self.L(f"WARN bore of {nm} in {self.cur} not cut (left solid)")
                    self.hide_sketch(hp)
        if ok:
            self.n["solids"] += 1
        else:
            self.n["solids_fail"] += 1
            self.L(f"FAIL solid {nm} in {self.cur}")

    # reference geometry, ported from the macro (construction lines in REF_SKETCH)
    def ref_open(self):
        sm = self.model.SketchManager
        self.clear()
        sm.AddToDB = True
        sm.DisplayWhenAdded = False
        C(sm, "Insert3DSketch", True)
        self.refq = []

    def ref_point(self, nm, x, y, z):
        h = 0.005
        sm = self.model.SketchManager
        a = C(sm, "CreateLine", x - h * 0.8, y - h * 0.48, z - h * 0.36, x + h * 0.8, y + h * 0.48, z + h * 0.36)
        b = C(sm, "CreateLine", x + h * 0.48, y - h * 0.6, z - h * 0.64, x - h * 0.48, y + h * 0.6, z + h * 0.64)
        self.refq.append(("P", nm, (x, y, z), a, b))
        self.part_refs.setdefault(self.cur, {})[nm] = ("P", (x, y, z))

    def ref_axis(self, nm, x1, y1, z1, x2, y2, z2):
        a = C(self.model.SketchManager, "CreateLine", x1, y1, z1, x2, y2, z2)
        self.refq.append(("A", nm, (x1, y1, z1), a, None))
        self.part_refs.setdefault(self.cur, {})[nm] = ("A", (x1, y1, z1, x2, y2, z2))

    def sel_seg(self, seg, app):
        sel = self.model.SelectionManager
        want = C(sel, "GetSelectedObjectCount2", -1) + 1 if app else 1
        try:
            ok = seg is not None and C(seg, "Select4", app, NOTHING)
        except pythoncom.com_error:
            ok = False
        return bool(ok) and C(sel, "GetSelectedObjectCount2", -1) == want

    def ref_fail(self, nm, why):
        self.n["refs_fail"] += 1
        self.L(f"FAIL reference {nm} in {self.cur}: {why}")
        self.clear()

    def build_refs(self):
        if not getattr(self, "refq", None):
            return
        sm = self.model.SketchManager
        C(sm, "Insert3DSketch", True)
        sm.AddToDB = False
        sm.DisplayWhenAdded = True
        sk = C(self.model, "FeatureByPositionReverse", 0)
        if sk is not None:
            sk.Name = "REF_SKETCH"
        for kind, nm, pt, a, b in self.refq:
            self.clear()
            if kind == "P":
                if not (self.sel_seg(a, False) and self.sel_seg(b, True)):
                    self.ref_fail(nm, "could not select its construction lines")
                    continue
                n0 = C(self.model, "GetFeatureCount")
                try:
                    C(self.model.FeatureManager, "InsertReferencePoint", 6, 0, 0.0, 1)
                except pythoncom.com_error:
                    pass
                f = self.newest(n0)
                if f is None:
                    self.ref_fail(nm, "InsertReferencePoint made no feature")
                    continue
                f.Name = nm
                try:
                    d = C(C(f, "GetSpecificFeature2"), "GetRefPoint").ArrayData
                    if sum(abs(u - v) for u, v in zip(d, pt)) > 1e-5:
                        self.clear()
                        if C(f, "Select2", False, 0):
                            C(self.model, "EditDelete")
                        self.ref_fail(nm, "made in the wrong place (deleted)")
                        continue
                except pythoncom.com_error:
                    self.L(f"unverified reference point {nm} in {self.cur}")
                self.n["refs"] += 1
            else:
                if not self.sel_seg(a, False):
                    self.ref_fail(nm, "could not select its construction line")
                    continue
                n0 = C(self.model, "GetFeatureCount")
                try:
                    C(self.model, "InsertAxis2", True)
                except pythoncom.com_error:
                    pass
                f = self.newest(n0)
                if f is None:
                    self.ref_fail(nm, "InsertAxis2 made no feature")
                    continue
                f.Name = nm
                self.n["refs"] += 1
        self.hide_sketch(sk)
        self.refq = []

    def sketch_open(self):
        self.ref_open()

    def sketch_line(self, *p):
        C(self.model.SketchManager, "CreateLine", *p)

    def sketch_close(self, nm):
        sm = self.model.SketchManager
        C(sm, "Insert3DSketch", True)
        sm.AddToDB = False
        sm.DisplayWhenAdded = True
        f = C(self.model, "FeatureByPositionReverse", 0)
        if f is not None:
            f.Name = nm
        self.refq = []

    def begin_part(self, nm):
        self.cur = nm
        self.model = self.new_doc(8)
        self.refq = []
        return self.model is not None

    def end_part(self, nm):
        self.build_refs()
        C(self.model, "ForceRebuild3", False)
        self.save_as(nm + ".SLDPRT")

    # ------------------------------------------------------------------ real damper
    def plane_offset_from_top(self, dist_m):
        """Reference plane parallel to the Top plane (the damper axis is Y)."""
        planes = []
        f = C(self.model, "FirstFeature")
        while f is not None and len(planes) < 3:
            if C(f, "GetTypeName2") == "RefPlane":
                planes.append(f)
            f = C(f, "GetNextFeature")
        top = planes[1]
        self.clear()
        C(top, "Select2", False, 0)
        n0 = C(self.model, "GetFeatureCount")
        C(self.model.FeatureManager, "InsertRefPlane", 8, abs(dist_m), 0, 0, 0, 0)
        p = self.newest(n0)
        if p is None:
            raise RuntimeError("could not create the offset plane")
        # make sure it sits at +dist (the offset can go to either side)
        y = C(p, "GetSpecificFeature2").Transform.ArrayData[10]
        if abs(y - dist_m) > 1e-6:
            d = C(p, "GetDefinition")
            d.ReversedReferenceFlip = not d.ReversedReferenceFlip
            C(p, "ModifyDefinition", d, self.model, NOTHING)
        return p

    def sketch_on(self, plane, draw):
        self.clear()
        C(plane, "Select2", False, 0)
        C(self.model.SketchManager, "InsertSketch", True)
        draw(self.model.SketchManager)
        C(self.model.SketchManager, "InsertSketch", True)
        return C(self.model, "FeatureByPositionReverse", 0)

    def extrude(self, sketch, cut, through, depth, reverse):
        self.clear()
        C(sketch, "Select2", False, 0)
        fm = self.model.FeatureManager
        t1 = 1 if through else 0
        if cut:
            return C(fm, "FeatureCut4", True, False, reverse, t1, 0, depth, depth, False, False, False, False,
                     0.0, 0.0, False, False, False, False, False, False, True, False, False, False, 0, 0.0,
                     False, False)
        return C(fm, "FeatureExtrusion3", True, False, reverse, t1, 0, depth, depth, False, False, False, False,
                 0.0, 0.0, False, False, False, False, True, True, True, 0, 0.0, False)

    def main_body_box(self):
        bodies = C(self.model, "GetBodies2", 0, False) or []
        boxes = [C(b, "GetBodyBox") for b in bodies]
        return [min(b[i] for b in boxes) for i in range(3)] + [max(b[i] for b in boxes) for i in range(3, 6)]

    def volume(self):
        return C(self.model.Extension, "CreateMassProperty").Volume

    def try_both(self, make, check, what):
        """Build a feature; if it went the wrong way, delete it and build it reversed."""
        for rev in (False, True):
            f = make(rev)
            if f is not None and check():
                return f
            if f is not None:
                self.clear()
                C(f, "Select2", False, 0)
                C(self.model, "EditDelete")
        raise RuntimeError(f"damper copy: {what} failed both ways")

    def make_damper_copies(self):
        src = self.damper_src
        e, w = byref_long(), byref_long()
        orig = C(self.sw, "GetOpenDocumentByName", src)
        if orig is not None:
            C(self.sw, "CloseDoc", C(orig, "GetTitle"))
        for fn in ("AMORTECEDOR_CORPO.SLDPRT", "AMORTECEDOR_HASTE.SLDPRT"):
            shutil.copyfile(src, self.dir + fn)
            os.chmod(self.dir + fn, 0o666)
        # ---- body side: everything from the body upward (y >= 95 mm) plus the spring
        self.cur = "AMORTECEDOR_CORPO"
        self.model = C(self.sw, "OpenDoc6", self.dir + "AMORTECEDOR_CORPO.SLDPRT", 1, 1, "", e, w)
        v0 = self.volume()
        p40 = self.plane_offset_from_top(0.040)
        p40.Name = "CORTE_HASTE_PLANO"
        sk = self.sketch_on(p40, lambda sm: C(sm, "CreateCornerRectangle", -0.12, -0.12, 0, 0.12, 0.12, 0))
        sk.Name = "CORTE_HASTE_ESBOCO"
        self.try_both(lambda r: self.extrude(sk, True, True, 0.0, r),
                      lambda: abs(self.main_body_box()[1] - 0.040) < 1e-4, "cut below the spring seat")
        p95 = self.plane_offset_from_top(0.095)
        p95.Name = "CORTE_HASTE_PLANO2"
        sk2 = self.sketch_on(p95, lambda sm: C(sm, "CreateCircleByRadius", 0, 0, 0, 0.015))
        sk2.Name = "CORTE_HASTE_ESBOCO2"
        v1 = self.volume()
        rod = math.pi * 0.006 ** 2 * 0.055
        self.try_both(lambda r: self.extrude(sk2, True, False, 0.055, r),
                      lambda: abs((v1 - self.volume()) - rod) < rod * 0.05, "rod removal")
        self.damper_refs(eye_y=0.220)
        C(self.model, "ForceRebuild3", False)
        C(self.model, "Save3", 1, e, w)
        self.L(f"AMORTECEDOR_CORPO: body side of {os.path.basename(src)} "
               f"(volume {v0*1e9:.0f} -> {self.volume()*1e9:.0f} mm3)")
        self.written.append("AMORTECEDOR_CORPO.SLDPRT")
        C(self.sw, "CloseDoc", C(self.model, "GetTitle"))
        # ---- shaft side: bottom eye, lower seat and rod (y <= 95 mm), rod continued inside the body
        self.cur = "AMORTECEDOR_HASTE"
        self.model = C(self.sw, "OpenDoc6", self.dir + "AMORTECEDOR_HASTE.SLDPRT", 1, 1, "", e, w)
        bodies = C(self.model, "GetBodies2", 0, False)
        spring = [b for b in bodies if C(b, "GetBodyBox")[1] > 0.03]  # the spring starts at y=40
        if len(spring) == 1 and len(bodies) == 2:
            self.clear()
            C(spring[0], "Select2", False, NOTHING)
            C(self.model.FeatureManager, "InsertDeleteBody2", False)
            f = C(self.model, "FeatureByPositionReverse", 0)
            f.Name = "REMOVE_MOLA"
        else:
            self.L("WARN shaft copy: could not tell the spring apart; left in")
        p95 = self.plane_offset_from_top(0.095)
        p95.Name = "CORTE_CORPO_PLANO"
        sk = self.sketch_on(p95, lambda sm: C(sm, "CreateCornerRectangle", -0.12, -0.12, 0, 0.12, 0.12, 0))
        sk.Name = "CORTE_CORPO_ESBOCO"
        self.try_both(lambda r: self.extrude(sk, True, True, 0.0, r),
                      lambda: abs(self.main_body_box()[4] - 0.095) < 1e-4, "cut above the rod")
        p95b = self.plane_offset_from_top(0.095)
        p95b.Name = "HASTE_INTERNA_PLANO"
        sk2 = self.sketch_on(p95b, lambda sm: C(sm, "CreateCircleByRadius", 0, 0, 0, 0.006))
        sk2.Name = "HASTE_INTERNA_ESBOCO"
        self.try_both(lambda r: self.extrude(sk2, False, False, 0.055, r),
                      lambda: abs(self.main_body_box()[4] - 0.150) < 1e-4, "rod inside the body")
        C(self.model, "FeatureByPositionReverse", 0).Name = "HASTE_INTERNA"
        self.damper_refs(eye_y=0.0)
        C(self.model, "ForceRebuild3", False)
        C(self.model, "Save3", 1, e, w)
        self.L(f"AMORTECEDOR_HASTE: shaft side (volume {self.volume()*1e9:.0f} mm3)")
        self.written.append("AMORTECEDOR_HASTE.SLDPRT")
        C(self.sw, "CloseDoc", C(self.model, "GetTitle"))

    def damper_refs(self, eye_y):
        self.ref_open()
        self.ref_point("P_EYE", 0.0, eye_y, 0.0)
        self.ref_axis("A_AXIS", 0.0, 0.0, 0.0, 0.0, 0.220, 0.0)
        self.ref_axis("A_PIN", 0.0, eye_y, -0.012, 0.0, eye_y, 0.012)
        self.build_refs()

    # ------------------------------------------------------------------ assemblies
    def begin_asm(self, nm):
        self.cur = nm
        self.model = self.new_doc(9)
        if self.model is None:
            return False
        self.comps = {}
        self.free_comps = set()
        if not self.save_as(nm + ".SLDASM"):
            return False
        self.asm_title = nm
        self.activate()
        return True

    def activate(self):
        e = byref_long()
        d = C(self.sw, "ActivateDoc3", C(self.model, "GetTitle"), False, 1, e)
        if d is not None:
            self.model = d

    def load(self, p, ext):
        if C(self.sw, "GetOpenDocumentByName", p) is not None:
            return
        e, w = byref_long(), byref_long()
        C(self.sw, "OpenDoc6", p, 2 if ext == ".SLDASM" else 1, 1, "", e, w)
        self.L(f"loaded {os.path.basename(p)} from disk (error {e.value})")
        self.activate()

    def identity(self):
        return C(self.math, "CreateTransform", dbl_array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]))

    def add_comp(self, nm, ext, fix, file_nm=None, transform=None):
        p = self.dir + (file_nm or nm) + ext
        self.load(p, ext)
        c = C(self.model, "AddComponent5", p, 0, "", False, "", 0, 0, 0)
        if c is None:
            self.n["other_fail"] += 1
            self.L(f"FAIL add component {nm} to {self.cur}")
            return None
        t = transform if transform is not None else self.identity()
        try:
            C(c, "SetTransformAndSolve3", t, False)
        except pythoncom.com_error:
            c.Transform2 = t
        self.comps[nm] = c
        if transform is not None:
            self.free_comps.add(nm)
        if fix:
            self.clear()
            if C(c, "Select4", False, NOTHING, False):
                C(self.model, "FixComponent")
            self.clear()
        return c

    def sel_ref(self, cn, rn, app):
        c = self.comps.get(cn)
        if c is None:
            return False
        sel = self.model.SelectionManager
        want = C(sel, "GetSelectedObjectCount2", -1) + 1 if app else 1
        ok = False
        try:
            f = C(c, "FeatureByName", rn)
            ok = f is not None and C(f, "Select2", app, 1) and C(sel, "GetSelectedObjectCount2", -1) == want
        except pythoncom.com_error:
            ok = False
        if not ok:
            typ = "AXIS" if rn.startswith("A_") else "DATUMPOINT"
            ok = C(self.model.Extension, "SelectByID2", f"{rn}@{c.Name2}@{self.asm_title}", typ,
                   0, 0, 0, app, 1, NOTHING, 0)
        return bool(ok)

    def try_mate(self, ca, ra, cb, rb, kind):
        self.clear()
        if not self.sel_ref(ca, ra, False):
            return None, -1
        if not self.sel_ref(cb, rb, True):
            return None, -2
        e = byref_long()
        try:
            m = C(self.model, "AddMate5", kind, 2, False, 0, 0, 0, 1, 1, 0, 0, 0, False, False, 0, e)
        except pythoncom.com_error as ex:
            self.L(f"  AddMate5 error {ex}")
            m = None
        self.clear()
        return m, e.value

    def mate_ref(self, ca, ra, cb, rb, label, kind=0):
        m, e = self.try_mate(ca, ra, cb, rb, kind)
        if m is None and kind == 0 and ra.startswith("A_") and rb.startswith("A_"):
            m, e = self.try_mate(ca, ra, cb, rb, 1)
        if m is None:
            self.n["mates_fail"] += 1
            why = {-1: "first reference not selected", -2: "second reference not selected"}.get(e, f"swAddMateError {e}")
            self.L(f"FAIL mate {label}: {ra}@{ca} to {rb}@{cb} ({why})")
        else:
            self.n["mates"] += 1
        return m

    def end_asm(self):
        self.clear()
        C(self.model, "ForceRebuild3", False)
        for nm, c in self.comps.items():
            if nm in self.free_comps:
                continue
            # the corner is a mechanism (it can travel), so the solver may nudge it by a few
            # microns; anything over 0.1 mm / ~0.01 deg is a real misplacement
            r = c.Transform2.ArrayData
            if sum(abs(v) for v in r[9:12]) > 1e-4 or abs(r[0] - 1) + abs(r[4] - 1) + abs(r[8] - 1) > 1e-7:
                self.n["other_fail"] += 1
                self.L(f"FAIL {c.Name2} is not where it was modelled in {self.cur}")
        # every mate must solve
        f = C(self.model, "FirstFeature")
        while f is not None:
            if C(f, "GetTypeName2") == "MateGroup":
                s = C(f, "GetFirstSubFeature")
                while s is not None:
                    if C(s, "GetErrorCode"):
                        self.n["mates_fail"] += 1
                        self.L(f"FAIL mate {s.Name} in {self.cur} does not solve (error {C(s, 'GetErrorCode')})")
                    s = C(s, "GetNextSubFeature")
            f = C(f, "GetNextFeature")
        e, w = byref_long(), byref_long()
        if not C(self.model, "Save3", 1, e, w):
            self.n["other_fail"] += 1
            self.L(f"FAIL save {self.cur} (error {e.value})")

    def make_flexible(self, nm):
        c = self.comps.get(nm)
        if c is None:
            return
        self.clear()
        if C(c, "Select4", False, NOTHING, False):
            C(self.model, "CompConfigProperties5", 2, 1, True, False, "", False, False)
        st = c.Solving
        self.clear()
        if st != 1:
            self.n["other_fail"] += 1
            self.L(f"FAIL make {nm} flexible")

    # ---- damper placement inside a corner
    def damper_transform(self, corner, eye_y):
        refs_c = self.part_refs[f"SUSP_{corner}_chassis_mounts"]
        refs_r = self.part_refs[f"SUSP_{corner}_rocker"]
        pc = refs_c["P_DAMPER_CHASSIS"][1]
        pr = refs_r["P_DAMPER"][1]
        ax = refs_c["A_ROCKER"][1]
        L = math.dist(pc, pr)
        u = [(a - b) / L for a, b in zip(pc, pr)]                 # local +Y: rocker eye -> chassis eye
        pin = [b - a for a, b in zip(ax[:3], ax[3:])]
        d = sum(p * q for p, q in zip(pin, u))
        z = [p - d * q for p, q in zip(pin, u)]
        zn = math.sqrt(sum(v * v for v in z))
        z = [v / zn for v in z]
        x = [u[1] * z[2] - u[2] * z[1], u[2] * z[0] - u[0] * z[2], u[0] * z[1] - u[1] * z[0]]
        a = math.radians(self.clocks.get(corner, 0.0))      # turn about the damper axis (local Y)
        x, z = ([math.cos(a) * p - math.sin(a) * q for p, q in zip(x, z)],
                [math.sin(a) * p + math.cos(a) * q for p, q in zip(x, z)])
        # local eye of the chassis-side part (body) is at y=0.220, of the shaft part at y=0
        origin = [c - 0.220 * k for c, k in zip(pc, u)] if eye_y > 0 else list(pr)
        self.damper_geom = dict(pc=pc, pr=pr, length=L, pin_dot=abs(d) / math.sqrt(sum(p * p for p in pin)))
        return x, u, z, origin

    def make_tr(self, rows, origin, transpose):
        x, y, z = rows
        if transpose:
            r = [x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]
        else:
            r = list(x) + list(y) + list(z)
        return C(self.math, "CreateTransform", dbl_array(r + list(origin) + [1, 0, 0, 0]))

    def place_damper(self, key, file_nm, corner, eye_y):
        x, y, z, origin = self.damper_transform(corner, eye_y)
        target = self.damper_geom["pc"] if eye_y > 0 else self.damper_geom["pr"]
        c = None
        for transpose in (False, True):
            tr = self.make_tr((x, y, z), origin, transpose)
            if c is None:
                c = self.add_comp(key, ".SLDPRT", False, file_nm=file_nm, transform=tr)
                if c is None:
                    return
            else:
                C(c, "SetTransformAndSolve3", tr, False)
            pt = C(self.math, "CreatePoint", dbl_array([0, eye_y, 0]))
            w = C(pt, "MultiplyTransform", c.Transform2).ArrayData
            if math.dist(w, target) < 1e-5:
                return
        self.L(f"WARN {key}: could not pre-place the damper exactly; left to the mates")

    def damper_extra_mates(self, corner):
        b, s = f"SUSP_{corner}_damper_body", f"SUSP_{corner}_damper_shaft"
        if b not in self.comps or s not in self.comps:
            return
        # Only the two halves are kept from spinning relative to each other. A "body pin //
        # rocker axis" mate over-defines the corner: the concepts' rocker axes are 0.001-0.003 deg
        # off perpendicular to the damper axis, and that alone breaks the rocker and damper mates.
        self.mate_ref(s, "A_PIN", b, "A_PIN", f"{corner} damper shaft pin // body pin", kind=3)
        C(self.model, "ForceRebuild3", False)

    # ------------------------------------------------------------------ interpreter
    def run_sub(self, name):
        for st in self.subs[name]:
            m = re.match(r"If Not (BeginPart|BeginAsm)\(([^)]*)\) Then Exit Sub", st)
            if m:
                arg = m.group(2).strip('"')
                if arg == "ASM_NAME":
                    arg = self.asm_name
                ok = self.begin_part(arg) if m.group(1) == "BeginPart" else self.begin_asm(arg)
                if not ok:
                    return
                continue
            word, _, rest = st.partition(" ")
            a = parse_args(rest) if rest else []
            if word in self.subs and re.match(r"(Build|Part_|Asm_)", word):
                if re.match(r"Part_SUSP_\w\w_damper_(body|shaft)$", word):
                    self.L(f"skip {word}: replaced by the real damper")
                    continue
                self.run_sub(word)
            elif word == "Cyl":
                self.cyl(*a)
            elif word == "RefOpen":
                self.ref_open()
            elif word == "RefPoint":
                self.ref_point(*a)
            elif word == "RefAxis":
                self.ref_axis(*a)
            elif word == "SketchOpen":
                self.sketch_open()
            elif word == "SketchLine":
                self.sketch_line(*a)
            elif word == "SketchClose":
                self.sketch_close(*a)
            elif word == "EndPart":
                self.end_part(*a)
            elif word == "AddComp":
                nm, ext, fix = a
                dm = re.match(r"SUSP_(\w\w)_damper_(body|shaft)$", nm)
                if dm:
                    corner, side = dm.groups()
                    self.place_damper(nm, "AMORTECEDOR_CORPO" if side == "body" else "AMORTECEDOR_HASTE",
                                      corner, 0.220 if side == "body" else 0.0)
                else:
                    self.add_comp(nm, ext, fix)
            elif word == "MateRef":
                self.mate_ref(*a)
            elif word == "MakeFlexible":
                self.make_flexible(*a)
            elif word == "EndAsm":
                cm = re.match(r"SUSP_(\w\w)$", self.cur)
                if cm:
                    self.damper_extra_mates(cm.group(1))
                self.end_asm()
            else:
                raise RuntimeError(f"unknown statement in {name}: {st}")

    def close_ours(self):
        names = set(self.written)
        for _ in range(3):
            asms, prts = [], []
            d = C(self.sw, "GetFirstDocument")
            while d is not None:
                fn = os.path.basename(C(d, "GetPathName")).upper()
                if fn and (fn in names or os.path.dirname(C(d, "GetPathName")).lower() + "\\" == self.dir.lower()):
                    (asms if C(d, "GetType") == 2 else prts).append(C(d, "GetTitle"))
                d = C(d, "GetNext")
            if not asms and not prts:
                return
            for t in asms + prts:
                C(self.sw, "CloseDoc", t)

    def planned_names(self):
        names = {"AMORTECEDOR_CORPO.SLDPRT", "AMORTECEDOR_HASTE.SLDPRT"}
        for body in self.subs.values():
            for st in body:
                m = re.match(r'If Not (BeginPart|BeginAsm)\("?([^")]*)"?\) Then Exit Sub', st)
                if m:
                    nm = self.asm_name if m.group(2) == "ASM_NAME" else m.group(2)
                    names.add((nm + (".SLDPRT" if m.group(1) == "BeginPart" else ".SLDASM")).upper())
        return names

    def close_conflicts(self):
        """SolidWorks cannot hold two documents with the same file name: close any open document
        (from any folder, e.g. another concept) named like a file this run writes."""
        names = self.planned_names()
        for _ in range(10):
            hits, docs, dirs = [], [], set()
            d = C(self.sw, "GetFirstDocument")
            while d is not None:
                path = C(d, "GetPathName") or ""
                docs.append((d, path))
                if os.path.basename(path).upper() in names:
                    if C(d, "GetSaveFlag"):
                        raise RuntimeError(f"{path} is open with unsaved changes; save or close it first")
                    hits.append((bool(d.Visible), C(d, "GetType") == 2, C(d, "GetTitle")))
                    dirs.add(os.path.dirname(path).lower())
                d = C(d, "GetNext")
            if not hits:
                return
            if not any(h[0] for h in hits):
                # hidden components held by a visible assembly from the same folder (e.g. the
                # previous concept's top assembly): close that assembly
                for d, path in docs:
                    if d.Visible and path and os.path.dirname(path).lower() in dirs and not C(d, "GetSaveFlag"):
                        hits.append((True, C(d, "GetType") == 2, C(d, "GetTitle")))
            # a document used by an open assembly cannot be closed on its own: close the
            # top-level (visible) documents first, assemblies before parts, then look again
            top = [h for h in hits if h[0]] or hits
            top.sort(key=lambda h: (not h[1], h[2]))
            C(self.sw, "CloseDoc", top[0][2])
            time.sleep(0.3)
        raise RuntimeError("could not close the documents that share file names with this build")

    def run(self):
        os.makedirs(self.dir, exist_ok=True)
        self.close_conflicts()
        t0 = time.time()
        C(self.sw, "DocumentVisible", False, 1)
        C(self.sw, "DocumentVisible", True, 2)
        try:
            self.run_sub("BuildParts")
            C(self.sw, "DocumentVisible", True, 1)
            self.make_damper_copies()
            self.run_sub("BuildCorners")
            self.run_sub("BuildTop")
        except Exception as ex:
            self.L(f"STOPPED in {self.cur}: {ex!r}")
            raise
        finally:
            C(self.sw, "DocumentVisible", True, 1)
            C(self.sw, "DocumentVisible", True, 2)
            self.close_ours()
            e, w = byref_long(), byref_long()
            top = C(self.sw, "OpenDoc6", self.dir + self.asm_name + ".SLDASM", 2, 1, "", e, w)
            if top is not None:
                C(top, "ShowNamedView2", "*Isometric", 7)
                C(top, "ViewZoomtofit2")
            n = self.n
            self.L(f"{self.asm_name}: {n['solids']} solids ({n['solids_fail']} failed), "
                   f"{n['refs']} reference points/axes ({n['refs_fail']} failed), "
                   f"{n['mates']} mates ({n['mates_fail']} failed), {n['other_fail']} other failures, "
                   f"{time.time() - t0:.0f} s")
            with open(self.dir + self.asm_name + "_log.txt", "w", encoding="utf-8") as fh:
                fh.write("\n".join(self.log) + "\n")


if __name__ == "__main__":
    clocks = {}
    if "--clock" in sys.argv:
        for kv in sys.argv[sys.argv.index("--clock") + 1:]:
            k, v = kv.split("=")
            clocks[k.upper()] = float(v)
    b = Builder(sys.argv[1], sys.argv[2], sys.argv[3], clocks)
    if clocks:
        b.L("damper clock (deg about its axis): " + ", ".join(f"{k} {v:g}" for k, v in clocks.items()))
    b.run()
