r"""
Replace the simplified wheel (SUSP_<k>_wheel, a tyre cylinder) with the team's wheel model in
every corner assembly of a concept, in place.

1. The wheel file is copied to SUSP_<c>\RODA_ESTRELA.SLDPRT (the original is not modified)
   and the copy gets the two reference points the corner mates use:
     P_CENTER   = the wheel centre, on the part's own mid plane "Plano medio" (y = -77.39 mm)
     P_CENTER_B = 40 mm from it along the spin axis (the part's Y), toward the hub face (+Y)
   The part's hub face is at its origin, on the +Y side of the rim, so +Y goes OUTBOARD, as
   P_CENTER_B does in the simplified wheel (on both sides of the car).
2. In each corner: the simplified wheel's P_CENTER / P_CENTER_B are read, the component is
   deleted (its two mates go with it), the new wheel is inserted at the pose that puts its
   points on those, and the same two mates are added again:
     P_CENTER   coincident  upright P_WHEEL_CENTER
     P_CENTER_B coincident  upright A_SPIN   (the wheel still spins freely)
   The corner is saved only if the new wheel centre lands on the old one.

Usage:  python -I swap_wheel.py <X|Y|Z> "<wheel .SLDPRT>"
"""
import math
import os
import shutil
import sys

import pythoncom
import win32com.client as wc
from win32com.client import VARIANT

OUT = r"C:\soft\FSAE\Suspension and Vehicle Dynamics\Ultimos conceitos"
CORNERS = ("FL", "FR", "RL", "RR")
NEW = "RODA_ESTRELA"
CENTER_Y = -0.07739            # "Plano medio" of the wheel part [m]
B_OFFSET = 0.040               # P_CENTER_B, toward +Y (the hub face, outboard) [m]
NOTHING = VARIANT(pythoncom.VT_DISPATCH, None)


def C(o, n, *a):
    o._FlagAsMethod(n)
    return getattr(o, n)(*a)


def BL():
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)


def arr(v):
    return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, [float(x) for x in v])


def newest(doc, n0):
    return C(doc, "FeatureByPositionReverse", 0) if C(doc, "GetFeatureCount") > n0 else None


def add_ref_points(sw, part, pts):
    """Reference points as intersections of two short 3D-sketch lines (build_susp's way)."""
    sm = part.SketchManager
    C(part, "ClearSelection2", True)
    sm.AddToDB = True
    sm.DisplayWhenAdded = False
    C(sm, "Insert3DSketch", True)
    segs = []
    h = 0.005
    for nm, (x, y, z) in pts.items():
        a = C(sm, "CreateLine", x - h * 0.8, y - h * 0.48, z - h * 0.36, x + h * 0.8, y + h * 0.48, z + h * 0.36)
        b = C(sm, "CreateLine", x + h * 0.48, y - h * 0.6, z - h * 0.64, x - h * 0.48, y + h * 0.6, z + h * 0.64)
        segs.append((nm, (x, y, z), a, b))
    C(sm, "Insert3DSketch", True)
    sm.AddToDB = False
    sm.DisplayWhenAdded = True
    sk = C(part, "FeatureByPositionReverse", 0)
    sk.Name = "REF_SKETCH"
    for nm, p, a, b in segs:
        C(part, "ClearSelection2", True)
        C(a, "Select4", False, NOTHING)
        C(b, "Select4", True, NOTHING)
        n0 = C(part, "GetFeatureCount")
        C(part.FeatureManager, "InsertReferencePoint", 6, 0, 0.0, 1)
        f = newest(part, n0)
        if f is None:
            raise RuntimeError(f"could not make reference point {nm}")
        f.Name = nm
        got = C(C(f, "GetSpecificFeature2"), "GetRefPoint").ArrayData
        if math.dist(got, p) > 1e-6:
            raise RuntimeError(f"{nm} made at {got}, wanted {p}")
    C(part, "ClearSelection2", True)
    C(sk, "Select2", False, 0)
    C(part, "BlankSketch")
    C(part, "ClearSelection2", True)


def make_wheel_copy(sw, src, dst):
    C(sw, "CloseAllDocuments", True)
    shutil.copyfile(src, dst)
    part = C(sw, "OpenDoc6", dst, 1, 1, "", BL(), BL())
    add_ref_points(sw, part, {"P_CENTER": (0.0, CENTER_Y, 0.0), "P_CENTER_B": (0.0, CENTER_Y + B_OFFSET, 0.0)})
    C(part, "ForceRebuild3", False)
    ok = C(part, "Save3", 1, BL(), BL())
    C(sw, "CloseAllDocuments", True)
    print(f"wheel copy {os.path.basename(dst)}: reference points added, saved {ok}", flush=True)


def world_ref(mu, comp, name):
    f = C(comp, "FeatureByName", name)
    local = C(C(f, "GetSpecificFeature2"), "GetRefPoint").ArrayData
    return C(C(mu, "CreatePoint", arr(local)), "MultiplyTransform", comp.Transform2).ArrayData


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def mate(doc, ca, ra, cb, rb):
    C(doc, "ClearSelection2", True)
    ok = C(C(ca, "FeatureByName", ra), "Select2", False, 1) and C(C(cb, "FeatureByName", rb), "Select2", True, 1)
    e = BL()
    m = C(doc, "AddMate5", 0, 2, False, 0, 0, 0, 1, 1, 0, 0, 0, False, False, 0, e) if ok else None
    C(doc, "ClearSelection2", True)
    return m is not None and e.value == 1


def swap_corner(sw, mu, c, k, wheel_path):
    doc = C(sw, "OpenDoc6", OUT + rf"\SUSP_{c}\SUSP_{k}.SLDASM", 2, 1, "", BL(), BL())
    comps = {x.Name2.rsplit("-", 1)[0]: x for x in C(doc, "GetComponents", True)}
    old, up = comps.get(f"SUSP_{k}_wheel"), comps[f"SUSP_{k}_upright"]
    if old is None:
        print(f"{c} {k}: no simplified wheel (already swapped?)", flush=True)
        C(sw, "CloseAllDocuments", True)
        return
    pc, pb = world_ref(mu, old, "P_CENTER"), world_ref(mu, old, "P_CENTER_B")
    y = unit([b - a for a, b in zip(pc, pb)])                     # part +Y: outboard spin axis
    ref = [0.0, 0.0, 1.0] if abs(y[2]) < 0.9 else [1.0, 0.0, 0.0]
    x = unit(cross(y, ref))
    z = cross(x, y)
    origin = [p - CENTER_Y * u for p, u in zip(pc, y)]            # local (0, CENTER_Y, 0) -> pc
    C(doc, "ClearSelection2", True)
    C(old, "Select4", False, NOTHING, False)
    C(doc, "EditDelete")
    C(doc, "ClearSelection2", True)
    C(sw, "OpenDoc6", wheel_path, 1, 1, "", BL(), BL())          # AddComponent5 needs it loaded
    C(sw, "ActivateDoc3", C(doc, "GetTitle"), False, 1, BL())
    new = C(doc, "AddComponent5", wheel_path, 0, "", False, "", 0, 0, 0)
    if new is None:
        raise SystemExit(f"{k}: could not insert {wheel_path}; corner not saved")
    C(new, "SetTransformAndSolve3", C(mu, "CreateTransform", arr(x + y + z + origin + [1, 0, 0, 0])), False)
    m1 = mate(doc, new, "P_CENTER", up, "P_WHEEL_CENTER")
    m2 = mate(doc, new, "P_CENTER_B", up, "A_SPIN")
    C(doc, "ForceRebuild3", False)
    comps = {x.Name2.rsplit("-", 1)[0]: x for x in C(doc, "GetComponents", True)}
    new = comps[NEW]
    d_c = math.dist(world_ref(mu, new, "P_CENTER"), pc) * 1000
    d_b = math.dist(world_ref(mu, new, "P_CENTER_B"), pb) * 1000
    ok = m1 and m2 and d_c < 0.01 and d_b < 0.01
    saved = C(doc, "Save3", 1, BL(), BL()) if ok else False
    print(f"{c} {k}: mates {m1}/{m2}, centre moved {d_c:.4f} mm, axis point moved {d_b:.4f} mm, saved {saved}",
          flush=True)
    C(sw, "CloseAllDocuments", True)
    if not ok:
        raise SystemExit(f"{k}: swap did not check out; corner not saved")


def main(c, src):
    sw = wc.GetActiveObject("SldWorks.Application")
    mu = C(sw, "GetMathUtility")
    dst = OUT + rf"\SUSP_{c}\{NEW}.SLDPRT"
    make_wheel_copy(sw, src, dst)
    for k in CORNERS:
        swap_corner(sw, mu, c, k, dst)


if __name__ == "__main__":
    main(sys.argv[1].upper(), sys.argv[2])
