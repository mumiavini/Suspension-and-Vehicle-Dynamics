r"""
Step 3 of the macro guide ("Colocar no carro") for one concept, on a COPY of the car:

  MONT_CHASSI_SUSP_MOTOR.SLDASM  --save as-->  Ultimos conceitos\MONT_CHASSI_SUSP_<c>.SLDASM
  suppress the old SUSP_2027-1, reopen (so none of its same-named parts is loaded),
  insert SUSP_<c>.SLDASM and mate its Front/Top/Right planes to Xasis 33.3's (coincident, aligned),
  exactly as SUSP_2027 was mated. The original car assembly is not modified.

Usage:  python -I place_in_car.py <X|Y|Z> [image.jpg]
"""
import os
import sys

import pythoncom
import win32com.client as wc
from win32com.client import VARIANT

ROOT = r"C:\soft\FSAE\Suspension and Vehicle Dynamics"
CAR = ROOT + r"\MONT_CHASSI_SUSP_MOTOR\MONT_CHASSI_SUSP_MOTOR.SLDASM"
OUT = ROOT + r"\Ultimos conceitos"
OLD = "SUSP_2027-1"
NOTHING = VARIANT(pythoncom.VT_DISPATCH, None)


def C(o, n, *a):
    o._FlagAsMethod(n)
    return getattr(o, n)(*a)


def BL():
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)


def open_docs(sw):
    d, out = C(sw, "GetFirstDocument"), []
    while d is not None:
        out.append(d)
        d = C(d, "GetNext")
    return out


def close_all(sw):
    """Close everything; save only documents this workflow generated (flexible
    subassemblies re-solve on open and come back 'modified')."""
    car_dir = os.path.dirname(CAR).lower() + "\\"
    for d in open_docs(sw):
        if C(d, "GetSaveFlag"):
            p = C(d, "GetPathName") or ""
            if p.lower().startswith(car_dir) and not d.Visible:
                # the car's parts (Xasis, motor) loaded as components of the copy come back
                # 'modified' after the open-time rebuild: discard it, the originals stay as they are
                continue
            if not p.lower().startswith((OUT + "\\SUSP_").lower()) and \
                    not os.path.basename(p).upper().startswith("MONT_CHASSI_SUSP_"):
                raise SystemExit(f"{p or C(d, 'GetTitle')} has unsaved changes: save or close it first")
            e, w = BL(), BL()
            C(d, "Save3", 1, e, w)
    C(sw, "CloseAllDocuments", True)


def planes(comp):
    out, f = [], C(C(comp, "GetModelDoc2"), "FirstFeature")
    while f is not None and len(out) < 3:
        if C(f, "GetTypeName2") == "RefPlane":
            out.append(f.Name)
        f = C(f, "GetNextFeature")
    return out


def main(c, image=None):
    sw = wc.GetActiveObject("SldWorks.Application")
    new = OUT + rf"\MONT_CHASSI_SUSP_{c}.SLDASM"
    susp = OUT + rf"\SUSP_{c}\SUSP_{c}.SLDASM"
    if os.path.exists(new):
        raise SystemExit(f"{new} already exists")
    if not os.path.exists(susp):
        raise SystemExit(f"{susp} not found")
    car_mtime = os.path.getmtime(CAR)

    # 1) copy of the car with the old suspension suppressed
    close_all(sw)
    e, w = BL(), BL()
    doc = C(sw, "OpenDoc6", CAR, 2, 1, "", e, w)
    if doc is None:
        raise SystemExit(f"could not open the car assembly (error {e.value})")
    e, w = BL(), BL()
    if not C(doc.Extension, "SaveAs", new, 0, 1, NOTHING, e, w):
        raise SystemExit(f"save as failed (error {e.value})")
    old = [k for k in C(doc, "GetComponents", True) if k.Name2 == OLD]
    if len(old) != 1:
        raise SystemExit(f"{OLD} not found in the car assembly")
    C(doc, "ClearSelection2", True)
    C(old[0], "Select4", False, NOTHING, False)
    if not C(doc, "EditSuppress2"):
        raise SystemExit(f"could not suppress {OLD}")
    C(doc, "ForceRebuild3", False)
    C(doc, "Save3", 1, BL(), BL())
    close_all(sw)
    print(f"MONT_CHASSI_SUSP_{c}: copy saved, {OLD} suppressed")

    # 2) reopen without the old suspension, insert the new one, mate the planes
    e, w = BL(), BL()
    doc = C(sw, "OpenDoc6", new, 2, 1, "", e, w)
    stale = [C(d, "GetPathName") for d in open_docs(sw) if "Suspension 2027 CAD" in (C(d, "GetPathName") or "")]
    if stale:
        raise SystemExit(f"old suspension documents still loaded: {stale[:3]}")
    C(sw, "OpenDoc6", susp, 2, 1, "", BL(), BL())
    C(sw, "ActivateDoc3", C(doc, "GetTitle"), False, 1, BL())
    xasis = [k for k in C(doc, "GetComponents", True) if k.Name2.startswith("Xasis")][0]
    comp = C(doc, "AddComponent5", susp, 0, "", False, "", 0, 0, 0)
    if comp is None:
        raise SystemExit("could not insert the suspension")
    C(comp, "SetTransformAndSolve3", xasis.Transform2, False)
    for a, b in zip(planes(comp), planes(xasis)):
        C(doc, "ClearSelection2", True)
        ok = C(C(comp, "FeatureByName", a), "Select2", False, 1) and \
            C(C(xasis, "FeatureByName", b), "Select2", True, 1)
        err = BL()
        m = C(doc, "AddMate5", 0, 0, False, 0, 0, 0, 1, 1, 0, 0, 0, False, False, 0, err) if ok else None
        if m is None:
            raise SystemExit(f"mate {a} <-> {b} failed (selected={ok}, error {err.value})")
        print(f"  mate {m.Name}: {a}@{comp.Name2} <-> {b}@{xasis.Name2}")
    C(doc, "ClearSelection2", True)
    C(doc, "ForceRebuild3", False)

    # 3) check and save
    diff = max(abs(p - q) for p, q in zip(xasis.Transform2.ArrayData[:12], comp.Transform2.ArrayData[:12]))
    errs = []
    f = C(doc, "FirstFeature")
    while f is not None:
        if C(f, "GetTypeName2") == "MateGroup":
            s = C(f, "GetFirstSubFeature")
            while s is not None:
                if C(s, "GetErrorCode") and not C(s, "IsSuppressed"):
                    errs.append((s.Name, C(s, "GetErrorCode")))
                s = C(s, "GetNextSubFeature")
        f = C(f, "GetNextFeature")
    e, w = BL(), BL()
    saved = C(doc, "Save3", 1, e, w)
    print(f"  {comp.Name2} vs Xasis transform diff {diff:.1e} | mate errors {errs} | saved {saved}")
    print(f"  original car assembly untouched: {os.path.getmtime(CAR) == car_mtime}")
    if image:
        C(doc, "ShowNamedView2", "*Isometric", 7)
        C(doc, "ViewZoomtofit2")
        C(doc.Extension, "SaveAs", image, 0, 3, NOTHING, BL(), BL())
    if diff > 1e-9 or errs or not saved:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1].upper(), sys.argv[2] if len(sys.argv) > 2 else None)
