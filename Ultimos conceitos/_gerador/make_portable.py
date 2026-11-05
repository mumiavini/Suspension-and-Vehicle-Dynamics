r"""
Make "Ultimos conceitos" self-contained, so the folder can be zipped and opened elsewhere:

1. the chassis and engine parts the car copies use are copied into
   Ultimos conceitos\Chassis and engine\ (the originals in MONT_CHASSI_SUSP_MOTOR are not touched);
2. each car copy MONT_CHASSI_SUSP_<c>.SLDASM is re-pointed to those copies
   (SldWorks.ReplaceReferencedDocument, with the documents closed);
3. the suppressed SUSP_2027-1 (the old suspension, hidden in the copies) is deleted from each
   car copy: it pulled in the whole Suspension 2027 CAD\SUSP_2027 tree;
4. every car's dependencies are listed again: all must be inside the folder and exist.

Usage:  python -I make_portable.py
"""
import os
import shutil
import sys

import pythoncom
import win32com.client as wc
from win32com.client import VARIANT

REPO = r"C:\soft\FSAE\Suspension and Vehicle Dynamics"
ROOT = REPO + r"\Ultimos conceitos"
SRC = REPO + r"\MONT_CHASSI_SUSP_MOTOR"
DST = ROOT + r"\Chassis and engine"
PARTS = ("Xasis 33.3.SLDPRT", "MOTOR_FIXADO.SLDPRT")
NOTHING = VARIANT(pythoncom.VT_DISPATCH, None)


def C(o, n, *a):
    o._FlagAsMethod(n)
    return getattr(o, n)(*a)


def BL():
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)


def deps(sw, path):
    d = C(sw, "GetDocumentDependencies2", path, True, True, False)
    return sorted(set(d[1::2])) if d else []


def main():
    sw = wc.GetActiveObject("SldWorks.Application")
    C(sw, "CloseAllDocuments", True)
    os.makedirs(DST, exist_ok=True)
    for p in PARTS:
        if not os.path.exists(os.path.join(DST, p)):
            shutil.copy2(os.path.join(SRC, p), os.path.join(DST, p))
    bad = False
    for c in "XYZ":
        car = ROOT + rf"\MONT_CHASSI_SUSP_{c}.SLDASM"
        for p in PARTS:
            ok = C(sw, "ReplaceReferencedDocument", car, os.path.join(SRC, p), os.path.join(DST, p))
            print(f"{c}: {p} -> Chassis and engine: {ok}", flush=True)
        doc = C(sw, "OpenDoc6", car, 2, 1, "", BL(), BL())
        old = [x for x in C(doc, "GetComponents", True) if x.Name2 == "SUSP_2027-1"]
        if old:
            C(doc, "ClearSelection2", True)
            C(old[0], "Select4", False, NOTHING, False)
            C(doc, "EditDelete")
            C(doc, "ClearSelection2", True)
        C(doc, "ForceRebuild3", False)
        saved = C(doc, "Save3", 1, BL(), BL())
        print(f"{c}: SUSP_2027-1 deleted: {bool(old)}, saved {saved}", flush=True)
        C(sw, "CloseAllDocuments", True)
        d = deps(sw, car)
        outside = [x for x in d if not x.lower().startswith(ROOT.lower())]
        missing = [x for x in d if not os.path.exists(x)]
        print(f"{c}: {len(d)} referenced files, outside the folder {len(outside)}, missing {len(missing)}", flush=True)
        for x in outside + missing:
            print("   ", x)
        bad |= bool(outside or missing)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
