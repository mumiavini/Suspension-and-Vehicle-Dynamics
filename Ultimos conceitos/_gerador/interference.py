r"""
Static interference check of a car copy (MONT_CHASSI_SUSP_<c>.SLDASM) in SolidWorks:
every interference that involves the chassis, the engine or the real damper copies.
Joint overlaps between the generated suspension parts themselves are expected (bushes,
ball joints, rod ends meet there) and are only counted.

Usage:  python -I interference.py <X|Y|Z> <out.json>
"""
import json
import os
import sys
import time

import pythoncom
import win32com.client as wc
from win32com.client import VARIANT

OUT = r"C:\soft\FSAE\Suspension and Vehicle Dynamics\Ultimos conceitos"


def C(o, n, *a):
    o._FlagAsMethod(n)
    return getattr(o, n)(*a)


def BL():
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)


def main(c, out):
    sw = wc.GetActiveObject("SldWorks.Application")
    path = OUT + rf"\MONT_CHASSI_SUSP_{c}.SLDASM"
    # one concept at a time: close everything (car parts are only rebuilt on open; discard that)
    d = C(sw, "GetFirstDocument")
    while d is not None:
        p = C(d, "GetPathName") or ""
        if C(d, "GetSaveFlag") and d.Visible and "MONT_CHASSI_SUSP_" not in p.upper():
            raise SystemExit(f"{p} has unsaved changes")
        d = C(d, "GetNext")
    C(sw, "CloseAllDocuments", True)
    doc = C(sw, "OpenDoc6", path, 2, 1, "", BL(), BL())
    t0 = time.time()
    idm = doc.InterferenceDetectionManager
    idm.TreatCoincidenceAsInterference = False
    idm.TreatSubAssembliesAsComponents = False
    idm.IncludeMultibodyPartInterferences = False
    idm.IgnoreHiddenBodies = True
    idm.MakeInterferingPartsTransparent = False
    ints = C(idm, "GetInterferences") or []
    rows, internal = [], 0
    for it in ints:
        comps = [k.Name2 for k in it.Components]
        vol = it.Volume * 1e9
        names = " ".join(comps).upper()
        if any(k in names for k in ("XASIS", "MOTOR", "AMORTECEDOR", "RODA")):
            rows.append({"components": comps, "volume_mm3": round(vol, 1)})
        else:
            internal += 1
    C(idm, "Done")
    res = {"concept": c, "seconds": round(time.time() - t0), "total": len(ints),
           "suspension_internal_joint_overlaps": internal, "relevant": rows}
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    C(sw, "CloseAllDocuments", True)


if __name__ == "__main__":
    main(sys.argv[1].upper(), sys.argv[2])
