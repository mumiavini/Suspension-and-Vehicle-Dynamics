r"""
Where on the real damper does it touch the frame? For each AMORTECEDOR_CORPO <-> Xasis
interference in a car copy, the interference body's box centre is taken into the damper's
own frame (axis = Y, chassis eye at y = 220 mm, reservoir toward -X) and classified.
Then the damper is clocked about its own axis in 30 deg steps (in memory, not saved) to see
which clock clears the frame.

Usage:  python -I damper_contacts.py <X|Y|Z>
"""
import math
import sys

import pythoncom
import win32com.client as wc
from win32com.client import VARIANT

OUT = r"C:\soft\FSAE\Suspension and Vehicle Dynamics\Ultimos conceitos"


def C(o, n, *a):
    o._FlagAsMethod(n)
    return getattr(o, n)(*a)


def BL():
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)


def arr(v):
    return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, [float(x) for x in v])


def region(p):
    x, y, z = (v * 1000 for v in p)
    r = math.hypot(x, z)
    if math.hypot(x + 50, z) < 25 and 100 < y < 215:
        return "reservoir"
    if y >= 175:
        return "head / chassis eye"
    if 40 <= y <= 165 and r > 20:
        return "spring"
    if y < 40:
        return "lower seat"
    return "body"


def damper_hits(sw, doc, mu):
    idm = doc.InterferenceDetectionManager
    idm.TreatCoincidenceAsInterference = False
    idm.TreatSubAssembliesAsComponents = False
    idm.IncludeMultibodyPartInterferences = False
    idm.IgnoreHiddenBodies = True
    out = []
    for it in C(idm, "GetInterferences") or []:
        comps = list(it.Components)
        names = [k.Name2 for k in comps]
        body = [k for k in comps if "AMORTECEDOR_CORPO" in k.Name2]
        if not body or not any("Xasis" in n for n in names):
            continue
        b = body[0]
        try:
            box = C(C(it, "GetInterferenceBody"), "GetBodyBox")
        except pythoncom.com_error as ex:
            print("   (could not read the interference body:", ex, ")")
            box = None
        corner = b.Name2.split("/")[1][5:7] if "/" in b.Name2 else "?"
        where = "?"
        if box is not None:
            centre = [(box[i] + box[i + 3]) / 2 for i in range(3)]
            inv = C(b.Transform2, "Inverse")
            local = C(C(mu, "CreatePoint", arr(centre)), "MultiplyTransform", inv).ArrayData
            where = f"{region(local)} (local x,y,z = {', '.join(f'{v*1000:.0f}' for v in local)} mm)"
        out.append((corner, round(it.Volume * 1e9, 1), where))
    C(idm, "Done")
    return out


def main(c):
    sw = wc.GetActiveObject("SldWorks.Application")
    mu = C(sw, "GetMathUtility")
    C(sw, "CloseAllDocuments", True)
    doc = C(sw, "OpenDoc6", OUT + rf"\MONT_CHASSI_SUSP_{c}.SLDASM", 2, 1, "", BL(), BL())
    hits = damper_hits(sw, doc, mu)
    print(f"== {c}: damper body vs frame")
    for h in hits:
        print(f"   {h[0]}  {h[1]:8.1f} mm3  {h[2]}")
    # clock sweep on the left corners that touch (the right is the mirror)
    corners = sorted({h[0] for h in hits if h[0] in ("FL", "RL")})
    susp = [k for k in C(doc, "GetComponents", True) if k.Name2 == f"SUSP_{c}-1"][0]
    for corner in corners:
        sub = [k for k in C(susp, "GetChildren") if k.Name2.endswith(f"SUSP_{corner}-1")][0]
        parts = {k.Name2.split("/")[-1].split("-")[0]: k for k in C(sub, "GetChildren")}
        body, rod = parts["AMORTECEDOR_CORPO"], parts["AMORTECEDOR_HASTE"]
        t0b, t0r = body.Transform2, rod.Transform2
        # damper axis in the corner's frame: local Y of the body
        res = []
        for deg in range(0, 360, 30):
            a = math.radians(deg)
            # rotation about local Y through the local origin (the axis), applied in local frame
            rot = C(mu, "CreateTransform", arr([math.cos(a), 0, -math.sin(a), 0, 1, 0, math.sin(a), 0, math.cos(a),
                                                 0, 0, 0, 1, 0, 0, 0]))
            body.Transform2 = C(rot, "Multiply", t0b)
            rod.Transform2 = C(rot, "Multiply", t0r)
            C(doc, "GraphicsRedraw2")
            hs = [h for h in damper_hits(sw, doc, mu) if h[0] == corner]
            res.append((deg, sum(h[1] for h in hs), ", ".join(sorted({h[2].split(' (')[0] for h in hs})) or "clear"))
        body.Transform2, rod.Transform2 = t0b, t0r
        print(f"   {corner} clock sweep (deg: mm3 touching frame):")
        for deg, v, w in res:
            print(f"      {deg:3d}: {v:8.1f}  {w}")
    C(sw, "CloseAllDocuments", True)   # nothing saved


if __name__ == "__main__":
    main(sys.argv[1].upper())
