"""Inspect a SolidWorks part through COM: configs, features, bodies, bbox, ref geometry."""
import sys
import pythoncom
import win32com.client as wc
from win32com.client import VARIANT


def C(o, name, *a):
    """Call a COM method on a late-bound object (pywin32 would treat a no-arg method as a property)."""
    o._FlagAsMethod(name)
    return getattr(o, name)(*a)

path = sys.argv[1]
sw = wc.GetActiveObject("SldWorks.Application")
err = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
wrn = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
doc = sw.GetOpenDocumentByName(path)
if doc is None:
    doc = sw.OpenDoc6(path, 1, 1 | 2, "", err, wrn)  # silent + readonly
print("open err", err.value, "warn", wrn.value, "title", C(doc, "GetTitle"))
print("configs:", C(doc, "GetConfigurationNames"))
print("active config:", doc.ConfigurationManager.ActiveConfiguration.Name)
f = C(doc, "FirstFeature")
while f is not None:
    t = C(f, "GetTypeName2")
    if t not in ("MaterialFolder", "DetailCabinet", "HistoryFolder", "SensorFolder",
                 "DocsFolder", "CommentsFolder", "FavoriteFolder", "SelectionSetFolder",
                 "EqnFolder", "InkMarkupFolder", "EnvFolder", "SolidBodyFolder",
                 "SurfaceBodyFolder", "MateReferenceGroupFolder"):
        extra = ""
        if t == "RefPoint":
            try:
                p = C(C(f, "GetSpecificFeature2"), "GetRefPoint").ArrayData
                extra = " @ " + ", ".join(f"{v*1000:.2f}" for v in p) + " mm"
            except Exception as e:
                extra = f" (? {e})"
        print(f"  {f.Name:40s} {t}{extra}  supp={C(f, "IsSuppressed")}")
    f = C(f, "GetNextFeature")
bodies = doc.GetBodies2(0, False)
print("solid bodies:", 0 if bodies is None else len(bodies))
for b in bodies or []:
    bx = C(b, "GetBodyBox")
    print("  ", b.Name, "box mm:", [round(v * 1000, 2) for v in bx])
mp = C(doc.Extension, "CreateMassProperty")
if mp:
    print("volume mm3:", round(mp.Volume * 1e9, 1), "CG mm:", [round(v * 1000, 2) for v in mp.CenterOfMass])
