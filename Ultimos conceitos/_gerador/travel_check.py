r"""
Heave check of a car copy (MONT_CHASSI_SUSP_<c>.SLDASM) with the real damper, in SolidWorks.

1. Reservoir clock: every corner's damper (body + shaft together) is turned about its own
   axis in 30 deg steps at ride; each corner gets the middle of its widest arc in which the
   damper body does not touch the frame (Xasis) or the engine.
2. Heave: a temporary distance mate (wheel centre to the chassis-mounts part's XY plane)
   in each corner sets the wheel travel; the corners are solved, then the car.
3. At each travel: interference detection, the damper's eye-to-eye length, and the wheel
   centre's actual travel.

Nothing is saved: the documents are closed without saving at the end.

Chassis-part frame (the macros'): forward -Y, left -X, UP -Z, metres.
Usage:  python -I travel_check.py <X|Y|Z> <out.json> [travel_mm ...]
"""
import json
import math
import sys
import time

import pythoncom
import win32com.client as wc
from win32com.client import VARIANT

OUT = r"C:\soft\FSAE\Suspension and Vehicle Dynamics\Ultimos conceitos"
CORNERS = ("FL", "FR", "RL", "RR")
NOTHING = VARIANT(pythoncom.VT_DISPATCH, None)


def C(o, n, *a):
    o._FlagAsMethod(n)
    return getattr(o, n)(*a)


def BL():
    return VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)


def arr(v):
    return VARIANT(pythoncom.VT_ARRAY | pythoncom.VT_R8, [float(x) for x in v])


class Car:
    def __init__(self, c):
        self.c = c
        self.sw = wc.GetActiveObject("SldWorks.Application")
        self.mu = C(self.sw, "GetMathUtility")
        C(self.sw, "CloseAllDocuments", True)
        self.car = C(self.sw, "OpenDoc6", OUT + rf"\MONT_CHASSI_SUSP_{c}.SLDASM", 2, 1, "", BL(), BL())
        self.susp = C(self.sw, "GetOpenDocumentByName", OUT + rf"\SUSP_{c}\SUSP_{c}.SLDASM")
        self.corner = {k: C(self.sw, "GetOpenDocumentByName", OUT + rf"\SUSP_{c}\SUSP_{k}.SLDASM") for k in CORNERS}
        self.comps = {}
        for k, d in self.corner.items():
            self.comps[k] = {x.Name2.rsplit("-", 1)[0]: x for x in C(d, "GetComponents", True)}
        self.capture()
        self.mates = {}
        self.wheel0 = {k: self.wheel_z(k) for k in CORNERS}

    # ---------------------------------------------------------------- geometry readback
    def world(self, comp, p):
        pt = C(self.mu, "CreatePoint", arr(p))
        return C(pt, "MultiplyTransform", comp.Transform2).ArrayData

    def ref_point(self, comp, name):
        f = C(comp, "FeatureByName", name)
        local = C(C(f, "GetSpecificFeature2"), "GetRefPoint").ArrayData   # in the part's frame
        return self.world(comp, local)

    def wheel_z(self, k):
        return self.ref_point(self.wheel(k), "P_CENTER")[2]

    def wheel(self, k):
        """The simplified wheel, or the team's wheel (RODA_ESTRELA) that swap_wheel.py put in."""
        return self.comps[k].get(f"SUSP_{k}_wheel") or self.comps[k]["RODA_ESTRELA"]

    def eye_to_eye(self, k):
        a = self.world(self.comps[k]["AMORTECEDOR_CORPO"], (0, 0.220, 0))
        b = self.world(self.comps[k]["AMORTECEDOR_HASTE"], (0, 0, 0))
        return math.dist(a, b) * 1000

    # ---------------------------------------------------------------- solving
    def rebuild(self):
        for d in self.corner.values():
            C(d, "ForceRebuild3", False)
        C(self.susp, "ForceRebuild3", False)
        C(self.car, "ForceRebuild3", False)

    def activate(self, d):
        C(self.sw, "ActivateDoc3", C(d, "GetTitle"), False, 1, BL())

    def car_dampers(self):
        """The damper components as instances in the car (nested): (body, shaft) per corner.
        Turning them here, without a re-solve, turns them in the car the interference check
        sees; turning them in the corner documents is undone by the next solve (their spin
        about the axis is free, the solver puts it back)."""
        # a fresh flat list every time: handles from GetChildren go stale after a re-solve
        out = {}
        for x in C(self.car, "GetComponents", False):
            parts = x.Name2.split("/")
            if len(parts) != 3 or parts[0] != f"SUSP_{self.c}-1":
                continue
            k, name = parts[1][5:7], parts[2].rsplit("-", 1)[0]
            if parts[1] == f"SUSP_{k}-1" and k in CORNERS and name in ("AMORTECEDOR_CORPO", "AMORTECEDOR_HASTE") \
                    and x.Transform2 is not None:          # the list can hold a stale duplicate
                out.setdefault(k, {})[name] = x
        out = {k: (v["AMORTECEDOR_CORPO"], v["AMORTECEDOR_HASTE"]) for k, v in out.items()}
        return out

    def capture(self):
        """Read the solved (unclocked) damper poses; call after every re-solve."""
        self.dampers = self.car_dampers()
        # plain tuples: holding the MathTransform objects across a rebuild breaks Multiply later
        self.base = {k: (tuple(b.Transform2.ArrayData), tuple(h.Transform2.ArrayData))
                     for k, (b, h) in self.dampers.items()}

    def clock(self, k, deg):
        """Turn corner k's damper (body and shaft together) about its own axis (local Y).
        SolidWorks transforms act on row vectors (p' = p.R + T), so a turn in the local
        frame is R' = Rot.R with T unchanged."""
        a = math.radians(deg)
        rot = ((math.cos(a), 0.0, -math.sin(a)), (0.0, 1.0, 0.0), (math.sin(a), 0.0, math.cos(a)))
        body, shaft = self.dampers[k]
        for comp, base in zip((body, shaft), self.base[k]):
            r = [base[0:3], base[3:6], base[6:9]]
            new = [sum(rot[i][m] * r[m][j] for m in range(3)) for i in range(3) for j in range(3)]
            comp.Transform2 = C(self.mu, "CreateTransform", arr(new + list(base[9:16])))

    @staticmethod
    def last_mate(d):
        f, last = C(d, "FirstFeature"), None
        while f is not None:
            if C(f, "GetTypeName2") == "MateGroup":
                s = C(f, "GetFirstSubFeature")
                while s is not None:
                    last = s
                    s = C(s, "GetNextSubFeature")
            f = C(f, "GetNextFeature")
        return last

    def set_travel(self, travel_mm):
        """Bump positive (wheel up = -Z in the chassis-part frame)."""
        for k, d in self.corner.items():
            self.activate(d)
            if k in self.mates:
                C(d, "ClearSelection2", True)
                C(self.mates[k], "Select2", False, 0)
                C(d, "EditDelete")
                del self.mates[k]
            if travel_mm is None:
                continue
            target = -(self.wheel0[k] - travel_mm / 1000.0)   # distance to the z = 0 plane
            wheel, mount = self.wheel(k), self.comps[k][f"SUSP_{k}_chassis_mounts"]
            plane = None
            f = C(C(mount, "GetModelDoc2"), "FirstFeature")
            while f is not None:
                if C(f, "GetTypeName2") == "RefPlane":
                    plane = f.Name        # the first default plane: XY (Front)
                    break
                f = C(f, "GetNextFeature")
            C(d, "ClearSelection2", True)
            ok = C(C(wheel, "FeatureByName", "P_CENTER"), "Select2", False, 1) and \
                C(C(mount, "FeatureByName", plane), "Select2", True, 1)
            err = BL()
            # flip=True keeps the wheel centre on its own side of the plane (False over-defines)
            m = C(d, "AddMate5", 5, 2, True, target, target, target, 1, 1, 0, 0, 0, False, False, 0, err) if ok else None
            feat = self.last_mate(d)
            if m is None or err.value != 1 or C(feat, "GetErrorCode"):
                raise RuntimeError(f"{k}: travel mate failed (selected={ok}, AddMate5 error {err.value}, "
                                   f"feature error {C(feat, 'GetErrorCode')})")
            self.mates[k] = feat
            C(d, "ClearSelection2", True)
        self.activate(self.car)
        self.rebuild()

    # ---------------------------------------------------------------- checks
    def interferences(self):
        idm = self.car.InterferenceDetectionManager
        idm.TreatCoincidenceAsInterference = False
        idm.TreatSubAssembliesAsComponents = False
        idm.IncludeMultibodyPartInterferences = False
        idm.IgnoreHiddenBodies = True
        out = []
        for it in C(idm, "GetInterferences") or []:
            names = [x.Name2 for x in it.Components]
            short = []
            corner = "-"
            for n in names:
                parts = n.split("/")
                if len(parts) >= 3:
                    corner = parts[1][5:7]
                short.append(parts[-1].rsplit("-", 1)[0].replace(f"SUSP_{corner}_", ""))
            out.append({"corner": corner, "pair": " <-> ".join(sorted(short)), "mm3": round(it.Volume * 1e9, 1)})
        C(idm, "Done")
        return out

    @staticmethod
    def damper_frame(rows):
        return {r["corner"]: r["mm3"] for r in rows
                if "AMORTECEDOR_CORPO" in r["pair"] and ("Xasis" in r["pair"] or "MOTOR" in r["pair"])}

    def close(self):
        C(self.sw, "CloseAllDocuments", True)   # discard: nothing is saved


def best_arc(clear):
    """clear: list of (deg, bool) every 30 deg -> middle of the widest clear arc (circular)."""
    n = len(clear)
    if all(ok for _, ok in clear):
        return 0, 360
    best, best_len = None, 0
    for i in range(n):
        if clear[i][1] and not clear[i - 1][1]:
            j, ln = i, 0
            while clear[j % n][1] and ln < n:
                ln += 1
                j += 1
            if ln > best_len:
                best_len, best = ln, (clear[i][0] + (ln - 1) * 15) % 360
    return (best if best is not None else 0), best_len * 30


def main(c, out, travels, given_clocks=None):
    t0 = time.time()
    car = Car(c)
    res = {"concept": c, "clock": {}, "travel": []}
    # the first detection after opening comes back short (components still loading): discard it
    car.interferences()
    # 1) reservoir clock at ride
    static = car.damper_frame(car.interferences())
    clocks = {k: 0 for k in CORNERS}
    if given_clocks:
        clocks.update(given_clocks)
        res["clock"] = {k: {"deg": clocks[k], "clear_arc_deg": "given (clock_sweep.py)"} for k in CORNERS}
    elif static:
        # turned in the car's context, no re-solve between angles (a re-solve undoes the turn)
        sweep = {k: [] for k in CORNERS}
        for deg in range(0, 360, 30):
            for k in CORNERS:
                car.clock(k, deg)
            hits = car.damper_frame(car.interferences())
            for k in CORNERS:
                sweep[k].append((deg, k not in hits))
        for k in CORNERS:
            deg, width = best_arc(sweep[k])
            clocks[k] = deg
            res["clock"][k] = {"deg": deg, "clear_arc_deg": width,
                               "clear_at": [d for d, ok in sweep[k] if ok]}
        for k in CORNERS:
            car.clock(k, 0)
    else:
        res["clock"] = {k: {"deg": 0, "clear_arc_deg": "not needed"} for k in CORNERS}
    print("clock:", {k: (v["deg"], v["clear_arc_deg"]) for k, v in res["clock"].items()}, flush=True)
    # 2) heave: solve, read the solved damper poses, turn each damper to its clock, check
    for t in travels:
        car.set_travel(t)
        if any(clocks.values()):        # the car's nested handles can go stale after a re-solve
            car.capture()
            for k in CORNERS:
                car.clock(k, clocks[k])
        car.interferences()            # warm-up: the first pass after a re-solve can come back short
        rows = car.interferences()
        rec = {
            "travel_mm": t,
            "wheel_travel_actual_mm": {k: round((car.wheel0[k] - car.wheel_z(k)) * 1000, 2) for k in CORNERS},
            "eye_to_eye_mm": {k: round(car.eye_to_eye(k), 1) for k in CORNERS},
            "interferences": rows,
        }
        res["travel"].append(rec)
        print(f"travel {t:+} mm: wheel {rec['wheel_travel_actual_mm']}, eye-eye {rec['eye_to_eye_mm']}, "
              f"{len(rows)} interferences", flush=True)
    car.set_travel(None)
    car.close()
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    rest = sys.argv[3:]
    given = {a.split("=")[0].upper(): float(a.split("=")[1]) for a in rest if "=" in a}
    main(sys.argv[1].upper(), sys.argv[2], [float(v) for v in rest if "=" not in v] or [-36, -18, 0, 18, 36],
         given)
