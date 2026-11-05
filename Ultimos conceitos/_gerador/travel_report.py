r"""
Summarise travel_check.py output: contacts that matter over heave, with the known modelling
artefacts left out (they are the same in every concept and at every travel):
  - wishbone / tie-rod / toe-link bushes and the stand-off cylinders of the chassis_mounts part
    sitting in the frame tubes (placeholder bushes and tabs, by design of the macros);
  - joints between generated suspension parts (bushes, ball joints, rod ends overlap there);
  - the two halves of the real damper (body and shaft overlap by construction);
  - the real damper's eyes in the rocker / chassis-mounts brackets (eye wider than the 20 mm
    the macros assumed; listed once as a static finding).
Everything else is reported with its volume at each travel.

Usage:  python -I travel_report.py <run_X.json> [<run_Y.json> ...]
"""
import json
import sys
from collections import defaultdict

SUSP = ("UCA", "LCA", "upright", "tie_rod", "toe_link", "pushrod", "pullrod", "rocker", "wheel", "chassis_mounts")
# pairs joined by a mate in the corner assemblies (any other contact between them is real)
JOINTS = [{"UCA", "chassis_mounts"}, {"LCA", "chassis_mounts"}, {"UCA", "upright"}, {"LCA", "upright"},
          {"tie_rod", "upright"}, {"tie_rod", "chassis_mounts"}, {"toe_link", "upright"},
          {"toe_link", "chassis_mounts"}, {"rocker", "chassis_mounts"}, {"wheel", "upright"},
          {"pushrod", "rocker"}, {"pullrod", "rocker"}, {"pushrod", "LCA"}, {"pushrod", "UCA"},
          {"pullrod", "LCA"}, {"pullrod", "UCA"}]


def artefact(pair):
    a, b = pair.split(" <-> ")
    s = {a, b}
    if s == {"AMORTECEDOR_CORPO", "AMORTECEDOR_HASTE"}:
        return True
    if "Xasis 33.3" in s and (s - {"Xasis 33.3"}).pop() in ("UCA", "LCA", "tie_rod", "toe_link", "chassis_mounts"):
        return True
    if s <= set(SUSP) and s in JOINTS:       # parts mated to each other: their joints overlap
        return True
    if s & {"AMORTECEDOR_CORPO", "AMORTECEDOR_HASTE"} and s & {"rocker", "chassis_mounts"}:
        return True                          # eyes in their brackets
    if s == {"MOTOR_FIXADO", "Xasis 33.3"}:
        return True                          # engine on its mounts (0.5 mm3, not ours)
    return False


def main(paths):
    for p in paths:
        r = json.load(open(p, encoding="utf-8"))
        print(f"== concept {r['concept']}  (clock: " +
              ", ".join(f"{k} {v['deg']}" for k, v in r["clock"].items()) + ")")
        table = defaultdict(dict)
        for t in r["travel"]:
            tr = t["travel_mm"]
            e2e = t["eye_to_eye_mm"]
            print(f"   travel {tr:+5.0f} mm: eye-to-eye " +
                  ", ".join(f"{k} {v:.1f}" for k, v in e2e.items()) +
                  f"  (damper margin to 165 / 220 mm: {min(e2e.values()) - 165:.1f} / {220 - max(e2e.values()):.1f})")
            for row in t["interferences"]:
                if not artefact(row["pair"]):
                    key = f"{row['corner']}: {row['pair']}"
                    table[key][tr] = table[key].get(tr, 0) + row["mm3"]
        travels = [t["travel_mm"] for t in r["travel"]]
        if not table:
            print("   no contact outside the known artefacts at any travel")
        else:
            print("   contacts (mm3) at travel " + " ".join(f"{t:+6.0f}" for t in travels))
            for key in sorted(table):
                print(f"   {key:52s} " + " ".join(f"{table[key].get(t, 0):6.0f}" for t in travels))


if __name__ == "__main__":
    main(sys.argv[1:])
