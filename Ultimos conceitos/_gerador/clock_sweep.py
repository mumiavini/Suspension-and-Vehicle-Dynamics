r"""
Reservoir clock sweep of a car copy at ride height: every corner's real damper (body and
shaft together) is turned about its own axis in steps, in the car's context (in memory,
nothing saved), and every contact of the damper outside the known modelling artefacts
(travel_report.artefact) is recorded per corner and angle.

Angle 0 is the damper as built (eye pin parallel to the rocker axis, reservoir toward the
part's -X); the turn is the one travel_check.Car.clock applies and build_susp --clock bakes in.

Usage:  python -I clock_sweep.py <X|Y|Z> <out.json> [step_deg]
"""
import json
import sys
import time

from travel_check import CORNERS, Car
from travel_report import artefact


def damper_rows(rows):
    out = {k: [] for k in CORNERS}
    for r in rows:
        if r["corner"] in out and "AMORTECEDOR" in r["pair"] and not artefact(r["pair"]):
            out[r["corner"]].append((r["pair"], r["mm3"]))
    return out


def main(c, out, step):
    t0 = time.time()
    car = Car(c)
    car.interferences()                      # warm-up (the first pass comes back short)
    res = {"concept": c, "step_deg": step, "angles": {}}
    for deg in range(0, 360, step):
        for k in CORNERS:
            car.clock(k, deg)
        car.interferences()
        hits = damper_rows(car.interferences())
        res["angles"][deg] = hits
        print(deg, {k: round(sum(v for _, v in h), 1) for k, h in hits.items()}, flush=True)
    car.close()
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1].upper(), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 15)
