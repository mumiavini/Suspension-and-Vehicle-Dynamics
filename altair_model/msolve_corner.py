"""Solve one suspension corner as a kinematic mechanism in Altair MotionSolve.

Builds a double-wishbone corner from the hardpoint CSV as a rigid-body
mechanism, drives the wheel centre through a vertical travel sweep, and writes
the resulting outboard-point positions to CSV. Nothing is interpreted here --
camber and toe are derived downstream by `validate_kinematics.py`, using
`vdcore`'s own extraction, so the only thing this file contributes is
MotionSolve's answer to *where the points go*.

Topology (RSSU + the upright, 1 DOF before the driving motion):

    LCA    revolute to chassis about LCA_IN_FRONT -- LCA_IN_REAR   (5 constraints)
    UCA    revolute to chassis about UCA_IN_FRONT -- UCA_IN_REAR   (5)
    LBJ    spherical, LCA to upright, at LCA_OUT                   (3)
    UBJ    spherical, UCA to upright, at UCA_OUT                   (3)
    TRO    spherical, tie rod to upright, at TIE_ROD_OUT           (3)
    TRI    universal, tie rod to chassis, at TIE_ROD_IN            (4)

    4 moving bodies x 6 = 24 DOF, 23 constraints => 1 DOF.
    One displacement motion on wheel-centre Z closes it to 0, which is what
    a KINEMATICS analysis requires.

The universal at the inner tie rod (rather than a second spherical) removes the
rod's idle spin about its own axis. Both of its cross axes are perpendicular to
the rod, so the rod still swings freely in the two directions that matter. This
is the same RSUR trick the shipped msolve 4bar example uses.

Mass and inertia are nominal placeholders: a KINEMATICS analysis is driven
entirely by constraints and never touches them. They are NOT a claim about the
real parts, which have not been massed.

Frame: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up. Units: MMKS.

Runs under Altair's Python 3.10 with the MotionSolve environment set. Drive it
from the project venv rather than calling it directly -- `msolve_driver.py`
owns the environment recipe, and its two callers are `validate_kinematics.py`
(differences this against DWSolver and exits non-zero on disagreement) and
`kpi_runner.py` (turns these positions into the app's Altair column).
"""

import argparse
import csv
import math
import sys
from pathlib import Path

from msolve import (
    Debug,
    Joint,
    Marker,
    Model,
    Motion,
    Output,
    Part,
    Point,
    Request,
    Units,
)

CORNERS = ("FL", "FR", "RL", "RR")

POINT_NAMES = (
    "UCA_IN_FRONT",
    "UCA_IN_REAR",
    "UCA_OUT",
    "LCA_IN_FRONT",
    "LCA_IN_REAR",
    "LCA_OUT",
    "TIE_ROD_IN",
    "TIE_ROD_OUT",
    "WHEEL_CENTER",
    "CONTACT_PATCH",
)

# Points tracked through the sweep. WHEEL_CENTER and the spin probe give the
# wheel frame; the three ball joints give the upright pose.
PROBES = ("UCA_OUT", "LCA_OUT", "TIE_ROD_OUT", "WHEEL_CENTER", "SPIN")


# --------------------------------------------------------------------------- #
# plain-tuple vector helpers (no numpy dependency in the Altair interpreter)
# --------------------------------------------------------------------------- #

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def mid(a, b):
    return scale(add(a, b), 0.5)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def unit(a):
    n = norm(a)
    if n < 1e-12:
        raise ValueError("cannot normalise a zero-length vector")
    return scale(a, 1.0 / n)


def any_perpendicular(axis):
    """Return a unit vector perpendicular to `axis`, chosen stably."""
    a = unit(axis)
    # Cross with whichever global axis is least aligned, so the result is
    # never near-degenerate.
    ref = (0.0, 0.0, 1.0) if abs(a[2]) < 0.9 else (1.0, 0.0, 0.0)
    return unit(cross(a, ref))


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #

def load_corner(csv_path, corner):
    """Return {point_name: (x, y, z)} for one corner, in mm."""
    csv_path = Path(csv_path)
    points = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["corner"].strip().upper() != corner:
                continue
            points[row["point"].strip().upper()] = (
                float(row["x_mm"]),
                float(row["y_mm"]),
                float(row["z_mm"]),
            )

    missing = [name for name in POINT_NAMES if name not in points]
    if missing:
        raise ValueError(f"{csv_path.name}: {corner} missing {', '.join(missing)}")
    return points


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

def build_corner_model(hp, corner, spin_axis, probe_arm_mm=100.0):
    """Assemble the corner mechanism. Return (model, {probe_name: Request})."""
    model = Model(output=f"corner_{corner.lower()}")
    Units(mass="KILOGRAM", length="MILLIMETER", time="SECOND", force="NEWTON")
    Debug(eprint=False, verbose=False, screen_output=False)

    def P(v):
        return Point(v[0], v[1], v[2])

    chassis = Part(ground=True, label=f"{corner} chassis")
    oxyz = Marker(body=chassis, label="global CS")

    # --- moving bodies -------------------------------------------------------
    # Mass/inertia are placeholders; KINEMATICS never reads them.
    def body(label, centroid):
        part = Part(mass=1.0, ip=[1.0e3, 1.0e3, 1.0e3, 0.0, 0.0, 0.0], label=label)
        part.cm = Marker(body=part, qp=P(centroid), label=f"{label} cm")
        return part

    uca_if, uca_ir, uca_o = hp["UCA_IN_FRONT"], hp["UCA_IN_REAR"], hp["UCA_OUT"]
    lca_if, lca_ir, lca_o = hp["LCA_IN_FRONT"], hp["LCA_IN_REAR"], hp["LCA_OUT"]
    tri, tro = hp["TIE_ROD_IN"], hp["TIE_ROD_OUT"]
    wc = hp["WHEEL_CENTER"]

    uca = body(f"{corner} UCA", scale(add(add(uca_if, uca_ir), uca_o), 1.0 / 3.0))
    lca = body(f"{corner} LCA", scale(add(add(lca_if, lca_ir), lca_o), 1.0 / 3.0))
    upright = body(f"{corner} upright", scale(add(add(uca_o, lca_o), tro), 1.0 / 3.0))
    tierod = body(f"{corner} tie rod", mid(tri, tro))

    # --- wishbone revolutes --------------------------------------------------
    def revolute(part, in_front, in_rear, label):
        """Hinge `part` to the chassis about the inboard pivot axis."""
        pivot = mid(in_front, in_rear)
        axis = sub(in_front, in_rear)
        z_pt = add(pivot, unit(axis))
        x_pt = add(pivot, any_perpendicular(axis))
        m_part = Marker(body=part, qp=P(pivot), zp=P(z_pt), xp=P(x_pt), label=f"{label} i")
        m_grnd = Marker(body=chassis, qp=P(pivot), zp=P(z_pt), xp=P(x_pt), label=f"{label} j")
        return Joint(type="REVOLUTE", i=m_part, j=m_grnd, label=label)

    revolute(uca, uca_if, uca_ir, f"{corner} UCA pivot")
    revolute(lca, lca_if, lca_ir, f"{corner} LCA pivot")

    # --- ball joints ---------------------------------------------------------
    def spherical(part_a, part_b, at, label):
        m_a = Marker(body=part_a, qp=P(at), label=f"{label} i")
        m_b = Marker(body=part_b, qp=P(at), label=f"{label} j")
        return Joint(type="SPHERICAL", i=m_a, j=m_b, label=label)

    spherical(uca, upright, uca_o, f"{corner} UBJ")
    spherical(lca, upright, lca_o, f"{corner} LBJ")
    spherical(tierod, upright, tro, f"{corner} TRO")

    # --- inner tie rod: universal, cross axes both perpendicular to the rod --
    rod = unit(sub(tro, tri))
    cross1 = any_perpendicular(rod)          # perpendicular to the rod
    cross2 = unit(cross(rod, cross1))        # perpendicular to the rod AND to cross1
    m_rod = Marker(
        body=tierod, qp=P(tri), zp=P(add(tri, cross1)), xp=P(add(tri, rod)),
        label=f"{corner} TRI i",
    )
    m_grnd = Marker(
        body=chassis, qp=P(tri), zp=P(add(tri, cross2)), xp=P(add(tri, rod)),
        label=f"{corner} TRI j",
    )
    Joint(type="UNIVERSAL", i=m_rod, j=m_grnd, label=f"{corner} TRI")

    # --- driving motion: wheel centre Z ---------------------------------------
    # DWSolver holds the wheel at its static height and heaves the chassis down;
    # here the chassis is fixed and the wheel rises. With no roll the two differ
    # by a rigid vertical translation, which leaves camber and toe untouched.
    m_wc_up = Marker(body=upright, qp=P(wc), label=f"{corner} WC motion i")
    m_wc_gr = Marker(body=chassis, qp=P(wc), label=f"{corner} WC motion j")
    motion = Motion(
        i=m_wc_up,
        j=m_wc_gr,
        jtype="TRANSLATION",
        direction="Z",
        dtype="DISPLACEMENT",
        function="TRAVEL_PLACEHOLDER",
        label=f"{corner} wheel travel",
    )

    # --- probes ---------------------------------------------------------------
    # All rigidly attached to the upright, so tracking them recovers its pose.
    probe_points = {
        "UCA_OUT": uca_o,
        "LCA_OUT": lca_o,
        "TIE_ROD_OUT": tro,
        "WHEEL_CENTER": wc,
        # Offset along the static spin axis: probe - WC is the spin axis at
        # every step, transported by the upright's own rigid motion.
        "SPIN": add(wc, scale(unit(spin_axis), probe_arm_mm)),
    }

    requests = {}
    for name, location in probe_points.items():
        marker = Marker(body=upright, qp=P(location), label=f"{corner} probe {name}")
        requests[name] = Request(
            type="EXPRESSION",
            label=f"{corner}_{name}",
            f1=f"DX({marker.id},{oxyz.id})",
            f2=f"DY({marker.id},{oxyz.id})",
            f3=f"DZ({marker.id},{oxyz.id})",
        )

    Output(reqsave=True)
    return model, motion, requests


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #

def run_sweep(csv_path, corner, spin_axis, droop_mm, bump_mm, steps, out_path):
    hp = load_corner(csv_path, corner)
    model, motion, requests = build_corner_model(hp, corner, spin_axis)

    # Ramp travel linearly from -droop to +bump over t in [0, 1], so the sample
    # spacing in travel is uniform and no interpolation is needed downstream.
    span = droop_mm + bump_mm
    motion.function = f"{-droop_mm} + {span}*time"

    results = model.simulate(
        type="KINEMATICS", end=1.0, steps=steps, returnResults=True,
    )
    if results is None:
        raise RuntimeError("MotionSolve returned no results")

    tracks = {}
    for name, request in requests.items():
        res = results.getObject(request)
        tracks[name] = (
            list(res.getComponent(0)),
            list(res.getComponent(1)),
            list(res.getComponent(2)),
        )
        times = list(res.times)

    header = ["time", "travel_mm"]
    for name in PROBES:
        header += [f"{name}_x", f"{name}_y", f"{name}_z"]

    out_path = Path(out_path)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i, t in enumerate(times):
            row = [f"{t:.10g}", f"{-droop_mm + span * t:.10g}"]
            for name in PROBES:
                xs, ys, zs = tracks[name]
                row += [f"{xs[i]:.10g}", f"{ys[i]:.10g}", f"{zs[i]:.10g}"]
            writer.writerow(row)

    print(f"{corner}: {len(times)} steps -> {out_path}")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="hardpoint CSV")
    parser.add_argument("--corner", required=True, choices=CORNERS)
    parser.add_argument(
        "--spin", required=True,
        help="static spin-axis unit vector 'x,y,z' in ISO 8855, from vdcore "
             "(supplied by validate_kinematics.py so there is one definition)",
    )
    parser.add_argument("--droop", type=float, default=25.0)
    parser.add_argument("--bump", type=float, default=25.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    spin = tuple(float(v) for v in args.spin.split(","))
    if len(spin) != 3:
        parser.error("--spin needs three comma-separated components")

    run_sweep(
        args.csv, args.corner, spin,
        args.droop, args.bump, args.steps, args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
