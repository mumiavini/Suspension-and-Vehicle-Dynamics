"""Cross-check vdcore's DWSolver against Altair MotionSolve on the same hardpoints.

Both solvers are given the identical corner geometry from the hardpoint CSV and
the identical wheel-travel sweep, then their answers are differenced.

They have nothing in common numerically:

  * ``vdcore.geometry.solver.DWSolver`` writes nine distance/sphere residuals in
    the ball-joint coordinates and drives them to zero with
    ``scipy.optimize.least_squares`` (trust-region, numerical Jacobian).
  * MotionSolve assembles revolute/spherical/universal joints into an index-3
    DAE and integrates it with DASPK.

So agreement is real evidence that the mechanism is being solved correctly, and
a disagreement localises to whichever quantity diverges.

What this does NOT independently check: the camber and toe *definitions*. Both
sides are run through ``DWSolver._extract_angles`` on purpose -- restating those
formulas here would only test whether they were copied correctly, and a sign
slip in a second copy would masquerade as a solver disagreement. MotionSolve's
contribution is the kinematics: where the upright ends up. The extraction is
vdcore's, applied to both.

Run from the project venv (it drives the Altair interpreter itself)::

    .venv\\Scripts\\python.exe altair_model\\validate_kinematics.py
    .venv\\Scripts\\python.exe altair_model\\validate_kinematics.py --corners FL FR RL RR

Frame: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up. Units: mm, deg.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from vdcore.geometry.solver import DWSolver  # noqa: E402
from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage  # noqa: E402

DEFAULT_CSV = REPO / "Geometry Summary" / "hardpoints_2027_merged.csv"
MSOLVE_SCRIPT = Path(__file__).resolve().parent / "msolve_corner.py"

ALTAIR_ROOT = Path(r"C:\Program Files\Altair\2026.1")

# The hardpoint CSV carries no provenance or tolerance columns. These stand in
# so the pydantic models validate; nothing in a kinematic cross-check reads
# either field, and neither is a claim about the real parts.
CSV_SOURCE = "design_intent"
CSV_TOL_MM = 0.0

CORNER_POINTS = {
    "uca_inboard_front": "UCA_IN_FRONT",
    "uca_inboard_rear": "UCA_IN_REAR",
    "uca_outboard": "UCA_OUT",
    "lca_inboard_front": "LCA_IN_FRONT",
    "lca_inboard_rear": "LCA_IN_REAR",
    "lca_outboard": "LCA_OUT",
    "tie_rod_inboard": "TIE_ROD_IN",
    "tie_rod_outboard": "TIE_ROD_OUT",
    "wheel_center": "WHEEL_CENTER",
}


# --------------------------------------------------------------------------- #
# building the corner from the CSV
# --------------------------------------------------------------------------- #

def read_csv_points(csv_path: Path) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Return {corner: {point_name: (x, y, z)}} in mm."""
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            corner = row["corner"].strip().upper()
            out.setdefault(corner, {})[row["point"].strip().upper()] = (
                float(row["x_mm"]), float(row["y_mm"]), float(row["z_mm"]),
            )
    return out


def static_camber_from_csv(points: dict[str, tuple[float, float, float]],
                           is_left: bool) -> tuple[float, float]:
    """Recover (static camber deg, loaded radius mm) from WHEEL_CENTER/CONTACT_PATCH.

    Inverts vdcore.geometry.solver's contact-patch construction, where for a
    left corner ``cp_y = wc_y - r*tan(camber)`` and ``cp_z = wc_z - r``. Taking
    the camber from the geometry keeps the CSV the single source of truth
    instead of restating -1.50 deg as a literal here.
    """
    wc = points["WHEEL_CENTER"]
    cp = points["CONTACT_PATCH"]
    radius = wc[2] - cp[2]
    if radius <= 0.0:
        raise ValueError("CONTACT_PATCH is not below WHEEL_CENTER")
    dy = cp[1] - wc[1]
    camber = math.degrees(math.atan2(dy, radius))
    return (-camber if is_left else camber), radius


def build_corner(corner_id: str, points: dict[str, tuple[float, float, float]],
                 static_toe_deg: float) -> Corner:
    is_left = corner_id in ("FL", "RL")
    camber_deg, radius_mm = static_camber_from_csv(points, is_left)

    hardpoints = {
        field: Hardpoint(
            name=name,
            x_mm=points[name][0], y_mm=points[name][1], z_mm=points[name][2],
            source=CSV_SOURCE, tol_mm=CSV_TOL_MM,
        )
        for field, name in CORNER_POINTS.items()
    }
    return Corner(
        corner_id=corner_id,
        tire=TirePackage(
            loaded_radius_mm=radius_mm, source=CSV_SOURCE, tol_mm=CSV_TOL_MM,
        ),
        static_camber_deg=camber_deg,
        static_toe_deg_per_side=static_toe_deg,
        **hardpoints,
    )


# --------------------------------------------------------------------------- #
# driving MotionSolve
# --------------------------------------------------------------------------- #

def altair_env() -> dict[str, str]:
    """Environment for the MotionSolve interpreter (mirrors motionsolve_jupyter.bat)."""
    python_home = ALTAIR_ROOT / "common" / "python" / "python3.10" / "win64"
    msolve_base = ALTAIR_ROOT / "hwsolvers" / "motionsolve"
    dll_dir = msolve_base / "bin" / "win64"

    env = dict(os.environ)
    env["PYTHONHOME"] = str(python_home)
    env["MSOLVE_BASE_DIR"] = str(msolve_base)
    env["NUSOL_DLL_DIR"] = str(dll_dir)
    env["PYTHONPATH"] = os.pathsep.join([str(msolve_base), str(dll_dir)])
    env["PATH"] = os.pathsep.join(
        [str(dll_dir), str(msolve_base), str(python_home), env.get("PATH", "")]
    )
    return env


def run_motionsolve(csv_path: Path, corner_id: str, spin_axis: np.ndarray,
                    droop: float, bump: float, steps: int,
                    workdir: Path) -> list[dict[str, float]]:
    """Run the corner sweep in MotionSolve. Return one dict per output step."""
    python_exe = ALTAIR_ROOT / "common" / "python" / "python3.10" / "win64" / "python.exe"
    if not python_exe.is_file():
        raise FileNotFoundError(
            f"Altair Python not found at {python_exe} -- is Altair 2026.1 installed?"
        )

    out_csv = workdir / f"msolve_{corner_id.lower()}.csv"
    # `--opt=value` rather than `--opt value`: the spin axis round-trips through
    # the upright body frame, so a nominally zero component can come back as
    # -1e-17 and argparse would read the leading minus as another option.
    cmd = [
        str(python_exe), str(MSOLVE_SCRIPT),
        f"--csv={csv_path}",
        f"--corner={corner_id}",
        "--spin=" + ",".join(f"{v:.17g}" for v in spin_axis),
        f"--droop={droop}",
        f"--bump={bump}",
        f"--steps={steps}",
        f"--out={out_csv}",
    ]
    proc = subprocess.run(
        cmd, cwd=workdir, env=altair_env(),
        capture_output=True, text=True, timeout=900,
    )
    if proc.returncode != 0 or not out_csv.is_file():
        raise RuntimeError(
            f"MotionSolve failed for {corner_id} (exit {proc.returncode}).\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}"
        )

    with open(out_csv, newline="", encoding="utf-8") as handle:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]

    # MotionSolve emits the initial state twice (assembly, then first step).
    deduped: list[dict[str, float]] = []
    for row in rows:
        if deduped and abs(row["time"] - deduped[-1]["time"]) < 1e-12:
            continue
        deduped.append(row)
    return deduped


# --------------------------------------------------------------------------- #
# comparison
# --------------------------------------------------------------------------- #

def compare_corner(corner_id: str, points: dict[str, tuple[float, float, float]],
                   csv_path: Path, static_toe_deg: float,
                   droop: float, bump: float, steps: int,
                   workdir: Path) -> dict[str, object]:
    corner = build_corner(corner_id, points, static_toe_deg)
    solver = DWSolver(corner)

    # vdcore's own static spin axis, so both sides share one definition of the
    # wheel plane. Reaching in for the definition (not the answer) is deliberate.
    spin_static = solver._reconstruct_spin_axis(
        solver._ubj_0, solver._lbj_0, solver._tro_0
    )

    ms_rows = run_motionsolve(
        csv_path, corner_id, spin_static, droop, bump, steps, workdir
    )

    records = []
    for row in ms_rows:
        travel = row["travel_mm"]

        ubj_ms = np.array([row["UCA_OUT_x"], row["UCA_OUT_y"], row["UCA_OUT_z"]])
        lbj_ms = np.array([row["LCA_OUT_x"], row["LCA_OUT_y"], row["LCA_OUT_z"]])
        tro_ms = np.array([row["TIE_ROD_OUT_x"], row["TIE_ROD_OUT_y"], row["TIE_ROD_OUT_z"]])
        wc_ms = np.array([
            row["WHEEL_CENTER_x"], row["WHEEL_CENTER_y"], row["WHEEL_CENTER_z"],
        ])
        spin_probe = np.array([row["SPIN_x"], row["SPIN_y"], row["SPIN_z"]])
        spin_ms = spin_probe - wc_ms
        spin_ms = spin_ms / np.linalg.norm(spin_ms)

        camber_ms, toe_ms, caster_ms, kpi_ms = solver._extract_angles(
            ubj_ms, lbj_ms, spin_ms
        )

        dw = solver.solve(wheel_travel_mm=travel)
        if not dw.converged:
            raise RuntimeError(
                f"{corner_id}: DWSolver did not converge at travel={travel:.3f} mm"
            )

        # DWSolver holds the wheel and drops the chassis; MotionSolve holds the
        # chassis and lifts the wheel. Undo that rigid vertical offset before
        # differencing positions. Angles are unaffected by it.
        shift = np.array([0.0, 0.0, travel])
        ubj_dw = np.array([dw.ubj.x_mm, dw.ubj.y_mm, dw.ubj.z_mm]) + shift
        lbj_dw = np.array([dw.lbj.x_mm, dw.lbj.y_mm, dw.lbj.z_mm]) + shift
        tro_dw = np.array([dw.tro.x_mm, dw.tro.y_mm, dw.tro.z_mm]) + shift
        wc_dw = np.array([
            dw.wheel_center.x_mm, dw.wheel_center.y_mm, dw.wheel_center.z_mm,
        ]) + shift

        records.append({
            "travel_mm": travel,
            "camber_ms": camber_ms, "camber_dw": dw.camber_deg,
            "toe_ms": toe_ms, "toe_dw": dw.toe_deg_per_side,
            "caster_ms": caster_ms, "caster_dw": dw.caster_deg,
            "kpi_ms": kpi_ms, "kpi_dw": dw.kpi_deg,
            "wc_y_ms": wc_ms[1], "wc_y_dw": wc_dw[1],
            "wc_z_ms": wc_ms[2], "wc_z_dw": wc_dw[2],
            "d_ubj": float(np.linalg.norm(ubj_ms - ubj_dw)),
            "d_lbj": float(np.linalg.norm(lbj_ms - lbj_dw)),
            "d_tro": float(np.linalg.norm(tro_ms - tro_dw)),
            "d_wc": float(np.linalg.norm(wc_ms - wc_dw)),
        })

    return {"corner_id": corner_id, "corner": corner, "records": records}


def _stats(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.max(np.abs(arr))), float(np.sqrt(np.mean(arr ** 2)))


def report(result: dict[str, object], verbose: bool) -> list[str]:
    corner_id = result["corner_id"]
    corner: Corner = result["corner"]  # type: ignore[assignment]
    records: list[dict[str, float]] = result["records"]  # type: ignore[assignment]

    lines: list[str] = []
    lines.append("")
    lines.append(f"  {corner_id}  ({len(records)} sweep points, "
                 f"static camber {corner.static_camber_deg:+.3f} deg, "
                 f"loaded radius {corner.tire.loaded_radius_mm:.1f} mm)")
    lines.append("  " + "-" * 74)

    if verbose:
        lines.append(f"  {'travel':>8} {'camber MS':>11} {'camber DW':>11} {'d':>9}"
                     f" {'toe MS':>10} {'toe DW':>10} {'d':>9}")
        for r in records:
            lines.append(
                f"  {r['travel_mm']:8.2f} {r['camber_ms']:11.5f} {r['camber_dw']:11.5f}"
                f" {r['camber_ms'] - r['camber_dw']:9.2e}"
                f" {r['toe_ms']:10.5f} {r['toe_dw']:10.5f}"
                f" {r['toe_ms'] - r['toe_dw']:9.2e}"
            )
        lines.append("  " + "-" * 74)

    checks = (
        ("camber", "deg", [r["camber_ms"] - r["camber_dw"] for r in records]),
        ("toe/side", "deg", [r["toe_ms"] - r["toe_dw"] for r in records]),
        ("caster", "deg", [r["caster_ms"] - r["caster_dw"] for r in records]),
        ("KPI", "deg", [r["kpi_ms"] - r["kpi_dw"] for r in records]),
        ("wheel centre Y", "mm", [r["wc_y_ms"] - r["wc_y_dw"] for r in records]),
        ("wheel centre Z", "mm", [r["wc_z_ms"] - r["wc_z_dw"] for r in records]),
        ("UBJ position", "mm", [r["d_ubj"] for r in records]),
        ("LBJ position", "mm", [r["d_lbj"] for r in records]),
        ("TRO position", "mm", [r["d_tro"] for r in records]),
    )
    lines.append(f"  {'quantity':<16} {'unit':<5} {'max |MS - DW|':>15} {'RMS':>13}")
    for label, unit, deltas in checks:
        peak, rms = _stats(deltas)
        lines.append(f"  {label:<16} {unit:<5} {peak:15.3e} {rms:13.3e}")

    # Camber gain from each solver's own curve, as a physical sanity check that
    # the sweep actually did something.
    travels = np.array([r["travel_mm"] for r in records])
    for tag in ("ms", "dw"):
        camber = np.array([r[f"camber_{tag}"] for r in records])
        slope = float(np.polyfit(travels, camber, 1)[0])
        lines.append(f"  camber gain ({tag.upper()}): {slope:+.5f} deg/mm")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--corners", nargs="+", default=["FL"],
                        choices=["FL", "FR", "RL", "RR"])
    parser.add_argument("--droop", type=float, default=25.0)
    parser.add_argument("--bump", type=float, default=25.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--static-toe", type=float, default=0.0,
                        help="static toe per side in deg; not recoverable from "
                             "the CSV, defaults to the design value of 0")
    parser.add_argument("--verbose", action="store_true",
                        help="print the full sweep table, not just the summary")
    parser.add_argument("--tol-deg", type=float, default=1e-4,
                        help="pass/fail threshold on angle deltas")
    parser.add_argument("--tol-mm", type=float, default=1e-3,
                        help="pass/fail threshold on position deltas")
    args = parser.parse_args(argv)

    all_points = read_csv_points(args.csv)

    print("=" * 78)
    print("  vdcore DWSolver  vs  Altair MotionSolve  --  kinematic cross-check")
    print("=" * 78)
    print(f"  hardpoints : {args.csv}")
    print(f"  sweep      : {-args.droop:+.1f} to {+args.bump:+.1f} mm wheel travel, "
          f"{args.steps} steps")

    worst_angle = 0.0
    worst_position = 0.0
    lines: list[str] = []

    with tempfile.TemporaryDirectory(prefix="msolve_validate_") as tmp:
        for corner_id in args.corners:
            if corner_id not in all_points:
                raise SystemExit(f"{args.csv.name} has no {corner_id} corner")
            result = compare_corner(
                corner_id, all_points[corner_id], args.csv, args.static_toe,
                args.droop, args.bump, args.steps, Path(tmp),
            )
            lines.extend(report(result, args.verbose))

            records: list[dict[str, float]] = result["records"]  # type: ignore[assignment]
            for key in ("camber", "toe", "caster", "kpi"):
                peak, _ = _stats([r[f"{key}_ms"] - r[f"{key}_dw"] for r in records])
                worst_angle = max(worst_angle, peak)
            for key in ("d_ubj", "d_lbj", "d_tro", "d_wc"):
                peak, _ = _stats([r[key] for r in records])
                worst_position = max(worst_position, peak)

    print("\n".join(lines))
    print("")
    print("=" * 78)
    print(f"  worst angle disagreement    : {worst_angle:.3e} deg  "
          f"(threshold {args.tol_deg:.0e})")
    print(f"  worst position disagreement : {worst_position:.3e} mm   "
          f"(threshold {args.tol_mm:.0e})")

    ok = worst_angle <= args.tol_deg and worst_position <= args.tol_mm
    print(f"  RESULT: {'AGREE' if ok else 'DISAGREE'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
