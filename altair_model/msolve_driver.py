"""Shared plumbing for driving Altair MotionSolve from the project venv.

Everything here is the machinery both Altair consumers need: locating the
Altair interpreter, building its environment, shelling out to
``msolve_corner.py``, and lifting a hardpoint CSV into ``vdcore`` models.

Two consumers:

  * ``validate_kinematics.py`` -- differences MotionSolve against DWSolver and
    exits non-zero on disagreement (the gate).
  * ``kpi_runner.py`` -- turns MotionSolve's positions into the KPI table the
    Streamlit app shows next to its vdcore column.

The project venv cannot import ``msolve`` and the Altair interpreter has no
pydantic/scipy, so every solve crosses a process boundary. That split is the
reason this module exists rather than the two callers each growing their own
copy of the environment recipe.

Frame: ISO 8855 -- X+ forward, Y+ LEFT, Z+ up. Units: mm, deg.
"""

from __future__ import annotations

import csv
import math
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from vdcore.models.hardpoint import Corner, Hardpoint, TirePackage

REPO = Path(__file__).resolve().parent.parent
MSOLVE_SCRIPT = Path(__file__).resolve().parent / "msolve_corner.py"
DEFAULT_CSV = REPO / "Geometry Summary" / "hardpoints_2027_merged.csv"

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


class AltairUnavailableError(RuntimeError):
    """Altair is not installed (or not where we expect it) on this machine.

    Kept distinct from a solve failure so callers can tell "cannot run here"
    apart from "ran and disagreed" -- the app greys the column out for the
    first and shows an error for the second.
    """


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #

def altair_python() -> Path:
    """Path to the Altair-bundled interpreter that can import ``msolve``."""
    return ALTAIR_ROOT / "common" / "python" / "python3.10" / "win64" / "python.exe"


def altair_available() -> bool:
    """True when MotionSolve can actually be driven from this machine."""
    return altair_python().is_file() and MSOLVE_SCRIPT.is_file()


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


# --------------------------------------------------------------------------- #
# input
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
                 static_toe_deg: float = 0.0,
                 *,
                 static_camber_deg: float | None = None,
                 loaded_radius_mm: float | None = None) -> Corner:
    """Lift one corner of the hardpoint CSV into a vdcore ``Corner``.

    ``static_camber_deg`` and ``loaded_radius_mm`` default to whatever the
    CSV's CONTACT_PATCH row implies. Pass them explicitly to build the corner
    from the same design inputs another consumer is using -- the file records
    camber only indirectly, at the file's rounding precision.
    """
    is_left = corner_id in ("FL", "RL")
    camber_deg, radius_mm = static_camber_from_csv(points, is_left)
    if static_camber_deg is not None:
        camber_deg = static_camber_deg
    if loaded_radius_mm is not None:
        radius_mm = loaded_radius_mm

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

def run_motionsolve(csv_path: Path, corner_id: str, spin_axis: np.ndarray,
                    droop: float, bump: float, steps: int,
                    workdir: Path) -> list[dict[str, float]]:
    """Run one corner sweep in MotionSolve. Return one dict per output step.

    The sweep runs from ``-droop`` to ``+bump`` in ``steps`` intervals, so the
    sampled travels are exactly ``linspace(-droop, +bump, steps + 1)``. Callers
    that need particular travels choose the grid to land on them rather than
    interpolating afterwards -- see ``kpi_runner.travel_grid``.

    Raises:
        AltairUnavailableError: Altair is not installed here.
        RuntimeError: MotionSolve ran but failed.
    """
    python_exe = altair_python()
    if not python_exe.is_file():
        raise AltairUnavailableError(
            f"Altair Python not found at {python_exe} -- is Altair 2026.1 installed?"
        )

    # The subprocess runs with cwd=workdir, so a relative CSV path would be
    # resolved against the wrong directory.
    csv_path = Path(csv_path).resolve()
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


def write_hardpoint_csv(path: Path,
                        points: dict[str, dict[str, tuple[float, float, float]]],
                        corners: Sequence[str]) -> Path:
    """Write a hardpoint CSV in the schema ``msolve_corner.py`` reads.

    Used to hand the app's in-memory geometry to the Altair interpreter, which
    has no way to receive a polars DataFrame.
    """
    order = (
        "UCA_IN_FRONT", "UCA_IN_REAR", "UCA_OUT",
        "LCA_IN_FRONT", "LCA_IN_REAR", "LCA_OUT",
        "TIE_ROD_IN", "TIE_ROD_OUT", "WHEEL_CENTER", "CONTACT_PATCH",
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corner", "point", "x_mm", "y_mm", "z_mm"])
        for corner in corners:
            for name in order:
                x, y, z = points[corner][name]
                writer.writerow([corner, name, f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])
    return path
