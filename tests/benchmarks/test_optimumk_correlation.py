"""Cross-validation against Optimum Kinematics export.

This test loads a CSV exported from Optimum Kinematics and compares
vdcore solver results against it. The CSV is produced by entering the
same hardpoint coordinates into OptimumK (via the export script at
scripts/export_hardpoints_for_optimumk.py) and running a wheel travel
sweep.

The CSV must live at tests/benchmarks/data/optimumk_sweep.csv with
required columns:
    wheel_travel_mm, camber_deg, caster_deg, kpi_deg, rc_height_mm

Plus EXACTLY ONE of the following toe columns:
    toe_per_side_deg   (one wheel's toe angle)
    toe_total_deg      (sum of left + right = 2x per-side for symmetric)

Do NOT name the column 'toe_deg' -- that is ambiguous and the loader
will reject it. OptimumK reports total toe per axle by default; our
solver returns toe_deg_per_side. A factor-of-2 error here would look
like a plausible offset, not an obvious failure.

RC height (rc_height_mm) is REQUIRED, not optional. RC migration is
the quantity most likely to be subtly wrong in our construction, and
the one that needs validation most for the design work.

RC is compared both at static and through the travel sweep.
roll_centre_height() accepts solved joint positions, so the
RC construction uses displaced UBJ/LBJ and contact patch at
each travel point.

Sign conventions in the CSV must be ISO 8855 (this project's native):
  - Negative camber = top inboard
  - Positive toe = toe-in
  - Positive caster = rearward tilt at top
  - Positive KPI = inboard tilt at top

Tolerances:
  - Angles: 0.05 deg
  - RC height: 2 mm

Test structure:
  1. TestOptimumKStatic -- compare static values first (zero travel).
     If the static point disagrees, the curve comparison tells us
     nothing useful.
  2. TestOptimumKSweep -- compare angle curves over travel.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from vdcore.analysis.roll_centre import roll_centre_height
from vdcore.geometry.solver import DWSolver, SolverResult
from vdcore.models.hardpoint import Axle, Corner, Hardpoint, TirePackage

_CSV_PATH = Path(__file__).parent / "data" / "optimumk_sweep.csv"
_CSV_EXISTS = _CSV_PATH.exists()

_ANGLE_TOL_DEG = 0.05
_RC_TOL_MM = 2.0


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire() -> TirePackage:
    return TirePackage(loaded_radius_mm=254.0, source="cad", tol_mm=1.0)


def _correlation_fl() -> Corner:
    """Front-left corner -- FSAE-representative geometry.

    Same geometry as tests/benchmarks/test_fsae_representative.py.
    This is the geometry entered into OptimumK.
    """
    return Corner(
        corner_id="FL",
        uca_inboard_front=_hp("UCA_IF", 70, 185, 290),
        uca_inboard_rear=_hp("UCA_IR", -70, 185, 290),
        uca_outboard=_hp("UCA_O", -12, 530, 312),
        lca_inboard_front=_hp("LCA_IF", 110, 115, 95),
        lca_inboard_rear=_hp("LCA_IR", -90, 115, 95),
        lca_outboard=_hp("LCA_O", 10, 590, 72),
        tie_rod_inboard=_hp("TR_I", -80, 150, 93.3),
        tie_rod_outboard=_hp("TR_O", -70, 555, 73.9),
        wheel_center=_hp("WC", 0, 610, 254),
        tire=_tire(),
        static_camber_deg=-2.0,
        static_toe_deg_per_side=0.1,
    )


def _correlation_fr() -> Corner:
    """Front-right mirror."""
    fl = _correlation_fl()
    return Corner(
        corner_id="FR",
        uca_inboard_front=_hp("UCA_IF", fl.uca_inboard_front.x_mm, -fl.uca_inboard_front.y_mm, fl.uca_inboard_front.z_mm),
        uca_inboard_rear=_hp("UCA_IR", fl.uca_inboard_rear.x_mm, -fl.uca_inboard_rear.y_mm, fl.uca_inboard_rear.z_mm),
        uca_outboard=_hp("UCA_O", fl.uca_outboard.x_mm, -fl.uca_outboard.y_mm, fl.uca_outboard.z_mm),
        lca_inboard_front=_hp("LCA_IF", fl.lca_inboard_front.x_mm, -fl.lca_inboard_front.y_mm, fl.lca_inboard_front.z_mm),
        lca_inboard_rear=_hp("LCA_IR", fl.lca_inboard_rear.x_mm, -fl.lca_inboard_rear.y_mm, fl.lca_inboard_rear.z_mm),
        lca_outboard=_hp("LCA_O", fl.lca_outboard.x_mm, -fl.lca_outboard.y_mm, fl.lca_outboard.z_mm),
        tie_rod_inboard=_hp("TR_I", fl.tie_rod_inboard.x_mm, -fl.tie_rod_inboard.y_mm, fl.tie_rod_inboard.z_mm),
        tie_rod_outboard=_hp("TR_O", fl.tie_rod_outboard.x_mm, -fl.tie_rod_outboard.y_mm, fl.tie_rod_outboard.z_mm),
        wheel_center=_hp("WC", fl.wheel_center.x_mm, -fl.wheel_center.y_mm, fl.wheel_center.z_mm),
        tire=_tire(),
        static_camber_deg=fl.static_camber_deg,
        static_toe_deg_per_side=fl.static_toe_deg_per_side,
    )


def _load_csv() -> tuple[list[dict[str, float]], str]:
    """Load the OptimumK sweep CSV.

    Returns (rows, toe_column_name) where toe_column_name is either
    'toe_per_side_deg' or 'toe_total_deg'.

    Raises ValueError if:
      - The ambiguous column name 'toe_deg' is present
      - Neither toe_per_side_deg nor toe_total_deg is present
      - Both toe columns are present
    """
    rows: list[dict[str, float]] = []
    with open(_CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        if "toe_deg" in headers:
            raise ValueError(
                "CSV column 'toe_deg' is ambiguous -- rename to either "
                "'toe_per_side_deg' (one wheel) or 'toe_total_deg' (sum of both wheels). "
                "OptimumK reports total toe by default."
            )

        has_per_side = "toe_per_side_deg" in headers
        has_total = "toe_total_deg" in headers

        if has_per_side and has_total:
            raise ValueError(
                "CSV has both 'toe_per_side_deg' and 'toe_total_deg' -- "
                "include exactly one."
            )
        if not has_per_side and not has_total:
            raise ValueError(
                "CSV must have either 'toe_per_side_deg' or 'toe_total_deg'. "
                "Do not use 'toe_deg' (ambiguous)."
            )

        toe_col = "toe_per_side_deg" if has_per_side else "toe_total_deg"

        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if v.strip()})

    return rows, toe_col


def _toe_per_side(row: dict[str, float], toe_col: str) -> float:
    """Extract toe per side from a CSV row, converting from total if needed."""
    if toe_col == "toe_per_side_deg":
        return row["toe_per_side_deg"]
    return row["toe_total_deg"] / 2.0


def _static_row(rows: list[dict[str, float]]) -> dict[str, float]:
    """Find the row closest to zero wheel travel."""
    return min(rows, key=lambda r: abs(r["wheel_travel_mm"]))


def _scrub_radius_mm(solver_result: object) -> float:
    """Compute scrub radius from a SolverResult.

    Scrub radius = CP_y - kingpin_ground_y (positive = CP outboard of kingpin).
    """
    r = solver_result
    ubj = np.array([r.ubj.x_mm, r.ubj.y_mm, r.ubj.z_mm])  # type: ignore[attr-defined]
    lbj = np.array([r.lbj.x_mm, r.lbj.y_mm, r.lbj.z_mm])  # type: ignore[attr-defined]
    kp_dir = ubj - lbj
    if abs(kp_dir[2]) < 1e-10:
        return 0.0
    t_ground = -lbj[2] / kp_dir[2]
    kp_ground = lbj + t_ground * kp_dir
    return float(r.contact_patch.y_mm - kp_ground[1])  # type: ignore[attr-defined]


@pytest.mark.skipif(not _CSV_EXISTS, reason="OptimumK CSV not yet exported")
class TestOptimumKStatic:
    """Compare static values (zero travel) before running sweep.

    If the static point disagrees, the curve comparison tells us nothing.
    """

    def test_csv_has_required_columns(self) -> None:
        rows, toe_col = _load_csv()
        assert len(rows) > 0, "CSV is empty"
        required = {"wheel_travel_mm", "camber_deg", "caster_deg", "kpi_deg", "rc_height_mm"}
        required.add(toe_col)
        assert required.issubset(rows[0].keys()), (
            f"Missing columns: {required - rows[0].keys()}"
        )

    def test_static_camber(self) -> None:
        rows, _ = _load_csv()
        row = _static_row(rows)
        r = DWSolver(_correlation_fl()).solve()
        assert r.converged
        err = abs(r.camber_deg - row["camber_deg"])
        assert err < _ANGLE_TOL_DEG, (
            f"Static camber: vdcore={r.camber_deg:.4f}, OptK={row['camber_deg']:.4f}, err={err:.4f}"
        )

    def test_static_toe(self) -> None:
        rows, toe_col = _load_csv()
        row = _static_row(rows)
        r = DWSolver(_correlation_fl()).solve()
        assert r.converged
        optk_toe = _toe_per_side(row, toe_col)
        err = abs(r.toe_deg_per_side - optk_toe)
        assert err < _ANGLE_TOL_DEG, (
            f"Static toe (per side): vdcore={r.toe_deg_per_side:.4f}, "
            f"OptK={optk_toe:.4f} (from {toe_col}), err={err:.4f}"
        )

    def test_static_caster(self) -> None:
        rows, _ = _load_csv()
        row = _static_row(rows)
        r = DWSolver(_correlation_fl()).solve()
        assert r.converged
        err = abs(r.caster_deg - row["caster_deg"])
        assert err < _ANGLE_TOL_DEG, (
            f"Static caster: vdcore={r.caster_deg:.4f}, OptK={row['caster_deg']:.4f}, err={err:.4f}"
        )

    def test_static_kpi(self) -> None:
        rows, _ = _load_csv()
        row = _static_row(rows)
        r = DWSolver(_correlation_fl()).solve()
        assert r.converged
        err = abs(r.kpi_deg - row["kpi_deg"])
        assert err < _ANGLE_TOL_DEG, (
            f"Static KPI: vdcore={r.kpi_deg:.4f}, OptK={row['kpi_deg']:.4f}, err={err:.4f}"
        )

    def test_static_rc_height(self) -> None:
        rows, _ = _load_csv()
        row = _static_row(rows)
        fl = _correlation_fl()
        fr = _correlation_fr()
        axle = Axle(left=fl, right=fr)
        rl = DWSolver(fl).solve()
        rr = DWSolver(fr).solve()
        assert rl.converged and rr.converged
        rc = roll_centre_height(axle, rl, rr)
        err = abs(rc.rc_height_mm - row["rc_height_mm"])
        assert err < _RC_TOL_MM, (
            f"Static RC: vdcore={rc.rc_height_mm:.2f}, OptK={row['rc_height_mm']:.2f}, err={err:.2f}"
        )

    def test_static_scrub_radius(self) -> None:
        """Scrub radius -- if available in the CSV."""
        rows, _ = _load_csv()
        row = _static_row(rows)
        if "scrub_radius_mm" not in row:
            pytest.skip("CSV does not include scrub_radius_mm")
        r = DWSolver(_correlation_fl()).solve()
        assert r.converged
        scrub = _scrub_radius_mm(r)
        err = abs(scrub - row["scrub_radius_mm"])
        assert err < _RC_TOL_MM, (
            f"Static scrub: vdcore={scrub:.2f}, OptK={row['scrub_radius_mm']:.2f}, err={err:.2f}"
        )


@pytest.mark.skipif(not _CSV_EXISTS, reason="OptimumK CSV not yet exported")
class TestOptimumKSweep:
    """Compare vdcore wheel-travel sweep against OptimumK export."""

    def test_camber_matches(self) -> None:
        rows, _ = _load_csv()
        solver = DWSolver(_correlation_fl())
        for row in rows:
            wt = row["wheel_travel_mm"]
            r = solver.solve(wheel_travel_mm=wt)
            assert r.converged, f"Solver failed at wt={wt}"
            err = abs(r.camber_deg - row["camber_deg"])
            assert err < _ANGLE_TOL_DEG, (
                f"Camber at wt={wt:.1f}mm: "
                f"vdcore={r.camber_deg:.4f}, OptK={row['camber_deg']:.4f}, err={err:.4f}"
            )

    def test_toe_matches(self) -> None:
        rows, toe_col = _load_csv()
        solver = DWSolver(_correlation_fl())
        for row in rows:
            wt = row["wheel_travel_mm"]
            r = solver.solve(wheel_travel_mm=wt)
            assert r.converged
            optk_toe = _toe_per_side(row, toe_col)
            err = abs(r.toe_deg_per_side - optk_toe)
            assert err < _ANGLE_TOL_DEG, (
                f"Toe at wt={wt:.1f}mm: "
                f"vdcore={r.toe_deg_per_side:.4f}, OptK={optk_toe:.4f} (from {toe_col})"
            )

    def test_caster_matches(self) -> None:
        rows, _ = _load_csv()
        solver = DWSolver(_correlation_fl())
        for row in rows:
            wt = row["wheel_travel_mm"]
            r = solver.solve(wheel_travel_mm=wt)
            assert r.converged
            err = abs(r.caster_deg - row["caster_deg"])
            assert err < _ANGLE_TOL_DEG, (
                f"Caster at wt={wt:.1f}mm: "
                f"vdcore={r.caster_deg:.4f}, OptK={row['caster_deg']:.4f}"
            )

    def test_kpi_matches(self) -> None:
        rows, _ = _load_csv()
        solver = DWSolver(_correlation_fl())
        for row in rows:
            wt = row["wheel_travel_mm"]
            r = solver.solve(wheel_travel_mm=wt)
            assert r.converged
            err = abs(r.kpi_deg - row["kpi_deg"])
            assert err < _ANGLE_TOL_DEG, (
                f"KPI at wt={wt:.1f}mm: "
                f"vdcore={r.kpi_deg:.4f}, OptK={row['kpi_deg']:.4f}"
            )

    def test_rc_height_matches(self) -> None:
        rows, _ = _load_csv()
        fl = _correlation_fl()
        fr = _correlation_fr()
        axle = Axle(left=fl, right=fr)
        solver_l = DWSolver(fl)
        solver_r = DWSolver(fr)
        for row in rows:
            wt = row["wheel_travel_mm"]
            rl = solver_l.solve(wheel_travel_mm=wt)
            rr = solver_r.solve(wheel_travel_mm=wt)
            assert rl.converged and rr.converged, f"Solver failed at wt={wt}"
            rc = roll_centre_height(axle, left_result=rl, right_result=rr)
            err = abs(rc.rc_height_mm - row["rc_height_mm"])
            assert err < _RC_TOL_MM, (
                f"RC at wt={wt:.1f}mm: "
                f"vdcore={rc.rc_height_mm:.2f}, OptK={row['rc_height_mm']:.2f}, err={err:.2f}"
            )

    def test_max_errors_summary(self) -> None:
        """Report max error per quantity across the sweep."""
        rows, toe_col = _load_csv()
        fl = _correlation_fl()
        fr = _correlation_fr()
        axle = Axle(left=fl, right=fr)
        solver_l = DWSolver(fl)
        solver_r = DWSolver(fr)
        errs: dict[str, float] = {
            "camber": 0.0, "toe": 0.0, "caster": 0.0, "kpi": 0.0, "rc": 0.0,
        }
        for row in rows:
            wt = row["wheel_travel_mm"]
            rl = solver_l.solve(wheel_travel_mm=wt)
            rr = solver_r.solve(wheel_travel_mm=wt)
            assert rl.converged and rr.converged
            errs["camber"] = max(errs["camber"], abs(rl.camber_deg - row["camber_deg"]))
            errs["toe"] = max(errs["toe"], abs(rl.toe_deg_per_side - _toe_per_side(row, toe_col)))
            errs["caster"] = max(errs["caster"], abs(rl.caster_deg - row["caster_deg"]))
            errs["kpi"] = max(errs["kpi"], abs(rl.kpi_deg - row["kpi_deg"]))
            rc = roll_centre_height(axle, left_result=rl, right_result=rr)
            errs["rc"] = max(errs["rc"], abs(rc.rc_height_mm - row["rc_height_mm"]))
        for k, v in errs.items():
            tol = _RC_TOL_MM if k == "rc" else _ANGLE_TOL_DEG
            unit = "mm" if k == "rc" else "deg"
            assert v < tol, f"Max {k} error: {v:.4f} {unit} (tol={tol})"
