#!/usr/bin/env python3
"""Export hardpoints from a vdcore fixture in Optimum Kinematics format.

Usage:
    python scripts/export_hardpoints_for_optimumk.py

Output is printed to stdout, ready to copy-paste into OptimumK's
point entry table. Each row is: PointName, X, Y, Z (in mm).

Frame transform (ISO 8855 → Optimum Kinematics):
    X_optk =  X_iso   (forward)
    Y_optk = -Y_iso   (ISO: left+, OptK: right+)
    Z_optk =  Z_iso   (up)

Angle sign conventions — OptimumK vs this project (ISO 8855):
    Camber:  same sign (negative = top inboard)
    Toe:     same sign (positive = toe-in) — verify in OptK settings
    Caster:  same sign (positive = rearward tilt)
    KPI:     same sign (positive = inboard tilt)

When exporting OptimumK sweep results to CSV for test_optimumk_correlation.py:
    1. Set up a FULL AXLE in OptimumK (left + right corners, mirrored)
    2. Run a wheel travel sweep (bump/droop, -30 to +30 mm, 1 mm steps)
    3. Export the results table
    4. Save as tests/benchmarks/data/optimumk_sweep.csv with headers:
         wheel_travel_mm,camber_deg,caster_deg,kpi_deg,rc_height_mm
       Plus EXACTLY ONE of: toe_per_side_deg or toe_total_deg
       (do NOT name it toe_deg -- the loader will reject it)
    5. rc_height_mm is REQUIRED — it is the primary quantity needing validation
    6. Negate OptimumK bump sign if it uses upward-positive for chassis
       (OptimumK convention may differ — check the sweep axis definition)
    7. Angle signs should match directly (no conversion needed)

OptimumK point names for a double-wishbone (SLA) front corner:
    UCA_CHASSIS_FRONT  = UCA inboard front
    UCA_CHASSIS_REAR   = UCA inboard rear
    UCA_UPRIGHT         = UCA outboard (upper ball joint)
    LCA_CHASSIS_FRONT  = LCA inboard front
    LCA_CHASSIS_REAR   = LCA inboard rear
    LCA_UPRIGHT         = LCA outboard (lower ball joint)
    TIEROD_CHASSIS      = Tie rod inboard (rack end)
    TIEROD_UPRIGHT      = Tie rod outboard
    WHEEL_CENTER        = Wheel centre
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vdcore.analysis.roll_centre import front_view_instant_centre, roll_centre_height
from vdcore.geometry.solver import DWSolver
from vdcore.models.hardpoint import Axle, Corner, Hardpoint, TirePackage

_OPTK_NAMES: dict[str, str] = {
    "uca_inboard_front": "UCA_CHASSIS_FRONT",
    "uca_inboard_rear": "UCA_CHASSIS_REAR",
    "uca_outboard": "UCA_UPRIGHT",
    "lca_inboard_front": "LCA_CHASSIS_FRONT",
    "lca_inboard_rear": "LCA_CHASSIS_REAR",
    "lca_outboard": "LCA_UPRIGHT",
    "tie_rod_inboard": "TIEROD_CHASSIS",
    "tie_rod_outboard": "TIEROD_UPRIGHT",
    "wheel_center": "WHEEL_CENTER",
}


def _hp(name: str, x: float, y: float, z: float) -> Hardpoint:
    return Hardpoint(name=name, x_mm=x, y_mm=y, z_mm=z, source="cad", tol_mm=0.5)


def _tire() -> TirePackage:
    return TirePackage(loaded_radius_mm=254.0, source="cad", tol_mm=1.0)


def _fsae_fl() -> Corner:
    """FSAE-representative front-left corner (same as test_fsae_representative.py)."""
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


def _fsae_fr() -> Corner:
    """FSAE-representative front-right corner (mirrored)."""
    fl = _fsae_fl()
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


def export_for_optimumk(corner: Corner) -> str:
    """Format hardpoints for Optimum Kinematics entry.

    Returns a string with one point per line:
        PointName, X_optk, Y_optk, Z_optk
    """
    lines: list[str] = []
    lines.append(f"# Optimum Kinematics hardpoints -- corner {corner.corner_id}")
    lines.append(f"# Frame: OptimumK (X+ forward, Y+ RIGHT, Z+ up)")
    lines.append(f"# Converted from ISO 8855 (Y negated)")
    lines.append(f"# Static camber: {corner.static_camber_deg:.1f} deg")
    lines.append(f"# Static toe (per side): {corner.static_toe_deg_per_side:.1f} deg")
    lines.append(f"# Tire loaded radius: {corner.tire.loaded_radius_mm:.1f} mm")
    lines.append(f"#")
    lines.append(f"# {'Point':<25s}  {'X':>10s}  {'Y':>10s}  {'Z':>10s}")

    for attr_name, optk_name in _OPTK_NAMES.items():
        hp: Hardpoint = getattr(corner, attr_name)
        x_optk = hp.x_mm
        y_optk = -hp.y_mm
        z_optk = hp.z_mm
        lines.append(f"  {optk_name:<25s}  {x_optk:10.2f}  {y_optk:10.2f}  {z_optk:10.2f}")

    return "\n".join(lines)


def print_static_values(fl: Corner, fr: Corner) -> None:
    """Print vdcore's static KPIs for comparison against OptimumK readout."""
    solver = DWSolver(fl)
    r = solver.solve()
    assert r.converged

    ubj = np.array([r.ubj.x_mm, r.ubj.y_mm, r.ubj.z_mm])
    lbj = np.array([r.lbj.x_mm, r.lbj.y_mm, r.lbj.z_mm])
    kp_dir = ubj - lbj
    t_ground = -lbj[2] / kp_dir[2]
    kp_ground = lbj + t_ground * kp_dir
    scrub = float(r.contact_patch.y_mm - kp_ground[1])
    mech_trail = float(kp_ground[0] - r.contact_patch.x_mm)

    fvic = front_view_instant_centre(fl, ubj=r.ubj, lbj=r.lbj, contact_patch=r.contact_patch)
    axle = Axle(left=fl, right=fr)
    rr = DWSolver(fr).solve()
    assert rr.converged
    rc = roll_centre_height(axle, r, rr)

    print("# -- vdcore static values (compare against OptimumK readout) --")
    print(f"#   Camber:          {r.camber_deg:+.4f} deg")
    print(f"#   Toe (per side):  {r.toe_deg_per_side:+.4f} deg")
    print(f"#   Caster:          {r.caster_deg:+.4f} deg")
    print(f"#   KPI:             {r.kpi_deg:+.4f} deg")
    print(f"#   Scrub radius:    {scrub:+.2f} mm")
    print(f"#   Mechanical trail:{mech_trail:+.2f} mm")
    print(f"#   FVSA:            {fvic.fvsa_mm:.1f} mm")
    print(f"#   RC height:       {rc.rc_height_mm:.2f} mm")
    print("#")
    print("# If KPI matches but caster is sign-flipped, X is also inverted")
    print("# in your OptimumK setup -- re-check the frame transform.")
    print("# If scrub sign is flipped, Y is inverted (most likely cause).")


def main() -> None:
    fl = _fsae_fl()
    fr = _fsae_fr()

    print(export_for_optimumk(fl))
    print()
    print_static_values(fl, fr)
    print()
    print("# -- Setup instructions --")
    print("#")
    print("#   1. Enter the LEFT corner hardpoints above into OptimumK")
    print("#   2. MIRROR to create the RIGHT corner (OptimumK should negate Y)")
    print("#      Verify right-side WC is at (0, 610, 254) in OptimumK coords")
    print("#   3. Set tire loaded radius = 254.0 mm")
    print("#   4. Set static camber = -2.0 deg, static toe = 0.1 deg")
    print("#   5. CHECK static readout against the values printed above")
    print("#      -- if any disagree, STOP and diagnose before running a sweep")
    print("#   6. Run a wheel travel sweep: -30 to +30 mm, 1 mm steps")
    print("#   7. Include RC height in the sweep output (requires full axle)")
    print("#   8. Export to CSV and save as:")
    print("#        tests/benchmarks/data/optimumk_sweep.csv")
    print("#   9. Required columns:")
    print("#        wheel_travel_mm,camber_deg,caster_deg,kpi_deg,rc_height_mm")
    print("#      Plus EXACTLY ONE of:")
    print("#        toe_per_side_deg  (one wheel's toe angle)")
    print("#        toe_total_deg     (sum of left + right)")
    print("#      Do NOT use 'toe_deg' -- the loader will reject it")


if __name__ == "__main__":
    main()
