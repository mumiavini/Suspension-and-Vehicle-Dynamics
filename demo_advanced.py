"""
demo_advanced.py
================
Demonstração consolidada dos Degraus 3-5 (Solver 3D, Sweeps, Otimizador).

Para o Degrau 6 (Streamlit), execute em separado:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from geometry.primitives import Point3D
from geometry.model_3d import ControlArm, SuspensionCorner
from geometry.solver_3d import TieRod, KinematicSolver3D
from analysis.sweeps import (
    SweepRunner,
    camber_gain_per_mm,
    bump_steer_per_mm,
    rc_migration_range,
)
from analysis.optimizer import (
    SuspensionOptimizer,
    DesignTargets,
    HardpointBounds,
)


def make_demo_corner() -> tuple[SuspensionCorner, TieRod]:
    """Geometria realista FSAE da ponta dianteira esquerda."""
    uca = ControlArm(
        inboard_front=Point3D( 60.0, 150.0, 295.0),
        inboard_rear =Point3D(-70.0, 150.0, 295.0),
        outboard     =Point3D( -5.0, 590.0, 280.0),
        name="UCA_FL",
    )
    lca = ControlArm(
        inboard_front=Point3D( 90.0, 130.0, 162.0),
        inboard_rear =Point3D(-70.0, 130.0, 162.0),
        outboard     =Point3D( 15.0, 600.0, 152.0),
        name="LCA_FL",
    )
    tr = TieRod(
        inboard =Point3D(-50.0, 180.0, 200.0),
        outboard=Point3D(-60.0, 580.0, 195.0),
        name="TR_FL",
    )
    corner = SuspensionCorner(
        upper_arm=uca,
        lower_arm=lca,
        wheel_center =Point3D(5.0, 610.0, 220.0),
        contact_patch=Point3D(5.0, 610.0,   0.0),
        corner_id="FL",
    )
    return corner, tr


# ═══════════════════════════════════════════════════════════════════════════
# DEGRAU 3 — Solver 3D
# ═══════════════════════════════════════════════════════════════════════════

def demo_solver_3d() -> None:
    print("\n" + "═" * 65)
    print("  DEGRAU 3 — SOLVER CINEMÁTICO 3D (Interseção de 3 esferas)")
    print("═" * 65)

    corner, tie_rod = make_demo_corner()
    solver = KinematicSolver3D(corner, tie_rod)

    cases = [
        ("Estático        ",  0.0,  0.0,  0.0),
        ("Bump +10 mm     ", 10.0,  0.0,  0.0),
        ("Rebound -10 mm  ",-10.0,  0.0,  0.0),
        ("Roll +2°        ",  0.0,  2.0,  0.0),
        ("Roll -2°        ",  0.0, -2.0,  0.0),
        ("Rack +20 mm     ",  0.0,  0.0, 20.0),
        ("Bump+Roll+Rack  ",  8.0,  1.5, 10.0),
    ]

    print(f"\n  {'Caso':<18} {'Camb (°)':>10} {'Caster':>10} {'KPI':>8} "
          f"{'Δtoe (°)':>10} {'residual':>12}")
    print("  " + "-" * 75)

    solver.reset_seed()
    s_ref = solver.solve(0.0, 0.0, 0.0)
    toe_ref = s_ref.toe_deg

    for label, h, r, s in cases:
        solver.reset_seed()
        state = solver.solve(h, r, s)
        dtoe = state.toe_deg - toe_ref
        print(f"  {label} {state.camber_deg:>+10.4f} {state.caster_deg:>+10.4f} "
              f"{state.kpi_deg:>+8.4f} {dtoe:>+10.4f} {state.residual_norm:>12.2e}")


# ═══════════════════════════════════════════════════════════════════════════
# DEGRAU 4 — Sweeps
# ═══════════════════════════════════════════════════════════════════════════

def demo_sweeps() -> None:
    print("\n" + "═" * 65)
    print("  DEGRAU 4 — SWEEPS PARAMÉTRICOS")
    print("═" * 65)

    corner, tie_rod = make_demo_corner()
    solver = KinematicSolver3D(corner, tie_rod)
    runner = SweepRunner(solver=solver)

    print("\n  Heave sweep (-25 → +25 mm, step 1 mm):")
    hs = runner.heave_sweep(-25.0, 25.0, 1.0)
    print(f"    pontos             : {len(hs)}")
    print(f"    convergência       : {hs['converged'].all()}  (max res {hs['residual'].max():.2e})")
    print(f"    camber range       : [{hs['camber_deg'].min():+.4f}, {hs['camber_deg'].max():+.4f}] °")
    print(f"    camber gain        : {camber_gain_per_mm(hs):+.5f} °/mm")
    print(f"    bump steer         : {bump_steer_per_mm(hs):+.5f} °/mm")
    dy, dz = rc_migration_range(hs)
    print(f"    RC migration       : ΔY = {dy:.2f} mm, ΔZ = {dz:.2f} mm")

    print("\n  Roll sweep (-3 → +3°, step 0.5°):")
    rs = runner.roll_sweep(-3.0, 3.0, 0.5)
    print(f"    pontos             : {len(rs)}")
    print(f"    camber range       : [{rs['camber_deg'].min():+.4f}, {rs['camber_deg'].max():+.4f}] °")
    # camber/roll ratio
    if rs['roll_deg'].max() - rs['roll_deg'].min() > 0:
        cr_ratio = np.polyfit(rs['roll_deg'], rs['camber_deg'], 1)[0]
        print(f"    camber/roll ratio  : {cr_ratio:+.4f} °/°")

    print("\n  Steer sweep (-30 → +30 mm, step 2 mm):")
    ss = runner.steer_sweep(-30.0, 30.0, 2.0)
    print(f"    pontos             : {len(ss)}")
    print(f"    toe range          : [{ss['toe_deg'].min():+.3f}, {ss['toe_deg'].max():+.3f}] °")
    print(f"    caster range       : [{ss['caster_deg'].min():+.4f}, {ss['caster_deg'].max():+.4f}] °")


# ═══════════════════════════════════════════════════════════════════════════
# DEGRAU 5 — Otimizador
# ═══════════════════════════════════════════════════════════════════════════

def demo_optimizer() -> None:
    print("\n" + "═" * 65)
    print("  DEGRAU 5 — OTIMIZADOR DE GEOMETRIA")
    print("═" * 65)

    corner, tie_rod = make_demo_corner()

    targets = DesignTargets(
        camber_gain_target_deg_per_mm=-0.020,   # alvo: −0.5°/inch
        bump_steer_max_abs_deg_per_mm=0.005,
        rc_height_target_mm=45.0,
        rc_y_migration_max_mm=20.0,
        heave_step_mm=5.0,                       # passos maiores → mais rápido
    )

    # Bounds restritivos: ±30 mm em torno dos hardpoints atuais
    def around(p: Point3D, dx=30, dy=30, dz=30):
        return HardpointBounds(p.x-dx, p.x+dx, p.y-dy, p.y+dy, p.z-dz, p.z+dz)

    opt = SuspensionOptimizer(
        seed_corner=corner,
        seed_tie_rod=tie_rod,
        targets=targets,
        bounds_uca_outboard=around(corner.upper_arm.outboard),
        bounds_lca_outboard=around(corner.lower_arm.outboard),
        bounds_tie_rod_in=around(tie_rod.inboard, 15, 15, 15),
        bounds_tie_rod_out=around(tie_rod.outboard, 15, 15, 15),
        population_size=8,
        max_iterations=20,
        verbose=False,
    )

    seed_cost = opt.objective(opt._initial_guess_vector())
    print(f"\n  Custo seed         : {seed_cost:.6e}")
    print(f"  Rodando differential_evolution ({opt.population_size}×{opt.max_iterations} ger.)...")

    result = opt.run()
    print(f"\n  Custo final        : {result.cost:.6e}")
    print(f"  Iterações          : {result.scipy_result.nit}")
    print(f"  Avaliações de obj  : {result.scipy_result.nfev}")
    print(f"  Mensagem           : {result.scipy_result.message}")
    print(f"\n  Hardpoints otimizados:")
    print(f"    UCA outboard     : {result.optimal_corner.upper_arm.outboard}")
    print(f"    LCA outboard     : {result.optimal_corner.lower_arm.outboard}")
    print(f"    Tie-rod inboard  : {result.optimal_tie_rod.inboard}")
    print(f"    Tie-rod outboard : {result.optimal_tie_rod.outboard}")

    # Valida o resultado: roda sweep no design otimizado
    new_solver = KinematicSolver3D(result.optimal_corner, result.optimal_tie_rod)
    new_runner = SweepRunner(solver=new_solver)
    hs = new_runner.heave_sweep(-25.0, 25.0, 5.0)
    print(f"\n  Performance da geometria otimizada:")
    print(f"    camber gain      : {camber_gain_per_mm(hs):+.5f} °/mm  (alvo: {targets.camber_gain_target_deg_per_mm:+.4f})")
    print(f"    bump steer       : {bump_steer_per_mm(hs):+.5f} °/mm  (max: ±{targets.bump_steer_max_abs_deg_per_mm:.4f})")
    rc_z = float(np.mean(hs['rc_z_mm']))
    print(f"    RC height médio  : {rc_z:.2f} mm           (alvo: {targets.rc_height_target_mm:.2f})")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n╔" + "═" * 63 + "╗")
    print("║      FSAE SUSPENSION ENGINE — DEMO ADVANCED (3..5)            ║")
    print("╚" + "═" * 63 + "╝")

    demo_solver_3d()
    demo_sweeps()
    demo_optimizer()

    print("\n" + "═" * 65)
    print("  Para o Degrau 6 (Streamlit), execute em outro terminal:")
    print("    $ streamlit run app.py")
    print("═" * 65 + "\n")