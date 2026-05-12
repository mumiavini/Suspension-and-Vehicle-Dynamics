"""
analysis/optimizer.py
=====================
Motor de SÍNTESE da geometria de suspensão.

OBJETIVO:
    Dado um veículo com hardpoints VARIÁVEIS dentro de bounding boxes
    (keep-out zones do chassi), encontrar a configuração que MINIMIZA um
    custo composto de metas (targets):

        cost = w_cg * (camber_gain - target)²
             + w_bs * Σ(Δtoe)²
             + w_rch * (rc_height - target)²
             + w_rcm * max(0, ΔY_rc - max_allowed)²

ALGORITMO: scipy.optimize.differential_evolution

    É um algoritmo evolutivo GLOBAL, robusto a espaços não-convexos com
    muitos mínimos locais (típico em problemas de hardpoint placement).
    Mais lento que gradiente, mas não precisa de derivadas.

FLUXO DE USO:
    1. Crie um SuspensionCorner e TieRod como "seed" (geometria inicial)
    2. Defina DesignTargets com seus alvos e pesos
    3. (Opcional) Defina HardpointBounds para restringir a busca
    4. Instancie SuspensionOptimizer e chame .run()
    5. Use OptimizationResult.optimal_corner e .optimal_tie_rod
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution, OptimizeResult

from geometry.primitives import Point3D
from geometry.model_3d import ControlArm, SuspensionCorner
from geometry.solver_3d import TieRod, KinematicSolver3D
from analysis.sweeps import (
    SweepRunner,
    camber_gain_per_mm,
    bump_steer_per_mm,
    rc_migration_range,
)


# =============================================================================
# HardpointBounds — Bounding box espacial para um hardpoint
# =============================================================================

@dataclass
class HardpointBounds:
    """
    Limites espaciais (caixa) para um hardpoint variável.

    Define a região do espaço onde o otimizador pode mover este hardpoint.
    Use para representar KEEP-OUT ZONES (regiões interditadas pelo chassi,
    pacote do motor, requisitos de packaging, etc).
    """
    x_min: float; x_max: float
    y_min: float; y_max: float
    z_min: float; z_max: float

    def as_bounds(self) -> list[tuple[float, float]]:
        """Formato esperado por scipy.optimize.differential_evolution."""
        return [
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (self.z_min, self.z_max),
        ]

    def contains(self, point: Point3D) -> bool:
        """Testa se um ponto está dentro da caixa."""
        return (self.x_min <= point.x <= self.x_max and
                self.y_min <= point.y <= self.y_max and
                self.z_min <= point.z <= self.z_max)


# =============================================================================
# DesignTargets — Metas e pesos da função objetivo
# =============================================================================

@dataclass
class DesignTargets:
    """
    Metas de projeto e pesos da função objetivo do otimizador.

    Pesos (w_*) controlam a IMPORTÂNCIA RELATIVA de cada termo no custo
    total. Definir um peso como 0.0 desativa o termo correspondente.
    """
    # ─── Alvos (valores desejados) ────────────────────────────────────────────
    camber_gain_target_deg_per_mm: float = -0.015   # ≈ −0.4°/inch
    bump_steer_max_abs_deg_per_mm: float =  0.010   # |bump_steer| < 0.01 °/mm
    rc_height_target_mm:           float =  50.0    # altura desejada do RC
    rc_y_migration_max_mm:         float =  30.0    # migração lateral máxima

    # ─── Faixa do sweep usado para avaliação ──────────────────────────────────
    heave_min_mm:  float = -25.0
    heave_max_mm:  float =  25.0
    heave_step_mm: float =   2.5

    # ─── Pesos da função objetivo ─────────────────────────────────────────────
    w_camber_gain:  float = 1.0
    w_bump_steer:   float = 10.0
    w_rc_height:    float = 0.01
    w_rc_migration: float = 0.05

    # ─── Penalidade para configurações que quebram o solver ──────────────────
    penalty_non_converged: float = 1e6


# =============================================================================
# SuspensionOptimizer — Loop principal de otimização
# =============================================================================

@dataclass
class SuspensionOptimizer:
    """
    Otimizador de geometria para UMA ponta de suspensão.

    VARIÁVEIS DE DESIGN (12 DOF):
        UCA outboard   (x, y, z)
        LCA outboard   (x, y, z)
        Tie-rod inboard  (x, y, z)
        Tie-rod outboard (x, y, z)

    Os pontos INBOARD do UCA/LCA são mantidos FIXOS (consideram-se determinados
    pelo packaging do chassi). Para libertar mais variáveis, estenda esta classe.
    """
    # ─── Geometria inicial (seed) ─────────────────────────────────────────────
    seed_corner:  SuspensionCorner
    seed_tie_rod: TieRod
    targets:      DesignTargets

    # ─── Limites para cada hardpoint variável ─────────────────────────────────
    bounds_uca_outboard: HardpointBounds = field(default=None)  # type: ignore
    bounds_lca_outboard: HardpointBounds = field(default=None)  # type: ignore
    bounds_tie_rod_in:   HardpointBounds = field(default=None)  # type: ignore
    bounds_tie_rod_out:  HardpointBounds = field(default=None)  # type: ignore

    # ─── Configurações do differential_evolution ──────────────────────────────
    population_size: int  = 15
    max_iterations:  int  = 60
    seed:            int  = 42
    workers:         int  = 1
    polish:          bool = True
    verbose:         bool = False

    # -------------------------------------------------------------------------
    # Inicialização: cria bounds default se não fornecidos
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Cria bounds default (caixa ±50mm) para hardpoints sem bounds."""
        def default_box(p: Point3D, margin: float = 50.0) -> HardpointBounds:
            return HardpointBounds(
                p.x - margin, p.x + margin,
                p.y - margin, p.y + margin,
                p.z - margin, p.z + margin,
            )

        if self.bounds_uca_outboard is None:
            self.bounds_uca_outboard = default_box(self.seed_corner.upper_arm.outboard)
        if self.bounds_lca_outboard is None:
            self.bounds_lca_outboard = default_box(self.seed_corner.lower_arm.outboard)
        if self.bounds_tie_rod_in is None:
            self.bounds_tie_rod_in = default_box(self.seed_tie_rod.inboard, 30.0)
        if self.bounds_tie_rod_out is None:
            self.bounds_tie_rod_out = default_box(self.seed_tie_rod.outboard, 30.0)

    # -------------------------------------------------------------------------
    # Mapeamento entre vetor de design e objetos de geometria
    # -------------------------------------------------------------------------

    def _design_bounds(self) -> list[tuple[float, float]]:
        """Concatena todos os bounds em uma única lista (12 tuplas)."""
        return (
            self.bounds_uca_outboard.as_bounds()
            + self.bounds_lca_outboard.as_bounds()
            + self.bounds_tie_rod_in.as_bounds()
            + self.bounds_tie_rod_out.as_bounds()
        )

    def _vector_to_geometry(
        self,
        x: NDArray[np.float64],
    ) -> tuple[SuspensionCorner, TieRod]:
        """
        Converte vetor de design (12 floats) em (SuspensionCorner, TieRod).

        Layout do vetor:
            [0:3]   UCA outboard
            [3:6]   LCA outboard
            [6:9]   Tie-rod inboard
            [9:12]  Tie-rod outboard
        """
        uca_out = Point3D(float(x[0]),  float(x[1]),  float(x[2]))
        lca_out = Point3D(float(x[3]),  float(x[4]),  float(x[5]))
        tr_in   = Point3D(float(x[6]),  float(x[7]),  float(x[8]))
        tr_out  = Point3D(float(x[9]),  float(x[10]), float(x[11]))

        # Mantém os inboards do UCA/LCA fixos (do seed)
        new_uca = ControlArm(
            inboard_front=self.seed_corner.upper_arm.inboard_front,
            inboard_rear =self.seed_corner.upper_arm.inboard_rear,
            outboard=uca_out,
            name=self.seed_corner.upper_arm.name,
        )
        new_lca = ControlArm(
            inboard_front=self.seed_corner.lower_arm.inboard_front,
            inboard_rear =self.seed_corner.lower_arm.inboard_rear,
            outboard=lca_out,
            name=self.seed_corner.lower_arm.name,
        )
        new_corner = SuspensionCorner(
            upper_arm=new_uca,
            lower_arm=new_lca,
            wheel_center =self.seed_corner.wheel_center,
            contact_patch=self.seed_corner.contact_patch,
            corner_id=self.seed_corner.corner_id,
        )
        new_tr = TieRod(inboard=tr_in, outboard=tr_out, name=self.seed_tie_rod.name)
        return new_corner, new_tr

    def _initial_guess_vector(self) -> NDArray[np.float64]:
        """Vetor de design correspondente ao seed (12 floats)."""
        return np.array([
            *self.seed_corner.upper_arm.outboard.to_array(),
            *self.seed_corner.lower_arm.outboard.to_array(),
            *self.seed_tie_rod.inboard.to_array(),
            *self.seed_tie_rod.outboard.to_array(),
        ])

    # =========================================================================
    # FUNÇÃO OBJETIVO
    # =========================================================================

    def objective(self, x: NDArray[np.float64]) -> float:
        """
        Avalia o custo de uma configuração de hardpoints.

        ETAPAS:
            1. Constrói SuspensionCorner + TieRod a partir do vetor x
            2. Cria solver 3D e roda um heave sweep curto
            3. Calcula métricas (camber_gain, bump_steer, rc_height, rc_migration)
            4. Soma os termos ponderados em um único custo escalar

        Se algo falhar (geometria inválida, solver não converge), retorna
        a penalidade gigante (targets.penalty_non_converged).
        """
        try:
            corner, tie_rod = self._vector_to_geometry(x)
            solver = KinematicSolver3D(corner, tie_rod)
            runner = SweepRunner(solver=solver)

            sweep = runner.heave_sweep(
                self.targets.heave_min_mm,
                self.targets.heave_max_mm,
                self.targets.heave_step_mm,
            )

            # Se algum ponto não convergiu, descarta esta configuração
            if not bool(sweep["converged"].all()):
                return float(self.targets.penalty_non_converged)

        except Exception:
            return float(self.targets.penalty_non_converged)

        # ─── Termo 1: camber gain ─────────────────────────────────────────────
        cg = camber_gain_per_mm(sweep)
        cost_cg = (cg - self.targets.camber_gain_target_deg_per_mm) ** 2

        # ─── Termo 2: bump steer (integral do quadrado do Δtoe) ──────────────
        # Como o solver retorna toe relativo ao estático, sweep["toe_deg"][zero]≈0
        cost_bs = float(np.mean(sweep["toe_deg"] ** 2))

        # ─── Termo 3: altura do Roll Center ──────────────────────────────────
        rc_z_mean = float(np.mean(sweep["rc_z_mm"]))
        cost_rch = (rc_z_mean - self.targets.rc_height_target_mm) ** 2

        # ─── Termo 4: migração do RC (penaliza só se passar do limite) ───────
        dy, _ = rc_migration_range(sweep)
        excess_y = max(0.0, dy - self.targets.rc_y_migration_max_mm)
        cost_rcm = excess_y ** 2

        return float(
            self.targets.w_camber_gain  * cost_cg
          + self.targets.w_bump_steer   * cost_bs
          + self.targets.w_rc_height    * cost_rch
          + self.targets.w_rc_migration * cost_rcm
        )

    # =========================================================================
    # LOOP PRINCIPAL DE OTIMIZAÇÃO
    # =========================================================================

    def run(self) -> "OptimizationResult":
        """
        Executa o differential evolution.

        ESTRATÉGIA DE INICIALIZAÇÃO DA POPULAÇÃO:
            - 1º indivíduo = seed (geometria inicial)
            - 50% dos demais = perturbação gaussiana em torno do seed
            - 50% restantes = sorteio uniforme nos bounds

        Isso acelera muito a convergência mantendo diversidade global.
        """
        bounds = self._design_bounds()
        seed_vec = self._initial_guess_vector()

        # ─── Constrói população inicial mista ─────────────────────────────────
        rng = np.random.default_rng(self.seed)
        n_dims = len(bounds)
        pop_size = self.population_size * n_dims
        init_pop = np.empty((pop_size, n_dims))

        for i in range(pop_size):
            for j in range(n_dims):
                lo, hi = bounds[j]
                if i == 0:
                    # 1º indivíduo = seed
                    init_pop[i, j] = seed_vec[j]
                elif rng.random() < 0.5:
                    # Perturbação gaussiana em torno do seed
                    sigma = (hi - lo) * 0.15
                    init_pop[i, j] = np.clip(
                        seed_vec[j] + rng.normal(0, sigma), lo, hi
                    )
                else:
                    # Amostragem uniforme nos bounds
                    init_pop[i, j] = rng.uniform(lo, hi)

        # ─── Roda differential_evolution ──────────────────────────────────────
        result = differential_evolution(
            func=self.objective,
            bounds=bounds,
            init=init_pop,
            maxiter=self.max_iterations,
            popsize=self.population_size,
            mutation=(0.5, 1.0),
            recombination=0.7,
            tol=1e-6,
            seed=self.seed,
            workers=self.workers,
            polish=self.polish,
            disp=self.verbose,
            updating="deferred" if self.workers != 1 else "immediate",
        )

        best_corner, best_tie_rod = self._vector_to_geometry(result.x)
        return OptimizationResult(
            optimal_corner=best_corner,
            optimal_tie_rod=best_tie_rod,
            cost=float(result.fun),
            x=result.x,
            scipy_result=result,
        )


# =============================================================================
# OptimizationResult — Encapsula o resultado
# =============================================================================

@dataclass
class OptimizationResult:
    """Resultado da otimização: geometria ótima + diagnóstico do solver."""
    optimal_corner:  SuspensionCorner
    optimal_tie_rod: TieRod
    cost:            float
    x:               NDArray[np.float64]
    scipy_result:    OptimizeResult

    def summary(self) -> str:
        """Resumo formatado do resultado."""
        return "\n".join([
            "═══ Optimization Result ═══",
            f"  Custo final         : {self.cost:.6e}",
            f"  Iterações           : {self.scipy_result.nit}",
            f"  Avaliações de obj   : {self.scipy_result.nfev}",
            f"  Sucesso             : {self.scipy_result.success}",
            f"  Mensagem            : {self.scipy_result.message}",
            "",
            "  Hardpoints otimizados:",
            f"    UCA outboard      : {self.optimal_corner.upper_arm.outboard}",
            f"    LCA outboard      : {self.optimal_corner.lower_arm.outboard}",
            f"    Tie-rod inboard   : {self.optimal_tie_rod.inboard}",
            f"    Tie-rod outboard  : {self.optimal_tie_rod.outboard}",
        ])