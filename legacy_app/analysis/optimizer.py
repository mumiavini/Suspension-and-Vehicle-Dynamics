"""
analysis/optimizer.py
=====================
Suspension geometry SYNTHESIS engine.

PURPOSE:
    Given a vehicle with hardpoints that are VARIABLE inside bounding boxes
    (chassis keep-out zones), find the configuration that MINIMIZES a composite
    cost of targets:

        cost = w_cg * (camber_gain - target)²
             + w_bs * Σ(Δtoe)²
             + w_rch * (rc_height - target)²
             + w_rcm * max(0, ΔY_rc - max_allowed)²

ALGORITHM: scipy.optimize.differential_evolution

    It is a GLOBAL evolutionary algorithm, robust to non-convex spaces with
    many local minima (typical in hardpoint-placement problems).
    Slower than gradient methods, but does not require derivatives.

USAGE FLOW:
    1. Create a SuspensionCorner and TieRod as the "seed" (initial geometry)
    2. Define DesignTargets with your targets and weights
    3. (Optional) Define HardpointBounds to constrain the search
    4. Instantiate SuspensionOptimizer and call .run()
    5. Use OptimizationResult.optimal_corner and .optimal_tie_rod
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

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
# HardpointBounds — Spatial bounding box for a hardpoint
# =============================================================================

@dataclass
class HardpointBounds:
    """
    Spatial limits (box) for a variable hardpoint.

    Defines the region of space where the optimizer may move this hardpoint.
    Use it to represent KEEP-OUT ZONES (regions forbidden by the chassis,
    the engine package, packaging requirements, etc).
    """
    x_min: float; x_max: float
    y_min: float; y_max: float
    z_min: float; z_max: float

    def as_bounds(self) -> list[tuple[float, float]]:
        """Format expected by scipy.optimize.differential_evolution."""
        return [
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (self.z_min, self.z_max),
        ]

    def contains(self, point: Point3D) -> bool:
        """Test whether a point is inside the box."""
        return (self.x_min <= point.x <= self.x_max and
                self.y_min <= point.y <= self.y_max and
                self.z_min <= point.z <= self.z_max)


# =============================================================================
# DesignTargets — Targets and weights of the objective function
# =============================================================================

@dataclass
class DesignTargets:
    """
    Design targets and weights of the optimizer's objective function.

    Weights (w_*) control the RELATIVE IMPORTANCE of each term in the total
    cost. Setting a weight to 0.0 disables the corresponding term.
    Targets of type Optional[float] are DISABLED when = None.
    """
    # ─── DYNAMIC TARGETS (measured along the heave sweep) ─────────────────────
    camber_gain_target_deg_per_mm: float = -0.015   # ≈ −0.4°/inch
    bump_steer_max_abs_deg_per_mm: float =  0.010   # |bump_steer| < 0.01 °/mm
    rc_height_target_mm:           float =  50.0    # desired RC height
    rc_y_migration_max_mm:         float =  30.0    # max lateral migration

    # ─── STATIC TARGETS (measured at the neutral position) ────────────────────
    # None = ignore this target
    caster_target_deg:           Optional[float] = None    # e.g. 4.0
    kpi_target_deg:              Optional[float] = None    # e.g. 7.0
    static_camber_target_deg:    Optional[float] = None    # e.g. -1.5
    scrub_radius_target_mm:      Optional[float] = None    # e.g. 15.0
    mechanical_trail_target_mm:  Optional[float] = None    # e.g. 20.0

    # ─── Sweep range used for evaluation ──────────────────────────────────────
    heave_min_mm:  float = -25.0
    heave_max_mm:  float =  25.0
    heave_step_mm: float =   2.5

    # ─── WEIGHTS — DYNAMIC TARGETS ────────────────────────────────────────────
    w_camber_gain:  float = 1.0
    w_bump_steer:   float = 10.0
    w_rc_height:    float = 0.01
    w_rc_migration: float = 0.05

    # ─── WEIGHTS — STATIC TARGETS ─────────────────────────────────────────────
    w_caster:        float = 1.0
    w_kpi:           float = 1.0
    w_static_camber: float = 5.0
    w_scrub:         float = 0.01
    w_trail:         float = 0.01

    # ─── Penalty for configurations that break the solver ─────────────────────
    penalty_non_converged: float = 1e6


# =============================================================================
# SuspensionOptimizer — Main optimization loop
# =============================================================================

@dataclass
class SuspensionOptimizer:
    """
    Geometry optimizer for ONE suspension corner.

    DESIGN VARIABLES (12 DOF):
        UCA outboard   (x, y, z)
        LCA outboard   (x, y, z)
        Tie-rod inboard  (x, y, z)
        Tie-rod outboard (x, y, z)

    The UCA/LCA INBOARD points are kept FIXED (considered determined by the
    chassis packaging). To free more variables, extend this class.
    """
    # ─── Initial geometry (seed) ──────────────────────────────────────────────
    seed_corner:  SuspensionCorner
    seed_tie_rod: TieRod
    targets:      DesignTargets

    # ─── Bounds for each variable hardpoint ───────────────────────────────────
    bounds_uca_outboard: HardpointBounds = field(default=None)  # type: ignore
    bounds_lca_outboard: HardpointBounds = field(default=None)  # type: ignore
    bounds_tie_rod_in:   HardpointBounds = field(default=None)  # type: ignore
    bounds_tie_rod_out:  HardpointBounds = field(default=None)  # type: ignore

    # ─── differential_evolution settings ──────────────────────────────────────
    population_size: int  = 15
    max_iterations:  int  = 60
    seed:            int  = 42
    workers:         int  = 1
    polish:          bool = True
    verbose:         bool = False

    # ─── Progress callback (optional) ─────────────────────────────────────────
    # Called once per generation: on_generation(generation, best_cost, convergence).
    # Returning True STOPS the optimization (early stop), keeping the best
    # individual found so far. Useful for progress bars and time limits in UIs.
    on_generation: Optional[Callable[[int, float, float], bool]] = None

    # -------------------------------------------------------------------------
    # Initialization: create default bounds if none are provided
    # -------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Create default bounds (±50mm box) for hardpoints without bounds."""
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
    # Mapping between the design vector and the geometry objects
    # -------------------------------------------------------------------------

    def _design_bounds(self) -> list[tuple[float, float]]:
        """Concatenate all bounds into a single list (12 tuples)."""
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
        Convert a design vector (12 floats) into (SuspensionCorner, TieRod).

        Vector layout:
            [0:3]   UCA outboard
            [3:6]   LCA outboard
            [6:9]   Tie-rod inboard
            [9:12]  Tie-rod outboard
        """
        uca_out = Point3D(float(x[0]),  float(x[1]),  float(x[2]))
        lca_out = Point3D(float(x[3]),  float(x[4]),  float(x[5]))
        tr_in   = Point3D(float(x[6]),  float(x[7]),  float(x[8]))
        tr_out  = Point3D(float(x[9]),  float(x[10]), float(x[11]))

        # Keep the UCA/LCA inboards fixed (from the seed)
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
        """Design vector corresponding to the seed (12 floats)."""
        return np.array([
            *self.seed_corner.upper_arm.outboard.to_array(),
            *self.seed_corner.lower_arm.outboard.to_array(),
            *self.seed_tie_rod.inboard.to_array(),
            *self.seed_tie_rod.outboard.to_array(),
        ])

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================

    def objective(self, x: NDArray[np.float64]) -> float:
        """
        Evaluate the cost of a hardpoint configuration.

        STEPS:
            1. Build SuspensionCorner + TieRod from the vector x
            2. Create a 3D solver and run a short heave sweep
            3. Compute metrics (camber_gain, bump_steer, rc_height, rc_migration)
            4. Sum the weighted terms into a single scalar cost

        If anything fails (invalid geometry, solver does not converge), return
        the huge penalty (targets.penalty_non_converged).
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

            # If any point did not converge, discard this configuration
            if not bool(sweep["converged"].all()):
                return float(self.targets.penalty_non_converged)

        except Exception:
            return float(self.targets.penalty_non_converged)

        # ─── Term 1: camber gain ──────────────────────────────────────────────
        cg = camber_gain_per_mm(sweep)
        cost_cg = (cg - self.targets.camber_gain_target_deg_per_mm) ** 2

        # ─── Term 2: bump steer (integral of the squared Δtoe) ───────────────
        # Since the solver returns toe relative to static, sweep["toe_deg"][zero]≈0
        cost_bs = float(np.mean(sweep["toe_deg"] ** 2))

        # ─── Term 3: Roll Center height ──────────────────────────────────────
        rc_z_mean = float(np.mean(sweep["rc_z_mm"]))
        cost_rch = (rc_z_mean - self.targets.rc_height_target_mm) ** 2

        # ─── Term 4: RC migration (penalize only if it exceeds the limit) ────
        dy, _ = rc_migration_range(sweep)
        excess_y = max(0.0, dy - self.targets.rc_y_migration_max_mm)
        cost_rcm = excess_y ** 2

        # ─── STATIC TERMS (only active when target != None) ──────────────────
        # Compute the static KPIs only once
        cost_caster = cost_kpi = cost_static_camber = cost_scrub = cost_trail = 0.0

        if self.targets.caster_target_deg is not None:
            cost_caster = (corner.static_caster_deg() - self.targets.caster_target_deg) ** 2

        if self.targets.kpi_target_deg is not None:
            cost_kpi = (corner.static_kpi_deg() - self.targets.kpi_target_deg) ** 2

        if self.targets.static_camber_target_deg is not None:
            cost_static_camber = (
                corner.static_camber_deg() - self.targets.static_camber_target_deg
            ) ** 2

        if self.targets.scrub_radius_target_mm is not None:
            cost_scrub = (
                corner.static_scrub_radius_mm() - self.targets.scrub_radius_target_mm
            ) ** 2

        if self.targets.mechanical_trail_target_mm is not None:
            cost_trail = (
                corner.static_mechanical_trail_mm() - self.targets.mechanical_trail_target_mm
            ) ** 2

        return float(
            # Dynamic
            self.targets.w_camber_gain    * cost_cg
          + self.targets.w_bump_steer     * cost_bs
          + self.targets.w_rc_height      * cost_rch
          + self.targets.w_rc_migration   * cost_rcm
            # Static
          + self.targets.w_caster         * cost_caster
          + self.targets.w_kpi            * cost_kpi
          + self.targets.w_static_camber  * cost_static_camber
          + self.targets.w_scrub          * cost_scrub
          + self.targets.w_trail          * cost_trail
        )

    # =========================================================================
    # MAIN OPTIMIZATION LOOP
    # =========================================================================

    def run(self) -> "OptimizationResult":
        """
        Run the differential evolution.

        POPULATION INITIALIZATION STRATEGY:
            - 1st individual = seed (initial geometry)
            - 50% of the rest = Gaussian perturbation around the seed
            - remaining 50% = uniform sampling within the bounds

        This greatly speeds up convergence while preserving global diversity.
        """
        bounds = self._design_bounds()
        seed_vec = self._initial_guess_vector()

        # ─── Build a mixed initial population ─────────────────────────────────
        rng = np.random.default_rng(self.seed)
        n_dims = len(bounds)
        pop_size = self.population_size * n_dims
        init_pop = np.empty((pop_size, n_dims))

        for i in range(pop_size):
            for j in range(n_dims):
                lo, hi = bounds[j]
                if i == 0:
                    # 1st individual = seed
                    init_pop[i, j] = seed_vec[j]
                elif rng.random() < 0.5:
                    # Gaussian perturbation around the seed
                    sigma = (hi - lo) * 0.15
                    init_pop[i, j] = np.clip(
                        seed_vec[j] + rng.normal(0, sigma), lo, hi
                    )
                else:
                    # Uniform sampling within the bounds
                    init_pop[i, j] = rng.uniform(lo, hi)

        # ─── Per-generation callback: convergence history + early stop ────────
        # Uses the legacy scipy signature `callback(xk, convergence)`, which
        # works in any version. `xk` is the best individual of the generation;
        # re-evaluating it costs 1 extra evaluation per generation (negligible
        # compared with the population_size × 12 evaluations per generation).
        history: list[float] = []

        def _de_callback(xk, convergence: float = 0.0) -> bool:
            best_cost = float(self.objective(np.asarray(xk)))
            history.append(best_cost)
            if self.on_generation is not None:
                return bool(self.on_generation(
                    len(history), best_cost, float(convergence),
                ))
            return False

        # ─── Run differential_evolution ───────────────────────────────────────
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
            callback=_de_callback,
            updating="deferred" if self.workers != 1 else "immediate",
        )

        best_corner, best_tie_rod = self._vector_to_geometry(result.x)
        return OptimizationResult(
            optimal_corner=best_corner,
            optimal_tie_rod=best_tie_rod,
            cost=float(result.fun),
            x=result.x,
            scipy_result=result,
            convergence_history=history,
        )


# =============================================================================
# OptimizationResult — Encapsulates the result
# =============================================================================

@dataclass
class OptimizationResult:
    """Optimization result: optimal geometry + solver diagnostics."""
    optimal_corner:  SuspensionCorner
    optimal_tie_rod: TieRod
    cost:            float
    x:               NDArray[np.float64]
    scipy_result:    OptimizeResult
    # Best cost at the end of each generation (index 0 = generation 1)
    convergence_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        """Formatted summary of the result."""
        return "\n".join([
            "═══ Optimization Result ═══",
            f"  Final cost          : {self.cost:.6e}",
            f"  Iterations          : {self.scipy_result.nit}",
            f"  Objective evals     : {self.scipy_result.nfev}",
            f"  Success             : {self.scipy_result.success}",
            f"  Message             : {self.scipy_result.message}",
            "",
            "  Optimized hardpoints:",
            f"    UCA outboard      : {self.optimal_corner.upper_arm.outboard}",
            f"    LCA outboard      : {self.optimal_corner.lower_arm.outboard}",
            f"    Tie-rod inboard   : {self.optimal_tie_rod.inboard}",
            f"    Tie-rod outboard  : {self.optimal_tie_rod.outboard}",
        ])


# =============================================================================
# Validation of a geometry against the targets
# =============================================================================

@dataclass
class TargetValidation:
    """
    Result of validating a geometry against a set of targets.

    Each report row contains: name, target, obtained, absolute error,
    and whether it is within an acceptable tolerance.
    """
    rows: list[dict[str, object]]

    def as_dict_list(self) -> list[dict[str, object]]:
        """Return the data as a list of dicts (easy to turn into a DataFrame)."""
        return self.rows

    def summary(self) -> str:
        """Formatted ASCII table."""
        lines = [
            f"{'Parameter':<22} {'Target':>10} {'Obtained':>10} {'Error':>10}  Status",
            "─" * 65,
        ]
        for r in self.rows:
            status = "OK " if r["ok"] else "OFF"
            lines.append(
                f"{r['name']:<22} "
                f"{r['target_str']:>10} "
                f"{r['obtained_str']:>10} "
                f"{r['error_str']:>10}  {status}"
            )
        return "\n".join(lines)


def validate_against_targets(
    corner: SuspensionCorner,
    tie_rod: TieRod,
    targets: DesignTargets,
) -> TargetValidation:
    """
    Evaluate a geometry against all active targets.

    Runs a short heave sweep to measure the dynamic targets and computes the
    static ones directly. Returns a row-by-row report.

    TYPICAL USE (post-optimization validation):
        result = optimizer.run()
        report = validate_against_targets(result.optimal_corner,
                                          result.optimal_tie_rod, targets)
        print(report.summary())
    """
    solver = KinematicSolver3D(corner, tie_rod)
    runner = SweepRunner(solver=solver)
    sweep = runner.heave_sweep(
        targets.heave_min_mm, targets.heave_max_mm, targets.heave_step_mm,
    )

    rows: list[dict[str, object]] = []

    def _row(name: str, target: float, obtained: float, unit: str,
             tol: float, fmt: str = "+.4f") -> None:
        err = obtained - target
        rows.append({
            "name":         f"{name} ({unit})",
            "target":       target,
            "obtained":     obtained,
            "error":        err,
            "tolerance":    tol,
            "ok":           abs(err) <= tol,
            # Pre-formatted strings for the summary
            "target_str":   format(target,   fmt),
            "obtained_str": format(obtained, fmt),
            "error_str":    format(err,      fmt),
        })

    # ─── STATIC ───────────────────────────────────────────────────────────────
    if targets.caster_target_deg is not None:
        _row("Caster", targets.caster_target_deg,
             corner.static_caster_deg(), "°", tol=0.5)

    if targets.kpi_target_deg is not None:
        _row("KPI", targets.kpi_target_deg,
             corner.static_kpi_deg(), "°", tol=0.5)

    if targets.static_camber_target_deg is not None:
        _row("Static camber", targets.static_camber_target_deg,
             corner.static_camber_deg(), "°", tol=0.25)

    if targets.scrub_radius_target_mm is not None:
        _row("Scrub Radius", targets.scrub_radius_target_mm,
             corner.static_scrub_radius_mm(), "mm", tol=3.0, fmt="+.2f")

    if targets.mechanical_trail_target_mm is not None:
        _row("Mechanical Trail", targets.mechanical_trail_target_mm,
             corner.static_mechanical_trail_mm(), "mm", tol=3.0, fmt="+.2f")

    # ─── DYNAMIC (weight 0 = term disabled, left out of the report) ───────────
    if targets.w_camber_gain > 0.0:
        _row("Camber Gain", targets.camber_gain_target_deg_per_mm,
             camber_gain_per_mm(sweep), "°/mm", tol=0.005, fmt="+.5f")

    if targets.w_bump_steer > 0.0:
        bs = bump_steer_per_mm(sweep)
        _row("Bump Steer (|max|)", 0.0, abs(bs), "°/mm",
             tol=targets.bump_steer_max_abs_deg_per_mm, fmt="+.5f")

    if targets.w_rc_height > 0.0:
        rc_z = float(np.mean(sweep["rc_z_mm"]))
        _row("RC Height (average)", targets.rc_height_target_mm,
             rc_z, "mm", tol=10.0, fmt="+.2f")

    if targets.w_rc_migration > 0.0:
        dy, _dz = rc_migration_range(sweep)
        _row("RC ΔY (migration)", targets.rc_y_migration_max_mm,
             dy, "mm", tol=targets.rc_y_migration_max_mm, fmt="+.2f")

    return TargetValidation(rows=rows)
