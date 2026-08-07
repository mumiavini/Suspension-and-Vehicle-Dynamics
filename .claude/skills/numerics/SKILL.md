---
description: "Numerical methods conventions for vdcore: residual formulation, tolerance policy, Jacobian conventions, scipy.optimize usage patterns, determinism and seeding, convergence failure protocol. Use when writing or reviewing solver code, optimizers, or any numerical computation in vdcore/."
---

## Residual formulation

Kinematic solvers express constraints as a residual vector `r(x)` where `x` is the unknown state and `r = 0` at the solution.

For the double-wishbone solver, residuals are:
- **Sphere constraints** (3): `||p - c||² - L² = 0` for each of UBJ, LBJ, TRO, where `c` is the moved inboard pivot and `L` is the link length.
- **Rigid-body constraints** (3): `||pi - pj||² - dij² = 0` for each pair in the upright triangle (UBJ-LBJ, UBJ-TRO, LBJ-TRO).

6 constraints, 9 unknowns. The remaining 3 DOF are resolved by the solver with a good initial guess (continuity from the previous sweep step).

## scipy.optimize patterns

### least_squares (for kinematic solvers)

```python
result = scipy.optimize.least_squares(
    fun=residuals,
    x0=seed,
    method='trf',      # trust-region reflective — supports bounds
    # DO NOT use 'lm' — it does not support bounds
    bounds=(lower, upper),  # optional, use if physical limits known
    ftol=1e-12,
    xtol=1e-12,
    gtol=1e-12,
    max_nfev=500,
)
```

**Why not LM**: `method='lm'` (Levenberg-Marquardt) does not support bounds in scipy. If you need bounds on the unknowns (e.g. to prevent the solver from finding a physically impossible configuration), you must use `'trf'` or `'dogbox'`.

### differential_evolution (for optimizers)

```python
result = scipy.optimize.differential_evolution(
    func=objective,
    bounds=design_bounds,
    seed=42,          # ALWAYS set for reproducibility
    maxiter=1000,
    tol=1e-6,
    polish=True,
    workers=1,        # or -1 for parallel, but ensure thread safety
)
```

## Jacobian conventions

**Default: numerical Jacobian** (`jac='2-point'` or `'3-point'`). This is the scipy default for `least_squares`.

Analytic Jacobians are only appropriate when:
1. The derivation is documented and peer-reviewed
2. A test compares the analytic Jacobian against finite-difference for multiple test points
3. The performance gain is measurable and necessary

Never add an analytic Jacobian as premature optimization. The numerical Jacobian is accurate to ~1e-8 and is rarely the bottleneck.

## Tolerance policy

| Context | ftol | xtol | Rationale |
|---|---|---|---|
| Kinematic solver | 1e-12 | 1e-12 | Sub-micron precision; residuals should be machine-epsilon |
| Optimizer objective | 1e-6 | 1e-6 | Design variables are mm and deg; 1e-6 mm is irrelevant |
| Convergence check | residual_norm < 1e-8 | — | If residuals > 1e-8 after solve, report non-convergence |

## Determinism and seeding

- Every `differential_evolution` call must set `seed=` for reproducibility.
- Every solver sweep must `reset_seed()` at the start to ensure the same initial guess path.
- Results must be bitwise reproducible across runs with the same inputs and seed.

## Convergence failure protocol

1. The solver returns a result object with `converged: bool` and `residual_norm: float`.
2. Non-convergence MUST be visible to the caller. Options:
   - Return `SolverResult(converged=False, ...)` — caller checks the flag.
   - Raise `SolverError` — for functions that cannot return a partial result.
3. **NEVER** return a plausible-looking number when the solver did not converge. This is the cardinal sin of numerical code.
4. Sweep functions must mark individual points as unconverged and report the count.
5. The `has_unconverged` property on sweep results lets callers test without scanning arrays.

## Numerical stability notes

- Normalize residuals by characteristic length (e.g., divide distance residuals by the arm length) to keep the Jacobian well-conditioned.
- For the upright rigid-body constraints, use `||p - q||² - d²` not `||p - q|| - d` to avoid the square root singularity when points coincide.
- Seed the solver from the previous sweep step's result for sweep continuity. Reset the seed at sweep boundaries.
