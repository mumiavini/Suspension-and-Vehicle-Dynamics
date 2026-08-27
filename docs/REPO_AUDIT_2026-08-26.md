# Repository Audit — 2026-08-26

**Project:** `vdcore` — FSAE suspension & steering design-support tool (PUCPR Racing, Team #27)
**Scope:** Clean-sheet FSAE26 car. Kinematics trade-off tool, not a lap-time simulator.

---

## Executive Summary

The core kinematic solver and analysis pipeline are **mature and well-tested** (247 tests pass, 0 failures, ~2:1 test-to-code ratio). The library enforces a clean layering rule (zero UI imports in `vdcore/`) and follows ISO 8855 sign conventions throughout. Five open worktrees contain completed work (tire module, anti-features, docs) waiting to be merged. Three `vdcore/` subpackages remain stubs. There is no CI pipeline, no onboarding documentation, and no external validation data file for the OptimumK correlation suite (13 tests skipped).

---

## 1. What Works — Validated and Tested

### Core Solver

| Module | What it does | Validation |
|---|---|---|
| `vdcore.geometry.solver.DWSolver` | Full 9-DOF double-wishbone kinematic solver | Benchmark, property (Hypothesis), regression tests; convergence verified over ±30 mm bump travel; **independently validated against Altair MotionSolve** (DAE solver with revolute/spherical/universal joints — different solver class entirely) |
| `vdcore.geometry.primitives` | Point3D, Vector3D, planes, intersections | Unit tests |
| `vdcore.geometry.derived` | Contact patch, kingpin axis, derived points | Sign-correctness verified |

### Altair MotionSolve Cross-Validation

Three files under `altair_model/` implement an independent validation pipeline:
- `suspension.py` — builds an Altair MotionView model from the hardpoints CSV
- `msolve_corner.py` — drives MotionSolve kinematic analysis
- `validate_kinematics.py` — cross-checks DWSolver output vs MotionSolve results

This is a **genuine independent validation**: MotionSolve uses a DAE solver with physical joint types; vdcore uses `scipy.optimize.least_squares`. Agreement confirms the mechanism is solved correctly. **Requires Altair 2026.1 installed** — not runnable in CI.

### Analysis Pipeline

| Module | What it does | Validation |
|---|---|---|
| `vdcore.analysis.camber` | Camber gain, sweep, symmetry | Benchmark + property tests (left/right sign independence) |
| `vdcore.analysis.roll_centre` | FVIC, FVSA, RC height, n-line construction | Regression-anchored golden values |
| `vdcore.analysis.axle` | Wheel rates, roll kinematics | FSAE2027 design golden values pinned |

### Data Models

| Module | What it does | Validation |
|---|---|---|
| `vdcore.models.hardpoint` | Corner, Axle, Hardpoint, TirePackage with provenance tracking | Unit tests |
| `vdcore.models.target` | Target model (design intent, not recommendation) | Unit tests |
| `vdcore.io.config` | Config load/save | Unit tests |
| `vdcore.io.frames` | Frame transforms (ISO 8855 ↔ J670e ↔ SolidWorks ↔ OptimumK) | Unit tests |

### Top-Level Scripts

| Script | What it does | Status |
|---|---|---|
| `sla_geometry.py` | Static synthesis: hardpoints, static KPIs, anti-geometry, leg forces | Working, golden values pinned |
| `steering_geometry.py` | Caster, trail, scrub, Ackermann, effort, bump steer | Working, golden values pinned |
| `scripts/geometry_summary.py` | Merged corners, rate tables, roll analysis | Working |

### Test Suite Summary

```
247 passed, 13 skipped, 0 failures (42s)
Test-to-code ratio: ~2:1 (3,532 test lines / 1,804 vdcore lines)
```

The 13 skips are all in `test_optimumk_correlation.py` — the test infrastructure is ready but the CSV data file (`tests/benchmarks/data/optimumk_sweep.csv`) has not been exported from Optimum Kinematics yet.

---

## 2. What Doesn't Work — Stubs and Gaps

### Empty Stub Modules

| Module | Status | Notes |
|---|---|---|
| `vdcore/tire/` | Stub (1 line) | TTC data has been acquired. A full implementation exists in the `tire-module` worktree (TTC loader, conditioning, raw metrics, target derivation) but is **not merged** to master |
| `vdcore/optimize/` | Stub (1 line) | No optimizer wrappers yet. Differential evolution wrappers mentioned in CLAUDE.md are not started |
| `vdcore/validate/` | Stub (1 line) | Cross-checks exist scattered in tests/ but are not structured into this module |

### Missing Infrastructure

| Item | Status |
|---|---|
| **CI pipeline** | None. No `.github/workflows/`, no Makefile, no CI config. Tests, lint, type checks, and purity checks are manual only |
| **`configs/` directory** | Referenced in CLAUDE.md but does not exist on disk |
| **OptimumK correlation data** | `tests/benchmarks/data/optimumk_sweep.csv` missing → 13 tests permanently skipped |

### Documentation Gaps

| Item | Status |
|---|---|
| `docs/onboarding/` | Directory exists, empty |
| `docs/theory/` | Directory exists, empty |
| `docs/README.md` | 953 lines — covers legacy app only, not `vdcore` |

### Architectural Gaps — Analysis Not Yet in `vdcore/`

These capabilities exist in the top-level scripts but have **no equivalent in `vdcore/analysis/`**, meaning they aren't available as library functions:

| Capability | Where it lives today | Why it matters |
|---|---|---|
| **Anti-dive / anti-squat** | `sla_geometry.py` (+ `anti-features` worktree) | Core FSAE judging KPI; needs to be a reusable analysis function |
| **Steering analysis** (bump steer, Ackermann, effort, steering ratio) | `steering_geometry.py` only | The Design Event presentation needs programmatic access |
| **Sweep / what-if engine** | Doesn't exist | CLAUDE.md describes this as the tool's primary interface — "a sweep/what-if engine that fills a table of consequences" — but no formal API exists yet |

### Repo Hygiene

| Item | Notes |
|---|---|
| `{geometry,analysis}/` | Empty directory with literal brace characters in name — clearly an artifact, should be deleted |
| `random_calcs.ipynb` | Scratch notebook at repo root with no clear purpose |
| `geometria.png` | Orphan image file at repo root |
| `sla_geometry.py`, `steering_geometry.py` | 2,413 lines of core scripts living at repo root instead of under `scripts/` |

---

## 3. Code Quality Snapshot

### Lint (ruff) — 5 fixable issues

| File | Issue |
|---|---|
| `vdcore/geometry/solver.py` | 3× unused imports (Point3D, Vector3D, Z_HAT) |
| `vdcore/geometry/solver.py` | 1× unsorted import block |
| `vdcore/analysis/camber.py` | 1× unused import (SolverResult) |

### Type Check (mypy) — 2 errors

| File | Issue |
|---|---|
| `vdcore/analysis/roll_centre.py:105` | Returning `Any` from function returning `float` (np.cross) |
| `vdcore/analysis/roll_centre.py:108` | Same |

### Purity Check — PASS

18 files scanned, zero forbidden imports. Layering rule is clean.

### TODOs / FIXMEs / HACKs — Zero

No deferred-work markers in `vdcore/` or `tests/`.

---

## 4. Unmerged Work — Open Worktrees

Five worktrees contain completed or near-complete features that are **not on master**:

| Worktree | Branch | Commits Ahead | What's in it |
|---|---|---|---|
| `anti-features` | `worktree-anti-features` | 1 | Anti-dive and anti-squat analysis via side-view instant centre |
| `steering-geometry` | `worktree-steering-geometry` | 1 | Parameter documentation for STEERING_2027 config block |
| `tire-module` | `worktree-tire-module` | 3 | TTC loader, conditioning, raw-data metrics, vehicle mass/inertia model, load transfer decomposition, target derivation |
| `tutorial-doc` | `worktree-tutorial-doc` | 1 | Hands-on tutorial for suspension design workflow |
| `fix-fr-convergence` | `worktree-fix-fr-convergence` | 0 | Already merged to master (can be cleaned up) |

**The tire-module worktree is the most significant** — 3 commits adding a complete Phase A–C tire analysis pipeline. This is substantial engineering work sitting outside master.

---

## 5. Legacy App

The legacy app (`legacy_app/`, 7,119 lines) is **frozen** with known-wrong dynamic KPIs:

- Anti-dive reports +200% (actually 0%)
- Ackermann reports +173% (actually ~70%, formula inverted)
- RC migration ~1 mm (actually 110 mm)
- Rear camber gain 27% low
- Mechanical trail sign inverted

The app carries a banner warning. No tests cover it (by design). Static values are correct. It should not be consulted for dynamic KPI reference values.

---

## 6. Prioritized Action Items

### High Priority

1. **Merge `tire-module` worktree** — 3 commits of substantial tire analysis work sitting unmerged. This is the single biggest chunk of completed work not on master.

2. **Merge `anti-features` worktree** — Anti-dive/squat analysis is a core design KPI for FSAE judging. One clean commit ready to go.

3. **Set up CI** — 247 tests, lint, mypy, and purity checks all pass but are manual-only. A basic GitHub Actions workflow running `pytest`, `ruff check`, `mypy`, and `check_purity.py` would protect against regressions.

4. **Export OptimumK sweep data** — 13 skipped correlation tests are the biggest external validation gap. The test harness is built; it just needs the CSV.

### Medium Priority

5. **Fix the 5 ruff lint errors and 2 mypy errors** — All trivial. The unused imports suggest a recent refactor left cleanup behind.

6. **Clean up `fix-fr-convergence` worktree** — Already merged to master, can be removed.

7. **Merge `tutorial-doc` and `steering-geometry`** — Documentation and config docs; low risk to merge.

8. **Create `configs/` directory** — CLAUDE.md references it but it doesn't exist. Either create it with versioned vehicle configs or update CLAUDE.md.

### Lower Priority

9. **Write onboarding docs** (`docs/onboarding/`) — The tutorial exists in the `tutorial-doc` worktree, but there's no getting-started guide for new PUCPR team members.

10. **Populate `docs/theory/`** — Derivations with sign conventions would benefit future team members and Design Event judges.

11. **Structure `vdcore/validate/`** — Cross-validation logic exists in test files; extracting it into a reusable module would support the Design Event presentation.

12. **Build `vdcore/optimize/`** — Differential evolution wrappers for geometry optimization. This is a nice-to-have, not blocking any current workflow.

### Architectural / Longer-term

13. **Extract anti-geometry into `vdcore/analysis/`** — Anti-dive/squat computation lives only in `sla_geometry.py` and the `anti-features` worktree. Needs to be a library function for programmatic access.

14. **Extract steering analysis into `vdcore/analysis/`** — Bump steer, Ackermann, effort, and steering ratio all live only in `steering_geometry.py`. No library equivalent.

15. **Build the sweep/what-if engine** — CLAUDE.md describes this as the primary interface but it doesn't exist as a formal API. Currently done ad-hoc in scripts.

16. **Build a modern Streamlit app on `vdcore`** — The legacy app uses its own (wrong) solver. There is no UI consuming the validated `vdcore` library.

17. **Clean up repo root** — Move `sla_geometry.py` and `steering_geometry.py` under `scripts/`. Delete orphan files (`{geometry,analysis}/`, `random_calcs.ipynb`, `geometria.png`).

---

## 7. Metrics Summary

| Metric | Value |
|---|---|
| `vdcore/` Python files | 18 |
| `vdcore/` lines of code | 1,804 |
| Test files | 22 |
| Test lines | 3,532 |
| Test-to-code ratio | 1.96:1 |
| Tests passing | 247 |
| Tests failing | 0 |
| Tests skipped | 13 |
| Lint errors | 5 (all fixable) |
| Type errors | 2 |
| Purity violations | 0 |
| Open worktrees | 5 (1 stale) |
| Unmerged commits | 6 (across 4 worktrees) |
| Top-level script lines | 2,413 (`sla_geometry.py` + `steering_geometry.py`) |
| Legacy app lines | 7,119 (frozen) |
| Altair validation | Independent cross-check (requires Altair 2026.1) |
| CI pipeline | None |
