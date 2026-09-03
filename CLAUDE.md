# CLAUDE.md

## Project

`vdcore` — a Python library and Streamlit application for FSAE suspension and steering design support. Built by PUCPR Racing (team #27) for the FSAE26 car. This is a kinematics design-support tool, **not** a lap-time simulator. It makes design trade-offs visible, quantified, and defensible — for internal decisions and Design Event judging.

## Design principle — trade-off tool, not decision-maker

The tool computes trade-offs. The designer chooses the geometry. The tool must **never present a recommended value**. It presents consequences and sensitivities; the designer picks and defends the pick.

Concretely: targets are arguments to a query, not properties of the design space. The primary interface is a sweep/what-if engine that fills a table of consequences for each combination the designer asks about. The solver fills the table; it does not pick a row. Telling the designer a combination is infeasible is information, not a decision.

## What this project is NOT

- Not a dynamics simulator (no wheel rates, damping, frequency response)
- Not an FEA tool (no stress, deflection, fatigue)
- Not a lap-time simulator
- Not an optimizer that picks geometry — it evaluates geometry the designer proposes
- Does not compute anything that requires tire data until TTC data is acquired

## Layering rule (absolute)

`vdcore/` is a **pure library**. It may import: `numpy`, `scipy`, `pydantic`, `polars`.
It may **NEVER** import: `streamlit`, `plotly`, `PySide6`, `pyqtgraph`, `pyvista`, `matplotlib`.
Every function in `vdcore/` must be callable from a bare Python REPL.
Enforced by `scripts/check_purity.py`, `tests/unit/test_layering.py`, and a PostToolUse hook.

## Coordinate system — ISO 8855

- **X+ forward, Y+ LEFT, Z+ up.** Right-handed: X × Y = Z.
- Origin: front axle centreline, ground plane, vehicle centreline.
- **Y+ is LEFT.** Left wheels (FL, RL) have positive Y. Right wheels (FR, RR) have negative Y. This is the opposite of the old codebase and most FSAE CAD defaults.
- Rotations follow the right-hand rule: roll about X, pitch about Y, yaw about Z.
- This is the native frame for MF-Tyre / Pacejka — no conversion needed for the tire module.
- See `.claude/skills/vd-conventions/references/frames.md` for conversion matrices to J670e, SolidWorks, Optimum Kinematics, and the legacy project frame.

## Units

mm (length), N (force), Nm (torque), deg (angles in user-facing I/O), rad (angles internally).
Never mix. Use unit suffixes on ambiguous variable names: `steer_angle_deg`, `rack_travel_mm`.

## Sign conventions

- Negative camber = top of wheel inboard (both sides).
- Toe-in positive.
- **Per-side vs total must always be explicit** on every toe quantity: `toe_deg_per_side`, `total_toe_deg`. This has caused confusion on the real car — never leave it ambiguous.

## Design scope — clean-sheet, no tire data

This is a clean-sheet FSAE26 design. There is no existing car geometry to validate against. The only fixed inputs are: powertrain, brake discs, springs, and dampers. Everything else is a design variable.

The tool produces geometrically sound designs but **cannot derive the performance targets it serves** — those come from literature (Milliken RCVD, Optimum G, SAE papers) and must be tagged `source="design_intent"` with a rationale string in every report and config that uses them. Never present a literature-derived target as a measured or computed value.

## Provenance

Every `Hardpoint` carries `source: Literal["cad", "measured", "estimate", "design_intent"]` and `tol_mm: float` (required, no default). `TirePackage` carries the same.

- `cad` — extracted from the team's CAD model
- `measured` — physically measured on the car
- `estimate` — engineering estimate, not yet validated
- `design_intent` — a value chosen by the designer, not measured or computed

`static_camber_deg` and `static_toe_deg_per_side` on `Corner` are design variables (source: `design_intent`), not measured properties.

Any analysis consuming an estimate-tagged input must be able to report that fact.

## Where numbers come from

There is one kinematic solver: `vdcore.geometry.solver.DWSolver`. Axle rates and
roll behaviour come from `vdcore.analysis.axle`, driven by
`scripts/geometry_summary.py` §3b — the merged corner is the only complete one,
because `sla_geometry.py` synthesises the wishbones and `steering_geometry.py`
owns the tie rod that closes the sixth DOF.

- `sla_geometry.py` — static synthesis only: hardpoints, static KPIs, checks,
  anti-geometry, leg forces. No sweeps. It cannot build a complete corner.
- `steering_geometry.py` — caster, tie rod, rack, bump steer, Ackermann, effort.
  Supersedes sla's zero-caster outboard ball joints.
- `legacy_app/` — the 📊 Analysis tab (the old "Analysis" and
  "vdcore (validated)" tabs were merged 2026-08-27) delegates every KPI to
  vdcore via `analysis/vdcore_bridge.py`: static values, swept dynamic KPIs
  (camber gain, RC migration/height, roll camber), scrub, mechanical trail,
  anti-dive/anti-squat (`model_3d.SuspensionCorner.anti_dive_percent`, a
  corrected pivot-axis-rake construction, cross-checked against
  `sla_geometry.py`'s construction and the full 3D linkage in
  `tests/benchmarks/test_anti_geometry_and_ackermann.py`), and Ackermann
  (`vdcore_bridge.solved_ackermann_pct`, a real rack sweep on `DWSolver`
  rather than the plan-view construction — reproduces `steering_geometry.py`'s
  formula to 3 decimals). All of it is quotable now.

  The old strut-to-midpoint solver (`geometry/solver_3d.py`'s
  `KinematicSolver3D`) still runs in exactly one place: the 🌐 View 3D tab's
  sweep animation (`analysis/viz3d.py`), for visualization only — flagged
  inline, do not read camber/toe off it. Automated hardpoint synthesis (the
  old `differential_evolution`-based optimizer, since removed) is retired for
  the same reason — it scored candidates with that solver — and has no
  drop-in replacement yet: `DWSolver` is too slow per-solve for the same cost
  function.

Track is measured at the **contact patches**. `loaded_radius_mm` is the vertical
wheel-centre-to-road distance, so patches sit at z=0 and negative camber moves
them OUTBOARD (widening ground track, increasing scrub). Static camber is built
into the upright, so ball joints, KPI and kingpin length are unaffected.

Rate and roll Z values are **chassis-referenced**. Ground-referenced RC
migration differs by exactly 1 mm per mm of travel — always state which.

## Numerical conventions

Every solver returns convergence status. Non-convergence raises or returns an explicit failure object — **never a plausible-looking number silently.** Use scipy's default trust-region method (not LM — LM does not support bounds). Numerical Jacobian by default; analytic only after separate validation.

## Language and typing

All English: comments, docstrings, identifiers, docs. Full type hints on `vdcore/`. mypy strict-equivalent flags on `vdcore/` (individual flags in `pyproject.toml` overrides). `apps/` may be looser.

## Commands

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check vdcore/
uv run ruff format vdcore/

# Type check
uv run mypy vdcore/

# Purity check
python scripts/check_purity.py

# Run the Streamlit app (existing, uses .venv)
& .venv\Scripts\streamlit.exe run app.py
```

## Repository structure

```
vdcore/                     # PURE LIBRARY — zero UI imports
  models/                   # pydantic v2: Hardpoint, Corner, Vehicle, Target, TirePackage
  geometry/                 # kinematic primitives, 3D solver, frame transforms
  analysis/                 # KPIs, sweeps, camber, steering
  tire/                     # MF5.2 fitting + evaluation (stub until TTC data)
  optimize/                 # differential_evolution wrappers
  io/                       # config load/save, frame transforms, CSV/JSON export
  validate/                 # cross-checks vs Optimum Kinematics, benchmark cases
legacy_app/                 # FROZEN — original Streamlit app. Dynamic KPIs are WRONG, see below
  geometry/                 #   primitives, 2D/3D solvers, model
  analysis/                 #   io, KPIs, optimizer, sweeps, viz3d
  ui/                       #   sidebar, tabs, theme
  app.py                    #   Streamlit entry point
tests/
  unit/                     # pytest
  property/                 # hypothesis
  benchmarks/               # known-answer cases
docs/
  README.md                 # full legacy app documentation
  onboarding/               # for new team members
  theory/                   # derivations with sign conventions
scripts/                    # check_purity.py, hooks
configs/                    # versioned vehicle configs
```

## Glossary

| Term | Symbol | Unit |
|---|---|---|
| Camber | γ | deg |
| Caster | τ | deg |
| Kingpin inclination (KPI) | σ | deg |
| Scrub radius | rs | mm |
| Mechanical trail | tm | mm |
| Toe (per side) | δtoe | deg |
| Roll centre height | hRC | mm |
| Instant centre | IC | — |
| Front-view swing arm (FVSA) | — | mm |
| Side-view swing arm (SVSA) | — | mm |
| Anti-dive | %AD | % |
| Anti-squat | %AS | % |
| Wheelbase | L | mm |
| Track width | T | mm |
| Steering ratio | — | :1 |
