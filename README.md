# FSAE Suspension & Vehicle Dynamics

Python tools for FSAE suspension and steering design. Built by PUCPR Racing (team #27) for the FSAE26 car.

This repository contains two things:

## `vdcore/` -- Pure design-support library (active development)

A typed Python library for suspension kinematics, steering analysis, and design-space exploration. Computes trade-offs -- the designer picks the geometry.

- Pydantic v2 models with provenance tracking
- ISO 8855 coordinate frame (X+ forward, Y+ left, Z+ up)
- 3D kinematic solver, camber/toe/caster extraction, roll centre, anti-features
- Sweep/what-if engine for design-space exploration
- No UI dependencies -- callable from a bare Python REPL

```bash
# Install (library + dev tools)
uv sync --all-extras

# Run tests
uv run pytest

# Type check
uv run mypy vdcore/
```

## `legacy_app/` -- Original Streamlit application (frozen)

The first version of the tool: a Streamlit app with analysis, synthesis/optimization, manual editing, comparison, and 3D visualization tabs. Still functional but not under active development -- new work goes into `vdcore/`.

```bash
cd legacy_app
pip install -r requirements.txt
streamlit run app.py
```

Full documentation for the legacy app: [`docs/README.md`](docs/README.md)

## Repository layout

```
vdcore/           Pure library (models, geometry, analysis, io, tire, optimize, validate)
legacy_app/       Original Streamlit app (geometry, analysis, ui, app.py, sample data)
tests/            pytest unit, property, and benchmark tests for vdcore
scripts/          Dev scripts (purity check, hooks, export utilities)
docs/             Documentation (legacy app README, onboarding, theory)
configs/          Versioned vehicle configurations
```
