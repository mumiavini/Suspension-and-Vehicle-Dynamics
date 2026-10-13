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

## `apps/vdstudio/` -- VD Studio desktop app

One desktop application (PySide6 + pyqtgraph) that carries the car through the
whole FSAE season: concept numbers and targets, SLA and steering synthesis with
live construction views, hardpoint editing and CAD exchange, kinematic sweeps,
a live 3D model driven by heave/roll/pitch/steer, a trade-off (what-if) engine,
quasi-static maneuvers, telemetry validation (MoTeC `.ld` / CSV) with kinematic
replay, OptimumK / Altair cross-checks, the setup sheet and a Markdown/PDF
design report. Every number comes from `vdcore` or the design scripts above,
and the tool never recommends a value.

Windows: double-click **`Install VD Studio shortcut.bat`** once, then open VD
Studio from the Desktop or Start-menu icon. Or, from a terminal:

```bash
uv sync --extra desktop --inexact
uv run --extra desktop python -m apps.vdstudio      # or double-click "VD Studio.bat"
```

User guide: [`docs/VD_STUDIO.md`](docs/VD_STUDIO.md)

![VD Studio](docs/images/vdstudio_dashboard.jpg)

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
apps/vdstudio/    VD Studio desktop app (engine/ is Qt-free, ui/ is PySide6)
legacy_app/       Original Streamlit app (geometry, analysis, ui, app.py, sample data)
tests/            pytest unit, property, and benchmark tests for vdcore
scripts/          Dev scripts (purity check, hooks, export utilities)
docs/             Documentation (legacy app README, onboarding, theory)
configs/          Versioned vehicle configurations
```
