# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FSAE suspension geometry engine for the PUCPR Racing team (FSAE26 car): analyzes and optimizes double-A-arm suspension hardpoints. The whole project (code, comments, UI, README) is in **English** — communicate with the user in English.

## Commands

```powershell
# Run the app (project venv in .venv\)
& .venv\Scripts\streamlit.exe run app.py

# Install dependencies
& .venv\Scripts\python.exe -m pip install -r requirements.txt

# Quick sanity check of the engine (without Streamlit)
& .venv\Scripts\python.exe -c "from geometry import Point3D; print(Point3D(1,2,3))"
```

There is no test suite or linter configured. To validate changes:

- **Smoke test**: `streamlit.testing.v1.AppTest.from_file("app.py", default_timeout=60)` — run with an empty session and with the demo (`generate_template_dataframe()`), check `at.exception`.
- **Visual check**: bring the app up on an alternate port (`--server.headless true --server.port 85xx`) and use **Selenium + Edge headless** (`--headless=new`). The one-shot `msedge --screenshot` does NOT render Streamlit. The sidebar is `section[data-testid='stSidebar']` (not `div`).
- On the Windows console, set `$env:PYTHONIOENCODING='utf-8'` before running Python scripts that print emoji/accents (the app uses emoji in labels).

## Architecture

Three layers, with dependencies flowing only top-down:

1. **`geometry/`** — pure mathematical engine (numpy/scipy, zero Streamlit).
   - `primitives.py`: `Point3D`/`Vector3D`/`Point2D` + intersections.
   - `model_3d.py`: `ControlArm`, `KingpinGeometry`, `SuspensionCorner`, `Vehicle` — static KPIs (caster, KPI, scrub, trail, RC height…).
   - `solver_3d.py`: `KinematicSolver3D` — treats the upright as a rigid body and solves the position for a state `(heave, roll, rack)` via 3-sphere intersection + `least_squares` (Levenberg-Marquardt). It is the heart of the dynamic computation.
   - `solver_2d.py`: four-bar mechanism in the front Y-Z view (used for the Roll Center).
2. **`analysis/`** — uses `geometry/`:
   - `io_hardpoints.py`: reading/validation/writing (csv/xlsx/json) and construction of `SuspensionCorner`/`Vehicle` from **polars** DataFrames. Defines `VALID_CORNERS` (FL/FR/RL/RR), `REQUIRED_POINTS_PER_CORNER` (10 points per corner) and `HardpointValidationError` — all input validation goes through here.
   - `sweeps.py`: `SweepRunner` (heave/roll/steer sweeps over the 3D solver) → numpy arrays; dynamic KPIs (camber gain, bump steer, RC migration) and Plotly plots.
   - `optimizer.py`: synthesis via `scipy.optimize.differential_evolution` — `DesignTargets` (static + dynamic targets), `HardpointBounds` (keep-out), `validate_against_targets()`.
   - `kpis.py`: full-vehicle KPIs (Ackermann, steer ratio, anti-dive, RC @ 1g, `build_full_report()`).
   - `viz3d.py`: 3D Plotly visualization (corner, vehicle, animation).
3. **`app.py` + `ui/`** — Streamlit layer. `app.py` is orchestration only (page config, theme, header, sidebar, `st.tabs`); each of the 5 tabs (Inputs / Analysis / View 3D / Synthesis / Comparison) lives in `ui/tab_*.py` exposing `render()`. Support: `ui/theme.py` (presets `THEMES`, CSS, header), `ui/sidebar.py` and `ui/shared.py` (empty-state, safe builders, sweep cache via `_geometry_signature()`).

### Data flow in the app

Hardpoints file → validated polars DataFrame → `st.session_state["hardpoints_df"]` (+ `"hardpoints_source"`) is the **single source of truth**; the tabs build corners from it on every rerun. Sweeps are cached with `@st.cache_data` keyed by `_geometry_signature()` in `ui/shared.py` (hashable tuple of all hardpoints) — if you add a new hardpoint to the model, include it in the signature. The Synthesis tab (`ui/tab_synthesis.py`) uses `st.fragment` and publishes `last_optimization`, consumed by the Comparison tab. The manual editor (`ui/tab_inputs.py`) keeps its own state (`manual_hardpoints`) synced via `manual_synced_source`.

### Domain conventions

- **SAE J670** axes: origin at the center of the front axle at ground level; X+ = front, Y+ = left, Z+ = up. Units: **mm** and **degrees** (never inches/radians).
- Scope is **pure kinematics** (does not compute wheel rate, motion ratio, frequency, damping — see README §2 and §15 before promising a new KPI).

## UI conventions (keep consistent)

- Width: `width="stretch"` — **never** `use_container_width` (deprecated).
- Metrics with `border=True`; `st.segmented_control`/`st.pills` return `None` when deselected — always handle the fallback.
- Standard empty-state via `render_empty_state()` from `ui/shared.py` (inline demo button); status badges in the header.
- **Themes**: `.streamlit/config.toml` defines only the boot state; the selectable presets live in the `THEMES` dict in `ui/theme.py`, applied via `st._config.set_option("theme.*")` + forced rerun (config is global to the process). If you change `config.toml`, update `_DEFAULT_THEME` to the equivalent preset.

## Documentation

The `README.md` is the end-user documentation (physics concepts, file format, tutorial, glossary) — update it when you change visible behavior. Note: the README's tab tour (§8) is partially out of date relative to the actual tab order in `app.py`.
