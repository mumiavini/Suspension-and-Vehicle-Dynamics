# VD Studio — user guide

VD Studio is the PUCPR Racing desktop workbench for suspension and steering. It
puts every tool in this repository behind one window and follows the car through
the FSAE season: concept numbers, SLA synthesis, steering, kinematic
simulation, trade-off sweeps, quasi-static maneuvers, telemetry validation,
CAD exchange, cross-checks and the Design Event documentation.

> **The tool computes trade-offs; the designer chooses the geometry.** Nothing
> in VD Studio recommends a value. Bands are the design-intent windows recorded
> in `CheckLimits` / `SteeringLimits` and are shown as *information*. A target
> is only ever an argument to a query ("which value of X puts Y at T?"), and a
> query result is applied only when you click to apply it — recorded and
> undoable.

![Dashboard](images/vdstudio_dashboard.jpg)

## Install and run

Windows (PowerShell, from the repository root):

```powershell
uv sync --extra desktop          # once; adds PySide6, pyqtgraph, PyOpenGL
uv run --extra desktop python -m apps.vdstudio
```

or double-click **`VD Studio.bat`** in the repository root. Open a project
directly with `python -m apps.vdstudio path\to\car.vdproj.json`.

Linux/macOS: `./run_vdstudio.sh`. Headless machines without OpenGL can set
`VDSTUDIO_NO_GL=1` (the 3D view is replaced by a notice; everything else works).

## The workflow, stage by stage

The Dashboard tracks nine FSAE stages (status + notes, saved in the project).
Each maps onto the pages below.

| Stage | Where | What you do there |
|---|---|---|
| 1. Concept & targets | **Concept & targets**, **Design log › targets** | Mass/CG (subsystem estimator), wheelbase/track rules, tilt margin, roll-stiffness budget from a roll-gradient target, chassis torsion factor, first-order camber recovery. Targets carry rationale + reference. |
| 2. Suspension geometry | **Suspension geometry** | `sla_geometry.py` synthesis: RC + FVSA place the FVIC, the kingpin places the ball joints, pickup planes place the chassis points, base/sweep/rise set the side view and anti-geometry. Live front/side/top construction views. |
| 3. Steering | **Steering** | `steering_geometry.py`: caster/offset, steering arm, rack, hardware. Ackermann and ratio solved on the real linkage. Target queries for zero bump steer and an Ackermann value. |
| 4. Kinematic validation | **Kinematic sweeps**, **3D motion**, **Trade-offs**, **Quasi-static maneuver** | Heave/roll/steer curves, live 3D, full-factorial sweeps, sensitivity matrix, steady-state load transfer and road cambers. |
| 5. Cross-check | **CAD & cross-checks** | OptimumK point table; Altair MotionSolve replay (needs a local Altair install). |
| 6. CAD | **Hardpoints**, **CAD & cross-checks** | Freeze the synthesis, export CSV in ISO / SolidWorks / OptimumK frames, SolidWorks VBA macro or live push/pull (Windows + pywin32), diff a CAD read-back. |
| 7. Manufacturing | **Hardpoints** | Edit measured points directly (provenance `measured`, tolerance per point), symmetric editing. |
| 8. Testing & telemetry | **Telemetry** | MoTeC `.ld` / CSV import, channel mapping, data quality, physics consistency, roll/pitch/understeer gradients vs targets, kinematic replay of the measured travel. |
| 9. Design Event | **Results & reports**, **Design log** | All KPIs with band and provenance, setup sheet, Markdown/PDF report, decision log with KPI snapshots. |

![Suspension geometry: live synthesis with the front-view IC/RC construction](images/vdstudio_geometry.jpg)

![3D motion: every frame is a DWSolver solve of all four corners](images/vdstudio_motion.jpg)

## Geometry sources

* **Synthesis** (default): the design inputs of `sla_geometry.py` and
  `steering_geometry.py` plus the rear toe link. The merge is the same one
  `scripts/geometry_summary.py` performs — a fresh project reproduces
  `Geometry Summary/hardpoints_2027_merged.csv` exactly (pinned by
  `tests/app/test_engine_geometry.py`).
* **Hardpoints**: points given directly — a frozen synthesis, a CAD read-back
  or measured points. Static camber, static toe and the tyre loaded radius are
  entered per axle because a point table cannot carry them.

Every hardpoint carries `source` (`cad` / `measured` / `estimate` /
`design_intent`) and a tolerance. The status bar counts estimate-tagged inputs.

## Real time

Every edit re-synthesises the car (~3 ms) and recomputes the static KPIs on
`DWSolver` (~10 ms), so views and tables follow the spin boxes. The swept KPIs
(rates, roll, bump steer, Ackermann, rod-end envelope; ~2 s) run on a worker
thread after a short pause; the status bar shows progress and stale values are
greyed until they land. Undo/redo (Ctrl+Z / Ctrl+Y) covers every edit.

## Conventions (same as the library)

* ISO 8855: **X forward, Y LEFT, Z up**; origin at the front axle centreline on
  the ground. Right-hand corners have negative Y.
* Camber negative = top inboard. Toe positive = toe-in, **always per side**
  unless a label says total.
* Wheel travel positive in bump. Rate and roll Z values are
  **chassis-referenced**.
* Synthesis inputs keep the scripts' DESIGN frame (y outboard, x rearward);
  every such field says so in its tooltip.
* Positive roll lifts the LEFT side (a left turn); positive rack steers left.

## Where every number comes from

| Quantity | Source |
|---|---|
| Static camber/toe/caster/KPI | `vdcore.geometry.solver.DWSolver` |
| Scrub radius, mechanical trail | `vdcore.geometry.derived` on the solved corner |
| Roll centre (static), FVIC, FVSA | `vdcore.analysis.roll_centre.roll_centre_height`: force-based RC (Milliken RCVD Ch. 17, SAE J670); FVIC = the upright's instantaneous front-view pole at the contact-patch station, from the solved UBJ/LBJ/tie rod (`instant_axis_fvic`). Chassis-referenced. |
| Camber gain, RC migration, half-track change | `vdcore.analysis.axle.axle_rates` (same instant-axis FVIC) |
| Roll cambers, RC in roll | `vdcore.analysis.axle.axle_roll` (both patches on the road; same instant-axis FVIC) |
| Bump steer (linear + peak) | `vdcore.analysis.toe.bump_steer` |
| Anti-dive / anti-squat | pivot-axis-rake construction (as `sla_geometry` / legacy `model_3d`) on the merged corner |
| Ackermann | rack sweep on `DWSolver` (reproduces `steering_geometry` to 0.01 %) |
| Steering ratio, effort | `steering_geometry` formulas on solved scrub/trail/rack-per-degree |
| Rod-end swing | `DWSolver` bump × steer envelope, both rod ends in one frame |
| Member lengths, e/a | the merged hardpoints (as `geometry_summary.member_lengths`) |
| Roll budget, load cases, leg forces | `sla_geometry.solve_vehicle` / `load_cases` / `leg_forces` |

## Telemetry

Supported: MoTeC i2 `.ld` (read natively, no extra package) and CSV/TSV
(metadata lines above the header and a units row are handled; `;` + decimal
comma accepted). Channels at different rates are resampled onto the fastest.

1. **Channel mapping** — auto-mapped by name, unit conversion guessed from the
   unit string or magnitude (and explained). Flip a sign where the logger uses a
   different convention. Motion ratio (damper per wheel travel) is an input:
   this tool does not model the pushrod, so it stays `estimate` until measured.
2. **Data quality** — per channel: rate, missing samples, flat-lines, spikes,
   saturation. Physics consistency with no model: yaw rate must equal
   `ay·g/V` in cornering (catches unit and sign errors in three channels at
   once, with a diagnosis), steering and roll must rise with `ay`, sensors must
   read zero at standstill, net heading tells the lap direction.
3. **Model vs measurement** — roll gradient per axle from the dampers against
   the design-intent target, pitch gradient, understeer gradient
   (`δ_road − L/R` vs `ay`), measured LLTD if wheel loads are logged.
4. **Kinematic replay** — the measured wheel travel and rack are pushed through
   camber/toe maps solved on `DWSolver` for the current geometry: road camber
   per corner over the lap, its distribution while cornering, and the share of
   cornering time the outer wheel spent inside the outer-camber band.

No tyre data is used anywhere — PUCPR Racing has no TTC data yet.

![Telemetry: kinematic replay of a log through the current geometry](images/vdstudio_telemetry.jpg)

## Trade-offs

* **Grid sweep** — 1 or 2 inputs, any outputs; the table is the result (not
  sorted by merit). Two inputs also give a map. Runs on a process pool.
* **Sensitivity matrix** — change of each output for one step of each input
  about the current design.
* **Target query** — every crossing of a target inside a range, by scan +
  `brentq`; "no crossing" is reported as such.

![Trade-offs: a 2-input grid, unranked](images/vdstudio_tradeoff.jpg)

## Out of scope (by design)

Wheel/ride rates from the linkage, damping, frequency response, stress and
buckling, lap time, and anything that needs tyre data. The setup sheet derives
wheel rate and natural frequency only from the spring and motion-ratio values
you type, and says so.

## For developers

```
apps/vdstudio/engine/   Qt-free logic (testable from a REPL)
  project.py            Project model, JSON (schema_version), dotted-path edits
  geometry.py           synthesis / hardpoint mode -> CarGeometry (vdcore Vehicle)
  kpis.py               every KPI with unit, source, band, failure state
  curves.py, pose.py    sweeps for plots; live 3D pose
  concept.py, maneuver.py, whatif.py, exports.py
  telemetry/            ld.py, session.py, mapping.py, analysis.py
apps/vdstudio/ui/       PySide6 + pyqtgraph presentation
```

* `vdcore/` is untouched and stays pure (the layering test still guards it).
* Tests: `uv run pytest tests/app` (the UI smoke test runs headless and skips
  when PySide6 is absent).
* A new KPI: add it in `engine/kpis.py` with its source and band; it appears in
  every table, the report, the trade-off outputs and the dashboard search.
