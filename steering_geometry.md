# steering_geometry.py — Tie Rod and Rack Synthesis

## Context

`sla_geometry.py` synthesises the double-A-arm hardpoints for all four corners but stops at the wishbones. It emits no tie rod, no rack, and — critically — no caster: `sla_geometry.py:355` builds the kingpin from KPI alone in the front view, and `build_corner` places both ball joints at `x = axle_x_mm`. Caster and mechanical trail are therefore identically zero, and mechanical trail is the dominant term in steering feel and effort.

`steering_geometry.py` is a sibling script that consumes `sla_geometry.py`'s front-axle solution, adds the steering-specific inputs in its own manually-edited config block, and outputs the tie rod and rack hardpoints plus the steering behaviour over bump and steer sweeps.

The geometry is underdetermined — 6 free parameters per side (TRO 3 + rack_x + rack_z + rack half-length) against roughly 3 target equations. Per `CLAUDE.md`, the script therefore never picks a value: `SteeringInputs` is all designer choice, and the two back-solvers are opt-in helpers in the style of the existing `dz_uca_for_anti()`.

---

## Decisions Taken

| Decision | Choice |
| --- | --- |
| **3D engine** | Reuse `vdcore.geometry.solver.DWSolver` via a frame bridge |
| **Dynamics scope** | Kinematics + steering effort (rack force, wheel torque). No tie rod structural loads. |
| **Synthesis mode** | Direct inputs + two opt-in back-solvers |
| **Caster** | `caster_deg` + `caster_offset_mm`; script emits corrected `UCA_OUT`/`LCA_OUT` |

---

## Why DWSolver

Bump steer is a rotation about the kingpin axis — it cannot be extracted from `sla_geometry.py`'s front-view four-bar, which resolves the upright in the $(y, z)$ plane only. `DWSolver` already solves the 9-unknown rigid-upright problem (3 ball joints × 3 coords against 3 sphere + 3 link + 3 inter-distance constraints), accepts `rack_mm`, returns `toe_deg_per_side`, and reports `converged`. Reimplementing it in a root script would duplicate ~300 lines of tested code.

> **Consequence to state in the report:** The 3D solve is more accurate than `sla_geometry.py`'s 2D sweep, so camber-vs-bump from this script will differ slightly from the sibling script's. That is expected, not a bug.

---

## File Structure

`steering_geometry.py` sits at the repo root, mirroring `sla_geometry.py`'s numbered-section layout, frozen `kw_only` dataclasses, `_flag`/`_band` report helpers, and CLI.

1. **INPUTS:** `SteeringLimits`, `SteeringInputs`
2. **CASTER:** `apply_caster()` -> corrected 3D ball joints, kingpin axis, trail, scrub
3. **STATIC SYNTHESIS:** `tro_from_arm_params()`, `tri_from_rack_params()` -> `SteeringGeometry`
4. **VDCORE BRIDGE:** `build_vdcore_corner()`
5. **KINEMATICS:** `SteeringKinematics`, `SteeringRates`
6. **BACK-SOLVERS:** `y_tri_for_zero_bump_steer()`, `arm_angle_for_ackermann_target()`
7. **EFFORT:** `rack_force_parking()`, `steering_wheel_torque()`
8. **HARDPOINTS:** `SteeringHardpoints` (ISO 8855)
9. **REPORT:** `steering_report()`, `notes_report()`
10. **PLOTTING:** `plot_steering()` (lazy matplotlib import, as in `sla_geometry.plot_all`)
11. **CONFIG:** `STEERING_2027` <- the manual input block
12. **TOP LEVEL:** `run()`, `main()`

### Imports

* **From `sla_geometry`:** `VehicleData`, `AxleInputs`, `AxleGeometry`, `solve_axle`, `build_corner`, `KinematicError`, `D2R`, `R2D`, `nz`, `Band`, `_flag`, `_band`, `_RULE`, `VEHICLE_2027`, `FRONT_2027`.
* **From `vdcore`:** `DWSolver` from `vdcore.geometry.solver`; `Corner`, `Hardpoint`, `TirePackage` from `vdcore.models.hardpoint`.

---

## Section 1 — The Manual Input Block

`SteeringInputs`, frozen + `kw_only`, docstring *"Everything the designer chooses for the steering system. Nothing here is derived."* — matching `AxleInputs`. Inputs are in the design frame (y outboard, z up, x positive REARWARD) so the designer thinks in the same terms as `sla_geometry.py`; conversion to ISO 8855 happens only in section 8.

```python
# kingpin axis in side view (absent from sla_geometry.py)
caster_deg: float = 5.0
caster_offset_mm: float = 0.0      # X shift of the kingpin axis at wheel-centre height

# steering arm — cylindrical parameterisation about the kingpin axis
tro_height_along_kingpin_mm: float   # from LBJ, up the kingpin axis
steer_arm_length_mm: float           # perpendicular from the kingpin axis
steer_arm_angle_deg: float           # 0 = forward, +ve toward INBOARD, mirror-symmetric

# rack — axis assumed lateral and horizontal (along Y)
rack_x_mm: float                     # +ve REARWARD of the front axle
rack_z_mm: float
rack_half_length_mm: float           # y of the inner joint at zero steer

# rack hardware
pinion_radius_mm: float
max_rack_travel_mm: float            # half-stroke, hard limit
steering_wheel_diameter_mm: float = 260.0

# static alignment (Corner requires it; sla_geometry.py has no toe input)
static_toe_deg_per_side: float = 0.0

# design intent targets (arguments to a query, never defaults the script chases)
target_ackermann_pct: float = 100.0
ackermann_at_steer_deg: float = 10.0   # Ackermann is not constant — the abscissa is required
target_bump_steer_deg_per_mm: float = 0.0

# effort
mu_parking: float = 1.0

# sweeps
steer_sweep_deg: float = 25.0
n_sweep: int = 21
hardpoint_tol_mm: float = 1.0        # feeds Hardpoint.tol_mm on the vdcore bridge
```

`SteeringLimits` mirrors `CheckLimits` with `Band` tuples for `bump_steer_deg_per_mm`, `ackermann_pct`, `steering_ratio`, `tie_rod_length_mm`, `rod_end_misalignment_deg` (max), `mechanical_trail_mm`, `steering_wheel_torque_Nm` (max), and `rack_x_window_mm` / `rack_z_window_mm` for packaging.

---

## Section 2 — Caster

`sla_geometry.py` hands over `geo.lbj` / `geo.ubj` as front-view `Vec2` at `x = axle_x_mm`. Lift to 3D and apply caster ($\tau$):

$$x_{lbj} = \text{axle\_x} + \text{caster\_offset\_mm} + (lbj_z - \text{loaded\_radius}) \cdot \tan(\tau)$$

$$x_{ubj} = \text{axle\_x} + \text{caster\_offset\_mm} + (ubj_z - \text{loaded\_radius}) \cdot \tan(\tau)$$

Pivoting about wheel-centre height means `caster_offset_mm` moves the axis bodily and `caster_deg` tilts it, giving independent control of caster angle and mechanical trail. Then:

* **Kingpin ground intercept, mechanical trail** = intercept X vs contact patch X (positive = intercept forward of the patch).
* **Scrub radius** recomputed in 3D — cross-check against `geo.scrub_radius_mm`, which is the zero-caster front-view value.

These corrected outboard points supersede `sla_geometry.py`'s and are emitted in section 8. The report must say so explicitly.

---

## Section 3 — TRO and TRI

Orthonormal frame on the kingpin axis:

$$e_{kp} = \text{unit}(UBJ - LBJ)$$

$$e_{fwd} = \text{unit component of forward perpendicular to } e_{kp}$$

$$e_{lat} = e_{kp} \times e_{fwd}$$

$$TRO = LBJ + h \cdot e_{kp} + L \cdot (\cos \theta \cdot e_{fwd} + \sin \theta \cdot e_{lat})$$

This is the right parameterisation because steering is rotation about the kingpin axis, so `steer_arm_length_mm` is directly the moment arm that sets C-factor and effort.

> **Sign hazard:** The main correctness risk in this file. `steer_arm_angle_deg` is measured from forward, positive toward the vehicle centreline, so the same number applies to both sides and the mirror is handled by the sign of $e_{lat}$. `CLAUDE.md` flags left/right sign handling as a known hazard; the symmetry test below exists to catch it.

* $TRI = (\text{rack\_x\_mm}, \pm\text{rack\_half\_length\_mm}, \text{rack\_z\_mm})$
* Tie rod length derived as $|TRI - TRO|$, never an input.

`SteeringGeometry` (frozen) carries: `tro`, `tri` per side, `tie_rod_length_mm`, `steer_arm_length_mm`, `kingpin_axis`, `mechanical_trail_mm`, `scrub_radius_mm`, `geometric_ackermann_pct` (from the kingpin→TRO line vs the rear axle centre).

---

## Section 4 — vdcore Bridge

`build_vdcore_corner(front_geo, steer, side, corner_id) -> Corner`

Takes `sla_geometry.build_corner()` output for the 4 inboard pickups and the wheel centre, substitutes the caster-corrected `UCA_OUT`/`LCA_OUT`, adds `TIE_ROD_IN`/`TIE_ROD_OUT`, and wraps each as a `Hardpoint`. `Hardpoint` requires source and `tol_mm` with no defaults — use `source="design_intent"` and `tol_mm=steer.hardpoint_tol_mm`. `TirePackage` from `FRONT_2027.loaded_radius_mm`. `static_camber_deg` from `AxleInputs`, `static_toe_deg_per_side` from `SteeringInputs`.

`Corner` validates the Y sign (FL/RL positive), so the design→ISO conversion $Y_{iso} = \pm y_{outboard}$, $X_{iso} = -x_{rearward}$ must be applied here, matching `build_corner`.

---

## Section 5 — Kinematics

`SteeringKinematics` holds a `DWSolver` for FL and FR and drives both with the same `rack_mm` — a translating rack carries both inner joints together, which is exactly what `_move_chassis_points` models.

Every `solve()` result is checked for `converged`; a failure raises `KinematicError` with the state that failed. Never return a plausible number silently.

`SteeringRates` (frozen):

* `bump_steer_deg_per_mm_per_side` — central difference of `toe_deg_per_side` about static. Also report total_toe change; `CLAUDE.md` requires per-side vs total to be explicit everywhere.
* `toe_at_full_bump_deg_per_side`, `toe_at_full_droop_deg_per_side`
* `c_factor_mm_per_deg` — numerical: solve at `rack_mm = ±ε`, take $2\varepsilon / \Delta\delta_{deg}$
* `steering_ratio` = $360 \cdot C / (2\pi \cdot \text{pinion\_radius\_mm})$
* `max_steer_at_stroke_deg` — road wheel angle at `±max_rack_travel_mm`
* `ackermann_pct_at_target` and the full Ackermann curve. Ideal inner angle for a given outer angle:

$$\cot \delta_{i,\text{ideal}} = \cot \delta_o - \frac{T}{L}$$

$$\text{Ackermann\%} = 100 \cdot \frac{\delta_{i,\text{actual}} - \delta_o}{\delta_{i,\text{ideal}} - \delta_o}$$

with $T$ from `FRONT_2027.track_mm` and $L$ from `VehicleData.wheelbase_mm`.

* `worst_rod_end_misalignment_deg` over the bump × steer envelope. Approximation to declare in the report: this is the swing of the tie-rod direction relative to its static direction, not the true per-end misalignment, which depends on how each rod end is clocked in its housing. It is a screening check.

---

## Section 6 — Back-Solvers (Opt-In)

Same shape as `dz_uca_for_anti()`: bracket, `brentq`, return `nan` when no sign change is found rather than guessing.

* `y_tri_for_zero_bump_steer(front_geo, steer, veh)` — sweeps `rack_half_length_mm` for `bump_steer_deg_per_mm == target_bump_steer_deg_per_mm`
* `arm_angle_for_ackermann_target(front_geo, steer, veh)` — sweeps `steer_arm_angle_deg` for `ackermann_pct_at_target == target_ackermann_pct`

Neither runs in the default `run()`. They are printed as suggestions-on-request only, and the report labels their output as "the value that would hit this target", not a recommendation.

---

## Section 7 — Steering Effort

Use virtual work with the solver-derived C-factor rather than the closed-form shorthand, so the real linkage is accounted for automatically:

* **Kingpin moment per wheel:**
$$M_{kp} = \mu \cdot F_z \cdot \sqrt{r_s^2 + t_m^2} \text{ [N·mm]}$$
(where $r_s$ is scrub radius, $t_m$ is mechanical trail).

* **Rack force:**
$$F_{rack} = \sum_{\text{wheels}} M_{kp} \cdot \left(\frac{\pi}{180}\right) / C \text{ [N]}$$
(from $F_{rack} \cdot dt = M_{kp} \cdot d\delta$).

* **Steering wheel torque:**
$$T_{sw} = \frac{F_{rack} \cdot \text{pinion\_radius\_mm}}{1000} \text{ [N·m]}$$

* **Rim force:**
$$F_{rim} = \frac{T_{sw}}{(\text{steering\_wheel\_diameter\_mm} / 2000)} \text{ [N]}$$

$F_z$ per front wheel from `VehicleData` (`total_mass_kg`, `front_mass_fraction`), reusing the same approach as `load_cases`. Report the effort budget split (friction / trail / scrub) per the steering-design skill table.

---

## Section 8 — Hardpoints Output

`SteeringHardpoints` mirroring `ModelHardpoints` with `rows()` and `to_dict()`, in ISO 8855, for FL and FR:

* `TIE_ROD_IN`, `TIE_ROD_OUT` — the deliverable
* `UCA_OUT`, `LCA_OUT` — caster-corrected, flagged as superseding `sla_geometry.py`'s

Same CSV columns as the sibling script (`corner`, `point`, `x_mm`, `y_mm`, `z_mm`) so both outputs concatenate into one hardpoint table.

---

## Sections 9–12

* **Report:** Reuses `_flag` / `_band` / `_RULE` from `sla_geometry` for band checks on bump steer, Ackermann, steering ratio, trail, rod-end misalignment, tie rod length, effort, and the rack packaging windows. `notes_report()` lists which values are `design_intent`, as `sla_geometry.py:1064` does.
* **Plotting:** `plot_steering()` — 4 panels: toe vs bump (bump steer), Ackermann vs steer angle, inner/outer steer vs rack travel, camber vs steer. Lazy matplotlib import inside the function, Agg backend.
* **Config:** `STEERING_2027` config block at the bottom, next to a comment pointing at it.
* **Top Level:** `main()` with `--json`, `--csv`, `--plot`, `--quiet`, `--no-sweep`, plus `--solve-bump-steer` and `--solve-ackermann` to invoke the back-solvers.

---

## Known Limitation to Document

`DWSolver` translates the inner joint purely along Y (`solver.py:264`), so the rack axis is assumed lateral and horizontal. That covers essentially every FSAE rack. An inclined rack would need a rack-axis direction added to `_move_chassis_points`. State this in the module docstring rather than working around it.

---

## Verification

### CLI Commands

```bash
# 1. Runs clean, report prints
& .venv\Scripts\python.exe steering_geometry.py

# 2. Exports
& .venv\Scripts\python.exe steering_geometry.py --json steering.json --csv steering.csv --plot

# 3. Library still pure, lint and types
python scripts/check_purity.py
uv run ruff check steering_geometry.py
uv run pytest
```

### Physics checks

To run in a REPL and record in the module docstring:

* **Static recovery** — At `bump=0`, `rack=0`, both corners must return `toe_deg_per_side == static_toe_deg_per_side` to solver tolerance.
* **Symmetry** — With a symmetric config, FL and FR must give equal camber and equal-and-opposite toe at zero rack; at a non-zero rack the inner/outer roles must swap cleanly when the rack sign flips. This is the test that catches a `steer_arm_angle_deg` sign error.
* **Zero bump steer construction** — Place the inner joint on the line through the FVIC (`front_geo.fvic`) and confirm the bump steer gradient collapses toward zero. Known-answer check on the whole chain.
* **100% Ackermann construction** — Place TRO on the line from the kingpin to the rear axle centre; `ackermann_pct_at_target` must read ≈100 at small steer angles.
* **C-factor cross-check** — The numerical C-factor must agree with the small-angle estimate from `steer_arm_length_mm` and the rack-to-arm angle, within a few percent.
* **Convergence** — Sweep the full bump × steer envelope and confirm every `SolverResult.converged` is `True`; any failure must raise, not return a number.
