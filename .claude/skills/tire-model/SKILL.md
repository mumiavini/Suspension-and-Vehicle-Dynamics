---
description: "Tire modelling: TTC data loading, raw-data analysis, sign conventions, load and camber sensitivity, MF5.2 fitting (future). Use when writing tire-related code in vdcore/tire/."
---

## Current status: RAW DATA ANALYSIS

PUCPR Racing has TTC (Tire Test Consortium) data. The current module loads raw .mat files, applies sign-convention corrections, conditions the data, and computes design-relevant metrics from binned raw data — no Magic Formula fitting yet. Fitting comes later; 80 % of design-relevant information is readable from binned raw data.

## Sign convention — adapted-SAE (TTC) → ISO 8855 (this project)

**This is the single most common source of bugs when importing tire data. Every channel must be derived from the frame definitions, not guessed.**

### Frame definitions

| Axis | Adapted SAE (TTC raw) | ISO 8855 (this project) |
|---|---|---|
| X | Forward | Forward |
| Y | **Right** | **Left** |
| Z | **Down** | **Up** |

Both frames are right-handed with X forward, but Y and Z are inverted.

### Channel-by-channel derivation

A force that points physically rightward has:
- Adapted SAE: positive FY (Y+ = right)
- ISO 8855: **negative** FY (Y+ = left)

Therefore **FY flips sign** when converting from adapted SAE to ISO 8855.

The same logic applies to every quantity whose physical direction involves Y or Z:

| Channel | Physical meaning | Axis involved | Flips? | Reason |
|---|---|---|---|---|
| SA (slip angle) | Angle of velocity from heading, positive = wheel pointed right of travel in SAE | Y (lateral) | **Yes — negate** | ISO 8855 positive SA = wheel pointed left of travel |
| FX | Longitudinal force | X | No | X axis is the same in both frames |
| FY | Lateral force | Y | **Yes — negate** | Y axis reverses |
| FZ | Normal (vertical) load | Z | **Yes — negate** | TTC FZ is negative (downward in Z-down); ISO FZ is positive (upward, Z-up). After negation a loaded tire has FZ > 0 |
| MX | Overturning moment | About X | **Yes — negate** | Moment about X follows from Y/Z axis reversal |
| MZ | Self-aligning torque | About Z | **Yes — negate** | Positive MZ in SAE = clockwise (top view); ISO 8855 positive MZ = counterclockwise (right-hand rule about Z-up) |
| IA | Inclination angle (camber) | Tilt about X, measured toward Y | **Yes — negate** | Positive IA in SAE = tilt toward Y+ (rightward); in ISO positive IA = tilt toward Y+ (leftward) |
| P | Inflation pressure | Scalar | No | |
| V | Travel velocity | Scalar magnitude | No | |
| RL | Loaded radius | Scalar length | No | |
| RE | Effective rolling radius | Scalar length | No | |
| SR | Slip ratio | Longitudinal | No | Defined as (Vx − Re·ω)/Vx; X axis is the same |
| ET | Elapsed time | Scalar | No | |
| TSTC/TSTI/TSTO | Tread temperatures | Scalar | No | |

### Validation rule

After conversion, a **positive slip angle** must produce a **positive lateral force** (leftward in ISO 8855) for a normal tire. This is the fundamental sanity check. If it fails, signs are wrong.

### Implementation

The conversion is applied once, at load time, in `vdcore.tire.ttc.load_ttc_mat()`. All downstream code sees only ISO 8855 signs and units. The raw file is never modified.

## Units after conversion

| Channel | Unit | Notes |
|---|---|---|
| SA | deg | Kept in degrees (TTC native); internal code converts to rad where needed |
| FX, FY, FZ | N | |
| MX, MZ | Nm | |
| IA | deg | |
| P | kPa | TTC ships kPa |
| V | km/h | TTC ships km/h |
| RL, RE | mm | TTC ships m; multiply by 1000 |
| SR | dimensionless | |
| ET | s | |
| TSTC/TSTI/TSTO | °C | |

## Conditioning

TTC runs contain warmup sweeps, pressure transients, and temperature drift that must not enter analysis. Standard filters:

1. **Warmup drop**: remove the first N seconds of each sweep
2. **Pressure band**: keep only rows within ±X kPa of target pressure
3. **Velocity band**: keep only rows within a target speed range
4. **Temperature window**: drop rows where tread temperature is outside a stated range

Every filter **must report how many rows it removed**. Silent data loss is not acceptable.

## Raw-data metrics (no curve fitting)

All metrics are computed per bin of (FZ, IA, P):

- **peak_mu_lateral** — max |FY|/FZ, and the slip angle where it occurs
- **cornering_stiffness** — dFY/dα near α ≈ 0, by linear regression over a stated window
- **load_sensitivity** — dμ/dFZ across FZ bins, as a slope with R²
- **camber_sensitivity** — dFY/dIA at peak slip angle
- **peak_sharpness** — fraction of peak FY retained at ±2° from peak slip angle
- **pneumatic_trail** — MZ/FY vs α
- **loaded_radius_vs_fz** — regression of RL against FZ with tolerance

**Critical**: report every metric over the FZ range the car actually runs (roughly 300–1200 N for FSAE), not over TTC's full test range. FSAE loads sit far below passenger-car test loads and tire rankings reorder in that band. The FZ window is an explicit argument with no default.

## Magic Formula 5.2 (Pacejka '02) — FUTURE

### Coefficient structure

The Pacejka Magic Formula: `Y = D · sin(C · arctan(B·x − E·(B·x − arctan(B·x))))`

Where `Y` is force or moment, `x` is slip angle or slip ratio, and:
- **B** = stiffness factor
- **C** = shape factor
- **D** = peak value
- **E** = curvature factor

MF5.2 adds combined slip, camber effects, and turn slip through ~100+ coefficients organised by name prefix:
- `pCx`, `pDx`, `pEx`, `pKx` — longitudinal pure slip
- `pCy`, `pDy`, `pEy`, `pKy` — lateral pure slip
- `rBx`, `rCx`, `rHx` — combined longitudinal
- `rBy`, `rCy`, `rHy` — combined lateral
- `qBz`, `qCz`, `qDz`, `qEz` — self-aligning torque

### Fitting workflow (not yet implemented)

1. Load raw TTC data and apply sign convention correction (done — see above)
2. Separate sweeps: pure lateral (Fx ≈ 0), pure longitudinal (α ≈ 0), combined
3. Fit pure slip coefficients first using `scipy.optimize.least_squares`
4. Fit combined slip coefficients using the pure slip results as fixed
5. Validate: compare measured vs fitted for each sweep, report RMS error
6. Store coefficients in a versioned JSON file with metadata

## Brush model fallback

For early design without tire data, a simplified brush model provides order-of-magnitude estimates:

- Cornering stiffness: `Cα ≈ 3 × Fz` (N/rad, for a typical FSAE slick)
- Peak lateral force: `Fy_peak ≈ μ × Fz` (μ ≈ 1.5 for FSAE slick on dry asphalt)
- Slip angle at peak: `α_peak ≈ Fy_peak / Cα` (typically 5–8°)

**These are estimates only.** Always label results computed from the brush model as `source: "estimate"`.
