---
description: "Tire modelling: MF5.2 / Pacejka coefficients, fitting workflow from TTC data, ISO vs SAE sign conventions, load and camber sensitivity, brush-model fallback. Use when writing tire-related code in vdcore/tire/. IMPORTANT: PUCPR Racing does NOT yet have tire data — say so explicitly when asked."
---

## Current status: NO TIRE DATA

PUCPR Racing does not yet have TTC (Tire Test Consortium) data. TTC acquisition is pending.

**When asked to compute anything that requires tire data:**
1. State clearly: "This requires tire data that the team does not have yet."
2. Raise `NotImplementedError` in code with a message naming the missing input.
3. Never invent placeholder coefficients — they produce plausible-looking but wrong results.

The brush-model fallback (section below) can provide order-of-magnitude estimates for early design, but must be clearly labelled as estimates.

## Magic Formula 5.2 (Pacejka '02)

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

### Load sensitivity

All peak forces scale with normal load Fz, but not linearly — the **load sensitivity** means the friction coefficient decreases at higher loads. This is why lightweight FSAE cars produce more grip per unit load than heavier cars.

### Camber sensitivity

Camber thrust: a cambered tire generates lateral force even at zero slip angle. Coefficient `pDy3` controls the magnitude. Typical: 0.5–1.5 N per degree of camber per kN of load.

## ISO 8855 vs adapted-SAE sign convention

**This is the single most common source of bugs when importing tire data.**

| Quantity | ISO 8855 (this project) | Adapted SAE (TTC raw data) |
|---|---|---|
| Slip angle | Positive = wheel pointed left of travel | Positive = wheel pointed right of travel |
| Lateral force | Positive = leftward | Positive = rightward |
| Self-aligning torque | Positive = counterclockwise (top view) | Positive = clockwise |

**When importing TTC .mat or .csv files:**
1. Check which convention the data uses (stated in the file header or README).
2. If adapted SAE: negate slip angle, lateral force Fy, and self-aligning moment Mz.
3. Validate by checking: positive slip angle should produce a positive (leftward) lateral force in ISO 8855.

## Fitting workflow (for when TTC data arrives)

1. Load raw TTC data (.mat format, typically from FSAE TTC rounds)
2. Apply sign convention correction (see above)
3. Separate sweeps: pure lateral (Fx ≈ 0), pure longitudinal (α ≈ 0), combined
4. Fit pure slip coefficients first using `scipy.optimize.least_squares`
5. Fit combined slip coefficients using the pure slip results as fixed
6. Validate: plot measured vs fitted for each sweep, report RMS error
7. Store coefficients in a versioned JSON file with metadata (tire name, test date, pressure, etc.)

## Brush model fallback

For early design without tire data, a simplified brush model provides order-of-magnitude estimates:

- Cornering stiffness: `Cα ≈ 3 × Fz` (N/rad, for a typical FSAE slick)
- Peak lateral force: `Fy_peak ≈ μ × Fz` (μ ≈ 1.5 for FSAE slick on dry asphalt)
- Slip angle at peak: `α_peak ≈ Fy_peak / Cα` (typically 5–8°)

**These are estimates only.** Always label results computed from the brush model as `source: "estimate"`.
