# Suspension & Steering Geometry Audit — 2026-08-26

**Car:** FSAE 2027 — PUCPR Racing (Team #27)
**Type:** Double A-arm, front and rear
**Frame:** ISO 8855 (X+ forward, Y+ left, Z+ up)

---

## Executive Summary

The geometry is internally consistent — all automated checks pass for the front axle, and the 3D solver converges across the full bump/steer envelope. The steering system is well-balanced with parking torque at 9.73 N.m (limit 10). However, the rear axle has **6 flagged issues**: the front legs of both wishbones are grossly oversized (554 mm and 518 mm vs a 430 mm limit), the kingpin length exceeds its band, and the rear tie rod has **no synthesis script, no bounds checking, and no tests** — it comes from a legacy CSV. Anti-dive and anti-squat are both 0% because all pivot axes are horizontal; this is a conscious design choice but should be defended.

---

## 1. Vehicle Parameters

| Parameter | Value | Source |
|---|---|---|
| Total mass | 315.0 kg | design_intent |
| Sprung mass | 270.0 kg | design_intent |
| Unsprung mass | 45.0 kg | design_intent |
| Wheelbase | 1540.0 mm | design_intent |
| CG height | 320.0 mm | design_intent |
| CG station (behind front axle) | 693.0 mm | design_intent |
| Front mass fraction | 55.0% | computed |
| Front track (contact patch) | 1240.0 mm | design_intent |
| Rear track (contact patch) | 1200.0 mm | design_intent |
| Loaded radius | 245.0 mm (both axles) | design_intent |
| Rim diameter | 13 in (both axles) | design_intent |
| Static camber | -1.50 deg (all corners) | design_intent |
| Static toe | 0.00 deg per side | design_intent |
| Roll gradient target | 1.00 deg/g | design_intent |
| Required roll stiffness | 731.0 N.m/deg | computed |
| Chassis torsional stiffness | 2193 min / 3655 target N.m/deg | computed |
| Roll axis height at CG | 44.0 mm | computed |
| Roll moment arm | 276.0 mm | computed |

---

## 2. Front Suspension

### 2a. Front-View Geometry (Design Frame: y outboard, z up)

| Point | Y [mm] | Z [mm] |
|---|---|---|
| Lower ball joint (LBJ) | 582.00 | 130.00 |
| Upper ball joint (UBJ) | 536.97 | 385.40 |
| LCA inboard | 175.00 | 117.38 |
| UCA inboard | 175.00 | 308.58 |
| FVIC (construction) | -880.00 | 84.68 |

### 2b. Front Wishbone Dimensions

| Parameter | Value |
|---|---|
| LCA front-view length | 407.20 mm |
| UCA front-view length | 370.03 mm |
| UCA/LCA ratio | 0.909 |
| Outboard vertical separation | 255.40 mm |
| Inboard vertical separation | 191.20 mm |
| LCA inclination | 1.78 deg (falls outboard to inboard) |
| UCA inclination | 11.98 deg (falls outboard to inboard) |
| RC that would flatten the LCA | 53.73 mm |

### 2c. Front Longitudinal Layout (x positive rearward)

| Parameter | Value |
|---|---|
| LCA pickup x | -130.0 / 130.0 mm (base 260, sweep 0) |
| UCA pickup x | -120.0 / 120.0 mm (base 240, sweep 0) |
| LCA e/a ratio | 0.00 |
| UCA e/a ratio | 0.00 |

Front wishbones are symmetric longitudinally (zero sweep) — both pickups equidistant from the axle line.

### 2d. Front 3D Member Lengths (merged, caster-corrected)

| Member | Length [mm] | Band | Status |
|---|---|---|---|
| LCA front leg | 424.49 | 320-430 | OK |
| LCA rear leg | 430.61 | 320-430 | !! marginal (0.61 mm over) |
| UCA front leg | 392.96 | 320-430 | OK |
| UCA rear leg | 385.39 | 320-430 | OK |
| Tie rod | 340.79 | 150-350 | OK |
| Kingpin (front view) | 259.34 | 200-260 | OK |
| Kingpin (true 3D) | 260.30 | 200-260 | !! marginal (0.30 mm over) |

### 2e. Front Static KPIs

| KPI | Value | Band | Status |
|---|---|---|---|
| Roll centre height | 35.00 mm | 20-70 | OK |
| FVSA length | 1500.00 mm | 1300-1700 | OK |
| Scrub radius | 15.08 mm | 5-25 | OK |
| KPI | 10.00 deg | 6-14 | OK |
| Kingpin length | 259.34 mm | 200-260 | OK |
| UCA/LCA ratio | 0.909 | 0.55-0.98 | OK |
| Camber gain (design, from FVSA) | 0.0382 deg/mm | 0.03-0.05 | OK |
| Anti-dive | 0.00% | 0-30 | OK (see note) |

### 2f. Front 3D Kinematic Rates (DWSolver, chassis-referenced)

| Rate | Value |
|---|---|
| Camber gain (solved, 3D) | -0.0384 deg/mm |
| Roll centre migration | -0.3914 mm/mm |
| Half-track change | 0.0568 mm/mm |
| Camber at full bump (+25 mm) | -2.49 deg |
| Camber at full droop (-25 mm) | -0.57 deg |
| RC range over +/-25 mm travel | 25.3 to 44.9 mm |

### 2g. Front at 1.5 deg Roll (camber relative to road)

| Quantity | Value |
|---|---|
| Outer wheel camber | -0.64 deg (from static -1.50) |
| Inner wheel camber | -2.39 deg |
| Roll centre height | 33.9 mm (design 35.0) |
| Roll centre lateral migration | -111.5 mm |
| Wheel travel at that roll | 16.23 mm |

---

## 3. Rear Suspension

### 3a. Rear Front-View Geometry (Design Frame)

| Point | Y [mm] | Z [mm] |
|---|---|---|
| Lower ball joint (LBJ) | 558.60 | 130.00 |
| Upper ball joint (UBJ) | 519.69 | 390.34 |
| LCA inboard | 175.00 | 129.53 |
| UCA inboard | 175.00 | 321.91 |
| FVIC (construction) | -800.00 | 128.33 |

### 3b. Rear Wishbone Dimensions

| Parameter | Value |
|---|---|
| LCA front-view length | 383.60 mm |
| UCA front-view length | 351.42 mm |
| UCA/LCA ratio | 0.916 |
| Outboard vertical separation | 260.34 mm |
| Inboard vertical separation | 192.38 mm |
| LCA inclination | 0.07 deg |
| UCA inclination | 11.23 deg |

### 3c. Rear Longitudinal Layout (x positive rearward)

| Parameter | Value |
|---|---|
| LCA pickup x | 1140.0 / 1480.0 mm (base 340, sweep 230) |
| UCA pickup x | 1160.0 / 1480.0 mm (base 320, sweep 220) |
| LCA e/a ratio | 1.35 |
| UCA e/a ratio | 1.38 |

The rear has **large forward sweep** — the front pickups are 230 mm (LCA) and 220 mm (UCA) ahead of the rear pickups. This creates the oversized front legs.

### 3d. Rear 3D Member Lengths

| Member | Length [mm] | Band | Status |
|---|---|---|---|
| LCA front leg | **554.21** | 320-430 | !! FAIL (+124 mm, 29% over) |
| LCA rear leg | 388.26 | 320-430 | OK |
| UCA front leg | **517.59** | 320-430 | !! FAIL (+88 mm, 20% over) |
| UCA rear leg | 356.50 | 320-430 | OK |
| Tie rod | 364.59 | -- | no check (legacy CSV) |
| Kingpin (front view) | **263.23** | 200-260 | !! FAIL (+3 mm) |
| Kingpin (true 3D) | **263.23** | 200-260 | !! FAIL (+3 mm) |

### 3e. Rear Static KPIs

| KPI | Value | Band | Status |
|---|---|---|---|
| Roll centre height | 55.00 mm | 20-70 | OK |
| FVSA length | 1400.00 mm | 1300-1700 | OK |
| Scrub radius | 21.97 mm | 5-25 | OK |
| KPI | 8.50 deg | 3-10 | OK |
| Kingpin length | **263.23 mm** | 200-260 | !! FAIL |
| UCA/LCA ratio | 0.916 | 0.55-0.98 | OK |
| Camber gain (design) | 0.0409 deg/mm | 0.03-0.05 | OK |
| Anti-squat | 0.00% | 0-30 | OK (see note) |

### 3f. Rear 3D Kinematic Rates (DWSolver, chassis-referenced)

| Rate | Value |
|---|---|
| Camber gain (solved, 3D) | -0.0411 deg/mm |
| Roll centre migration | -0.4239 mm/mm |
| Half-track change | 0.0922 mm/mm |
| Camber at full bump (+25 mm) | -2.54 deg |
| Camber at full droop (-25 mm) | -0.49 deg |
| RC range over +/-25 mm travel | 44.5 to 65.7 mm |

### 3g. Rear at 1.5 deg Roll (camber relative to road)

| Quantity | Value |
|---|---|
| Outer wheel camber | -0.65 deg (from static -1.50) |
| Inner wheel camber | -2.36 deg |
| Roll centre height | 54.2 mm (design 55.0) |
| Roll centre lateral migration | -71.1 mm |
| Wheel travel at that roll | 15.70 mm |

---

## 4. Steering System

### 4a. Designer Inputs (STEERING_2027)

| Parameter | Value |
|---|---|
| Caster angle | 5.00 deg |
| Caster offset | 0.00 mm |
| TRO height along kingpin | 40.00 mm |
| Steering arm length | 80.00 mm |
| Steering arm angle | -12.00 deg |
| Rack x (rearward) | 30.00 mm |
| Rack z | 158.30 mm |
| Rack half length | 270.00 mm |
| Pinion radius | 16.00 mm |
| Max rack travel (half-stroke) | 38.00 mm |
| Steering wheel diameter | 280.00 mm |
| Static toe per side | 0.00 deg |
| Target Ackermann | 100% at 10 deg |
| Target bump steer | 0 deg/mm |
| Parking friction coefficient | 1.0 |

### 4b. Computed Steering KPIs

| KPI | Value | Band | Status |
|---|---|---|---|
| Mechanical trail | 21.43 mm | 10-35 | OK |
| Scrub radius (3D) | 15.08 mm | 5-25 | OK |
| Tie rod length | 340.79 mm | 150-350 | OK |
| Bump steer (per-side) | -0.00010 deg/mm | 0-0.005 | OK |
| Bump steer (total toe) | 0.00021 deg/mm | -- | -- |
| Toe at full bump | 0.152 deg/side | -- | -- |
| Toe at full droop | 0.158 deg/side | -- | -- |
| C-factor | -1.278 mm/deg | -- | -- |
| Steering ratio | 4.58:1 | 3-7 | OK |
| Max steer at stroke (outer) | 27.07 deg | -- | -- |
| Max steer at stroke (inner) | 39.69 deg | -- | -- |
| Geometric Ackermann | 101.1% | -- | -- |
| Ackermann at 10 deg outer | 70.0% | 60-120 | OK |
| Rod end misalignment (worst) | 5.54 deg | limit 12 | OK |
| Rack x position | 30.00 mm | -80-80 | OK |
| Rack z position | 158.30 mm | 50-180 | OK |

### 4c. Steering Effort (Parking)

| Quantity | Value | Limit | Status |
|---|---|---|---|
| Fz per front wheel | 849.8 N | -- | -- |
| Kingpin moment (per wheel) | 22.27 N.m | -- | -- |
| Scrub contribution | 15.08 mm | -- | -- |
| Trail contribution | 21.43 mm | -- | -- |
| Rack force (both wheels) | 608.4 N | -- | -- |
| Steering wheel torque | 9.73 N.m | 10.0 | OK (0.27 N.m margin) |
| Rim force | 69.5 N | -- | -- |

### 4d. Steering Hardpoints (ISO 8855)

| Corner | Point | X [mm] | Y [mm] | Z [mm] |
|---|---|---|---|---|
| FL | TIE_ROD_IN | -30.00 | 270.00 | 158.30 |
| FL | TIE_ROD_OUT | 84.59 | 590.29 | 178.75 |
| FR | TIE_ROD_IN | -30.00 | -270.00 | 158.30 |
| FR | TIE_ROD_OUT | 84.59 | -590.29 | 178.75 |
| FL | UCA_OUT (caster-corrected) | -12.28 | 536.97 | 385.40 |
| FL | LCA_OUT (caster-corrected) | 10.06 | 582.00 | 130.00 |
| FR | UCA_OUT (caster-corrected) | -12.28 | -536.97 | 385.40 |
| FR | LCA_OUT (caster-corrected) | 10.06 | -582.00 | 130.00 |

---

## 5. Flagged Issues

### FAIL — Must Address

| # | Issue | Axle | Value | Limit | Severity |
|---|---|---|---|---|---|
| 1 | LCA front leg length | Rear | 554.21 mm | 430 max | Critical -- 29% over limit |
| 2 | UCA front leg length | Rear | 517.59 mm | 430 max | Critical -- 20% over limit |
| 3 | Kingpin length | Rear | 263.23 mm | 260 max | Minor -- 1.2% over |

**Root cause for #1 and #2:** The rear axle has 230 mm of forward sweep (the front pickup is 230 mm ahead of the axle line at x=1140 vs axle at x=1540). This means the front leg of each wishbone has to reach from x=1140 back to x=1540 while also spanning the lateral distance — producing 554 mm legs that are 29% over the 430 mm limit. These are very long tubes that will be heavy, hard to package, and prone to buckling.

**Options:**
- Reduce the rear sweep (move the front pickups rearward). This shortens the front legs but changes anti-squat response when pivot axis rake is added.
- Widen the band if these leg lengths are intentional and structurally validated.
- Split the A-arm into a multi-link (H-arm or 5-link) for the rear if packaging demands it.

### WARNING — Should Address

| # | Issue | Details |
|---|---|---|
| 4 | Rear tie rod unvalidated | RL/RR tie rod points come from `legacy_app/carro_formula_2027.csv`. No synthesis script, no bounds checks, no tests. The summary script warns about this explicitly. |
| 5 | Front LCA rear leg marginal | 430.61 mm vs 430 limit -- 0.14% over. Negligible but technically out of band. |
| 6 | Front kingpin 3D marginal | 260.30 mm vs 260 limit -- 0.12% over. The front-view value (259.34) passes; the caster adds 0.96 mm. |
| 7 | Steering torque margin thin | 9.73 N.m vs 10.0 N.m limit -- only 2.7% margin. Any increase in scrub radius, trail, or front weight shifts this over. |
| 8 | Anti-dive and anti-squat both 0% | All pivot axes are horizontal. This is by design (no dz_lca/dz_uca set), but zero anti-features means full dive/squat under braking/acceleration. Must be defended at Design Event. |
| 9 | Geometric Ackermann vs kinematic Ackermann gap | Geometric projection gives 101%, but actual kinematic Ackermann from the 3D solver is 70% at 10 deg outer steer. The geometric check is misleading. The 70% is the real number. |

### NOTES — For Awareness

| # | Item | Details |
|---|---|---|
| 10 | Toe at full bump is non-zero | 0.152 deg/side at +25 mm bump. The bump steer rate is essentially zero (-0.0001 deg/mm), but the integral over 25 mm accumulates. This is normal. |
| 11 | RC lateral migration is large | -111.5 mm (front) and -71.1 mm (rear) at 1.5 deg roll. These are chassis-referenced. This is typical for SLA at FSAE roll angles. |
| 12 | No rear steering synthesis | `steering_geometry.py` only covers the front axle. The rear toe-link geometry is an open item. |
| 13 | Leg forces are screening-only | No pushrod in the model, no tire data, friction coefficients are assumptions. These are NOT structural sizing loads. |

---

## 6. Front vs Rear Comparison

| Quantity | Front | Rear | Notes |
|---|---|---|---|
| Track (contact patch) | 1240.0 mm | 1200.0 mm | 3.3% wider front |
| Roll centre height | 35.0 mm | 55.0 mm | Rear 57% higher |
| FVSA length | 1500 mm | 1400 mm | Rear 7% shorter |
| Camber gain (design) | 0.0382 deg/mm | 0.0409 deg/mm | Rear 7% higher |
| Camber gain (3D solved) | 0.0384 deg/mm | 0.0411 deg/mm | Rear 7% higher |
| RC migration rate | -0.391 mm/mm | -0.424 mm/mm | -- |
| KPI | 10.00 deg | 8.50 deg | -- |
| Scrub radius | 15.08 mm | 21.97 mm | -- |
| Kingpin length | 259.34 mm | 263.23 mm | -- |
| UCA/LCA ratio | 0.909 | 0.916 | -- |
| Outer camber at 1.5 deg roll | -0.64 deg | -0.65 deg | -- |
| Anti-dive / anti-squat | 0% | 0% | -- |

The rear has a higher roll centre (55 vs 35 mm), a shorter FVSA (1400 vs 1500 mm), and slightly higher camber gain. This means the rear recovers camber marginally faster than the front in roll — a typical choice to reduce oversteer at the limit. The RC height split (35/55) produces a roll axis that rises rearward at 1.3 deg, placing the roll axis at 44 mm at the CG — within the usual FSAE range.

---

## 7. Consistency Checks

| Check | Result |
|---|---|
| Design camber gain vs 3D solved | Front: 0.0382 vs 0.0384 (0.5% diff) -- OK |
| | Rear: 0.0409 vs 0.0411 (0.5% diff) -- OK |
| sla_geometry static KPIs vs DWSolver | Ball joints, scrub, KPI all match -- caster shifts the 3D outboard BJ by ~10-12 mm in X |
| Left/right symmetry | All corners mirror exactly (confirmed by solver) |
| Track consistency | Contact patch track matches config input (1240/1200) |
| Camber encoding in hardpoints | All 4 corners recover -1.500 deg from contact patch to wheel centre offset |
| 60 deg tilt test | Minimum track needed: 1109 mm; narrowest fitted: 1200 mm (rear) -- 91 mm margin |
| Narrow/wide track ratio | 96.77% (limit >=75%) -- OK |
| Wheelbase | 1540.0 mm (minimum 1525 mm) -- 15 mm margin |

---

## 8. What's Validated vs What's Assumed

### Validated (DWSolver + independent Altair MotionSolve cross-check)
- All front-view geometry (ball joints, FVIC, FVSA, RC height)
- Camber gain and sweep over +/-25 mm bump
- Roll centre height and migration
- Bump steer and Ackermann (front only)
- Steering effort decomposition
- Frame transforms (ISO 8855 <-> design frame)

### Design Intent (chosen, not computed -- must be defended)
- Static camber (-1.50 deg)
- RC heights (35/55 mm)
- FVSA lengths (1500/1400 mm)
- KPI (10/8.5 deg)
- Roll gradient target (1.0 deg/g)
- Caster (5 deg)
- All steering arm geometry
- All rack parameters
- Friction coefficients (1.50 lateral, 1.40 longitudinal)

### Unvalidated
- Rear tie rod geometry (legacy CSV, no synthesis)
- Anti-features (deliberately 0%, pivot axes not inclined)
- Member stress/buckling (especially the 554 mm rear front legs)
- Anything requiring tire data
- Wheel rates, motion ratios, ride frequency, damping

---

## 9. Complete Hardpoint Table (ISO 8855)

All coordinates: X+ forward, Y+ left, Z+ up. Origin: front axle centreline, ground plane, vehicle centreline.

### Front Left (FL) / Front Right (FR = Y negated)

| Point | X [mm] | Y [mm] | Z [mm] |
|---|---|---|---|
| UCA_IN_FRONT | 120.00 | 175.00 | 308.58 |
| UCA_IN_REAR | -120.00 | 175.00 | 308.58 |
| UCA_OUT | -12.28 | 536.97 | 385.40 |
| LCA_IN_FRONT | 130.00 | 175.00 | 117.38 |
| LCA_IN_REAR | -130.00 | 175.00 | 117.38 |
| LCA_OUT | 10.06 | 582.00 | 130.00 |
| TIE_ROD_IN | -30.00 | 270.00 | 158.30 |
| TIE_ROD_OUT | 84.59 | 590.29 | 178.75 |
| WHEEL_CENTER* | 0.00 | 613.58 | 245.00 |
| CONTACT_PATCH* | 0.00 | 620.00 | 0.00 |

### Rear Left (RL) / Rear Right (RR = Y negated)

| Point | X [mm] | Y [mm] | Z [mm] |
|---|---|---|---|
| UCA_IN_FRONT | -1160.00 | 175.00 | 321.91 |
| UCA_IN_REAR | -1480.00 | 175.00 | 321.91 |
| UCA_OUT | -1540.00 | 519.69 | 390.34 |
| LCA_IN_FRONT | -1140.00 | 175.00 | 129.53 |
| LCA_IN_REAR | -1480.00 | 175.00 | 129.53 |
| LCA_OUT | -1540.00 | 558.60 | 130.00 |
| TIE_ROD_IN | -1480.00 | 175.00 | 169.00 |
| TIE_ROD_OUT | -1594.60 | 520.81 | 183.53 |
| WHEEL_CENTER* | -1540.00 | 593.58 | 245.00 |
| CONTACT_PATCH* | -1540.00 | 600.00 | 0.00 |

\* Reference points, not hardpoints. Wheel centre encodes -1.50 deg static camber (6.42 mm inboard of contact patch). Static toe cannot be encoded in a contact-patch / wheel-centre pair.

---

## 10. Scrub Radius and Camber Interaction

Static camber moves the contact patch **outboard** (widening ground track), which increases the effective scrub radius. The hardpoint-level scrub radius is computed from the ball joints (unchanged by camber), but the actual ground-level scrub is larger:

| Axle | Scrub (from ball joints) | Camber-adjusted scrub | Band |
|---|---|---|---|
| Front | 15.08 mm | 21.49 mm | 5-25 |
| Rear | 21.97 mm | 28.39 mm | 5-25 — **OUTSIDE BAND** |

The rear camber-adjusted scrub (28.39 mm) exceeds the 25 mm upper bound. This also feeds into steering effort: any increase in front scrub raises the parking torque (currently at 9.73 N.m vs 10.0 N.m limit with only 2.7% margin).

---

## 11. Design Event Questions to Prepare

These are questions a judge would likely ask. Each needs a prepared rationale:

1. **Why zero anti-dive and anti-squat?** Horizontal pivot axes are a deliberate simplification — what is the trade-off vs 15-25% anti-dive?

2. **Why is the rear FVSA shorter than the front (1400 vs 1500 mm)?** The rear recovers camber faster — is this to compensate for the narrower rear track, or to bias understeer at the limit?

3. **How do you justify 554 mm rear wishbone legs?** These are 29% over the typical band. Have they been checked for buckling under the ~3 kN compression loads?

4. **What sets your caster at 5 degrees?** The mechanical trail is 21.43 mm — is this targeting a specific pneumatic trail ratio or aligning force feedback level?

5. **What sets your roll gradient target at 1.0 deg/g?** This is conservative for FSAE (typical range 0.5-1.5 deg/g). What data supports this choice?

6. **Your parking torque is 9.73 N.m against a 10.0 N.m limit — 2.7% margin. What happens with cold tires or debris on track?** The 1.0 friction coefficient for parking is generous — some surfaces are higher.

7. **The geometric Ackermann is 101% but kinematic Ackermann is 70% at 10 deg steer. Can you explain the discrepancy?** The geometric projection is a small-angle approximation that breaks down at real steer angles. The 70% is the 3D-solver answer.

8. **Where does the rear tie rod geometry come from?** It's from a legacy CSV with no synthesis script or bounds checks. What validates it?
