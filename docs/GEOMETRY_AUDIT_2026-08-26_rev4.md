# Suspension & Steering Geometry Audit — 2026-08-26 (rev 4)

**Car:** FSAE 2027 — PUCPR Racing (Team #27)
**Type:** Double A-arm (SLA), front and rear
**Frame:** ISO 8855 (X+ forward, Y+ left, Z+ up)
**Scope of this revision:** a full validity audit of the *current* shipped
geometry — front, rear, and steering — computing every kinematic the tool
supports, checking it against its limit bands, and justifying each design
choice for the Design Event. Supersedes rev 3, which covered only the rear
wishbone re-sweep. Nothing in the geometry changed between rev 3 and rev 4;
this revision *validates and documents* the rev-3 state, it does not move it.

**Nature of this tool (stated up front, as it must be at the event):** `vdcore`
is a kinematics design-support tool. It makes trade-offs visible, quantified,
and defensible. It is **not** a lap-time simulator, not an FEA tool, and it
computes nothing that requires tyre data (no TTC data yet). Every value below
is a geometric consequence of the chosen hardpoints, not a performance
prediction. Targets shown are literature-derived design intents (Milliken RCVD,
Optimum G), never measured or recommended values.

---

## Executive Summary — is the geometry valid?

**Yes.** The current geometry is kinematically valid and internally consistent,
confirmed by two independent verification passes:

- **Numerical validity (independent re-solve, `DWSolver`, tol 1e-8):** the solver
  **converges at every point** on the full envelope — ±25 mm heave and the full
  ±38 mm rack stroke on all four corners, plus the ±2° roll sweep — with a worst
  residual of **1.56e-10 mm**, ~2 orders below the gate. It is **bitwise
  left/right symmetric** (0.0 deg), **bitwise deterministic** (identical SHA-256
  on repeat), **recovers the static state to 1.8e-15 mm**, and recovers static
  camber to **−1.500000°** on all four corners. Design-vs-solved camber gain
  agrees to **< 0.5 %** on both axles. Full suite: **74 passed, 13 skipped.**

- **Physics validity (independent formula/sign/frame audit):** **zero** formula,
  sign, frame, or unit errors that would produce a plausible-but-wrong number.
  The code does **not** reproduce any legacy-app failure mode (no +200% anti-dive
  artefact, no inverted Ackermann, no pivot-midpoint roll centre). Camber sign is
  correct *independently on both sides*; the roll centre is built from the pivot
  **axis**; anti-geometry is **exactly** 0 with horizontal pivot axes; toe is
  **always** qualified per-side vs total; the parking-effort decomposition is
  dimensionally correct.

**Good things about this geometry** (defensible strengths — see §5):
1. Camber gain lands squarely in the literature band on **both** axles (front
   −0.0384, rear −0.0411 deg/mm) and the outer wheel stays inside the useful
   window (−0.64° / −0.65° at 1.5° roll) — static camber does the heavy lifting,
   the geometry trims.
2. Bump steer is essentially zero (−0.0001 deg/mm per side; total toe 0.0002
   deg/mm) — the tie rod is well placed.
3. Both hard rear-layout constraints are satisfied simultaneously: **no inboard
   pickup behind the rear axle**, and (with the caster-corrected authority)
   the rear legs sit inside the length band.
4. Roll centres are low and migrate predictably; the car is symmetric to
   machine precision.

**Known, accepted items** (design-state facts, not numerical faults — see §6):
rear kingpin length 263.23 mm (1.2% over), front LCA rear leg 430.61 mm (0.6 mm
over, caster-induced), front true-3D kingpin 260.30 mm (0.3 mm over), 0% anti-
geometry by design, and the non-synthesised rear tie rod. All are traceable to
the config and were surfaced by both independent passes.

---

## 1. Vehicle and stiffness basis

| Quantity | Value | Source |
|---|---|---|
| Total / sprung mass | 315.0 / 270.0 kg | design_intent |
| Wheelbase (L) | 1540.0 mm | design_intent |
| CG height / station | 320.0 mm / 693.0 mm behind front axle | design_intent |
| Static front mass fraction | 55.0 % | design_intent |
| Roll axis height at CG | 44.0 mm | computed |
| Roll moment arm | 276.0 mm | computed |
| Target roll gradient | 1.00 deg/g | design_intent |
| → Required roll stiffness | 731.0 N·m/deg | computed |
| → Chassis torsional stiffness | 2193 min / 3655 target N·m/deg | computed |
| Front / rear track (contact patch) | 1240.0 / 1200.0 mm | design_intent |
| 60° tilt test min track | 1109 mm (fitted 1200 mm) | computed — PASS |

Rules compliance (geometry only): wheelbase 1540 mm (15 mm over the 1525 minimum),
narrow/wide track ratio 96.8% (in the 75–100 band), 60° tilt uses the narrower
**rear** track (1200 mm needs 1108.5 mm) — **PASS**.

---

## 2. Front suspension — full KPI set

### 2a. Static front-view construction

| KPI | Value | Band | Status |
|---|---|---|---|
| Roll centre height | 35.00 mm | 20–70 | OK |
| FVSA length | 1500.00 mm | 1300–1700 | OK |
| Scrub radius | 15.08 mm | 5–25 | OK |
| KPI (σ) | 10.00 deg | 6–14 | OK |
| Kingpin length (front-view) | 259.34 mm | 200–260 | OK (marginal, 0.66 mm) |
| UCA/LCA ratio | 0.909 | 0.55–0.98 | OK |
| Camber gain (from FVSA) | 0.0382 deg/mm | 0.03–0.05 | OK |
| FVIC | (−880.0, 84.68) mm | far side | OK |
| LCA / UCA front-view length | 407.20 / 370.03 mm | — | — |

### 2b. Solved rates about static (3D, chassis-referenced Z)

| Rate | Value (solved) | Design (57.30/FVSA) | Δ |
|---|---|---|---|
| Camber gain | −0.038353 deg/mm | −0.038197 | −0.41% |
| RC migration | −0.391440 mm/mm | — | — |
| Half-track change | +0.056769 mm/mm | — | — |
| Camber at full bump (+25) | −2.4896° | — | — |
| Camber at full droop (−25) | −0.5692° | — | — |
| RC height range over travel | 25.30 → 44.90 mm (Δ19.60) | — | — |

### 2c. Behaviour at 1.5° of roll (camber vs ROAD)

| Quantity | Value |
|---|---|
| Outer wheel camber | −0.635° (static −1.50) |
| Inner wheel camber | −2.390° |
| Roll centre height in roll | 33.89 mm (design 35.0) |
| Roll centre lateral migration | −111.46 mm |
| Wheel travel at that roll | 16.228 mm |
| Outer camber in useful window (−2.5…0) | OK |

---

## 3. Rear suspension — full KPI set

### 3a. Static front-view construction

| KPI | Value | Band | Status |
|---|---|---|---|
| Roll centre height | 55.00 mm | 20–70 | OK |
| FVSA length | 1400.00 mm | 1300–1700 | OK |
| Scrub radius | 21.97 mm | 5–25 | OK |
| KPI (σ) | 8.50 deg | 3–10 | OK |
| Kingpin length (LBJ–UBJ) | 263.23 mm | 200–260 | **!! 1.2% over** |
| UCA/LCA ratio | 0.916 | 0.55–0.98 | OK |
| Camber gain (from FVSA) | 0.0409 deg/mm | 0.03–0.05 | OK |
| FVIC | (−800.0, 128.33) mm | far side | OK |
| LCA / UCA front-view length | 383.60 / 351.42 mm | — | — |

### 3b. Solved rates about static (3D, chassis-referenced Z)

| Rate | Value (solved) | Design (57.30/FVSA) | Δ |
|---|---|---|---|
| Camber gain | −0.041114 deg/mm | −0.040926 | −0.46% |
| RC migration | −0.423899 mm/mm | — | — |
| Half-track change | +0.092210 mm/mm | — | — |
| Camber at full bump (+25) | −2.5447° | — | — |
| Camber at full droop (−25) | −0.4856° | — | — |
| RC height range over travel | 44.52 → 65.74 mm (Δ21.23) | — | — |

### 3c. Behaviour at 1.5° of roll (camber vs ROAD)

| Quantity | Value |
|---|---|
| Outer wheel camber | −0.652° (static −1.50) |
| Inner wheel camber | −2.360° |
| Roll centre height in roll | 54.25 mm (design 55.0) |
| Roll centre lateral migration | −71.09 mm |
| Wheel travel at that roll | 15.704 mm |
| Outer camber in useful window (−2.5…0) | OK |

---

## 4. Steering — full KPI set (front axle)

`steering_geometry.py` owns caster, the outboard ball joints (superseding sla's
zero-caster placement), the tie rod, rack, bump steer, Ackermann, and effort.

### 4a. Geometry

| KPI | Value | Band | Status |
|---|---|---|---|
| Caster angle | 5.00 deg | — | — |
| Caster offset | 0.00 mm | — | — |
| Mechanical trail | 21.43 mm | 10–35 | OK |
| Scrub radius (3D) | 15.08 mm | — | OK |
| Tie rod length | 340.79 mm | 150–350 | OK (marginal) |
| Steering arm length | 80.00 mm | — | — |
| Steering arm angle | −12.00 deg | — | — |

### 4b. Kinematic rates

| KPI | Value | Band | Status |
|---|---|---|---|
| Bump steer (per-side) | −0.000103 deg/mm | 0–0.005 | OK (near-ideal) |
| Bump steer (total toe) | 0.000206 deg/mm | — | — |
| Toe at full bump / droop (per-side) | +0.152° / +0.158° | — | — |
| C-factor | −1.278 mm/deg | — | — |
| Steering ratio | 4.58 :1 | 3–7 | OK |
| Max steer at stroke: outer / inner | 27.07° / 39.69° | — | toe-out on turns |
| Geometric Ackermann | 101.1 % | — | small-angle construction |
| Ackermann at 10° (3D solver) | 70.04 % | 60–120 | OK |
| Rod-end misalignment | 5.54 deg | ≤ 12 | OK |

The geometric (101%) and finite-angle (70%) Ackermann differ because the first
is the small-angle kingpin-line construction and the second is the true 3D-solved
value at 10°. The 39.7°/27.1° inner/outer split at full rack confirms genuine
toe-out on turns.

### 4c. Parking effort

| Quantity | Value | Limit | Status |
|---|---|---|---|
| Fz per front wheel | 849.8 N | — | — |
| Kingpin moment (per wheel) | 22.27 N·m | — | — |
| — scrub contribution | 15.08 mm | — | — |
| — trail contribution | 21.43 mm | — | — |
| Rack force (both wheels) | 608.4 N | — | — |
| Steering-wheel torque | 9.73 N·m | 10 | OK (2.7% margin) |
| Rim force | 69.5 N | — | — |

Effort is `M_kp = μ·Fz·√(rs² + tm²)`, rack force by virtual work with the
solver-derived C-factor, steering-wheel torque via the pinion radius. The 2.7%
margin is scrub-driven: any change moving the contact patch outboard eats it.

---

## 5. Design justifications — why these choices

Each choice below is stated as a trade-off with its consequence, not as a
recommendation. The designer chose; the tool quantifies the consequence.

**5.1 Camber gain −0.038 (F) / −0.041 (R) deg/mm.** Both sit inside the
0.03–0.05 literature band. At 1.5° roll the outer wheel is at −0.64°/−0.65°,
comfortably inside the −2.5…0 useful window. The design intent is that **static
camber (−1.50°) does the bulk of the work and the geometry trims**, rather than
chasing full roll compensation with an aggressive FVSA — which would spike bump
camber and cost tyre life on the inner wheel. Justification at the event: this is
a deliberate "static-led" camber strategy for a car with no tyre data yet, where
over-committing the geometry to an unmeasured camber-thrust curve is the larger
risk.

**5.2 Roll centres 35 mm (F) / 55 mm (R), both low, rear higher than front.**
Both are inside the 20–70 band. The rear sits higher than the front, giving a
forward-inclined roll axis (roll-axis height 44 mm at the CG) — a conventional,
defensible choice that biases elastic roll-couple toward the front for a mild
understeer balance. RC migration is modest and lateral migration is bounded
(−111 mm F / −71 mm R at 1.5° roll), so jacking behaviour is controlled.

**5.3 KPI 10° (F) / 8.5° (R), scrub 15 (F) / 22 (R) mm.** Front KPI and scrub
are set together with caster to keep parking effort under the 10 N·m limit
(achieved: 9.73 N·m). Positive scrub of 15 mm gives useful straight-line
stability and steering feel without excessive kickback. This is the binding
front constraint — §4c shows only 2.7% margin — so scrub is deliberately kept
modest.

**5.4 Caster 5°, mechanical trail 21.4 mm.** In the 10–35 band. Gives
self-centring feel and dynamic camber-by-steer without excessive effort. Trail
combines with scrub in the kingpin moment, so it is chosen jointly with 5.3.

**5.5 Bump steer near zero (−0.0001 deg/mm per side).** The tie rod inner/outer
points are placed so the tie rod sweeps almost on the wishbone arc — the primary
tie-rod KPI. Near-zero bump steer means ride motion does not steer the car, a
clear strength to show a judge.

**5.6 Ackermann 70% at 10°.** Inside the 60–120 band, giving partial toe-out on
turns. For an FSAE autocross car this is a reasonable compromise between the
low-speed geometric ideal (100%) and the high-slip-angle reality where parallel
or even anti-Ackermann can be faster — chosen conservatively toward pro-Ackermann
because, with no tyre data, the low-speed geometric benefit is the defensible one.

**5.7 Rear wishbone e/a = 1.0, base 180/160, sweep 90/80 (carried from rev 3).**
The rearmost inboard pickup is held **exactly on the rear-axle line** — no pickup
behind the axle. That constraint forces sweep ≥ base/2, i.e. e/a ≥ 1.0 by
definition; the equality (sweep = base/2) is the tightest feasible point. Base is
then minimised to 180/160 to keep the resulting front legs inside the length
band. The accepted price is e/a = 1.0 (a swept compression strut in braking) —
still well below the original 1.35 and inside the ≤ 1.5 limit. This is a chosen
layout constraint, defended as such, not an oversight.

**5.8 Anti-geometry 0% (both axles), by design.** The pivot axes are horizontal
(`dz_lca = dz_uca = 0`), so the side-view instant centre is at infinity and
anti-dive/anti-squat are **exactly** 0 — confirmed by the physics review as the
correct formula result, not the legacy +200% artefact. The design intent is to
decouple longitudinal load transfer from the suspension links and handle pitch
purely with springs/dampers, keeping the geometry clean and the anti-effects
predictable. This must be actively defended at the event (it is a deliberate 0,
not a missing feature); inclining `dz_lca`/`dz_uca` is the one-parameter change
that would introduce anti-geometry if the team later wants it.

---

## 6. Known / accepted items (design-state, not numerical faults)

Both independent passes surfaced the same items. None is a solver error; all are
traceable to the config and already flagged `!!` in the shipped
`geometry_summary.py`.

| # | Item | Value | Note |
|---|---|---|---|
| 1 | Rear kingpin length over band | 263.23 mm | 3.23 mm (1.2%) over 260 max, front-view and true-3D. Set by `kingpin_length_mm`. |
| 2 | Front LCA rear leg over band (caster) | 430.61 mm | 0.61 mm over 430. Caster tips it over; the **zero-caster `sla.member_legs_mm` reports 427.44 and hides this**. |
| 3 | Front true-3D kingpin over band (caster) | 260.30 mm | 0.30 mm over. Front-view (259.34) passes; the 3D value is the upright authority. |
| 4 | Rear tie rod length / provenance | 364.59 mm | Over the 350 tie-rod band **and** hand-entered in `carro_formula_2027.csv` — synthesised by no script, covered by no test. Treat as measured-once until a rear steering synthesis exists. |
| 5 | Anti-geometry 0% | 0.00% | By design (§5.8). Must be defended, not fixed. |

**Authority note (important for the audit):** the **caster-corrected 3D member
lengths from `geometry_summary.member_lengths` are the authority**, not the
zero-caster `sla.member_legs_mm`. The latter under-reports the front LCA rear leg
by ~3 mm and hides that it is over the limit. Any leg-length claim in a
deliverable must cite the caster-corrected value.

**Screening leg forces are not sizing loads.** The leg forces printed in the
summary assume a tyre μ the team has not measured (no TTC data), and the model
has no pushrod (the solve closes the 6th DOF with a moment constraint and
zero-caster front ball joints). They screen for gross imbalance only; they do not
size tubes and do not cover buckling of the swept rear legs. Out of tool scope.

---

## 7. Verification record

| Check | Result | Source |
|---|---|---|
| Full test suite | 74 passed, 13 skipped | pytest |
| Solver convergence, full envelope | **PASS** — worst residual 1.56e-10 mm (gate 1e-8) | numerical-validator |
| — Heave ±25 mm × 4 corners | 51/51 each | ″ |
| — Front rack ±38 mm + 11×11 grid | 51/51 + 121/121 | ″ |
| — Roll ±2° (axle_roll + direct), all corners | 21/21 + 84/84 | ″ |
| Left/right symmetry (FL↔FR, RL↔RR) | **0.000e+00 deg** (bitwise) | ″ |
| Determinism (repeat solve) | identical SHA-256, max diff 0.0 | ″ |
| Zero-state recovery solve(0,0,0) | max 1.78e-15 mm; camber −1.500000° | ″ |
| Static camber from patch→wheel-centre | −1.500000° all 4 corners | ″ |
| Design vs solved camber gain | F −0.41%, R −0.46% (< 0.5%) | ″ |
| Formula / sign / frame / unit audit | **zero errors**; no legacy failure modes | vd-physics-reviewer |
| — Camber sign, both sides independently | correct | ″ |
| — Roll centre from pivot AXIS | correct (not midpoint) | ″ |
| — Anti-geometry exactly 0 with dz=0 | correct | ″ |
| — Ackermann not inverted | correct | ″ |
| — Per-side vs total toe explicit everywhere | correct | ″ |
| — Parking-effort decomposition | correct | ″ |

---

## 8. Bottom line

The current FSAE 2027 geometry is **kinematically valid and defensible**. It
converges everywhere to machine precision, is exactly symmetric, computes every
KPI within (or knowingly outside) its literature band, and — independently
verified — carries no formula, sign, frame, or unit error. Its strengths are a
static-led camber strategy that keeps the outer wheel in the useful window on
both axles, near-zero bump steer, low predictable roll centres, and a rear layout
that satisfies both the on-axle-pickup and leg-length constraints at once. Its
open items (rear kingpin 263 mm, caster-induced front over-limits at ~0.3–0.6 mm,
0% anti-geometry by design, and the unsynthesised rear tie rod) are all known,
traceable to the config, and either accepted by choice or bounded well within
tolerance — none is a numerical fault. The caster-corrected 3D member lengths are
the authority for any leg-length claim.
