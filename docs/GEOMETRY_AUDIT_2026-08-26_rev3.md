# Suspension & Steering Geometry Audit — 2026-08-26 (rev 3)

**Car:** FSAE 2027 — PUCPR Racing (Team #27)
**Type:** Double A-arm, front and rear
**Frame:** ISO 8855 (X+ forward, Y+ left, Z+ up)
**Change under audit:** rear wishbone re-swept to hold the rearmost inboard
pickup on the axle line. `REAR_2027`: `lca_base` 340→180, `lca_sweep` 0→90,
`uca_base` 320→160, `uca_sweep` 0→80.
**Supersedes:** rev 2 for the rear axle. Front axle and steering unchanged from rev 1.

---

## Executive Summary

Rev 2 removed the oversized rear legs by dropping sweep to zero, but that pushed
the rearmost inboard pickup **170 mm behind the rear axle** — an unacceptable
chassis layout. Rev 3 resolves that: the rear arms are re-swept so the **rearmost
pickup lands exactly on the rear axle line (x = 1540)**, and the base is shrunk to
**180 / 160 mm** so the now-longer front legs still fit under the 430 mm limit.

The result satisfies both hard constraints simultaneously:
- **No inboard pickup sits behind the rear axle** (rearmost pickup clearance = 0 mm).
- **All four rear legs are inside the 320–430 band** (423.73 / 386.13 / 383.60 / 351.42 mm).

The unavoidable price, stated plainly: **e/a = 1.00 on both arms.** Holding a
pickup no further back than the axle *forces* sweep ≥ base/2, which is e/a ≥ 1.0
by definition — you cannot have a symmetric (e/a < 1) arm and keep the pickup off
the rear of the axle. So the front leg is again a swept compression strut in
braking, but at 423.7 mm and e/a 1.0 it is far milder than the original 554 mm /
e/a 1.35, and it is now the deliberate consequence of a chosen layout constraint
rather than an unmanaged side effect.

As with every sweep/base change, **nothing kinematic moved.** Camber gain, roll
centre, RC migration, scrub, and anti-squat are numerically identical to rev 1/2.

---

## 1. The constraint and why e/a = 1.0 is forced

The inboard pickups sit at `mid ± base/2` where `mid = axle_x − sweep`. The
rearmost pickup is at `axle_x − sweep + base/2`. Requiring it not to go behind
the axle (`≤ axle_x`) gives:

$$\text{axle\_x} - \text{sweep} + \tfrac{\text{base}}{2} \le \text{axle\_x}
\iff \text{sweep} \ge \tfrac{\text{base}}{2} \iff \text{e/a} = \tfrac{2\,\text{sweep}}{\text{base}} \ge 1.0$$

The tightest feasible point is the equality: **sweep = base/2**, which puts the
rearmost pickup exactly on the axle at e/a = 1.0. With e/a pinned, **base is the
only free knob**, and it trades leg length against structural robustness.

## 2. Base selection (with e/a = 1.0)

Computed leg lengths at sweep = base/2, rearmost pickup on the axle:

| LCA/UCA base | LCA front leg | UCA front leg | Legs < 430? |
|---|---|---|---|
| 340 / 320 | 512.6 | 475.3 | ❌ both over |
| 260 / 240 | 463.4 | 425.6 | ❌ LCA over |
| 200 / 180 | 432.6 | 394.8 | ❌ LCA over (2.6 mm) |
| **180 / 160** | **423.7** | **386.1** | ✅ chosen — max base that fits |
| 160 / 140 | 415.6 | 378.3 | ✅ |
| 140 / 120 | 408.3 | 371.3 | ✅ |

**180 / 160 was chosen** — the widest base that keeps both front legs under the
limit, so it retains the most longitudinal stiffness and the lowest per-pickup
load of any feasible option. Leg margin is thin on the binding LCA front leg
(6.3 mm), which is the accepted cost of maximising base.

**Structural caveat (out of tool scope):** a narrower base reacts fore/aft
(braking) loads on a shorter moment arm, so pickup reaction loads rise roughly as
1/base and the inboard mount is less stiff. `vdcore` computes kinematics only —
it does **not** score stiffness, member loads, or buckling (CLAUDE.md scope). The
screening leg forces below are explicitly *not* sizing loads. The base choice
trades a structural quantity this tool cannot measure; 180/160 is the
kinematically-feasible option that gives that quantity the most headroom.

---

## 3. Final rear geometry — before / after

### 3a. Member lengths (true 3D)

| Member | rev 1 (sweep 230/220) | rev 2 (sweep 0) | **rev 3 (base 180/160, e/a 1.0)** | Band |
|---|---|---|---|---|
| LCA front leg | 554.21 ❌ | 419.58 | **423.73 ✓** | 320–430 |
| LCA rear leg | 388.26 | 419.58 | **383.60 ✓** | 320–430 |
| UCA front leg | 517.59 ❌ | 386.13 | **386.13 ✓** | 320–430 |
| UCA rear leg | 356.50 | 386.13 | **351.42 ✓** | 320–430 |

Front and rear legs differ again (the arm is swept), but every leg is in band.

### 3b. Pickup positions (design frame, x positive rearward; axle at 1540)

| Pickup | rev 2 (sweep 0) | **rev 3** | Clearance to axle |
|---|---|---|---|
| LCA front | 1370 | 1360 | +180 mm ahead |
| LCA rear | 1710 (170 behind ❌) | **1540** | **0 mm — on the axle ✓** |
| UCA front | 1380 | 1380 | +160 mm ahead |
| UCA rear | 1700 (160 behind ❌) | **1540** | **0 mm — on the axle ✓** |

The behind-axle pickup (rev 2 issue N1) is **resolved**: both rearmost pickups
now sit exactly on the axle line, none behind it.

### 3c. e/a ratio

| Arm | rev 1 | rev 2 | rev 3 | Target |
|---|---|---|---|---|
| LCA | 1.353 | 0.000 | **1.000** | ≤ 1.5 |
| UCA | 1.375 | 0.000 | **1.000** | ≤ 1.5 |

e/a rose from 0 back to 1.0 — the unavoidable cost of the on-axle constraint.
Still inside the ≤ 1.5 limit, and well below the original 1.35/1.38.

---

## 4. What did NOT change (kinematic invariants)

Front-view KPIs are **byte-identical** across rev 1 / rev 2 / rev 3 — verified by
`solve_axle` direct compare:

| Invariant | All revisions |
|---|---|
| LCA / UCA front-view length | 383.60 / 351.42 mm |
| Scrub radius | 21.97 mm |
| RC that flattens the LCA | 55.71 mm |
| Roll centre height | 55.00 mm |
| FVSA length | 1400.00 mm |
| Camber gain (design / 3D solved) | 0.0409 / −0.0411 deg/mm |
| RC migration | −0.4239 mm/mm |
| Half-track change | 0.0922 mm/mm |
| Anti-squat | 0.00% |

Sweep and base move pickups along X only; with horizontal pivot axes (`dz = 0`)
the front-view Y–Z kinematics are untouched and anti-squat stays 0%.

---

## 5. Flagged issues (rear axle)

### CLEARED

| Issue | Status |
|---|---|
| LCA front leg over limit (rev 1) | 554→**423.73 mm — CLEARED** |
| UCA front leg over limit (rev 1) | 518→**386.13 mm — CLEARED** |
| Rearmost pickup behind axle (rev 2 N1) | **CLEARED — now on the axle line** |
| Stale docstring (rev 2 N3) | **FIXED** (`member_legs_mm`, no longer says "170 mm") |

### STILL OPEN (unchanged by this edit)

| # | Issue | Value | Note |
|---|---|---|---|
| 3 | Rear kingpin length over limit | 263.23 mm | 3.23 mm (1.2%) over 260 max. Set by `kingpin_length_mm`, not by sweep/base. |
| 4 | Rear tie rod unvalidated | — | Still from `legacy_app/carro_formula_2027.csv`; no synthesis, no test. |
| 8 | Anti-squat 0% | 0.00% | By design (horizontal pivot axes). Must be defended at Design Event. |
| N2 | Stale CSV | — | `carro_formula_2027.csv` still holds the *original* swept rear pickups; `--verify` will flag mismatches. Regenerate (`--csv`) once the rear tie rod dependency is handled. |

### DESIGN NOTE

| Item | Details |
|---|---|
| e/a = 1.0 accepted by choice | The front leg is a swept compression strut in braking again. This is the deliberate price of the on-axle pickup constraint, not an oversight. If a judge asks "why e/a 1.0 at the rear," the answer is: it is the minimum sweep that keeps every inboard pickup off the rear of the axle, and the base was minimised (180/160) to keep the resulting legs inside the length band. |

---

## 6. Verification

| Check | Result |
|---|---|
| Full test suite | **247 passed, 13 skipped** |
| Benchmark `test_fsae2027_design.py` | 24 passed; golden legs updated to 423.73 / 386.13 / 383.60 / 351.42 |
| 3D solver convergence, all 4 corners | **PASS** — 1224 envelope pts + roll sweep, 0 failures, max residual 1.56e-10 mm |
| Rear KPIs vs prior audit | max deviation 2.9e-5 (4-decimal rounding only) — camber gain, RC migration, half-track all match |
| Sweep/base decoupling proof | rear KPIs **bitwise identical** (Δ = 0.00e+00) vs a zero-sweep rerun — edit is longitudinal-only |
| Member legs off merged hardpoints | 423.73 / 383.60 / 386.13 / 351.42 mm, all in band; e/a = 1.000000 both arms |
| Rearmost rear pickup (ISO 8855) | LCA_IN_REAR & UCA_IN_REAR both at x = −1540.00 — on the axle line, not behind it |
| Zero-state recovery `solve(0,0,0)` | ball joints recovered to 0.00e+00 mm on all 4 corners |
| Left/right symmetry (RL vs RR) | `max‖|L|−|R|‖ = 0.00e+00 deg` over the full sweep — exact mirror |

Convergence campaign (independent numerical-validator, `DWSolver`, tol 1e-8):

| Corner | Envelope | Converged | Max residual |
|---|---|---|---|
| FL / FR | ±25 mm heave × ±38 mm rack (561 pts each) | 561/561 | 1.56e-10 mm |
| RL / RR | ±25 mm heave (51 pts each) | 51/51 | 4.14e-11 mm |
| Rear roll sweep | −2° to +2° (21 pts) | 21/21 | converged |

---

## 7. Bottom Line

Rev 3 satisfies both hard constraints the designer set: **no inboard pickup
behind the rear axle**, and **all rear legs under 430 mm**. It does so by re-sweeping
to sweep = base/2 (rearmost pickup on the axle line) and shrinking the base to
180/160 to keep the legs short. The tradeoff is e/a = 1.0 — an accepted, defensible
consequence of the on-axle constraint, and still far below the original 1.35. No
kinematic KPI changed. Remaining open items (kingpin length, rear tie rod, 0%
anti-squat, stale CSV) are pre-existing and untouched by this edit.
