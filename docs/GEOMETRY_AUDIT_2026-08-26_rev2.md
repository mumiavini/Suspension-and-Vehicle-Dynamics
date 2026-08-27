# Suspension & Steering Geometry Audit — 2026-08-26 (rev 2)

**Car:** FSAE 2027 — PUCPR Racing (Team #27)
**Type:** Double A-arm, front and rear
**Frame:** ISO 8855 (X+ forward, Y+ left, Z+ up)
**Change under audit:** rear wishbone sweep reduced to zero (`REAR_2027`: `lca_sweep_mm` 230→0, `uca_sweep_mm` 220→0)
**Supersedes:** `GEOMETRY_AUDIT_2026-08-26.md` (rev 1) for the rear axle only

Regenerated with `scripts/geometry_summary.py` (single source of truth). The
front axle, kingpin, steering, and all solver code are unchanged; every front
axle number in rev 1 still holds. This document reports only what the rear-sweep
edit moved.

---

## Executive Summary

**Both critical FAILs from rev 1 are cleared.** The rear front legs — 554 mm and
518 mm against a 430 mm limit in rev 1 — are now **419.58 mm and 386.13 mm**,
both inside the band. The e/a ratios dropped from 1.35/1.38 to **0.00/0.00**.

The fix cost **nothing kinematically**: every front-view KPI (camber gain, roll
centre, scrub, FVSA, RC-migration, half-track change) is **numerically
identical** before and after, because sweep only positions the inboard pickups
longitudinally and never touches the front-view plane. Anti-squat was 0% before
and remains 0% — the sweep was providing no kinematic benefit, only packaging
liability. The full test suite (247 passed, 13 skipped) and the 3D solver over
the ±25 mm envelope confirm nothing downstream shifted.

**Remaining rear issues unchanged from rev 1:** kingpin length still 263.23 mm
(3.23 mm over the 260 limit), the rear tie rod is still unsynthesised, and
anti-squat is still 0% by design. Three **new** layout notes below.

---

## 1. What Changed — Before / After

### 1a. Rear member lengths (true 3D, off the merged hardpoints)

| Member | rev 1 (sweep 230/220) | rev 2 (sweep 0) | Δ | Band | rev 2 status |
|---|---|---|---|---|---|
| LCA front leg | **554.21 mm** ❌ | **419.58 mm** | −134.6 mm | 320–430 | **OK** |
| LCA rear leg | 388.26 mm | 419.58 mm | +31.3 mm | 320–430 | OK |
| UCA front leg | **517.59 mm** ❌ | **386.13 mm** | −131.5 mm | 320–430 | **OK** |
| UCA rear leg | 356.50 mm | 386.13 mm | +29.6 mm | 320–430 | OK |

Front and rear legs are now **equal** on each arm — the pickups sit symmetrically
fore/aft of the ball joint, so both legs share the same 3D length. The front legs
came down 24–25%; the rear legs rose ~8% but stay well inside the band.

### 1b. e/a ratio (2·sweep / base)

| Arm | rev 1 | rev 2 | Target |
|---|---|---|---|
| LCA | 1.353 | **0.000** | ≤ 1.5 (comfortable < 1.0) |
| UCA | 1.375 | **0.000** | ≤ 1.5 (comfortable < 1.0) |

The front legs are no longer the doubly-loaded compression members they were
under braking. Symmetric arm = simplest, most structurally efficient layout.

### 1c. Rear inboard pickup positions (design frame, x positive rearward)

| Pickup | rev 1 | rev 2 | Note |
|---|---|---|---|
| LCA front | x = 1140 | x = 1370 | moved 230 mm rearward |
| LCA rear | x = 1480 | x = 1710 | moved 230 mm rearward |
| UCA front | x = 1160 | x = 1380 | moved 220 mm rearward |
| UCA rear | x = 1480 | x = 1700 | moved 220 mm rearward |

Rear axle line is at x = 1540. The pickups are now symmetric about it (±170 mm
LCA, ±160 mm UCA). **See new layout note N1 below** — the rearmost pickup now
sits *behind* the axle.

---

## 2. What Did NOT Change (proof the edit is purely longitudinal)

Computed directly from `solve_axle()` on both geometries — these front-view
quantities are **bit-identical** old vs new:

| Invariant | rev 1 | rev 2 |
|---|---|---|
| LCA front-view length | 383.60 mm | 383.60 mm |
| UCA front-view length | 351.42 mm | 351.42 mm |
| Scrub radius | 21.97 mm | 21.97 mm |
| RC that flattens LCA | 55.71 mm | 55.71 mm |

And every rev-1 rear KPI carries over unchanged (regenerated, confirmed identical):

| KPI | Value | Band | Status |
|---|---|---|---|
| Roll centre height | 55.00 mm | 20–70 | OK |
| FVSA length | 1400.00 mm | 1300–1700 | OK |
| Scrub radius | 21.97 mm | 5–25 | OK |
| KPI | 8.50 deg | 3–10 | OK |
| UCA/LCA ratio | 0.916 | 0.55–0.98 | OK |
| Camber gain (design) | 0.0409 deg/mm | 0.03–0.05 | OK |
| Camber gain (3D solved) | −0.0411 deg/mm | — | unchanged |
| RC migration | −0.4239 mm/mm | — | unchanged |
| Half-track change | 0.0922 mm/mm | — | unchanged |
| Camber at full bump / droop | −2.5447 / −0.4856 deg | — | unchanged |
| RC range over ±25 mm | 44.5 → 65.7 mm | — | unchanged |
| Anti-squat | 0.00% | 0–30 | OK (by design) |

**Why this must be true:** sweep enters the geometry only through `pickups_x`
(`sla_geometry.py:406`), which sets the inboard pickup X. It never touches the
Y or Z of any point. Anti-geometry comes from the pivot-axis *z-slope*
(`dz_lca_mm`/`dz_uca_mm`, both 0.0), not from pickup spacing, so a horizontal
pivot axis puts the SVIC at infinity regardless of sweep — anti stays 0%.

---

## 3. Leg Forces (screening only — NOT sizing loads)

Rear leg forces, `Fz` outer ≈ 1252 N (unchanged assumptions):

| Case | LCA front | LCA rear | UCA front | UCA rear |
|---|---|---|---|---|
| Cornering | −1418 | −1418 | 402 | 402 |
| Braking | 2269 | −2056 | −108 | −108 |
| Combined | 553 | −2474 | 249 | 249 |

(positive = tension, negative = compression)

The leg-force *magnitudes* are set by the load case and the front-view geometry,
not by sweep, so they are essentially the rev-1 values. What changed is the
**structural quality** of carrying them: at e/a 1.35 the 554 mm front leg was a
slender strut in braking compression; at e/a 0 the load path is a symmetric,
short, buckling-resistant pair. Same force, far better member. These remain
screening numbers — no pushrod in the model, no tire data, friction assumed.

---

## 4. Flagged Issues (rear axle)

### CLEARED since rev 1

| rev 1 # | Issue | rev 1 value | rev 2 |
|---|---|---|---|
| 1 | LCA front leg over limit | 554.21 mm | **419.58 mm — CLEARED** |
| 2 | UCA front leg over limit | 517.59 mm | **386.13 mm — CLEARED** |

### STILL OPEN (unchanged by this edit)

| # | Issue | Value | Note |
|---|---|---|---|
| 3 | Rear kingpin length over limit | 263.23 mm | 3.23 mm (1.2%) over the 260 max. Set by `kingpin_length_mm` input, not by sweep. |
| 4 | Rear tie rod unvalidated | — | RL/RR tie rod still from `legacy_app/carro_formula_2027.csv`; no synthesis, no bounds check, no test. |
| 8 | Anti-squat 0% | 0.00% | Horizontal pivot axes, by design. Must be defended at Design Event. The sweep removal makes this cleaner to defend: there is now no unused swept geometry implying anti was intended. |

### NEW — introduced or surfaced by this edit

| # | Item | Details |
|---|---|---|
| N1 | **Rearmost pickup now sits behind the axle** | rev 1 pickups were all ahead of the rear axle (rearmost at x=1480, 60 mm *in front*). rev 2 pickups are symmetric about the axle, so the rear pickup is at x=1710 — **170 mm behind the axle line**. The chassis/subframe must now provide a pickup plane behind the axle. Packaging change, not a kinematic one; confirm against the frame. |
| N2 | **Stale CSV — `carro_formula_2027.csv` no longer matches** | `geometry_summary.py --verify` reports 12 mismatches: the committed CSV still carries the old swept rear pickups (Δ up to 230 mm on rear LCA/UCA IN points). The CSV is a stale artifact of the pre-edit geometry. Regenerate it (`--csv`) or stop treating it as authoritative. Note the CSV also feeds the unsynthesised rear tie rod (#4), so it can't simply be deleted yet. |
| N3 | **Stale docstring** | `member_legs_mm` docstring (`sla_geometry.py:738`) still reads "the two differ by 170 mm" — true at sweep 230, now false (front and rear legs are equal at sweep 0). Cosmetic. |

---

## 5. Verification

| Check | Result |
|---|---|
| Full test suite | **247 passed, 13 skipped** (`.venv/Scripts/python -m pytest`) |
| Benchmark `test_fsae2027_design.py` | 24 passed; `test_real_member_legs_rear_within_limits` asserts the new in-band legs |
| 3D solver convergence, all 4 corners | **PASS** — 1224 envelope points, 0 failures, max residual 1.56e-10 mm (machine-epsilon class) |
| Rear KPIs vs rev-1 values | max deviation 1.4e-5 (4-decimal rounding only); RC migration, camber gain, half-track all match |
| Sweep-decoupling proof | rear front-view KPIs **bitwise identical** (Δ = 0.00e+00) between old swept and new zero-sweep config, run in-memory |
| Zero-state recovery | `solve(0,0,0)` recovers static ball joints to 0.00e+00 mm on all 4 corners |
| Left/right symmetry (RL vs RR) | `max‖|L|−|R|‖ = 0.00e+00 deg` over the full sweep — exact mirror |
| `geometry_summary.py --verify` vs CSV | 12 mismatches — **expected**, CSV is stale (see N2) |

Convergence campaign (independent numerical-validator run, `DWSolver`, tol 1e-8):

| Corner | Envelope | Converged | Max residual |
|---|---|---|---|
| FL / FR | ±25 mm bump × ±38 mm rack (561 pts each) | 561/561 | 1.56e-10 mm |
| RL / RR | ±25 mm bump (51 pts each) | 51/51 | 4.14e-11 mm |

---

## 6. Bottom Line

The rear-sweep reduction is a clean win: it removed a 29%/20% packaging violation
and a doubled front-leg compression load at **zero kinematic cost**, because the
anti-squat the sweep nominally served was already 0% (pivot axes horizontal).
Every roll, camber, scrub, and RC number is unchanged; only the four leg lengths
and the two e/a ratios moved, all in the right direction. Three follow-ups remain,
all pre-existing or cosmetic: regenerate the stale rear CSV (N2), confirm the
behind-axle pickup plane against the chassis (N1), and address the still-open
kingpin/tie-rod items (#3, #4) that this edit did not touch.
