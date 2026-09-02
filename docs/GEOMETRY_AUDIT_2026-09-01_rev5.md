# Geometry revision 5 — chassis packaging and front anti-dive

**Date:** 2026-09-01
**Supersedes:** `GEOMETRY_AUDIT_2026-08-26_rev4.md` (and rev 3's rear layout)
**Driven by:** the chassis team, after they tried to build around the rev‑3/4 hardpoints

---

## 0. Summary

Five changes. Changes 1, 2 and 4 are chassis packaging requests; change 3 is a
performance target the team asked for at the same time; change 5 is the analysis
capability that change 4 needed in order to be evaluated at all:

| # | Change | Where |
|---|---|---|
| 1 | Rearmost rear inboard pickups moved **80 mm ahead of the rear axle line** | `REAR_2027` base/sweep 180/90 and 160/80 → **160/160** both arms |
| 2 | Inboard **UCA pickup plane moved outboard to y = 210** (LCA stays 175) | new `uca_inner_y_mm` on both axles |
| 3 | **7.5 % front anti-dive** (was exactly 0 %) | `FRONT_2027.dz_uca_mm = −11.305` |
| 4 | **Rear toe link inboard moved forward to X = −1460**, sharing the LCA rear bracket | new `REAR_TOE_LINK_INBOARD` in `geometry_summary.py` |
| 5 | **Bump steer now computed on both axles** (the rear had no number at all) | new `vdcore/analysis/toe.py`, report §3c |

Two check bands were widened to admit change 1: `ea_ratio_max` 1.5 → **2.0** and
`lca_length_mm` (320, 430) → **(320, 460)**. Both were approved as part of the
decision, and both are stated as prices below rather than buried.

Everything else on the car is unchanged: track, wheelbase, tyre package, roll
centre heights, FVSA lengths, KPI, caster, static camber and toe.

---

## 1. Change 1 — rear pickups off the driveshaft plane

### The constraint

Rev 3 held both rearmost rear pickups **exactly on the rear axle line**
(ISO `X = −1540`) to satisfy "no pickup behind the axle". That put a chassis
bracket at `X = −1540, Y = 175` — the same X plane as the driveshaft, and right
where the diff and its mounts live. The chassis team could not build a good
connection there and asked for real clearance, not zero.

### The trade

Let Δ be the clearance of the rearmost pickup ahead of the axle. Because the
rear ball joints sit on the axle line, `sweep = Δ + base/2`, so:

$$\text{e/a} = \frac{2\,\text{sweep}}{\text{base}} = 1 + \frac{2\Delta}{\text{base}}
\qquad
\text{front leg} = \sqrt{\text{proj}^2 + (\Delta + \text{base})^2}$$

The two pull in opposite directions: e/a wants a **wide** base, the front leg
wants a **narrow** one. Under rev 4's bands (e/a ≤ 1.5, legs ≤ 430 mm) the two
together cap Δ at **≈ 38 mm** — less than half of what was asked for. Δ = 80 mm
is only reachable by widening a band.

Frontier at Δ = 80 mm, LCA (front-view projection 383.6 mm):

| e/a cap | base | LCA front leg | LCA rear leg |
|---|---|---|---|
| 1.5 | — | infeasible | — |
| **2.0** | **160** | **452.5** | **391.9** |
| 2.5 | 106.7 | 426.6 | 391.9 |
| 3.0 | 80 | 415.6 | 391.9 |

`base = 2Δ = 160` is the **narrowest base meeting the e/a ≤ 2.0 cap**, and
therefore the one giving the shortest possible front leg at that cap. Both arms
use it, so the two pivot axes stay parallel in plan view.

### Result

| Pickup | rev 3/4 | **rev 5** | Clearance to the axle |
|---|---|---|---|
| LCA front | `X = −1360` | **`−1300`** | 240 mm ahead |
| LCA rear | `X = −1540` | **`−1460`** | **80 mm ahead** |
| UCA front | `X = −1380` | **`−1300`** | 240 mm ahead |
| UCA rear | `X = −1540` | **`−1460`** | **80 mm ahead** |

| Member | rev 3/4 | **rev 5** | Band |
|---|---|---|---|
| LCA front leg | 423.7 | **452.5** | 320–460 |
| LCA rear leg | 383.6 | **391.9** | 320–460 |
| UCA front leg | 386.1 | **396.6** | 320–460 |
| UCA rear leg | 351.4 | **325.7** | 320–460 |
| e/a, both arms | 1.00 | **2.00** | ≤ 2.0 |

### The price, stated plainly

**The rear LCA front leg goes 423.7 → 452.5 mm at e/a 1.0 → 2.0.** That is a
longer and more swept compression strut under braking. For scale: rev 1 was
rejected at 554 mm / e/a 1.35; rev 5 sits between rev 1 and rev 3.

`vdcore` computes kinematics only. It does **not** score buckling, stiffness or
member loads (CLAUDE.md scope), and the screening leg forces in the summary are
explicitly not sizing loads. **This member needs a structural check by the team
before the geometry is released.** If it fails, the recovery is to trade back
along the frontier table above — raising the e/a cap shortens the leg.

---

## 2. Change 2 — split inboard pickup plane

The chassis carries the upper wishbone on a wider rail than the lower one, so
the inboard UCA must sit outboard of the inboard LCA. Both arms shared one plane
at `y = 175` until now; the upper arm moves to `y = 210` on both axles.

### Why this was free to grant

An inboard pickup lies on the **ball joint → FVIC** line. Moving it along that
line changes the arm *length* and nothing else in the front-view construction:
the FVIC is where it was, so FVSA, roll centre height and the design camber gain
are untouched.

This is not just an argument from the construction — it was measured through the
full 3D solver. Isolating the two changes on the front axle:

| Front geometry | Solved static RC height |
|---|---|
| pre-rev-5 baseline | 35.0000 mm |
| **UCA pickup to y = 210 only** | **35.0000 mm** |
| UCA rake for anti-dive only | 35.4515 mm |
| both (shipped rev 5) | 35.4999 mm |

The pickup move is exactly neutral to five decimal places. The 0.50 mm of static
RC rise belongs entirely to change 3.

What *does* move is the second-order arc curvature: the ball joint now swings on
a tighter radius, so full-travel values shift slightly and roll-centre migration
**improves**. That was not a target — it is a side effect worth naming so nobody
later mistakes it for one.

| | Front before | Front after | Rear before | Rear after |
|---|---|---|---|---|
| Camber gain [deg/mm] | −0.0384 | −0.0386 | −0.0411 | −0.0411 |
| RC migration [mm/mm] | −0.3914 | **−0.3106** | −0.4239 | **−0.3380** |
| Camber @ full bump [deg] | −2.4896 | −2.5155 | −2.5447 | −2.5649 |
| RC min/max [mm] | 25.30 / 44.90 | 27.90 / 43.44 | 44.52 / 65.74 | 46.75 / 63.67 |
| RC lateral @ 1.5° roll [mm] | −111.46 | −86.90 | −71.09 | −56.34 |

Arm lengths (front-view projection): front UCA 370.03 → **334.25** mm,
rear UCA 351.42 → **315.74** mm. UCA/LCA ratio 0.909 → **0.821** (front) and
0.916 → **0.823** (rear), both inside the 0.55–0.98 band.

---

## 3. Change 3 — 7.5 % front anti-dive

Requested band was 5–10 %; 7.5 % is the midpoint. Carried entirely by raking the
**upper** pivot axis: `dz_uca_mm = −11.305` mm over a 240 mm base (−2.70°). The
lower axis stays horizontal.

### Why the UCA and not the LCA

All three routes reach 7.5 %. They differ in how much geometry they need, and
therefore in how much of the result survives fabrication tolerance:

| Route | Pivot rake needed | SVIC | Sensitivity to ±2 mm of weld tolerance |
|---|---|---|---|
| LCA only | `dz_lca = +4.13` mm (0.91°) | x = 16074, z = 385 | ±3.5 % — a 4 mm z-split is inside fab noise |
| **UCA only (chosen)** | **`dz_uca = −11.305` mm (−2.70°)** | **x = 5422, z = 130** | **±1.3 %** |
| Parallel, both | +6.23 / +5.75 mm (1.37°) | at infinity | ±2.2 % |

The UCA route needs the largest z-split for the same anti, which is exactly why
it is the most robust: the anti-dive the car is built with is closest to the
anti-dive that was drawn.

### Sign — do not flip it

With the LCA axis horizontal, a **positive** `dz_uca` gives **pro**-dive. The
rear UCA pickup must sit *below* the front one. `dz` is defined in the DESIGN
frame (x positive **rearward**), so in the exported ISO table the **forward**
pickup is the higher one: `UCA_IN_FRONT` z = 321.66, `UCA_IN_REAR` z = 310.36.

Because the lower axis is horizontal, the lower ball joint moves purely
vertically and the SVIC sits at **its** height, 130 mm. That is the corrected
ball-joint construction from the 2026-08-27 audit, not the pivot-midpoint one.

### Consequential change: the rack moved 0.9 mm

Raking the upper axis makes the upper ball joint move fore/aft in bump, which
broke the zero-bump-steer solution (−0.0022 deg/mm per side). `rack_z_mm` was
re-solved 158.3 → **157.3915** mm, restoring bump steer to 0.00000 deg/mm.
Ackermann and steering ratio barely moved (70.08 → 70.03 %, 4.58:1).

Note for whoever repeats this: `y_tri_for_zero_bump_steer` cannot fix it. It
sweeps `rack_half_length_mm` over 0.5–1.5× and finds no sign change, returning
`nan`. `rack_z_mm` is the knob, and it is strongly monotonic over 120–210 mm.

### Rear anti-squat is still exactly 0 %

Anti-squat was not requested. Both rear pivot axes remain horizontal, the rear
SVIC stays at infinity, and the rear anti stays at an exact zero. Sweeping the
rear pickups forward does **not** create anti-squat on its own — only `dz` does.

---

## 4. Change 4 — rear toe link onto the LCA bracket

Moving the wishbones forward left the toe link inboard at `X = −1480` as the
**rearmost chassis point on the car** — 20 mm behind the wishbone it was
supposed to be clearing, and only 60 mm ahead of the driveshaft plane. It moves
to **`X = −1460, Y = 175, Z = 169`**.

### Why that point

Bump steer over a grid of candidate positions showed the two axes are not
comparable in importance:

| | Effect on peak toe over ±25 mm |
|---|---|
| **X**, −1460 → −1400 | 0.0128 → 0.0119 deg — essentially nothing |
| **Z**, ±11 mm about 169 | ±0.5 deg — the whole knob |

So X is free to be chosen on packaging grounds alone, and the best packaging
answer is `X = −1460`: identical X **and** Y to the LCA rear pickup, sitting
39.5 mm above it. One bracket carries both, instead of two brackets 20 mm apart.
Nothing on the rear chassis is now behind the wishbone.

Z stays at 169. The exact null is 169.065 mm, which is well inside the 0.5 mm
build tolerance, so the drawing carries the round number and
`toe_link_z_for_zero_bump_steer` re-solves it if the geometry moves again.

### What it did to the kinematics

| Rear toe link | X = −1480 (old) | **X = −1460 (new)** |
|---|---|---|
| Peak toe over ±25 mm | 0.0313 deg | **0.0128 deg** |
| Linear rate | −0.00018 deg/mm | −0.00015 deg/mm |
| Link length | 364.6 mm | 371.4 mm |
| Plan-view sweep | 18.3° | 21.3° |
| Clearance to the LCA rear leg | 44.2 mm | 39.5 mm |

The peak **more than halved**. The move was made for packaging; the kinematic
improvement is a side effect, and is pinned by a test so a later change cannot
quietly undo it. The slightly greater plan-view sweep is mildly favourable for
toe stiffness under lateral load, which this tool does not compute.

### The option not taken

Mounting the toe link inboard on the **lower wishbone arm** was considered and
rejected. It is kinematically valid — the inboard end riding on the arm is still
one constraint, so the corner keeps 1 DOF — but `vdcore`'s `Corner` models
chassis hardpoints as fixed, so it would need a solver change; it loads the
wishbone in bending; and it buys nothing here, because the chassis pickup option
already lands below the baseline bump steer with a shared bracket.

### Provenance

The rear toe link is now a **declared design input** in `geometry_summary.py`
rather than a value read back out of `legacy_app/carro_formula_2027.csv`. The
provenance report's standing `!!` warning — *"REAR TIE ROD IS NOT SYNTHESISED BY
ANY SCRIPT … not regenerated, not bounds-checked, and not covered by any test"* —
is retired: it is regenerated, it is bounds-checked in §3c, and it is covered by
five tests.

---

## 5. Change 5 — bump steer is computed now

`vdcore/analysis/toe.py` is new: `toe_sweep()` and `bump_steer()`, mirroring the
API of `camber.py`. Report section **3c** covers both axles.

The rear had **no bump-steer number anywhere** before this. `steering_geometry.py`
owns bump steer and is front-axle only, so the rear toe link's effect on toe was
never computed — it is an ordinary five-link constraint and `DWSolver` could
always see it; nothing was asking. That is why change 4 could be evaluated at all.

Two numbers are reported, not one:

| Axle | Linear rate [deg/mm per side] | Peak over ±25 mm [deg per side] |
|---|---|---|
| Front | −0.00002 | **0.1598** |
| Rear | −0.00015 | **0.0128** |

**The front's linear rate is nulled, and its peak is 0.16 deg.** Those are not in
contradiction: nulling the linear term leaves a quadratic, so the toe curve is a
parabola about ride height and the wheel toes the *same way* in bump and droop —
a straight-line fit through that reads zero. The front axle has always behaved
this way (it measured 0.1581 deg before this revision); it was invisible because
only the linear rate was ever reported. Reporting one number would have called
this geometry "zero bump steer".

Toe is reported **per side** throughout, with `total_toe_deg` available
explicitly, per the CLAUDE.md rule that no toe quantity is ever left ambiguous.

---

## 6. Open items — for the chassis team, not decided here

1. **The rear toe link OUTBOARD point sits 54.6 mm behind the wishbone ball
   joints** (`X = −1594.6`). It was left alone: it is a rear-upright steering-arm
   feature at `Y = 521`, not a chassis connection, and nowhere near the
   driveshaft. Bringing it ahead of the axle would be an upright redesign and
   would flip the sign of toe change under longitudinal load. The report now
   prints this as a TO CONFIRM item.
2. **The rear inboard plane is no longer a single Y.** Structure is now needed
   at `y = 175` (LCA + toe link) and `y = 210` (UCA). Confirm both are reachable.
3. **Structural check on the 452.5 mm rear LCA front leg** — see §1.
4. **CAD clash check** on the toe link against the LCA rear leg: 39.5 mm of
   minimum separation, down from 44.2 mm. Comfortable, but it is a straight-line
   measure between member centrelines and takes no account of tube diameter,
   rod-end bodies or the shared bracket's own geometry.

## 7. Carried forward unresolved from rev 4

- Rear kingpin length 263.23 mm against a 260 mm limit (1.2 % over), and
  260.30 mm on the true 3D measure. Pre-existing, untouched by this revision.

---

## 8. Regenerated artefacts

Every downstream deliverable was rebuilt from the changed config in the same
commit — the merged CSV feeds the chassis team, the MotionSolve model, the
SolidWorks skeleton and six tests:

- `Geometry Summary/hardpoints_2027_merged.csv`
- `vdcore/analysis/toe.py` (new), `tests/unit/test_toe.py` (new)
- `legacy_app/carro_formula_2027.csv` (kept in sync; `--verify` compares against it)
- `Geometry Summary/Geometry Summary 2027.md`
- `Geometry Summary/Hardpoints Suspensão 2027.docx` / `.pdf`
- `Geometry Summary/solidworks_skeleton.bas`

## 9. Verification run

| Check | Result |
|---|---|
| `pytest` | **425 passed**, 14 skipped |
| `scripts/check_purity.py` | PASS, no forbidden imports in `vdcore/` |
| `geometry_summary.py --verify` | merged hardpoints match the CSV exactly |
| `ruff` | identical counts before and after (5 in `vdcore/`, 101 elsewhere); the new files add none |

Golden values in `tests/benchmarks/test_fsae2027_design.py`,
`test_anti_geometry_and_ackermann.py`, `tests/property/test_fsae2027_invariants.py`
and `tests/unit/test_vdcore_bridge.py` were re-anchored deliberately. Four new
tests were added that pin this revision's *intent* rather than its output:

- front anti-dive is 7.5 % with a finite SVIC at the lower ball joint height;
- the inboard UCA is outboard of the inboard LCA on both axles;
- every rear inboard pickup clears the axle plane by ≥ 80 mm;
- moving `uca_inner_y_mm` leaves the FVIC bit-identical — the property that made
  change 2 free, and which nothing tested before.

One test got *better* as a side effect. `test_legacy_app_anti_features_...` could
previously only check the trivial 0 % case, because the shipped design had no
pivot rake — the exact blind spot that hid the pivot-midpoint bug until the
2026-08-27 audit. With real rake on the front it is now a live comparison: the
legacy construction gives 6.908 % against the correct 6.923 %. They agree closely
only because the LCA axis is horizontal; that is not a licence to quote the
legacy solver.
