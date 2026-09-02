# Geometry revision 6 — asymmetric rear inboard clearance (LCA 100 mm, UCA 80 mm)

**Date:** 2026-09-02
**Supersedes:** the symmetric-160/160 rear layout in `GEOMETRY_AUDIT_2026-09-01_rev5.md`
**Driven by:** the driveshaft/diff packaging around the rear LCA rear bracket

---

## 0. Summary

One change, requested directly: the rear **LCA** rear-inboard pickup needed
more room off the driveshaft plane than rev 5 gave it. The **UCA** is
unaffected and stays exactly where rev 5 put it.

| # | Change | Where |
|---|---|---|
| 1 | LCA rearmost inboard clearance **80 mm → 100 mm ahead of the rear axle**, UCA stays at 80 mm | `REAR_2027.lca_base_mm`/`lca_sweep_mm` 160/160 → **200/200**; `uca_base_mm`/`uca_sweep_mm` unchanged at 160/160 |
| 2 | Rear toe link inboard follows the LCA rear bracket it shares | `REAR_TOE_LINK_INBOARD` X −1460 → **−1440** |

One check band moved to admit change 1: `lca_length_mm` top **460 → 490 mm**.
`ea_ratio_max` (2.0) is untouched — this is the point of the option chosen
below.

The rear inboard layout is now genuinely asymmetric front-to-back: the LCA
and UCA rear-inboard pickups no longer sit on the same X. This was already
true for Y since rev 5 (change 2, `uca_inner_y_mm` split); rev 6 adds the X
split. Everything else on the car is unchanged: track, wheelbase, tyre
package, front axle, rear roll centre construction, static camber/toe.

---

## 1. Change 1 — LCA to 100 mm, UCA held at 80 mm

### The constraint

Rev 5 held both rear arms at a symmetric Δ = 80 mm (`base = sweep = 160` on
both). The team asked for the LCA rear bracket alone to clear the driveshaft
by 100 mm, for space around the diff mounts; the UCA bracket has no such
conflict and stays at 80 mm.

### The trade

With the rear ball joints on the axle line, `sweep = Δ + base/2`, so raising
Δ alone (holding the rev-5 base) breaches e/a; narrowing the base to hold e/a
instead lengthens the front leg. Two points on that frontier were computed at
Δ = 100 mm for the LCA:

| Option | `lca_base_mm` | `lca_sweep_mm` | e/a | LCA front leg | Price |
|---|---|---|---|---|---|
| A | 160 (unchanged) | 180 | **2.25** (breaches 2.0 cap) | **~463.4 mm** (breaches 460 band) | two small breaches |
| **B (chosen)** | **200** | **200** | **2.00** (exactly the existing cap) | **~487 mm** | one band change, no cap change |

**Option B was chosen.** It lands exactly on the e/a cap already in force —
no new trade-off dimension is opened, only the existing leg-length band moves
to admit a longer member. Option A would have breached *two* bands at once
for a smaller nominal base change, which is a worse-shaped trade even though
the numbers look smaller.

### Result (solved, not hand-calculated)

| Pickup | rev 5 | **rev 6** | Clearance to the axle |
|---|---|---|---|
| LCA front | `X = −1300` | **`−1240`** | 300 mm ahead |
| LCA rear | `X = −1460` | **`−1440`** | **100 mm ahead** |
| UCA front | `X = −1300` | `−1300` (unchanged) | 240 mm ahead |
| UCA rear | `X = −1460` | `−1460` (unchanged) | **80 mm ahead** |

| Member | rev 5 | **rev 6** | Band |
|---|---|---|---|
| LCA front leg | 452.49 | **486.98** | 320–**490** |
| LCA rear leg | 391.85 | **396.42** | 320–490 |
| UCA front leg | 396.60 | 396.60 (unchanged) | 320–490 |
| UCA rear leg | 325.71 | 325.71 (unchanged) | 320–490 |
| e/a, LCA | 2.00 | **2.00** (unchanged, at cap) | ≤ 2.0 |
| e/a, UCA | 2.00 | 2.00 (unchanged, at cap) | ≤ 2.0 |

`design.rear.lca_length_mm` (front-view 2D projection, ~383.6 mm) is
unaffected, confirmed by running the solver rather than assumed: the
front-view arm length depends only on `inner_pickup_y_mm` and the FVIC, never
on `axle_x`/base/sweep. Only the true 3D leg lengths move.

### The price, stated plainly

**The rear LCA front leg goes 452.5 → 487.0 mm**, now the single longest
member in the whole rear corner and the sole binding one against the
320–490 mm band (previously the UCA front leg was closer to its own limit;
it hasn't moved). The UCA legs are completely unaffected by this change —
the two arms are no longer symmetric in either plan-view sweep or leg length,
which is new for this car.

`vdcore` computes kinematics only, not buckling, stiffness or member loads.
**This member needs the same structural check flagged in rev 5 for the
452.5 mm leg — now against a longer, more swept 487 mm leg.** If it fails,
the recovery is the same frontier: trade the LCA e/a cap upward to shorten
the leg (see rev 5 §1's frontier table for the shape of that trade — it
applies per-arm now, not to both arms at once).

---

## 2. Change 2 — rear toe link follows the LCA bracket

The rear toe link inboard shares a bracket with `LCA_IN_REAR` (rev 5 change
4). Moving the LCA rear pickup forward by 20 mm (`X: −1460 → −1440`) moves
the toe link inboard the same 20 mm, to keep one bracket instead of splitting
it into two.

### Re-verified against bump steer, not assumed

Rev 5's own sensitivity study found rear bump steer nearly insensitive to the
toe-link inboard X (peak toe moved only 0.0128 → 0.0119 deg across
`X = −1460 → −1400`, a 60 mm sweep) and entirely governed by Z. The 20 mm move
here is inside that already-characterized range, but was re-run live rather
than reused:

| Rear toe link | X = −1460 (rev 5) | **X = −1440 (rev 6)** |
|---|---|---|
| Linear rate [deg/mm per side] | −0.000154 | **−0.000150** |
| Peak over ±25 mm [deg per side] | 0.0128 | **0.0125** |
| Zero-bump-steer null Z [mm] | 169.058 | 169.065 |

Both numbers move in the favourable direction, by an amount well below
measurement noise. Z stays at the round 169 mm — the null sits within
0.1 mm of it either way, far inside the 0.5 mm build tolerance already
accepted in rev 5 — and both the linear rate and peak stay well inside
`BUMP_STEER_LINEAR_LIMIT` (0.005 deg/mm) and `BUMP_STEER_PEAK_LIMIT`
(0.30 deg) in `scripts/geometry_summary.py`.

---

## 3. What did NOT change

- UCA base/sweep, UCA leg lengths, UCA e/a — bit-identical to rev 5.
- Front axle — untouched.
- Rear roll centre construction, rear anti-squat (still exactly 0 %),
  rear camber gain band (0.030–0.050 deg/mm) — see verification run below.
- `ea_ratio_max` — stays at 2.0; this is the reason Option B was preferred
  over Option A.
- Rear toe link Z, outboard point, link length ballpark — Z unchanged,
  outboard untouched, length shifts only with the 20 mm inboard X move.

---

## 4. Open items — carried forward, not decided here

1. **Structural check on the 487 mm rear LCA front leg** (raised, not
   lowered, from rev 5's already-flagged 452.5 mm member) — see §1.
2. **CAD clash check** on the toe link against the LCA rear leg, now at the
   new shared-bracket X — rev 5's 39.5 mm separation figure needs
   re-measurement at the new geometry; not recomputed here because it is a
   solid/tube-diameter question outside `vdcore`'s scope (kinematics only).
3. Rear kingpin length (263.23 mm against a 260 mm limit, pre-existing since
   rev 4) — untouched by this revision, still open.

---

## 5. Regenerated artefacts

- `sla_geometry.py` — `REAR_2027.lca_base_mm`/`lca_sweep_mm`,
  `CheckLimits.lca_length_mm` top bound
- `scripts/geometry_summary.py` — `REAR_TOE_LINK_INBOARD`
- `Geometry Summary/hardpoints_2027_merged.csv`
- `Geometry Summary/Geometry Summary 2027.md`
- `Geometry Summary/solidworks_skeleton.bas`
- `legacy_app/carro_formula_2027.csv` (kept in sync; `--verify` compares
  against it — RL/RR `LCA_IN_FRONT`, `LCA_IN_REAR`, `TIE_ROD_IN` rows moved;
  UCA rows untouched)
- `tests/benchmarks/test_fsae2027_design.py` — golden values re-anchored:
  `test_real_member_legs_rear_within_limits`,
  `test_rear_pickups_clear_the_driveshaft_plane` (now per-arm thresholds:
  LCA ≥ 100 mm, UCA ≥ 80 mm)

## 6. Verification run

| Check | Result |
|---|---|
| `tests/benchmarks/test_fsae2027_design.py` | **36 passed** |
| `pytest` (full suite, minus one pre-existing unrelated collection error — see §7) | **425 passed**, 14 skipped |
| `scripts/geometry_summary.py --verify` | merged hardpoints match `carro_formula_2027.csv` exactly |

Only the two tests predicted by the plan needed golden-value updates
(`test_real_member_legs_rear_within_limits`,
`test_rear_pickups_clear_the_driveshaft_plane`); every other rear test in the
file — including `test_rear_rates`, `test_rear_roll`,
`test_rear_anti_squat_is_exactly_zero`,
`test_rear_improved_when_the_toe_link_moved`,
`test_no_chassis_point_sits_behind_the_wishbone`, and the toe-link bracket
share check — passed unmodified, confirming the change is contained to the
LCA leg lengths and the toe-link X shift's already-small bump-steer effect.

## 7. Byproduct findings, not part of this revision

- **`.venv` was in a corrupted, partially-installed state before this
  session** (missing `pydantic`/`polars`, a numpy install missing
  `RECORD`/`METADATA`), independent of this geometry change. Core packages
  (`numpy`, `scipy`, `pydantic`, `polars`, `pytest`, `hypothesis`) were
  reinstalled via targeted `uv pip install` (a respawning `ruff.exe`
  language-server process kept `uv sync` itself from completing by locking
  its own executable — not a project issue). `tests/unit/test_app_agrees_with_summary.py`
  still fails to collect (`streamlit` → `google.protobuf` chain incomplete)
  and was excluded from the run above; this is orthogonal to `vdcore` and to
  this revision.
- **`scripts/build_summary_doc.py` has a pre-existing bug**, unrelated to
  this change: it references `MergedHardpoints.rear_tie_rod_from_csv`, an
  attribute that no longer exists after rev 5 made the rear toe link a
  declared design input rather than a CSV-sourced value (§4 of rev 5). Docx/pdf
  regeneration was not in this revision's scope and was not attempted beyond
  noting the failure; the `.docx`/`.pdf` artefacts in `Geometry Summary/` are
  therefore stale relative to rev 6 and should be treated as such until that
  script is fixed separately.
