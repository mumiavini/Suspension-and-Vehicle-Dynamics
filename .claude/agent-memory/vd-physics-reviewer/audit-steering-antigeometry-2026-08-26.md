---
name: audit-steering-antigeometry-2026-08-26
description: Judge-facing correctness audit of sla_geometry.py + steering_geometry.py + axle.py -- anti-geometry, camber signs, Ackermann, parking effort. No CRITICAL errors found.
metadata:
  type: project
---

Audited 2026-08-26 for a Design-Event-facing validity review. Focus: sla_geometry.py,
steering_geometry.py, vdcore/analysis/axle.py, vdcore/geometry/solver.py.

**VERDICT: no CRITICAL formula/sign/frame/unit errors. Numbers are trustworthy for a
judge-facing audit.** Only WARNING/INFO items below.

Verified correct (these had legacy-app failure modes CLAUDE.md warns about; the current
code does NOT reproduce them):
1. Camber sign round-trips per-side. solver `_compute_static_spin_axis` builds spin
   `(+Y,+Z)` left / `(-Y,+Z)` right for negative camber (top inboard); `_extract_angles`
   uses distinct per-side formula (`-atan2(z, y)` left, `-atan2(z, -y)` right). Both
   independently return negative. Correct.
2. Anti-geometry: with dz_lca=dz_uca=0 the two side-view pivot lines are horizontal &
   parallel, `line_intersection` returns None, `_side_view_anti` returns anti=0 EXACTLY
   (`sla_geometry.py:465-482`). Nonzero dz inclines them -> finite SVIC -> nonzero anti.
   Correct; does NOT reproduce the legacy +200% anti-dive artifact.
3. Roll centre / FVIC built from the pivot AXIS via `_effective_pivot_at_x`
   (roll_centre.py), not the pivot midpoint. Correct.
4. Ackermann NOT inverted. `_ackermann_at_steer` (steering_geometry.py:601-652) uses
   cot_inner_ideal = cot_outer - T/L (inner turns MORE than outer). % = 100*(actual-outer)
   /(ideal-outer). Matches its docstring. Correct.
5. Camber gain textbook rate R2D/FVSA = 57.2958/FVSA deg/mm, labeled "from FVSA" and
   explicitly distinguished from the solved rate. Correct.
6. Per-side vs total toe explicit throughout: `toe_deg_per_side`,
   `bump_steer_total_toe_deg_per_mm`, `static_toe_deg_per_side`. No bare `toe`.
7. Parking effort: M_kp = mu*Fz*sqrt(rs^2+tm^2), F_rack via virtual work
   2*M_kp*(pi/180)/C, T_sw = F_rack*pinion_r, unit conversions (Nmm->Nm, mm->m) present
   and correct (steering_geometry.py:841-871).

WARNING/INFO items (do NOT change geometry to fix; code notes only):
- INFO: `_side_view_anti` uses `is_front = axle_x_mm < wheelbase/2` as a front/rear
  proxy. Works for the 2027 config (front x=0, rear x=1540) but is a fragile heuristic
  if someone sets axle_x to a real longitudinal coordinate near mid-wheelbase.
- INFO (carried from [[audit-roll-centre-v1]]): asymmetric RC in roll_centre.py uses
  geometric intersection in `roll_centre_height` (good) but `axle_roll` in axle.py does
  its own intersection via `_line_intersection` after de-rotating -- two RC code paths
  exist; confirm they agree if both are quoted.
- INFO: `_geometric_ackermann_pct` guards d[0] then divides by d[1]; if the kingpin->TRO
  line were exactly fore-aft (d[1]=0) it returns 0.0 via the nan guard. Fine for real
  arms, benign.

Related: [[audit-findings-v1]] (solver internals, now mostly resolved),
[[audit-roll-centre-v1]] (RC intersection).

**Why:** This is the load-bearing conclusion for the judge-facing audit -- the kinematics
are computed correctly.
**How to apply:** On any PR touching these 4 files, re-verify items 1-3 (the ones with
known legacy failure modes) before re-signing off.
