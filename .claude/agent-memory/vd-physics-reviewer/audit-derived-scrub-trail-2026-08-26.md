---
name: audit-derived-scrub-trail
description: Audit of vdcore/geometry/derived.py scrub_radius_mm, mechanical_trail_mm, _kingpin_ground_intercept — signs/frames correct, no critical errors (2026-08-26)
metadata:
  type: project
---

Audited three new functions in `vdcore/geometry/derived.py`: `_kingpin_ground_intercept`, `scrub_radius_mm`, `mechanical_trail_mm`. Verdict: NO critical errors. Signs, frames, units all correct and would stay correct for other geometries (negative scrub if patch inboard, negative trail if intercept behind patch), not just for the pinned benchmark numbers.

**Why:** Vinicius asked for a physics/sign confirmation (not just number-matching) before relying on these for the 2027 geometry reports. Empirical: FL/FR scrub +15.077 (bench 15.08), rear scrub +21.971 (bench 21.97), front trail +21.435 (bench 21.43), rear trail ≈0.

**How to apply:** Treat these three functions as sign-verified. If a future change touches the trail/scrub sign or the side-detection, re-check against the frame table below.

### The three-frame trail sign chain (load-bearing — memorize)
Positive mechanical trail = intercept AHEAD of patch = self-aligning ("positive trail").
- Root `steering_geometry.py` — design frame X+ REARWARD — `cp_x - kp_gnd_x` (line 201). Correct there.
- New `derived.py` — ISO 8855 X+ FORWARD — `kp_ground_x - cp_x` (line 198). Correct flip.
- Legacy `model_3d.py:246` — X+ forward — `contact_patch.x - intercept.x`. INVERTED (CLAUDE.md flags it). The new code is the exact negation of the legacy bug, i.e. the corrected form. Confirmed NOT repeating the legacy error.

### Scrub fold
`scrub_radius_mm` = `cp_y - kp_ground_y` on left, negated on right, so positive = patch OUTBOARD both sides. Side detected by `contact_patch.y_mm >= 0` (ISO: left FL/RL +Y, right FR/RR -Y) — sound. Matches OptimumK-correlated `cp_y - kp_ground_y` in `tests/benchmarks/test_optimumk_correlation.py:188`, which is validated on the LEFT corner where the fold is a no-op.

### Ground intercept
`t = -lbj_z / kp_z` walks LBJ along (ubj-lbj) to z=0; patch also at z=0 per `contact_patch`, same plane. Guard `abs(kp_z) < 1e-10` raises ValueError (horizontal kingpin) rather than returning a fictitious huge intercept — correct per the "never a plausible number silently" rule.

### Minor notes (INFO, not blocking)
- `scrub_radius_mm` side test `contact_patch.y_mm >= 0` puts the exact Y=0 patch on the "left" branch. Harmless (delta≈0 there) but a centreline corner is undefined anyway.
- The OptimumK test helper `_scrub_radius_mm` (test line 175) does NOT fold the right side (returns raw `cp_y - kp_ground_y`); it is only exercised on the left corner, so it agrees with the production function there. If that test is ever extended to a right corner, it will disagree in sign with production — flag then.
