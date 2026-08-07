---
name: audit-roll-centre-v1
description: Physics audit of roll_centre.py FVIC and RC construction -- asymmetric RC bug, degenerate fallback bug, test comment error (2026-08-05)
metadata:
  type: project
---

Audited 2026-08-05. Key findings for `vdcore/analysis/roll_centre.py`:

1. **CRITICAL: Asymmetric axle RC uses averaging instead of geometric intersection** (line 228).
   Milliken RCVD Ch. 17 defines RC as intersection of the two CP-to-FVIC lines, not the average of their Y=0 Z-intercepts. For symmetric cars the difference is zero; for asymmetric setups it can be significant.

2. **WARNING: Degenerate fallback in `_rc_height_from_cp_and_ic`** (lines 250-252).
   When ic_y == cp_y and cp_y != 0, the vertical line never crosses Y=0 but the code returns ic_z. Should return inf or raise.

3. **WARNING: Missing frame annotations on private helpers** (`_effective_pivot_at_x`, `_rc_height_from_cp_and_ic`).

4. **INFO: Test comment is wrong** in test_roll_centre.py line 143: says "IC between wheel and centreline gives negative RC" but the expected result is +400 (positive).

Related to [[audit-findings-v1]] which covers solver.py and derived.py.

**Why:** Roll centre correctness is load-bearing for anti-roll bar sizing and jacking analysis.
**How to apply:** When reviewing any change to roll_centre.py or the Axle model, re-check the asymmetric intersection logic.
