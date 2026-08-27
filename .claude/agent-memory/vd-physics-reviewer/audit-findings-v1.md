---
name: audit-findings-v1
description: Audit of primitives.py, solver.py, derived.py, camber.py -- sign conventions, solver constraints, contact patch. Some items resolved by 2026-08-26 re-audit.
metadata:
  type: project
---

Audited 2026-08-05; re-verified 2026-08-26. Status of prior findings:

- **RESOLVED (verify before quoting):** solver.py contact-patch lateral shift.
  As of 2026-08-26 `solver.py:503-504` reads `lateral_shift = -r*tan(gamma)` then
  `cp_y = wc[1] + lateral_shift if left else wc[1] - lateral_shift`. Traced both
  sides for negative camber: patch moves OUTBOARD on both (correct). The v1
  "sign is WRONG" finding no longer applies -- code was fixed.
- solver.py 9th constraint: REWRITTEN. Now `r[8]` pins the wheel-centre height in
  the CHASSIS frame (`solver.py:348-350`) -- a real driving equation, not the old
  invariant-zero projection. This fixed a documented 8.1% bump overshoot / 7.3%
  camber-gain error. Good.
- solver.py `toe_deg_per_side` is now the field name everywhere (SolverResult and
  internals) -- the old bare `toe_deg` is gone. Good.
- solver.py rack applied as +Y translation for both sides (`solver.py:274`). Still
  present but this is CORRECT: TRI points are in ISO 8855 where +Y is a single
  physical rack-shift direction for the whole axle. Not a bug.

**Why:** These are the physics-correctness findings tracked across sessions.
**How to apply:** Re-check on any PR touching solver.py. The contact-patch and 9th
constraint items were the load-bearing ones and are now correct -- do not re-flag
them from stale memory without re-reading the lines.
