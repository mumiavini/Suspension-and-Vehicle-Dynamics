---
name: audit-findings-v1
description: First full audit of primitives.py, solver.py, derived.py, camber.py -- sign conventions, solver constraints, contact patch formulas (2026-08-05)
metadata:
  type: project
---

Audited 2026-08-05. Key findings:
- solver.py contact patch lateral shift sign is WRONG for negative camber
- solver.py 9th constraint uses an unused variable and has inconsistent reference direction
- solver.py _extract_angles uses bare `toe_deg` variable internally (minor, renamed on output)
- solver.py rack applied as +Y translation for both sides -- may be physically incorrect for right corners
- solver.py UCA axis projection residual uses the displaced axis_unit for reference projection but the initial reference used the same displaced axis -- potential subtle error

**Why:** These are the physics-correctness findings that must be tracked across sessions.
**How to apply:** When reviewing any PR touching solver.py or derived.py, re-check these items.
