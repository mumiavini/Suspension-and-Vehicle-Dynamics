---
description: "Vehicle dynamics coordinate frame, sign conventions, unit policy, and glossary. Use whenever writing or reviewing code that handles coordinates, angles, forces, or dimensions in vdcore/. The transform table in references/frames.md converts between ISO 8855, J670e, the legacy project frame, Optimum Kinematics, and SolidWorks."
---

## Coordinate frame — ISO 8855

- **X+ forward, Y+ LEFT, Z+ up.** Right-handed: X × Y = Z.
- Origin: front axle centreline, ground plane, vehicle centreline.

### Y+ is LEFT — read this carefully

Left wheels (FL, RL) have **positive** Y coordinates.
Right wheels (FR, RR) have **negative** Y coordinates.

This is the opposite of most FSAE CAD defaults and the old codebase. Every function that returns or consumes a Y coordinate must document which side is positive.

### Rotations

Right-hand rule about the respective axis:
- **Roll** about X (positive roll = right side down)
- **Pitch** about Y (positive pitch = nose up)
- **Yaw** about Z (positive yaw = turn left)

### Why ISO 8855

It is a citable standard. Z-up keeps roll-centre heights and ride heights positive. It is the native frame for MF-Tyre / Pacejka, so the tire module needs no conversion.

## Frame conversions

See [references/frames.md](references/frames.md) for explicit 3×3 sign/permutation matrices converting between ISO 8855, J670e (z-down), the legacy project frame, Optimum Kinematics, and SolidWorks.

## Units

| Quantity | Unit | Notes |
|---|---|---|
| Length | mm | Always. Never inches. |
| Force | N | |
| Torque | Nm | |
| Angle (user-facing) | deg | I/O, plots, reports |
| Angle (internal) | rad | Computation, solvers |

Use unit suffixes on ambiguous variable names: `steer_angle_deg`, `rack_travel_mm`, `force_n`.

## Sign conventions

| Quantity | Positive | Notes |
|---|---|---|
| Camber | Top of wheel outboard | Negative camber = top inboard (the typical FSAE setup). Same sign convention on both sides. |
| Toe | Toe-in | Per-side. **Always state per-side or total explicitly**: `toe_deg_per_side`, `total_toe_deg`. |
| Caster | Kingpin axis tilted rearward at top | Standard SAE definition. |
| KPI | Kingpin axis tilted inboard at top | Standard SAE definition. |
| Scrub radius | Kingpin-ground intersection outboard of contact patch | Positive = kingpin hits ground inboard of CP. |

### Per-side vs total toe — THE RULE

Every toe variable, function parameter, return value, table column, and plot axis label must state whether it is per-side or total. No exceptions. Use `toe_deg_per_side` or `total_toe_deg`. Never bare `toe` or `toe_deg`.

## Glossary

See [references/glossary.md](references/glossary.md) for the full term list with symbols and units.
