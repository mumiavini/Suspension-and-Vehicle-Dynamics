---
description: "Steering system design: rack force, parking effort, KPI, scrub radius, mechanical and pneumatic trail, Ackermann error, steering ratio, effort budget decomposition. Use when writing or reviewing steering-related code in vdcore/. References Lenkungshandbuch and Milliken RCVD."
---

## Steering geometry

### Kingpin geometry (ISO 8855: X+ fwd, Y+ LEFT, Z+ up)

The **kingpin axis** is the line from LBJ to UBJ. It defines:

- **KPI (σ)**: inclination of the kingpin axis in the Y-Z (front) view, measured from the vertical. Positive = tilted inboard at top.
- **Caster (τ)**: inclination in the X-Z (side) view, measured from the vertical. Positive = tilted rearward at top.
- **Scrub radius (rs)**: lateral distance from the kingpin-ground intercept to the contact patch centre, in the front view. Positive when the kingpin axis intercepts the ground inboard of the contact patch.
- **Mechanical trail (tm)**: longitudinal distance from the kingpin-ground intercept to the contact patch centre, in the side view. Positive when the intercept is forward of the contact patch (normal for positive caster).
- **Pneumatic trail (tp)**: longitudinal offset from the contact patch centre to the centroid of the tyre's lateral force distribution. Load-, slip-, and speed-dependent. Requires tire data.

### Ackermann geometry

**Ackermann %** = 100 × (actual inner-outer steer difference) / (ideal Ackermann difference).

Ideal Ackermann: the inner and outer wheels' axes both pass through the same instantaneous turn centre on the rear axle line.

Geometric Ackermann from steering arms: extend the line from the kingpin axis to the TRO toward the rear axle. If the left and right lines intersect on the rear axle centreline, Ackermann = 100%.

### Steering ratio and C-factor

- **C-factor** (rack sensitivity): mm of rack travel per degree of road wheel steer. Computed from a small rack displacement test.
- **Steering ratio**: (steering wheel angle) / (average road wheel angle) = (pinion linear pitch × 360°) / (C-factor × 2π × pinion radius). Typical FSAE: 3:1 to 5:1.

## Steering effort

### Parking effort (worst case)

The parking rack force is the maximum force required to steer from lock to lock while stationary. It is set by:

1. **Axle load** on the steered wheels
2. **Scrub radius** and **mechanical trail** (moment arms)
3. **Tire size** (contact patch width affects the restoring moment)
4. **Inflation pressure** (affects contact patch area and friction coefficient)
5. **Surface friction** (μ ≈ 1.0 for dry asphalt, μ ≈ 0.8 for painted surface)

Formula (Lenkungshandbuch ch. 4): `F_rack = μ × Fz × √(rs² + tm²) / (C_factor)` (simplified; full model includes tire self-aligning torque).

### Effort budget decomposition

| Source | Typical % of total |
|---|---|
| Tire friction (μ × Fz) | 50–70% |
| Self-aligning torque | 15–25% |
| Kingpin moment (scrub × Fz) | 10–20% |
| Mechanical friction | 5–10% |

### Design levers for reducing steering effort

1. Reduce scrub radius (move kingpin ground intercept toward CP)
2. Increase C-factor (longer steering arm → more rack travel per degree)
3. Reduce caster / mechanical trail (less self-centering moment)
4. Reduce tire width or increase inflation pressure
5. Increase steering ratio (trade-off: slower response)

The PUCPR FSAE26 car reduced effort from 29.87 Nm to 14.73 Nm (−50.7%), mainly through steering ratio and KPI/scrub radius changes.

## Key references

- **Lenkungshandbuch** (Pfeffer & Harrer): ch. 4 (steering forces), ch. 5 (kinematics), ch. 7 (power assist sizing)
- **Race Car Vehicle Dynamics** (Milliken & Milliken): ch. 17 (steering geometry), ch. 19 (alignment effects)
- **The Automotive Chassis** (Reimpell, Stoll, Betzler): ch. 3.5 (steering kinematics)
