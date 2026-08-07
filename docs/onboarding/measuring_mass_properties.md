# Measuring Vehicle Mass Properties

This guide is written for a second-year engineering student joining PUCPR Racing. It covers the four measurements that turn our `source: "estimate"` tags into `source: "measured"` in the suspension tool — and explains WHY each measurement matters for vehicle dynamics.

## Why mass properties matter

Every load-transfer calculation in the tool depends on three numbers: total mass, CG height, and CG longitudinal position. Of these, **CG height is the most dangerous estimate**. A 10 mm error in CG height changes lateral load transfer by roughly 3%, which is comparable to a full degree of camber change. The tool flags estimated inputs in every result — your job is to replace those flags with measured values.

## 1. Corner weighting — total mass and weight distribution

### What it measures
- Total vehicle mass (with and without driver)
- Front/rear weight distribution
- Left/right weight balance (cross-weight)

### What you need
- Four individual corner scales (bathroom scales are NOT accurate enough — use industrial platform scales rated to at least 200 kg each, resolution 0.1 kg)
- A flat, level surface
- Shims to level the car if the scales have different heights
- A spirit level across the chassis

### Procedure
1. Set tire pressures to the target race pressure
2. Place each wheel on its own scale
3. Level the car — the scales may be different heights. Use shims under the thinner scales. Check with a spirit level on a flat chassis surface. **If the car is not level, the weight distribution is wrong.**
4. Zero the scales with the car on them (or record tare weights)
5. Record all four corner weights
6. Repeat with the driver seated in driving position, arms on the wheel

### Computing the results

```
total_mass_kg = FL + FR + RL + RR
front_fraction = (FL + FR) / total_mass_kg
cross_weight  = (FL + RR) / total_mass_kg   (should be ~0.50 for symmetric car)
```

For the longitudinal CG position (distance behind front axle):

```
cg_x_mm = wheelbase_mm * (1 - front_fraction)
```

### Accuracy
- Typical accuracy: ±0.5 kg per corner with good industrial scales
- Total mass: ±1 kg
- Weight distribution: ±0.3%
- **Record `tol`: 1.0 kg for total mass, 5 mm for cg_x_mm**

### Common mistakes
- Not levelling the car — tilting 0.5 deg shifts ~1% of weight distribution
- Measuring with cold tires (pressures different from race setup)
- Driver not in race position (arms on wheel, helmet on, feet on pedals)

---

## 2. Tilt test — CG height

### Why this is the most important measurement
CG height is the single input with the largest effect on lateral load transfer. It cannot be measured with corner scales alone — you need a tilt test.

### What it measures
- CG height above the ground

### What you need
- A tilt platform or ramp (the car is tilted sideways, one pair of wheels raised)
- Corner scales on the low-side wheels
- An inclinometer or angle measurement (phone app is NOT accurate enough — use a digital inclinometer with 0.1 deg resolution)
- Two people for safety (the car can roll off!)
- Wheel chocks on the low side

### Procedure
1. Record the level (0 deg) corner weights first (from step 1)
2. Raise one side of the car to a known angle θ (typically 5-10 deg). **Do not exceed the angle where the car feels unstable.**
3. Secure the car — wheel chocks on the low-side wheels, someone spotting
4. Record the two low-side corner weights at the tilted angle
5. Repeat at 2-3 different angles for a consistency check

### Computing CG height

At tilt angle θ, the weight shift to the low side is:

```
ΔW = (FL_tilted + RL_tilted) - (FL_level + RL_level)
h_cg = (ΔW * track_mm) / (total_weight * sin(θ))
```

where track_mm is the contact-patch-to-contact-patch track width, and total_weight is in the same units as ΔW (Newtons or kg — be consistent).

If you measured at multiple angles, each should give the same h_cg. If they disagree by more than 5 mm, something is wrong (car not rigid, scales shifted, angle measurement off).

### Accuracy
- Typical accuracy: ±5-10 mm for a careful tilt test
- The dominant error source is the angle measurement — a 0.2 deg error at 8 deg tilt gives ~2.5% CG height error
- **Record `tol`: 10 mm (honest — do not claim 1 mm)**

### Common mistakes
- Using a phone inclinometer (±1 deg accuracy — useless for this measurement)
- Not locking the steering (the car shifts laterally on the tilted platform)
- Not accounting for fluid shift (fuel, coolant) — measure with the tank at race level
- Tilting too far — nonlinear effects and safety risk

---

## 3. Pendulum test — yaw moment of inertia

### What it measures
- Yaw moment of inertia (Izz) — how hard the car is to rotate in plan view

### Why it matters
Yaw inertia affects transient response: how quickly the car changes direction. High yaw inertia means sluggish turn-in but stable at high speed. Low yaw inertia means sharp turn-in but potentially nervous. For FSAE autocross, low yaw inertia is generally better.

### What you need
- A pivot point at the CG position (or a known offset from CG)
- A way to measure the natural oscillation period (stopwatch, or better: an accelerometer logging at 100+ Hz)
- A method to suspend the car so it can rotate freely — this is the hard part. Common approaches:
  - **Trifilar pendulum**: hang the car from three cables. Most accurate but requires a strong overhead structure
  - **Knife-edge pivot**: balance the car on a single pivot line. Simpler but friction adds damping

### Procedure (trifilar pendulum)
1. Suspend the car from three cables of equal length L, attached at known radii R from the CG
2. Give the car a small rotational displacement (~5 deg) and release
3. Measure the period T of the resulting oscillation (average over 10+ cycles for accuracy)
4. Compute:

```
I_zz = (m * g * R^2 * T^2) / (4 * pi^2 * L)
```

where m is total mass, g = 9.81, R is the cable attachment radius, T is the period, L is the cable length.

### Accuracy
- Typical accuracy: ±5-10% for a well-executed trifilar test
- Error sources: cable stretch, friction, CG not exactly centred
- **Record `tol`: 10-15% of the measured value**

### For a first estimate (before you build the test rig)
A reasonable estimate for an FSAE car: I_zz ≈ 0.15 × m × L² (where L is wheelbase in metres). For a 300 kg car with 1.55 m wheelbase: I_zz ≈ 108 kg·m². **Tag this as `source: "estimate"` with `tol: 30 kg·m²`.**

---

## 4. Roll inertia

### What it measures
- Roll moment of inertia (Ixx) — resistance to roll

### Why it matters
Roll inertia affects how quickly the car rolls in response to lateral load. It is much smaller than yaw inertia (concentrated mass near the centreline) and harder to measure accurately.

### Practical approach
For FSAE, roll inertia is almost always estimated rather than measured. A reasonable estimate:

```
I_xx ≈ 0.15 × m × (track/2)^2
```

For a 300 kg car with 1.22 m track: I_xx ≈ 17 kg·m².

Tag this as `source: "estimate"` with `tol: 5 kg·m²`.

If you build a bifilar pendulum (two cables, measuring lateral oscillation), you can measure it. The procedure is similar to the trifilar test but with the car swinging about the longitudinal axis. Accuracy is typically ±15%.

---

## Recording results in the tool

After each measurement, update the vehicle config:

```python
from vdcore.models.mass import MassProperties, ProvenanceFloat

mass = MassProperties(
    total_mass_kg=ProvenanceFloat(value=298.0, source="measured", tol=1.0),
    driver_mass_kg=ProvenanceFloat(value=75.0, source="measured", tol=0.5),
    cg_height_mm=ProvenanceFloat(value=295.0, source="measured", tol=10.0),
    cg_x_mm=ProvenanceFloat(value=820.0, source="measured", tol=5.0),
    front_mass_fraction=ProvenanceFloat(value=0.47, source="measured", tol=0.003),
    yaw_inertia_kgm2=ProvenanceFloat(value=108.0, source="estimate", tol=30.0),
    roll_inertia_kgm2=ProvenanceFloat(value=17.0, source="estimate", tol=5.0),
)
```

Note that `yaw_inertia_kgm2` and `roll_inertia_kgm2` are still estimates — update them when you run the pendulum tests.

## Measurement priority

If you only have time for some measurements, do them in this order:

1. **Corner weighting** (1 hour) — total mass and weight distribution. Everything downstream needs this.
2. **Tilt test** (2 hours) — CG height. Without this, every load-transfer number is suspect.
3. **Yaw inertia** (half a day) — only needed for transient analysis, which is not yet in the tool.
4. **Roll inertia** — estimate is usually sufficient.
