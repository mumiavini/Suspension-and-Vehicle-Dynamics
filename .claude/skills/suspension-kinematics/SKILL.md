---
description: "Double-wishbone suspension kinematics: solver formulation, instant centres, roll centre, camber/toe/caster extraction, anti-features, motion ratio, bump steer. Use when writing or reviewing kinematic solver code, KPI calculations, or sweep analysis in vdcore/geometry/ or vdcore/analysis/. Includes benchmark cases with known answers in references/benchmark_cases.md."
---

## Double-wishbone kinematics

### Geometry definition

A double-wishbone corner has 10 hardpoints (in ISO 8855: X+ fwd, Y+ LEFT, Z+ up):
- UCA: inboard front, inboard rear, outboard (upper ball joint)
- LCA: inboard front, inboard rear, outboard (lower ball joint)
- Tie rod: inboard (rack end), outboard (TRO on upright)
- Wheel centre, contact patch (derived)

The **upright** is the rigid body connecting UBJ, LBJ, and TRO. Its three internal distances are invariants.

### Solver formulation

The 3D solver finds the displaced positions of UBJ, LBJ, and TRO given inputs `(heave_mm, roll_deg, rack_mm)`.

**Constraints** (9 unknowns: 3 coordinates × 3 points):
1. UBJ lies on a sphere centred at the moved UCA effective inboard, radius = UCA length (1 equation: distance²)
2. LBJ lies on a sphere centred at the moved LCA effective inboard, radius = LCA length (1 equation)
3. TRO lies on a sphere centred at the moved tie-rod inboard, radius = tie-rod length (1 equation)
4. dist(UBJ, LBJ) = d_ubj_lbj (rigid upright, 1 equation)
5. dist(UBJ, TRO) = d_ubj_tro (1 equation)
6. dist(LBJ, TRO) = d_lbj_tro (1 equation)

6 constraints for 9 unknowns — the system is under-determined. The remaining 3 DOF correspond to the upright's rotational orientation. `least_squares` with a good seed resolves this: the seed from the previous state provides continuity across a sweep.

**Method**: `scipy.optimize.least_squares`, default trust-region (NOT Levenberg-Marquardt — LM does not support bounds). Numerical Jacobian by default.

**Chassis point motion**: inboard points move with heave (Z translation) and roll (rotation about the vehicle X axis through the roll centre approximation or ground level).

### Instant centres

- **Front-view instant centre (FVIC)**: intersection of the UCA and LCA projected into the Y-Z plane. The line from the contact patch to the FVIC defines the FVSA.
- **Side-view instant centre (SVIC)**: intersection in the X-Z plane. Defines the SVSA.
- **Roll centre**: intersection of the line (contact patch → FVIC) with the vehicle centreline (Y=0 plane), for each side. The roll centre is the average or geometric construction.

### Angle extraction

**Camber**: angle of the wheel plane vs vertical in the Y-Z plane.
- Build the upright frame from UBJ, LBJ, TRO.
- The wheel-plane normal is perpendicular to the upright axle (UBJ-LBJ direction) in the Y-Z plane.
- **Left/right differ**: because Y+ is LEFT, the sign of the wheel-plane normal's Y component flips between sides. The camber extraction function MUST handle this explicitly.
- Negative camber = top of wheel inboard (both sides).

**Caster**: angle of the kingpin axis (LBJ→UBJ) projected onto the X-Z plane, measured from the vertical. Positive = rearward tilt at top.

**KPI**: angle of the kingpin axis projected onto the Y-Z plane, measured from the vertical. Positive = inboard tilt at top.

**Toe**: angle of the wheel centreline vs vehicle centreline in the X-Y (plan) view. Positive = toe-in. Always report as `toe_deg_per_side`.

### Anti-features

- **Anti-dive** (front, braking): determined by the side-view instant centre height and the braking force line. %AD = (tan(θ_SVIC) / tan(θ_brake)) × 100, where θ is the angle from the contact patch.
- **Anti-squat** (rear, traction): analogous for the rear axle under drive torque.
- **Anti-lift** (rear, braking): rear geometry's resistance to front-end lift during braking.

### Bump steer

Change in toe per unit heave: Δδtoe / Δheave (deg/mm). Computed from a heave sweep with rack fixed.

### Benchmark cases

See [references/benchmark_cases.md](references/benchmark_cases.md) for worked examples with known answers.
