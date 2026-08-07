# Benchmark cases for kinematic solver validation

## Case 1: Symmetric planar four-bar (2D equivalent)

A perfectly symmetric front-view geometry where UCA and LCA are equal length and horizontal. At zero heave, camber = 0°, roll centre at ground level (Z=0). This validates the basic solver and angle extraction.

### Geometry (ISO 8855, left corner FL)

| Point | X (mm) | Y (mm) | Z (mm) |
|---|---|---|---|
| UCA inboard front | 50 | 200 | 300 |
| UCA inboard rear | -50 | 200 | 300 |
| UCA outboard | 0 | 600 | 300 |
| LCA inboard front | 50 | 200 | 100 |
| LCA inboard rear | -50 | 200 | 100 |
| LCA outboard | 0 | 600 | 100 |
| Tie rod inboard | 0 | 200 | 200 |
| Tie rod outboard | 0 | 600 | 200 |
| Wheel centre | 0 | 620 | 200 |

Tire loaded radius: 228 mm.

### Expected results at heave = 0

- Static camber: 0°
- Caster: 0° (all points in the Y-Z plane)
- KPI: 0° (UBJ directly above LBJ)
- Toe (per side): 0° (tie rod aligned with vehicle axis)

### Expected behaviour in heave

- Symmetric bump produces symmetric camber change.
- Equal-length parallel arms → zero camber gain (first-order).
- Roll centre stays at Z ≈ 0 (for small displacements).

## Case 2: Typical FSAE front corner

A realistic FSAE double-wishbone front-left corner. Validate against Optimum Kinematics or hand calculation.

### Geometry (ISO 8855, FL)

| Point | X (mm) | Y (mm) | Z (mm) |
|---|---|---|---|
| UCA inboard front | 80 | 150 | 280 |
| UCA inboard rear | -80 | 150 | 280 |
| UCA outboard | 0 | 530 | 290 |
| LCA inboard front | 100 | 130 | 80 |
| LCA inboard rear | -100 | 130 | 80 |
| LCA outboard | 0 | 580 | 75 |
| Tie rod inboard | -60 | 160 | 120 |
| Tie rod outboard | -50 | 540 | 110 |
| Wheel centre | 0 | 600 | 200 |

Tire loaded radius: 228 mm (10" rim, typical FSAE slick).

### Expected results (approximate)

- Static camber: ~ -1° to -2° (slight negative, typical FSAE)
- Caster: ~ 4° to 6° (typical FSAE)
- KPI: ~ 6° to 8° (moderate)
- Camber gain in bump: ~ -0.3 to -0.8 deg/25mm (gains negative camber in bump)

These values should be cross-validated with Optimum Kinematics exports.

## Case 3: Left-right symmetry check

Mirror Case 2 to create the FR corner by negating all Y coordinates. The solver must produce:
- Camber(FL) = Camber(FR) in magnitude (same sign by convention: negative = top inboard)
- Toe, caster, KPI equal in magnitude
- Roll centre contributions symmetric about Y=0

This test catches Y-sign errors in the angle extraction.
