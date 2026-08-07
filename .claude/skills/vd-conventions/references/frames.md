# Coordinate frame transform table

All transforms are 3×3 sign/permutation matrices. To convert a vector **from** the source frame **to** the target frame, multiply: `v_target = M @ v_source`.

## Frame definitions

| Frame | X+ | Y+ | Z+ | Handedness | Origin |
|---|---|---|---|---|---|
| **ISO 8855 (this project)** | Forward | Left | Up | Right | Front axle CL, ground, vehicle CL |
| **J670e (SAE z-down)** | Forward | Right | Down | Right | Front axle CL, ground, vehicle CL |
| **Legacy project frame** | Forward | Left | Up | Right | Front axle CL, ground, vehicle CL |
| **Optimum Kinematics** | Forward | Right | Up | Left | Varies (usually front axle) |
| **SolidWorks (typical FSAE)** | Rearward | Right | Up | Left | Varies (usually CG or rear axle) |

## Transform matrices

### ISO 8855 → J670e (SAE z-down)

Y and Z are negated (Y: left→right, Z: up→down).

```
M = [[ 1,  0,  0],
     [ 0, -1,  0],
     [ 0,  0, -1]]
```

### J670e → ISO 8855

Same matrix (self-inverse).

```
M = [[ 1,  0,  0],
     [ 0, -1,  0],
     [ 0,  0, -1]]
```

### ISO 8855 → Legacy project frame

The legacy `geometry/` package uses X+ forward, Y+ left, Z+ up — same axes as ISO 8855. The transform is identity. Note: the legacy code may have implicit sign assumptions in some functions; validate individual results.

```
M = [[ 1,  0,  0],
     [ 0,  1,  0],
     [ 0,  0,  1]]
```

### ISO 8855 → Optimum Kinematics

Optimum Kinematics uses X+ forward, Y+ right, Z+ up (left-handed). Negate Y.

```
M = [[ 1,  0,  0],
     [ 0, -1,  0],
     [ 0,  0,  1]]
```

### ISO 8855 → SolidWorks (typical FSAE setup)

SolidWorks FSAE models commonly use X+ rearward, Y+ right, Z+ up (left-handed). Negate X and Y.

```
M = [[-1,  0,  0],
     [ 0, -1,  0],
     [ 0,  0,  1]]
```

## Usage in code

Use `vdcore.io.frames` for executable transforms:
```python
from vdcore.io.frames import iso8855_to_j670e, j670e_to_iso8855
v_j670e = iso8855_to_j670e(v_iso8855)
```

## Verifying a transform

Every transform matrix must satisfy:
- `det(M)` is +1 (proper rotation/reflection) or -1 (improper — includes sign flips)
- `M @ M_inverse = I` (round-trip)
- For self-inverse matrices: `M @ M = I`
