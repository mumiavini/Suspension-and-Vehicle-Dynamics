# Ultimos conceitos: rocker concepts X, Y, Z in the car

Three suspension layouts for the FSAE 2027 car, each built in SolidWorks inside a copy of the
car (chassis Xasis 33.3 + engine). Every layout uses the team's real damper (Amortecedor) and
wheel (RODA ESTRELA FORMULA V1).

This folder is **self-contained**: zip it, send it, unzip it anywhere and the assemblies open.
It does not need the rest of the repository.

## How to open

- Use **SolidWorks 2025 or newer**. These files were saved in 2025, and older versions cannot open them.
- Open one of the cars: `MONT_CHASSI_SUSP_X.SLDASM`, `MONT_CHASSI_SUSP_Y.SLDASM` or `MONT_CHASSI_SUSP_Z.SLDASM`.
  - If SolidWorks says the references were found in another folder, accept it. That message is
    expected after unzipping somewhere else.
- **Open one concept at a time.** `SUSP_X`, `SUSP_Y` and `SUSP_Z` use the same file names
  (`SUSP_FL_UCA.SLDPRT`, `AMORTECEDOR_CORPO.SLDPRT`, ...). SolidWorks keeps only one document of a
  given name open, so close one concept before opening the next.
- Keep the folder structure as it is. The assemblies find their parts by relative path.

## What is in the folder

| Path | What it is |
|---|---|
| `MONT_CHASSI_SUSP_<c>.SLDASM` | The car with concept `<c>`: chassis, engine and the suspension assembly |
| `SUSP_<c>\SUSP_<c>.SLDASM` | Concept `<c>`'s suspension: four corner assemblies (`SUSP_FL/FR/RL/RR`), the rack and a centreline skeleton |
| `SUSP_<c>\SUSP_<k>_*.SLDPRT` | The corner's parts: wishbones, upright, tie rod / toe link, push/pull rod, rocker, chassis mounts. Each is made of editable features: a 3D-sketch centreline and a circular sweep whose diameter is a dimension |
| `SUSP_<c>\AMORTECEDOR_CORPO / _HASTE.SLDPRT` | The real damper, split in two so it slides: the body (reservoir, spring, chassis eye) and the shaft (rocker eye, rod) |
| `SUSP_<c>\RODA_ESTRELA.SLDPRT` | The real wheel (tyre + rim), a copy with the two reference points the corners mate to |
| `Chassis and engine\` | Copies of `Xasis 33.3.SLDPRT` and `MOTOR_FIXADO.SLDPRT` used by the three cars |
| `Source parts\` | The original damper and wheel files these copies were made from (not used by the assemblies) |
| `Macros\` | The SolidWorks macros (.bas) each suspension was generated from (`assembly` = this mated build, `solids` = one part with all the bodies) |
| `_gerador\` | The Python scripts that built and checked all of this (see the end) |

Each corner moves. The wheel travels and the rocker, rod and damper follow, so you can drag a
wheel up and down in SolidWorks.

## The three concepts

All three use 550 lb/in springs at a motion ratio of 0.506 at ride, front and rear. MR is a
comparison choice, not a decision. Kinematic hardpoints are the 2027 project's.

| | Front | Rear |
|---|---|---|
| **X** | Pushrod from the LCA to a rocker just **ahead of the front hoop**, low on the front triangle (z 545-615 mm). The coil-overs lie across the car and end on a bracket off the front face of the hoop's top bar. | R1: pushrod; the coil-over runs forward along the upper side rail |
| **Y** | Pullrod from the UCA; the coil-overs lie low and across the car, ahead of the rack | R2: pullrod, low rocker; the coil-over goes up outboard of the chain line |
| **Z** | Pullrod; the rocker sits on the front hoop's side node and the coil-over goes up to its upper corner | R3: pushrod; the coil-over runs rearward above the rear box |

The front and rear are independent, so a front from one concept works with a rear from another.

## What was checked, and what to know

- **Wheel travel.** Each concept was swept over +-36 mm of heave with the real damper.
  - The damper keeps about 8 mm before bottoming at +36 mm and 10 mm before full extension at -36 mm.
  - The +-36 mm limit assumes a bump stop and a droop limiter. Neither is in the model.
- **Concept X front** was moved twice: from over the hoop, to ahead of it, then lower onto the frame.
  - The current layout is clean in the static check.
  - The evaluator's packaging checks, over +-36 mm with full steer, also pass.
  - The SolidWorks heave sweep was run on the previous (higher) X front, not on this one.
- **Concepts Y and Z: damper clocking.** The damper body hits the frame unless it is turned
  about its own axis. That means rotating the chassis bracket; it assumes a spherical bearing at
  the chassis eye.
  - At ride height, the angles that clear the frame are below. Angle 0 is the damper as built:
    eye bolt parallel to the rocker axis, reservoir toward the part's -X.

    | | FL | FR | RL | RR |
    |---|---|---|---|---|
    | Y | 15-345 | 15-345 | 15-255 | 15-285 |
    | Z | 60-345, 0 | 135-180, 225-345, 0 | 45-345, 0 | 165-330 |

  - These angles are **not set in the models**. The solver puts the damper back on rebuild, so
    turn and fix it by hand.
  - Over travel they were not checked at the turned angle.
- **Rear wheel vs. rear upper ball joint.** In all three concepts, the rim's inboard lip overlaps
  the rear upper ball-joint housing by about 6.5 mm (613 mm³ per side).
  - The housing is a 34 mm placeholder, but the 2027 rear upper ball joint really is at the rim's
    inner edge.
  - Moving it about 10-20 mm inboard or down clears it. That changes the rear kinematics, so it
    has not been done.
- **Placeholders.** Wishbones, uprights, rockers, rods, brackets and chassis mounts are
  placeholder cylinders on the project's points: a modelling aid, not detailed or structural
  parts.
  - Bracket couples and node distances are flagged in the concept report.
  - Brackets on the front hoop are the chassis team's and the rules' call.

The full numbers (loads, motion ratios, every clearance, bracket lengths) are in the repository's
`projects/rocker_concepts/concept_xyz.md`.

## Regenerating (only with the repository)

The scripts in `_gerador\` drive SolidWorks through COM. They need Windows, SolidWorks and Python
with pywin32.

1. `build_susp.py <Macros\...assembly...bas> <SUSP_c folder> <Source parts\Amortecedor.SLDPRT>` builds a concept from its macro.
2. `place_in_car.py <c>` puts it into a copy of the car. It needs the repository's `MONT_CHASSI_SUSP_MOTOR` folder.
3. `swap_wheel.py <c> "<Source parts\RODA ESTRELA FORMULA V1.SLDPRT>"` replaces the simplified wheel.
4. `make_portable.py` re-points the cars to `Chassis and engine\` and drops outside references.
5. These check the result: `interference.py`, `travel_check.py`, `travel_report.py` and `clock_sweep.py`.

The macros themselves come from the repository's `scripts/export_suspension_assembly.py`.
