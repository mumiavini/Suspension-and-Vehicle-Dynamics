# hardpoints_2026_car.csv — provenance

Reference geometry of the **2026 car** (the previous car, not the FSAE26/2027 design).
Kept for comparison only; nothing in the repo reads it.

- **Source:** `c:\soft\FSAE\carro_formula.xlsx` (outside this repo, last modified
  2026-05-18), exported to `carro_formula.csv` by `c:\soft\FSAE\sketch.ipynb`.
  Whether the points came from CAD or from the car is not recorded. Treat them as
  `estimate`.
- **Right side is mirrored.** Only FL and RL in the source are real. Its FR and RR
  rows were the unedited demo values from `hardpoints_template.csv`, so FR/RR here
  are FL/RL with Y negated. This assumes the car was symmetric.
- **No static camber recorded.** CONTACT_PATCH sits directly under WHEEL_CENTER, so
  the file describes 0° camber. The loaded radius implied by the file is 262.5 mm.
- **Frame:** ISO 8855 (X+ forward, Y+ left, Z+ up), origin at front axle, ground,
  centreline. Same as `carro_formula_2027.csv`.
- **Carry-over:** the 2027 rear toe-link outboard point (RL `TIE_ROD_OUT`
  -1594.60, 520.81, 183.53) is this car's point, copied unchanged.

Headline dimensions: track 1145 mm (front and rear), wheelbase 1525 mm.

Static KPIs (vdcore `DWSolver` static solve at 0° camber, r = 262.5 mm; matches the
front-view construction to 0.01 mm):

| axle  | KPI     | caster  | scrub radius | mech. trail |
|-------|---------|---------|--------------|-------------|
| front | 12.15°  | -0.07°  | 24.46 mm     | 2.42 mm     |
| rear  | 11.54°  | 5.06°   | 24.09 mm     | 11.97 mm    |

Scrub is positive when the contact patch is outboard of the kingpin ground point.
Each degree of negative static camber adds 4.58 mm to it.
