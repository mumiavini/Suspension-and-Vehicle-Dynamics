# Inboard pivots moved off the frame nodes

Source: `FSAE_2027.vdproj.json`, by `scripts/pivot_standoff_study.py`. Every number is solved on `DWSolver` with the ball joints, tie-rod ends and wheel centres where they are, so the static alignment (camber, toe, caster, KPI, scrub, trail, track) is the same in every column. Roll-centre heights are chassis-referenced.

- **leg D**: each inboard wishbone pivot D mm along its own leg toward the ball joint (the leg is D mm shorter, on the same line).
- **lateral D**: each inboard pivot D mm outboard in Y only.
- **leg D + toe link**: leg D, plus the rear toe link's inboard end D mm along the toe link. It changes nothing at the front.

A pivot slid along its leg stays in the wishbone's plane, and a ball joint moves perpendicular to that plane about any axis lying in it: the first-order motion at ride is the same and only the curvature changes. A lateral shift tilts the plane.

## Front axle

First order, at ride:

| Quantity | as is | leg 30 | leg 35 | lateral 35 | leg 35 + toe link |
|---|---|---|---|---|---|
| Roll centre height, static [mm] | 33.61 | 33.61 | 33.61 | 37.73 | 33.61 |
| FVSA length [mm] | 1491.2 | 1491.2 | 1491.2 | 1324.5 | 1491.2 |
| Camber gain [deg/mm] | -0.0386 | -0.0386 | -0.0386 | -0.0435 | -0.0386 |
| Roll camber (chassis-ref.) [deg/deg] | -0.418 | -0.418 | -0.418 | -0.471 | -0.418 |
| Half-track change [mm/mm] | 0.055 | 0.055 | 0.055 | 0.061 | 0.055 |
| Bump steer, linear (per side) [deg/mm] | -0.0020 | -0.0020 | -0.0020 | -0.0082 | -0.0020 |
| Anti-dive (front) / anti-squat (rear), 3D virtual work [%] | 6.93 | 6.93 | 6.93 | 6.36 | 6.93 |

Curvature, over the travel:

| Quantity | as is | leg 30 | leg 35 | lateral 35 | leg 35 + toe link |
|---|---|---|---|---|---|
| RC migration (chassis-ref.) [mm/mm] | -0.278 | -0.404 | -0.426 | -0.426 | -0.426 |
| RC height at -36 / +36 mm (chassis-ref.) [mm] | 44.4 / 24.3 | 48.9 / 19.9 | 49.8 / 19.1 | 53.9 / 23.3 | 49.8 / 19.1 |
| RC lateral at 1.5 deg of roll [mm] | 81.9 | 119.3 | 126.0 | 111.8 | 126.0 |
| RC height at 1.5 deg of roll [mm] | 33.16 | 32.50 | 32.36 | 36.66 | 32.36 |
| Outer wheel camber at 1.5 deg of roll (road) [deg] | -0.652 | -0.655 | -0.655 | -0.735 | -0.655 |
| Camber at -36 / +36 mm [deg] | -0.229 / -3.023 | -0.239 / -3.035 | -0.241 / -3.037 | -0.066 / -3.216 | -0.241 / -3.037 |
| Toe change at -36 / +36 mm (per side) [deg] | 0.284 / 0.138 | 0.163 / 0.018 | 0.141 / -0.004 | 0.359 / -0.238 | 0.141 / -0.004 |
| Peak |toe change| over +-36 mm (per side) [deg] | 0.284 | 0.163 | 0.141 | 0.359 | 0.141 |
| Half-track change at -36 / +36 mm [mm] | -3.32 / 0.64 | -3.45 / 0.51 | -3.48 / 0.49 | -3.72 / 0.73 | -3.48 / 0.49 |
| Motion ratio at -36 / 0 / +36 mm (the project's rocker) [-] | 0.477 / 0.506 / 0.506 | 0.471 / 0.503 / 0.508 | 0.470 / 0.502 / 0.508 | 0.470 / 0.503 / 0.509 | 0.470 / 0.502 / 0.508 |

## Rear axle

First order, at ride:

| Quantity | as is | leg 30 | leg 35 | lateral 35 | leg 35 + toe link |
|---|---|---|---|---|---|
| Roll centre height, static [mm] | 53.77 | 53.77 | 53.77 | 59.63 | 53.77 |
| FVSA length [mm] | 1397.5 | 1397.5 | 1397.5 | 1260.9 | 1397.5 |
| Camber gain [deg/mm] | -0.0412 | -0.0412 | -0.0412 | -0.0457 | -0.0412 |
| Roll camber (chassis-ref.) [deg/deg] | -0.432 | -0.432 | -0.432 | -0.479 | -0.432 |
| Half-track change [mm/mm] | 0.090 | 0.090 | 0.090 | 0.100 | 0.090 |
| Bump steer, linear (per side) [deg/mm] | 0.0020 | 0.0020 | 0.0020 | 0.0066 | 0.0020 |
| Anti-dive (front) / anti-squat (rear), 3D virtual work [%] | -1.21 | -1.21 | -1.21 | -4.87 | -1.21 |

Curvature, over the travel:

| Quantity | as is | leg 30 | leg 35 | lateral 35 | leg 35 + toe link |
|---|---|---|---|---|---|
| RC migration (chassis-ref.) [mm/mm] | -0.283 | -0.380 | -0.397 | -0.381 | -0.397 |
| RC height at -36 / +36 mm (chassis-ref.) [mm] | 64.3 / 43.9 | 67.9 / 40.5 | 68.6 / 39.8 | 73.8 / 46.3 | 68.6 / 39.9 |
| RC lateral at 1.5 deg of roll [mm] | 48.1 | 65.1 | 68.1 | 58.5 | 68.1 |
| RC height at 1.5 deg of roll [mm] | 53.47 | 53.19 | 53.13 | 59.12 | 53.14 |
| Outer wheel camber at 1.5 deg of roll (road) [deg] | -0.658 | -0.661 | -0.661 | -0.731 | -0.661 |
| Camber at -36 / +36 mm [deg] | -0.071 / -3.046 | -0.081 / -3.058 | -0.083 / -3.061 | 0.079 / -3.222 | -0.083 / -3.061 |
| Toe change at -36 / +36 mm (per side) [deg] | -0.024 / 0.118 | 0.101 / 0.242 | 0.124 / 0.265 | -0.043 / 0.433 | -0.022 / 0.120 |
| Peak |toe change| over +-36 mm (per side) [deg] | 0.118 | 0.242 | 0.265 | 0.433 | 0.120 |
| Half-track change at -36 / +36 mm [mm] | -4.65 / 1.87 | -4.76 / 1.76 | -4.78 / 1.74 | -5.11 / 2.12 | -4.78 / 1.75 |
| Motion ratio at -36 / 0 / +36 mm (the project's rocker) [-] | 0.491 / 0.506 / 0.512 | 0.481 / 0.506 / 0.525 | 0.479 / 0.506 / 0.527 | 0.501 / 0.530 / 0.551 | 0.479 / 0.506 / 0.527 |

## Not covered

- The pivot brackets themselves (tabs, bolts, bearing housings) and their fit next to the rocker and coil-over brackets that sit near the same nodes.
- Loads: a shorter pivot base raises the leg forces for the same wheel load; no structural number is computed here.
- The motion-ratio rows use the project's rocker as it is: a rocker is re-tuned to its target MR after the hardpoints move (`scripts/rocker_concepts.py`).
- The side-view construction KPI (`anti`) uses the pivot-axis rake and moves a little with a leg slide; the 3D virtual-work figure (`anti_3d`) is the one tabulated.
