# Tutorial: Designing FSAE Suspension with this Tool

This tutorial walks through the complete workflow for using the Suspension Geometry Engine to design, analyze, and optimize a Formula SAE suspension. It assumes you know what a double-wishbone suspension is but have never used this software before.

For installation, file format, and troubleshooting, see [README.md](README.md).

---

## Contents

1. [Understand what the tool does (and doesn't)](#1-understand-what-the-tool-does-and-doesnt)
2. [Prepare your hardpoints](#2-prepare-your-hardpoints)
3. [First look: load and inspect](#3-first-look-load-and-inspect)
4. [Read the static KPIs](#4-read-the-static-kpis)
5. [Run a heave sweep](#5-run-a-heave-sweep)
6. [Run roll and steer sweeps](#6-run-roll-and-steer-sweeps)
7. [Use the setup sheet](#7-use-the-setup-sheet)
8. [Optimize with the Synthesis tab](#8-optimize-with-the-synthesis-tab)
9. [Compare before and after](#9-compare-before-and-after)
10. [Edit hardpoints manually](#10-edit-hardpoints-manually)
11. [Use Python scripts instead of the UI](#11-use-python-scripts-instead-of-the-ui)
12. [Design workflow: putting it all together](#12-design-workflow-putting-it-all-together)
13. [Common mistakes and how to avoid them](#13-common-mistakes-and-how-to-avoid-them)

---

## 1. Understand what the tool does (and doesn't)

This is a **kinematics** tool. It answers one question: *given these hardpoint locations, what are the geometric consequences?*

It computes:

- **Static KPIs** — caster, KPI, camber, scrub radius, mechanical trail, roll centre height
- **Dynamic behavior** — how camber, toe, and roll centre change with wheel travel, body roll, and steering
- **Steering geometry** — Ackermann percentage, steer ratio, steer arm length
- **Side-view geometry** — anti-dive, anti-squat (simplified)

It does **not** compute anything that requires spring rates, damper curves, masses, or tire data. No wheel rates, no natural frequencies, no lap times, no load transfer. Those need different tools.

### The design philosophy

The tool shows consequences. It never recommends a value.

When you set a target in the optimizer, you are telling the tool *your* goal. It finds hardpoints that achieve it. Whether that goal makes sense for your car is your decision — the tool will not second-guess it.

This matters because suspension parameters are coupled. Increasing caster tends to increase trail. Raising the lower control arm inboard changes camber gain *and* roll centre height simultaneously. The tool makes these couplings visible so you can make informed trade-offs.

---

## 2. Prepare your hardpoints

A "hardpoint" is a pivot or attachment point on the suspension. Each corner of the car has 10:

```
 CHASSIS SIDE                    WHEEL SIDE

 UCA_IN_FRONT  ────────────────  UCA_OUT (upper ball joint)
 UCA_IN_REAR   ──────────────╱
                              ╲
 LCA_IN_FRONT  ──────────────╲
 LCA_IN_REAR   ────────────────  LCA_OUT (lower ball joint)

 TIE_ROD_IN    ────────────────  TIE_ROD_OUT

                                 WHEEL_CENTER
                                 CONTACT_PATCH (Z = 0)
```

### Where hardpoints come from

| Source | How to get them |
|---|---|
| **Existing CAD** | Measure each point in SolidWorks / Fusion 360. Use "Mass Properties" or "Measure" on the sketch point or center of a hole. |
| **Previous year's car** | Export from last year's geometry spreadsheet or CAD model. |
| **Starting from scratch** | Use the built-in demo template as a seed. It gives you realistic FSAE-range values to start iterating from. |
| **Literature** | Some textbooks (Milliken RCVD, Optimum G) list example geometries. Enter them manually. |

### Coordinate system

The software uses **ISO 8855**:

- **X+ forward**, **Y+ left**, **Z+ up**
- Origin at the front axle centerline, ground level, vehicle centerline

Before entering coordinates, verify:

| Check | Expected |
|---|---|
| `WHEEL_CENTER.z` | Tire loaded radius, ~220-260 mm (positive) |
| `CONTACT_PATCH.z` | 0 |
| `UCA_IN.z > LCA_IN.z` | The upper arm is higher than the lower arm |
| FL points have positive Y | Y+ is left |
| FR points have negative Y | Mirrored across the centerline |

If your CAD uses a different convention (Z-down, Y-right, origin at wheel center), convert before loading. The README section 14.5 covers common transformations.

### File format

A CSV with 5 columns and 40 rows (4 corners x 10 points):

```csv
corner,point,x_mm,y_mm,z_mm
FL,UCA_IN_FRONT,60,150,295
FL,UCA_IN_REAR,-70,150,295
...
```

The sidebar's "Template" button downloads a correctly formatted file you can fill in.

---

## 3. First look: load and inspect

Start the app:

```bash
streamlit run app.py
```

### Option A: Use the demo geometry

Click **"Demo"** in the sidebar. This loads a realistic FSAE geometry — good for learning the interface before using your own data.

### Option B: Load your file

1. In the sidebar, click **"Browse files"** and select your CSV/XLSX/JSON
2. A preview appears showing the point count and corners found
3. Click **"Apply file"** to load it into the session

### Option C: Start from the template

1. Click **"Template"** to download `hardpoints_template.csv`
2. Open it in a spreadsheet editor, replace the numbers with your data
3. Save and upload via Option B

After loading, the sidebar shows the filename and point count. You are ready to analyze.

---

## 4. Read the static KPIs

Go to the **Analysis** tab. Six cards appear at the top:

```
 ┌──────────┐  ┌──────────┐  ┌──────────┐
 │ Caster   │  │ KPI      │  │ Camber   │
 │  +5.2°   │  │  +7.1°   │  │  -1.5°   │
 └──────────┘  └──────────┘  └──────────┘
 ┌──────────┐  ┌──────────┐  ┌──────────┐
 │ Scrub    │  │ Trail    │  │ RC Height│
 │  +12 mm  │  │  +18 mm  │  │  +45 mm  │
 └──────────┘  └──────────┘  └──────────┘
```

### What to look for

| KPI | Typical FSAE range | Red flags |
|---|---|---|
| **Caster** | 3-7 deg | 0 deg means UCA_OUT and LCA_OUT have the same X coordinate |
| **KPI** | 5-10 deg | 0 deg means they have the same Y coordinate |
| **Camber** | -1 to -3 deg | Positive camber on a race car is unusual |
| **Scrub Radius** | -10 to +30 mm | > 100 mm means WHEEL_CENTER.y is wrong |
| **Mechanical Trail** | 5-25 mm | Negative trail causes steering instability |
| **RC Height** | 20-80 mm | Negative means the roll centre is underground |

If any value is wildly off, the hardpoints likely have a coordinate error. Check the troubleshooting in the README (section 14.3) before continuing.

### Selecting corners

Use the corner selector to switch between FL, FR, RL, RR. For a symmetric car, FL and FR should have the same magnitudes with appropriate sign changes.

---

## 5. Run a heave sweep

A heave sweep moves the wheel vertically relative to the chassis and measures what happens to camber, toe, and the roll centre. This is the single most informative analysis you can run.

1. In the Analysis tab, set **Sweep type** to **Heave**
2. Set the range: **-25 mm** to **+25 mm**, step **1 mm**
3. The sweep runs automatically (~2 seconds)

Three things appear:

### Camber vs. Heave chart

Shows how camber changes as the wheel moves up (bump) and down (rebound).

**What you want:** A roughly linear negative slope. As the wheel bumps into the chassis, camber should go more negative (top of wheel tilts inboard), compensating for body roll.

**The number that matters:** Camber gain, reported in **deg/mm**. Typical target: **-0.015 to -0.025 deg/mm**.

- Steeper (more negative) = more camber compensation in roll = better cornering grip retention
- But too steep = excessive camber change over bumps on straights

### Toe vs. Heave chart (bump steer)

Shows involuntary toe change as the wheel travels.

**What you want:** As flat as possible. Bump steer is generally undesirable — it means the car's toe changes when it hits a bump, causing unpredictable handling.

**The number that matters:** Bump steer, reported in **deg/mm**. Target: **< 0.005 deg/mm** in magnitude.

If bump steer is too high, the tie rod is not following the same arc as the control arms. Adjusting `TIE_ROD_IN` and `TIE_ROD_OUT` positions is the primary fix.

### Roll Centre migration chart

Shows the roll centre's Y and Z position as the wheel travels. The RC should stay relatively stable.

**What you want:** Small migration. Target: **deltaY < 30 mm**, **deltaZ < 30 mm** over the sweep range.

Large RC migration means the car's roll behavior changes significantly with wheel position — the car "feels different" at different ride heights.

---

## 6. Run roll and steer sweeps

### Roll sweep

Set sweep type to **Roll**. This rotates the chassis about X and shows how the suspension responds.

**Key output:** Roll camber, in **deg/deg**. This tells you how much camber the outside wheel gains per degree of body roll. Typical range: **-0.5 to -1.5 deg/deg**.

- -1.0 deg/deg means 1 degree of body roll costs 1 degree of camber on the outside wheel
- More negative = more camber compensation = better cornering

### Steer sweep

Set sweep type to **Steer**. This moves the rack laterally and shows caster and KPI variation with steering angle.

Useful for checking that caster and KPI remain stable through the steering range. Large variations indicate the kingpin axis geometry is sensitive to steer angle.

---

## 7. Use the setup sheet

The **Complete KPIs** tab (or the full setup sheet in the Analysis tab) shows every computed parameter in one place, organized by category:

- **Dimensions** — wheelbase, front/rear track width
- **Kinematics** — all static KPIs for each corner
- **Steering** — Ackermann percentage, steer ratio, steer arm lengths
- **Dynamic** — camber gain, bump steer, ride/roll camber, RC migration
- **Side view** — anti-dive, anti-squat

Each row shows the source: "calculated" (from hardpoints), "input" (from sidebar settings), or "derived" (from a sweep).

Use the category filter pills to focus on one area at a time. This is the view you would print for a Design Event judge or paste into a design report.

### Vehicle-level setup

Some KPIs need additional inputs from the sidebar:

| Sidebar input | What it affects |
|---|---|
| **Brake bias** | Anti-dive calculation |
| **Steering c-factor** (mm/rev) | Steer ratio |
| **Steering lock** (deg) | Maximum steer angle display |

Set these before reading the setup sheet if you care about those KPIs.

---

## 8. Optimize with the Synthesis tab

The optimizer finds hardpoints that meet your performance targets. It uses differential evolution — a global search algorithm that explores many geometries simultaneously.

### Step 1: Choose your seed

The optimizer starts from an existing geometry (the "seed") and searches nearby. Load your starting geometry first — either from a file, the demo template, or the manual editor.

The quality of the seed matters. A reasonable starting geometry converges faster and finds better solutions than a random one.

### Step 2: Define targets

The Synthesis tab has two sections:

**Static targets** — check the ones you care about and set values:

| Target | Example value | Notes |
|---|---|---|
| Caster | 4.5 deg | Self-centering feel |
| KPI | 7.0 deg | Camber change during steer |
| Camber | -1.5 deg | Cornering grip |
| Scrub radius | 15 mm | Steering effort |
| Mechanical trail | 18 mm | Steering sensitivity |

**Dynamic targets** — these are computed from a heave sweep during optimization:

| Target | Example value | Notes |
|---|---|---|
| Camber gain | -0.020 deg/mm | Camber compensation in roll |
| Bump steer | 0.002 deg/mm | Steering stability over bumps |
| RC height | 50 mm | Roll behavior |
| RC deltaY | 15 mm | Roll centre stability |

You do not need to enable every target. Start with the 3-4 that matter most for your car. More targets = more constrained = harder to satisfy simultaneously.

### Step 3: Set bounds

Expand **"Bounding Boxes"**. This controls how far each hardpoint can move from the seed.

- **Wide bounds** (+-50 to +-100 mm): more freedom, slower convergence, more likely to find a global optimum
- **Tight bounds** (+-10 to +-25 mm): faster convergence, may miss better solutions, useful when you are close to a good design and want to fine-tune

The optimizer moves 4 points (UCA_OUT, LCA_OUT, TIE_ROD_IN, TIE_ROD_OUT) — 12 degrees of freedom total. The inboard pickup points stay fixed because they are usually constrained by chassis packaging.

### Step 4: Configure the solver

| Parameter | Default | When to change |
|---|---|---|
| Population size | 15 | Increase to 20-30 for difficult multi-target problems |
| Max iterations | 60 | Increase to 100-200 if cost is still dropping when it stops |
| Seed (random) | 42 | Change to get a different solution from the same problem |
| Workers | -1 (all cores) | Leave at -1 unless your machine overheats |
| Polish | on | Refines the best solution with a local optimizer at the end |

### Step 5: Run and interpret

Click **"Run Optimization"**. Typical times:

- 60 iterations, population 15 = ~30 seconds
- 100 iterations, population 20 = ~2 minutes

A results table appears:

```
Parameter        Target    Seed     Optimized   Seed OK   Opt OK
Caster (deg)      +4.50    +8.88      +4.49       no       yes
KPI (deg)         +7.00    +4.47      +6.82       no       yes
Camber (deg)      -1.50    +0.00      -1.51       no       yes
Camber gain       -0.020   -0.012     -0.019      no       yes
```

- "yes" means the optimized value is within tolerance of the target
- "no" means it fell short — consider loosening that target or increasing its weight

**If multiple targets show "no":** the targets may be geometrically incompatible. This is real information — it tells you where the physics forces a trade-off. Decide which target to relax.

### Step 6: Apply and download

- **"Apply to session"** loads the optimized geometry into the app so you can analyze it further
- **"Mirror to opposite corner"** copies FL to FR (or vice versa) with the appropriate Y sign flip
- **"Download CSV"** saves the optimized hardpoints for import into your CAD tool

---

## 9. Compare before and after

The **Comparison** tab puts two geometries side by side:

1. Select **Source A** (e.g., "Last SEED geometry")
2. Select **Source B** (e.g., "Last OPTIMIZED geometry")

You see:

- **Static KPI table** with absolute values and deltas
- **Heave sweep overlay** — camber and toe curves from both geometries on the same axes
- **RC migration overlay** — how each geometry's roll centre moves

This is where you confirm the optimizer actually improved what you asked for without breaking something else. Pay attention to KPIs you did *not* target — they may have shifted.

---

## 10. Edit hardpoints manually

The **Manual Editor** tab (Inputs) lets you change individual coordinates and see the effect in real time.

### When to use manual editing

- **Learning:** move one point at a time and watch how KPIs change. This builds intuition for which hardpoints control which behaviors.
- **Packaging:** your chassis tube frame has a fixed node at (X, Y, Z) — type it directly.
- **Fine-tuning:** the optimizer got you close, now you want to round coordinates to integers or adjust for manufacturing.

### How it works

1. Select a corner (FL, FR, RL, RR)
2. Edit X, Y, Z values in the table
3. Three 2D views update live:
   - **Front view (YZ)** — shows arm heights, upright angle
   - **Side view (XZ)** — shows arm sweep, caster angle
   - **Top view (XY)** — shows arm convergence
4. Click **"Apply as loaded hardpoints"** to commit the changes to the session
5. Switch to the Analysis tab to see the resulting KPIs

### Mirror button

**"Mirror Left to Right"** copies the selected corner to its opposite side, negating Y coordinates. Use this to enforce symmetry after editing one side.

### Learning exercise: one point at a time

Try this to build intuition:

1. Load the demo template
2. Go to the Analysis tab, note the static KPIs for FL
3. Go to the Manual Editor, select FL
4. Change `LCA_OUT.z` by +10 mm (raise the lower ball joint)
5. Apply, go back to Analysis
6. Note which KPIs changed and by how much

Repeat for other points. You will quickly learn:

- Moving outboard points (UCA_OUT, LCA_OUT) has the largest effect on caster, KPI, and camber
- Moving TIE_ROD positions primarily affects bump steer
- The inboard points (on the chassis) affect camber gain and roll centre more than static alignment

---

## 11. Use Python scripts instead of the UI

Everything the Streamlit app does is available as Python functions. Scripts are useful for:

- **Batch analysis** — sweep 50 configurations overnight
- **Custom optimization** — define your own objective function
- **Reports** — generate data for a design report
- **Version control** — check your analysis scripts into git alongside your geometry files

### Load and analyze a corner

```python
from analysis.io_hardpoints import read_hardpoints, build_corner_from_dataframe
from geometry import KinematicSolver3D
from analysis.sweeps import SweepRunner, camber_gain_per_mm, bump_steer_per_mm

df = read_hardpoints("my_car.csv")
corner, tie_rod = build_corner_from_dataframe(df, "FL")

# Static KPIs
print(f"Caster:       {corner.static_caster_deg():+.2f} deg")
print(f"KPI:          {corner.static_kpi_deg():+.2f} deg")
print(f"Scrub radius: {corner.static_scrub_radius_mm():+.1f} mm")
print(f"Mech. trail:  {corner.static_mechanical_trail_mm():+.1f} mm")
print(f"RC height:    {corner.roll_center_height_mm():+.1f} mm")

# Heave sweep
solver = KinematicSolver3D(corner, tie_rod)
runner = SweepRunner(solver=solver)
sweep = runner.heave_sweep(-25.0, 25.0, 1.0)

print(f"Camber gain:  {camber_gain_per_mm(sweep):+.5f} deg/mm")
print(f"Bump steer:   {bump_steer_per_mm(sweep):+.5f} deg/mm")
```

### Build a full vehicle report

```python
from analysis.io_hardpoints import read_hardpoints, build_vehicle_from_dataframe
from analysis.kpis import build_full_report

df = read_hardpoints("my_car.csv")
vehicle, tie_rods = build_vehicle_from_dataframe(df)

report = build_full_report(
    vehicle, tie_rods,
    cg_height_mm=280.0,
    brake_bias_pct=60.0,
    drive_type="RWD",
    roll_stiffness_deg_per_g=1.5,
)

print(f"Wheelbase:    {report.wheelbase_mm:.1f} mm")
print(f"Track front:  {report.track_front_mm:.1f} mm")
print(f"Track rear:   {report.track_rear_mm:.1f} mm")
```

### Run an optimization from a script

```python
from analysis.io_hardpoints import read_hardpoints, build_corner_from_dataframe
from analysis.optimizer import (
    SuspensionOptimizer, DesignTargets, validate_against_targets,
)

df = read_hardpoints("my_car.csv")
corner, tie_rod = build_corner_from_dataframe(df, "FL")

targets = DesignTargets(
    caster_target_deg=4.5,
    kpi_target_deg=7.0,
    static_camber_target_deg=-1.5,
    camber_gain_target_deg_per_mm=-0.020,
    rc_height_target_mm=50.0,
    bump_steer_max_deg_per_mm=0.003,
    heave_step_mm=5.0,
)

opt = SuspensionOptimizer(
    seed_corner=corner,
    seed_tie_rod=tie_rod,
    targets=targets,
    population_size=15,
    max_iterations=100,
    workers=-1,
)
result = opt.run()
print(result.summary())

# Validate
report = validate_against_targets(
    result.optimal_corner, result.optimal_tie_rod, targets,
)
print(report.summary())
```

### Export results

```python
from analysis.io_hardpoints import dataframe_from_corner, save_dataframe

df_out = dataframe_from_corner(result.optimal_corner, result.optimal_tie_rod)
save_dataframe(df_out, "optimized_FL.csv")
```

---

## 12. Design workflow: putting it all together

Here is a complete workflow for designing the front suspension of a new FSAE car.

### Phase 1: Establish constraints

Before touching the tool, write down what is fixed:

- **Tire**: loaded radius (e.g., 228 mm for a 13" Hoosier)
- **Wheel**: offset, width, bolt pattern
- **Chassis**: frame node locations that constrain inboard pickups
- **Packaging**: minimum ground clearance, maximum track width, steering rack location

These become the fixed coordinates and bounds for the optimizer.

### Phase 2: Set performance goals

Decide what you are targeting and why. Document the source of each target:

| Target | Value | Source |
|---|---|---|
| Caster | 4-5 deg | Milliken RCVD Table 18.2 |
| KPI | 6-8 deg | Optimum G Tech Tip 14 |
| Static camber | -1.5 deg | Team preference from testing |
| Camber gain | -0.020 deg/mm | Milliken RCVD Ch. 17 |
| Bump steer | < 0.003 deg/mm | Team requirement |
| RC height | 40-60 mm | Compromise: low for CG, high enough for jacking force |

These are *your* targets, not the tool's. The tool will tell you what the geometry achieves; you decide if the achievement is good enough.

### Phase 3: First pass — optimizer

1. Load the demo template or last year's geometry as a seed
2. Open the Synthesis tab
3. Enter your targets
4. Set wide bounds (+-50 mm on outboard points)
5. Run with 100 iterations, population 20
6. Review results

If all targets are met: good, move to Phase 4.

If some targets conflict: this is useful information. You now know the trade-off frontier. Decide which target to relax and re-run.

### Phase 4: Verify with sweeps

1. Apply the optimized geometry
2. Run a heave sweep (-30 to +30 mm)
3. Run a roll sweep (-3 to +3 deg)
4. Check the full setup sheet

Look at KPIs you did *not* target. The optimizer only minimizes what you asked about — everything else is free to drift.

### Phase 5: Package check

Open the Manual Editor and verify that the optimized points are physically realizable:

- Are the inboard pickups close to actual chassis nodes?
- Is there clearance between the tie rod and the control arms?
- Does the upright geometry make sense (ball joint spacing, tie rod location)?
- Can you actually build a control arm that reaches from inboard to outboard without interfering with the tire?

The tool does not check packaging. This is a visual and CAD exercise.

### Phase 6: Fine-tune

Adjust individual points in the Manual Editor to accommodate packaging realities. After each change, check the Analysis tab to verify KPIs are still acceptable.

Round coordinates to values that are practical to manufacture (whole millimeters, or whatever your chassis jig can achieve).

### Phase 7: Mirror and check the rear

1. Mirror FL to FR
2. Repeat the entire process for the RL corner
3. Mirror RL to RR
4. Check the full vehicle setup sheet

Verify:

- Front and rear track widths are as intended
- Wheelbase matches your chassis design
- Roll axis inclination is reasonable (the line from front RC to rear RC)
- Front and rear roll camber are in the desired ratio

### Phase 8: Export and document

1. Download the final CSV
2. Import the coordinates into your CAD model
3. Save the setup sheet (screenshot or export) for your design report

---

## 13. Common mistakes and how to avoid them

### Wrong coordinate system

**Symptom:** KPIs are nonsensical (caster = 0, RC height negative, scrub radius > 100 mm).

**Fix:** Verify your coordinate system matches ISO 8855. The most common issues:
- Z-down instead of Z-up: negate Z
- Y-right instead of Y-left: negate Y
- Origin at wheel center instead of ground: add tire radius to Z

### Targeting everything at once

**Symptom:** Optimizer runs a long time and still shows "no" on multiple targets.

**Fix:** Start with 2-3 targets. Add more only after the first pass succeeds. Suspension geometry has finite degrees of freedom — you cannot independently control every parameter.

### Ignoring untargeted KPIs

**Symptom:** Optimizer hits all targets, but bump steer is 0.05 deg/mm (10x too high) because you did not target it.

**Fix:** Always run a full sweep after optimization and check the complete setup sheet. If an untargeted KPI drifts too far, add it to the next optimization run.

### Not checking both sides

**Symptom:** FL is perfect, FR has different KPIs.

**Fix:** After optimizing one side, use the mirror function. Then verify the mirrored side in the Analysis tab — a correct mirror should produce identical magnitudes.

### Trusting the optimizer blindly

**Symptom:** The optimizer finds a geometry with great numbers, but the hardpoints are physically impossible (arm goes through the wheel, tie rod intersects the brake disc).

**Fix:** Always verify the optimized geometry in the 3D View tab and in your CAD tool. The optimizer has no awareness of physical interference.

### Over-constraining the optimizer bounds

**Symptom:** Optimizer converges immediately to the seed geometry (cost = 0 from the start) or produces trivially different results.

**Fix:** Widen the bounds. If an outboard point can only move +-5 mm, the optimizer has almost no room to work. Start with +-50 mm and tighten only after you know which direction the solution goes.

### Confusing per-side and total toe

**Symptom:** Toe values seem doubled or halved compared to expectations.

**Fix:** The tool reports per-side toe and sum toe separately. Make sure you are reading the right one. Per-side toe is the angle of one wheel; sum toe (total toe) is left + right. This distinction has caused real-world setup errors — always be explicit.
