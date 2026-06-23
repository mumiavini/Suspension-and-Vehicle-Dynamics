# 🏎️ FSAE Suspension Geometry Engine

Python software to **design, analyze and optimize** the suspension geometry of a Formula SAE car.

It works in two modes:

- **Analysis** — you have a car (or CAD assembly) and want to understand how it behaves
- **Synthesis** — you define behavior targets and the software finds the hardpoints that meet them

Interface via **Streamlit** (recommended) or **Python scripts** (for automation).

---

## 📑 Table of contents

1. [Who it's for](#1-who-its-for)
2. [What the software DOES and DOESN'T do](#2-what-the-software-does-and-doesnt-do)
3. [Why weight/engine/spring are NOT needed for most KPIs](#3-why-weightenginespring-are-not-needed-for-most-kpis)
4. [Physics concepts](#4-physics-concepts)
5. [Conventions and units](#5-conventions-and-units)
6. [Installation](#6-installation)
7. [Getting started (5 min)](#7-getting-started-5-min)
8. [App tabs tour](#8-app-tabs-tour)
9. [Complete tutorial: from CAD to optimized](#9-complete-tutorial-from-cad-to-optimized)
10. [Hardpoints file format](#10-hardpoints-file-format)
11. [Project structure](#11-project-structure)
12. [Python usage (scripts)](#12-python-usage-scripts)
13. [Complete KPI list](#13-complete-kpi-list)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitations](#15-limitations)
16. [Glossary](#16-glossary)

---

## 1. Who it's for

It was designed for:

- **FSAE suspension engineer** who needs to iterate geometries fast
- **Student** wanting to understand the effect of each hardpoint
- **Formula SAE team** that wants to document/version their choices
- **SolidWorks users** who want a calculation engine decoupled from the CAD

**It does NOT replace:**
- Structural analysis (needs FEA)
- Full vehicle-dynamics simulation (use OptimumLap, CarMaker)
- Experimental validation (k-rig, track testing)

---

## 2. What the software DOES and DOESN'T do

### ✅ DOES — Kinematics and geometry

| Category | Computed KPIs |
|---|---|
| **Static** | Caster, KPI, Camber, Scrub Radius, Mechanical Trail, RC Height, Kingpin Offset @ WC |
| **Dimensions** | Wheelbase, Track Width F/R |
| **Steering** | Steer Arm Length, Ackermann %, Rack/degree, Steer Ratio (with c-factor input) |
| **Dynamic** | Camber Gain (°/mm), Ride Camber (°/m), Roll Camber (°/°), Bump Steer (°/mm) |
| **Roll Center** | ΔY/ΔZ migration during sweeps, RC @ 1g lateral (approximation) |
| **Side view** | Anti-dive %, Anti-squat % (simplified version) |
| **State** | Static Sum Toe |
| **Synthesis** | Global optimization with static + dynamic targets, bounding boxes, validation |

### ❌ DOESN'T DO (currently) — Dynamics and structure

| Category | Why it's missing |
|---|---|
| **Wheel Rate** (N/mm) | Needs **spring stiffness** + motion ratio |
| **Roll Rate** (Nm/°) | Wheel rate + ARB + track |
| **Sprung Mass Frequency** (Hz) | Wheel rate + **sprung mass** |
| **Motion Ratio** | Needs to model the **pushrod/pullrod/rocker** |
| **Jounce/Rebound Damping** | F×v curve of the **damper** |
| **FEA of the arms** | A different kind of software (Ansys, etc.) |
| **Lap time simulation** | A different kind of software (OptimumLap) |

---

## 3. Why weight/engine/spring are NOT needed for most KPIs

This is a common question, and the answer is important:

### 🟢 Pure kinematics — geometry only

Most KPIs depend **only on the hardpoint positions** and on **how they move**:

```
Caster, KPI, Camber, Scrub, Trail, Roll Center
Camber Gain, Bump Steer, Ride/Roll Camber
Ackermann %, Steer Arm Length
```

These parameters are **invariant with respect to mass**. A 200 kg car and a 300 kg car with the same hardpoint geometry will have the same Caster, same Camber, same Ackermann.

Mass only matters to:
- Compute the **natural frequency** (needs wheel rate × mass)
- Compute **absolute loads** (for FEA)
- Compute **load transfer**

The software computes none of these three.

### 🟡 Approximations that use external parameters (with reasonable defaults)

**Anti-dive / Anti-squat:**
- Formula: `tan(θ_SVIC) × wheelbase/cg_height × brake_bias × 100`
- Needs the **CG height** and **brake bias** → you provide them in the vehicle setup in the sidebar
- **Does NOT need the absolute weight** — only the relative CG position

**Roll Center @ 1g lateral:**
- Formula: applies an equivalent 1g roll and measures where the RC ends up
- Needs the **roll stiffness** (degrees per g) → user input, default 1.5 °/g
- This value depends on spring+ARB+track, but as an approximation the typical FSAE value is accepted

### 🔴 KPIs that require external data (future)

If you want **wheel rate, roll rate, motion ratio, natural frequency, damping** — I need to add:

1. **Pushrod/rocker** model (motion ratio)
2. **Spring stiffness** as input
3. **Sprung mass** as input
4. **Damper curves** as input

That would multiply the size of the project. That is why the current scope is **only kinematics** — which already covers roughly 70% of the KPIs on a typical FSAE setup sheet.

### Table summary

| You need to... | Does the software compute it TODAY? |
|---|---|
| Move hardpoints and see Caster/KPI/Camber/etc. | ✅ Yes |
| Optimize geometry for targets | ✅ Yes |
| See Ackermann, Steer Ratio | ✅ Yes |
| Anti-dive/squat (needs CG and brake bias) | ✅ Yes, with inputs |
| RC @ 1g (needs roll stiffness) | ✅ Yes, approximate |
| Natural frequency, wheel rate, damping | ❌ No |
| FEA, stress analysis | ❌ No |
| Lap time simulation | ❌ No |

---

## 4. Physics concepts

### 4.1 Hardpoints

**Hardpoints** are the suspension's pivot/attachment points. Defining the hardpoints is defining how the car behaves.

```
┌─────── CHASSIS ───────┐
│                       │
│  UCA_IN_FRONT  ●─────────●  UCA_OUT (on the upright)
│  UCA_IN_REAR   ●──────╱
│                       │ ╲
│  LCA_IN_FRONT  ●──────╲    ● (upright)
│  LCA_IN_REAR   ●─────────●  LCA_OUT
│                       │
│  TIE_ROD_IN    ●─────────●  TIE_ROD_OUT
└───────────────────────┘
                            ●  WHEEL_CENTER (wheel center)
                            │
                            ●  CONTACT_PATCH (tire-ground contact)
```

Each **corner** (FL, FR, RL, RR) has **10 hardpoints**.

### 4.2 Typical FSAE values

**Static:**

| Parameter | Typical value | What it affects |
|---|---|---|
| **Caster** | 3° to 7° | Steering self-centering |
| **KPI** | 5° to 10° | Camber variation during steer |
| **Camber** | −1° to −3° | Cornering grip |
| **Scrub Radius** | −10 to +30 mm | Steering effort |
| **Mechanical Trail** | 5 to 25 mm | Steering sensitivity |
| **RC Height** | 20 to 80 mm | Chassis roll |

**Dynamic:**

| Parameter | Typical target |
|---|---|
| **Camber Gain** | −0.015 to −0.025 °/mm |
| **Bump Steer** | < 0.005 °/mm in magnitude |
| **RC ΔY** | < 30 mm of lateral migration |
| **Roll Camber** | −0.5 to −1.5 °/° |
| **Anti-dive** | 0 to 30% |
| **Ackermann** | 30% to 100% |

### 4.3 Coupling of the parameters

⚠️ All these parameters are **geometrically coupled** — you cannot change one without affecting the others:

- Increasing Caster → tends to increase Trail
- Increasing KPI → reduces Scrub Radius
- Raising the LCA inboard → affects Camber Gain AND RC at the same time

That is why **global optimization** (Synthesis tab) is more effective than trial-and-error.

---

## 5. Conventions and units

### 5.1 Axis system (SAE J670)

```
                 Z (up, positive)
                 ▲
                 │
                 │
                 ●─────────► Y (vehicle left, positive)
                /
               /
              ▼
              X (vehicle front, positive)
```

- **Origin:** center of the front axle, at ground level
- **X+** = front · **Y+** = left · **Z+** = up

> ⚠️ If your CAD origin is not SAE, **transform** the coordinates before entering them.
> Common signs of a problem:
> - `WHEEL_CENTER.z` ≈ 0 → your Z is not "height above ground" (it should be the tire radius)
> - `LCA_IN.z > UCA_IN.z` → your Z points downward

### 5.2 Signs

| Parameter | + means |
|---|---|
| **Camber** | Top of the wheel OUTWARD (negative = racing) |
| **Caster** | Top of the kingpin BEHIND the base |
| **KPI** | Top of the kingpin INWARD |
| **Scrub** | Kingpin crosses the ground INWARD of the contact |
| **Heave** | Bump (wheel rises relative to the chassis) |
| **Roll** | Chassis rolls to the RIGHT |
| **Rack** | Rack moves to the LEFT |

### 5.3 Units

- Lengths: **mm**
- Angles: **degrees (°)**
- Camber gain: **°/mm**
- Ride camber: **°/m**

No inches or radians. Convert beforehand (1 in = 25.4 mm).

---

## 6. Installation

### 6.1 Prerequisites

- Python 3.10+ ([download](https://www.python.org/downloads/))
- ~500 MB free

### 6.2 Step by step

**1. Download the files** (folder `fsae_suspension_clean/`)

**2. Open a terminal in the project folder**

**3. Virtual environment (recommended):**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**4. Install dependencies:**
```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
numpy>=1.24
scipy>=1.10
plotly>=5.0
polars>=0.20
openpyxl>=3.0
fastexcel>=0.10
streamlit>=1.30
```

**5. Test:**
```bash
python -c "from geometry import Point3D; print(Point3D(1,2,3))"
```

### 6.3 Running the app

```bash
streamlit run app.py
```
Opens in the browser at `http://localhost:8501`.

---

## 7. Getting started (5 min)

```
1. Terminal:                    streamlit run app.py
2. Sidebar → "📋 Demo"
3. "📊 Analysis" tab
4. See the FL KPIs appear
5. "Sweep type" → "Heave"
6. Charts appear automatically
```

---

## 8. App tabs tour

The app has **5 areas**:

### 📂 Sidebar — Hardpoints and Setup

```
┌─ Hardpoints ─────────────────┐
│ 1. Load file                 │  ← upload of .csv/.xlsx/.json
│ [file.csv shown here]        │
│ ✅ '...' — 40 points        │  ← preview
│ [🔄 Apply file]              │  ← click here to apply
│                              │
│ Or use:                      │
│ [📋 Demo] [⬇️ Template]      │
│                              │
│ ─────────────                │
│ 📊 In use: file.csv          │
│ [🗑️ Clear]                  │
│                              │
│ ─────────────                │
│ ⚙️ Vehicle setup             │
│ Brake bias front:    0.60    │  ← affects anti-dive
│ ▼ Steering                   │
│   c-factor:          100 mm  │  ← affects steer ratio
│   Steering lock:     270°    │
└──────────────────────────────┘
```

**Recommended flow:**

1. **Load a file** OR click **Demo** OR edit manually (tab 5)
2. Check the preview, click **🔄 Apply file**
3. Adjust the **Vehicle setup** if you want correct anti-dive/steer ratio
4. Go to the tabs

### 📊 Tab 1: Analysis

Shows the KPIs and charts of the selected corner.

- **6 cards at the top:** Caster, KPI, Camber, Scrub, Trail, RC Height
- **Sweep selection:** Heave / Roll / Steer with inline configuration
- **Dynamic KPIs:** Camber Gain, Bump Steer, RC ΔY/ΔZ
- **Plotly charts:** Camber vs Heave, Δ Toe vs Heave, RC trajectory
- **"Full data" expander:** table with all the sweep points

### 🎯 Tab 2: Synthesis / Optimization

Reverse engineering — you define goals and the software finds the hardpoints.

**Structure:**

```
┌─ Target definition ─────────────────────────┐
│ STATIC                      DYNAMIC         │
│ ☑ Caster        4°          Camber Gain     │
│ ☑ KPI           7°          Bump Steer max  │
│ ☑ Camber       -1.5°        RC Height       │
│ ☐ Scrub                     RC ΔY max       │
│ ☐ Trail                                     │
└─────────────────────────────────────────────┘

▼ ⚙️ Objective-function weights (advanced)
▼ 📦 Bounding Boxes / Keep-out zones
▼ 🔧 Evolutionary solver configuration

[🚀 Run Optimization]

────────────────────────────────────────
Table: Target × Seed × Optimized
Table: optimized hardpoints
[⬇️ Download CSV]
```

### 🔄 Tab 3: Comparison

Compares two geometries (A vs B) side by side.

- Sources: file / seed / last optimization
- Table with static KPIs and Δ
- Overlaid charts

### ✏️ Tab 4: Manual Editor

Manual editing of the hardpoints with live 2D visualization.

```
┌─────────────────────┬─────────────────────┐
│ Editable table      │ 3 2D views:         │
│ (10 pts per corner) │  - YZ (front)       │
│                     │  - XZ (side)        │
│                     │  - XY (top)         │
│                     │ Updates in real     │
│                     │ time as you type    │
└─────────────────────┴─────────────────────┘
[✅ Apply as hardpoints]  [⬇️ Download CSV]
[🪞 Mirror FL→FR]
[📋 Load template into this corner]
[🔁 Reload from file]
```

### 📋 Tab 5: Complete KPIs

Complete report in the car's setup-sheet format.

```
DIMENSIONS       | Wheelbase, Track F, Track R
FRONT            | Camber L/R, Sum Toe, Caster L/R, KPI L/R,
                 | Scrub L/R, Trail L/R
                 | RC static / @ 1g (Y, Z)
                 | Ride Camber, Roll Camber, Anti-dive
                 | Ackermann, Steer Arm, Steer Ratio
REAR             | (same structure, without caster/ackermann)
NOT COMPUTED     | List of what would need a spring/damper
```

---

## 9. Complete tutorial: from CAD to optimized

### Scenario 1: I have a finished car, I want to analyze it

**Step 1 — Extract hardpoints from SolidWorks**

For each of the 10 hardpoints of each corner:
1. Click on the point/sketch
2. "Mass Properties" or "Measure" → read X, Y, Z
3. Record them in a spreadsheet

> 💡 **Origin check:** before spending time, confirm that:
> - `WHEEL_CENTER.z ≈ tire radius` (positive, 220-260 mm typical)
> - `CONTACT_PATCH.z = 0`
> - `UCA_IN.z > LCA_IN.z` (the UCA is higher)
> If these do not match, your CAD origin is not SAE — transform it.

**Step 2 — Fill in the template**

1. Sidebar → **"⬇️ Template"** downloads `hardpoints_template.csv`
2. Open it in Excel, replace the values
3. Save as `my_car.csv`

**Step 3 — Load and analyze**

1. Sidebar → upload `my_car.csv`
2. Check the preview
3. Click **"🔄 Apply file"**
4. Go to **Tab 1 (Analysis)**
5. See the 6 KPIs in the cards — if any is WAY off, review it

**Step 4 — Dynamic sweep**

1. "Sweep type" → **"Heave"**
2. Min: −25 mm, Max: +25 mm, Step: 1 mm
3. Wait ~2s. These appear:
 - **Camber Gain** (°/mm) — target −0.015 to −0.025
 - **Bump Steer** (°/mm) — target < 0.005
 - **RC ΔY, ΔZ** — target ΔY < 30 mm

**Step 5 — See everything on the setup sheet**

The **"📋 Complete KPIs"** tab shows all the vehicle parameters in a table.

### Scenario 2: I'm going to design a new car, I want to find the ideal hardpoints

**Step 1 — Seed geometry**

Load a starting point (Demo, last year's car, etc.).

**Step 2 — 🎯 Synthesis tab**

1. Choose the **seed corner** (FL)
2. **Check the boxes** of the targets that matter:
 - ☑ Caster = 4.5°
 - ☑ KPI = 7°
 - ☑ Static camber = −1.5°
3. Adjust the dynamic targets:
 - Camber Gain = −0.020 °/mm
 - RC Height = 50 mm

**Step 3 — (Optional) Bounding boxes**

Expand **"📦 Bounding Boxes"**. Typical margins:
- UCA out / LCA out: ±50 mm
- TR in/out: ±25 mm

Start wide (±50-100), then tighten.

**Step 4 — Run**

Click **"🚀 Run Optimization"**. Time:
- 40 iter × 12 pop = ~6000 evaluations → 30s-2min
- 100 iter = 3-5 min

**Step 5 — Interpret**

A table appears:
```
Parameter          Target   Seed    Optimized  Seed OK  Opt OK
Caster (°)         +4.50    +8.88   +4.49      ❌       ✅
KPI (°)            +7.00    +4.47   +6.82      ❌       ✅
Camber (°)         -1.50    +0.00   -1.51      ❌       ✅
...
```

- ❌ Seed = did not meet it
- ✅ Opt = met after optimization
- If any stays ❌ → trade-off; increase the weight or loosen it

**Step 6 — Download the CSV and apply it in CAD**

1. **"⬇️ Download optimized hardpoints"** button
2. Open the CSV
3. In SolidWorks, edit each sketch with the new X, Y, Z

**Step 7 — Validate**

**🔄 Comparison** tab:
- A = "Last SEED geometry"
- B = "Last OPTIMIZED geometry"
- See overlaid charts confirming the gain

### Scenario 3: I want to play with hardpoints manually

**✏️ Manual Editor tab:**

1. Choose a corner to edit
2. Edit the X, Y, Z values directly in the table
3. The 3 2D views update live
4. "🪞 Mirror Left → Right" button if you want symmetry
5. Click **"✅ Apply as loaded hardpoints"** when finished
6. Go to Tab 1 and see the resulting KPIs

---

## 10. Hardpoints file format

### 10.1 Structure

5 columns, 40 rows (4 corners × 10 points):

| Column | Type | Description |
|---|---|---|
| `corner` | text | "FL", "FR", "RL", "RR" |
| `point` | text | point name |
| `x_mm` | number | X coordinate in mm |
| `y_mm` | number | Y coordinate in mm |
| `z_mm` | number | Z coordinate in mm |

### 10.2 The 10 points per corner

| Name | What it is |
|---|---|
| `UCA_IN_FRONT` | Front inboard of the upper arm |
| `UCA_IN_REAR` | Rear inboard of the upper arm |
| `UCA_OUT` | Outboard of the upper arm (= UBJ) |
| `LCA_IN_FRONT` | Front inboard of the lower arm |
| `LCA_IN_REAR` | Rear inboard of the lower arm |
| `LCA_OUT` | Outboard of the lower arm (= LBJ) |
| `TIE_ROD_IN` | Inboard of the tie-rod (on the rack) |
| `TIE_ROD_OUT` | Outboard of the tie-rod (on the upright) |
| `WHEEL_CENTER` | Wheel center |
| `CONTACT_PATCH` | Tire-ground contact (always Z=0) |

### 10.3 CSV example

```csv
corner,point,x_mm,y_mm,z_mm
FL,UCA_IN_FRONT,60,150,295
FL,UCA_IN_REAR,-70,150,295
FL,UCA_OUT,-5,590,280
FL,LCA_IN_FRONT,90,130,162
FL,LCA_IN_REAR,-70,130,162
FL,LCA_OUT,15,600,152
FL,TIE_ROD_IN,-50,180,200
FL,TIE_ROD_OUT,-60,580,195
FL,WHEEL_CENTER,5,610,220
FL,CONTACT_PATCH,5,610,0
... (repeat for FR, RL, RR)
```

### 10.4 Validation errors

| Message | Cause | Fix |
|---|---|---|
| `Invalid corners: ['fl']` | Lowercase | Use uppercase FL |
| `Unknown points: ['UCA_INBOARD']` | Wrong name | Use `UCA_IN_FRONT` |
| `Corner 'FL' missing points: ['...']` | Missing row | Add it |
| `Column 'x_mm' contains nulls` | Empty cell | Fill it in |
| `Column 'x_mm' must be numeric` | Text | Use a decimal point, not a comma |

---

## 11. Project structure

```
fsae_suspension_clean/
│
├── geometry/                       # Pure mathematical engine
│   ├── __init__.py
│   ├── primitives.py               # Point3D, Vector3D, Point2D, intersections
│   ├── solver_2d.py                # Four-bar mechanism (front view Y-Z)
│   ├── model_3d.py                 # ControlArm, KingpinGeometry,
│   │                                 SuspensionCorner, Vehicle
│   └── solver_3d.py                # 3D kinematic solver (3 spheres + LM)
│
├── analysis/                       # Analysis + I/O + KPIs + Optimization
│   ├── __init__.py
│   ├── sweeps.py                   # SweepRunner + Plotly plots
│   ├── optimizer.py                # DesignTargets, SuspensionOptimizer
│   ├── io_hardpoints.py            # read/write csv/xlsx/json
│   └── kpis.py                     # Ackermann, Steer Ratio, Ride/Roll Camber,
│                                     RC@1g, Anti-dive, build_full_report
│
├── app.py                          # 🌐 Streamlit (5 areas)
├── README.md                       # This file
└── requirements.txt
```

### What each module does

**`geometry/primitives.py`** — Base types (`Point3D`, `Vector3D`, `Point2D`) and intersection functions (circle-circle, line-line). Pure math.

**`geometry/solver_2d.py`** — Solves the suspension as a four-bar mechanism in the front view (Y-Z plane). Used for the Roll Center.

**`geometry/model_3d.py`** — OOP classes: `ControlArm`, `KingpinGeometry`, `SuspensionCorner`, `Vehicle`. Computes static KPIs.

**`geometry/solver_3d.py`** — 3D kinematic solver. Treats the upright as a rigid body. Solves the position at (heave, roll, rack) via 3-sphere intersection + `least_squares` (Levenberg-Marquardt).

**`analysis/sweeps.py`** — Runs sweeps (`SweepRunner`). Computes camber gain, bump steer, RC migration. Generates Plotly plots.

**`analysis/optimizer.py`** — Global optimization (`differential_evolution`). `DesignTargets` with static + dynamic targets, `HardpointBounds` for keep-out zones, `validate_against_targets()` for a report.

**`analysis/io_hardpoints.py`** — Reading, validation, writing. Builds `SuspensionCorner` and `Vehicle` from DataFrames.

**`analysis/kpis.py`** — Advanced KPIs (Ackermann, Steer Ratio, Ride/Roll Camber, RC@1g, Anti-dive). `build_full_report()` generates a complete report.

**`app.py`** — Streamlit with 5 areas: Sidebar + 4 tabs + Manual Editor.

---

## 12. Python usage (scripts)

### 12.1 Load and analyze

```python
from analysis.io_hardpoints import read_hardpoints, build_corner_from_dataframe
from geometry import KinematicSolver3D
from analysis.sweeps import SweepRunner, camber_gain_per_mm, bump_steer_per_mm

df = read_hardpoints("my_car.xlsx")
corner, tie_rod = build_corner_from_dataframe(df, "FL")

print(f"Caster: {corner.static_caster_deg():+.3f}°")
print(f"KPI:    {corner.static_kpi_deg():+.3f}°")

solver = KinematicSolver3D(corner, tie_rod)
runner = SweepRunner(solver=solver)
sweep  = runner.heave_sweep(-25.0, 25.0, 1.0)

print(f"Camber gain: {camber_gain_per_mm(sweep):+.5f} °/mm")
print(f"Bump steer:  {bump_steer_per_mm(sweep):+.5f} °/mm")
```

### 12.2 Optimization

```python
from analysis.optimizer import (
    SuspensionOptimizer, DesignTargets, validate_against_targets,
)

targets = DesignTargets(
    caster_target_deg=4.5,
    kpi_target_deg=7.0,
    static_camber_target_deg=-1.5,
    camber_gain_target_deg_per_mm=-0.020,
    rc_height_target_mm=50.0,
    heave_step_mm=5.0,
)

opt = SuspensionOptimizer(
    seed_corner=corner, seed_tie_rod=tie_rod, targets=targets,
    population_size=15, max_iterations=60, workers=-1,
)
result = opt.run()
print(result.summary())

report = validate_against_targets(
    result.optimal_corner, result.optimal_tie_rod, targets,
)
print(report.summary())
```

### 12.3 Complete report

```python
from analysis.io_hardpoints import build_vehicle_from_dataframe
from analysis.kpis import build_full_report

vehicle, tie_rods = build_vehicle_from_dataframe(df)

report = build_full_report(
    vehicle, tie_rods,
    cg_height_mm=280.0,
    brake_bias_pct=60.0,
    drive_type="RWD",
    roll_stiffness_deg_per_g=1.5,
)

print(f"Wheelbase: {report.wheelbase_mm:.1f} mm")
print(f"Track F:   {report.track_front_mm:.1f} mm")
print(f"Front:     {report.front}")
```

### 12.4 Export

```python
from analysis.io_hardpoints import dataframe_from_corner, save_dataframe

df_out = dataframe_from_corner(result.optimal_corner, result.optimal_tie_rod)
save_dataframe(df_out, "optimized_geometry.xlsx")
```

---

## 13. Complete KPI list

### 13.1 Per corner (`SuspensionCorner`)

| Method | Returns | Unit |
|---|---|---|
| `static_caster_deg()` | Caster | ° |
| `static_kpi_deg()` | Kingpin Inclination | ° |
| `static_camber_deg()` | Static camber (constructive input) | ° |
| `static_scrub_radius_mm()` | Scrub Radius | mm |
| `static_mechanical_trail_mm()` | Mechanical trail | mm |
| `static_kingpin_offset_mm()` | Kingpin offset at WC height | mm |
| `roll_center_height_mm()` | Static RC Height | mm |
| `steer_arm_length_mm(tro)` | Steering arm length | mm |
| `anti_dive_percent(...)` | Simplified anti-dive | % |
| `anti_squat_percent(...)` | Simplified anti-squat | % |

### 13.2 Advanced (`analysis/kpis.py`)

| Function | Returns |
|---|---|
| `wheelbase_mm(front, rear)` | Wheelbase |
| `track_width_mm(left, right)` | Track width |
| `ride_camber_deg_per_m(corner, tr)` | Ride Camber (°/m) |
| `roll_camber_deg_per_deg(corner, tr)` | Roll Camber (°/°) |
| `static_toe_deg(corner, tr)` | Static toe |
| `static_sum_toe_deg(L, R, ...)` | Sum Toe |
| `ackermann_geometry(...)` | Dict with Ackermann %, steer arms |
| `steer_ratio_and_cfactor(...)` | Dict with rack/wheel° and wheel°/rack |
| `steer_ratio_from_pinion(...)` | Steer Ratio (x:1) |
| `roll_center_at_1g_lat(...)` | RC under 1g lateral |
| `anti_dive_percent(...)` | Anti-dive |
| `anti_squat_percent(...)` | Anti-squat |
| `build_full_report(...)` | Complete `FullKPIReport` |

### 13.3 Dynamic (from sweeps)

| Function | Computes |
|---|---|
| `camber_gain_per_mm(sweep)` | Slope of camber vs heave |
| `bump_steer_per_mm(sweep)` | Slope of toe vs heave |
| `rc_migration_range(sweep)` | (ΔY, ΔZ) of the RC during a sweep |

---

## 14. Troubleshooting

### 14.1 Streamlit

| Problem | Solution |
|---|---|
| `command not found: streamlit` | Activate the venv and `pip install streamlit` |
| Blank page | `streamlit run app.py --server.port 8502` |
| polars import error | `pip install polars openpyxl fastexcel` |
| `[Errno 2] No such file or directory: '/tmp/...'` | Windows bug — update to the latest app.py version |

### 14.2 File upload

| Message | Solution |
|---|---|
| `Invalid corners` | Use uppercase FL/FR/RL/RR |
| `Column x_mm contains nulls` | Fill in all 40 rows |
| `ModuleNotFoundError: openpyxl` | `pip install openpyxl` |

### 14.3 Absurd KPI values

| Result | Cause | Check |
|---|---|---|
| Caster = 0° | Outboards at the same X | Difference of X between UCA_OUT and LCA_OUT |
| KPI = 0° | Outboards at the same Y | Difference of Y between UCA_OUT and LCA_OUT |
| Camber/KPI = ±70° | Narrow upright (Z UBJ ≈ Z LBJ) | Vertical distance 80-180 mm |
| RC Height < 0 | RC below the ground | UCA inboard lower than outboard? Z swapped? |
| Scrub > 100 mm | WC at the wrong Y | Check WHEEL_CENTER.y |

### 14.4 Upright diagnostics

The upright (UBJ-LBJ) should have:
- **Vertical height (Z)**: 80-180 mm
- **Total distance**: 100-200 mm

```python
upright = corner.upper_arm.outboard.distance_to(corner.lower_arm.outboard)
height_z = abs(corner.upper_arm.outboard.z - corner.lower_arm.outboard.z)
print(f"Upright: {upright:.1f} mm, Z height: {height_z:.1f} mm")
```

If the upright < 60 mm or Z height < 50 mm → review the hardpoints.

### 14.5 CAD origin different from SAE

Obvious signs:
- `WHEEL_CENTER.z` should be the **tire radius** (~220-260 mm, positive)
- `CONTACT_PATCH.z` should be **0**

Typical conversion:
```python
# If the CAD Z points downward with the origin at the wheel center:
Z_sae = TIRE_RADIUS - Z_user

# If Y is negative for the left side:
Y_sae = -Y_user
```

### 14.6 Optimization

| Symptom | Solution |
|---|---|
| Cost > 100 after many iterations | Targets impossible simultaneously; loosen one |
| Hardpoints with no variation | Bounds too tight; increase them |
| Result worse than the seed | Increase iterations (≥100) |
| Takes > 10 min | `heave_step_mm=5` or reduce the population |
| "Mechanism out of reach" | Bounds generate an impossible geometry |

### 14.7 Manual editor loses data

If clicking "🔄 Apply file" makes the manual editor discard your edits:
- This is the correct behavior: applying a file **overwrites** the editor state
- To keep your edits: click "✅ Apply as hardpoints" in the editor BEFORE loading another file

---

## 15. Limitations

### 15.1 What it DOES ✅

- 3D kinematics in (heave, roll, steer)
- 6+ static KPIs
- 10+ dynamic KPIs
- Global optimization with mixed targets
- Bounding boxes
- Validation
- CSV/Excel/JSON export

### 15.2 What it DOESN'T do ❌

- **Vertical dynamics** (mass-spring-damper)
- **Arm compliance**
- **Pushrod/pullrod/rocker** (no motion ratio)
- **3D visualization of the hardpoints** (2D only)
- **Optimization of all 4 corners together**
- **Physical interference detection**
- **Tire load** (load transfer)
- **Wheel rate, roll rate, natural frequency, damping**

### 15.3 Important approximations

- **Roll axis at the origin** — rotation about X passing through (0,0,0). For roll < 3°, error < 1 mm
- **Non-interpretable absolute toe** — reports **Δ toe** relative to static
- **Rigid upright** — no compliance
- **Static camber = input** — not inferred from the hardpoints
- **Simplified anti-dive** — assumes an outboard brake
- **Approximate RC @ 1g** — uses a fixed roll stiffness (default 1.5 °/g)

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **A-arm / Wishbone** | "A"-shaped arm, FSAE standard |
| **Anti-dive / Anti-squat** | Side-view geometries that reduce dive/squat |
| **Ball joint** | Spherical joint (rod end) arm↔upright |
| **Bounding box** | 3D box for keep-out zones |
| **Bump** | Wheel rising relative to the chassis (heave +) |
| **Bump steer** | INVOLUNTARY toe variation with heave |
| **Camber** | Wheel inclination vs vertical |
| **Camber Gain** | d(camber)/d(heave) in °/mm |
| **Caster** | Kingpin inclination in the side view |
| **Compliance** | Elastic deformation (bushings, arms) |
| **Contact patch (CP)** | Tire-ground contact area |
| **Differential evolution** | Evolutionary algorithm of the optimizer |
| **DOF** | Degree of Freedom |
| **FSAE** | Formula SAE — student competition |
| **Hardpoint** | Pivot/attachment point |
| **Heave** | Vertical chassis-wheel displacement |
| **Inboard / Outboard** | Chassis side / wheel side |
| **Instant Center (IC)** | Instant center of rotation of the upright |
| **Jounce** | Synonym of bump |
| **KPI** | Kingpin Inclination |
| **LBJ / UBJ** | Lower / Upper Ball Joint |
| **LCA / UCA** | Lower / Upper Control Arm |
| **Levenberg-Marquardt (LM)** | least_squares algorithm |
| **Mechanical Trail** | Longitudinal distance kingpin-ground to CP |
| **Motion Ratio** | Wheel displacement / spring ratio |
| **Pickup point** | Synonym of hardpoint |
| **Pushrod / Pullrod** | Upright → rocker bar |
| **Rack** | Steering rack |
| **Rebound** | Wheel dropping (heave −) |
| **Rocker / Bell-crank** | Pushrod → spring lever |
| **Roll** | Chassis rotation about X |
| **Roll Axis** | Line joining RC F and RC R |
| **Roll Center (RC)** | Instant roll pivot (front view) |
| **Scrub Radius** | Lateral distance kingpin-ground to CP |
| **Seed** | Initial geometry of the optimization |
| **Steer** | Steering |
| **Sweep** | Parametric sweep |
| **SVIC** | Side View Instant Center |
| **Tie-rod** | Steering bar rack→upright |
| **Toe** | Convergence/divergence |
| **TRO / TRI** | Tie Rod Outboard / Inboard |
| **Upright** | Steering upright |
| **Wheel Center (WC)** | Wheel center |

---

## 📞 About

Software developed as an educational project for Formula SAE teams. It is not a commercial product.

The modular structure allows extension:
- Pushrod/pullrod → extend `SuspensionCorner` and the 3D solver
- 3D visualization → use `plotly.graph_objects.Scatter3d`
- Full side view (anti-dive) → create `solver_xz.py`
- Wheel rate / natural frequency → new module + spring inputs
- SolidWorks integration → use the COM API (Windows)

**Version:** 2026
