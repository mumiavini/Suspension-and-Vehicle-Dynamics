# 🏎️ FSAE Suspension Geometry Engine

Documentation for **`legacy_app/`** — the original Streamlit tool for FSAE suspension
kinematics: **analyze** a geometry (static KPIs, sweeps, setup sheet) and inspect it in
2D/3D. It is still functional but frozen — new engineering work happens in `vdcore/`,
a pure Python library with a validated 3D solver (see the repository root
[`README.md`](../README.md) and [`CLAUDE.md`](../CLAUDE.md)). Every dynamic KPI in this
app is now computed by that same `vdcore` solver via a bridge module — see §8.

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
9. [Complete tutorial: from CAD to setup sheet](#9-complete-tutorial-from-cad-to-setup-sheet)
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

It also doesn't decide anything for you. Every number here is a consequence of the
hardpoints you gave it, not a recommendation — see `CLAUDE.md`'s design principle for
why that's deliberate.

---

## 2. What the software DOES and DOESN'T do

### ✅ DOES — Kinematics and geometry

| Category | Computed KPIs |
|---|---|
| **Static** | Caster, KPI, Camber, Scrub Radius, Mechanical Trail, RC Height, Kingpin Offset @ WC |
| **Dimensions** | Wheelbase, Track Width F/R |
| **Steering** | Steer Arm Length, Ackermann % (real rack sweep on the linkage), Rack/degree, Steer Ratio (with c-factor input) |
| **Dynamic** (`vdcore`/`DWSolver`) | Camber Gain, Ride Camber, Roll Camber, Bump Steer (linear rate + peak over travel), RC migration/height, half-track change (scrub in bump) |
| **Roll Center at a chosen roll angle** | RC height and lateral position from an actual kinematic sweep that holds both contact patches on the tilted road — not an approximation |
| **Side view** | Anti-dive %, Anti-squat % (pivot-axis-rake construction) |
| **State** | Static Sum Toe |
| **Setup sheet (given user inputs)** | Wheel Rate, Roll Rate, Sprung-mass Natural Frequency — plain algebra from a typed spring rate, motion ratio and mass |
| **Reverse engineering** | Snapshot of a seed corner's KPIs (static + dynamic) as a reference for manual redesign — **not** an automated search, see §9 |

### ❌ DOESN'T DO (currently) — Dynamics, structure, and automated search

| Category | Why it's missing |
|---|---|
| **Motion Ratio** | Needs a **pushrod/pullrod/rocker** model — not derivable from the 10 hardpoints per corner; type it in |
| **Jounce/Rebound Damping** | Recorded as a typed `% critical` value on the setup sheet, not computed from an F×v damper curve |
| **FEA of the arms** | A different kind of software (Ansys, etc.) |
| **Lap time simulation** | A different kind of software (OptimumLap) |
| **Automated hardpoint synthesis** | **Retired.** The old `differential_evolution` optimizer scored candidates with a solver that didn't close the real linkage (see `CLAUDE.md`). There is no drop-in `DWSolver`-based replacement yet — it is too slow per-solve for the same cost function |

---

## 3. Why weight/engine/spring are NOT needed for most KPIs

This is a common question, and the answer is important:

### 🟢 Pure kinematics — geometry (and a chosen roll angle) only

Most KPIs depend **only on the hardpoint positions**, on **how they move**, and — for
the roll-centre-at-roll figures — on the roll angle you ask to evaluate:

```
Caster, KPI, Camber, Scrub, Trail, Roll Center (static and at a given roll angle)
Camber Gain, Bump Steer, Ride/Roll Camber
Ackermann %, Steer Arm Length
```

These parameters are **invariant with respect to mass**. A 200 kg car and a 300 kg car
with the same hardpoint geometry will have the same Caster, same Camber, same
Ackermann. The roll-centre-at-roll figures likewise only need the roll angle itself
(a design load case you pick, e.g. 1.5°) — not roll stiffness, spring rate or mass.

Mass only matters to:
- Compute the **natural frequency** (needs wheel rate × mass)
- Compute **absolute loads** (for FEA)
- Compute **load transfer**

The software computes the first of these three *if you type in* spring rate, motion
ratio and mass (see the setup sheet in §8); it computes none of the other two.

### 🟡 The one KPI that needs an external parameter

**Anti-dive / Anti-squat:**
- Computed by `SuspensionCorner.anti_dive_percent(brake_bias, wheelbase_mm, cg_height_mm)`
  — a pivot-axis-rake construction (Milliken & Milliken *RCVD* eq. 17.21), cross-checked
  against `sla_geometry.py`'s construction and the full 3D linkage
- Needs the **CG height** and **brake bias** → you provide them in the vehicle setup
  (sidebar) and the setup sheet's mass inputs
- **Does NOT need the absolute weight** — only the CG-height/wheelbase ratio and the
  brake-bias fraction

### 🔴 KPIs that require external data (future)

If you want **motion ratio derived from the pushrod geometry** or a real **damper F×v
curve**, the pushrod/pullrod/rocker linkage would need to be modeled and added as its
own hardpoint set — that would multiply the size of the project. That is why the
current scope is **kinematics of the wishbone/upright/tie-rod linkage** — which already
covers the majority of a typical FSAE setup sheet.

### Table summary

| You need to... | Does the software compute it TODAY? |
|---|---|
| Move hardpoints and see Caster/KPI/Camber/etc. | ✅ Yes |
| See Ackermann, Steer Ratio | ✅ Yes |
| Anti-dive/squat (needs CG height and brake bias) | ✅ Yes, with inputs |
| RC height/lateral at a chosen roll angle | ✅ Yes, from a real kinematic sweep |
| Wheel rate / roll rate / natural frequency | ✅ Yes, if you type in spring rate + motion ratio + mass |
| Automated hardpoint search from targets | ❌ No — retired, see §9 |
| Motion ratio from pushrod geometry, damper curves | ❌ No |
| FEA, stress analysis, lap time simulation | ❌ No |

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
| **RC height range** | < 30 mm over the bump/droop sweep |
| **Roll Camber** | −0.5 to −1.5 °/° |
| **Anti-dive** | 0 to 30% |
| **Ackermann** | 30% to 100% |

### 4.3 Coupling of the parameters

⚠️ All these parameters are **geometrically coupled** — you cannot change one without affecting the others:

- Increasing Caster → tends to increase Trail
- Increasing KPI → reduces Scrub Radius
- Raising the LCA inboard → affects Camber Gain AND RC at the same time

This coupling is exactly why changing hardpoints is a trade-off exercise rather than a
single "fix". The Analysis tab's setup sheet and sweeps (§8) exist to make each
trade-off visible so you can iterate deliberately — see §9 for a manual workflow.
Automated hardpoint search used to exist for this; it's retired (§2, §9).

---

## 5. Conventions and units

### 5.1 Axis system (ISO 8855)

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

- **Origin:** center of the front axle, at ground level, on the vehicle centreline
- **X+** = front · **Y+** = left · **Z+** = up · right-handed (X × Y = Z)

> ⚠️ **Not the same as SAE J670e.** J670e's body axis system is X forward, Y to the
> right, Z down — the opposite Y and Z sense of the frame used here. Don't mix the two:
> `vdcore.io.frames` carries the conversion matrices if you need to go from CAD/J670e
> into this frame. Between this app's hardpoints DataFrame and `vdcore`, the mapping is
> the identity — no sign flip needed.
> Common signs your CAD export is in a different frame:
> - `WHEEL_CENTER.z` ≈ 0 → your Z is not "height above ground" (it should be the tire radius)
> - `LCA_IN.z > UCA_IN.z` → your Z points downward
> - The left corners (FL/RL) come out with **negative** Y → your Y points right, not left

### 5.2 Signs

| Parameter | + means |
|---|---|
| **Camber** | Top of the wheel OUTWARD (negative = racing camber, top inboard) |
| **Caster** | Top of the kingpin BEHIND the base |
| **KPI** | Top of the kingpin INWARD |
| **Scrub** | Kingpin crosses the ground INWARD of the contact patch |
| **Toe (per side)** | Toe-IN (front of the wheel points toward the vehicle centerline) |
| **Sum Toe** | Total toe-in across both wheels on an axle |
| **Heave** | Bump (wheel rises relative to the chassis) |
| **Roll** | Chassis rolls to the RIGHT |
| **Rack** | Rack moves to the LEFT |

Always state whether a toe or camber value is **per side** or a **sum/total** — the two
have caused real confusion on the car. `static_toe_deg` is per side; `static_sum_toe_deg`
is the total.

### 5.3 Units

- Lengths: **mm**
- Angles: **degrees (°)**
- Camber gain: **°/mm**
- Ride camber: **°/m**

No inches or radians in the UI or file format. Convert beforehand (1 in = 25.4 mm).

---

## 6. Installation

### 6.1 Prerequisites

- Python 3.10+ ([download](https://www.python.org/downloads/))
- ~500 MB free

### 6.2 Step by step

**1. Clone the repository** and move into the legacy app folder:
```bash
git clone <this-repo-url>
cd Suspension-and-Vehicle-Dynamics/legacy_app
```

**2. Virtual environment (recommended):**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```
`requirements.txt` is a fully pinned lock file (streamlit, plotly, polars, numpy, scipy,
pydantic, and their transitive dependencies) — install it as-is rather than picking
individual packages.

**4. Test the import** (run from inside `legacy_app/`, so `geometry`/`analysis`/`ui`
resolve as top-level packages):
```bash
python -c "from geometry.primitives import Point3D; print(Point3D(1, 2, 3))"
```

### 6.3 Running the app

From inside `legacy_app/`:
```bash
streamlit run app.py
```
Opens in the browser at `http://localhost:8501`.

`app.py` puts the repository root on `sys.path` itself (needed to `import vdcore`), so
you do not need to install `vdcore` separately or set `PYTHONPATH` — just run the
command from `legacy_app/`.

---

## 7. Getting started (5 min)

```
1. Terminal (from legacy_app/):    streamlit run app.py
2. Sidebar → "📋 Demo"
3. "📊 Analysis" tab
4. See the front/rear axle cards appear (camber gain, RC migration, ...)
5. Expand "📈 Detailed sweeps" → "Heave"
6. Charts appear automatically
```

---

## 8. App tabs tour

The app has a **sidebar** and **5 tabs**.

### 📂 Sidebar — Hardpoints and vehicle setup

```
┌─ 1️⃣ Load data ────────────────┐
│ [Hardpoints file uploader]   │  ← .csv/.xlsx/.json
│ ✅ '...' — 40 points ·       │  ← preview, corners shown
│    corners: FL, FR, RL, RR   │
│ [🔄 Apply file]              │  ← click here to apply
│                              │
│ Or use a shortcut:            │
│ [📋 Demo] [⬇️ Template]      │
│ [🗑️ Clear session]           │
│                              │
│ ─────────────                │
│ 2️⃣ Vehicle setup             │
│ Brake bias front:    0.60    │  ← affects anti-dive/anti-squat
│ ▼ 🔬 vdcore inputs           │
│   Static camber:    -1.50°   │  ← not in the hardpoints file
│   Loaded radius:     245 mm  │
│ ▼ 🔧 Steering                 │
│   c-factor:          100 mm  │  ← affects steer ratio
│   Steering lock:     270°    │
│                              │
│ 🎨 App theme                 │
└──────────────────────────────┘
```

**Recommended flow:**

1. **Load a file** OR click **Demo** OR build one in the **✏️ Inputs** tab
2. Check the preview, click **🔄 Apply file**
3. Fill in **Brake bias**, **static camber**, **loaded radius** and **c-factor** for
   correct anti-dive, dynamic-KPI and steer-ratio numbers
4. Go to the tabs

### ✏️ Tab 1: Inputs

Manual hardpoint editor with live 2D visualization — the tool for building a geometry
from scratch or inspecting one point by point.

```
┌─────────────────────┬─────────────────────┐
│ Editable table       │ 3 2D views (tabs):  │
│ (10 pts, one corner) │  - Front (YZ)       │
│                      │  - Side (XZ)        │
│                      │  - Top (XY)         │
│                      │  Updates live as     │
│                      │  you type            │
└─────────────────────┴─────────────────────┘
[✅ Apply as loaded hardpoints]  [⬇️ Download all (CSV)]
[🪞 Mirror Left → Right (Y → −Y)]
[📋 Load template into this corner]
[🔁 Reload from file]  (only shown once a file/demo is loaded)
```

Applying a file in the sidebar **overwrites** whatever is in this editor — apply your
edits first if you want to keep them (see §14.7).

### 📊 Tab 2: Analysis

Every KPI here is computed by the validated `vdcore` 3D solver (`DWSolver`), which
constrains all six degrees of freedom with the real wishbone/tie-rod linkage, is
covered by the test suite, and agrees with an independent Altair MotionSolve run to
~1e-7 mm.

```
▼ 🅰️ Altair MotionSolve cross-check (optional, ~150 s to run)
──────────────────────────────────────
Front axle card: Camber gain · RC migration · Half-track change
                 Camber @ full bump/droop · RC height range
                 Roll (at chosen angle): outer/inner camber, RC height, RC lateral
Rear axle card:  (same structure)
──────────────────────────────────────
Chart: Camber vs wheel travel        Chart: RC height vs wheel travel
──────────────────────────────────────
▼ 📈 Detailed sweeps
   Sweep: Heave | Steer | Roll (axle)   → per-corner or per-axle charts + data table
──────────────────────────────────────
▼ 🔧 Additional inputs (tires, suspension/spring, masses, damper, other)
📋 Complete Setup Sheet — category-filterable table:
   🛞 Tires & Wheels · 🔩 Suspension & Rates (incl. wheel rate, roll rate,
   natural frequency IF you filled in spring rate/motion ratio/mass) ·
   🎢 Kinematics (ride/roll camber, anti-dive/squat, bump steer, RC) ·
   📐 Static alignment (KPI, caster, scrub, trail, sum toe, camber) ·
   🕹️ Steering (Ackermann, steer ratio, steer arm length) · ⚖️ Masses
[⬇️ Download setup sheet (CSV)]
```

Two things worth knowing:
- **Roll is solved at axle level**, not per corner: `vdcore` finds the wheel travel on
  each side that keeps both contact patches on the tilted road at your chosen roll
  angle (default 1.5°, a slider — not derived from a roll-stiffness input).
- **Ackermann** comes from an actual rack sweep on `DWSolver` to a target outer-wheel
  steer angle, not the classic plan-view "extend the steering arms to the rear axle"
  construction — that construction assumes a vertical kingpin and is unusable on a car
  with double-digit KPI (see `vdcore_bridge.solved_ackermann_pct`'s docstring).

### 🌐 Tab 3: View 3D

Interactive 3D visualization: rotate, zoom, and inspect the layout.

```
Mode: 🏎️ Complete vehicle | 🔍 Single corner | 🎬 Sweep animation
  Complete vehicle → full car, toggle tires / chassis wireframe
  Single corner     → one corner + a small KPI expander (caster/KPI/scrub)
  Sweep animation   → heave/roll/steer slider + ▶ Play
```

⚠️ **The sweep animation is the one place the legacy solver still runs**
(`analysis/viz3d.py` → `KinematicSolver3D`, which models each wishbone as a strut to
its pivot midpoint). It's fine for seeing how the linkage *looks* through a sweep; do
not read camber or toe values off it — use the Analysis tab for numbers.

### 🎯 Tab 4: Synthesis / Optimization

🚧 **Automated hardpoint search is retired.** This tab used to run a
`differential_evolution` optimizer against user-picked targets; that optimizer scored
candidates with the legacy strut-to-pivot-midpoint solver, which reported camber gain
and roll-centre migration with the wrong sign and magnitude. There's no drop-in
`DWSolver`-based replacement yet (`DWSolver` is much slower per solve, and the old cost
function evaluated it thousands of times per run).

What's left is a **seed-corner snapshot**: pick a corner, see its static KPIs
(caster/KPI/camber/RC height) and dynamic KPIs (camber gain, bump steer, both computed
via `DWSolver`) as a reference while you adjust hardpoints manually — check the result
on the **Analysis** tab.

### 🔄 Tab 5: Comparison

Compares two geometries (A vs B) side by side, computed via `DWSolver`.

```
Source A / B: File corner | Last SEED | Last OPTIMIZED
Table: static KPIs (Caster, KPI, camber, toe, scrub, trail, steer arm) with Δ (B−A)
Heave sweep overlay: camber gain / bump steer metrics + Camber vs Heave,
                      Δ Toe vs Heave, and RC trajectory (Y × Z) charts
```

The **"Last SEED"** / **"Last OPTIMIZED"** source options are left over from the
retired optimizer (§Tab 4): they only populate if something in this session writes
`st.session_state["last_optimization"]`, which nothing currently does. In practice,
compare **"File corner"** against **"File corner"** — e.g. two corners of the same
file, or reload a different file between picks.

---

## 9. Complete tutorial: from CAD to setup sheet

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
> - The left corners (FL/RL) have **positive** Y
> If these do not match, your CAD export is not in this app's frame (§5.1) — transform it.

**Step 2 — Fill in the template**

1. Sidebar → **"⬇️ Template"** downloads `hardpoints_template.csv`
2. Open it in Excel, replace the values
3. Save as `my_car.csv`

**Step 3 — Load and analyze**

1. Sidebar → upload `my_car.csv`
2. Check the preview
3. Click **"🔄 Apply file"**
4. Go to **📊 Analysis**
5. Check the axle cards — if a number looks wrong, review the hardpoints (§14.3)

**Step 4 — Dynamic sweeps**

Expand **"📈 Detailed sweeps"**:
1. **"Heave"** → Min −25 mm, Max +25 mm, Step 1 mm → camber-vs-heave and bump-steer
   charts, RC migration
2. **"Roll (axle)"** → pick Front/Rear, a roll range → outer/inner camber and RC
   height/lateral vs chassis roll

**Step 5 — See everything on the setup sheet**

Expand **"🔧 Additional inputs"** to fill in springs/motion ratio/masses if you want
wheel rate, roll rate and natural frequency on the sheet, then check **"📋 Complete
Setup Sheet"** for the full documentation table, and download it as CSV.

### Scenario 2: I'm iterating a geometry by hand toward some targets

There is no automated search anymore (§2, §8 Tab 4) — this is a manual loop:

**Step 1 — Load a seed geometry**

Demo, last year's car, or a file you've built in the **✏️ Inputs** tab.

**Step 2 — Snapshot the seed**

**🎯 Synthesis / Optimization** tab → pick the corner → note its current caster, KPI,
camber, camber gain, bump steer, RC height as your baseline.

**Step 3 — Change a hardpoint, re-check**

Edit the value in the **✏️ Inputs** tab (or upload a new file), click **"✅ Apply as
loaded hardpoints"**, and re-read the KPIs on the **📊 Analysis** tab. Because the
parameters are coupled (§4.3), expect a change aimed at one KPI to move several others
— that coupling is the trade-off the tool is making visible, not something to fight.

**Step 4 — Compare before/after**

**🔄 Comparison** tab, both sides set to **"File corner"**: keep two files (or two
corners) around and diff them — static KPI table with Δ, plus overlaid heave-sweep
charts.

**Step 5 — Export**

**"⬇️ Download all (CSV)"** in the Inputs tab, or the setup sheet's CSV, once you're
happy with a candidate — hand it to CAD.

### Scenario 3: I want to play with hardpoints manually

**✏️ Inputs tab:**

1. Choose a corner to edit (segmented control)
2. Edit the X, Y, Z values directly in the table
3. The 3 2D view tabs update live
4. "🪞 Mirror Left → Right" button if you want symmetry
5. Click **"✅ Apply as loaded hardpoints"** when finished
6. Go to **📊 Analysis** and see the resulting KPIs

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
Suspension-and-Vehicle-Dynamics/
│
├── vdcore/                          # Pure library — the validated solver (see CLAUDE.md)
│   ├── models/                      # Hardpoint, Corner, Axle, Target, TirePackage
│   ├── geometry/                    # primitives, DWSolver, derived KPIs (scrub, trail...)
│   ├── analysis/                    # axle rates/roll, camber, roll centre, toe
│   ├── io/                          # config load/save, frame transforms
│   ├── tire/, optimize/, validate/  # stubs / benchmark cross-checks
│
├── legacy_app/                      # This documentation's subject — frozen, still functional
│   ├── geometry/
│   │   ├── primitives.py            # Point3D, Vector3D, Point2D, intersections
│   │   ├── solver_2d.py             # Four-bar mechanism (front view Y-Z)
│   │   ├── model_3d.py              # ControlArm, SuspensionCorner, Vehicle,
│   │   │                              anti_dive_percent/anti_squat_percent
│   │   └── solver_3d.py             # Legacy strut-to-midpoint solver — View 3D
│   │                                  animation only, see §8 Tab 3
│   ├── analysis/
│   │   ├── io_hardpoints.py         # read/write csv/xlsx/json, validation
│   │   ├── kpis.py                  # wheelbase, track, sum toe, steer ratio, steer arm
│   │   ├── sweeps.py                # sweep result layout + plotly plots
│   │   ├── vdcore_bridge.py         # bridge: loaded hardpoints → vdcore/DWSolver
│   │   ├── viz3d.py                 # legacy solver → 3D plotly figures (View 3D)
│   │   └── altair_bridge.py         # optional Altair MotionSolve cross-check
│   ├── ui/                          # one module per tab: tab_inputs, tab_vdcore
│   │   │                              (the Analysis tab), tab_view3d, tab_synthesis,
│   │   │                              tab_compare, sidebar, theme, shared
│   ├── app.py                       # Streamlit entry point: sidebar + 5 tabs
│   └── requirements.txt             # pinned lock file for this app
│
├── tests/                           # unit, property (hypothesis), benchmark tests for vdcore
├── scripts/                         # purity check, dev hooks, export utilities
├── docs/                            # this file, onboarding, theory notes
├── configs/                         # versioned vehicle configurations
└── README.md                        # repository-level overview
```

### What each `legacy_app/` module does

**`geometry/primitives.py`** — Base types (`Point3D`, `Vector3D`, `Point2D`) and intersection functions (circle-circle, line-line). Pure math.

**`geometry/solver_2d.py`** — Solves the suspension as a four-bar mechanism in the front view (Y-Z plane). Used for the static Roll Center.

**`geometry/model_3d.py`** — OOP classes: `ControlArm`, `KingpinGeometry`, `SuspensionCorner`, `Vehicle`. Computes static KPIs, plus the corrected `anti_dive_percent`/`anti_squat_percent`.

**`geometry/solver_3d.py`** — The original 3D kinematic solver: treats the upright as a rigid body positioned by 3-sphere intersection + `least_squares`, modeling each wishbone as a strut to its pivot midpoint. Only used today by the View 3D tab's sweep animation — its dynamic output is not otherwise trusted (see `CLAUDE.md`).

**`analysis/sweeps.py`** — Defines the sweep result layout (`SWEEP_DTYPE`) and derives camber gain, bump steer, RC migration from it. Sweeps themselves are produced by `analysis/vdcore_bridge.py` (vdcore/DWSolver), not by this module. Generates Plotly plots.

**`analysis/io_hardpoints.py`** — Reading, validation, writing. Builds `SuspensionCorner` and `Vehicle` from DataFrames.

**`analysis/kpis.py`** — Wheelbase, track width, static sum toe, steer ratio/C-factor, steering arm length. Dynamic KPIs (camber gain, RC migration, Ackermann %, anti-dive/anti-squat) come from `vdcore`/`DWSolver` via `analysis/vdcore_bridge.py`; see CLAUDE.md.

**`analysis/vdcore_bridge.py`** — Lifts the loaded hardpoints DataFrame into `vdcore` models (`Corner`, `Axle`) and returns validated dynamic KPIs; the one place `legacy_app` calls into `vdcore`.

**`analysis/viz3d.py`** — Builds the plotly 3D figures for the View 3D tab, driven by the legacy `solver_3d.py` for the animated mode.

**`analysis/altair_bridge.py`** — Optional independent cross-check: runs the same geometry through Altair MotionSolve and caches the result against a signature of the geometry + inputs.

**`app.py`** — Streamlit entry point: page config, theme, header, sidebar, and 5 tabs (Inputs, Analysis, View 3D, Synthesis/Optimization, Comparison), each delegating to its `ui/tab_*.py` module.

---

## 12. Python usage (scripts)

### 12.1 Load and analyze

```python
from analysis.io_hardpoints import read_hardpoints, build_corner_from_dataframe
from analysis.vdcore_bridge import CornerInputs, df_to_vdcore_corner, vdcore_sweep
from analysis.sweeps import camber_gain_per_mm, bump_steer_per_mm

df = read_hardpoints("my_car.xlsx")
corner, tie_rod = build_corner_from_dataframe(df, "FL")

print(f"Caster: {corner.static_caster_deg():+.3f}°")
print(f"KPI:    {corner.static_kpi_deg():+.3f}°")

vd_corner = df_to_vdcore_corner(df, "FL", CornerInputs.from_vehicle_setup({}))
sweep = vdcore_sweep(vd_corner, "Heave", (-25.0, 25.0, 1.0))

print(f"Camber gain: {camber_gain_per_mm(sweep):+.5f} °/mm")
print(f"Bump steer:  {bump_steer_per_mm(sweep):+.5f} °/mm")
```

Run this from inside `legacy_app/` (or with the repository root and `legacy_app/` both
on `sys.path`) so `analysis`/`geometry` resolve and `import vdcore` succeeds.

### 12.2 Export

```python
from analysis.io_hardpoints import dataframe_from_corner, save_dataframe

df_out = dataframe_from_corner(corner, tie_rod)
save_dataframe(df_out, "modified_geometry.xlsx")
```

---

## 13. Complete KPI list

### 13.1 Per corner (`SuspensionCorner`, `legacy_app/geometry/model_3d.py`)

| Method | Returns | Unit |
|---|---|---|
| `static_caster_deg()` | Caster | ° |
| `static_kpi_deg()` | Kingpin Inclination | ° |
| `static_camber_deg()` | Always `0.0` — the legacy model cannot infer camber from hardpoints. The Analysis tab reads static camber from the sidebar's `vdcore` inputs instead | ° |
| `static_scrub_radius_mm()` | Scrub Radius | mm |
| `static_mechanical_trail_mm()` | Mechanical trail | mm |
| `static_kingpin_offset_mm()` | Kingpin offset at WC height | mm |
| `roll_center_height_mm()` | Static RC Height | mm |
| `steer_arm_length_mm(tro)` | Steering arm length | mm |
| `anti_dive_percent(brake_bias, wheelbase_mm, cg_height_mm)` | Anti-dive (pivot-axis-rake construction) | % |
| `anti_squat_percent(...)` | Anti-squat (same construction, brake_bias=1.0, rear corner) | % |

The Analysis tab's setup sheet (§8 Tab 2) shows the `vdcore`/`DWSolver` equivalents of
the static rows above (camber, KPI, caster, scrub, trail, sum toe) alongside these —
they should agree to a few thousandths of a degree/mm except for camber, which
`SuspensionCorner` doesn't compute at all.

### 13.2 Advanced (`legacy_app/analysis/kpis.py`)

| Function | Returns |
|---|---|
| `wheelbase_mm(front, rear)` | Wheelbase |
| `track_width_mm(left, right)` | Track width |
| `static_toe_deg(corner, tr)` | Static toe, per side |
| `static_sum_toe_deg(L, R, ...)` | Sum Toe |
| `steering_arm_lengths(fl, fl_tr, fr, fr_tr)` | Dict with left/right steering-arm length |
| `steer_ratio_from_pinion(...)` | Steer Ratio (x:1) |

Ackermann %, anti-dive/anti-squat, camber gain, and RC migration are dynamic
KPIs computed via `analysis/vdcore_bridge.py` (vdcore/DWSolver), not by this
module — see CLAUDE.md.

### 13.3 Dynamic (from sweeps, `legacy_app/analysis/sweeps.py`)

| Function | Computes |
|---|---|
| `camber_gain_per_mm(sweep)` | Slope of camber vs heave |
| `bump_steer_per_mm(sweep)` | Slope of toe vs heave |
| `rc_migration_range(sweep)` | (ΔY, ΔZ) of the RC during a sweep |

### 13.4 Setup-sheet-only KPIs (`legacy_app/ui/tab_vdcore.py`)

These are simple presentation-layer formulas over user-typed values, not derived from
the hardpoints — they only appear once you fill in the setup sheet's spring/mass inputs
(§8 Tab 2):

| Function | Computes | Needs |
|---|---|---|
| `_wheel_rate(spring_rate, mr)` | Wheel rate = spring_rate × MR² | spring rate, motion ratio |
| `_roll_rate(wheel_rate, track_mm)` | Roll rate per wheel | wheel rate, track width |
| `_natural_freq(wheel_rate, sprung_mass)` | Sprung-mass natural frequency | wheel rate, sprung mass/corner |

---

## 14. Troubleshooting

### 14.1 Streamlit

| Problem | Solution |
|---|---|
| `command not found: streamlit` | Activate the venv and `pip install -r requirements.txt` |
| Blank page | `streamlit run app.py --server.port 8502` |
| polars/plotly import error | Re-run `pip install -r requirements.txt` from inside `legacy_app/` |
| `import vdcore` fails / `ModuleNotFoundError: No module named 'vdcore'` | Make sure you launched with `streamlit run app.py` from inside `legacy_app/` — the app inserts the repo root onto `sys.path` itself at import time |

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
| Synthesis tab shows Camber = 0.00° but Analysis shows −1.50° | Expected, not a bug | `SuspensionCorner.static_camber_deg()` always returns 0 (§13.1); the Analysis tab's camber comes from the sidebar's `vdcore` inputs instead |

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

### 14.5 CAD frame different from this app's (§5.1)

Obvious signs:
- `WHEEL_CENTER.z` should be the **tire radius** (~220-260 mm, positive)
- `CONTACT_PATCH.z` should be **0**
- Left corners (FL/RL) should have **positive** Y

Typical conversion:
```python
# If the CAD Z points downward with the origin at the wheel center:
Z_iso = TIRE_RADIUS - Z_cad

# If your CAD's Y is positive to the right (J670e-style):
Y_iso = -Y_cad
```

### 14.6 Automated synthesis

There isn't one anymore (§2, §8 Tab 4). If you're looking for the old
"set targets, click Run" workflow: it's retired because the optimizer scored candidates
on the legacy strut-to-midpoint solver. Iterate manually per §9 Scenario 2 instead.

### 14.7 Inputs tab loses data

If clicking "🔄 Apply file" in the sidebar makes the Inputs tab's editor discard your edits:
- This is the correct behavior: applying a file **overwrites** the editor state
- To keep your edits: click "✅ Apply as loaded hardpoints" in the Inputs tab BEFORE loading another file

---

## 15. Limitations

### 15.1 What it DOES ✅

- 3D kinematics in (heave, roll, steer), on the validated `DWSolver`
- 6+ static KPIs, 10+ dynamic KPIs (camber gain, RC migration/height, roll camber, bump steer, half-track change)
- Anti-dive/anti-squat and Ackermann from the real linkage
- Wheel rate / roll rate / natural frequency, given typed spring rate + motion ratio + mass
- CSV/Excel/JSON import and export
- An optional independent cross-check against Altair MotionSolve

### 15.2 What it DOESN'T do ❌

- **Vertical dynamics** beyond the plain wheel-rate/frequency algebra above (no ride simulation)
- **Arm compliance**
- **Pushrod/pullrod/rocker** modeling (no motion ratio derived from geometry — it's a typed input)
- **3D visualization driven by the validated solver** (the View 3D animation still uses the legacy solver, §8 Tab 3)
- **Automated hardpoint search** (retired, §2)
- **Optimization across all 4 corners together** (never existed even when the optimizer did)
- **Physical interference detection**
- **Tire load** (load transfer)
- **Damper F×v curves** (jounce/rebound damping are typed `% critical` values, not computed)

### 15.3 Important approximations

- **Roll axle-level, not per corner** — `DWSolver` finds the wheel travel per side that
  keeps both patches on the tilted road at your chosen roll angle; there is no
  roll-stiffness or spring input involved (§8 Tab 2)
- **Non-interpretable absolute toe** — reports **Δ toe** relative to static in sweeps
- **Rigid upright** — no compliance
- **Ackermann from a rack sweep**, not the plan-view construction — see §8 Tab 2
- **View 3D's sweep animation** still runs the legacy strut-to-midpoint solver (§8 Tab 3) — visualization only

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **A-arm / Wishbone** | "A"-shaped arm, FSAE standard |
| **Anti-dive / Anti-squat** | Side-view geometries that reduce dive/squat |
| **Ball joint** | Spherical joint (rod end) arm↔upright |
| **Bump** | Wheel rising relative to the chassis (heave +) |
| **Bump steer** | INVOLUNTARY toe variation with heave |
| **Camber** | Wheel inclination vs vertical |
| **Camber Gain** | d(camber)/d(heave) in °/mm |
| **Caster** | Kingpin inclination in the side view |
| **Compliance** | Elastic deformation (bushings, arms) |
| **Contact patch (CP)** | Tire-ground contact area |
| **DOF** | Degree of Freedom |
| **DWSolver** | `vdcore`'s validated 6-DOF kinematic solver (`vdcore.geometry.solver.DWSolver`) |
| **FSAE** | Formula SAE — student competition |
| **Hardpoint** | Pivot/attachment point |
| **Heave** | Vertical chassis-wheel displacement |
| **Inboard / Outboard** | Chassis side / wheel side |
| **Instant Center (IC)** | Instant center of rotation of the upright |
| **ISO 8855** | The axis convention this app and `vdcore` use — X forward, Y left, Z up (§5.1) |
| **Jounce** | Synonym of bump |
| **KPI** | Kingpin Inclination |
| **LBJ / UBJ** | Lower / Upper Ball Joint |
| **LCA / UCA** | Lower / Upper Control Arm |
| **Levenberg-Marquardt (LM)** | The `least_squares` algorithm used by the legacy `solver_3d.py` |
| **Mechanical Trail** | Longitudinal distance kingpin-ground to CP |
| **Motion Ratio** | Wheel displacement / spring displacement ratio — a typed input here, not derived |
| **Pickup point** | Synonym of hardpoint |
| **Pushrod / Pullrod** | Upright → rocker bar |
| **Rack** | Steering rack |
| **Rebound** | Wheel dropping (heave −) |
| **Rocker / Bell-crank** | Pushrod → spring lever |
| **Roll** | Chassis rotation about X |
| **Roll Axis** | Line joining RC F and RC R |
| **Roll Center (RC)** | Instant roll pivot (front view) |
| **Scrub Radius** | Lateral distance kingpin-ground to CP |
| **Seed** | Starting geometry used as a reference on the Synthesis tab (§8 Tab 4) |
| **Steer** | Steering |
| **Sweep** | Parametric sweep |
| **SVIC** | Side View Instant Center |
| **Tie-rod** | Steering bar rack→upright |
| **Toe** | Convergence/divergence |
| **TRO / TRI** | Tie Rod Outboard / Inboard |
| **Upright** | Steering upright |
| **`vdcore`** | The pure-library, actively-developed 3D kinematics engine this app delegates its dynamic KPIs to — see root `README.md` and `CLAUDE.md` |
| **Wheel Center (WC)** | Wheel center |

---

## 📞 About

Software developed as an educational project for Formula SAE teams. It is not a commercial product.

New engineering work belongs in `vdcore/`, not here — this app is frozen. Plausible
extensions, if this app is touched again:
- Pushrod/pullrod → extend `SuspensionCorner` and the 3D solver with a rocker model,
  so Motion Ratio stops being a typed input
- A `DWSolver`-based replacement for the retired synthesis optimizer, with a cost
  function cheap enough to call thousands of times per run
- SolidWorks integration → the COM API (Windows)

**Version:** 2026
