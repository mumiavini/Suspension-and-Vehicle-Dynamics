# CAD surveys

`scripts/survey_chassis.py` (or drag a `.SLDPRT` onto `Survey chassis CAD.bat`)
writes `<model>_survey.json` and an isometric `.png` here: the model's sketches,
reference geometry and bounding boxes in its own frame, in mm. Analysed by
anyone, without SolidWorks. See `docs/VD_STUDIO.md`, "Checking the chassis CAD".

## Suspension solids for SolidWorks (front of concept DF, rear of concept H)

The team project's rocker layout (`projects/rocker_concepts/`; earlier
projects are in `projects/archive/`):

- front, concept DF (`concept_nodes_DEF.md`; 220 lb/in springs, MR 0.80):
  pushrod on the LCA; rocker pivot 49.0 mm from the front-hoop side node,
  coil-over chassis eye 57.5 mm from the front-hoop bottom node. A front for
  550 lb/in springs is still being chosen;
- rear, concept H (`concept_550_lbin.md`; 550 lb/in springs, MR 0.506, the
  same 24.7 N/mm at the wheel): pushrod from a clevis on the UCA rear leg (the
  bracket is in the UCA part) up to a rocker 53.5 mm from the UCA rear pivot
  node, coil-over down to an eye 52.1 mm from the LCA rear pivot node. Checked
  to +-36 mm of wheel travel: it needs a bump stop and a droop limiter, which
  are not drawn.

`suspension_DF-H_solids_*.bas` build the whole suspension of the team
project as primitive solids (one cylinder per member: wishbones, bushes, ball
joints, simplified uprights and hubs, hollow tyres, tie rods / toe links, rack,
pushrods / pullrods and rod ends, rockers with pickup bolts, coil-overs with
eyes and spring, and stand-offs from the rocker pivots and coil-over ends to
the nearest bodied chassis tube) plus a centreline skeleton per corner.
Regenerate with
`uv run python scripts/export_suspension_solids.py --name suspension_DF-H_solids`
(both versions below); `*_preview.svg` shows the result over the chassis and the engine.

Two versions, same geometry:

- `..._Xasis_33.3.bas` is in the chassis part's coordinates. Open any document,
  run the macro (Tools > Macro > New, save a blank .swp, File > Import File...
  the .bas, F5): it creates a new part with the solids. Save it, insert it in
  `MONT_CHASSI_SUSP_MOTOR.SLDASM` and mate its origin and three planes to
  Xasis 33.3's.
- `..._iso.bas` is in ISO 8855 numbers, like `susp.SLDPRT`. Open `susp.SLDPRT`
  (already placed in the assembly) and run it there: a part whose name
  contains "susp" is used as is.

Re-running replaces the `SUSP_...` bodies and sketches it made before. The
diameters are placeholders (`SIZES` in the script), not a detailed or
structural design. Use Tools > Evaluate > Interference Detection in the
assembly: the real engine mesh is only there (the evaluator sees its features).

## Suspension assembly for SolidWorks (one part per component, mated)

`suspension_DF-H_assembly_Xasis_33.3.bas` builds the same solids split
into parts, in a folder it asks for (default: `SUSP_2027\` next to the .swp):

```
SUSP_2027.SLDASM             top: corners (flexible), SUSP_rack, SUSP_skeleton (fixed)
├─ SUSP_FL.SLDASM            one per corner (FL, FR, RL, RR)
│   ├─ SUSP_FL_chassis_mounts    fixed: rocker / coil-over stand-offs (frame tabs
│   │                            for the chassis team) + chassis pick-up references
│   ├─ SUSP_FL_UCA, SUSP_FL_LCA  legs, bushes (+ the rod bracket if the rod is on it)
│   ├─ SUSP_FL_upright           kingpin, spindle, arm, hub, ball-joint housings
│   ├─ SUSP_FL_tie_rod           (rear: toe_link)
│   ├─ SUSP_FL_pushrod, SUSP_FL_rocker   (rear: SUSP_RL_pushrod; its clevis
│   │                            bracket is in SUSP_RL_UCA)
│   ├─ SUSP_FL_damper_body       body, chassis eye, spring
│   ├─ SUSP_FL_damper_shaft      shaft, rocker eye
│   └─ SUSP_FL_wheel
```

Every part is modelled in the chassis part's coordinates and carries named
reference points (`P_...`) and axes (`A_...`) at its joints. The corner mates are
coincident mates between those references: wishbones and rocker revolve on the
chassis axes, ball joints and rod ends are spherical, the damper's body and shaft
slide on one axis, the wheel spins on the upright. Steering is held straight
ahead (the tie rods' inner joints are on the chassis part). Drag a wheel and the
corner moves; the generator checks the scheme before writing the macro (no
redundant mate, one DOF that moves the joints, damper/wheel ratio equal to
vdcore's: 0.800 front, 0.506 rear at ride).

How to run: open SolidWorks with no suspension file open, Tools > Macro > New
(save a blank .swp where the files should go), File > Import File... the .bas,
open the module `SUSP_2027_build` it adds, click inside `Sub main` and press F5
(the .swp's own module has an empty `main`). It writes 47 files and
`SUSP_2027_log.txt` (send the log if a step fails), then opens `SUSP_2027`.
The parts are built without windows; each assembly opens a window while its
components and mates go in (SolidWorks inserts nothing into a hidden assembly).
Then insert `SUSP_2027.SLDASM` into `MONT_CHASSI_SUSP_MOTOR.SLDASM` and mate its
origin and three planes to Xasis 33.3's; suppress `susp` there.

Re-running rebuilds every file it made, so close the assemblies that use them
first; edits made inside those files are lost. Swap in real parts (the
team's wheel, coil-over) once the layout is settled: Replace Components, then
re-mate. Wheel: its axis concentric with `A_SPIN` of the upright, its mid-plane
coincident with `P_WHEEL_CENTER`. Coil-over: each eye centre coincident with
`P_DAMPER_CHASSIS` (chassis_mounts) and `P_DAMPER` (rocker), the coil-over made
a flexible sub-assembly. Regenerate with
`uv run python scripts/export_suspension_assembly.py --name suspension_DF-H_assembly`
(`--frame iso` for ISO numbers). An anti-roll bar is not in the assembly yet (its arms and drop links
need their own joints); the single-part macro takes `--arb`.
