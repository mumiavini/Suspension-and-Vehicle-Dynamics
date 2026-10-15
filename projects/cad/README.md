# CAD surveys

`scripts/survey_chassis.py` (or drag a `.SLDPRT` onto `Survey chassis CAD.bat`)
writes `<model>_survey.json` and an isometric `.png` here: the model's sketches,
reference geometry and bounding boxes in its own frame, in mm. Analysed by
anyone, without SolidWorks. See `docs/VD_STUDIO.md`, "Checking the chassis CAD".

## Suspension solids for SolidWorks (concept A refined)

`suspension_A_refined_solids_*.bas` build the whole suspension of the team
project as primitive solids (one cylinder per member: wishbones, bushes, ball
joints, simplified uprights and hubs, hollow tyres, tie rods / toe links, rack,
pushrods and rod ends, rockers with pickup bolts, coil-overs with eyes and
spring, and stand-offs from the rocker pivots and coil-over ends to the nearest
bodied chassis tube) plus a centreline skeleton per corner. Regenerate with
`uv run python scripts/export_suspension_solids.py --name suspension_A_refined_solids`;
`*_preview.svg` shows the result over the chassis and the engine.

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
