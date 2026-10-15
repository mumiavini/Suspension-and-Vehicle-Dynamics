# Projects

VD Studio project files (`*.vdproj.json`), one JSON file per car or variant.
Each file holds everything the designer owns. Derived values are never stored;
they are recomputed on open.

- **`FSAE_2027.vdproj.json`** is the team's shared car. VD Studio opens it on
  first start, and whenever there is no "last project" to reopen.
- Save your variants here too (Save as…), e.g. `FSAE_2027_high_rc.vdproj.json`.
  Commit the ones the team should see.
- Start-up order and saving: see `docs/VD_STUDIO.md`, "Project files".
- **`rocker_concepts/`**: three push/pull-rod + rocker layouts checked against the chassis CAD (`scripts/rocker_concepts.py build`), with a side-by-side report (`README.md`) and views. Each is also a variant here, `FSAE_2027_rocker_concept_<A|B|C>.vdproj.json`. Consequences only (snapshot of 2026-10-14).
- The designer chose concept A; its refinement (`rocker_concepts/concept_A_refined.md`) is now the team project's rocker layout. The previous team project is in `archive/FSAE_2027_before_rocker_concept_A.vdproj.json`.
