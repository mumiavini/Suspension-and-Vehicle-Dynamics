# Projects

VD Studio project files (`*.vdproj.json`), one JSON file per car or variant.
Each file holds everything the designer owns. Derived values are never stored;
they are recomputed on open.

- **`FSAE_2027.vdproj.json`** is the team's shared car. VD Studio opens it on
  first start, and whenever there is no "last project" to reopen.
- Save your variants here too (Save as…), e.g. `FSAE_2027_high_rc.vdproj.json`.
  Commit the ones the team should see.
- Start-up order and saving: see `docs/VD_STUDIO.md`, "Project files".
