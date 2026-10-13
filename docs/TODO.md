# TODO — possible future upgrades

Ideas that are deliberately **not** done yet, in priority order. None of them
is needed for the current numbers to be right; each says why it would matter
and what would trigger doing it.

## Priority list

### 1. Rod-end misalignment per joint, with the bolt axis as an input

The KPI is a screening value today: the swing of the rod direction over the
bump × steer envelope. Rotation about the bolt is free, so this over-states
the real misalignment. A per-joint cone angle needs each rod end's bolt axis
as an input: the chassis frame for the inner joint, the upright frame for the
outer one. Show it only when the axes are given, and keep the screening value
otherwise. This matters for rod-end selection, once the screening value gets
close to its band.

### 2. "Outer overhang" input for the hairpin check

The turning radius (L / sin δo) is the path of the outer front contact-patch
centre. The car's outermost point (tyre sidewall, wing endplate) lies
outboard of it. Add a design_intent input for that overhang and plot its path
against the FSAE hairpin's 4.5 m outside radius. The default car has about
1 m of margin, so this is not urgent.

### 3. Cosmetic polish

- Re-style existing plots when the theme is toggled: they keep the old
  background until they are redrawn.
- Column widths and elided long text on the Setup sheet, the Concept
  estimator and the Hardpoints pages.

### 4. Compiled kernel for the kinematic solver (Rust, via PyO3/maturin)

Everything is Python today (`vdcore` on numpy/scipy). That is fast enough for
the app: a static KPI recompute takes about 30 ms and the swept KPIs about
2 s. It is not fast enough for automated hardpoint synthesis: `DWSolver` is
too slow per solve for a `differential_evolution`-style cost function (see
CLAUDE.md, "Where numbers come from"). Trigger: bringing back an optimizer, or
large trade-off grids.

1. Profile, and first try the cheap Python wins: an analytic Jacobian for the
   9 residuals (validated against the numerical one, per the numerics
   convention), vectorised sweeps, and warm-starting each solve from the
   previous travel.
2. Only if that is not enough, move the residuals and the solver loop into a
   Rust extension module. It must keep the same `CornerSolver` interface and
   the same convergence reporting (never a plausible number on failure), and
   pass the existing benchmarks within tolerance.

`vdcore` must stay importable from a bare Python REPL. The Rust module is
therefore an optional accelerator, and the Python solver stays as the
fallback and the reference.

## Done

- Full 6×6 upright equilibrium for the link forces, braking included
  (`vdcore.analysis.upright_loads`; Quasi-static maneuver page, "Link forces"
  tab). Validated against virtual work on `DWSolver`.

## Blocked — waits for TTC tyre data

Anything that needs tyre data: pneumatic trail, slip-angle steering effort,
and ride rate from a measured tyre vertical rate.
