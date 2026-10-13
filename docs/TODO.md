# TODO — possible future upgrades

Ideas that are deliberately **not** done yet. None of them is needed for the
current numbers to be right; each says why it would matter and what would
trigger doing it.

## Performance

- **Compiled kernel for the kinematic solver (Rust, via PyO3/maturin).**
  Everything is Python today (`vdcore` on numpy/scipy). That is fast enough
  for the app: a static KPI recompute takes about 30 ms, the swept KPIs about
  2 s. It is not fast enough for automated hardpoint synthesis:
  `DWSolver` is too slow per solve for a `differential_evolution`-style cost
  function (see CLAUDE.md, "Where numbers come from"). Trigger: bringing back
  an optimizer or large trade-off grids.
  Do it in steps:
  1. Profile, and first try the cheap Python wins: an analytic Jacobian for
     the 9 residuals (validated against the numerical one, per the numerics
     convention), vectorised sweeps, and warm-starting each solve from the
     previous travel.
  2. Only if that is not enough, move the residuals + solver loop to a Rust
     extension module. It must keep the same `CornerSolver` interface, the
     same convergence reporting (never a plausible number on failure) and pass
     the existing benchmarks bit-for-bit within tolerance.
  `vdcore` must stay importable from a bare Python REPL, so the Rust module
  is an optional accelerator with the Python solver as the fallback and the
  reference.

## Analysis

- **Wishbone leg forces in braking from a full upright equilibrium.** The
  current two-force UCA cannot react the Fx moment, so the braking and
  combined cases are reported as "not computed". The fix is a 6×6 upright
  solve: 4 legs + tie rod/toe link + pushrod, with the pushrod direction as
  an input tagged `estimate`.
- **Rod-end misalignment per joint.** The KPI today is a screening value: the
  swing of the rod direction. A per-joint cone angle needs the bolt axis of
  each rod end as an input (chassis frame for the inner joint, upright frame
  for the outer one).
- **Outer overhang for the hairpin check.** The turning radius uses the
  contact-patch centre. Add an input for the car's outermost point (tyre
  sidewall, wing endplate) and plot its path against the 4.5 m line.
- **Anything that needs tyre data** (pneumatic trail, slip-angle effort,
  ride rate from a measured tyre rate) waits for TTC data.

## UI polish

- Re-style existing plots when the theme is toggled (they keep the old
  background until redrawn).
- Column widths and elided long text on the Setup sheet, Concept estimator
  and Hardpoints pages.
