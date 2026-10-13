# TODO — possible future upgrades

Ideas that are deliberately **not** done yet, in priority order. None of them
is needed for the current numbers to be right; each says why it would matter
and what would trigger doing it.

## Priority list

### 1. Tyre data (TTC) — acquire, fit, then unlock the tyre-dependent analyses

PUCPR Racing has no tyre data yet, so nothing in the tool uses a tyre model
(CLAUDE.md, "no tire data"). Several numbers are therefore assumptions tagged
`estimate`, or are left out on purpose. This is the biggest single upgrade in
what the tool can say, and it is blocked only on data.

1. **Acquire.** Join the FSAE Tire Test Consortium (Calspan TIRF data) and get
   the rounds for the tyre the team will run (size, compound, rim width):
   cornering (Fy, Mz, Mx vs slip angle), drive/brake (Fx vs slip ratio) and
   combined runs, over the loads, inflation pressures and camber angles the car
   will see. Record the round, tyre, rim and pressure as provenance.
2. **Fit.** `vdcore/tire/` is a stub. Fit Pacejka MF 5.2 (Fy, Fx, Mz, combined
   slip) in the ISO 8855 frame (the TTC data is SAE: convert once, on import,
   with a test that pins the signs; see the `tire-model` skill). Report the fit
   residuals by load and camber. Keep the raw data and the fit with their
   provenance.
3. **Unlock, each as information, never as a pick:**
   - **Loaded radius and vertical rate** vs load and pressure. They replace
     the `estimate` loaded radius (synthesis and hardpoint mode) and the
     optional tyre vertical rate on the setup sheet, which gives ride rate and
     ride frequency.
   - **Pneumatic trail and aligning torque.** Steering effort at speed, and
     parking effort with patch twist (bore torque) instead of the lever-arm
     friction model with an assumed μ.
   - **Camber sensitivity** (Fy vs camber and load). It turns the road-camber
     results of the kinematic sweeps, the maneuver and the telemetry replay
     into lateral-force consequences. The design-intent camber bands then get
     data next to them.
   - **Load sensitivity.** The front/rear lateral capacity as a function of
     LLTD, shown as a consequence curve; the tool still does not choose the
     LLTD.
   - **Friction coefficients.** μ lateral / longitudinal in the load cases
     and the link-force tables from the data (at the case's load and camber),
     instead of the 1.5 / 1.4 assumptions.
   - **Telemetry.** Slip-angle estimates and predicted vs measured ay at a
     given steer.
4. **Provenance.** Every number that moves from an assumption to the fit
   changes its source tag and cites the TTC round in reports.

### 2. Compiled kernel for the kinematic solver (Rust, via PyO3/maturin)

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
- Rod-end misalignment per joint, with each bolt axis as an input
  (`kpis.rod_end_misalignment`; Steering page, "Rod-end bolts" tab): front tie
  rod and rear toe link, inner (chassis-fixed) and outer (upright-fixed) bolt,
  asin(|rod · bolt|) over the travel × rack envelope.
- "Outer overhang" for the hairpin check (`kpis.turning_radii_m`; Steering
  page, "Turning envelope" tab and "Turning radius" plot): the outer front
  tyre's plan-view outline and an optional body point, compared against the
  FSAE hairpin's 4.5 m outside radius.
- Cosmetic polish: plots restyle on a theme toggle; tables no longer wrap long
  source text (elided, full text in the tooltip); the Concept, Hardpoints and
  Setup-sheet pages fit their columns.
