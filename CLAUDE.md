# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in the M-body project.
A sibling of `~/spherex/SPHEREx-L4-Cosmology-Pipeline/`; see its CLAUDE.md for
shared conventions and the broader SPHEREx context.

## Session start

Read all memory files in
`~/.claude/projects/-Users-jamie-spherex-mbody/memory/` in parallel before
starting work. The MEMORY.md index loads automatically; the individual files
do not.

## Project description

M-body is a differentiable particle-mesh (PM) N-body toy built on MLX (Apple's
array framework) and run on Apple Silicon. The goal is a fully differentiable
forward model of structure formation -- Gaussian or local-f_NL ICs, LPT
displacement, a few leapfrog PM steps -- so that `mx.grad` yields gradients of
summary statistics w.r.t. f_NL (and later cosmology / bias). The headline
target is autodiff `dlnP(k)/df_NL` vs the analytic Dalal et al. (2008)
scale-dependent bias. See README.md and ROADMAP.md.

It is an educational / intuition-building project, **not** production science.
No halo finding (FoF mass is non-differentiable; a PM under-resolves halos) --
chase the field-level `dP/df_NL`. Droppable for pmwd / FlowPM / DISCO-DJ for
real runs.

## Relationship to SPHEREx

Not on the SPHEREx critical path. The production f_NL path is DasResultat
(Cartesian P_l(k) inference, Cobaya) fed by estimators (pypower / SuperFaB),
with an SFB emulator for cheap likelihoods. The scientific reason this toy
exists: the dominant SPHEREx f_NL systematic is the galaxy bias `b_phi` and
its degeneracy with f_NL (the v28 forecast assumes the Dalal / Barreira
universality relation `b_phi = 2 * delta_c * (b1 - 1)`). A differentiable
forward model is a sandbox for that lever. See memory:
`project_mbody_overview.md`, `reference_fnl_bphi_target.md`.

## Environment

Pixi, pinned to `osx-arm64` (MLX is Apple-Silicon only):

```bash
pixi install
pixi run probe         # day-0: autodiff through mx.fft (ROADMAP step 0)
pixi run test          # pytest smoke tests
pixi run format        # black --skip-string-normalization
pixi run check-format  # black --check (no modify)
pixi run lint          # flake8
```

`pixi.lock` co-commits with `pyproject.toml`: any pyproject change requires
re-running `pixi install` and committing the regenerated lock in the same
commit.

## Code style

- `black --skip-string-normalization` (matches the parent SPHEREx repos).
- `flake8`, max line length 88 (`.flake8`).
- **No unicode in source files** (comments, strings, identifiers). Spell out
  Greek (`sigma`, `delta`, `phi`) and use ASCII arrows (`->`).
- Python scripts, not notebooks. Save figures as PNGs under `outputs/`
  (gitignored).

## Hardware

128 GB unified-memory Apple Silicon MacBook Pro -- generous for PM meshes. The
binding memory constraint is the autodiff graph: it unrolls over leapfrog
steps, so reverse-mode memory scales with (grid size) x (number of steps).
Plan grids / steps around that, and use checkpointing when it bites.

## Layout

```
mbody/      package: PM kernels, ICs, statistics (being built per ROADMAP)
scripts/    runnable experiments / demos
tests/      unit + smoke tests
```

## Diagnostics & the run driver

- `mbody.run(SimConfig) -> RunResult` (`mbody/driver.py`) is the single entry
  point: it threads the whole config through ic -> LPT -> leapfrog, and the
  result measures + renders + serializes itself (`.power`, `.cross_with_ic`,
  `.growth_history`, `.dashboard`, `.save`).
- `mbody/diagnostics.py` holds the shared, off-AD-path estimators (P(k),
  cross-correlation `r(k)`, growth history, skewness / one-point PDF) and the
  memory-gated `SnapshotRecorder` (the `snapshot(step, a, x, p)` seam, stores
  only bounded reductions). Reuse these -- do not re-implement diagnostics per
  script.
- `mbody/viz.py` is the optional matplotlib layer (`density_slice`,
  `animate_slab`, six-panel `dashboard`). It is intentionally NOT imported by
  `mbody.__init__`, so `import mbody` stays plotting-free / off the hot path --
  import `mbody.viz` explicitly in scripts.
- **Honest-config invariant:** SimConfig defaults name only implemented physics.
  Currently `integrator="fastpm"` (growth-corrected kick/drift, exact linear
  growth at any step count) and `lpt_order=2` (2LPT); `"exact"` and `lpt_order=1`
  are the simpler implemented alternatives. `"bullfrog"` (2LPT-accurate
  drift-kick-drift, Rampf+24) is now ALSO implemented and selectable -- all three
  integrator enum values run; `_check_supported` stays as a guard for any future
  reserved value. fastpm remains the default (bullfrog's per-step advantage is
  resolution-gated; see docs/bullfrog.md). When adding an integrator or LPT order,
  implement + validate it before making it the default.
  The loose API agrees: `integrate.initial_state` / `leapfrog` / `adjoint_grad_fnl`
  default `lpt_order=2`, and `leapfrog` resolves `integrator` from `time.integrator`
  (default fastpm) -- so the loose and config paths give the same physics. Pass
  `lpt_order=1` / `integrator="exact"` explicitly for the simpler variants.
  Redshift-space distortions are the same way: `SimConfig.rsd =
  RedshiftSpace(enabled=False, los_axis=0, f_growth=1.0)` is OFF by default (a
  plain run measures the real-space field); enable it to map the final particles
  to redshift space and expose `RunResult.power_multipoles()`.

## Redshift-space distortions (RSD)

- Opt-in via `RedshiftSpace`. The map (`mbody.rsd.redshift_space_positions`) shifts
  the line-of-sight coordinate by `Delta_s = f_growth * p_los/(a^2 E)` -- derived
  from mbody's H0=1 drift, so NO stray h factor (verified: a Zeldovich field's RSD
  shift = f x real-displacement to ~2e-7). **Use a FULL fft axis for the LOS**
  (`los_axis` 0/1), never the rfft axis 2 (its `kz=0` plane skews the discrete
  multipole average).
- Multipoles: `fields.band_power_multipole` (raw, differentiable -- the Fisher
  data vector) and `fields.power_multipoles` (diagnostic, with the discrete-shell
  `multipole_decoupling` so it matches continuum Kaiser). **`P_0`+`P_2` are
  science-grade; `P_4` is noise-limited at toy box sizes** (signal ~0.04 P0).
  `fields.interlaced_density_contrast` removes the CIC aliasing upturn near Nyquist
  (validated). It is now the **standard measurement painter** for ALL P(k)/band-power
  diagnostics and the Fisher Jacobians (real- and redshift-space), via
  `diagnostics.particle_power(interlace=True)`, the driver's `ic_field`/`final_field`,
  and `fisher.pm_{logP,multipole}_jacobian`; ONLY the force solve
  (`forces.forces_on_particles`) keeps plain CIC. The f_NL signal (low k) and the
  autodiff `d ln P/d theta` (window cancels in the log-derivative) are unchanged --
  the win is honest high-k diagnostics. On the nonlinear z=0 field interlacing is a
  small correction (genuine power dominates aliasing); the big gain is on
  low-amplitude/clean fields.
- Fisher: `fisher.{linear,pm}_multipole_jacobian` over `PARAM_NAMES_RSD =
  {f_NL,b1,b2,A,f_growth}` with LINEAR `P_ell` data (P2 can be negative -> no log)
  and a Gaussian mock block covariance (`multipole_gaussian_covariance`);
  `FisherForecast` now takes `covariance=` (full) as well as `variances=` (diag).
  `f_NL`/`A` use `adjoint_grad_ic(..., loss_uses_momentum=True)` -- the adjoint seed
  was generalized to a loss of (x_final, p_final) because the redshift field depends
  on the final velocities; `b1`/`b2`/`f_growth` are downstream cheap grads.
- **Honest finding:** the quadrupole sharply pins `f_growth` (>10x) and PARTIALLY
  recovers `sigma(f_NL)` (~1.6x), but does NOT break the `b_phi*f_NL` degeneracy
  (only the k^-2 shape + multi-tracer do). Fingers-of-god are absent in a PM, so
  trust RSD only at k < ~0.1 h/Mpc. See `docs/rsd.md`, `pixi run rsd`.

## External convergence cross-check

The forward model is validated against references OUTSIDE mbody (ROADMAP
"External convergence cross-check"). Two layers built; pmwd (a differentiable-PM
peer) is deferred. (1) **Analytic linear theory** -- matched-phase transfer
`T(k) = P_PM/P_lin -> 1` at large scales, propagator `r(k)`, and growth
convergence (`scripts/probe_pk_convergence.py`, `pixi run convergence`,
`tests/test_convergence.py`); added `fields.cic_window` + a
`power_spectrum(deconvolve_cic=)` flag (particle-painted P(k) carries the CIC
window; a grid field does not). (2) **Absolute f_NL bias amplitude** --
`dlnP_h/df_NL = 4 b2 sigma^2/(b1 M(k))` (`bias.scale_dependent_bias_response`,
its bin-averaged sibling `_binned`, and `bias.mesh_variance`), matched to autodiff
to ~1-2% at low k; this upgrades the old SHAPE-only `scale_dependent_shape`
overlay to an ABSOLUTE one (use `_binned` to match the band-power shell sum, like
`ic.local_bispectrum_binned`). (3) **CCL anchor** -- `pyccl` is now a pixi dep;
`scripts/probe_external_ccl.py` (`pixi run external-ccl`) + the GATED
`tests/test_external_ccl.py` (skip if `pyccl` absent) cross-check growth / linear
P(k) / sigma_R against the community code (EH98 vs CCL ~1e-6). Mind the units: CCL
is physical Mpc, mbody is Mpc/h (`k_ccl = k h`, `P_mbody = P_ccl h^3`,
`R_ccl = R/h`).

## Key references

- Dalal et al. 2008 (arXiv:0710.4560) -- local-f_NL scale-dependent bias.
- pmwd, FlowPM, FastPM, DISCO-DJ -- differentiable / PM prior art.
- Bock et al. 2025 (arXiv:2511.02985) -- SPHEREx mission (sigma(f_NL) < 0.5).
- MLX: https://github.com/ml-explore/mlx
