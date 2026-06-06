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
  are the simpler implemented alternatives. `"bullfrog"` is the one reserved enum
  value `run()` still rejects with `NotImplementedError`. When adding an
  integrator or LPT order, implement + validate it before making it the default.
  NB `integrate.leapfrog` / `initial_state` keep their own `lpt_order=1` default
  (loose API); the fastpm+2LPT defaults live in SimConfig (the driver path).

## Key references

- Dalal et al. 2008 (arXiv:0710.4560) -- local-f_NL scale-dependent bias.
- pmwd, FlowPM, FastPM, DISCO-DJ -- differentiable / PM prior art.
- Bock et al. 2025 (arXiv:2511.02985) -- SPHEREx mission (sigma(f_NL) < 0.5).
- MLX: https://github.com/ml-explore/mlx
