# M-body

A differentiable particle-mesh (PM) N-body toy built on
[MLX](https://github.com/ml-explore/mlx), Apple's array framework, run on
Apple Silicon. "M" for Mac / MLX; "body" for N-body.

The aim is a fully differentiable forward model of large-scale structure --
Gaussian or local-`f_NL` initial conditions, LPT displacement, a handful of
leapfrog PM steps -- so that gradients of summary statistics with respect to
cosmological and primordial-non-Gaussianity parameters fall straight out of
`mx.grad`. The headline target is **autodiff `dlnP(k)/df_NL`**, compared
against the analytic Dalal et al. (2008) scale-dependent bias
`Delta b(k) ~ f_NL / k^2`.

This is an educational / intuition-building project, not production science
code (see "Scope").

## Why MLX / Apple Silicon

- **Unified memory.** No host<->device copies; a 128 GB Apple Silicon laptop
  can hold the PM meshes and the unrolled autodiff graph that a discrete-GPU
  setup of the same nominal size could not.
- **Composable autodiff + lazy evaluation** behind a NumPy-like API.
- The interesting (and risky) part is autodiff *through the FFT* -- which the
  PM force solve and the P(k) estimator both depend on. `pixi run probe`
  checks this on day one.

## Scope

- **Is:** a small, readable, differentiable PM for studying how the late-time
  field and its power spectrum respond to `f_NL` (and, later, cosmology and
  galaxy bias).
- **Is not:** a production sim. No halo finding -- FoF mass is a step function
  (non-differentiable) and a PM under-resolves halo mass, so the whole point is
  to chase field-level `dP/df_NL`, not a mass function. For any real science
  run this is droppable for [pmwd](https://github.com/eelregit/pmwd),
  [FlowPM](https://github.com/DifferentiableUniverseInitiative/flowpm), or
  DISCO-DJ.

## Install

```bash
pixi install
```

macOS on Apple Silicon only (MLX requirement); the pixi platform is pinned to
`osx-arm64`.

## Run

```bash
pixi run probe    # day-0 feasibility: does autodiff flow through mx.fft?
pixi run test     # smoke tests (package import + MLX array / grad)
pixi run format   # black --skip-string-normalization
pixi run lint     # flake8
```

## Layout

```
mbody/      package: PM kernels, ICs, statistics (being built; see ROADMAP)
scripts/    runnable experiments and demos (probe_ad_fft.py is the first)
tests/      unit / smoke tests
```

## Relationship to SPHEREx

A sibling of the SPHEREx L4 cosmology work in `~/spherex/`. It is **not** on
the critical path: the production `f_NL` inference path is DasResultat
(Cartesian `P_l(k)`, Cobaya) fed by the estimators (pypower / SuperFaB), with
an SFB emulator for cheap likelihoods. M-body exists to build intuition for
the real scientific lever -- the galaxy bias `b_phi` and its degeneracy with
`f_NL` -- in a setting where the whole forward model is differentiable end to
end.

## References

- Dalal, Dore, Huterer & Shirokov 2008,
  [arXiv:0710.4560](https://arxiv.org/abs/0710.4560) -- scale-dependent bias
  from local `f_NL`.
- Differentiable / PM prior art: [pmwd](https://github.com/eelregit/pmwd),
  [FlowPM](https://github.com/DifferentiableUniverseInitiative/flowpm),
  FastPM, DISCO-DJ.
- Bock et al. 2025, [arXiv:2511.02985](https://arxiv.org/abs/2511.02985) --
  SPHEREx mission (`sigma(f_NL) < 0.5` target).
- [MLX](https://github.com/ml-explore/mlx).
