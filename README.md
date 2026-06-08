# M-body

[![CI](https://github.com/jrcheshire/mbody/actions/workflows/ci.yml/badge.svg)](https://github.com/jrcheshire/mbody/actions/workflows/ci.yml)
![license](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)
![python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![platform](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey.svg)
![built with MLX](https://img.shields.io/badge/built%20with-MLX-orange.svg)

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

## Features

- **Cosmology backend** (`mbody.cosmology`) -- flat-LCDM background `E(z)`, linear
  growth `D(z)` / `f(z)`, and the linear power spectrum from either CAMB (default)
  or the Eisenstein & Hu (1998) fitting formula; cross-checked against CCL.
- **Initial conditions** (`mbody.ic`) -- Gaussian random fields and local-`f_NL`
  ICs (`phi = phi_G + f_NL (phi_G^2 - <phi_G^2>)`), differentiable in `f_NL`, plus
  analytic squeezed-bispectrum templates.
- **LPT displacement** (`mbody.lpt`) -- first-order (Zel'dovich) and second-order
  (2LPT) displacement fields.
- **Particle-mesh dynamics** (`mbody.painting`, `mbody.forces`, `mbody.integrate`)
  -- CIC paint / read, an FFT Poisson force solve, and three leapfrog integrators:
  exact-background, FastPM (growth-corrected), and BullFrog (2LPT-accurate).
- **Reversible adjoint** (`integrate.adjoint_grad_ic` / `_fnl`) -- O(1)-in-steps
  memory gradients of summary statistics w.r.t. the IC parameters (`f_NL`, the
  linear amplitude `A`), the lever that keeps deep PM runs differentiable.
- **Redshift-space distortions** (`mbody.rsd`, `mbody.fields`) -- the line-of-sight
  velocity map, Kaiser multipoles (`P_0`, `P_2`) with discrete-shell decoupling,
  and CIC interlacing.
- **Summary statistics** (`mbody.fields`, `mbody.diagnostics`) -- `P(k)`,
  cross-power / propagator `r(k)`, the Scoccimarro bispectrum, the one-point PDF,
  skewness, and growth history.
- **Biased tracers** (`mbody.bias`) -- a local quadratic-bias tracer and the Dalal
  scale-dependent-bias reference (both shape and absolute amplitude).
- **Autodiff Fisher forecasts** (`mbody.fisher`) -- forecasts over
  `{f_NL, b1, b2, A, f_growth}`, including redshift-space multipoles and
  multi-tracer sample-variance cancellation (the `b_phi`-`f_NL` capstone).
- **Run driver, diagnostics, visualization** (`mbody.driver`, `mbody.diagnostics`,
  `mbody.viz`) -- one `run(SimConfig) -> RunResult` entry point that measures,
  renders a multi-panel dashboard, and serializes itself; an optional matplotlib
  layer kept off the import and hot paths.

## Install

```bash
pixi install
```

macOS on Apple Silicon only (MLX requirement); the pixi platform is pinned to
`osx-arm64`.

## Run

```bash
pixi run test     # the test suite
pixi run probe    # day-0 feasibility: does autodiff flow through mx.fft?
pixi run format   # black --skip-string-normalization
pixi run lint     # flake8
```

Headline demos (each writes figures under `outputs/`):

```bash
pixi run fisher            # linear-vs-PM autodiff Fisher over {f_NL, b1, b2, A}
pixi run convergence       # transfer / propagator / growth vs analytic linear theory
pixi run external-ccl      # cross-check the cosmology layer against CCL
pixi run bullfrog          # the BullFrog integrator probe
pixi run rsd               # redshift-space multipoles + the f_growth Fisher
pixi run multitracer       # the b_phi-f_NL multi-tracer cancellation capstone
pixi run rsd-multitracer   # the redshift-space multi-tracer composition
```

A whole run from a single config:

```python
import mbody

result = mbody.run(mbody.SimConfig())     # FastPM + 2LPT defaults, Gaussian ICs
k, Pk, n_modes = result.power()           # measured P(k) of the final field
result.dashboard(out="outputs/run.png")   # multi-panel diagnostic figure
```

## Documentation

- **[`docs/api.md`](docs/api.md)** -- a tour of the public API, module by module.
- **[`examples/`](examples/)** -- short, commented, runnable end-to-end scripts:
  a quickstart, the autodiff `dlnP/df_NL` headline, a Fisher forecast, and
  redshift-space multipoles.
- Topic notes in [`docs/`](docs/): the BullFrog integrator
  ([`bullfrog.md`](docs/bullfrog.md)), redshift-space distortions
  ([`rsd.md`](docs/rsd.md)), the multi-tracer capstone
  ([`multitracer.md`](docs/multitracer.md)), the external convergence cross-check
  ([`convergence_crosscheck.md`](docs/convergence_crosscheck.md)), and the
  as-built architecture ([`architecture-plan.md`](docs/architecture-plan.md)).
- [`ROADMAP.md`](ROADMAP.md) -- the staged build plan and validation history.

## Layout

```
mbody/      package: cosmology, ICs, LPT, PM force + integrators, painting,
            statistics, RSD, the autodiff Fisher, and the run driver
examples/   short, commented end-to-end usage examples (start here)
scripts/    richer experiments and figure-making demos (probe_*, plot_*)
tests/      unit and validation tests
docs/       the API reference and topic notes (integrators, RSD, multi-tracer)
```

## Relationship to SPHEREx

A sibling of the author's SPHEREx L4 cosmology work. It is **not** on
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
