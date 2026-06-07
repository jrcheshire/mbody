# Multi-tracer Fisher: the b_phi-f_NL degeneracy capstone

The differentiable forward model's on-target capstone. The dominant SPHEREx
`f_NL` systematic is the galaxy bias `b_phi` and its degeneracy with `f_NL`: a
single tracer measures only the *product* `f_NL * b_phi` (the amplitude of the
`k^-2` scale-dependent bias), not `f_NL`. The multi-tracer technique
(Seljak 2009; Barreira & Krause 2023) is the lever against this. This module
reproduces that structure in mbody's autodiff Fisher -- a thing the production
finite-difference Fisher cannot do.

All of it lives in `mbody/fisher.py` (+ `bias.scale_dependent_bias_tracer`), with
`scripts/probe_multitracer.py`, `scripts/plot_multitracer_fisher.py`, and
`tests/test_multitracer.py`. It is loose-API-only, like the single-tracer Fisher
(not a `SimConfig` run-mode).

## The two key facts

1. **The degeneracy is a PRODUCT degeneracy, invisible at `f_NL = 0`.** In the
   scale-dependent bias `delta_h(k) = [b1 + b_phi f_NL / M(k)] delta(k)`, the
   derivatives `dP/df_NL ~ b_phi` and `dP/db_phi ~ f_NL` share the SAME `1/M(k)`
   shape, so they are perfectly degenerate (the Fisher block is rank-1) for any
   `f_NL != 0`. At `f_NL = 0` the `dP/db_phi` term vanishes and the two decouple
   -- a Fisher *null forecast* sees no degeneracy. So the degeneracy must be
   demonstrated at a **non-zero fiducial `f_NL`** (a "marginal detection").

2. **The multi-tracer gain is shot-noise-limited.** Two tracers of the same field
   cancel cosmic variance only down to their shot noise (`~sqrt(P n / 2)`); with
   noiseless continuous fields the cancellation is formally perfect and the
   forecast vacuous. So per-tracer Poisson shot noise `1/n_i` is built into the
   covariance.

## Two tracer models

**Native tracer** (`bias.local_bias_tracer`, the b2-sourced one). Here `b_phi` is
emergent: `dlnP_h/df_NL = 4 b2 A sigma^2 / (b1 M(k))`, i.e. `b_phi = 2 b2 A
sigma^2`. Used for the **detection regime** (`f_NL = 0`): it shows the
sample-variance cancellation cleanly *through the PM* (the f_NL/A columns flow
through the reversible adjoint). Its limitation: at high `k` the `b2` broadband
loop self-calibrates `b_phi` (a handle the real `b_phi` lacks), so it reproduces
the degeneracy faithfully only at low `k`.

**Explicit-b_phi tracer** (`bias.scale_dependent_bias_tracer`). `b_phi` is a free
`k^-2` parameter on a Gaussian field, decoupled from `b2` -- so the degeneracy is
clean at **all `k`**. Used for the **degeneracy regime** (`f_NL != 0`). It is a
linear-bias statement, so it is forecast on the linear field only (the PM
evolution does not change the (f_NL, b_phi) structure; the PM autodiff showcase is
the native model's job). Universality is the clean `b_phi = 2 delta_c (b1 - 1)`.

## Parameters, data vector, covariance

- Native: `PARAM_NAMES_MT = (f_NL, A, b1_A, b2_A, b1_B, b2_B)`. Explicit:
  `PARAM_NAMES_MT_BPHI = (f_NL, A, b1_A, bphi_A, b1_B, bphi_B)`. Shared IC params
  (f_NL, A) first; per-tracer bias blocks after.
- Data vector `mu = [P_AA(k), P_AB(k), P_BB(k)]`, spectrum-major, RAW linear band
  powers (the cross `P_AB` is not positive-definite -> no log) -> a full block
  covariance.
- Jacobians: `linear_multitracer_jacobian` / `pm_multitracer_jacobian` (native),
  `linear_multitracer_bphi_jacobian` (explicit). In the PM case f_NL and A share
  ONE reversible-adjoint sweep per data component (the loss paints both tracers
  from the same final field), and the four b's are cheap downstream gradients.
- Covariance: `multitracer_gaussian_covariance` (native) /
  `multitracer_bphi_gaussian_covariance` (explicit) are mock-based with per-tracer
  shot noise -- the ground truth (they capture the colored field + weak
  non-Gaussianity). `multitracer_analytic_covariance` is the clean Gaussian 3x3
  block `Cov(P_ij,P_kl) = (Ptot_ik Ptot_jl + Ptot_il Ptot_jk)/N_modes` with
  `Ptot_ii = P_ii + 1/n_i`, `N_modes = 2/Var[ln P_b]` (carrying the rfft-plane
  double-count); it is exact on white noise and inflates only the steep low-k bins
  on a colored field.
- Universality tie: `universality_tie_matrix` (native) /
  `universality_bphi_tie_matrix` (explicit) give a constant chain-rule map `T`;
  `J_tied = J @ T` reparametrizes to the tied basis `(f_NL, A, b1_A, b1_B)`.
  `multitracer_forecast(J, cov, fiducial, param_names, priors=, tie=)` assembles
  the free or tied `FisherForecast`.

## Results (measured; see `pixi run probe-multitracer`)

**Detection regime (native, f_NL=0): sample-variance cancellation.** Two tracers
tighten `sigma(f_NL)` (universality-tied) over one:
- linear field: `836 -> 335` (**2.5x**),
- PM-evolved: `6677 -> 1547` (**4.3x**).

**Degeneracy regime (explicit b_phi, f_NL=100, L=1024):**
- **FREE** (`b_phi` marginalized): `f_NL` alone is degenerate -- the Fisher is
  singular in (f_NL, b_phi) (condition number `~1e23`), and only the product
  `f_NL*b_phi` is constrained (`sigma(f_NL*b_phi_A) ~ 153`, fiducial `169`). A
  second tracer does NOT break this; it sharpens the products and relaxes the
  prior requirement (Barreira & Krause).
- **TIED** (universality): `f_NL` is recovered; multi-tracer `sigma(f_NL)`
  `95 -> 36` (**2.7x**), the `|b1_A - b1_B|` differential-bias gain. The plot's
  panel 3 shows it grows with the bias separation and vanishes as `b1_A -> b1_B`.

## Honest caveats

- **The degeneracy needs `f_NL != 0`** (it is a product degeneracy; see fact 1).
- **The cancellation needs shot noise** (fact 2). `n_A, n_B` are a free knob; the
  demo picks them so `P*n` is order a few.
- The native tracer self-calibrates `b_phi` at high `k`; trust its degeneracy
  demo only at low `k`. The explicit-`b_phi` tracer is clean at all `k` but is a
  linear-bias model (no PM).
- The multi-tracer cancellation lives in the *differential* bias, so it sharpens
  the universality-tied `sigma(f_NL)` (and the relative-bias combination), not the
  individual product `sigma(f_NL*b_phi_A)`.

## Reproduce

```
pixi run probe-multitracer   # all validation numbers + both headlines (no files)
pixi run multitracer         # outputs/multitracer_fisher.png (the 3-panel figure)
pixi run python -m pytest tests/test_multitracer.py
```

## References (verified against the primary sources)

- Seljak 2009 (arXiv:0807.1770) -- multi-tracer cosmic-variance cancellation.
- McDonald & Seljak 2009 (arXiv:0810.0323) -- the two-tracer Fisher formalism.
- Barreira 2022 (arXiv:2205.05673) -- a single tracer constrains `f_NL*b_phi`.
- Barreira & Krause 2023 (arXiv:2302.09066) -- multi-tracer constrains the
  products with no `b_phi` prior; constraining power `~ |b1_B b_phi_A - b1_A
  b_phi_B|`; it relaxes (not eliminates) the `b_phi` prior requirement.
- Dalal et al. 2008 (arXiv:0710.4560) -- the `k^-2` scale-dependent bias.
