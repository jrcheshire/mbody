# M-body roadmap

Staged plan, smallest-runnable-thing first. Each step should leave a runnable
script in `scripts/` and (where it makes sense) a test in `tests/`. v0 leans on
an analytic Eisenstein & Hu (1998) transfer function so the toy stays
self-contained; CAMB is an optional upgrade later.

## Step 0 -- de-risk autodiff through the FFT  [in progress]

`scripts/probe_ad_fft.py` (`pixi run probe`). The PM force solve and the P(k)
estimator both route through `mx.fft`, so before building anything, confirm
`mx.grad` flows through the transform and matches an analytic gradient. If MLX
cannot differentiate the FFT cleanly, that dictates the whole architecture
(custom VJP, real-FFT workaround, or fall back to JAX / pmwd).

Also probe the other differentiability-critical primitive: **scatter-add**
(CIC painting is a scatter, CIC read is a gather). Confirm `mx.grad` flows
through `mx.scatter` / `array.at[idx].add(...)` before relying on it in step 3.

## Step 1 -- linear theory + Gaussian ICs

- EH98 transfer function -> linear `P(k)` at the target redshift.
- Generate a Gaussian random field `delta_lin` on an `N^3` mesh from `P(k)`
  with the correct Hermitian symmetry.
- Validate: the measured `P(k)` of the realization recovers the input to
  cosmic variance.

## Step 2 -- LPT displacement -> particles

- Zel'dovich (1LPT) displacement field from `delta_lin`; 2LPT later.
- Place `N_p` particles on a grid, displace, assign velocities.
- Validate: cross-correlation with the linear field; visual slice plots.

## Step 3 -- PM force solve + leapfrog

- CIC paint particles -> density mesh.
- FFT -> solve Poisson in Fourier space -> gradient -> force mesh.
- CIC read forces back to particles.
- Kick-drift-kick leapfrog over a few steps.
- Measure final `P(k)`; sanity-check growth against linear theory on large
  scales.

## Step 4 -- local f_NL initial conditions

- `phi = phi_G + f_NL * (phi_G^2 - <phi_G^2>)` on the potential, then scale to
  `delta_lin` via the transfer function.
- Validate the injected `f_NL` against the field bispectrum (squeezed limit)
  on large scales.

## Step 5 -- the headline: autodiff dlnP/df_NL

- Differentiate the measured `P(k)` (or a band power) with respect to `f_NL`
  through the entire pipeline with `mx.grad`.
- Overlay against the analytic Dalal et al. (2008) scale-dependent bias
  `Delta b(k) ~ f_NL / k^2` (here the matter-field analogue / the large-scale
  `dlnP/df_NL` shape).
- This closes the loop and is the project's "it works" figure.

## Stretch

- Gradients w.r.t. cosmological parameters (autodiff Fisher).
- A toy galaxy bias and exploration of `b_phi` -- the real SPHEREx lever; its
  degeneracy with `f_NL` is the dominant systematic (see CLAUDE.md).
- Gradient-checkpointing the leapfrog: reverse-mode memory grows with the
  number of steps because the graph unrolls; checkpointing trades compute for
  memory.
- 2LPT ICs; more steps; convergence study vs a reference (pmwd / analytic).

## Known risks / open questions

- **AD through FFT and scatter-add** (step 0) -- the gating unknowns.
- **Float32 default.** MLX defaults to float32; check whether PM accuracy and
  gradient stability need float64 (and whether MLX supports it well on Metal).
- **AD memory vs steps.** The unrolled graph can dominate memory; budget grids
  and step counts accordingly even with 128 GB unified memory.
- **Validation.** Cross-check field statistics and gradients against pmwd or
  analytic limits before trusting any `dlnP/df_NL` number.
