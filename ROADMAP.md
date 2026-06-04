# M-body roadmap

Staged plan, smallest-runnable-thing first. Each step should leave a runnable
script in `scripts/` and (where it makes sense) a test in `tests/`. Linear
theory uses the **CAMB Boltzmann solver as the default** transfer/P(k) backend,
with the analytic Eisenstein & Hu (1998) fitting formula retained behind the
same interface as a fast, dependency-light, differentiable-friendly
alternative (the two agree to ~4%).

**Cross-cutting: diagnostics & visualization.** Every stage should be
inspectable. Each evolving stage (LPT, the PM leapfrog) takes an optional
snapshot/callback hook so state can be captured per step without changing the
physics or autodiff path; a small optional `mbody` viz layer then turns a
sequence of snapshots into slice/scatter frames and stitches them into an
**animation of structure forming** from the initial conditions. Shared
diagnostics (P(k), cross-correlation, later the bispectrum) are first-class and
reused across steps. Keep all of this optional and off the hot/AD path.

## Step 0 -- de-risk autodiff through the FFT  [done]

Verified on Apple Silicon (MLX 0.31.2): `mx.grad` flows cleanly through
`rfftn`/`irfftn`, through scatter-add (`mesh.at[idx].add`, accumulates) and
gather, and `mx.checkpoint` / `@mx.custom_function` work. float64 raises on the
GPU (CPU-stream islands only). See docs/architecture-plan.md "Local
verification".

`scripts/probe_ad_fft.py` (`pixi run probe`). The PM force solve and the P(k)
estimator both route through `mx.fft`, so before building anything, confirm
`mx.grad` flows through the transform and matches an analytic gradient. If MLX
cannot differentiate the FFT cleanly, that dictates the whole architecture
(custom VJP, real-FFT workaround, or fall back to JAX / pmwd).

Also probe the other differentiability-critical primitive: **scatter-add**
(CIC painting is a scatter, CIC read is a gather). Confirm `mx.grad` flows
through `mx.scatter` / `array.at[idx].add(...)` before relying on it in step 3.

## Step 1 -- linear theory + Gaussian ICs  [done]

- `mbody/cosmology.py`: CAMB (default) + EH98 backends for the transfer
  function / linear `P(k)`, exact flat-LCDM growth, sigma8 normalization.
- `mbody/fields.py`: Gaussian random field on the `N^3` mesh (real white noise
  -> rfftn -> colour by sqrt(P(k)/V_cell) -> irfftn, automatically Hermitian),
  plus a `P(k)` estimator.
- Validated: EH98 vs CAMB agree to ~4%; the measured `P(k)` of a realization
  recovers the input to <0.2% (modes-weighted over 30 realizations). Figures
  under `outputs/`; scaling via `scripts/bench_fields.py`.

## Step 2 -- LPT displacement -> particles  [1LPT done]

- `mbody/lpt.py`: Zel'dovich (1LPT) displacement `Psi_1 = grad lap^-1 delta` via
  FFT; particles placed on the grid and displaced, growth-scaled by D(z).
- Validated: the identity `div Psi_1 = -delta` holds on all resolved Fourier
  modes (<1e-4; the only residual is the Nyquist-plane spectral-gradient
  ambiguity). Structure-formation figure + animation under `outputs/`.
- TODO: 2LPT; velocities (deferred to the integrator); and -- once CIC painting
  exists -- the displaced-density cross-correlation / P(k) recovery check.

## Step 3 -- PM force solve + leapfrog  [in progress]

- DONE `mbody/painting.py`: CIC paint particles -> density mesh, and CIC read
  (the trilinear adjoint). Differentiable -- gradients flow through the CIC
  weights to particle positions; the integer cell indices are detached
  (`mx.stop_gradient`) because MLX refuses a VJP w.r.t. scatter/gather indices.
  Validated: mass conservation, exact paint/read adjoint, mx.grad vs fp64 finite
  difference, and displaced-density recovers linear P(k) to ~1% at high z.
- TODO `mbody/forces.py`: FFT -> Poisson in Fourier (`Phi_k = -delta_k/k^2`) ->
  force `-grad(Phi)` -> `cic_read` forces back to particles. (Note: the force
  kernel `i k/k^2` is the same one `lpt._k_components` already builds.)
- TODO `mbody/integrate.py`: kick-drift-kick leapfrog (FastPM kernels) with an
  optional per-step snapshot hook for the animation framework; the velocity /
  time convention is fixed here.
- Validate: final `P(k)` growth vs linear on large scales; the cosmic web
  sharpens relative to the Zel'dovich washout.

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
