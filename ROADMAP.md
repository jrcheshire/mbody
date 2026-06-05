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

## Step 3 -- PM force solve + leapfrog  [done]

- DONE `mbody/painting.py`: CIC paint particles -> density mesh, and CIC read
  (the trilinear adjoint). Differentiable -- gradients flow through the CIC
  weights to particle positions; the integer cell indices are detached
  (`mx.stop_gradient`) because MLX refuses a VJP w.r.t. scatter/gather indices.
  Validated: mass conservation, exact paint/read adjoint, mx.grad vs fp64 finite
  difference, and displaced-density recovers linear P(k) to ~1% at high z.
- DONE `mbody/forces.py`: FFT Poisson solve. `potential` (`phi_k =
  -delta_k/k^2`), `acceleration_field` (`g_j = i k_j delta_k/k^2`, the same
  kernel `lpt._k_components` builds), and `forces_on_particles` (paint -> solve
  -> `cic_read`), all differentiable. Validated: the potential inverts the
  Laplacian and `div g = -delta` on every resolved mode (<1e-4); the
  acceleration of a realization equals its Zel'dovich displacement to fp32 (the
  headline identity); a single-mode analytic case; zero net force by parity
  (~1e-8); and `mx.grad` vs an fp64 directional finite difference (~1%).
  `scripts/plot_forces.py` renders the potential + acceleration quiver. The
  cosmological prefactor `(3/2) Omega_m H0^2/a` is deferred to the integrator.
- DONE `mbody/integrate.py`: kick-drift-kick leapfrog in scale-factor time.
  EOM (H0 = 1 units) `dx/da = p/(a^3 E)`, `dp/da = (3/2) Omega_m g/(a^2 E)`, so
  the whole cosmological prefactor lives in the kick/drift factors (exact
  background integrals, fp64 CPU) and forces.py stays geometric. Zel'dovich
  growing-mode velocity IC `p = a^2 E D f Psi`. Optional off-AD-path `snapshot`
  callback for diagnostics / animation. `evolve_state` is a pure differentiable
  function of (x, p): `mx.grad` flows through the whole unrolled trajectory.
  Validated: large-scale growth tracks linear `D(a)` and converges as steps
  rise (this is the exact-background leapfrog; ~2% deficit at low step count is
  the expected discretization error); kick/drift additivity; growing-mode IC;
  snapshot bookkeeping; AD-through-leapfrog vs fp64 FD. Determinism is to fp32
  round-off only, not bit-exact (GPU scatter-add is nondeterministic).
  `scripts/animate_pm.py` (structure-formation gif via snapshot) and
  `scripts/plot_growth.py` (P(k) + D(a) validation figure).
- NEXT (stage 2, parked in Stretch): FastPM growth-corrected kick/drift kernels
  (Feng et al. 2016) to reproduce linear growth at very low step count, checked
  against this exact-background baseline.

## Step 4 -- local f_NL initial conditions  [done]

- DONE `mbody/ic.py`: `phi = phi_G + f_NL * (phi_G^2 - <phi_G^2>)` on the
  potential, then scale to `delta_lin` via the Poisson/transfer factor
  `M(k, z)`. Differentiable, kink-free, bit-reproducible in f_NL.
- DONE `mbody/fields.py`: Scoccimarro FFT bispectrum estimator
  (`bispectrum`, and a differentiable single-triangle `bispectrum_single`).
  Shell-filter `delta_k` onto `|k|` bins, `B = (V^2/N^9) sum_x I1 I2 I3 /
  sum_x J1 J2 J3` (the `V^2/N^9` prefactor is exact in the same DFT convention
  as `power_spectrum`'s `V/N^6`). Reductions are fp64 on the CPU stream.
- DONE `mbody/ic.py`: the analytic local templates -- `local_bispectrum_template`
  (continuum tree `B = 2 f_NL [M3/(M1 M2) P1 P2 + perms]`) and
  `local_bispectrum_binned` (the exact bin-averaged prediction, via the same
  shell-product identity, which removes the binning systematic so the estimator
  matches with unit calibration).
- Validated (`scripts/probe_bispectrum.py`, `tests/test_bispectrum.py`):
  normalization is exact two ways -- a deterministic closed-triangle plane-wave
  field gives `B = L^6 a^3/(4 n_tri)` to ~5e-8, and `n_tri` matches a
  brute-force triangle count; in a real measurement the matched-phase
  cosmic-variance-cancelled squeezed signal recovers the binned template with
  `c_cal ~ 1.0` (1.01 over 40 seeds at 256/128; 1.006 over 16 seeds at the test
  box). The squeezed `1/k_long^2` divergence (the scale-dependent-bias
  signature) is reproduced, `B` is odd/linear in f_NL, `B(f_NL=0) ~ 0`, and
  `mx.grad` dB/df_NL matches an FD to ~5e-5. `scripts/plot_fnl.py` ->
  `outputs/fnl_bispectrum.png` (density slice, skewed PDF, measured-vs-template
  squeezed B).

## Step 5 -- the headline: autodiff dlnP/df_NL

The matter power spectrum has no O(f_NL) term (`dlnP_matter/df_NL = 0`; the
correction is a Gaussian 3-point that vanishes), so the Dalal `1/k^2` is shown
on a **biased tracer**, with the matter field as the null contrast. Tracer =
Eulerian local quadratic bias `delta_h = b1 delta + (b2/2)(delta^2 - <delta^2>)`,
whose f_NL response enters via the squeezed bispectrum (Step 4 physics).

- DONE (Stage 5a, linear field): `mbody/fields.py` differentiable `band_power` /
  `cross_power` (fixed |k| shells, not a histogram); `mbody/bias.py`
  `local_bias_tracer` + `scale_dependent_shape` (1/M(k), the Dalal reference).
  `mx.grad` of the tracer's dlnP/df_NL matches a matched-phase finite difference
  to ~1e-4, tracks 1/M(k) at large scales, and the matter field is a ~1% null.
  `tests/test_bias.py`, `scripts/probe_fnl_bias.py`,
  `scripts/plot_dlnp_dfnl.py` -> `outputs/dlnp_dfnl_linear.png`.
  NOTE: forward-mode `mx.jvp` is wrong through the FFT (returns ~half); use
  reverse-mode `mx.grad`.
- NEXT (Stage 5b, PM): thread `f_NL` through `lpt.zeldovich_displacement` and
  `integrate.{initial_state,leapfrog}` (via `ic.linear_density`), apply the
  tracer to the CIC-painted evolved field, and differentiate dlnP/df_NL through
  the whole unrolled leapfrog -- showing the large-scale 1/k^2 survives.
- Amplitude: the 1/M(k) overlay is shape-only (normalized at the largest scale);
  a first-principles `b_phi` and the universality relation
  `b_phi = 2 delta_c (b1-1)` are Stretch (the variance-modulation tracer).

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
