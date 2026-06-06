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
reused across steps. Keep all of this optional and off the hot/AD path. This is
now built -- see Step 6 (`mbody.diagnostics` + `mbody.viz` + the run driver).

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
- DONE: 2LPT (`lpt.second_order_displacement`, sourced by `lpt2_source`;
  `div Psi_2 = -delta_2` on resolved modes); velocities (in the integrator);
  displaced-density P(k) recovery (Step 3). The 2LPT IC carries the correct
  positive (gravitational) skewness the Zel'dovich field lacks.

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
- DONE (stage 2): FastPM growth-corrected kick/drift kernels (Feng et al. 2016,
  `integrate.fastpm_kick_factor` / `fastpm_drift_factor`, selected by
  `integrator="fastpm"`) -- a single linear mode grows as `D(a)` exactly at any
  step count (validated to <1e-4 at 2-8 steps), where the exact-background
  leapfrog is off by 2-8%. The kernels are constants of the step, so `mx.grad`
  and the reversible adjoint extend unchanged.

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
- DONE (Stage 5b, PM): `f_NL` is threaded through `lpt.zeldovich_displacement`
  and `integrate.{initial_state,leapfrog}` (via `ic.linear_density`), so the
  whole forward model `f_NL -> LPT -> PM leapfrog -> CIC -> tracer -> P(k)` is
  differentiable. `mx.grad` of dlnP/df_NL through the unrolled leapfrog matches
  a matched-phase FD to ~5e-3 (the CIC scatter-add floor). The f_NL signal
  survives evolution (large-scale dlnP/df_NL ~13x the matter null) but its shape
  flattens vs the linear 1/M(k) -- nonlinear evolution + CIC + the Eulerian bias
  mix scales (the real-world complication behind the b_phi systematic), so the
  clean Dalal 1/k^2 is a linear-theory result. `scripts/plot_dlnp_dfnl_pm.py`
  -> `outputs/dlnp_dfnl_pm.png`. The end-to-end differentiability is the
  headline; the clean 1/M(k) lives in Stage 5a.
- Amplitude: the 1/M(k) overlay is shape-only (normalized at the largest scale);
  a first-principles `b_phi` and the universality relation
  `b_phi = 2 delta_c (b1-1)` are Stretch (the variance-modulation tracer).

## Step 6 -- diagnostics layer, viz, and the run driver  [done]

The cross-cutting diagnostics/visualization plan above is built, and a single run
driver ties the config to it.

- `mbody/diagnostics.py`: read-only, off-AD-path measurement reused across the
  scripts and the dashboard -- `cross_correlation` (the k-resolved coefficient
  r(k)), `particle_power`, `growth_amplitude` / `linear_growth_reference`,
  `skewness` / `one_point_pdf`, and the memory-gated `SnapshotRecorder` (the
  `snapshot(step, a, x, p)` seam; stores only bounded reductions -- a slab
  projection + low-k modes -- never the full particle state).
- `mbody/viz.py`: the optional matplotlib layer, deliberately NOT imported by
  `mbody.__init__` so `import mbody` stays plotting-free and off the hot path --
  `density_slice`, `animate_slab`, and the six-panel `dashboard` (density slice;
  P(k) vs linear at z_init/z_final; one-point PDF + skewness; growth vs D(a);
  r(k) vs the IC; a config-summary text panel).
- `mbody/driver.py`: `run(SimConfig) -> RunResult`, the single entry point that
  threads a whole config through ic -> LPT -> leapfrog and measures it. RunResult
  measures (`power`, `cross_with_ic`, `growth_history`) and renders / serializes
  itself (`dashboard`, `save`). `scripts/run_demo.py` is the headline; the inline
  growth / animation in `plot_growth.py` and `animate_pm.py` now comes from this
  shared layer.
- Honest config: SimConfig defaults name only built physics. With FastPM + 2LPT
  now implemented (Steps 2/3), the defaults are `integrator="fastpm"` and
  `lpt_order=2` (the better physics, validated); `"exact"` and `lpt_order=1`
  remain available, and `"bullfrog"` is the one reserved name `run()` still
  rejects with `NotImplementedError`. `scripts/compare_integrators.py` shows the
  payoff (exact-vs-fastpm growth accuracy, ZA-vs-2LPT skewness).

## Stretch

- Gradients w.r.t. cosmological parameters (autodiff Fisher).
- A toy galaxy bias and exploration of `b_phi` -- the real SPHEREx lever; its
  degeneracy with `f_NL` is the dominant systematic (see CLAUDE.md).
- Leapfrog AD memory (reverse-mode grows with step count as the graph unrolls).
  RESOLVED via a reversible adjoint, NOT checkpointing: `mx.checkpoint` was
  measured to give no memory reduction here (the PM solve is near-linear, so
  reverse-mode retains almost nothing to recompute-away). The reversible-leapfrog
  adjoint (`integrate.adjoint_grad_fnl`) gives an O(1)-in-steps gradient instead
  -- flat memory vs step count (23x less than replay at 32 steps, N=64) at ~2x
  compute, grad matching replay to ~1e-6. `mx.compile` of the force solve is
  ~1x (FFT-bound). See `scripts/bench_pm.py` and the module docstrings. Next
  here: generalize the adjoint IC step to cosmology params (autodiff Fisher).
- More steps / higher resolution. Convergence cross-check vs analytic linear
  theory + CCL is now DONE (see "External convergence cross-check" below); the
  pmwd differentiable-PM peer comparison remains the open follow-on. (2LPT ICs and
  FastPM kernels are also DONE -- Steps 2/3.)

## Known risks / open questions

- **AD through FFT and scatter-add** (step 0) -- the gating unknowns.
- **Float32 default.** MLX defaults to float32; check whether PM accuracy and
  gradient stability need float64 (and whether MLX supports it well on Metal).
- **AD memory vs steps.** The unrolled graph can dominate memory; budget grids
  and step counts accordingly even with 128 GB unified memory.
- **Validation.** Cross-check field statistics and gradients against pmwd or
  analytic limits before trusting any `dlnP/df_NL` number. DONE for analytic
  linear theory + CCL (see below); pmwd is the remaining peer cross-check.

## External convergence cross-check  [analytic + CCL done]

Before trusting any number, validate the forward model against references outside
mbody. Two layers built (a pmwd differentiable-PM peer comparison is deferred).
Full writeup with the measured numbers: `docs/convergence_crosscheck.md`.

- **Analytic linear theory** (`scripts/probe_pk_convergence.py`,
  `scripts/plot_convergence.py` -> `outputs/convergence.png`,
  `tests/test_convergence.py`). On matched phases (cosmic variance cancels):
  the matched transfer `T(k) = P_PM/P_lin -> 1` at the largest scales (absolute
  growth + normalization; high-k droop is the coarse PM under-resolving small
  scales, characterized not hidden); the propagator `r(k) -> 1` at low k and
  decoheres as a standard PM; and large-scale growth converges to linear `D(a)`
  (fastpm step-independent, exact converges up to it). Adds the CIC mass-assignment
  window `fields.cic_window` + a `power_spectrum(deconvolve_cic=)` flag.
- **Absolute f_NL bias amplitude** (`scripts/probe_fnl_amplitude.py`,
  `bias.scale_dependent_bias_response[_binned]`). The Step-5 1/M(k) overlay was
  shape-only; the first-principles amplitude is `dlnP_h/df_NL = 4 b2 sigma^2/(b1
  M(k))` (a long mode modulating the small-scale variance), with `sigma^2 =
  bias.mesh_variance` the mesh variance. The bin-AVERAGED predictor (matching the
  band-power shell sum, like `ic.local_bispectrum_binned`) matches the autodiff
  derivative to ~1-2% across the squeezed bins -- an absolute check with no free
  normalization.
- **CCL community-code anchor** (`scripts/probe_external_ccl.py`,
  `tests/test_external_ccl.py`, gated on `pyccl`). Independent cross-check of the
  cosmology layer: growth `D(z)` (<0.3% to z=9; the residual is mbody neglecting
  radiation in `E(z)`), linear `P(k)` (EH98 vs CCL EH98 to ~1e-6; CAMB vs CCL CAMB
  to <0.5%), and `sigma_R` / sigma8 (to ~1e-4). Catches convention bugs the
  internal EH98-vs-CAMB check could share.
