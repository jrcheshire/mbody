# M-body API reference

A tour of the public API, module by module. This is a curated reference, not an
exhaustive autodoc dump -- the docstrings in the source carry the full detail.
For runnable end-to-end usage, see [`examples/`](../examples/); for the topic
notes, see the rest of [`docs/`](.).

**Conventions.** Lengths are in Mpc/h, wavenumbers in h/Mpc; internal time uses
H0 = 1. Arrays are MLX (`mlx.core`) float32 on the GPU by default; float64 work
runs on a CPU stream (see `mbody.precision`). `backend=` selects the linear-theory
transfer function, `"camb"` (default) or `"eh98"`. Fields are real `(N, N, N)`
meshes (`N = box.n_mesh`); particle positions are `(N_p^3, 3)` in Mpc/h.

The public surface is also re-exported from the top level: `import mbody` gives
`mbody.run`, `mbody.RunResult`, the config dataclasses, and every submodule
(`mbody.cosmology`, `mbody.fisher`, ...). `import mbody` does not pull in
matplotlib.

---

## `mbody.config` -- run configuration

Frozen dataclasses (pure data, no MLX). A single `SimConfig` fully specifies a run.

- **`Cosmology(Omega_m=0.31, Omega_b=0.049, h=0.677, n_s=0.965, sigma8=0.81,
  T_cmb_K=2.7255)`** -- flat-LCDM parameters. Properties: `Omega_cdm`,
  `Omega_Lambda`, `H0`.
- **`BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)`** -- the periodic
  volume and meshes. Properties: `cell_size`, `n_mesh_total`, `n_particles_total`,
  `k_fundamental`, `k_nyquist`.
- **`TimeStepping(z_init=9.0, z_final=0.0, n_steps=10, integrator="fastpm",
  memory_mode="replay")`** -- the PM march. `integrator` is `"fastpm"` /
  `"exact"` / `"bullfrog"`.
- **`InitialConditions(f_NL=0.0, seed=0, lpt_order=2, kind="gaussian")`** -- how
  the IC field is generated (`kind` is `"gaussian"` / `"local_fnl"`).
- **`Tracer(b1=2.0, b2=1.0, A=1.0)`** -- the local quadratic-bias tracer and the
  linear amplitude `A`.
- **`RedshiftSpace(enabled=False, los_axis=0, f_growth=1.0)`** -- opt-in RSD
  settings for the measured field.
- **`SimConfig(cosmology, box, time, ic, tracer, rsd)`** -- the aggregate; all
  fields default. `.summary()` returns a one-screen text summary.

## `mbody.cosmology` -- background, growth, linear P(k)

- **`transfer(k_hmpc, cosmo, backend="camb")`** -- linear transfer function
  T(k); `transfer_eh98` / `_nowiggle` / `_zerobaryon` / `transfer_camb` are the
  backends.
- **`linear_power(k_hmpc, cosmo, z=0.0, backend="camb")`** -- linear P(k),
  sigma8-normalized. `dimensionless_power(...)` returns Delta^2(k).
- **`sigma_R(R, cosmo, z=0.0, backend="camb")`** -- rms fluctuation in spheres of
  radius R (Mpc/h).
- **`E(z, cosmo)`** -- H(z)/H0. `mean_matter_density(cosmo)` -- rho_m in
  (Msun/h)/(Mpc/h)^3.
- **`growth_factor(z, cosmo)`** / **`growth_rate(z, cosmo)`** -- D(z) (normalized
  D(0)=1) and f = dlnD/dlna. `growth_factor_md` is the matter-dominated
  normalization used by the Dalal M(k). `growth_factor_2` / `growth_rate_2` are the
  second-order (2LPT) growth.

## `mbody.ic` -- initial conditions

- **`linear_density(box, cosmo, seed=0, z=0.0, f_NL=0.0, backend="camb")`** -- the
  linear density contrast delta on the mesh; `f_NL` (float or mx scalar) injects
  local non-Gaussianity and keeps delta differentiable in `f_NL`.
- **`primordial_potential(box, cosmo, seed=0, ...)`** -- the Gaussian potential
  phi_G. **`poisson_M(k, cosmo, ...)`** / **`poisson_factor(box, cosmo, ...)`** --
  the M(k) with delta = M(k) phi.
- **`local_bispectrum_template(triangles, cosmo, f_NL, ...)`** /
  **`local_bispectrum_binned(...)`** -- analytic tree-level squeezed bispectrum
  templates (the latter bin-averaged to match the estimator).

## `mbody.lpt` -- Lagrangian perturbation theory

- **`displacement(box, cosmo, order=1, seed=0, f_NL=0.0, backend="camb",
  amplitude=1.0)`** -- the LPT displacement field Psi (order 1 or 2).
- **`zeldovich_displacement(...)`** / **`second_order_displacement(...)`** --
  the 1LPT and 2LPT pieces; **`lpt2_source(delta, box)`** the 2LPT source.
- **`lpt_positions(box, cosmo, seed=0, z=0.0, f_NL=0.0, backend="camb")`** --
  displaced particle positions at redshift z. `displace`, `lagrangian_grid`,
  `divergence` are the building blocks.

## `mbody.painting` -- mass assignment (CIC)

- **`cic_paint(positions, box, weights=None)`** -- scatter particles to a density
  mesh (differentiable through the trilinear weights).
- **`density_contrast(positions, box)`** -- paint and convert to delta.
- **`cic_read(field, positions, box)`** / **`cic_read_vector(fx, fy, fz,
  positions, box)`** -- gather a mesh (or three) back to particle locations.

## `mbody.forces` -- the PM force solve

- **`forces_on_particles(positions, box, delta=None)`** -- paint -> FFT Poisson ->
  read: the geometric acceleration on each particle.
- **`acceleration_field(delta, box)`** -- g = i k delta / k^2 on the mesh;
  **`potential(delta, box)`** -- the (diagnostic) potential phi.
- **`make_force_fn(box, compiled=False, checkpoint=False, recompute_cic=False)`**
  -- build the force closure the integrator calls.

## `mbody.integrate` -- time stepping and the adjoint

- **`initial_state(box, cosmo, time, seed=0, f_NL=0.0, backend="camb",
  lpt_order=2, amplitude=1.0)`** -- LPT positions and momenta at `z_init`.
- **`leapfrog(box, cosmo, time, seed=0, f_NL=0.0, backend="camb",
  spacing="linear", integrator=None, lpt_order=2, amplitude=1.0, ...)`** -- the
  full forward evolution (IC -> evolve), returning final `(x, p)`. Differentiable
  with `mx.grad`.
- **`evolve_state(x, p, box, cosmo, a_steps, integrator="fastpm", snapshot=None)`**
  -- evolve a given state across the scale-factor grid (`a_grid(time, spacing)`).
- **`adjoint_grad_ic(loss_field, box, cosmo, time, seed=0, f_NL=0.0,
  amplitude=1.0, ..., loss_uses_momentum=False, recompute_cic=True)`** -- the
  reversible-adjoint gradient of `loss_field(x_final[, p_final])` w.r.t. the IC
  parameters `(f_NL, A)`; O(1) memory in step count. **`adjoint_grad_fnl(...)`**
  is the f_NL-only wrapper.
- Kernels: `kick_factor` / `drift_factor` (exact), `fastpm_kick_factor` /
  `fastpm_drift_factor`, `bullfrog_coeffs`, `a_grid`.

## `mbody.rsd` -- redshift-space distortions

- **`redshift_space_positions(x, p, box, cosmo, z, los_axis=0, f_growth=1.0)`** --
  shift the line-of-sight coordinate by the peculiar velocity (the RSD map).
- **`apply_linear_kaiser(delta, box, b1, f, los_axis=0)`** -- the linear Kaiser
  factor applied in Fourier space (a quick analytic check).

## `mbody.fields` -- statistics and estimators

- **`gaussian_random_field(box, cosmo, seed=0, z=0.0, backend="camb")`** -- a GRF
  with the linear P(k). `k_grid(box)`, `cic_window(box)` are helpers.
- **`power_spectrum(delta, box, dk=None, kmin=None, kmax=None,
  deconvolve_cic=False)`** -- binned P(k) (returns k, P, n_modes).
- **`band_power(delta, box, k_bins, dk=None)`** / **`cross_power(a, b, box,
  k_bins, dk=None)`** -- differentiable fixed-shell band powers (the Fisher data).
- **`band_power_multipole(delta, box, k_bins, ell, los_axis=0, dk=None)`** /
  **`cross_power_multipole(a, b, ...)`** -- raw redshift-space multipoles;
  **`power_multipoles(...)`** the decoupled, CIC-deconvolved diagnostic;
  **`multipole_decoupling(...)`** the discrete-shell correction matrix.
- **`interlaced_density_contrast(positions, box)`** -- the interlaced CIC painter
  (the standard measurement painter).
- **`bispectrum(delta, box, triangles, dk=None)`** /
  **`bispectrum_single(delta, box, triangle, dk=None)`** -- the Scoccimarro FFT
  bispectrum estimator.

## `mbody.bias` -- biased tracers and the Dalal reference

- **`local_bias_tracer(delta, b1, b2)`** -- `b1 delta + (b2/2)(delta^2 -
  <delta^2>)`, the minimal differentiable tracer.
- **`scale_dependent_bias_tracer(delta, box, cosmo, b1, b_phi, f_NL, ...)`** -- a
  tracer with an EXPLICIT k^-2 scale-dependent bias (b_phi a free parameter).
- **`scale_dependent_shape(k, cosmo, ...)`** -- the 1/M(k) Dalal shape;
  **`scale_dependent_bias_response[_binned](box, cosmo, k_bins, b1, b2, ...)`** --
  the absolute amplitude 4 b2 sigma^2/(b1 M(k)). `mesh_variance(box, cosmo, ...)`
  is the sigma^2.

## `mbody.diagnostics` -- off-AD-path measurement

- **`particle_power(positions, box, interlace=True, **kwargs)`** -- P(k) of a
  particle set. **`cross_correlation(a, b, box, ...)`** -- the propagator r(k).
- **`growth_amplitude(modes_seq)`** / **`linear_growth_reference(a_arr, cosmo)`**
  -- measured vs linear growth. **`skewness(field)`**, **`one_point_pdf(field,
  ...)`** -- the one-point statistics.
- **`SnapshotRecorder(box, ...)`** -- the memory-gated `snapshot(step, a, x, p)`
  seam for trajectories (stores only bounded reductions); `.growth_history()`.

## `mbody.driver` -- the run driver

- **`run(cfg, backend="camb", record=False, recorder=None, spacing="linear")`** --
  the single entry point: threads a `SimConfig` through IC -> LPT -> leapfrog and
  measures it. Returns a `RunResult`.
- **`RunResult`** -- holds the final state and fields; methods: `.power(**kw)`
  (returns k, P, n_modes), `.power_multipoles(ells=(0, 2), **kw)` (requires
  `cfg.rsd.enabled`), `.cross_with_ic(**kw)`, `.growth_history()` (needs
  `record=True`), `.dashboard(out=...)`, `.save(path)`.

## `mbody.fisher` -- autodiff Fisher forecasts

- **`FisherForecast(jacobian, variances=None, fiducial_params=..., param_names=...,
  priors=None, covariance=None)`** -- a forecast from a Jacobian and a data
  covariance (diagonal `variances` or full `covariance`). Methods: `.sigma(name)`
  (marginalized) and `.sigma_conditional(name)`, `.marginalized_2d(i, j)`,
  `.conditional_2d(i, j)`, `.ellipse_xy(i, j, n_sigma=1.0)`,
  `.condition_number()`, `.summary(name)`.
- **`band_power_log_variance(box, k_bins, dk=None)`** -- the Gaussian Var[ln P_b]
  (with the rfft-plane double-count correction).
- Jacobians: **`linear_logP_jacobian`** / **`pm_logP_jacobian`** over
  `PARAM_NAMES = {f_NL, b1, b2, A}`; the multipole and multi-tracer variants
  (`linear_multipole_jacobian`, `pm_multipole_jacobian`,
  `linear_multitracer_jacobian`, `pm_multitracer_jacobian`, and the RSD multi-tracer
  multipole versions) over the corresponding `PARAM_NAMES_*`.
- Covariances: `multipole_gaussian_covariance`, `multitracer_analytic_covariance`,
  `multitracer_gaussian_covariance`, and the multipole/bphi variants (mock block
  covariances with per-tracer shot noise).
- Multi-tracer + universality: **`multitracer_forecast(J, cov, fiducial,
  param_names, priors=None, tie=None)`**; `universality_fiducial` /
  `universality_tie_matrix` (native), `universality_bphi_*` (explicit b_phi), and
  `universality_rsd_*` (redshift-space) build the chain-rule tie.

## `mbody.viz` -- optional matplotlib layer

Not imported by `mbody`; import `mbody.viz` explicitly.

- **`density_slice(field, box, ..., smooth=None)`** -- a projected density slice.
- **`dashboard(...)`** -- the six-panel run dashboard (used by
  `RunResult.dashboard`).
- **`animate_slab(slabs, a_arr, box, out, ...)`** -- a structure-formation
  animation from a snapshot sequence.

## `mbody.precision` -- dtype policy

- **`fp64_cpu()`** -- context manager for a float64 CPU-stream island.
- **`as_real(x)`** / **`as_complex(x)`** / **`ensure_gpu_safe(x, name=...)`** --
  dtype guards. **`accurate_sum`** / **`accurate_mean`** / **`subtract_mean`** --
  stable reductions for global float32 sums.
