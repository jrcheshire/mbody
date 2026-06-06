# External convergence cross-check

Validation of the M-body forward model against references **outside** mbody. Until
now everything was checked internally (autodiff-vs-finite-difference,
adjoint-vs-replay, integrator self-consistency) and the f_NL scale-dependent-bias
overlay was *shape-only* (the 1/M(k) curve normalized to the data at the largest
scale). This closes the standing ROADMAP "known risk": no absolute-normalization
check, and no comparison to an external code.

Three layers were built. A fourth -- a **pmwd** differentiable-PM peer comparison --
is deliberately deferred (the heavy convention-matching one); the scripts here are
kept pmwd-ready.

All ratio checks use **matched phases** (the same RNG seed for the PM run and the
linear reference), so cosmic variance cancels and the comparison is deterministic
rather than ensemble-noisy.

## Reproduce

```bash
pixi run python scripts/probe_pk_convergence.py   # Layer A numbers (stdout)
pixi run convergence                              # -> outputs/convergence.png
pixi run python scripts/probe_fnl_amplitude.py    # absolute f_NL amplitude
pixi run external-ccl                             # -> Layer B (CCL) numbers
pixi run test                                     # tests/test_convergence.py + test_external_ccl.py
```

Figures land in `outputs/` (gitignored): `convergence.png` (transfer, propagator,
growth) and `dlnp_dfnl_linear.png` (the now-absolute f_NL overlay).

---

## Layer A -- analytic linear theory

`scripts/probe_pk_convergence.py`, `scripts/plot_convergence.py`,
`tests/test_convergence.py`. Numbers below are L=512 Mpc/h, N=64, fastpm+2LPT,
z=9->0 in 10 steps, 8 seeds.

### A1. Matched transfer T(k) = P_PM / P_lin (z=0)

A particle-painted P(k) carries the **CIC mass-assignment window**
`W(k) = prod_i sinc^2(k_i / 2 k_nyq)` (new `fields.cic_window`); a grid field
(`ic.linear_density`) does not. So the raw transfer droops at high k for a
non-physics reason -- deconvolve P_PM with `power_spectrum(deconvolve_cic=True)`.

| k / k_nyq | T_raw | T_dec (CIC-deconvolved) |
|----------:|------:|------------------------:|
| 0.031     | 0.988 | **0.991 +/- 0.006**     |
| 0.062     | 0.972 | **0.979 +/- 0.007**     |
| 0.094     | 0.947 | 0.962                   |
| 0.125     | 0.916 | 0.940                   |
| 0.250     | 0.781 | 0.866                   |
| 0.500     | 0.475 | 0.721                   |

At the largest scales `T -> 1` (absolute growth + normalization, validated for the
first time). The high-k droop is the coarse PM **under-resolving small scales**
(force softening at the cell scale), the expected limitation -- characterized, not
hidden. The IC-stage transfer `P(x0)/P_lin(z_init)` is likewise ~1 (+/-5%) out to
~0.4 k_nyq.

### A2. Propagator r(k) = <delta_PM delta_lin> / sqrt(P_PM P_lin)

Window-independent (the CIC window cancels in the ratio).

| k / k_nyq | r(k)   |
|----------:|-------:|
| 0.031     | 0.9998 |
| 0.094     | 0.9885 |
| 0.250     | 0.843  |
| 0.500     | 0.429  |

`r -> 1` at the fundamental (phases track linear theory) and decoheres toward small
scales -- the standard PM propagator (first drops below 0.99 at ~0.09 k_nyq).

### A3. Growth convergence vs step count

Large-scale growth `R(a_final) / [D(0)/D(z_init)]`; linear target D-ratio = 7.85.

| n_steps | exact  | fastpm |
|--------:|-------:|-------:|
| 2       | 0.903  | 0.975  |
| 5       | 0.948  | 0.974  |
| 10      | 0.966  | 0.974  |
| 20      | 0.972  | 0.974  |

FastPM gets linear growth right at **any** step count (flat); the exact-background
leapfrog has a low-step deficit that converges upward to it. (Both sit ~2.6% below
1 -- the large-scale field-growth measurement from the painted low-k modes, not an
integrator error, since the two integrators agree in the limit.)

---

## Absolute f_NL bias amplitude

`scripts/probe_fnl_amplitude.py`, `bias.scale_dependent_bias_response[_binned]`,
`bias.mesh_variance`. Upgrades the shape-only overlay to an **absolute** prediction
with no free normalization.

For the local quadratic-bias tracer `delta_h = b1 delta + (b2/2)(delta^2 -
<delta^2>)`, a long-wavelength potential modulates the small-scale variance by
`(1 + 4 f_NL phi_L)`, giving

```
dlnP_h/df_NL(k) = 4 b2 sigma^2 / (b1 M(k)),   sigma^2 = mesh variance
```

(the squeezed limit of the integrated tree-level bispectrum, `I(k)/P(k) ->
4 sigma^2/M(k)`). The mesh variance matches the realized linear field's `<delta^2>`
to **0.15%**, and `mx.grad` matches matched-phase finite difference to ~1e-4.

| k [h/Mpc] | autodiff dlnP/df | pred (bin centre) | pred (shell avg) | meas/centre | meas/binned |
|----------:|-----------------:|------------------:|-----------------:|------------:|------------:|
| 0.0123    | 1.403e-3         | 1.925e-3          | 1.419e-3         | 0.729       | **0.989**   |
| 0.0245    | 6.21e-4          | 6.90e-4           | 6.25e-4          | 0.900       | **0.994**   |
| 0.0368    | 4.29e-4          | 4.36e-4           | 4.22e-4          | 0.984       | 1.017       |
| 0.0982    | 2.03e-4          | 1.68e-4           | 1.68e-4          | 1.207       | 1.208       |

The bin-**centre** 1/M form is 27% low at the fundamental: the band power is a
P(k)-weighted shell average of `1/M(q)` and 1/M is steep, so the centre value is
biased -- the same binning systematic `ic.local_bispectrum_binned` removes for the
bispectrum. The **bin-averaged** predictor matches autodiff to ~1-2% across the
squeezed bins; high-k bins drift up (the leading term omits the non-squeezed and
b2^2 contributions). The absolute amplitude is asserted only on the *linear* field
(tree-level exact); through the PM pipeline it is a measured deviation, not a
pass/fail.

---

## Layer B -- CCL (pyccl) community-code anchor

`scripts/probe_external_ccl.py`, `tests/test_external_ccl.py` (gated: skipped if
`pyccl` is absent, so the core suite never depends on it). mbody's CAMB backend is
the same CAMB it would self-check against, and the internal EH98-vs-CAMB agreement
could hide a shared convention bug; CCL reimplements the growth factor, transfer
function, and sigma8 normalization independently.

**Units are the whole game** -- CCL works in physical Mpc, mbody in Mpc/h:

```
k_ccl [1/Mpc]       = k_mbody [h/Mpc] * h
P_mbody [(Mpc/h)^3] = P_ccl [Mpc^3] * h^3
R_ccl [Mpc]         = R_mbody [Mpc/h] / h
```

| quantity | comparison | agreement |
|----------|------------|----------:|
| growth D(z), z=0..9 | mbody vs `ccl.growth_factor` | < 0.3% (1.6e-3 at z=9) |
| linear P(k), EH98 | mbody `eh98` vs CCL `eisenstein_hu` | ~1e-6 |
| linear P(k), CAMB | mbody `camb` vs CCL `boltzmann_camb` | < 0.5% (1.6e-3) |
| sigma_R(R), sigma8 | mbody `sigma_R` vs `ccl.sigmaR`/`sigma8` | ~1e-4 |

The growth residual at high z is mbody **neglecting radiation** in
`E(z) = sqrt(Omega_m a^-3 + Omega_Lambda)` -- an intentional toy simplification, not
a bug. Everything else agrees to the level of the underlying fitting formula.

---

## Status and follow-on

- 138 tests pass (10 new); the CCL test skips cleanly without `pyccl`.
- **Deferred:** the pmwd differentiable-PM peer comparison -- run pmwd (JAX
  CPU/fp64 on Apple Silicon) from matched ICs and compare nonlinear P(k)/r(k) and,
  as a stretch, the gradient dP/df_NL. The hard part is matching pmwd's
  transfer/growth/LPT/stepping conventions so a discrepancy means "bug", not
  "different setup". The Layer A scripts are structured to drop a pmwd run in.
