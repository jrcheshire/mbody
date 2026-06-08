# Redshift-space distortions (RSD)

M-body can map the final particles into **redshift space** -- shifting each along
a line of sight by its peculiar velocity -- and measure the anisotropic power
spectrum multipoles `P_0` (monopole) and `P_2` (quadrupole). This is the regime
real galaxy `f_NL` constraints are actually measured in (Cartesian `P_l(k)`), and
the large-scale Kaiser signal is squarely in mbody's validated regime.

Opt-in via `RedshiftSpace(enabled=True)` on a `SimConfig` (off by default -- the
honest-config invariant: a plain run measures the real-space field).

## The velocity -> shift conversion

In the plane-parallel approximation a particle at comoving `x` with physical
peculiar velocity `v_pec` is observed at `s = x + (v_pec . n)/(a H) n` (Kaiser
1987). mbody integrates in scale factor with the momentum rescaled so `H0 = 1`
(`dx/da = p/(a^3 E)`); pushing the Kaiser shift through that drift gives, with the
box already in Mpc/h,

```
Delta_s_los = f_growth * p_los / (a^2 E(a))      (Mpc/h, no stray h factor)
```

The `h` factor only appears if one detours through physical km/s, which mbody
never does. `f_growth` (default 1 = the simulation's own velocities) is a
multiplier that doubles as a differentiable growth-rate amplitude in the Fisher.
**Validated** (`mbody.rsd.redshift_space_positions`): on a Zeldovich field the RSD
shift equals `f x (real-space displacement)`, recovering the linear growth rate to
`rel err ~2e-7`.

## The multipole estimator

For a periodic box with a fixed Cartesian line of sight the plane-parallel
estimator is `P_ell(k) = (2 ell + 1) * mean_shell[ |delta_k|^2 L_ell(mu) ]`,
`mu = k_los/|k|` (`mbody.fields.band_power_multipole`, `power_multipoles`). Two
finite-mesh subtleties, both **measured not assumed** (`scripts/probe_rsd.py`):

- **Line of sight must be a full fft axis** (`los_axis` 0 or 1), not the rfft
  (last) axis 2: along the rfft axis the `kz = 0` plane forces `mu = 0` for a whole
  plane of modes, skewing the discrete Legendre average.
- **Discrete-shell mode coupling.** A thin `|k|` shell samples solid angle
  non-uniformly, so the raw estimator leaks power between multipoles
  (`<L_ell>_shell != 0`). `multipole_decoupling` inverts the per-bin coupling
  matrix `M[ell,L] = (2 ell+1)<L_ell L_L>_shell` to recover the continuum
  multipoles. It is a fixed geometric (field-independent) linear map, so it keeps
  the estimator differentiable -- and a Fisher forecast is invariant to it, so the
  Fisher uses the raw multipoles and only the diagnostics/plots decouple.

With decoupling the estimator recovers the closed-form linear Kaiser ratios
(`P0 = b1^2(1 + 2/3 beta + 1/5 beta^2)`, `P2/P0 = (4/3 beta + 4/7 beta^2)/(...)`,
Hamilton 1992) at the few-percent level for `P_0`, `P_2`. **`P_4` is
noise-dominated** at toy box sizes (its signal is `~0.04 P_0`, below a 64^3 box's
floor), so the science uses `P_0 + P_2`.

## Interlacing

`fields.interlaced_density_contrast` paints the particles twice (the second on a
half-cell-shifted grid) and averages in Fourier space with the realigning phase,
cancelling the leading CIC aliasing image (Sefusatti et al. 2016,
arXiv:1512.07295). It is now the **standard measurement painter** for all P(k) /
band-power diagnostics and the Fisher Jacobians (real- and redshift-space); only
the force solve (`forces.forces_on_particles`) keeps plain CIC. Measured (at the
probe config, L=256 Mpc/h, N=64): near Nyquist the plain CIC power turns up from
aliasing while the interlaced estimator stays flat; the two are identical at low
k -- so the f_NL signal (low k) and the autodiff `d ln P / d theta` (the window
cancels in the log-derivative) are unchanged, the win is honest high-k
diagnostics.

## The Fisher: does the quadrupole help break the f_NL degeneracy?

The autodiff Fisher (`mbody.fisher`) is extended to redshift space over
`theta = {f_NL, b1, b2, A, f_growth}`, with the linear multipole band powers
`P_ell(k)` as the data vector (not `ln P` -- the quadrupole can be negative) and a
Gaussian mock block covariance (`multipole_gaussian_covariance`) that couples
`P_0`/`P_2` within a k-bin. `f_growth` is downstream of the trajectory (a cheap
fixed-field `mx.grad`); `f_NL`, `A` remain IC-stage and share the reversible
adjoint, now **seeded with the momentum cotangent** (`loss_uses_momentum=True`)
because the redshift-space field depends on the final velocities.

Adding the quadrupole (L=512 Mpc/h, prior `sigma(A)=0.1`):

| forward model | `sigma(f_NL)` mono | mono+quad | `sigma(f_growth)` mono | mono+quad |
|---|---|---|---|---|
| linear field | ~620 | ~390 | ~3.8 | ~0.30 |
| PM-evolved   | ~940 | ~580 | ~1.2 | ~0.08 |

The quadrupole sharply constrains the growth amplitude `f_growth` (`>10x`) and
**partially recovers** `sigma(f_NL)` (`~1.6x`) lost to the bias/amplitude
degeneracy.

### Honest caveats

- **Fingers-of-god are absent.** A PM under-resolves virial velocities, so RSD is
  trustworthy only in the large-scale Kaiser regime (`k < ~0.1 h/Mpc`); the
  small-scale quadrupole is biased. Fine for `f_NL`, which lives at `k -> 0`.
- **RSD does not break the `b_phi * f_NL` degeneracy** -- and neither does
  multi-tracer by itself. Both a single tracer's scale-dependent-bias amplitude
  and the multi-tracer cross spectra constrain the *product* `f_NL * b_phi`, not
  `f_NL` alone (Barreira 2022, arXiv:2205.05673; Barreira & Krause 2023,
  arXiv:2302.09066). Pinning `f_NL` needs a `b_phi(b1)` relation (e.g.
  universality); multi-tracer *relaxes and robustifies* that prior and sharpens
  the products via sample-variance cancellation. That is the SPHEREx strategy, now
  built and quantified in `docs/multitracer.md` (`pixi run multitracer`). The
  **redshift-space multi-tracer** combines both levers -- the cancellation tightens
  `sigma(f_NL)` and the quadrupole pins `f_growth` -- see the composition section of
  `docs/multitracer.md` (`pixi run rsd-multitracer`).
- The PM forecast reuses the **Gaussian (disconnected) mock covariance**; the PM
  nonlinearity enters the signal/Jacobian, not the covariance.

## Reproduce

```
pixi run rsd                 # probe: all tolerances above (conversion, Kaiser,
                             #   interlacing, Jacobian vs FD, adjoint vs replay,
                             #   mono-vs-quad Fisher)
pixi run rsd-multipoles      # outputs/rsd_multipoles.png  (P0/P2 vs Kaiser)
pixi run fisher-rsd          # outputs/fisher_rsd.png + PM headline sigmas
pixi run python -m pytest tests/test_rsd.py
```

## References

- Kaiser 1987, MNRAS 227, 1 -- the redshift-space mapping.
- Hamilton 1992, ApJ 385, L5 -- linear multipole coefficients (predates arXiv;
  see also the Hamilton 1998 review, arXiv:astro-ph/9708102).
- Sefusatti et al. 2016 (arXiv:1512.07295) -- interlacing / accurate estimators.
- Barreira 2022 (arXiv:2205.05673) -- `b_phi`-`f_NL` degeneracy, why multi-tracer.
