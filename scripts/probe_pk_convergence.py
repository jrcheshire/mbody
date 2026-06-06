"""Probe: does the PM forward model converge to analytic linear theory?

The external cross-check, self-contained half (Layer A): compare the evolved field
to linear theory with ABSOLUTE normalization, on matched phases so cosmic variance
cancels. Three measurements, each printed with the numbers that set the tolerances
in tests/test_convergence.py:

  A1. Matched transfer T(k) = P_PM(k) / P_lin(k): the evolved matter power divided
      by the SAME-seed linear field's power. -> 1 at low k (absolute growth +
      normalization); at high k it droops because a coarse PM under-resolves small
      scales (and, raw, the CIC window suppresses the particle-painted P_PM -- shown
      both raw and deconvolved). Also the IC-stage check P(x0) / P_lin(z_init) -> 1.
  A2. Propagator r(k) = <delta_PM delta_lin> / sqrt(P_PM P_lin): -> 1 at low k,
      decohering toward high k (the FastPM validation metric).
  A3. Growth convergence: the large-scale growth R(a_final) vs the linear
      D(z=0)/D(z_init), as a function of step count for "exact" (converges) and
      "fastpm" (exact at any step count).

Run: pixi run python scripts/probe_pk_convergence.py
Nothing is written; scripts/plot_convergence.py renders the figure.
"""

import numpy as np

from mbody.config import (
    BoxConfig,
    Cosmology,
    InitialConditions,
    TimeStepping,
    SimConfig,
)
from mbody import cosmology as C
from mbody import fields as F
from mbody import fisher as FI
from mbody import ic as IC
from mbody import diagnostics as D
from mbody import driver

COSMO = Cosmology()
BOX = BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
TIME = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10)
NSEED = 8


def linear_field(seed):
    """The z=0 linear density realization with the same phases the run uses."""
    return IC.linear_density(BOX, COSMO, seed=seed, z=0.0)


def main():
    cfg0 = SimConfig(box=BOX, time=TIME)
    print(cfg0.summary())
    print(f"\nseeds = {NSEED}\n")

    # --- A1 / A2: matched transfer T(k), IC transfer, propagator r(k) ---
    # T_raw = P_PM/P_lin (particle CIC window suppresses high k); T_dec deconvolves
    # the CIC window (P_lin is a grid field, no window). Both -> 1 at low k.
    k_pm = None
    Traw_sum = Tdec_sum = Tdec2_sum = r_sum = Tic_sum = None
    for s in range(NSEED):
        cfg = SimConfig(box=BOX, time=TIME, ic=InitialConditions(seed=s))
        res = driver.run(cfg)
        kk, P_pm, _ = res.power()
        _, P_pm_dec, _ = res.power(deconvolve_cic=True)
        dlin = linear_field(s)
        _, P_lin_real, _ = F.power_spectrum(dlin, BOX)
        _, P_ic, _ = F.power_spectrum(res.ic_field, BOX, deconvolve_cic=True)
        _, r, _ = D.cross_correlation(res.final_field, dlin, BOX)
        if k_pm is None:
            k_pm = kk
            Traw_sum = np.zeros_like(kk)
            Tdec_sum = np.zeros_like(kk)
            Tdec2_sum = np.zeros_like(kk)
            r_sum = np.zeros_like(kk)
            Tic_sum = np.zeros_like(kk)
        Traw_sum += P_pm / P_lin_real
        Tdec_sum += P_pm_dec / P_lin_real
        Tdec2_sum += (P_pm_dec / P_lin_real) ** 2
        r_sum += r
        Tic_sum += P_ic / C.linear_power(kk, COSMO, z=TIME.z_init)

    T_raw = Traw_sum / NSEED
    T_dec = Tdec_sum / NSEED
    T_err = np.sqrt(np.maximum(Tdec2_sum / NSEED - T_dec**2, 0.0)) / np.sqrt(NSEED)
    r = r_sum / NSEED
    Tic = Tic_sum / NSEED
    cv = np.sqrt(FI.band_power_log_variance(BOX, k_pm))

    knyq = BOX.k_nyquist
    print("A1. matched transfer T(k) = P_PM / P_lin (z=0), and IC transfer (z_init):")
    print("  (T_raw = particle CIC window; T_dec = CIC-deconvolved; per-bin cv shown)")
    print("  k/knyq   k        T_raw    T_dec+/-SEM        IC:P/P_lin(z_init)")
    for i, k in enumerate(k_pm):
        print(
            f"  {k/knyq:5.3f}  {k:7.4f}  {T_raw[i]:6.3f}   "
            f"{T_dec[i]:6.3f} +/- {T_err[i]:5.3f} (cv {cv[i]:.2f})   {Tic[i]:6.3f}"
        )
    lowk = k_pm < 0.15 * knyq
    print(
        f"\n  low-k (k<0.15 knyq) deconvolved T(k): mean={T_dec[lowk].mean():.4f}, "
        f"max|T-1|={np.max(np.abs(T_dec[lowk]-1)):.4f}"
    )
    print(
        f"  low-k IC transfer P(x0)/P_lin(z_init): mean={Tic[lowk].mean():.4f}, "
        f"max|.-1|={np.max(np.abs(Tic[lowk]-1)):.4f}"
    )

    print("\nA2. propagator r(k)=<delta_PM delta_lin>/sqrt(P_PM P_lin):")
    print("  k/knyq   k         r(k)")
    for i, k in enumerate(k_pm):
        print(f"  {k/knyq:5.3f}  {k:7.4f}  {r[i]:7.4f}")
    print(f"\n  low-k (k<0.15 knyq) r(k): min={r[lowk].min():.4f}")
    # Decoherence scale: first k where r drops below 0.99.
    below = np.where(r < 0.99)[0]
    if below.size:
        print(
            f"  r first drops below 0.99 at k={k_pm[below[0]]:.4f} "
            f"({k_pm[below[0]]/knyq:.3f} knyq)"
        )

    # --- A3: growth convergence vs step count ---
    print("\nA3. large-scale growth R(a_final) / [D(0)/D(z_init)] vs n_steps:")
    D_target = C.growth_factor(0.0, COSMO) / C.growth_factor(TIME.z_init, COSMO)
    print(f"  linear target D(0)/D(z_init) = {D_target:.4f}")
    print("  n_steps   exact      fastpm")
    for nst in (2, 5, 10, 20):
        row = {}
        for integ in ("exact", "fastpm"):
            t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=nst, integrator=integ)
            res = driver.run(SimConfig(box=BOX, time=t), record=True)
            a_arr, R = res.growth_history()
            Dref = D.linear_growth_reference(a_arr, COSMO)
            # R and Dref are both normalized to 1 at a_init; final ratio to linear.
            row[integ] = R[-1] / Dref[-1]
        print(f"  {nst:5d}    {row['exact']:7.4f}    {row['fastpm']:7.4f}")


if __name__ == "__main__":
    main()
