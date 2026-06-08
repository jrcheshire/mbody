"""Validation probe for the galaxy mock catalog (mbody.catalog) -- Stage B.

Three measurements over an ensemble of seeds, in the framing the tests use:

  (i)   nbar recovery -- N_gal / V vs the realized intensity nbar (and the small
        inflation of the realized nbar above the target from clipping).

  (ii)  sampler fidelity -- the Poisson + sub-cell-placement draw is UNBIASED: the
        sampled catalog reproduces the *noiseless* clipped intensity field's
        low-k cross-bias with the matter field. Measured as
        cross(catalog, matter) / cross(noiseless_clipped, matter), which -> 1 at
        low k (a mild droop at higher k is the uniform-in-cell placement window,
        a sinc that -> 1 as k -> 0). This isolates the sampler from the clip.

  (iii) clip bias-shift -- the *physical* effect of the clip non-negativity map:
        clipping deep voids lowers the effective large-scale bias by a
        scale-independent factor that grows with the clip fraction. Reported as
        b_eff / b1 = cross(noiseless_clipped, matter) / P_matter / b1 at low k,
        for a few input biases. This is a documented feature of clipped mocks,
        not an error (and it grows with bias -- relevant for high-b shells).

This PRINTS the numbers behind the locked test tolerances; nothing here asserts.
Run: ``pixi run python scripts/probe_mock_validation.py``.
"""

import numpy as np
import mlx.core as mx

from mbody import bias as B
from mbody import catalog as CAT
from mbody import fields as F
from mbody import painting as PA
from mbody.config import (
    BoxConfig,
    CatalogSampling,
    Cosmology,
    InitialConditions,
    SimConfig,
    TimeStepping,
    Tracer,
)
from mbody.driver import run

COSMO = Cosmology()
BOX = BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
TIME = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5, integrator="fastpm")
NBAR = 1.0e-2
NSEED = 24
BACKEND = "eh98"


def _kbins(n_low=6):
    return np.arange(1, n_low + 1) * BOX.k_fundamental


def _gal_field(xyz):
    return PA.density_contrast(mx.array(np.ascontiguousarray(xyz, np.float32)), BOX)


def _run(seed, b1, nbar=NBAR):
    cfg = SimConfig(
        cosmology=COSMO,
        box=BOX,
        time=TIME,
        ic=InitialConditions(f_NL=0.0, kind="gaussian", seed=seed),
        tracer=Tracer(b1=b1, b2=0.0, A=1.0),
        catalog=CatalogSampling(enabled=True, nbar=nbar, draw_seed=seed),
    )
    return run(cfg, backend=BACKEND)


def _noiseless_clipped(delta_m, b1, nbar=NBAR):
    """The mean galaxy overdensity field (clipped intensity), no Poisson noise."""
    delta_g = B.local_bias_tracer(delta_m, b1, 0.0)
    lam = CAT.intensity_field(delta_g, nbar, BOX)
    return mx.array(np.asarray(lam / lam.mean() - 1.0, np.float32))


def main():
    kb = _kbins()
    print(
        f"box L={BOX.box_size} n_mesh={BOX.n_mesh} steps={TIME.n_steps} "
        f"nbar={NBAR} nseed={NSEED} backend={BACKEND}"
    )
    print(f"k bins (h/Mpc): {np.array2string(kb, precision=4)}\n")

    # ---- (i) nbar recovery + (ii) sampler fidelity (b1 = 1.5) ----
    b1 = 1.5
    nbar_ratio, clip, realized, fidelity = [], [], [], []
    for s in range(NSEED):
        res = _run(s, b1)
        cat = res.catalog
        delta_m = PA.density_contrast(res.x, BOX)
        delta_e = _noiseless_clipped(delta_m, b1)
        x_em = np.asarray(F.cross_power(delta_e, delta_m, BOX, kb), np.float64)
        x_cm = np.asarray(
            F.cross_power(_gal_field(cat.xyz), delta_m, BOX, kb), np.float64
        )
        fidelity.append(x_cm / x_em)
        nbar_ratio.append(cat.n_galaxies / BOX.box_size**3 / cat.realized_nbar)
        clip.append(cat.clip_fraction)
        realized.append(cat.realized_nbar)
    nbar_ratio = np.array(nbar_ratio)
    fidelity = np.array(fidelity)

    print("--- (i) nbar recovery (b1=1.5) ---")
    print(f"  target nbar       : {NBAR:.5e}")
    print(
        f"  realized (clipped): {np.mean(realized):.5e}  "
        f"(inflation {np.mean(realized)/NBAR - 1:+.4f}; "
        f"clip mean={np.mean(clip):.4f} max={np.max(clip):.4f})"
    )
    print(
        f"  N_gal/V / realized: mean={nbar_ratio.mean():.5f} "
        f"max|dev|={np.abs(nbar_ratio - 1).max():.5f}   [tol < 0.01]\n"
    )

    print("--- (ii) sampler fidelity: cross(catalog,m)/cross(noiseless,m) ---")
    print("   bin    k        ratio    sem      |dev|")
    f_mean = fidelity.mean(axis=0)
    f_sem = fidelity.std(axis=0, ddof=1) / np.sqrt(NSEED)
    for i, k in enumerate(kb):
        print(
            f"   {i+1:>3}  {k:7.4f}   {f_mean[i]:6.4f}  {f_sem[i]:6.4f}  "
            f"{abs(f_mean[i]-1):6.4f}"
        )
    print(
        f"  lowest 2 bins max|dev| = {np.abs(f_mean[:2]-1).max():.4f}   "
        f"[tol < 0.02]\n"
    )

    # ---- (iii) clip bias-shift: b_eff/b1 vs clip fraction ----
    print("--- (iii) clip bias-shift (physical; diagnostic only) ---")
    print("   b1     clip      b_eff/b1 (bins 1-3)")
    for b1 in (1.0, 1.5, 2.0):
        ratios, clips = [], []
        for s in range(max(8, NSEED // 2)):
            res = _run(s, b1)
            delta_m = PA.density_contrast(res.x, BOX)
            delta_e = _noiseless_clipped(delta_m, b1)
            x_em = np.asarray(F.cross_power(delta_e, delta_m, BOX, kb[:3]), np.float64)
            p_mm = np.asarray(F.band_power(delta_m, BOX, kb[:3]), np.float64)
            ratios.append(x_em / p_mm / b1)
            clips.append(res.catalog.clip_fraction)
        r = np.array(ratios).mean(axis=0)
        print(
            f"   {b1:<5}  {np.mean(clips):.4f}    " f"{np.array2string(r, precision=4)}"
        )


if __name__ == "__main__":
    main()
