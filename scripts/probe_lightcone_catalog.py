"""Lightcone galaxy-catalog validation (task #5).

Assembles the full capability -- a radially-evolving Lagrangian-bias tracer on a
multi-snapshot lightcone, Poisson-sampled (catalog.lightcone_catalog) -- and checks
that the bias and its f_NL response EVOLVE correctly along the line of sight (the point
of rejecting an effective redshift):

  1. b(r): per shell, the low-k cross/auto with the matter field recovers the input
     b1(z_shell) (the linear bias evolves as prescribed);
  2. b_phi(r): per shell, the matched-phase through-PM d ln P_g/df_NL matches the linear
     Dalal prediction 2 b_phi(z)/(b1(z) M(k)) with b_phi(z) = 2 delta_c (b1(z) - p), so
     the injected f_NL scale-dependent bias is undiluted AND varies along the LoS;
  3. assembly: catalog.lightcone_catalog produces galaxies that fall in their shells'
     comoving ranges with the prescribed radial number density.

Run: ``pixi run python scripts/probe_lightcone_catalog.py``.
"""

import mlx.core as mx
import numpy as np

from mbody import bias as B
from mbody import catalog as CAT
from mbody import cosmology as C
from mbody import fields as F
from mbody import ic as IC
from mbody import integrate as IG
from mbody import painting as PA
from mbody.config import BoxConfig, Cosmology, CatalogSampling, TimeStepping

BACKEND = "eh98"
DELTA_C = 1.686
P_BPHI = 0.55


def bias_fn(z):
    """A monotonic linear bias evolution b1(z) for the validation."""
    return 1.5 + 1.0 * z


def b_phi_of(z):
    return 2.0 * DELTA_C * (bias_fn(z) - P_BPHI)


def pweighted_inv_M(box, cosmo, k_bins, dk, z):
    _, _, k_mag = F.k_grid(box)
    km = k_mag.ravel()
    P_lin = C.linear_power(km, cosmo, z=z, backend=BACKEND)
    M = IC.poisson_M(np.where(km > 0, km, 1.0), cosmo, z=z, backend=BACKEND)
    inv_M = np.where(km > 0, P_lin / M, 0.0)
    P_wt = np.where(km > 0, P_lin, 0.0)
    out = np.empty(len(k_bins))
    for b, kc in enumerate(k_bins):
        m = (km >= kc - 0.5 * dk) & (km < kc + 0.5 * dk)
        out[b] = inv_M[m].sum() / P_wt[m].sum()
    return out


def run_pm_snaps(box, cosmo, time, seed, f_NL):
    """One forward run; return (a_snaps, snaps=[(a, positions_np), ...])."""
    snaps = []
    x0, p0 = IG.initial_state(
        box, cosmo, time, seed=seed, f_NL=f_NL, backend=BACKEND, lpt_order=2
    )
    IG.evolve_state(
        x0,
        p0,
        box,
        cosmo,
        IG.a_grid(time),
        snapshot=lambda step, a, x, p: snaps.append(
            (float(a), np.asarray(x, np.float32))
        ),
        integrator=time.integrator,
    )
    return np.array([a for a, _ in snaps]), snaps


def shell_delta_g(snaps, a_snaps, box, cosmo, z, f_NL, seed):
    """Lagrangian-bias tracer overdensity for the shell at z (nearest snapshot)."""
    si = int(np.argmin(np.abs(a_snaps - 1.0 / (1.0 + z))))
    b1 = bias_fn(z)
    delta1 = IC.linear_density(box, cosmo, seed=seed, z=z, f_NL=f_NL, backend=BACKEND)
    phi_G = IC.primordial_potential(box, cosmo, seed=seed, z=z, backend=BACKEND)
    dgL = B.lagrangian_bias_field(
        delta1, box, b1=b1 - 1.0, b_phi=b_phi_of(z), f_NL=f_NL, phi_G=phi_G
    )
    w = (1.0 + dgL).reshape(-1)
    n_g = PA.cic_paint(mx.array(snaps[si][1]), box, weights=w)
    return np.asarray(n_g) / (float(w.sum()) / box.n_mesh**3) - 1.0, si


def main():
    cosmo = Cosmology()
    L, N, seed = 6000.0, 96, 0
    box = BoxConfig(box_size=L, n_mesh=N)
    z_edges = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3])
    z_mid = 0.5 * (z_edges[1:] + z_edges[:-1])
    time = TimeStepping(
        z_init=9.0, z_final=float(z_edges[0]), n_steps=24, integrator="bullfrog"
    )
    dk = box.k_fundamental
    k_low = dk * np.arange(1, 6)

    # (1) b(r): per-shell linear bias from the clean run.
    a0, snaps0 = run_pm_snaps(box, cosmo, time, seed, 0.0)
    print("=== (1) b(r): per-shell linear bias recovers input b1(z) ===")
    print(f"  {'z':>6} {'b1_in':>7} {'b1_meas':>8} {'meas/in':>8}")
    for zc in z_mid:
        dg, si = shell_delta_g(snaps0, a0, box, cosmo, float(zc), 0.0, seed)
        dm = np.asarray(PA.density_contrast(mx.array(snaps0[si][1]), box))
        Pc = np.asarray(
            F.cross_power(
                mx.array(dg.astype(np.float32)),
                mx.array(dm.astype(np.float32)),
                box,
                k_low,
                dk=dk,
            )
        )
        Pm = np.asarray(
            F.band_power(mx.array(dm.astype(np.float32)), box, k_low, dk=dk)
        )
        b_meas = float(np.mean(Pc / Pm))
        print(f"  {zc:6.2f} {bias_fn(zc):7.3f} {b_meas:8.3f} {b_meas/bias_fn(zc):8.4f}")

    # (2) b_phi(r): matched-phase through-PM f_NL response per shell.
    eps = 10.0
    ap, snapsp = run_pm_snaps(box, cosmo, time, seed, +eps)
    am, snapsm = run_pm_snaps(box, cosmo, time, seed, -eps)
    print("\n=== (2) b_phi(r): through-PM dlnP_g/df_NL vs linear Dalal (per shell) ===")
    print(f"  {'z':>6} {'b1':>6} {'b_phi':>7} {'meas/pred':>10}")
    for zc in z_mid:
        zc = float(zc)
        dgp, _ = shell_delta_g(snapsp, ap, box, cosmo, zc, +eps, seed)
        dgm, _ = shell_delta_g(snapsm, am, box, cosmo, zc, -eps, seed)
        Pp = np.asarray(
            F.band_power(mx.array(dgp.astype(np.float32)), box, k_low, dk=dk)
        )
        Pm = np.asarray(
            F.band_power(mx.array(dgm.astype(np.float32)), box, k_low, dk=dk)
        )
        meas = (np.log(Pp) - np.log(Pm)) / (2.0 * eps)
        pred = (
            2.0
            * b_phi_of(zc)
            / bias_fn(zc)
            * pweighted_inv_M(box, cosmo, k_low, dk, zc)
        )
        print(
            f"  {zc:6.2f} {bias_fn(zc):6.3f} {b_phi_of(zc):7.3f}"
            f" {float(np.mean(meas / pred)):10.4f}"
        )

    # (3) assembly: lightcone_catalog galaxies land in their shells with n(r).
    observer = (L / 2, L / 2, L / 2)
    sampling = CatalogSampling(nbar=1e-4, draw_seed=0)
    cat = CAT.lightcone_catalog(
        box,
        cosmo,
        time,
        observer,
        z_edges,
        bias_fn,
        sampling.nbar,
        sampling,
        f_NL=0.0,
        seed=seed,
        p=P_BPHI,
        backend=BACKEND,
    )
    r = np.sqrt((cat.xyz**2).sum(1))  # observer-centered output -> r = |xyz|
    chi_edges = C.comoving_distance(z_edges, cosmo)
    print(
        f"\n=== (3) assembly: {cat.n_galaxies} galaxies, r in "
        f"[{r.min():.0f}, {r.max():.0f}] Mpc/h ==="
    )
    print(
        f"  shell range [{chi_edges[0]:.0f}, {chi_edges[-1]:.0f}];"
        f" in-range fraction {np.mean((r >= chi_edges[0]) & (r <= chi_edges[-1])):.4f}"
    )
    vol = 4.0 / 3.0 * np.pi * (chi_edges[1:] ** 3 - chi_edges[:-1] ** 3)
    print(f"  {'z':>6} {'n_gal':>9} {'density/nbar':>13}")
    for s in range(len(z_mid)):
        n = int(((r >= chi_edges[s]) & (r < chi_edges[s + 1])).sum())
        print(f"  {z_mid[s]:6.2f} {n:9d} {n / vol[s] / sampling.nbar:13.4f}")


if __name__ == "__main__":
    main()
