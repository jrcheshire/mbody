"""Multi-snapshot lightcone validation (1b) -- the effective-redshift fix.

A single PM snapshot puts every galaxy at one growth D(z_eff); a real survey is a
lightcone where the clustering amplitude declines with comoving distance as D(z(r))
and the bias evolves -- b1(r), b_phi(r) and growth D(r) vary continuously along the
line of sight (as spherical-harmonic / SFB analyses integrate, with no single
effective redshift). This builds an ONION lightcone -- capture snapshots during one
forward run, then assign each radial shell the snapshot at the scale factor a(r)
matching its comoving distance -- and checks three things:

  1. snapshot growth: the low-k cross-amplitude (cross-power with the linear field)
     of each captured snapshot tracks D(a) -- the PM growth the lightcone draws on;
  2. the EFFECTIVE-Z ERROR: a single-snapshot mock has a flat amplitude vs r, while
     the lightcone amplitude follows D(z(r)) -- so effective-z mis-states the
     radial clustering by up to D(z_hi)/D(z_lo) across the sample;
  3. geometry: the onion-stitched matter number density is continuous across shell
     boundaries (clean comoving-distance stitching, no gaps/pileups).

The faithful per-particle lightcone-crossing interpolation is the natural production
refinement; the onion here validates the mechanism cheaply.

Run: ``pixi run python scripts/probe_lagrangian_lightcone.py``.
"""

import mlx.core as mx
import numpy as np

from mbody import cosmology as C
from mbody import fields as F
from mbody import ic as IC
from mbody import integrate as IG
from mbody import painting as PA
from mbody.config import BoxConfig, Cosmology, TimeStepping

BACKEND = "eh98"


def growth_amplitude(snap_positions, delta_lin_ref, box, k_bins, dk):
    """Large-scale growth amplitude D(z) of a snapshot, shot-free.

    Cross-power of the snapshot density with the (noiseless) z=0 linear field over
    the lowest |k| shells, divided by the linear auto power: P_x/P_lin = D(z) r(k),
    with the propagator r -> 1 at low k. Avoids the CIC shot/discreteness floor that
    inflates the total rms where the signal is small (high z).
    """
    d = PA.density_contrast(mx.array(snap_positions), box)
    Px = np.asarray(F.cross_power(d, delta_lin_ref, box, k_bins, dk=dk))
    Pl = np.asarray(F.band_power(delta_lin_ref, box, k_bins, dk=dk))
    return float(np.mean(Px / Pl))


def main():
    cosmo = Cosmology()
    L, N = 7000.0, 96
    box = BoxConfig(box_size=L, n_mesh=N)
    obs = np.array([L / 2, L / 2, L / 2], dtype=np.float32)

    z_edges = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    z_mid = 0.5 * (z_edges[1:] + z_edges[:-1])
    chi_edges = C.comoving_distance(z_edges, cosmo)
    assert chi_edges.max() < 0.5 * L - box.cell_size, "shell exceeds inscribed sphere"

    # One forward run, capturing positions at every step (the snapshot ladder).
    time = TimeStepping(
        z_init=9.0, z_final=float(z_edges[0]), n_steps=24, integrator="bullfrog"
    )
    snaps = []

    def rec(step, a, x, p):
        snaps.append((float(a), np.asarray(x, dtype=np.float32) % L))

    x0, p0 = IG.initial_state(box, cosmo, time, seed=0, backend=BACKEND, lpt_order=2)
    IG.evolve_state(
        x0, p0, box, cosmo, IG.a_grid(time), snapshot=rec, integrator="bullfrog"
    )
    a_snaps = np.array([a for a, _ in snaps])

    # Shot-free growth amplitude: cross-power with the z=0 linear field at low k.
    dk = box.k_fundamental
    k_low = dk * np.arange(1, 6)
    delta_lin_ref = IC.linear_density(box, cosmo, seed=0, z=0.0, backend=BACKEND)
    amp_snap = np.array(
        [growth_amplitude(x, delta_lin_ref, box, k_low, dk) for _, x in snaps]
    )

    # (1) snapshot growth: low-k cross amplitude vs D(a).
    z_snap = 1.0 / a_snaps - 1.0
    D_snap = np.array([C.growth_factor(z, cosmo) for z in z_snap])
    ratio = amp_snap / D_snap
    print("=== (1) snapshot growth: low-k cross-amplitude tracks D(a) ===")
    print(f"  {'a':>6} {'z':>6} {'amp':>8} {'D':>7} {'amp/D':>8}")
    for i in range(0, len(snaps), max(1, len(snaps) // 8)):
        print(
            f"  {a_snaps[i]:6.3f} {z_snap[i]:6.3f} {amp_snap[i]:8.4f}"
            f" {D_snap[i]:7.4f} {ratio[i]:8.4f}"
        )
    print(
        f"  -> amp/D spread over the shell range (z<1.5): "
        f"{ratio[a_snaps > 0.4].min():.3f}-{ratio[a_snaps > 0.4].max():.3f}\n"
    )

    # (2) effective-z error vs the lightcone radial growth profile.
    D_mid = np.array([C.growth_factor(z, cosmo) for z in z_mid])
    picked, amp_lc, counts = [], [], []
    for s in range(len(z_mid)):
        si = int(np.argmin(np.abs(a_snaps - 1.0 / (1.0 + z_mid[s]))))
        picked.append(si)
        amp_lc.append(amp_snap[si])
        xnp = snaps[si][1]
        r = np.sqrt(((xnp - obs) ** 2).sum(1))
        counts.append(int(((r >= chi_edges[s]) & (r < chi_edges[s + 1])).sum()))
    amp_lc = np.array(amp_lc)
    counts = np.array(counts)
    truth = D_mid / D_mid[0]  # correct radial amplitude profile
    lc = amp_lc / amp_lc[0]  # lightcone reproduces it
    effz = np.ones_like(truth)  # single-snapshot mock: flat amplitude
    print("=== (2) effective-z ERROR vs lightcone (radial growth profile) ===")
    print(
        f"  {'z':>6} {'chi':>7} {'D/D0 (truth)':>13} {'lightcone':>10}"
        f" {'eff-z':>7} {'eff-z err':>10}"
    )
    for s in range(len(z_mid)):
        chi_mid = 0.5 * (chi_edges[s] + chi_edges[s + 1])
        print(
            f"  {z_mid[s]:6.2f} {chi_mid:7.0f} {truth[s]:13.4f} {lc[s]:10.4f}"
            f" {effz[s]:7.4f} {effz[s]/truth[s]-1:+9.1%}"
        )
    print(
        f"  -> lightcone tracks D(z(r)) to "
        f"{np.max(np.abs(lc/truth-1)):.1%}; effective-z is off up to "
        f"{np.max(np.abs(effz/truth-1)):.0%}\n"
    )

    # (3) geometry: onion-stitched matter density continuous across shells.
    vol = 4.0 / 3.0 * np.pi * (chi_edges[1:] ** 3 - chi_edges[:-1] ** 3)
    dens = counts / vol
    print("=== (3) geometry: stitched matter density vs shell (should be flat) ===")
    print(f"  {'z':>6} {'n_part':>10} {'density/d0':>11}")
    for s in range(len(z_mid)):
        print(f"  {z_mid[s]:6.2f} {counts[s]:10d} {dens[s]/dens[0]:11.4f}")
    print(
        f"  -> density spread across shells: "
        f"{(dens/dens[0]).min():.3f}-{(dens/dens[0]).max():.3f}"
    )


if __name__ == "__main__":
    main()
