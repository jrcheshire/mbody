"""Plot the linear matter power spectrum: CAMB vs EH98 (full and no-wiggle).

The first milestone with a real, physical curve. Top panel: P(k) from the CAMB
Boltzmann solver and the two EH98 fitting forms. Bottom panel: the ratio of
each EH98 form to CAMB -- you can read off the ~few-percent accuracy of the
fit, and see the baryon acoustic oscillations as the wiggles the no-wiggle
form deliberately omits.

Run: pixi run python scripts/plot_linear_power.py
Saves: outputs/linear_power.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology  # noqa: E402
from mbody import cosmology as C  # noqa: E402


def main():
    cosmo = Cosmology()
    box = BoxConfig()

    k = np.logspace(-3, 1, 600)
    P_camb = C.linear_power(k, cosmo, backend="camb")
    P_eh98 = C.linear_power(k, cosmo, backend="eh98")
    P_nw = C.linear_power(k, cosmo, backend="eh98_nowiggle")

    # A few derived numbers worth printing alongside the figure.
    p = C._EH98(cosmo)
    k_eq = p.k_eq / cosmo.h
    kk = np.logspace(-3, 0, 3000)
    k_peak = kk[np.argmax(C.linear_power(kk, cosmo, backend="camb"))]
    broad = (k > 1e-3) & (k < 10)
    max_dev = np.max(np.abs(P_eh98[broad] / P_camb[broad] - 1.0))
    m_particle = C.mean_matter_density(cosmo) * (box.box_size / box.n_particles) ** 3

    print("Linear power spectrum summary (default cosmology):")
    print(f"  sigma8 (CAMB)          = {C.sigma_R(8.0, cosmo, backend='camb'):.4f}")
    print(f"  k_eq                   = {k_eq:.4f} h/Mpc")
    print(f"  P(k) turnover          = {k_peak:.4f} h/Mpc")
    print(f"  EH98 vs CAMB max dev   = {100 * max_dev:.1f}% over [1e-3,10]")
    print(f"  box k_fundamental      = {box.k_fundamental:.4f} h/Mpc")
    print(f"  box k_nyquist          = {box.k_nyquist:.3f} h/Mpc")
    print(
        f"  particle mass (L={box.box_size:.0f}, Np={box.n_particles}^3)"
        f" = {m_particle:.3e} M_sun/h"
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.loglog(k, P_camb, "k-", lw=2, label="CAMB (Boltzmann)")
    ax1.loglog(k, P_eh98, "C0--", lw=1.5, label="EH98 (full, with BAO)")
    ax1.loglog(k, P_nw, "C1:", lw=1.5, label="EH98 (no-wiggle)")
    ax1.axvline(k_eq, color="gray", ls="-", lw=0.8, alpha=0.6)
    ax1.text(k_eq * 1.1, P_camb.min() * 3, "k_eq", color="gray", fontsize=9)
    # Mark the box's accessible range (fundamental mode to mesh Nyquist).
    ax1.axvspan(box.k_fundamental, box.k_nyquist, color="C2", alpha=0.07)
    ax1.set_ylabel(r"$P(k)$  [$(\mathrm{Mpc}/h)^3$]")
    ax1.set_title("Linear matter power spectrum (z=0)")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.2)

    ax2.semilogx(k, P_eh98 / P_camb - 1.0, "C0--", lw=1.5, label="EH98 full")
    ax2.semilogx(k, P_nw / P_camb - 1.0, "C1:", lw=1.5, label="EH98 no-wiggle")
    ax2.axhline(0.0, color="k", lw=0.8)
    ax2.set_ylim(-0.1, 0.1)
    ax2.set_xlabel(r"$k$  [$h/\mathrm{Mpc}$]")
    ax2.set_ylabel("ratio - 1")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.2)

    os.makedirs("outputs", exist_ok=True)
    out = "outputs/linear_power.png"
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
