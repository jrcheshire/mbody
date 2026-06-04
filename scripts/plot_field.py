"""Visualize a Gaussian random field: a density slice + measured vs input P(k).

The "we made a universe" figure: a 2D slice through the 3D density field, next to
its measured power spectrum overlaid on the input P(k). The scatter in the
measured points (especially at low k, where few modes exist) is real cosmic
variance, not error.

Run: pixi run python scripts/plot_field.py
Saves: outputs/field_demo.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology  # noqa: E402
from mbody import cosmology as C  # noqa: E402
from mbody import fields as F  # noqa: E402


def main():
    cosmo = Cosmology()
    box = BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)

    delta = F.gaussian_random_field(box, cosmo, seed=0)
    mx.eval(delta)
    sl = np.asarray(delta[:, :, box.n_mesh // 2], dtype=np.float64)
    ks, pk, nmodes = F.power_spectrum(delta, box)
    kfine = np.logspace(np.log10(ks.min()), np.log10(ks.max()), 200)
    P_in = C.linear_power(kfine, cosmo, backend="camb")

    sel = nmodes >= 20
    ratio = np.average((pk / C.linear_power(ks, cosmo))[sel], weights=nmodes[sel])
    print("field demo (256 Mpc/h, 128^3):")
    print("  std(delta)                         = %.3f" % float(mx.std(delta)))
    print("  modes-weighted measured/input P(k) = %.3f (one realization)" % ratio)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    extent = [0, box.box_size, 0, box.box_size]
    im = ax1.imshow(sl.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-3, vmax=3)
    ax1.set_xlabel("x [Mpc/h]")
    ax1.set_ylabel("y [Mpc/h]")
    ax1.set_title("Gaussian density field delta(x), mid-plane slice")
    fig.colorbar(im, ax=ax1, label="delta", fraction=0.046)

    ax2.loglog(ks, pk, "o", ms=4, label="measured (1 realization)")
    ax2.loglog(kfine, P_in, "k-", lw=1.5, label="input P(k) [CAMB]")
    ax2.axvline(box.k_nyquist, color="gray", ls="--", lw=0.8)
    ax2.text(box.k_nyquist * 0.6, P_in.max(), "Nyquist", color="gray", fontsize=8)
    ax2.set_xlabel("k [h/Mpc]")
    ax2.set_ylabel(r"$P(k)$ [$(\mathrm{Mpc}/h)^3$]")
    ax2.set_title("measured vs input power spectrum")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.2)

    os.makedirs("outputs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/field_demo.png", dpi=130)
    print("wrote outputs/field_demo.png")


if __name__ == "__main__":
    main()
