"""Zel'dovich structure formation: particle slabs at several redshifts.

The same Lagrangian slab of particles, displaced by the Zel'dovich approximation
at decreasing redshift -- a uniform grid flowing into the filamentary pattern as
the growth factor rises. (Still no gravity solve: this is LPT + D(z).)

Run: pixi run python scripts/plot_lpt.py
Saves: outputs/lpt_growth.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology  # noqa: E402
from mbody import cosmology as C  # noqa: E402
from mbody import lpt as L  # noqa: E402


def main():
    cosmo = Cosmology()
    box = BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)
    N = box.n_mesh
    psi = L.zeldovich_displacement(box, cosmo, seed=0)
    thick = 4  # Lagrangian z-slab thickness, in cells
    redshifts = [49.0, 4.0, 1.0, 0.0]

    fig, axes = plt.subplots(1, len(redshifts), figsize=(4 * len(redshifts), 4.2))
    for ax, z in zip(axes, redshifts):
        D = C.growth_factor(z, cosmo)
        pos = np.asarray(L.displace(box, psi, D), np.float64).reshape(N, N, N, 3)
        slab = pos[:, :, :thick, :].reshape(-1, 3)
        ax.scatter(slab[:, 0], slab[:, 1], s=0.3, c="k", alpha=0.5, linewidths=0)
        ax.set_xlim(0, box.box_size)
        ax.set_ylim(0, box.box_size)
        ax.set_aspect("equal")
        ax.set_title("z = %.0f   (D = %.3f)" % (z, D))
        ax.set_xlabel("x [Mpc/h]")
    axes[0].set_ylabel("y [Mpc/h]")
    fig.suptitle("Zel'dovich displacement: structure emerging (Lagrangian slab)")

    os.makedirs("outputs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/lpt_growth.png", dpi=130)
    print("wrote outputs/lpt_growth.png")


if __name__ == "__main__":
    main()
