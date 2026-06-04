"""CIC density field of Zel'dovich-displaced particles: slab map + P(k).

Paints the LPT particle distribution onto the mesh and shows (left) a projected
density slab -- the cosmic web rendered as a smooth field rather than a point
scatter -- and (right) its measured P(k) against the linear theory. At z = 0 the
measured power falls below linear at high k: the Zel'dovich washout (particles
streaming through each other with no gravity to bind them), which the PM force
solve will fix.

Run: pixi run python scripts/plot_density.py
Saves: outputs/density.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology  # noqa: E402
from mbody import cosmology as C  # noqa: E402
from mbody import fields as F  # noqa: E402
from mbody import lpt as L  # noqa: E402
from mbody import painting as PA  # noqa: E402


def main():
    cosmo = Cosmology()
    box = BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)

    pos = L.lpt_positions(box, cosmo, seed=0, z=0.0)
    delta = PA.density_contrast(pos, box)
    ks, pk, _ = F.power_spectrum(delta, box)
    P_lin = C.linear_power(ks, cosmo, z=0.0, backend="camb")

    # Projected density of a slab (a few cells thick), in 1 + delta for log color.
    thick = 16
    dn = np.asarray(delta, np.float64)
    slab = (1.0 + dn[:, :, :thick]).mean(axis=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    extent = [0, box.box_size, 0, box.box_size]
    im = ax1.imshow(
        np.log10(np.clip(slab, 1e-2, None)).T,
        origin="lower",
        extent=extent,
        cmap="inferno",
    )
    ax1.set_xlabel("x [Mpc/h]")
    ax1.set_ylabel("y [Mpc/h]")
    ax1.set_title("CIC density of Zel'dovich particles, z=0 (log 1+delta)")
    fig.colorbar(im, ax=ax1, label="log10(1 + delta)", fraction=0.046)

    ax2.loglog(ks, pk, "o", ms=4, label="measured (Zel'dovich + CIC)")
    ax2.loglog(ks, P_lin, "k-", lw=1.5, label="linear P(k) [CAMB, z=0]")
    ax2.axvline(box.k_nyquist, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("k [h/Mpc]")
    ax2.set_ylabel(r"$P(k)$ [$(\mathrm{Mpc}/h)^3$]")
    ax2.set_title("power: Zel'dovich washout at high k")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.2)

    os.makedirs("outputs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/density.png", dpi=130)
    print("wrote outputs/density.png")


if __name__ == "__main__":
    main()
