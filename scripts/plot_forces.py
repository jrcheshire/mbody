"""Gravity made visible: the potential and the acceleration field of structure.

Takes the Zel'dovich density at z = 0, solves Poisson on the mesh, and shows
(left) the density slab with the in-plane acceleration g = -grad phi overlaid as
arrows -- they point *into* the overdense filaments and knots, which is gravity
pulling matter onto the cosmic web -- and (right) the peculiar potential phi,
whose wells (blue) sit under the overdensities. The arrows here are exactly the
Zel'dovich displacement direction (same i k / k^2 kernel); the integrator will
turn this field into motion.

Run: pixi run python scripts/plot_forces.py
Saves: outputs/forces.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology  # noqa: E402
from mbody import forces as FO  # noqa: E402
from mbody import lpt as L  # noqa: E402
from mbody import painting as PA  # noqa: E402


def main():
    cosmo = Cosmology()
    box = BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)

    pos = L.lpt_positions(box, cosmo, seed=0, z=0.0)
    delta = PA.density_contrast(pos, box)
    phi = FO.potential(delta, box)
    gx, gy, _ = FO.acceleration_field(delta, box)

    N, Lh = box.n_mesh, box.box_size
    thick = 8  # slab thickness in cells, averaged for a smooth 2D view
    dn = np.asarray(delta, np.float64)[:, :, :thick].mean(axis=2)
    ph = np.asarray(phi, np.float64)[:, :, :thick].mean(axis=2)
    ux = np.asarray(gx, np.float64)[:, :, :thick].mean(axis=2)
    uy = np.asarray(gy, np.float64)[:, :, :thick].mean(axis=2)

    # Downsample the field to a readable arrow grid; cell-centred coordinates.
    step = N // 24
    s = slice(step // 2, N, step)
    cc = (np.arange(N) + 0.5) * box.cell_size
    Xq, Yq = np.meshgrid(cc[s], cc[s], indexing="ij")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    extent = [0, Lh, 0, Lh]

    im1 = ax1.imshow(
        np.log10(np.clip(1.0 + dn, 1e-2, None)).T,
        origin="lower",
        extent=extent,
        cmap="inferno",
    )
    # Default autoscaling; arrow length tracks |g| (large-scale flows dominate).
    ax1.quiver(Xq, Yq, ux[s, s], uy[s, s], color="cyan", alpha=0.9, width=0.004)
    ax1.set_xlabel("x [Mpc/h]")
    ax1.set_ylabel("y [Mpc/h]")
    ax1.set_title("density + acceleration g = -grad phi (arrows pull into structure)")
    fig.colorbar(im1, ax=ax1, label="log10(1 + delta)", fraction=0.046)

    vmax = np.percentile(np.abs(ph), 99)
    im2 = ax2.imshow(
        ph.T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax2.set_xlabel("x [Mpc/h]")
    ax2.set_ylabel("y [Mpc/h]")
    ax2.set_title("peculiar potential phi (wells under overdensities)")
    fig.colorbar(im2, ax=ax2, label="phi (unit-source)", fraction=0.046)

    os.makedirs("outputs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/forces.png", dpi=130)
    print("wrote outputs/forces.png")


if __name__ == "__main__":
    main()
