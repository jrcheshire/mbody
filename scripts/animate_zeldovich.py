"""Animate Zel'dovich structure formation from the initial conditions.

Streams a Lagrangian slab of particles along the Zel'dovich displacement as the
scale factor grows from the initial redshift to today, and stitches the frames
into a gif -- the first "structure forming" movie. No gravity solve yet, just
LPT scaled by the growth factor D(z); the PM steps will give the real nonlinear
version later. This is the prototype of the optional snapshot -> animation
framework that the PM loop will plug into.

Run: pixi run python scripts/animate_zeldovich.py
Saves: outputs/zeldovich.gif
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology  # noqa: E402
from mbody import cosmology as C  # noqa: E402
from mbody import lpt as L  # noqa: E402


def main():
    cosmo = Cosmology()
    box = BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)
    N = box.n_mesh
    thick = 4
    n_frames = 50

    psi = L.zeldovich_displacement(box, cosmo, seed=0)
    scale_factors = np.linspace(1.0 / 50.0, 1.0, n_frames)  # a: z=49 -> z=0

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=0.3, c="k", alpha=0.5, linewidths=0)
    ax.set_xlim(0, box.box_size)
    ax.set_ylim(0, box.box_size)
    ax.set_aspect("equal")
    ax.set_xlabel("x [Mpc/h]")
    ax.set_ylabel("y [Mpc/h]")
    title = ax.set_title("")

    def frame(i):
        a = scale_factors[i]
        z = 1.0 / a - 1.0
        D = C.growth_factor(z, cosmo)
        pos = np.asarray(L.displace(box, psi, D), np.float64).reshape(N, N, N, 3)
        slab = pos[:, :, :thick, :].reshape(-1, 3)
        scat.set_offsets(slab[:, :2])
        title.set_text("z = %.2f    D = %.3f" % (z, D))
        return scat, title

    anim = animation.FuncAnimation(fig, frame, frames=n_frames, blit=False)
    os.makedirs("outputs", exist_ok=True)
    out = "outputs/zeldovich.gif"
    anim.save(out, writer=animation.PillowWriter(fps=12))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
