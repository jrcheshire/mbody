"""Animate the PM leapfrog: real gravitational structure formation.

This is the Zel'dovich animation's grown-up sibling. Instead of streaming
particles along a fixed LPT displacement, it runs the full particle-mesh
leapfrog -- solving for gravity at every step -- and films the result through
the integrator's `snapshot` callback. Watch the Zel'dovich sheet collapse into a
sharp cosmic web: filaments thin, knots brighten, voids empty. That sharpening
(relative to the LPT version, which just smears) is gravity doing its work.

The snapshot callback is the optional, off-the-autodiff-path diagnostics seam
built into mbody.integrate: it hands each step's (step, a, positions, momenta)
to whatever consumer you like -- here, a slab projection accumulated into frames.

Run: pixi run python scripts/animate_pm.py
Saves: outputs/structure_formation.gif
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology, TimeStepping  # noqa: E402
from mbody import integrate as IG  # noqa: E402


def main():
    cosmo = Cosmology()
    box = BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=80)
    N = box.n_mesh
    thick = 4  # Lagrangian slab thickness in cells (track the same sheet)

    frames = []

    def snapshot(step, a, x, p):
        # Off the AD path: eval to numpy and reduce to a slab projection now.
        xn = np.asarray(x, np.float64).reshape(N, N, N, 3)
        slab = xn[:, :, :thick, :].reshape(-1, 3)
        frames.append((a, slab[:, :2].copy()))

    IG.leapfrog(box, cosmo, time, seed=0, snapshot=snapshot)

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=0.3, c="k", alpha=0.5, linewidths=0)
    ax.set_xlim(0, box.box_size)
    ax.set_ylim(0, box.box_size)
    ax.set_aspect("equal")
    ax.set_xlabel("x [Mpc/h]")
    ax.set_ylabel("y [Mpc/h]")
    title = ax.set_title("")

    def draw(i):
        a, xy = frames[i]
        scat.set_offsets(xy)
        title.set_text("PM leapfrog   z = %.2f   a = %.3f" % (1.0 / a - 1.0, a))
        return scat, title

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    os.makedirs("outputs", exist_ok=True)
    out = "outputs/structure_formation.gif"
    anim.save(out, writer=animation.PillowWriter(fps=15))
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
