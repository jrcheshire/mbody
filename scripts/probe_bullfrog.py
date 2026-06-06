"""BullFrog: the 2LPT-accurate integrator, and when its advantage shows.

BullFrog (Rampf, List & Hahn 2024, arXiv:2409.19049) is a drift-kick-drift
integrator whose single step is 2LPT-accurate, so it should converge to the exact
solution in fewer steps than FastPM (1LPT-per-step) or the exact-background
leapfrog. This probe measures that, and an honest caveat: the advantage is realized
only when the PM force is resolved well enough to carry the second-order mode
coupling BullFrog's kick is calibrated for. The metric is self-convergence -- the
large-scale cross-correlation 1 - r(k) of an n-step run against the SAME
integrator's high-step limit (so it measures each scheme's per-step accuracy, not a
cross-scheme offset), matched-phase and seed-averaged.

  - At n_mesh=64 BullFrog clearly beats FastPM (~3x fewer steps for the same 1-r).
  - At n_mesh=32 the coarse CIC force under-resolves the 2LPT coupling, so BullFrog
    does NOT beat FastPM there -- though it still converges to the same field and
    beats the exact-background leapfrog.

Run: pixi run python scripts/probe_bullfrog.py   (-> outputs/bullfrog_convergence.png)
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology, TimeStepping  # noqa: E402
from mbody import integrate as IG  # noqa: E402
from mbody import diagnostics as D  # noqa: E402
from mbody import painting as PA  # noqa: E402

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
INTEGRATORS = ("bullfrog", "fastpm", "exact")
NSTEPS = (2, 3, 4, 6, 10)
NREF = 24
SEEDS = (0, 1, 2)


def _run(box, integrator, n_steps, seed):
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=n_steps, integrator=integrator)
    x, _ = IG.leapfrog(box, COSMO, t, seed=seed, backend="eh98", integrator=integrator)
    return x


def _self_convergence(box):
    """1 - r(k<0.2) of n-step vs the same integrator's NREF-step limit, seed-mean."""
    out = {ig: [] for ig in INTEGRATORS}
    for ig in INTEGRATORS:
        for n in NSTEPS:
            errs = []
            for s in SEEDS:
                x = _run(box, ig, n, s)
                xr = _run(box, ig, NREF, s)
                k, r, _ = D.cross_correlation(
                    PA.density_contrast(x, box), PA.density_contrast(xr, box), box
                )
                errs.append(1.0 - float(np.mean(r[k < 0.2])))
            out[ig].append(np.mean(errs))
    return out


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, nmesh in zip(axes, (64, 32)):
        box = BoxConfig(box_size=256.0, n_mesh=nmesh, n_particles=nmesh)
        print(f"n_mesh={nmesh}: measuring self-convergence ...")
        conv = _self_convergence(box)
        print("  nsteps  " + "  ".join(f"{ig:>9s}" for ig in INTEGRATORS))
        for j, n in enumerate(NSTEPS):
            print(
                f"  {n:5d}   " + "  ".join(f"{conv[ig][j]:9.5f}" for ig in INTEGRATORS)
            )
        styles = {
            "bullfrog": ("C3", "o-"),
            "fastpm": ("C0", "s-"),
            "exact": ("C1", "^-"),
        }
        for ig in INTEGRATORS:
            c, m = styles[ig]
            ax.semilogy(NSTEPS, np.maximum(conv[ig], 1e-6), m, color=c, label=ig)
        ax.set_xlabel("number of PM steps")
        ax.set_title(
            f"n_mesh={nmesh}"
            + ("  (BullFrog wins)" if nmesh == 64 else "  (coarse force: no advantage)")
        )
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("1 - r(k) vs the high-step limit  (lower = more converged)")
    fig.suptitle(
        "BullFrog convergence: 2LPT-per-step accuracy needs a resolved force "
        "(L=256, z=9->0, 3 seeds)"
    )
    fig.tight_layout()
    path = OUT / "bullfrog_convergence.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
