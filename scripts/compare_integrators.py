"""The payoff of FastPM + 2LPT: integrator accuracy and IC skewness.

Left panel: the large-scale linear growth D(z=0)/D(z_init) recovered by the
exact-background vs FastPM kick/drift kernels, as a function of step count. A
single linear mode (whose geometric acceleration equals its displacement) is
integrated with each scheme's coefficients; FastPM is exact at every step count,
while the exact-background leapfrog carries a step-count growth deficit (the ~2%
motivation for FastPM, larger at very low step count).

Right panel: the one-point PDF of the initial density for Zel'dovich (1LPT) vs
2LPT initial conditions at z_init, annotated with the skewness -- 2LPT sources
the gravitational (positive) skewness the linear/Zel'dovich field lacks.

Run: pixi run python scripts/compare_integrators.py
Saves: outputs/compare_integrators.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology, TimeStepping  # noqa: E402
from mbody import cosmology as C  # noqa: E402
from mbody import diagnostics as D  # noqa: E402
from mbody import integrate as IG  # noqa: E402
from mbody import painting as PA  # noqa: E402


def linear_mode_growth(integrator, n_steps, cosmo, z_init=9.0):
    """D(z=0)/D(z_init) recovered for a single linear mode by `integrator`.

    The mode's geometric acceleration equals its displacement amplitude (both
    track D(a)), so the KDK coefficients alone set the growth.
    """
    a = IG.a_grid(TimeStepping(z_init=z_init, z_final=0.0, n_steps=n_steps))
    Di = C.growth_factor(z_init, cosmo)
    x = Di
    p = a[0] ** 2 * float(C.E(z_init, cosmo)) * Di * C.growth_rate(z_init, cosmo)
    for i in range(len(a) - 1):
        a0, a1 = float(a[i]), float(a[i + 1])
        a_c = 0.5 * (a0 + a1)
        if integrator == "exact":
            k1 = IG.kick_factor(a0, a_c, cosmo)
            dr = IG.drift_factor(a0, a1, cosmo)
            k2 = IG.kick_factor(a_c, a1, cosmo)
        else:
            k1 = IG.fastpm_kick_factor(a0, a_c, a0, cosmo)
            dr = IG.fastpm_drift_factor(a0, a1, a_c, cosmo)
            k2 = IG.fastpm_kick_factor(a_c, a1, a1, cosmo)
        p = p + k1 * x
        x = x + dr * p
        p = p + k2 * x
    return x / Di


def main():
    cosmo = Cosmology()
    target = C.growth_factor(0.0, cosmo) / C.growth_factor(9.0, cosmo)
    nsteps = [2, 3, 4, 6, 8, 12, 20, 40]
    err = {
        ig: [abs(linear_mode_growth(ig, n, cosmo) / target - 1.0) * 100 for n in nsteps]
        for ig in ("exact", "fastpm")
    }

    box = BoxConfig(box_size=256.0, n_mesh=64, n_particles=64)
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10)
    x1, _ = IG.initial_state(box, cosmo, t, seed=0, backend="eh98", lpt_order=1)
    x2, _ = IG.initial_state(box, cosmo, t, seed=0, backend="eh98", lpt_order=2)
    d1, d2 = PA.density_contrast(x1, box), PA.density_contrast(x2, box)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.semilogy(nsteps, np.maximum(err["exact"], 1e-4), "o-", label="exact-background")
    ax1.semilogy(nsteps, np.maximum(err["fastpm"], 1e-4), "s-", label="FastPM")
    ax1.set_xlabel("number of PM steps")
    ax1.set_ylabel(r"linear-growth error  $|D/D_{\rm lin} - 1|$  [%]")
    ax1.set_title("FastPM is exact at any step count")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.2)

    for d, label, color in ((d1, "Zel'dovich (1LPT)", "C0"), (d2, "2LPT", "C3")):
        c, dens, _, sk = D.one_point_pdf(d)
        ax2.step(
            c, dens, where="mid", color=color, label="%s (skew %+.3f)" % (label, sk)
        )
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$\delta / \sigma$")
    ax2.set_ylabel("PDF")
    ax2.set_title("2LPT sources the gravitational skewness")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.2)

    os.makedirs("outputs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/compare_integrators.png", dpi=130)
    print("wrote outputs/compare_integrators.png")
    print(
        "  growth error at 2 steps:  exact=%.2f%%  fastpm=%.4f%%"
        % (err["exact"][0], err["fastpm"][0])
    )
    print("  IC skewness:  ZA=%+.3f  2LPT=%+.3f" % (D.skewness(d1), D.skewness(d2)))


if __name__ == "__main__":
    main()
