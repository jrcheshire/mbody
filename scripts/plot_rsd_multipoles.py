"""Redshift-space multipoles of the PM-evolved tracer, vs linear Kaiser.

Runs the differentiable PM forward model with redshift-space distortions enabled
(mbody.driver), measures the monopole P_0 and quadrupole P_2 of the local-bias
tracer (seed-averaged, CIC-window-deconvolved, discrete-shell decoupled), and
compares to linear Kaiser theory:

  * left  -- P_0(k) and |P_2(k)| in redshift space, with the real-space tracer
    P(k) for reference (the monopole is Kaiser-boosted; the quadrupole is the new
    anisotropic signal RSD adds);
  * right -- the ratio P_2/P_0(k) against the constant linear-Kaiser prediction
    (4/3 beta + 4/7 beta^2)/(1 + 2/3 beta + 1/5 beta^2), beta = f/b1. It tracks
    Kaiser at large scales and deviates at high k (nonlinear growth; fingers-of-god
    are absent in a PM, so the small-scale quadrupole is not trustworthy).

Writes outputs/rsd_multipoles.png (gitignored). Run:
    pixi run python scripts/plot_rsd_multipoles.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody import cosmology as C  # noqa: E402
from mbody import run  # noqa: E402
from mbody.config import (  # noqa: E402
    BoxConfig,
    Cosmology,
    InitialConditions,
    RedshiftSpace,
    SimConfig,
    TimeStepping,
    Tracer,
)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
BOX = BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
TIME = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5)
TRACER = Tracer(b1=2.0, b2=1.0)
LOS = 0
NSEED = 6
ELLS = (0, 2)


def measure(seed):
    cfg = SimConfig(
        cosmology=COSMO,
        box=BOX,
        time=TIME,
        ic=InitialConditions(f_NL=0.0, seed=seed, kind="gaussian"),
        tracer=TRACER,
        rsd=RedshiftSpace(enabled=True, los_axis=LOS, f_growth=1.0),
    )
    res = run(cfg, backend="eh98")
    # tracer in redshift space: apply the local bias to the (decoupled) field
    k, P, _ = res.power_multipoles(ells=ELLS, kmax=0.4 * BOX.k_nyquist)
    # real-space tracer monopole for reference
    kr, Pr, _ = res.power(deconvolve_cic=True, kmax=0.4 * BOX.k_nyquist)
    return k, P, Pr


def main():
    acc = None
    Pr_acc = None
    for s in range(NSEED):
        k, P, Pr = measure(s)
        acc = {el: P[el] if acc is None else acc[el] + P[el] for el in P}
        Pr_acc = Pr if Pr_acc is None else Pr_acc + Pr
    P = {el: acc[el] / NSEED for el in acc}
    Pr = Pr_acc / NSEED

    f0 = C.growth_rate(0.0, COSMO)
    beta = f0 / TRACER.b1
    kaiser_ratio = (4 / 3 * beta + 4 / 7 * beta**2) / (
        1 + 2 / 3 * beta + 1 / 5 * beta**2
    )
    print(f"f(z=0)={f0:.4f}, beta=f/b1={beta:.4f}, Kaiser P2/P0={kaiser_ratio:+.4f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    axL.loglog(k, P[0], "o-", color="C0", label="redshift P_0")
    axL.loglog(k, np.abs(P[2]), "s-", color="C3", label="redshift |P_2|")
    axL.loglog(k, Pr, "x--", color="0.5", label="real-space P (tracer-free)")
    axL.set_xlabel("k [h/Mpc]")
    axL.set_ylabel("P(k) [(Mpc/h)^3]")
    axL.set_title("Redshift-space multipoles (PM, seed-averaged)")
    axL.legend(fontsize=8)

    axR.semilogx(k, P[2] / P[0], "o-", color="C3", label="measured P_2/P_0")
    axR.axhline(
        kaiser_ratio, color="k", ls="--", label=f"linear Kaiser {kaiser_ratio:.3f}"
    )
    axR.axhline(0.0, color="0.8", lw=0.8)
    axR.set_xlabel("k [h/Mpc]")
    axR.set_ylabel("P_2 / P_0")
    axR.set_title("Quadrupole/monopole vs Kaiser (b1=%.1f)" % TRACER.b1)
    axR.legend(fontsize=8)

    fig.suptitle(
        f"M-body redshift-space distortions: L={BOX.box_size:.0f} Mpc/h, "
        f"N={BOX.n_mesh}, {TIME.n_steps} PM steps, {NSEED} seeds, los={LOS}\n"
        "Kaiser at large scales; high-k deviation is nonlinear (no fingers-of-god "
        "in a PM)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = OUT / "rsd_multipoles.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
