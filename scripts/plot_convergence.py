"""Figure: the PM forward model vs analytic linear theory (the cross-check).

Three panels, all on matched phases (cosmic variance cancels in the ratios):

  Left:   matched transfer T(k) = P_PM/P_lin at z=0, raw (CIC window) and
          CIC-deconvolved, plus the IC-stage transfer P(x0)/P_lin(z_init). The
          deconvolved transfer -> 1 at the largest scales (absolute growth +
          normalization) and droops at high k as the coarse PM under-resolves
          small scales.
  Middle: the propagator r(k) = <delta_PM delta_lin>/sqrt(P_PM P_lin) -> 1 at low
          k, decohering toward small scales (the standard PM diagnostic).
  Right:  large-scale growth R(a_final)/[D(0)/D(z_init)] vs step count for the
          exact-background leapfrog (converges) and fastpm (step-independent).

Writes outputs/convergence.png. Run: pixi run python scripts/plot_convergence.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import (  # noqa: E402
    BoxConfig,
    Cosmology,
    InitialConditions,
    TimeStepping,
    SimConfig,
)
from mbody import cosmology as C  # noqa: E402
from mbody import fields as F  # noqa: E402
from mbody import ic as IC  # noqa: E402
from mbody import diagnostics as D  # noqa: E402
from mbody import driver  # noqa: E402

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
BOX = BoxConfig(box_size=512.0, n_mesh=128, n_particles=128)
TIME = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10)
NSEED = 4


def main():
    print(
        f"transfer + propagator over {NSEED} seeds (L={BOX.box_size}, "
        f"N={BOX.n_mesh}) ..."
    )
    k = None
    Traw = Tdec = r = Tic = None
    for s in range(NSEED):
        res = driver.run(SimConfig(box=BOX, time=TIME, ic=InitialConditions(seed=s)))
        kk, P_pm, _ = res.power()
        _, P_pm_dec, _ = res.power(deconvolve_cic=True)
        dlin = IC.linear_density(BOX, COSMO, seed=s, z=0.0)
        _, P_lin, _ = F.power_spectrum(dlin, BOX)
        _, P_ic, _ = F.power_spectrum(res.ic_field, BOX, deconvolve_cic=True)
        _, rr, _ = D.cross_correlation(res.final_field, dlin, BOX)
        if k is None:
            k = kk
            Traw, Tdec, Tic, r = (np.zeros_like(kk) for _ in range(4))
        Traw = Traw + P_pm / P_lin
        Tdec = Tdec + P_pm_dec / P_lin
        Tic = Tic + P_ic / C.linear_power(kk, COSMO, z=TIME.z_init)
        r = r + rr
    Traw, Tdec, Tic, r = (a / NSEED for a in (Traw, Tdec, Tic, r))
    knyq = BOX.k_nyquist

    print("growth convergence vs step count ...")
    steps = [2, 3, 5, 10, 20]
    D_lin = {}
    for integ in ("exact", "fastpm"):
        ratios = []
        for nst in steps:
            t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=nst, integrator=integ)
            res = driver.run(SimConfig(box=BOX, time=t), record=True)
            a_arr, R = res.growth_history()
            Dref = D.linear_growth_reference(a_arr, COSMO)
            ratios.append(R[-1] / Dref[-1])
        D_lin[integ] = np.array(ratios)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.6))

    ax0.axhline(1.0, color="k", lw=0.8, ls=":")
    ax0.plot(
        k / knyq, Tic, "^", ms=4, color="C2", alpha=0.7, label="IC: P(x0)/P_lin(z_init)"
    )
    ax0.plot(k / knyq, Traw, "s", ms=3, color="0.6", label="T(k) raw (CIC window)")
    ax0.plot(k / knyq, Tdec, "o", ms=4, color="C3", label="T(k) CIC-deconvolved")
    ax0.set_xlabel("k / k_nyquist")
    ax0.set_ylabel("P_PM / P_lin")
    ax0.set_ylim(0.0, 1.5)
    ax0.set_title("matched transfer -> 1 at large scales")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.2)

    ax1.axhline(1.0, color="k", lw=0.8, ls=":")
    ax1.plot(k / knyq, r, "o-", ms=3, color="C0")
    ax1.axhline(0.99, color="0.7", lw=0.8, ls="--")
    ax1.set_xlabel("k / k_nyquist")
    ax1.set_ylabel("r(k)")
    ax1.set_title("propagator: phases track linear IC, decohere at small scales")
    ax1.grid(True, alpha=0.2)

    target = C.growth_factor(0.0, COSMO) / C.growth_factor(TIME.z_init, COSMO)
    ax2.axhline(1.0, color="k", lw=0.8, ls=":", label="linear growth")
    ax2.plot(steps, D_lin["exact"], "o-", color="C1", label="exact (converges)")
    ax2.plot(steps, D_lin["fastpm"], "s-", color="C3", label="fastpm (flat)")
    ax2.set_xlabel("n_steps")
    ax2.set_ylabel("R(a_final) / [D(0)/D(z_init)]")
    ax2.set_title("growth vs linear theory (target D-ratio = %.2f)" % target)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    fig.suptitle(
        f"M-body convergence cross-check vs linear theory  "
        f"(L={BOX.box_size:.0f}, N={BOX.n_mesh}, fastpm+2LPT, {NSEED} seeds)"
    )
    fig.tight_layout()
    path = OUT / "convergence.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
