"""Redshift-space Fisher: does the quadrupole help break the f_NL degeneracy?

Extends the autodiff Fisher (mbody.fisher) to redshift space over
theta = {f_NL, b1, b2, A, f_growth}, with the multipole band powers P_ell(k) as
the data vector and a Gaussian mock block covariance (cross-multipole correlated).
It contrasts two forecasts on the same box/seeds:

  * monopole only  -- the redshift-space P_0, which carries growth info only
    weakly (through the angle-averaged Kaiser boost);
  * mono + quad    -- adds P_2, the line-of-sight growth-rate signal.

The ellipses (linear field) are drawn mono (dashed grey) vs mono+quad (solid
colour) for the science pairs. The honest takeaway, printed as sigmas for both
the linear and PM forward models: the quadrupole sharply constrains the growth
amplitude f_growth and partially recovers sigma(f_NL) lost to the bias/amplitude
degeneracy -- but it does NOT break the fundamental b_phi*f_NL degeneracy. Neither
does multi-tracer alone: both constrain the product f_NL*b_phi, and pinning f_NL
needs a b_phi(b1) relation (universality), which multi-tracer relaxes + sharpens
(see docs/multitracer.md, pixi run multitracer).

Writes outputs/fisher_rsd.png (gitignored). Run:
    pixi run python scripts/plot_fisher_rsd.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody import fisher as FI  # noqa: E402
from mbody.config import BoxConfig, Cosmology, TimeStepping, Tracer  # noqa: E402

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
BOX = BoxConfig(box_size=512.0, n_mesh=48, n_particles=48)
TIME = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5)
TRACER = Tracer(b1=2.0, b2=1.0)
THETA = {"f_NL": 0.0, "b1": TRACER.b1, "b2": TRACER.b2, "A": TRACER.A, "f_growth": 1.0}
PRIORS = {"A": 0.1}
LOS = 0
NSEED_LIN = 8
NSEED_PM = 3
N_MOCK = 500
KBINS = np.arange(1.5, 18.0, 1.0)  # in units of k_fundamental
PAIRS = [("f_NL", "f_growth"), ("f_NL", "b1"), ("b1", "f_growth")]


def _ellipse(cov2, cx, cy, n_sigma):
    evals, evecs = np.linalg.eigh(cov2)
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    axes = n_sigma * np.sqrt(np.maximum(evals, 0.0))
    pts = evecs @ (axes[:, None] * np.stack([np.cos(t), np.sin(t)]))
    return cx + pts[0], cy + pts[1]


def linear_forecast(ells):
    kb = KBINS * BOX.k_fundamental
    J = np.zeros((len(kb) * len(ells), len(FI.PARAM_NAMES_RSD)))
    for s in range(NSEED_LIN):
        J += FI.linear_multipole_jacobian(
            BOX, COSMO, THETA, kb, ells=ells, los_axis=LOS, seed=s, backend="eh98"
        )[0]
    J /= NSEED_LIN
    cov = FI.multipole_gaussian_covariance(
        BOX, COSMO, THETA, kb, ells=ells, los_axis=LOS, n_mock=N_MOCK, backend="eh98"
    )
    return FI.FisherForecast(
        J,
        covariance=cov,
        fiducial_params=THETA,
        param_names=FI.PARAM_NAMES_RSD,
        priors=PRIORS,
    )


def pm_sigmas(ells):
    kb = KBINS * BOX.k_fundamental
    J = np.zeros((len(kb) * len(ells), len(FI.PARAM_NAMES_RSD)))
    for s in range(NSEED_PM):
        J += FI.pm_multipole_jacobian(
            BOX,
            COSMO,
            TIME,
            THETA,
            kb,
            ells=ells,
            los_axis=LOS,
            seed=s,
            backend="eh98",
            integrator="fastpm",
        )[0]
    J /= NSEED_PM
    cov = FI.multipole_gaussian_covariance(
        BOX, COSMO, THETA, kb, ells=ells, los_axis=LOS, n_mock=N_MOCK, backend="eh98"
    )
    fc = FI.FisherForecast(
        J,
        covariance=cov,
        fiducial_params=THETA,
        param_names=FI.PARAM_NAMES_RSD,
        priors=PRIORS,
    )
    return fc.sigma("f_NL"), fc.sigma("f_growth")


def draw(axes, fc, color, label, dashed=False):
    style = "--" if dashed else "-"
    for ax, (pi, pj) in zip(axes, PAIRS):
        cx, cy = THETA[pi], THETA[pj]
        marg = fc.marginalized_2d(pi, pj)
        x, y = _ellipse(marg["cov_2d"], cx, cy, 1.0)
        ax.plot(x, y, style, color=color, lw=2.0, label=label)
        ax.plot(cx, cy, "+", color="k", ms=9, mew=1.4)
        ax.set_xlabel(pi)
        ax.set_ylabel(pj)


def main():
    fc_mono = linear_forecast((0,))
    fc_quad = linear_forecast((0, 2))
    print(fc_mono.summary("linear, monopole only"))
    print(fc_quad.summary("linear, mono+quad"))

    print("\nPM-evolved forecast (the headline):")
    s_fnl_m, s_fg_m = pm_sigmas((0,))
    s_fnl_q, s_fg_q = pm_sigmas((0, 2))
    print(f"  monopole only: sigma(f_NL)={s_fnl_m:8.1f}  sigma(f_growth)={s_fg_m:.3f}")
    print(f"  mono + quad  : sigma(f_NL)={s_fnl_q:8.1f}  sigma(f_growth)={s_fg_q:.3f}")
    print(f"  f_NL recovery factor from the quadrupole: {s_fnl_m / s_fnl_q:.2f}x")

    fig, axes = plt.subplots(1, len(PAIRS), figsize=(4.2 * len(PAIRS), 4.0))
    draw(axes, fc_mono, "0.5", "monopole only", dashed=True)
    draw(axes, fc_quad, "C0", "mono + quad")
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "M-body redshift-space Fisher (linear field): monopole only (grey dashed) "
        "vs mono+quad (blue)\n"
        f"L={BOX.box_size:.0f}, N={BOX.n_mesh}, los={LOS}, prior sigma(A)={PRIORS['A']}; "
        f"linear sigma(f_NL) {fc_mono.sigma('f_NL'):.0f} -> {fc_quad.sigma('f_NL'):.0f}, "
        f"sigma(f_growth) {fc_mono.sigma('f_growth'):.2f} -> {fc_quad.sigma('f_growth'):.2f}",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = OUT / "fisher_rsd.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
