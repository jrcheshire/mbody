"""The autodiff-Fisher headline figure: forecast ellipses over {f_NL, b1, b2, A}.

Two rows on the same box and seeds: the linear-field forecast (top, blue) and the
PM-evolved forecast (bottom, red). The Jacobian d ln P_b / d theta is built by
reverse-mode autodiff of the local-bias tracer's band power -- plain mx.grad for
the linear field; for the PM field, the reversible-leapfrog adjoint for the
upstream (f_NL, A) columns and a cheap fixed-field mx.grad for the downstream
(b1, b2) columns -- seed-averaged, and fed to a diagonal-Gaussian-covariance
Fisher (mbody.fisher). Each row shows 1- and 2-sigma error ellipses for the
science-relevant pairs:

  * (f_NL, b1): the SPHEREx-relevant f_NL-bias degeneracy;
  * (f_NL, b2): f_NL vs the quadratic bias that carries the scale-dependent
    signal;
  * (A, b1): the strongly anti-correlated amplitude direction (both scale ln P
    by a constant), the near-degenerate "cigar" broken only by f_NL / b2 shape.

Marginalized ellipses (solid) are overlaid on conditional ones (dashed, all
other parameters held fixed) to show how marginalizing inflates each constraint.
A weak prior sigma(A) keeps the otherwise near-singular amplitude direction
well-posed. The PM f_NL constraint is broader than the linear one -- nonlinear
evolution and CIC flatten dlnP/df_NL -- which the printed sigmas quantify.

Writes outputs/fisher.png (gitignored). Run:
    pixi run python scripts/plot_fisher.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology, TimeStepping, Tracer  # noqa: E402
from mbody import fisher as FI  # noqa: E402

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
BOX = BoxConfig(box_size=512.0, n_mesh=48, n_particles=48)
TIME = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5)
TRACER = Tracer()
THETA_FID = {"f_NL": 0.0, "b1": TRACER.b1, "b2": TRACER.b2, "A": TRACER.A}
PRIORS = {"A": 0.1}  # a 10% external amplitude (sigma8-like) prior
NSEED = 8
KBINS = np.array([1, 2, 3, 4, 6, 8])
PAIRS = [("f_NL", "b1"), ("f_NL", "b2"), ("A", "b1")]


def _ellipse(cov2, cx, cy, n_sigma):
    """Points tracing the n_sigma error ellipse of a 2x2 covariance at (cx, cy)."""
    evals, evecs = np.linalg.eigh(cov2)
    t = np.linspace(0.0, 2.0 * np.pi, 200)
    axes = n_sigma * np.sqrt(np.maximum(evals, 0.0))
    pts = evecs @ (axes[:, None] * np.stack([np.cos(t), np.sin(t)]))
    return cx + pts[0], cy + pts[1]


def _forecast(jacobian_fn, label):
    kb = KBINS * BOX.k_fundamental
    print(f"building {label} Jacobian over {NSEED} seeds ...")
    J = np.zeros((len(kb), len(FI.PARAM_NAMES)))
    for s in range(NSEED):
        J += jacobian_fn(kb, s)
    J /= NSEED
    var = FI.band_power_log_variance(BOX, kb)
    return FI.FisherForecast(J, var, THETA_FID, priors=PRIORS)


def linear_forecast():
    def jac(kb, s):
        return FI.linear_logP_jacobian(BOX, COSMO, THETA_FID, kb, seed=s)[0]

    return _forecast(jac, "linear-field")


def pm_forecast():
    def jac(kb, s):
        return FI.pm_logP_jacobian(BOX, COSMO, TIME, THETA_FID, kb, seed=s)[0]

    return _forecast(jac, "PM-evolved")


def draw_panels(axes, fc, color):
    for ax, (pi, pj) in zip(axes, PAIRS):
        cx, cy = THETA_FID[pi], THETA_FID[pj]
        marg = fc.marginalized_2d(pi, pj)
        cond = fc.conditional_2d(pi, pj)
        for ns, lw in ((1.0, 2.0), (2.0, 1.0)):
            x, y = _ellipse(marg["cov_2d"], cx, cy, ns)
            ax.plot(x, y, "-", color=color, lw=lw)
            xc, yc = _ellipse(cond["cov_2d"], cx, cy, ns)
            ax.plot(xc, yc, "--", color="0.5", lw=lw)
        ax.plot(cx, cy, "+", color="k", ms=10, mew=1.5)
        ax.set_xlabel(pi)
        ax.set_ylabel(pj)
        ax.set_title(f"rho = {marg['rho']:+.2f}")


def _row_label(fc, name, color):
    sig = ", ".join(f"s({p})={fc.sigma(p):.3g}" for p in FI.PARAM_NAMES)
    return f"{name}:  {sig}"


def main():
    fc_lin = linear_forecast()
    print(fc_lin.summary("linear field"))
    fc_pm = pm_forecast()
    print(fc_pm.summary("PM-evolved"))

    fig, axes = plt.subplots(2, len(PAIRS), figsize=(4.2 * len(PAIRS), 7.6))
    draw_panels(axes[0], fc_lin, color="C0")
    draw_panels(axes[1], fc_pm, color="C3")
    axes[0][0].set_ylabel("b1\n(linear field)")
    axes[1][0].set_ylabel("b1\n(PM-evolved)")

    fig.suptitle(
        "M-body autodiff Fisher over {f_NL, b1, b2, A}: linear (blue) vs "
        f"PM-evolved (red)\n(L={BOX.box_size:.0f}, N={BOX.n_mesh}, "
        f"{TIME.n_steps} PM steps, prior sigma(A)={PRIORS['A']}, {NSEED} seeds;  "
        "solid = marginalized, dashed = conditional)\n"
        + _row_label(fc_lin, "linear", "C0")
        + "\n"
        + _row_label(fc_pm, "PM", "C3"),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = OUT / "fisher.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
