"""Redshift-space multi-tracer Fisher: the two f_NL levers, composed.

The final-bow figure. Two local-bias tracers A, B painted from the SAME
redshift-space field combine two handles on the b_phi-f_NL system:

  * the multi-tracer sample-variance cancellation tightens sigma(f_NL) in the
    universality-tied basis (Seljak 2009; Barreira & Krause 2023), and
  * the line-of-sight quadrupole sharply pins f_growth (the RSD growth rate),

and -- the point of this figure -- the two stack. The scientific conclusion is
unchanged from the isotropic case: with b_phi free per tracer only the products
f_NL*b_phi are constrained, so f_NL itself is recovered only under the universality
tie b_phi = 2 delta_c (b1 - 1); neither RSD nor multi-tracer alone breaks that
degeneracy. This is the composition / completeness piece (linear field, a large
cheap box -- box size is free in this toy).

Three panels (native tracer, tied universality basis, f_NL = 0 detection regime):
  1. sigma(f_NL): mono+quad, 1-tracer vs 2-tracer -- the cancellation on f_NL.
  2. sigma(f_growth): 2-tracer, monopole-only vs mono+quad -- the quadrupole pins
     the growth rate (the f_NL,f_growth ellipse collapses along f_growth).
  3. the 2x2 bar summary of sigma(f_NL): {mono, mono+quad} x {1 tracer, 2 tracer}.

Writes outputs/rsd_multitracer.png (gitignored). Run:
    pixi run python scripts/plot_rsd_multitracer.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody import fisher as FI  # noqa: E402
from mbody import bias as B  # noqa: E402
from mbody.config import BoxConfig, Cosmology  # noqa: E402

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
BOX = BoxConfig(box_size=1024.0, n_mesh=48, n_particles=48)
BACKEND = "eh98"
DELTA_C = 1.686
LOS = 0
B1_A, B1_B = 1.5, 2.5
N_A = N_B = 5.0e-4
N_SEED = 8
N_MOCK = 400
PRIORS = {"A": 0.1}
KBINS = np.arange(1.5, 12.0, 1.0)  # in units of k_fundamental
THETA = FI.universality_rsd_fiducial(
    0.0, 1.0, 1.0, B1_A, B1_B, BOX, COSMO, backend=BACKEND
)


def _ellipse(cov2, cx, cy, n_sigma=1.0):
    evals, evecs = np.linalg.eigh(cov2)
    t = np.linspace(0.0, 2.0 * np.pi, 240)
    axes = n_sigma * np.sqrt(np.maximum(evals, 0.0))
    pts = evecs @ (axes[:, None] * np.stack([np.cos(t), np.sin(t)]))
    return cx + pts[0], cy + pts[1]


def forecasts(ells):
    """1- and 2-tracer tied RSD-MT forecasts at a fixed ells set."""
    kb = KBINS * BOX.k_fundamental
    ne, nb = len(ells), len(kb)
    J = np.zeros((3 * ne * nb, 7))
    for s in range(N_SEED):
        J += FI.linear_multitracer_multipole_jacobian(
            BOX, COSMO, THETA, kb, ells=ells, los_axis=LOS, seed=s, backend=BACKEND
        )[0]
    J /= N_SEED
    cov = FI.multitracer_multipole_gaussian_covariance(
        BOX,
        COSMO,
        THETA,
        kb,
        N_A,
        N_B,
        ells=ells,
        los_axis=LOS,
        n_mock=N_MOCK,
        backend=BACKEND,
    )
    tie = FI.universality_rsd_tie_matrix(THETA, BOX, COSMO, backend=BACKEND)
    fc2 = FI.multitracer_forecast(
        J, cov, THETA, FI.PARAM_NAMES_MT_RSD, priors=PRIORS, tie=tie
    )
    # 1-tracer (AA only): rows of the AA block, columns [f_NL,A,f_growth,b1_A,b2_A].
    naa = ne * nb
    JA, covA = J[:naa][:, [0, 1, 2, 3, 4]], cov[:naa, :naa]
    s2 = B.mesh_variance(BOX, COSMO, backend=BACKEND)
    T1 = np.zeros((5, 4))
    T1[0, 0] = T1[1, 1] = T1[2, 2] = T1[3, 3] = 1.0
    T1[4, 1] = -DELTA_C * (B1_A - 1) / (1.0**2 * s2)
    T1[4, 3] = DELTA_C / (1.0 * s2)
    fc1 = FI.FisherForecast(
        JA @ T1,
        covariance=covA,
        fiducial_params={n: THETA[n] for n in ("f_NL", "A", "f_growth", "b1_A")},
        param_names=("f_NL", "A", "f_growth", "b1_A"),
        priors=PRIORS,
    )
    return fc1, fc2


def main():
    fc1_m, fc2_m = forecasts((0,))
    fc1_q, fc2_q = forecasts((0, 2))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # Panel 1: the cancellation -- sigma(f_NL), 1-tracer vs 2-tracer (mono+quad).
    ax = axes[0]
    for fc, color, lab in [(fc1_q, "0.55", "1 tracer"), (fc2_q, "C0", "2 tracers")]:
        m = fc.marginalized_2d("f_NL", "b1_A")
        x, y = _ellipse(m["cov_2d"], 0.0, B1_A)
        ax.plot(
            x,
            y,
            "-",
            color=color,
            lw=2.0,
            label=f"{lab}: sigma(f_NL)={fc.sigma('f_NL'):.0f}",
        )
    ax.plot(0.0, B1_A, "k+", ms=10, mew=1.5)
    ax.set_xlabel("f_NL")
    ax.set_ylabel("b1 (tracer A)")
    ax.set_title(
        "Sample-variance cancellation (mono+quad):\n"
        "a 2nd tracer tightens sigma(f_NL) in the\n"
        "universality-tied basis",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: the quadrupole -- sigma(f_growth), monopole-only vs mono+quad (2 tracer).
    ax = axes[1]
    for fc, color, lab in [
        (fc2_m, "0.55", "monopole only"),
        (fc2_q, "C3", "mono + quad"),
    ]:
        m = fc.marginalized_2d("f_NL", "f_growth")
        x, y = _ellipse(m["cov_2d"], 0.0, 1.0)
        ax.plot(
            x,
            y,
            "-",
            color=color,
            lw=2.0,
            label=f"{lab}: sigma(fg)={fc.sigma('f_growth'):.2f}",
        )
    ax.plot(0.0, 1.0, "k+", ms=10, mew=1.5)
    ax.set_xlabel("f_NL")
    ax.set_ylabel("f_growth")
    ax.set_title(
        "The quadrupole pins f_growth (2 tracers):\n"
        "the (f_NL, f_growth) ellipse collapses\n"
        "along the growth-rate axis",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper right")

    # Panel 3: the 2x2 summary of sigma(f_NL).
    ax = axes[2]
    groups = ["monopole\nonly", "mono +\nquad"]
    s1 = [fc1_m.sigma("f_NL"), fc1_q.sigma("f_NL")]
    s2 = [fc2_m.sigma("f_NL"), fc2_q.sigma("f_NL")]
    xpos = np.arange(2)
    w = 0.36
    ax.bar(xpos - w / 2, s1, w, color="0.55", label="1 tracer")
    ax.bar(xpos + w / 2, s2, w, color="C0", label="2 tracers")
    for x, v in zip(xpos - w / 2, s1):
        ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    for x, v in zip(xpos + w / 2, s2):
        ax.text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(groups)
    ax.set_ylabel("sigma(f_NL)")
    ax.set_title(
        "Both levers stack: the cancellation (1->2\n"
        "tracers) and the quadrupole each tighten\n"
        "sigma(f_NL)",
        fontsize=9,
    )
    ax.legend(fontsize=8)

    fig.suptitle(
        f"M-body redshift-space multi-tracer Fisher (native tracer, linear field, "
        f"L={BOX.box_size:.0f}, f_NL=0, b1=({B1_A},{B1_B})): the cancellation and the "
        f"quadrupole compose",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT / "rsd_multitracer.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")
    print(
        f"  sigma(f_NL): mono 1t {fc1_m.sigma('f_NL'):.0f} 2t {fc2_m.sigma('f_NL'):.0f}"
        f" | mono+quad 1t {fc1_q.sigma('f_NL'):.0f} 2t {fc2_q.sigma('f_NL'):.0f}"
    )
    print(
        f"  sigma(f_growth): 2t mono {fc2_m.sigma('f_growth'):.3f}"
        f" -> mono+quad {fc2_q.sigma('f_growth'):.3f}"
    )


if __name__ == "__main__":
    main()
