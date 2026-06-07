"""Multi-tracer Fisher: does a second tracer break the b_phi-f_NL degeneracy?

The capstone figure for the differentiable forward model. Two tracers A, B with
different bias, painted from the SAME field, sample the same modes -- so the
cross spectrum and the relative bias carry f_NL information that cosmic variance
cannot wash out (Seljak 2009; Barreira & Krause 2023). Uses the EXPLICIT-b_phi
tracer (b_phi a free k^-2 scale-dependent-bias parameter) at a non-zero fiducial
f_NL, where the b_phi-f_NL product degeneracy is real (it is invisible at f_NL=0).

Three panels (linear field, a large cheap box):
  1. FREE (b_phi marginalized): the (f_NL, b_phi_A) constraint is a degenerate
     streak along the product f_NL*b_phi -- you measure the product, not f_NL.
     A second tracer tightens the product (the streak shrinks) but stays degenerate
     in f_NL: pinning f_NL needs a b_phi relation (Barreira & Krause).
  2. TIED (universality b_phi=2 delta_c(b1-1)): f_NL is recovered, and 2 tracers
     are much tighter than 1 -- the |b1_A-b1_B| differential-bias gain.
  3. The gain vs |b1_A-b1_B|: 2-tracer sigma(f_NL) (tied) falls as the two biases
     separate; it meets the 1-tracer value when the tracers coincide.

Writes outputs/multitracer_fisher.png (gitignored). Run:
    pixi run python scripts/plot_multitracer_fisher.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody import fisher as FI  # noqa: E402
from mbody.config import BoxConfig, Cosmology  # noqa: E402

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
BOX = BoxConfig(box_size=1024.0, n_mesh=48, n_particles=48)
BACKEND = "eh98"
DELTA_C = 1.686
FNL = 100.0
B1_A = 1.5
N_A = N_B = 5.0e-4
N_SEED = 8
N_MOCK = 400
PRIORS = {"A": 0.1}
# A weak b_phi prior only to make the FREE degenerate streak finite for drawing
# (it does not constrain the degenerate direction; panel 1 annotation says so).
WEAK_BPHI = 5.0
KBINS = np.array([1, 2, 3, 4, 6, 8, 12, 16])


def _ellipse(cov2, cx, cy, n_sigma=1.0):
    evals, evecs = np.linalg.eigh(cov2)
    t = np.linspace(0.0, 2.0 * np.pi, 240)
    axes = n_sigma * np.sqrt(np.maximum(evals, 0.0))
    pts = evecs @ (axes[:, None] * np.stack([np.cos(t), np.sin(t)]))
    return cx + pts[0], cy + pts[1]


def forecasts(b1_B, priors_free=None):
    """Build the 1- and 2-tracer free/tied forecasts at fiducial b1_B."""
    theta = FI.universality_bphi_fiducial(FNL, 1.0, B1_A, b1_B, delta_c=DELTA_C)
    kb = KBINS * BOX.k_fundamental
    nb = len(kb)
    J = np.zeros((3 * nb, 6))
    for s in range(N_SEED):
        J += FI.linear_multitracer_bphi_jacobian(
            BOX, COSMO, theta, kb, seed=s, backend=BACKEND
        )[0]
    J /= N_SEED
    cov = FI.multitracer_bphi_gaussian_covariance(
        BOX, COSMO, theta, kb, N_A, N_B, n_mock=N_MOCK, backend=BACKEND
    )
    tie = FI.universality_bphi_tie_matrix(theta, delta_c=DELTA_C)
    pf = dict(PRIORS, **(priors_free or {}))
    fc2_free = FI.multitracer_forecast(J, cov, theta, FI.PARAM_NAMES_MT_BPHI, priors=pf)
    fc2_tied = FI.multitracer_forecast(
        J, cov, theta, FI.PARAM_NAMES_MT_BPHI, priors=PRIORS, tie=tie
    )
    # 1-tracer (AA only)
    JA, covA = J[:nb][:, [0, 1, 2, 3]], cov[:nb, :nb]
    n4 = ("f_NL", "A", "b1_A", "bphi_A")
    pf1 = {k: pf[k] for k in pf if k in n4}
    fc1_free = FI.FisherForecast(
        JA,
        covariance=covA,
        fiducial_params={k: theta[k] for k in n4},
        param_names=n4,
        priors=pf1,
    )
    T1 = np.zeros((4, 3))
    T1[0, 0] = T1[1, 1] = T1[2, 2] = 1.0
    T1[3, 2] = 2 * DELTA_C
    fc1_tied = FI.FisherForecast(
        JA @ T1,
        covariance=covA,
        fiducial_params={k: theta[k] for k in ("f_NL", "A", "b1_A")},
        param_names=("f_NL", "A", "b1_A"),
        priors=PRIORS,
    )
    return theta, fc1_free, fc2_free, fc1_tied, fc2_tied


def main():
    bphi_prior = {"bphi_A": WEAK_BPHI, "bphi_B": WEAK_BPHI}
    theta, fc1_free, fc2_free, fc1_tied, fc2_tied = forecasts(
        2.5, priors_free=bphi_prior
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    # Panel 1: FREE -- the (f_NL, b_phi_A) degenerate streak.
    ax = axes[0]
    for fc, color, lab in [
        (fc1_free, "0.55", "1 tracer"),
        (fc2_free, "C3", "2 tracers"),
    ]:
        m = fc.marginalized_2d("f_NL", "bphi_A")
        x, y = _ellipse(m["cov_2d"], FNL, theta["bphi_A"], 1.0)
        ax.plot(x, y, "-", color=color, lw=2.0, label=lab)
    ax.plot(FNL, theta["bphi_A"], "k+", ms=10, mew=1.5)
    ax.set_xlabel("f_NL")
    ax.set_ylabel("b_phi (tracer A)")
    ax.set_title(
        "b_phi marginalized: a degenerate streak along\n"
        "the product f_NL*b_phi. A 2nd tracer tightens the\n"
        "product but still needs a b_phi relation for f_NL",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper right")

    # Panel 2: TIED -- f_NL recovered, multi-tracer tighter.
    ax = axes[1]
    for fc, color, lab in [
        (fc1_tied, "0.55", "1 tracer"),
        (fc2_tied, "C0", "2 tracers"),
    ]:
        m = fc.marginalized_2d("f_NL", "b1_A")
        x, y = _ellipse(m["cov_2d"], FNL, B1_A, 1.0)
        ax.plot(
            x,
            y,
            "-",
            color=color,
            lw=2.0,
            label=f"{lab}: sigma(f_NL)={fc.sigma('f_NL'):.0f}",
        )
    ax.plot(FNL, B1_A, "k+", ms=10, mew=1.5)
    ax.set_xlabel("f_NL")
    ax.set_ylabel("b1 (tracer A)")
    ax.set_title(
        "Universality TIED (b_phi=2 delta_c(b1-1)):\nf_NL recovered; "
        "2 tracers break the\ndegeneracy via the differential bias",
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="upper right")

    # Panel 3: the |b1_A - b1_B| scaling of the multi-tracer gain (tied).
    ax = axes[2]
    b1_Bs = np.array([1.55, 1.8, 2.1, 2.5, 3.0])
    s1, s2 = [], []
    for b1_B in b1_Bs:
        _, _, _, f1t, f2t = forecasts(b1_B)
        s1.append(f1t.sigma("f_NL"))
        s2.append(f2t.sigma("f_NL"))
    sep = b1_Bs - B1_A
    ax.plot(sep, s2, "o-", color="C0", lw=2.0, label="2 tracers (tied)")
    ax.plot(sep, s1, "s--", color="0.55", lw=1.6, label="1 tracer (tied)")
    ax.set_xlabel("|b1_A - b1_B|")
    ax.set_ylabel("sigma(f_NL)")
    ax.set_title(
        "Multi-tracer gain grows with the bias\nseparation " "(power ~ |b1_A - b1_B|)",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    fig.suptitle(
        f"M-body multi-tracer Fisher (explicit b_phi, linear field, L={BOX.box_size:.0f}, "
        f"f_NL={FNL:.0f}, b1_A={B1_A}): the b_phi-f_NL degeneracy and how a second tracer addresses it",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT / "multitracer_fisher.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")
    print(
        f"  tied sigma(f_NL): 1-tracer {fc1_tied.sigma('f_NL'):.1f}  "
        f"2-tracer {fc2_tied.sigma('f_NL'):.1f}  gain {fc1_tied.sigma('f_NL')/fc2_tied.sigma('f_NL'):.2f}x"
    )


if __name__ == "__main__":
    main()
