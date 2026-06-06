"""The Step 5 headline figure (linear stage): autodiff dlnP/df_NL and the Dalal
scale-dependent bias.

Left: dlnP/df_NL of a local-bias tracer of the linear f_NL field, from
reverse-mode mx.grad (points, with seed scatter), overlaid on the matched-phase
finite difference (open markers, the ground truth) and the ABSOLUTE first-
principles prediction 4 b2 sigma^2/(b1 M(k)) (bin-averaged, solid curve -- NO free
normalization). The faint dashed curve is the same 1/M(k) shape normalized at the
largest scale, the older shape-only overlay. The matter field's response (b2 = 0)
sits at ~0 -- the null that says the scale-dependent bias is a *tracer* effect.

Right: the same physics as a power split -- the tracer auto-power P_h(k) at
f_NL = -F, 0, +F (seed-averaged); local f_NL pushes large-scale power apart as
1/k^2.

Writes outputs/dlnp_dfnl_linear.png (gitignored). Run:
    pixi run python scripts/plot_dlnp_dfnl.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology  # noqa: E402
from mbody import fields as F  # noqa: E402
from mbody import ic as IC  # noqa: E402
from mbody import bias as B  # noqa: E402

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
BOX = BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
B1, B2 = 2.0, 1.0
EPS = 50.0
F_SPLIT = 300.0
NSEED = 20


def tracer_power(f_NL, seed, kb, b2=B2):
    delta = IC.linear_density(BOX, COSMO, seed=seed, f_NL=f_NL)
    return F.band_power(B.local_bias_tracer(delta, B1, b2), BOX, kb)


def grad_dlnp(seed, kb, b2=B2):
    P = np.asarray(tracer_power(mx.array(0.0), seed, kb, b2), np.float64)
    dP = np.array(
        [
            float(
                mx.grad(lambda f, i=i: tracer_power(f, seed, kb, b2)[i])(mx.array(0.0))
            )
            for i in range(len(kb))
        ]
    )
    return dP / P


def fd_dlnp(seed, kb, b2=B2):
    lp = mx.log(tracer_power(mx.array(EPS), seed, kb, b2))
    lm = mx.log(tracer_power(mx.array(-EPS), seed, kb, b2))
    return np.asarray((lp - lm) / (2 * EPS), np.float64)


def main():
    kf = BOX.k_fundamental
    kb = np.array([1, 2, 3, 4, 6, 8, 12, 16]) * kf

    print(f"measuring dlnP/df_NL over {NSEED} seeds ...")
    grad_t = np.array([grad_dlnp(s, kb) for s in range(NSEED)])
    grad_m = np.array([grad_dlnp(s, kb, b2=0.0) for s in range(NSEED)])
    fd_t = np.array([fd_dlnp(s, kb) for s in range(NSEED)])
    gt, gt_e = grad_t.mean(0), grad_t.std(0) / np.sqrt(NSEED)
    gm = grad_m.mean(0)
    ft = fd_t.mean(0)

    pred_abs = B.scale_dependent_bias_response_binned(BOX, COSMO, kb, B1, B2)
    ref_shape = B.scale_dependent_shape(kb, COSMO)
    ref_shape = ref_shape * (gt[0] / ref_shape[0])  # shape-only (largest-scale norm)

    # Right panel: the power split.
    pk = {f: np.zeros(len(kb)) for f in (-F_SPLIT, 0.0, F_SPLIT)}
    for s in range(NSEED):
        for f in pk:
            pk[f] += np.asarray(tracer_power(mx.array(f), s, kb), np.float64)
    for f in pk:
        pk[f] /= NSEED

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax0.plot(kb, pred_abs, "-", color="0.3", label="absolute: 4 b2 sigma^2/(b1 M(k))")
    ax0.plot(kb, ref_shape, "--", color="0.7", label="1/M(k) shape (normalized)")
    ax0.errorbar(kb, gt, yerr=gt_e, fmt="o", color="C3", label="tracer: mx.grad")
    ax0.plot(kb, ft, "x", color="C0", mew=2, label="tracer: finite diff")
    ax0.plot(
        kb, np.abs(gm), "s", color="0.5", mfc="none", label="matter null |dlnP/df|"
    )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("k  [h/Mpc]")
    ax0.set_ylabel("dlnP/df_NL")
    ax0.set_title("autodiff dlnP/df_NL vs absolute scale-dependent bias")
    ax0.legend(fontsize=8)

    for f, c in ((-F_SPLIT, "C0"), (0.0, "k"), (F_SPLIT, "C3")):
        ax1.plot(kb, pk[f], "o-", color=c, label=f"f_NL = {f:+.0f}")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("k  [h/Mpc]")
    ax1.set_ylabel("P_h(k)  [(Mpc/h)^3]")
    ax1.set_title("tracer power split: local f_NL pushes large scales apart")
    ax1.legend(fontsize=8)

    fig.suptitle(
        f"M-body Step 5 (linear): biased-tracer dlnP/df_NL  "
        f"(L={BOX.box_size:.0f}, N={BOX.n_mesh}, b1={B1}, b2={B2}, {NSEED} seeds)"
    )
    fig.tight_layout()
    path = OUT / "dlnp_dfnl_linear.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
