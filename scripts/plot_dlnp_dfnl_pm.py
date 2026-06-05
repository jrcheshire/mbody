"""Step 5 headline, PM stage: dlnP/df_NL through the full differentiable forward
model.

The pipeline f_NL -> ic.linear_density -> LPT -> PM leapfrog -> CIC -> local-bias
tracer -> band power is differentiable end to end. This figure shows:

Left: dlnP/df_NL of the PM-evolved tracer from reverse-mode mx.grad (points,
seed scatter) overlaid on the matched-phase finite difference (open markers) --
they agree, so autodiff works through the whole unrolled leapfrog. For
reference, the linear-field tracer (dashed) tracks 1/M(k) cleanly; the evolved
tracer's signal survives and stays well above the matter null but is flattened
(nonlinear evolution + CIC + Eulerian bias mix scales -- the real-world
complication behind the b_phi systematic). 1/M(k) is the gray curve.

Right: a slice of the CIC-painted evolved density the forward model produced.

Writes outputs/dlnp_dfnl_pm.png (gitignored). Run:
    pixi run python scripts/plot_dlnp_dfnl_pm.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology, TimeStepping  # noqa: E402
from mbody import fields as F  # noqa: E402
from mbody import ic as IC  # noqa: E402
from mbody import bias as B  # noqa: E402
from mbody import integrate as IN  # noqa: E402
from mbody import painting as PA  # noqa: E402

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
BOX = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
TIME = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5)
B1, B2 = 2.0, 1.0
EPS = 50.0
NSEED = 16


def pm_density(f_NL, seed):
    x, _ = IN.leapfrog(BOX, COSMO, TIME, seed=seed, f_NL=f_NL)
    return PA.density_contrast(x, BOX)


def pm_power(f_NL, seed, kb, b2=B2):
    return F.band_power(B.local_bias_tracer(pm_density(f_NL, seed), B1, b2), BOX, kb)


def lin_power(f_NL, seed, kb, b2=B2):
    d = IC.linear_density(BOX, COSMO, seed=seed, f_NL=f_NL)
    return F.band_power(B.local_bias_tracer(d, B1, b2), BOX, kb)


def grad_dlnp(power_fn, seed, kb, b2=B2):
    P = np.asarray(power_fn(mx.array(0.0), seed, kb, b2), np.float64)
    dP = np.array(
        [
            float(mx.grad(lambda f, i=i: power_fn(f, seed, kb, b2)[i])(mx.array(0.0)))
            for i in range(len(kb))
        ]
    )
    return dP / P


def fd_dlnp(power_fn, seed, kb, b2=B2):
    lp = mx.log(power_fn(mx.array(EPS), seed, kb, b2))
    lm = mx.log(power_fn(mx.array(-EPS), seed, kb, b2))
    return np.asarray((lp - lm) / (2 * EPS), np.float64)


def main():
    kf = BOX.k_fundamental
    kb = np.array([1, 2, 3, 4, 6]) * kf

    print(f"measuring dlnP/df_NL through {TIME.n_steps}-step PM, {NSEED} seeds ...")
    g_pm = np.array([grad_dlnp(pm_power, s, kb) for s in range(NSEED)])
    g_pm0 = np.array([grad_dlnp(pm_power, s, kb, 0.0) for s in range(NSEED)])
    fd_pm = np.array([fd_dlnp(pm_power, s, kb) for s in range(NSEED)])
    g_lin = np.array([grad_dlnp(lin_power, s, kb) for s in range(NSEED)])

    gp, gp_e = g_pm.mean(0), g_pm.std(0) / np.sqrt(NSEED)
    fp = fd_pm.mean(0)
    gp0 = g_pm0.mean(0)
    gl = g_lin.mean(0)

    ref = B.scale_dependent_shape(kb, COSMO)
    ref = ref * (gl[0] / ref[0])  # normalize 1/M(k) to the linear tracer

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax0.plot(kb, ref, "-", color="0.6", label="1/M(k) (Dalal)")
    ax0.plot(kb, gl, "--", color="C1", label="linear tracer (mx.grad)")
    ax0.errorbar(kb, gp, yerr=gp_e, fmt="o", color="C3", label="PM tracer: mx.grad")
    ax0.plot(kb, fp, "x", color="C0", mew=2, label="PM tracer: finite diff")
    ax0.plot(kb, np.abs(gp0), "s", color="0.5", mfc="none", label="PM matter null")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("k  [h/Mpc]")
    ax0.set_ylabel("dlnP/df_NL")
    ax0.set_title("dlnP/df_NL through the full differentiable PM pipeline")
    ax0.legend(fontsize=8)

    slab = np.asarray(pm_density(mx.array(0.0), 0)[:, :, BOX.n_mesh // 2])
    im = ax1.imshow(
        np.log10(1.0 + slab - slab.min()),
        cmap="magma",
        extent=[0, BOX.box_size] * 2,
        origin="lower",
    )
    ax1.set_title(f"evolved density (z={TIME.z_final:.0f}, {TIME.n_steps} PM steps)")
    ax1.set_xlabel("Mpc/h")
    ax1.set_ylabel("Mpc/h")
    fig.colorbar(im, ax=ax1, fraction=0.046, label="log10(1 + delta - min)")

    fig.suptitle(
        f"M-body Step 5 (PM): autodiff dlnP/df_NL end to end  "
        f"(L={BOX.box_size:.0f}, N={BOX.n_mesh}, {TIME.n_steps} steps, {NSEED} seeds)"
    )
    fig.tight_layout()
    path = OUT / "dlnp_dfnl_pm.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
