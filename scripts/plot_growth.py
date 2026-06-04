"""Validation figure: does the PM leapfrog grow structure like linear theory?

Two checks in one figure. Left: the measured power spectrum at the start
(z = z_init) and end (z = 0) of the run, each against linear theory. On large
scales both track linear P(k). The small scales are limited by the force-mesh
resolution here -- CIC smoothing suppresses power near the Nyquist scale at both
epochs -- so the nonlinear sharpening of the web is easier to see in the
animation than in this coarse-mesh spectrum. Right: the large-scale growth of
the density amplitude across the run, measured by cross-correlating each
snapshot with the initial field, overlaid on the linear growth factor
D(a)/D(a_init). They should lie on top of each other -- the headline check that
the integrator reproduces linear growth where it must.

Run: pixi run python scripts/plot_growth.py
Saves: outputs/growth.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology, TimeStepping  # noqa: E402
from mbody import cosmology as C  # noqa: E402
from mbody import fields as F  # noqa: E402
from mbody import integrate as IG  # noqa: E402
from mbody import painting as PA  # noqa: E402

import mlx.core as mx  # noqa: E402


def main():
    cosmo = Cosmology()
    box = BoxConfig(box_size=512.0, n_mesh=128, n_particles=128)
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=20)
    z_init = time.z_init

    _, _, kmag = F.k_grid(box)
    low_k = (kmag > 0) & (kmag < 0.05)  # deeply linear band for the growth check

    history = []  # (a, low-k Fourier modes of the density)

    def snapshot(step, a, x, p):
        dk = np.asarray(mx.fft.rfftn(PA.density_contrast(x, box)))
        history.append((a, dk[low_k].copy()))

    x0, _ = IG.initial_state(box, cosmo, time, seed=0)
    xf, _ = IG.leapfrog(box, cosmo, time, seed=0, snapshot=snapshot)

    # Left panel: P(k) at the two epochs vs linear theory.
    ks_i, pk_i, _ = F.power_spectrum(PA.density_contrast(x0, box), box)
    ks_f, pk_f, _ = F.power_spectrum(PA.density_contrast(xf, box), box)
    Plin_i = C.linear_power(ks_i, cosmo, z=z_init)
    Plin_f = C.linear_power(ks_f, cosmo, z=0.0)

    # Right panel: large-scale growth = Re<dk(a) dk(a_i)*> / <|dk(a_i)|^2>.
    dk0 = history[0][1]
    a_arr = np.array([a for a, _ in history])
    R = np.array(
        [
            np.real(np.sum(dk * np.conj(dk0)) / np.sum(np.abs(dk0) ** 2))
            for _, dk in history
        ]
    )
    D_lin = np.array([C.growth_factor(1.0 / a - 1.0, cosmo) for a in a_arr])
    D_lin = D_lin / D_lin[0]  # normalize to the initial epoch

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.loglog(ks_f, pk_f, "o", ms=4, color="C3", label="measured z=0")
    ax1.loglog(ks_f, Plin_f, "-", color="C3", lw=1.2, label="linear z=0")
    ax1.loglog(
        ks_i, pk_i, "s", ms=3, color="C0", alpha=0.7, label="measured z=%g" % z_init
    )
    ax1.loglog(ks_i, Plin_i, "--", color="C0", lw=1.0, label="linear z=%g" % z_init)
    ax1.axvline(box.k_nyquist, color="gray", ls=":", lw=0.8)
    ax1.set_xlabel("k [h/Mpc]")
    ax1.set_ylabel(r"$P(k)$ [$(\mathrm{Mpc}/h)^3$]")
    ax1.set_title("power spectrum vs linear theory (z = 9 and z = 0)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.2)

    ax2.plot(a_arr, D_lin, "-", color="k", lw=1.5, label="linear D(a)/D(a_init)")
    ax2.plot(a_arr, R, "o", ms=5, color="C3", label="measured (PM, large scales)")
    ax2.set_xlabel("scale factor a")
    ax2.set_ylabel("large-scale density growth")
    ax2.set_title("growth tracks linear theory on large scales")
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    os.makedirs("outputs", exist_ok=True)
    fig.tight_layout()
    fig.savefig("outputs/growth.png", dpi=130)
    print("wrote outputs/growth.png")


if __name__ == "__main__":
    main()
