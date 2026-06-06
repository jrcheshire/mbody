"""Validation figure: does the PM leapfrog grow structure like linear theory?

Two checks in one figure. Left: the measured power spectrum at the start
(z = z_init) and end (z = 0) of the run, each against linear theory. On large
scales both track linear P(k); small scales are limited by the force-mesh
resolution (CIC smoothing suppresses power near Nyquist at both epochs), so the
nonlinear sharpening of the web is easier to see in the animation than in this
coarse-mesh spectrum. Right: the large-scale growth of the density amplitude
across the run, overlaid on the linear growth factor D(a)/D(a_init). They should
lie on top of each other -- the headline check that the integrator reproduces
linear growth where it must.

Both panels are built from mbody.diagnostics (particle_power, the
SnapshotRecorder growth history, linear_growth_reference) -- the same estimators
the dashboard and the tests use -- rather than re-deriving them here.

Run: pixi run python scripts/plot_growth.py
Saves: outputs/growth.png
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mbody.config import BoxConfig, Cosmology, TimeStepping  # noqa: E402
from mbody import cosmology as C  # noqa: E402
from mbody import diagnostics as D  # noqa: E402
from mbody import integrate as IG  # noqa: E402


def main():
    cosmo = Cosmology()
    box = BoxConfig(box_size=512.0, n_mesh=128, n_particles=128)
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=20)
    z_init = time.z_init

    # Growth only -- skip the slab projections this figure does not use.
    rec = D.SnapshotRecorder(box, record_slab=False)
    x0, _ = IG.initial_state(box, cosmo, time, seed=0)
    xf, _ = IG.leapfrog(box, cosmo, time, seed=0, snapshot=rec)

    # Left panel: P(k) at the two epochs vs linear theory.
    ks_i, pk_i, _ = D.particle_power(x0, box)
    ks_f, pk_f, _ = D.particle_power(xf, box)
    Plin_i = C.linear_power(ks_i, cosmo, z=z_init)
    Plin_f = C.linear_power(ks_f, cosmo, z=0.0)

    # Right panel: large-scale growth vs linear D(a)/D(a_init).
    a_arr, R = rec.growth_history()
    D_lin = D.linear_growth_reference(a_arr, cosmo)

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
