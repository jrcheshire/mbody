"""Probe the ABSOLUTE amplitude of the f_NL scale-dependent bias.

The Step 5 headline figure overlays the autodiff dlnP/df_NL on the 1/M(k) Dalal
shape *normalized at the largest scale* -- a shape-only check that leaves the
amplitude free. This probe closes that gap on the linear field: it compares the
seed-averaged autodiff dlnP/df_NL to the first-principles absolute prediction

    dlnP_h/df_NL(k) = 4 b2 sigma^2 / (b1 M(k))            (bias.scale_dependent_bias_response)

with sigma^2 = bias.mesh_variance(box) the mesh variance of the linear field --
NO free normalization. The prediction is the squeezed-limit leading term, so it
should track the measurement at low k and ride above it (sub-leading + b2^2 terms)
toward smaller scales. The numbers here set the tolerance in tests/test_bias.py.

Run: pixi run python scripts/probe_fnl_amplitude.py
Nothing is written.
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import fields as F
from mbody import ic as IC
from mbody import bias as B

COSMO = Cosmology()
# Big box for low-k reach (deeply squeezed first bins), modest N for fast grads.
BOX = BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
B1, B2 = 2.0, 1.0
EPS = 50.0
NSEED = 24


def k_bins(box):
    kf = box.k_fundamental
    return np.array([n * kf for n in (1, 2, 3, 4, 6, 8)])


def tracer_power(f_NL, seed, kb, b2=B2):
    delta = IC.linear_density(BOX, COSMO, seed=seed, f_NL=f_NL)
    return F.band_power(B.local_bias_tracer(delta, B1, b2), BOX, kb)


def power_and_deriv(seed, kb, b2=B2):
    P = np.asarray(tracer_power(mx.array(0.0), seed, kb, b2), dtype=np.float64)
    dP = np.array(
        [
            float(
                mx.grad(lambda f, i=i: tracer_power(f, seed, kb, b2)[i])(mx.array(0.0))
            )
            for i in range(len(kb))
        ]
    )
    return P, dP


def dlnp_fd(seed, kb, b2=B2):
    lp = mx.log(tracer_power(mx.array(EPS), seed, kb, b2))
    lm = mx.log(tracer_power(mx.array(-EPS), seed, kb, b2))
    return np.asarray((lp - lm) / (2 * EPS), dtype=np.float64)


def main():
    kb = k_bins(BOX)

    sigma2 = B.mesh_variance(BOX, COSMO)
    # Cross-check the ensemble mesh variance against a realized field's <delta^2>.
    real_var = np.mean(
        [
            float(mx.mean(IC.linear_density(BOX, COSMO, seed=1000 + s) ** 2))
            for s in range(8)
        ]
    )
    print(f"mesh_variance (ensemble) = {sigma2:.5e}")
    print(f"<delta^2> realized (8 seeds mean) = {real_var:.5e}")
    print(f"  ratio realized/ensemble = {real_var / sigma2:.4f}")

    # grad vs matched-phase FD (a few seeds): the derivative is trustworthy.
    rel = []
    for s in range(4):
        P, dP = power_and_deriv(1000 + s, kb)
        rel.append(np.abs((dP / P) / dlnp_fd(1000 + s, kb) - 1.0))
    print(
        "\ngrad vs matched-phase FD, max rel.err per bin over 4 seeds:\n  "
        + "  ".join(f"{r:.2e}" for r in np.array(rel).max(axis=0))
    )

    # Per-seed matched log-derivative dlnP/df_NL, then mean +/- SEM across seeds,
    # vs the ABSOLUTE analytic prediction (no free normalization).
    per_seed = np.empty((NSEED, len(kb)))
    for s in range(NSEED):
        P, dP = power_and_deriv(2000 + s, kb)
        per_seed[s] = dP / P
    dlnp = per_seed.mean(axis=0)
    sem = per_seed.std(axis=0, ddof=1) / np.sqrt(NSEED)
    pred_c = B.scale_dependent_bias_response(kb, BOX, COSMO, B1, B2)  # bin-centre
    pred_b = B.scale_dependent_bias_response_binned(BOX, COSMO, kb, B1, B2)  # shell-avg

    print(f"\nabsolute dlnP/df_NL vs 4 b2 sigma^2 / (b1 M(q)), {NSEED} seeds:")
    print("  (pred_c = bin-centre 1/M; pred_b = P-weighted shell average, exact)")
    print("  k        measured     +/-SEM     pred_c   pred_b   meas/c  meas/b")
    for i, k in enumerate(kb):
        print(
            f"  {k:6.4f}  {dlnp[i]:11.4e}  {sem[i]:9.2e}  {pred_c[i]:8.3e} "
            f"{pred_b[i]:8.3e}  {dlnp[i]/pred_c[i]:6.4f}  {dlnp[i]/pred_b[i]:6.4f}"
        )
    print(
        "\n  low-k (first 3 bins) meas/pred_binned: "
        + "  ".join(f"{dlnp[i]/pred_b[i]:.4f}" for i in range(3))
    )


if __name__ == "__main__":
    main()
