"""Probe dlnP/df_NL of a biased tracer before fixing any test tolerances.

The headline of Step 5: differentiate a tracer's band power with respect to
f_NL and recover the Dalal scale-dependent bias ~ 1/k^2. This probe checks, on
the linear field (Stage 5a):

  * mx.grad (reverse mode) gives dlnP/df_NL that matches a matched-phase finite
    difference -- NOTE forward-mode mx.jvp is WRONG through the FFT here (it
    returned ~half the correct derivative; reverse mode matches FD to ~1e-4), so
    we differentiate per bin with mx.grad;
  * the tracer's seed-averaged dlnP/df_NL follows the 1/M(k) reference shape at
    large scales;
  * the matter field (b2 = 0) has dlnP/df_NL ~ 0 (the null contrast).

Run: pixi run python scripts/probe_fnl_bias.py
Nothing is written; the printed numbers set tolerances in tests/test_bias.py.
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import fields as F
from mbody import ic as IC
from mbody import bias as B

COSMO = Cosmology()
BOX = BoxConfig(box_size=256.0, n_mesh=64, n_particles=64)
B1, B2 = 2.0, 1.0
EPS = 50.0
NSEED = 24


def k_bins(box):
    kf = box.k_fundamental
    return np.array([n * kf for n in (1, 2, 3, 4, 6, 8, 12)])


def tracer_power(f_NL, seed, kb, b2=B2):
    """P(k) of the local-bias tracer of the linear f_NL field (MLX vector)."""
    delta = IC.linear_density(BOX, COSMO, seed=seed, f_NL=f_NL)
    delta_h = B.local_bias_tracer(delta, B1, b2)
    return F.band_power(delta_h, BOX, kb)


def power_and_deriv(seed, kb, b2=B2):
    """(P, dP/df_NL) at f_NL = 0, the derivative per bin via reverse-mode grad."""
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
    """Matched-phase finite difference of dlnP/df_NL at f_NL = 0."""
    lp = mx.log(tracer_power(mx.array(EPS), seed, kb, b2))
    lm = mx.log(tracer_power(mx.array(-EPS), seed, kb, b2))
    return np.asarray((lp - lm) / (2 * EPS), dtype=np.float64)


def main():
    kb = k_bins(BOX)
    shape = B.scale_dependent_shape(kb, COSMO)  # 1 / M(k)

    print("=== grad vs matched-phase FD (per seed, dlnP/df_NL) ===")
    rel = []
    for s in range(4):
        P, dP = power_and_deriv(1000 + s, kb)
        rel.append(np.abs((dP / P) / dlnp_fd(1000 + s, kb) - 1.0))
    print("  max rel.err over 4 seeds, per bin:")
    print("  " + "  ".join(f"{r:.2e}" for r in np.array(rel).max(axis=0)))

    print("\n=== shape: <dP/df> / <P> vs 1/M(k) ===")
    sP = np.zeros(len(kb))
    sdP = np.zeros(len(kb))
    sP0 = np.zeros(len(kb))
    sdP0 = np.zeros(len(kb))
    for s in range(NSEED):
        P, dP = power_and_deriv(2000 + s, kb)
        P0, dP0 = power_and_deriv(2000 + s, kb, b2=0.0)
        sP += P
        sdP += dP
        sP0 += P0
        sdP0 += dP0
    dlnp = sdP / sP
    dlnp0 = sdP0 / sP0
    ref = shape * (dlnp[0] / shape[0])  # normalize 1/M(k) at the largest scale
    print("  k        dlnP/df_NL   1/M(k)*norm   ratio    matter(b2=0)")
    for i, k in enumerate(kb):
        print(
            f"  {k:6.4f}  {dlnp[i]:11.4e}  {ref[i]:11.4e}  "
            f"{dlnp[i]/ref[i]:6.3f}   {dlnp0[i]:11.4e}"
        )
    print(
        "\n  matter null |dlnP0|/|dlnP_tracer| (max): "
        f"{np.max(np.abs(dlnp0) / np.abs(dlnp)):.4f}"
    )


if __name__ == "__main__":
    main()
