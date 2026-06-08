"""Example 2 -- the headline: autodiff dlnP/df_NL on a biased tracer.

The matter power spectrum has no linear response to local f_NL, but a *biased*
tracer does: its large-scale power picks up the Dalal scale-dependent bias,
dlnP/df_NL ~ 1/M(k) ~ 1/k^2. Here we differentiate a local quadratic-bias
tracer's band power straight through with reverse-mode mx.grad and compare it to
(a) the absolute first-principles prediction 4 b2 sigma^2 / (b1 M(k)) -- no free
normalization -- and (b) the matter-field null (b2 = 0), which sits near zero.

Run:  pixi run python examples/02_dlnp_dfnl.py
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import bias as B
from mbody import fields as F
from mbody import ic as IC

COSMO = Cosmology()
BOX = BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
B1, B2 = 2.0, 1.0
NSEED = 8
BACKEND = "eh98"


def tracer_power(f_NL, seed, kb, b2):
    """Band power of a local-bias tracer of the linear f_NL density field."""
    delta = IC.linear_density(BOX, COSMO, seed=seed, f_NL=f_NL, backend=BACKEND)
    return F.band_power(B.local_bias_tracer(delta, B1, b2), BOX, kb)


def dlnp_dfnl(seed, kb, b2):
    """d ln P_b / d f_NL at f_NL = 0 -- one reverse-mode sweep per band."""
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


def main():
    kb = np.array([1, 2, 3, 4, 6, 8]) * BOX.k_fundamental

    # Seed-average the autodiff derivative for the tracer (b2=B2) and matter (b2=0).
    tracer = np.mean([dlnp_dfnl(s, kb, B2) for s in range(NSEED)], axis=0)
    matter = np.mean([dlnp_dfnl(s, kb, 0.0) for s in range(NSEED)], axis=0)

    # The absolute, first-principles prediction (no free normalization).
    pred = B.scale_dependent_bias_response_binned(
        BOX, COSMO, kb, B1, B2, backend=BACKEND
    )

    print(f"dlnP/df_NL of a local-bias tracer (b1={B1}, b2={B2}), {NSEED} seeds:")
    print(f"{'k [h/Mpc]':>10} {'autodiff':>12} {'prediction':>12} {'matter null':>12}")
    for ki, t, p, m in zip(kb, tracer, pred, matter):
        print(f"{ki:10.3f} {t:12.4f} {p:12.4f} {m:12.4f}")
    print("\nThe autodiff tracer response tracks the 1/M(k) prediction and rises")
    print("toward large scales (small k); the matter null sits near zero.")


if __name__ == "__main__":
    main()
