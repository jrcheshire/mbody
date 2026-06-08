"""Example 3 -- a Fisher forecast over {f_NL, b1, b2, A} on the linear field.

Builds the band-power Jacobian d ln P_b / d theta by reverse-mode autodiff
(seed-averaged), pairs it with the Gaussian band-power variance, and reports the
marginalized and conditional 1-sigma constraints. The gap between them is the
f_NL-bias degeneracy -- the lever the b_phi / multi-tracer work is all about.

This uses the linear-field Jacobian for speed; mbody.fisher also has the
PM-evolved (adjoint), redshift-space, and multi-tracer variants.

Run:  pixi run python examples/03_fisher_forecast.py
"""

import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import fisher as FI

COSMO = Cosmology()
BOX = BoxConfig(box_size=512.0, n_mesh=48, n_particles=48)
BACKEND = "eh98"
THETA_FID = {"f_NL": 0.0, "b1": 2.0, "b2": 1.0, "A": 1.0}
PRIORS = {"A": 0.1}  # a 10% external amplitude (sigma8-like) prior
NSEED = 6
KBINS = np.array([1, 2, 3, 4, 6, 8])


def main():
    kb = KBINS * BOX.k_fundamental

    # Seed-average the linear-field Jacobian d ln P_b / d theta.
    J = np.zeros((len(kb), len(FI.PARAM_NAMES)))
    for s in range(NSEED):
        J += FI.linear_logP_jacobian(
            BOX, COSMO, THETA_FID, kb, seed=s, backend=BACKEND
        )[0]
    J /= NSEED

    # Diagonal Gaussian data covariance: the band-power log-variance.
    var = FI.band_power_log_variance(BOX, kb)
    fc = FI.FisherForecast(J, var, THETA_FID, priors=PRIORS)

    print(fc.summary("linear field"))
    print("\nmarginalized vs conditional 1-sigma (inflation = the degeneracy cost):")
    for p in FI.PARAM_NAMES:
        marg = fc.sigma(p)
        cond = fc.sigma_conditional(p)
        print(
            f"  {p:>5}:  marg = {marg:10.3g}   cond = {cond:10.3g}   {marg / cond:5.1f}x"
        )


if __name__ == "__main__":
    main()
