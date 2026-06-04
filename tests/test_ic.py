"""Tests for mbody.ic: Gaussian + local-f_NL initial conditions.

At f_NL = 0 the local machinery must round-trip to the Gaussian field. The f_NL
term enters delta linearly and as a single multiplicative term, so the field's
f_NL-dependence is exactly proportional to f_NL; the field develops a positive
skewness growing with f_NL; the primordial potential comes out at the physical
~1e-5 scale (a check on the Poisson-factor normalization); and the field is
differentiable in f_NL (kink-free, so a finite difference matches tightly).
"""

import numpy as np
import mlx.core as mx

from mbody.config import BoxConfig, Cosmology
from mbody import cosmology as C
from mbody import fields as F
from mbody import ic as IC

COSMO = Cosmology()
BOX = BoxConfig(box_size=256.0, n_mesh=64, n_particles=64)


def _skewness(field):
    d = np.asarray(field, np.float64).ravel()
    d = d - d.mean()
    return np.mean(d**3) / np.mean(d**2) ** 1.5


def test_fnl_zero_recovers_gaussian():
    delta_g = F.gaussian_random_field(BOX, COSMO, seed=0, z=0.0)
    delta_0 = IC.linear_density(BOX, COSMO, seed=0, z=0.0, f_NL=0.0)
    scale = float(mx.sqrt(mx.mean(delta_g**2)))
    rel = float(mx.max(mx.abs(delta_0 - delta_g))) / scale
    assert rel < 1e-4  # M * (1/M) round-trip, to FFT round-off


def test_fnl_enters_linearly():
    # delta(f_NL) = delta_G + f_NL * X with X fixed, so (delta(f) - delta_0)/f is
    # independent of f. This is exact (the path is kink-free, bit-reproducible).
    d0 = IC.linear_density(BOX, COSMO, seed=1, f_NL=0.0)
    x1 = (IC.linear_density(BOX, COSMO, seed=1, f_NL=100.0) - d0) / 100.0
    x2 = (IC.linear_density(BOX, COSMO, seed=1, f_NL=1000.0) - d0) / 1000.0
    scale = float(mx.sqrt(mx.mean(x1**2)))
    assert float(mx.max(mx.abs(x1 - x2))) / scale < 1e-3


def test_skewness_grows_with_fnl():
    s0 = _skewness(IC.linear_density(BOX, COSMO, seed=0, f_NL=0.0))
    s_lo = _skewness(IC.linear_density(BOX, COSMO, seed=0, f_NL=250.0))
    s_hi = _skewness(IC.linear_density(BOX, COSMO, seed=0, f_NL=500.0))
    assert abs(s0) < 0.02  # Gaussian: skewness ~ 0
    assert s_lo > 0.0 and s_hi > s_lo  # positive f_NL -> positive, growing skew
    assert 1.6 < s_hi / s_lo < 2.4  # approximately linear in f_NL


def test_primordial_potential_scale():
    phi = IC.primordial_potential(BOX, COSMO, seed=0, z=0.0)
    rms = float(mx.sqrt(mx.mean(phi**2)))
    assert float(mx.abs(mx.mean(phi))) < 1e-6  # zero mean (DC killed)
    assert 1e-6 < rms < 1e-3  # physical primordial-potential scale (~1e-5)


def test_poisson_factor_structure():
    # M(k) = (2/3)(c/H0)^2 D_md/Omega_m * k^2 T(k). (Can't test T -> 1 at low k:
    # this box's fundamental, 0.0245 h/Mpc, is already past k_eq ~ 0.015.) Verify
    # instead that M factorizes as that constant times k^2 T(k) over all k.
    M = IC.poisson_factor(BOX, COSMO, z=0.0)
    _, _, kmag = F.k_grid(BOX)
    sel = kmag > 0
    T = C.transfer(kmag[sel], COSMO, "camb")
    ratio = M[sel] / (kmag[sel] ** 2 * T)
    const = (
        (2.0 / 3.0) * IC.C_OVER_H0**2 * C.growth_factor_md(0.0, COSMO) / COSMO.Omega_m
    )
    assert np.all(M[sel] > 0)
    assert np.ptp(ratio) / np.mean(ratio) < 1e-4  # M is exactly const * k^2 T(k)
    assert abs(np.mean(ratio) / const - 1.0) < 1e-4  # with the analytic prefactor


def test_differentiable_in_fnl():
    def loss(fnl):
        return mx.sum(IC.linear_density(BOX, COSMO, seed=0, f_NL=fnl) ** 2)

    g = mx.grad(loss)(mx.array(0.0))
    mx.eval(g)
    assert bool(mx.isfinite(g)) and abs(float(g)) > 0.0

    def loss64(fnl):
        d = IC.linear_density(BOX, COSMO, seed=0, f_NL=fnl)
        return float(np.sum(np.asarray(d, np.float64) ** 2))

    h = 5.0
    fd = (loss64(h) - loss64(-h)) / (2.0 * h)
    ad = float(g)
    assert abs(fd - ad) / abs(ad) < 1e-2  # kink-free: FD matches AD tightly
