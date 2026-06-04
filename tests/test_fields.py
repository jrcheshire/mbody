"""Tests for mbody.fields: k-grid, Gaussian random field, P(k) estimator.

The headline check is the round trip: a field generated with input P(k) must
measure back the same P(k) -- the generator and estimator normalizations are
inverse. We average several realizations and compare where cosmic variance and
low-k binning bias are small.
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import cosmology as C
from mbody import fields as F

COSMO = Cosmology()
BOX = BoxConfig(box_size=128.0, n_mesh=64, n_particles=64)


def test_k_grid_shape_and_extremes():
    k1d, kz1d, kmag = F.k_grid(BOX)
    N = BOX.n_mesh
    assert kmag.shape == (N, N, N // 2 + 1)
    assert kmag[0, 0, 0] == 0.0
    assert np.isclose(kz1d[-1], BOX.k_nyquist)
    assert np.isclose(np.sort(np.abs(k1d))[1], BOX.k_fundamental)


def test_field_is_real_zero_mean_finite():
    d = F.gaussian_random_field(BOX, COSMO, seed=0)
    mx.eval(d)
    assert d.dtype == mx.float32
    assert tuple(d.shape) == (BOX.n_mesh, BOX.n_mesh, BOX.n_mesh)
    assert abs(float(mx.mean(d))) < 1e-4
    assert bool(mx.all(mx.isfinite(d)))


def test_field_is_deterministic_in_seed():
    a = F.gaussian_random_field(BOX, COSMO, seed=7)
    b = F.gaussian_random_field(BOX, COSMO, seed=7)
    c = F.gaussian_random_field(BOX, COSMO, seed=8)
    mx.eval(a, b, c)
    assert float(mx.max(mx.abs(a - b))) == 0.0
    assert float(mx.max(mx.abs(a - c))) > 0.0


def test_measured_pk_recovers_input():
    # The normalization round trip. The modes-weighted mean over sampled bins
    # pins the absolute normalization (measured ~0.9985 input in dev runs);
    # well-sampled bins also agree per-bin within cosmic variance.
    nreal = 16
    ks, _, nmodes = F.power_spectrum(F.gaussian_random_field(BOX, COSMO, seed=0), BOX)
    acc = np.zeros_like(ks)
    for s in range(nreal):
        _, pk, _ = F.power_spectrum(
            F.gaussian_random_field(BOX, COSMO, seed=1000 + s), BOX
        )
        acc += pk
    ratio = (acc / nreal) / C.linear_power(ks, COSMO, backend="camb")
    sel = nmodes >= 20
    weighted = np.average(ratio[sel], weights=nmodes[sel])
    assert 0.97 < weighted < 1.03
    well = nmodes >= 200
    assert np.all(np.abs(ratio[well] - 1.0) < 0.12)


def test_power_spectrum_range_and_positive():
    d = F.gaussian_random_field(BOX, COSMO, seed=3)
    ks, pk, nmodes = F.power_spectrum(d, BOX)
    assert np.all(pk > 0)
    assert ks.min() >= 0.5 * BOX.k_fundamental
    assert ks.max() <= BOX.k_nyquist
    assert np.all(nmodes >= 1)
