"""Tests for the differentiable band powers (mbody.fields) and the biased-tracer
f_NL response (mbody.bias) -- Step 5, on the linear field.

The headline: mx.grad of a local-bias tracer's band power w.r.t. f_NL recovers
the Dalal scale-dependent bias ~ 1/M(k), matches a matched-phase finite
difference, and is ~0 for the matter field (the null). Forward-mode mx.jvp is
NOT used: it is wrong through the FFT here (returns ~half the correct
derivative); reverse-mode mx.grad matches FD to ~1e-4. Tolerances measured with
scripts/probe_fnl_bias.py, not guessed.
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import fields as F
from mbody import ic as IC
from mbody import bias as B

COSMO = Cosmology()
# L = 256 gives a small fundamental (low-k reach for the 1/k^2 shape); n = 32
# keeps the per-bin reverse-mode grads fast.
BOX = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
B1, B2 = 2.0, 1.0
EPS = 50.0
NSEED = 8


def _kb():
    kf = BOX.k_fundamental
    return np.array([1, 2, 3, 4]) * kf


def _tracer_power(f_NL, seed, kb, b2=B2):
    delta = IC.linear_density(BOX, COSMO, seed=seed, f_NL=f_NL)
    return F.band_power(B.local_bias_tracer(delta, B1, b2), BOX, kb)


def _power_and_deriv(seed, kb, b2=B2):
    P = np.asarray(_tracer_power(mx.array(0.0), seed, kb, b2), dtype=np.float64)
    dP = np.array(
        [
            float(
                mx.grad(lambda f, i=i: _tracer_power(f, seed, kb, b2)[i])(mx.array(0.0))
            )
            for i in range(len(kb))
        ]
    )
    return P, dP


def _dlnp_fd(seed, kb, b2=B2):
    lp = mx.log(_tracer_power(mx.array(EPS), seed, kb, b2))
    lm = mx.log(_tracer_power(mx.array(-EPS), seed, kb, b2))
    return np.asarray((lp - lm) / (2 * EPS), dtype=np.float64)


def _seed_mean_dlnp(b2):
    kb = _kb()
    sP = np.zeros(len(kb))
    sdP = np.zeros(len(kb))
    for s in range(NSEED):
        P, dP = _power_and_deriv(s, kb, b2)
        sP += P
        sdP += dP
    return sdP / sP  # <dP/df> / <P>


def test_band_power_matches_power_spectrum():
    # Same shells, same V/N^6 normalization as the (numpy) power_spectrum.
    d = F.gaussian_random_field(BOX, COSMO, seed=0)
    centers, pk, nmodes = F.power_spectrum(d, BOX)
    bp = np.asarray(F.band_power(d, BOX, centers, dk=BOX.k_fundamental), np.float64)
    well = nmodes >= 20
    assert np.allclose(bp[well], pk[well], rtol=1e-3)


def test_cross_power_auto_equals_band_power():
    d = IC.linear_density(BOX, COSMO, seed=2, f_NL=100.0)
    kb = _kb()
    auto = np.asarray(F.cross_power(d, d, BOX, kb), np.float64)
    band = np.asarray(F.band_power(d, BOX, kb), np.float64)
    assert np.allclose(auto, band, rtol=1e-5)


def test_local_bias_tracer():
    d = IC.linear_density(BOX, COSMO, seed=0, f_NL=100.0)
    # b2 = 0 is pure linear bias.
    assert np.allclose(
        np.asarray(B.local_bias_tracer(d, 2.0, 0.0)), np.asarray(2.0 * d), rtol=1e-6
    )
    # The quadratic tracer is ~zero-mean (the <delta^2> subtraction).
    dh = B.local_bias_tracer(d, 2.0, 1.0)
    rms = float(mx.sqrt(mx.mean(dh**2)))
    assert abs(float(mx.mean(dh))) < 1e-2 * rms


def test_grad_dlnp_matches_fd():
    # Reverse-mode autodiff of dlnP/df_NL matches the matched-phase FD per seed
    # (measured max rel.err ~2.6e-4).
    kb = _kb()
    P, dP = _power_and_deriv(0, kb)
    assert np.all(np.abs((dP / P) / _dlnp_fd(0, kb) - 1.0) < 1e-3)


def test_matter_null():
    # The matter field (b2 = 0) has no O(f_NL) power response: dlnP/df_NL ~ 0,
    # tiny next to the tracer signal (measured |ratio| ~0.04 at nseed=8).
    tracer = _seed_mean_dlnp(B2)
    matter = _seed_mean_dlnp(0.0)
    assert np.all(np.abs(matter) < 0.1 * np.abs(tracer))


def test_scale_dependent_bias_shape():
    # The Dalal signature: dlnP/df_NL is positive, falls with k, and tracks the
    # 1/M(k) reference shape at large scales (measured first-4-bin ratio 1.0-1.31).
    kb = _kb()
    dlnp = _seed_mean_dlnp(B2)
    ref = B.scale_dependent_shape(kb, COSMO)
    ref = ref * (dlnp[0] / ref[0])  # normalize at the largest scale
    assert np.all(dlnp > 0)
    assert np.all(np.diff(dlnp) < 0)  # monotonically decreasing in k
    assert np.all(np.abs(dlnp / ref - 1.0) < 0.4)


def test_scale_dependent_shape_is_inverse_M():
    # The helper is 1/M(k); at large scales M ~ k^2 T so it steepens toward 1/k^2.
    kb = _kb()
    shape = B.scale_dependent_shape(kb, COSMO)
    M = IC.poisson_M(kb, COSMO)
    assert np.allclose(shape, 1.0 / M, rtol=1e-10)
    assert np.all(np.diff(shape) < 0)  # falls with k
