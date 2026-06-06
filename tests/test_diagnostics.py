"""Tests for the shared diagnostics layer (mbody.diagnostics).

The cross-correlation estimator and the growth-amplitude reduction are checked
exactly on synthetic inputs; the SnapshotRecorder is exercised end to end on a
short leapfrog. The integrator uses the EH98 backend throughout to keep the
suite fast and CAMB-free.
"""

import numpy as np

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import diagnostics as D
from mbody import fields as F
from mbody import integrate as IG


def _box():
    return BoxConfig(box_size=200.0, n_mesh=32, n_particles=32)


def test_cross_correlation_self_is_unity():
    box, cosmo = _box(), Cosmology()
    d = F.gaussian_random_field(box, cosmo, seed=3, backend="eh98")
    k, r, n = D.cross_correlation(d, d, box)
    assert np.all(n > 0)
    assert np.allclose(r, 1.0, atol=1e-5)


def test_cross_correlation_invariant_to_amplitude():
    # r is amplitude-independent: a perfectly rescaled copy still has r = 1.
    box, cosmo = _box(), Cosmology()
    d = F.gaussian_random_field(box, cosmo, seed=4, backend="eh98")
    k, r, n = D.cross_correlation(d, 2.5 * d, box)
    assert np.allclose(r, 1.0, atol=1e-5)


def test_cross_correlation_bounded():
    box, cosmo = _box(), Cosmology()
    a = F.gaussian_random_field(box, cosmo, seed=1, backend="eh98")
    b = F.gaussian_random_field(box, cosmo, seed=2, backend="eh98")
    k, r, n = D.cross_correlation(a, b, box)
    assert np.all(r <= 1.0 + 1e-6)
    assert np.all(r >= -1.0 - 1e-6)


def test_growth_amplitude_exact_on_rescaled_modes():
    # m_i = s_i m_0 must give back exactly s_i, isolating growth from phases.
    rng = np.random.default_rng(0)
    ref = rng.normal(size=64) + 1j * rng.normal(size=64)
    factors = [1.0, 1.3, 2.0, 3.7]
    R = D.growth_amplitude([f * ref for f in factors])
    assert np.allclose(R, factors, atol=1e-12)


def test_linear_growth_reference_normalized():
    cosmo = Cosmology()
    a = np.array([0.1, 0.3, 0.6, 1.0])
    Dref = D.linear_growth_reference(a, cosmo)
    assert Dref[0] == 1.0
    assert np.all(np.diff(Dref) > 0)  # growth increases with a


def test_recorder_runs_with_leapfrog():
    box, cosmo = _box(), Cosmology()
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=8)
    rec = D.SnapshotRecorder(box, slab_thick=2)
    IG.leapfrog(box, cosmo, time, seed=0, backend="eh98", snapshot=rec)

    n_frames = time.n_steps + 1
    assert len(rec.a) == n_frames
    assert len(rec.steps) == n_frames
    assert len(rec.slabs) == n_frames
    # each slab is the 2D projection of a 2-cell-thick Lagrangian slab
    assert rec.slabs[0].shape == (box.n_mesh * box.n_mesh * 2, 2)

    a_arr, R = rec.growth_history()
    assert np.isclose(R[0], 1.0)
    assert R[-1] > R[0]  # structure grows under gravity
    # Large-scale growth tracks linear D(a) to a loose band; the precise ~2%
    # low-step-count deficit is a separately measured figure-level result, not
    # pinned here.
    Dref = D.linear_growth_reference(a_arr, cosmo)
    assert abs(R[-1] / Dref[-1] - 1.0) < 0.1


def test_skewness_zero_for_gaussian_positive_for_skewed():
    rng = np.random.default_rng(0)
    g = rng.normal(size=200000)
    assert abs(D.skewness(g)) < 0.05  # symmetric -> ~0
    # an exponential sample is right-skewed (skewness = 2 in the limit)
    e = rng.exponential(size=200000)
    assert D.skewness(e) > 1.0


def test_skewness_flips_sign():
    rng = np.random.default_rng(1)
    e = rng.exponential(size=100000)
    assert np.isclose(D.skewness(-e), -D.skewness(e), rtol=1e-12)


def test_one_point_pdf_normalized():
    box, cosmo = _box(), Cosmology()
    d = F.gaussian_random_field(box, cosmo, seed=5, backend="eh98")
    centers, density, sigma, sk = D.one_point_pdf(d, bins=100, span=6.0)
    assert centers.shape == density.shape == (100,)
    assert sigma > 0
    width = centers[1] - centers[0]
    assert abs(np.sum(density) * width - 1.0) < 0.05  # integrates to ~1


def test_particle_power_recovers_power_spectrum():
    box, cosmo = _box(), Cosmology()
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=4)
    xf, _ = IG.leapfrog(box, cosmo, time, seed=0, backend="eh98")
    k, pk, n = D.particle_power(xf, box)
    assert np.all(n > 0)
    assert np.all(pk > 0)
    assert k.shape == pk.shape == n.shape
