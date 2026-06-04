"""Tests for mbody.cosmology: transfer functions, growth, P(k), normalization.

Validation strategy (no tolerances relaxed without a measured basis):

- The EH98 port is pinned by hardcoded regression anchors (catches a typo in
  any coefficient) and cross-checked against the CAMB Boltzmann solver, which
  it must match to ~a few percent -- the published EH98 accuracy.
- sigma8 must be recovered by every backend (the amplitude normalization).
- Growth is checked against exact LCDM limits (D(0)=1, D ~ a at high z,
  f(0) ~ Omega_m^0.55).
"""

import math

import numpy as np
import pytest

from mbody.config import BoxConfig, Cosmology
from mbody import cosmology as C

COSMO = Cosmology()

# EH98 transfer anchors at the default cosmology, from this implementation.
# Pure regression guard on the port (independently reproducible by compiling
# the authors' tf_fit.c). Columns: k [h/Mpc], T_full, T_nowiggle, T_zerobaryon.
_EH98_ANCHORS = np.array(
    [
        [1.000e-04, 9.99906808e-01, 9.99899339e-01, 9.99899339e-01],
        [1.000e-03, 9.92249532e-01, 9.92070436e-01, 9.92070446e-01],
        [1.000e-02, 7.80210566e-01, 7.86253578e-01, 7.87797017e-01],
        [1.540e-02, 6.67048081e-01, 6.72918993e-01, 6.82405230e-01],
        [5.000e-02, 2.71274686e-01, 2.78617295e-01, 3.35985246e-01],
        [1.000e-01, 1.31247778e-01, 1.33502235e-01, 1.72447865e-01],
        [2.000e-01, 5.51201976e-02, 5.47496600e-02, 7.43513739e-02],
        [5.000e-01, 1.41176416e-02, 1.41225775e-02, 2.00071457e-02],
        [1.000e00, 4.69301170e-03, 4.66460143e-03, 6.73402717e-03],
    ]
)


def test_eh98_anchor_regression():
    k = _EH98_ANCHORS[:, 0]
    assert np.allclose(C.transfer_eh98(k, COSMO), _EH98_ANCHORS[:, 1], rtol=1e-5)
    assert np.allclose(
        C.transfer_eh98_nowiggle(k, COSMO), _EH98_ANCHORS[:, 2], rtol=1e-5
    )
    assert np.allclose(
        C.transfer_eh98_zerobaryon(k, COSMO), _EH98_ANCHORS[:, 3], rtol=1e-5
    )


def test_transfer_low_k_limit():
    # Every transfer function -> 1 as k -> 0.
    for backend in ("eh98", "eh98_nowiggle", "eh98_zerobaryon", "camb"):
        if backend == "camb":
            pytest.importorskip("camb")
        assert abs(C.transfer(1e-4, COSMO, backend=backend) - 1.0) < 2e-3


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        C.transfer(0.1, COSMO, backend="halofit")


def test_eh98_matches_camb_broadband():
    # The physics validation: the EH98 fit must reproduce the CAMB Boltzmann
    # solve to ~a few percent. Measured worst-case ~3.9% over [1e-3, 10] h/Mpc
    # (at a BAO feature); bound set just above that.
    pytest.importorskip("camb")
    k = np.logspace(-3, 1, 400)
    ratio = C.linear_power(k, COSMO, backend="eh98") / C.linear_power(
        k, COSMO, backend="camb"
    )
    assert np.max(np.abs(ratio - 1.0)) < 0.05
    assert np.median(np.abs(ratio - 1.0)) < 0.02


def test_sigma8_normalization():
    for backend in ("eh98", "eh98_nowiggle", "camb"):
        if backend == "camb":
            pytest.importorskip("camb")
        s8 = C.sigma_R(8.0, COSMO, z=0.0, backend=backend)
        assert abs(s8 - COSMO.sigma8) / COSMO.sigma8 < 5e-3


def test_growth_factor_limits():
    assert abs(C.growth_factor(0.0, COSMO) - 1.0) < 1e-12
    # Monotonic: more growth at later times.
    assert C.growth_factor(0.0, COSMO) > C.growth_factor(1.0, COSMO)
    assert C.growth_factor(1.0, COSMO) > C.growth_factor(9.0, COSMO)
    # Deep in matter domination D ~ a, so D*(1+z) is constant.
    d49 = C.growth_factor(49.0, COSMO) * 50.0
    d99 = C.growth_factor(99.0, COSMO) * 100.0
    assert abs(d49 / d99 - 1.0) < 1e-3


def test_growth_rate_limits():
    # f(0) ~ Omega_m^0.55 (the standard approximation), f -> 1 at high z.
    assert abs(C.growth_rate(0.0, COSMO) - COSMO.Omega_m**0.55) < 0.01
    assert abs(C.growth_rate(9.0, COSMO) - 1.0) < 5e-3


def test_linear_power_shape():
    pytest.importorskip("camb")
    k = np.logspace(-3, 0, 2000)
    Pk = C.linear_power(k, COSMO, backend="camb")
    assert np.all(Pk > 0)
    # Turns over near matter-radiation equality (~0.015 h/Mpc).
    assert 0.008 < k[np.argmax(Pk)] < 0.03
    # Falls at high k.
    assert Pk[-1] < Pk[np.argmax(Pk)]


def test_growth_scales_power():
    pytest.importorskip("camb")
    k = np.array([0.05, 0.1, 0.2])
    p0 = C.linear_power(k, COSMO, z=0.0, backend="camb")
    p1 = C.linear_power(k, COSMO, z=1.0, backend="camb")
    expected = (C.growth_factor(1.0, COSMO) / C.growth_factor(0.0, COSMO)) ** 2
    assert np.allclose(p1 / p0, expected, rtol=1e-6)


def test_mean_matter_density_and_particle_mass():
    assert C.mean_matter_density(COSMO) == COSMO.Omega_m * C.RHO_CRIT
    box = BoxConfig()  # 256 Mpc/h, 128^3 particles
    m_particle = C.mean_matter_density(COSMO) * (box.box_size / box.n_particles) ** 3
    # Each particle is ~7e11 M_sun/h at the default resolution.
    assert math.isclose(m_particle, 6.88e11, rel_tol=0.02)
