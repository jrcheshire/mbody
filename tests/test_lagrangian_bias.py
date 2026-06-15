"""Tests for the Lagrangian-bias tracer.

Covers the second-order bias operators (tidal s^2, Laplacian, the operator
decomposition), the linear-field b_phi response (gate 1: the explicit b_phi imprints
the Dalal kernel 1/M(k)), and a lean through-PM check (gate 2: the advected tracer's
d ln P_g/df_NL matches the linear Dalal value with no Eulerian-style ~0.25 dilution).
The full validation, with tables, is in scripts/probe_lagrangian_bias.py.
"""

import mlx.core as mx
import numpy as np

from mbody import bias as B
from mbody import cosmology as C
from mbody import fields as F
from mbody import ic as IC
from mbody import integrate as IG
from mbody import painting as PA
from mbody.config import BoxConfig, Cosmology, TimeStepping

BACKEND = "eh98"  # closed-form transfer: no CAMB call, keeps the tests fast
DELTA_C = 1.686


def _axis0_cosine(box, m):
    """A single cosine mode cos(k0 x) along axis 0, k0 = 2 pi m / L (real field)."""
    N = box.n_mesh
    line = np.cos(2.0 * np.pi * m * np.arange(N) / N).astype(np.float32)
    field = np.broadcast_to(line[:, None, None], (N, N, N)).copy()
    return mx.array(field), 2.0 * np.pi * m / box.box_size


def _pweighted_inv_M(box, cosmo, k_bins, dk, z):
    """P_lin-weighted shell average of 1/M(k), matching the cross/auto estimator."""
    _, _, k_mag = F.k_grid(box)
    km = k_mag.ravel()
    P_lin = C.linear_power(km, cosmo, z=z, backend=BACKEND)
    M = IC.poisson_M(np.where(km > 0, km, 1.0), cosmo, z=z, backend=BACKEND)
    inv_M = np.where(km > 0, P_lin / M, 0.0)
    P_wt = np.where(km > 0, P_lin, 0.0)
    out = np.empty(len(k_bins))
    for b, kc in enumerate(k_bins):
        m = (km >= kc - 0.5 * dk) & (km < kc + 0.5 * dk)
        out[b] = inv_M[m].sum() / P_wt[m].sum()
    return out


def _tracer_field(box, cosmo, time, seed, f_NL, b1, b_phi):
    """Advected Lagrangian-bias tracer overdensity (forward-only)."""
    z = time.z_final
    d = IC.linear_density(box, cosmo, seed=seed, z=z, f_NL=f_NL, backend=BACKEND)
    phi = IC.primordial_potential(box, cosmo, seed=seed, z=z, backend=BACKEND)
    dgL = B.lagrangian_bias_field(d, box, b1=b1, b_phi=b_phi, f_NL=f_NL, phi_G=phi)
    w = (1.0 + dgL).reshape(-1)
    x0, p0 = IG.initial_state(
        box, cosmo, time, seed=seed, f_NL=f_NL, backend=BACKEND, lpt_order=2
    )
    x_final, _ = IG.evolve_state(
        x0, p0, box, cosmo, IG.a_grid(time), integrator=time.integrator
    )
    n_g = PA.cic_paint(x_final, box, weights=w)
    return n_g / (float(w.sum()) / box.n_mesh**3) - 1.0


def test_laplacian_pure_mode():
    box = BoxConfig(box_size=200.0, n_mesh=16)
    field, k0 = _axis0_cosine(box, m=2)
    lap = np.asarray(B.laplacian_field(field, box))
    assert np.allclose(lap, -(k0**2) * np.asarray(field), rtol=2e-3, atol=1e-4)


def test_tidal_sq_axis_mode():
    # For an axis-aligned mode s^2 = (2/3) delta^2 (s_xx = 2/3 delta, s_yy = s_zz =
    # -1/3 delta, off-diagonals zero).
    box = BoxConfig(box_size=200.0, n_mesh=16)
    field, _ = _axis0_cosine(box, m=3)
    s2 = np.asarray(B.tidal_sq(field, box))
    assert np.allclose(s2, (2.0 / 3.0) * np.asarray(field) ** 2, rtol=3e-3, atol=1e-4)


def test_tidal_sq_nonneg():
    box = BoxConfig(box_size=512.0, n_mesh=32)
    d = IC.linear_density(box, Cosmology(), seed=1, z=0.0, backend=BACKEND)
    s2 = np.asarray(B.tidal_sq(d, box))
    assert s2.shape == (32, 32, 32)
    assert np.all(s2 >= -1e-5)  # a sum of squares, up to float round-off


def test_lagrangian_bias_components():
    box = BoxConfig(box_size=512.0, n_mesh=32)
    cosmo = Cosmology()
    d = IC.linear_density(box, cosmo, seed=2, z=0.0, backend=BACKEND)
    phi = IC.primordial_potential(box, cosmo, seed=2, z=0.0, backend=BACKEND)
    # b1 only -> b1 delta
    f1 = np.asarray(B.lagrangian_bias_field(d, box, b1=1.7))
    assert np.allclose(f1, 1.7 * np.asarray(d), rtol=1e-5, atol=1e-6)
    # b_phi only -> b_phi f_NL phi_G (the explicit scale-dependent bias term)
    fphi = np.asarray(
        B.lagrangian_bias_field(d, box, b1=0.0, b_phi=3.0, f_NL=50.0, phi_G=phi)
    )
    assert np.allclose(fphi, 150.0 * np.asarray(phi), rtol=1e-5, atol=1e-8)
    # the b2 operator is mean-subtracted
    fb2 = np.asarray(B.lagrangian_bias_field(d, box, b1=0.0, b2=2.0))
    assert abs(fb2.mean()) < 1e-3 * np.std(fb2)


def test_lagrangian_bias_per_cell_coeffs():
    # Coefficients may be per-cell (N, N, N) fields, for radially-evolving bias.
    box = BoxConfig(box_size=512.0, n_mesh=16)
    cosmo = Cosmology()
    N = box.n_mesh
    d = IC.linear_density(box, cosmo, seed=3, z=0.0, backend=BACKEND)
    phi = IC.primordial_potential(box, cosmo, seed=3, z=0.0, backend=BACKEND)
    grad = np.broadcast_to(
        (1.0 + 0.5 * np.arange(N) / N)[:, None, None], (N, N, N)
    ).copy()
    f1 = np.asarray(B.lagrangian_bias_field(d, box, b1=grad))
    assert np.allclose(f1, grad * np.asarray(d), rtol=1e-5, atol=1e-6)
    fp = np.asarray(
        B.lagrangian_bias_field(d, box, b1=0.0, b_phi=grad, f_NL=20.0, phi_G=phi)
    )
    assert np.allclose(fp, grad * 20.0 * np.asarray(phi), rtol=1e-5, atol=1e-8)


def test_gate1_bphi_linear_response():
    box = BoxConfig(box_size=2048.0, n_mesh=32)
    cosmo = Cosmology()
    b1, b_phi, f_NL = 1.0, 5.0, 100.0
    dG = IC.linear_density(box, cosmo, seed=0, z=0.0, f_NL=0.0, backend=BACKEND)
    phi = IC.primordial_potential(box, cosmo, seed=0, z=0.0, backend=BACKEND)
    dgL = B.lagrangian_bias_field(dG, box, b1=b1, b_phi=b_phi, f_NL=f_NL, phi_G=phi)
    dk = box.k_fundamental
    kb = dk * np.arange(1, 8)
    ratio = np.asarray(F.cross_power(dgL, dG, box, kb, dk=dk)) / np.asarray(
        F.band_power(dG, box, kb, dk=dk)
    )
    measured = ratio - b1
    pred = b_phi * f_NL * _pweighted_inv_M(box, cosmo, kb, dk, 0.0)
    # mid bins (skip the noisy fundamental): the Dalal kernel to within 5%
    r = measured[1:6] / pred[1:6]
    assert np.all(np.abs(r - 1.0) < 0.05)


def test_gate2_through_pm_no_dilution():
    box = BoxConfig(box_size=2048.0, n_mesh=32)
    cosmo = Cosmology()
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=4, integrator="bullfrog")
    b1, eps = 1.0, 10.0
    b_phi = 2.0 * DELTA_C * (1.0 + b1 - 0.55)  # Barreira, b1_E = 1 + b1
    dk = box.k_fundamental
    kb = dk * np.arange(1, 5)
    derivs = []
    for seed in (0, 1):
        dp = _tracer_field(box, cosmo, time, seed, +eps, b1, b_phi)
        dm = _tracer_field(box, cosmo, time, seed, -eps, b1, b_phi)
        Pp = np.asarray(F.band_power(dp, box, kb, dk=dk))
        Pm = np.asarray(F.band_power(dm, box, kb, dk=dk))
        derivs.append((np.log(Pp) - np.log(Pm)) / (2.0 * eps))
    meas = np.mean(derivs, axis=0)
    pred = 2.0 * b_phi / (1.0 + b1) * _pweighted_inv_M(box, cosmo, kb, dk, 0.0)
    r = meas / pred
    # lowest 2 bins: clean Dalal recovery, no Eulerian-style ~0.25 dilution
    assert np.all(np.abs(r[:2] - 1.0) < 0.08)
    assert np.all(r[:2] > 0.7)
