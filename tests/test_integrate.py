"""Tests for mbody.integrate: the scale-factor leapfrog PM stepper.

The load-bearing check is linear growth: on the largest scales the evolved field
must grow as the linear growth factor D(a), and the plain leapfrog must converge
toward it as the step count rises (the residual at low step count is the
expected discretization error that FastPM kernels later remove). Plus the
background kick/drift integrals, the Zel'dovich growing-mode velocity IC,
determinism, the snapshot callback, and that mx.grad flows through the whole
unrolled trajectory.
"""

import numpy as np
import mlx.core as mx

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import cosmology as C
from mbody import fields as F
from mbody import lpt as L
from mbody import painting as PA
from mbody import integrate as IG

COSMO = Cosmology()
SMALL = BoxConfig(box_size=128.0, n_mesh=16, n_particles=16)


def test_a_grid_endpoints_and_spacing():
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10)
    for spacing in ("linear", "log"):
        ag = IG.a_grid(t, spacing=spacing)
        assert ag.shape == (11,)
        assert abs(ag[0] - 0.1) < 1e-12  # a = 1/(1+9)
        assert abs(ag[-1] - 1.0) < 1e-12  # a = 1/(1+0)
        assert np.all(np.diff(ag) > 0)
    lin = IG.a_grid(t, "linear")
    assert np.allclose(np.diff(lin), lin[1] - lin[0])  # equal steps in a


def test_kick_drift_factors_positive_and_additive():
    a0, ac, a1 = 0.2, 0.35, 0.5
    for fac in (IG.kick_factor, IG.drift_factor):
        assert fac(a0, a1, COSMO) > 0
        whole = fac(a0, a1, COSMO)
        split = fac(a0, ac, COSMO) + fac(ac, a1, COSMO)
        assert abs(split - whole) / whole < 1e-6  # integral is additive


def test_initial_state_positions_and_growing_mode_velocity():
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10)
    x, p = IG.initial_state(SMALL, COSMO, t, seed=0)
    npart = SMALL.n_particles**3
    assert tuple(x.shape) == (npart, 3) and tuple(p.shape) == (npart, 3)
    assert x.dtype == mx.float32 and p.dtype == mx.float32
    assert float(mx.min(x)) >= 0.0 and float(mx.max(x)) < SMALL.box_size

    # Positions are exactly the z_init Zel'dovich layout.
    x_lpt = L.lpt_positions(SMALL, COSMO, seed=0, z=9.0)
    assert float(mx.max(mx.abs(x - x_lpt))) == 0.0

    # Velocity is the growing mode p = a^2 E f * (displacement), with the
    # displacement = x - q (minimum image). This is the IC that grows as D(a).
    a_i = 0.1
    q = np.asarray(L.lagrangian_grid(SMALL), np.float64)
    xn = np.asarray(x, np.float64)
    Lh = SMALL.box_size
    disp = (xn - q + Lh / 2) % Lh - Lh / 2
    coef = a_i**2 * float(C.E(9.0, COSMO)) * float(C.growth_rate(9.0, COSMO))
    p_expected = coef * disp
    rel = np.max(np.abs(np.asarray(p, np.float64) - p_expected)) / np.max(
        np.abs(p_expected)
    )
    assert rel < 1e-5


def test_determinism():
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
    xa, _ = IG.leapfrog(SMALL, COSMO, t, seed=4)
    xb, _ = IG.leapfrog(SMALL, COSMO, t, seed=4)
    xc, _ = IG.leapfrog(SMALL, COSMO, t, seed=5)
    mx.eval(xa, xb, xc)
    # Same seed agrees up to GPU scatter-add round-off, not bit-exactly: CIC's
    # atomic accumulation order is not reproducible on Metal (~1e-6 per paint,
    # growing mildly over steps), so this is a tiny tolerance, not == 0.
    assert float(mx.max(mx.abs(xa - xb))) < 1e-3
    assert float(mx.max(mx.abs(xa - xc))) > 1.0


def test_snapshot_callback():
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=4)
    calls = []

    def snap(step, a, x, p):
        calls.append((step, a, tuple(x.shape), tuple(p.shape)))

    IG.leapfrog(SMALL, COSMO, t, seed=0, snapshot=snap)
    assert [c[0] for c in calls] == [0, 1, 2, 3, 4]  # step 0 (IC) + n_steps
    a_vals = [c[1] for c in calls]
    assert abs(a_vals[0] - 0.1) < 1e-12 and abs(a_vals[-1] - 1.0) < 1e-12
    assert all(np.diff(a_vals) > 0)
    npart = SMALL.n_particles**3
    assert all(c[2] == (npart, 3) and c[3] == (npart, 3) for c in calls)


def test_linear_growth_converges_to_D():
    # Largest-scale modes must grow as the linear growth factor D, and the plain
    # leapfrog must converge toward it as the step count rises. The cross-
    # correlation estimator of the per-mode growth is measured on k < 0.04 h/Mpc
    # (deeply linear) of a large box; the residual at high step count is the
    # expected leapfrog discretization (plus a little CIC-resolution) error.
    box = BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
    ratio = C.growth_factor(0.0, COSMO) / C.growth_factor(9.0, COSMO)
    _, _, kmag = F.k_grid(box)
    mask = (kmag > 0) & (kmag < 0.04)

    def growth_estimate(n_steps):
        t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=n_steps)
        x0, _ = IG.initial_state(box, COSMO, t, seed=0)
        d_i = np.asarray(mx.fft.rfftn(PA.density_contrast(x0, box)))
        xf, _ = IG.leapfrog(box, COSMO, t, seed=0)
        d_f = np.asarray(mx.fft.rfftn(PA.density_contrast(xf, box)))
        return np.real(
            np.sum(d_f[mask] * np.conj(d_i[mask])) / np.sum(np.abs(d_i[mask]) ** 2)
        )

    err_coarse = abs(growth_estimate(6) / ratio - 1.0)
    err_fine = abs(growth_estimate(24) / ratio - 1.0)
    assert err_fine < err_coarse  # converging toward linear growth
    assert err_fine < 0.03  # within 3% of D on the largest scales


def test_differentiable_through_leapfrog():
    # The architecture headline: mx.grad flows through the full unrolled
    # trajectory (every kick, drift, force solve and CIC op over all steps).
    # Checked against a random directional finite difference; the ~few-percent
    # floor is the FD reference straddling CIC kinks across steps (the AD itself
    # is exact), and (as for the force) smaller h is the accurate end here.
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
    x0, p0 = IG.initial_state(SMALL, COSMO, t, seed=0)
    ag = IG.a_grid(t)

    def loss(x):
        xf, _ = IG.evolve_state(x, p0, SMALL, COSMO, ag)
        return mx.sum(PA.density_contrast(xf, SMALL) ** 2)

    g = mx.grad(loss)(x0)
    mx.eval(g)
    assert bool(mx.all(mx.isfinite(g)))
    assert float(mx.max(mx.abs(g))) > 0.0

    def loss64(x):
        xf, _ = IG.evolve_state(x, p0, SMALL, COSMO, ag)
        return float(
            np.sum(np.asarray(PA.density_contrast(xf, SMALL), np.float64) ** 2)
        )

    rng = np.random.default_rng(0)
    v = rng.standard_normal(tuple(x0.shape)).astype(np.float32)
    v /= np.linalg.norm(v)
    vmx = mx.array(v)
    h = 0.02 * SMALL.cell_size
    fd = (loss64(x0 + h * vmx) - loss64(x0 - h * vmx)) / (2.0 * h)
    ad = float(mx.sum(g * vmx))
    assert abs(fd - ad) / abs(ad) < 0.10
