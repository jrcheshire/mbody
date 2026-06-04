"""Tests for mbody.forces: the FFT Poisson solve and the force on particles.

The force kernel is i k / k^2, the same Green's-function operation as the
Zel'dovich displacement. So the checks triangulate the solve from three sides:
the potential inverts the Laplacian (lap phi = delta), Gauss's law holds for the
acceleration (div g = -delta), and -- the headline -- the acceleration field of
a realization is identical to its Zel'dovich displacement. Plus a single-mode
analytic case, zero net force (parity of the kernel on the symmetric grid), and
that mx.grad flows through paint -> solve -> read to particle positions.
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import fields as F
from mbody import forces as FO
from mbody import lpt as L
from mbody import precision as P

COSMO = Cosmology()
BOX = BoxConfig(box_size=64.0, n_mesh=32, n_particles=32)


def _resolved_mask(N):
    """Boolean mask of non-Nyquist Fourier modes on the rfftn half-grid, plus
    the (KX, KY, KZ) integer-frequency grids, for spectral-identity checks.
    """
    nyq = N // 2
    kxi = np.fft.fftfreq(N)
    kzi = np.fft.rfftfreq(N)
    KX, KY, KZ = np.meshgrid(kxi, kxi, kzi, indexing="ij")
    resolved = (KX != kxi[nyq]) & (KY != kxi[nyq]) & (KZ != kzi[nyq])
    return resolved, KX, KY, KZ


def test_potential_inverts_poisson():
    # lap phi = delta: take phi = potential(delta), apply -k^2 in Fourier, and
    # recover delta on every resolved mode.
    delta = F.gaussian_random_field(BOX, COSMO, seed=0, z=0.0)
    phi = FO.potential(delta, BOX)

    N = BOX.n_mesh
    resolved, KX, KY, KZ = _resolved_mask(N)
    # Physical k uses the cell spacing; the constant cancels in the ratio below,
    # but keep it explicit for clarity.
    two_pi_over_d = 2.0 * np.pi / BOX.cell_size
    k2 = (two_pi_over_d**2) * (KX**2 + KY**2 + KZ**2)
    phik = np.fft.rfftn(np.asarray(phi, dtype=np.float64))
    dk = np.fft.rfftn(np.asarray(delta, dtype=np.float64))
    lap_phi_k = -k2 * phik  # Fourier-space Laplacian of phi
    res = np.max(np.abs((lap_phi_k - dk)[resolved])) / np.max(np.abs(dk))
    assert res < 1e-4


def test_gauss_law_div_acceleration():
    # div g = div(-grad phi) = -lap phi = -delta.
    delta = F.gaussian_random_field(BOX, COSMO, seed=3, z=0.0)
    gx, gy, gz = FO.acceleration_field(delta, BOX)
    div = L.divergence(BOX, gx, gy, gz)

    a = np.asarray(div, dtype=np.float64).ravel()
    b = -np.asarray(delta, dtype=np.float64).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.99

    N = BOX.n_mesh
    resolved, _, _, _ = _resolved_mask(N)
    divk = np.fft.rfftn(a.reshape(N, N, N))
    dk = np.fft.rfftn(np.asarray(delta, dtype=np.float64))
    res = np.max(np.abs((divk + dk)[resolved])) / np.max(np.abs(dk))
    assert res < 1e-4


def test_acceleration_equals_zeldovich():
    # The headline identity: the linear gravitational force IS the Zel'dovich
    # displacement (same i k / k^2 kernel on the same realization).
    delta = F.gaussian_random_field(BOX, COSMO, seed=0, z=0.0)
    g = FO.acceleration_field(delta, BOX)
    psi = L.zeldovich_displacement(BOX, COSMO, seed=0)
    for gc, pc in zip(g, psi):
        scale = float(mx.max(mx.abs(pc)))
        rel = float(mx.max(mx.abs(gc - pc))) / scale
        assert rel < 1e-5


def test_single_mode_analytic():
    # delta = cos(k0 x) along x with k0 = m * k_fundamental (an exact resolved
    # mode). Then phi = -delta / k0^2 and g_x = -sin(k0 x) / k0 analytically.
    N = BOX.n_mesh
    m = 2
    k0 = m * BOX.k_fundamental
    x = (np.arange(N) * BOX.cell_size)[:, None, None]
    delta_np = np.broadcast_to(np.cos(k0 * x), (N, N, N)).astype(np.float32)
    delta = mx.array(delta_np)

    phi = np.asarray(FO.potential(delta, BOX), dtype=np.float64)
    phi_exact = -np.cos(k0 * x) / k0**2
    assert np.max(np.abs(phi - phi_exact)) / (1.0 / k0**2) < 1e-4

    gx, _, _ = FO.acceleration_field(delta, BOX)
    gx = np.asarray(gx, dtype=np.float64)
    gx_exact = -np.sin(k0 * x) / k0
    assert np.max(np.abs(gx - gx_exact)) / (1.0 / k0) < 1e-4


def test_acceleration_field_zero_mean():
    # k = 0 is killed, so the mesh acceleration has exactly zero mean (no DC
    # force on the box). This is rigorous to fp32 round-off.
    delta = F.gaussian_random_field(BOX, COSMO, seed=7, z=0.0)
    for g in FO.acceleration_field(delta, BOX):
        scale = float(mx.sqrt(mx.mean(g**2)))
        assert abs(float(mx.mean(g))) / scale < 1e-4


def test_zero_net_force_on_particles():
    # Summed over particles the force vanishes by parity: the kernel i k / k^2
    # is odd, |delta_k|^2 is even, so the net force is zero up to the Nyquist
    # plane (where k and -k coincide) and fp32 round-off. The fp64 reductions
    # are converted to numpy so the ratio is taken on the CPU, not the GPU.
    pos = L.lpt_positions(BOX, COSMO, seed=0, z=0.0)
    f = FO.forces_on_particles(pos, BOX)
    net = np.asarray(P.accurate_sum(f, axis=0))  # fp64 reduction, on the CPU
    total_abs = np.asarray(P.accurate_sum(mx.abs(f), axis=0))
    rel = float(np.max(np.abs(net) / total_abs))
    assert rel < 1e-6


def test_forces_differentiable():
    # mx.grad flows through density_contrast -> FFT solve -> cic_read to the
    # particle positions. Check it with a random directional finite difference:
    # projecting the gradient on a random direction and averaging over all
    # particles washes out the isolated cell-boundary kinks (CIC is C0, not C1)
    # that make a single particle's FD unreliable. The residual floor is fp32
    # force round-off, which (as for painting) makes the FD *worse* at small h.
    pos = L.lpt_positions(BOX, COSMO, seed=2, z=0.0)
    g = mx.grad(lambda p: mx.sum(FO.forces_on_particles(p, BOX) ** 2))(pos)
    mx.eval(g)
    assert bool(mx.all(mx.isfinite(g)))

    def loss64(p):  # fp64 reduction; the per-particle forces stay fp32
        f = FO.forces_on_particles(p, BOX)
        return float(np.sum(np.asarray(f, np.float64) ** 2))

    rng = np.random.default_rng(0)
    v = rng.standard_normal(tuple(pos.shape)).astype(np.float32)
    v /= np.linalg.norm(v)
    vmx = mx.array(v)
    h = 0.05 * BOX.cell_size
    fd = (loss64(pos + h * vmx) - loss64(pos - h * vmx)) / (2.0 * h)
    ad = float(mx.sum(g * vmx))
    assert abs(fd - ad) / abs(ad) < 0.05
