"""Tests for mbody.lpt: the Zel'dovich (1LPT) displacement and particle layout.

The core check is the defining identity div(Psi_1) = -delta_0. It holds exactly
for every resolved Fourier mode; the only residual is on the Nyquist plane,
where a real field's spectral gradient is fundamentally ambiguous (it spreads
across real space as plane waves, so we validate in Fourier on resolved modes
and via the real-space correlation coefficient).
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import cosmology as C
from mbody import fields as F
from mbody import ic as IC
from mbody import lpt as L

COSMO = Cosmology()
BOX = BoxConfig(box_size=128.0, n_mesh=64, n_particles=64)


def test_lagrangian_grid():
    q = L.lagrangian_grid(BOX)
    mx.eval(q)
    assert tuple(q.shape) == (BOX.n_mesh**3, 3)
    assert float(mx.min(q)) == 0.0
    # Last grid point sits one cell short of the box edge.
    assert abs(float(mx.max(q)) - (BOX.box_size - BOX.cell_size)) < 1e-3


def test_zeldovich_divergence_identity():
    psi = L.zeldovich_displacement(BOX, COSMO, seed=0)
    # The displacement is built from ic.linear_density (f_NL = 0 here), so the
    # identity div(Psi) = -delta is checked against that exact source.
    delta0 = IC.linear_density(BOX, COSMO, seed=0, z=0.0, f_NL=0.0)
    div = L.divergence(BOX, *psi)

    a = np.asarray(div, dtype=np.float64).ravel()
    b = -np.asarray(delta0, dtype=np.float64).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.99

    # Rigorous: every resolved (non-Nyquist) Fourier mode satisfies div = -delta.
    N = BOX.n_mesh
    nyq = N // 2
    divk = np.fft.rfftn(np.asarray(div, dtype=np.float64))
    dk = np.fft.rfftn(np.asarray(delta0, dtype=np.float64))
    kxi = np.fft.fftfreq(N)
    kzi = np.fft.rfftfreq(N)
    KX, KY, KZ = np.meshgrid(kxi, kxi, kzi, indexing="ij")
    resolved = (KX != kxi[nyq]) & (KY != kxi[nyq]) & (KZ != kzi[nyq])
    res = np.max(np.abs((divk + dk)[resolved])) / np.max(np.abs(dk))
    assert res < 1e-4


def test_displacement_real_and_finite():
    psi = L.zeldovich_displacement(BOX, COSMO, seed=1)
    for comp in psi:
        mx.eval(comp)
        assert comp.dtype == mx.float32
        assert bool(mx.all(mx.isfinite(comp)))


def test_growth_scaling():
    # Displacement amplitude scales with the growth factor D(z). Use z=1 and
    # z=9 where the max displacement stays below L/2, so minimum-image is exact.
    psi = L.zeldovich_displacement(BOX, COSMO, seed=0)
    grid = np.asarray(L.lagrangian_grid(BOX), dtype=np.float64)
    Lh = BOX.box_size

    def rms_disp(z):
        pos = np.asarray(L.displace(BOX, psi, C.growth_factor(z, COSMO)), np.float64)
        disp = (pos - grid + Lh / 2) % Lh - Lh / 2  # minimum image
        return np.sqrt(np.mean(disp**2))

    ratio = rms_disp(1.0) / rms_disp(9.0)
    expected = C.growth_factor(1.0, COSMO) / C.growth_factor(9.0, COSMO)
    assert abs(ratio / expected - 1.0) < 1e-3


def test_positions_periodic_and_deterministic():
    a = L.lpt_positions(BOX, COSMO, seed=5, z=0.0)
    b = L.lpt_positions(BOX, COSMO, seed=5, z=0.0)
    c = L.lpt_positions(BOX, COSMO, seed=6, z=0.0)
    mx.eval(a, b, c)
    assert float(mx.min(a)) >= 0.0 and float(mx.max(a)) < BOX.box_size
    assert float(mx.max(mx.abs(a - b))) == 0.0
    assert float(mx.max(mx.abs(a - c))) > 0.0


def test_zero_growth_returns_grid():
    psi = L.zeldovich_displacement(BOX, COSMO, seed=0)
    pos = L.displace(BOX, psi, 0.0)
    grid = L.lagrangian_grid(BOX)
    mx.eval(pos, grid)
    assert float(mx.max(mx.abs(pos - grid))) == 0.0


def test_npart_must_equal_nmesh():
    import pytest

    bad = BoxConfig(box_size=128.0, n_mesh=64, n_particles=32)
    with pytest.raises(ValueError):
        L.lpt_positions(bad, COSMO, seed=0, z=0.0)


# --- 2LPT: second-order displacement -----------------------------------------


def test_lpt2_divergence_identity():
    # div(Psi2) = -delta2, the second-order analogue of the Zel'dovich identity.
    # delta2 (a quadratic field) piles power near Nyquist, so the real-space L2
    # residual is larger than for Psi1; the operator itself is exact on every
    # resolved mode, which is what the Fourier check below pins.
    delta = IC.linear_density(BOX, COSMO, seed=0, z=0.0, f_NL=0.0)
    delta2 = L.lpt2_source(delta, BOX)
    div = L.divergence(BOX, *L.second_order_displacement(BOX, COSMO, seed=0))

    a = np.asarray(div, dtype=np.float64).ravel()
    b = -np.asarray(delta2, dtype=np.float64).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.99

    N = BOX.n_mesh
    nyq = N // 2
    divk = np.fft.rfftn(np.asarray(div, dtype=np.float64))
    d2k = np.fft.rfftn(np.asarray(delta2, dtype=np.float64))
    kxi = np.fft.fftfreq(N)
    kzi = np.fft.rfftfreq(N)
    KX, KY, KZ = np.meshgrid(kxi, kxi, kzi, indexing="ij")
    # Exclude the Nyquist planes (gradient ambiguity) and the k=0 mode: delta2
    # has a nonzero mean, but the i k / k^2 operator gives Psi2 zero mean, so the
    # identity holds only for the fluctuating part.
    dc = (KX == 0) & (KY == 0) & (KZ == 0)
    resolved = (KX != kxi[nyq]) & (KY != kxi[nyq]) & (KZ != kzi[nyq]) & ~dc
    res = np.max(np.abs((divk + d2k)[resolved])) / np.max(np.abs(d2k[resolved]))
    assert res < 1e-4


def test_lpt2_source_differentiable_in_fnl():
    # The 2LPT source is quadratic in the linear density, so f_NL flows through
    # it (pure FFT path -- no CIC floor). mx.grad matches a central difference.
    box = BoxConfig(box_size=200.0, n_mesh=32, n_particles=32)
    k_bins = [2.0 * box.k_fundamental]

    def loss(f):
        d2 = L.lpt2_source(
            IC.linear_density(box, COSMO, seed=2, z=0.0, f_NL=f, backend="eh98"), box
        )
        return F.band_power(d2, box, k_bins)[0]

    g = float(mx.grad(loss)(mx.array(50.0)))
    h = 5.0
    fd = (float(loss(mx.array(50.0 + h))) - float(loss(mx.array(50.0 - h)))) / (2 * h)
    assert abs(g / fd - 1.0) < 1e-2


def test_displacement_wrapper_orders():
    import pytest

    one = L.displacement(BOX, COSMO, order=1, seed=0)
    two = L.displacement(BOX, COSMO, order=2, seed=0)
    assert len(one) == 1 and len(two) == 2
    # order-1 component of the wrapper is exactly the Zel'dovich displacement.
    psi1 = L.zeldovich_displacement(BOX, COSMO, seed=0)
    assert float(mx.max(mx.abs(two[0][0] - psi1[0]))) == 0.0
    with pytest.raises(ValueError):
        L.displacement(BOX, COSMO, order=3, seed=0)
