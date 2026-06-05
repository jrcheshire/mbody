"""Tests for the Scoccimarro FFT bispectrum estimator (mbody.fields) and the
local-f_NL template (mbody.ic).

The normalization is checked two independent ways. A deterministic plane-wave
field with a closed triangle has an analytic triple product, so the estimator
must return B = L^6 a^3 / (4 n_tri) exactly (no cosmic variance) -- a direct
test of the V^2/N^9 prefactor; n_tri is cross-checked against a brute-force
triangle count. The statistical path is checked with the matched-phase
antisymmetric combination [B(+f) - B(-f)]/2, which cancels the Gaussian
cosmic-variance term and isolates the tree signal, so a handful of seeds on a
small box already recover the binned template (squeezed c_cal ~ 1; measured
~1.006 over 16 seeds in dev). Tolerances here were measured with
scripts/probe_bispectrum.py, not guessed.
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import fields as F
from mbody import ic as IC

COSMO = Cosmology()
# Small box for the statistical tests; matches test_fields' resolution.
BOX = BoxConfig(box_size=128.0, n_mesh=64, n_particles=64)


def _plane_wave_field(N, m1, m2, m3, a):
    """a[cos(q1.x) + cos(q2.x) + cos(q3.x)] on an N^3 grid, m the fftfreq index
    vectors (so q = (2 pi / L) m and the phase per cell is (2 pi / N) m)."""
    ax = (2.0 * np.pi / N) * np.arange(N)

    def cos(m):
        return np.cos(
            m[0] * ax[:, None, None]
            + m[1] * ax[None, :, None]
            + m[2] * ax[None, None, :]
        )

    return a * (cos(m1) + cos(m2) + cos(m3))


def _brute_n_tri(N, centers, dk_mult=1.0):
    """Count full-grid mode-triplets (k1, k2, k3) with |k_i| in shell i and
    k1 + k2 + k3 = 0 (mod N), independently of the FFT J-product."""
    idx = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing="ij")
    mag = np.sqrt(gx**2 + gy**2 + gz**2)
    masks = [(mag >= c - 0.5 * dk_mult) & (mag < c + 0.5 * dk_mult) for c in centers]
    m1 = np.stack([gx[masks[0]], gy[masks[0]], gz[masks[0]]], axis=1)
    m2 = np.stack([gx[masks[1]], gy[masks[1]], gz[masks[1]]], axis=1)
    count = 0
    for a in m1:
        c3 = (-(a + m2)) % N
        signed = np.where(c3 <= N // 2, c3, c3 - N)
        mag3 = np.sqrt((signed**2).sum(axis=1))
        count += int(((mag3 >= centers[2] - 0.5) & (mag3 < centers[2] + 0.5)).sum())
    return count


def test_normalization_deterministic():
    # Closed 3-4-5 triangle of distinct-magnitude modes: each shell holds one
    # populated mode pair, so I_i = a cos(qi x) and sum_x I1 I2 I3 = a^3 N^3 / 4,
    # giving B = L^6 a^3 / (4 n_tri) exactly. Tests alpha + data path + count.
    N, L, a = 16, 200.0, 0.5
    box = BoxConfig(box_size=L, n_mesh=N, n_particles=N)
    kf = box.k_fundamental
    field = _plane_wave_field(N, (3, 0, 0), (0, 4, 0), (-3, -4, 0), a)
    B, n_tri = F.bispectrum(
        mx.array(field.astype(np.float32)), box, [(3 * kf, 4 * kf, 5 * kf)]
    )

    n_brute = _brute_n_tri(N, [3, 4, 5])
    assert abs(n_tri[0] - n_brute) < 0.5  # J-product count == brute force
    predicted = L**6 * a**3 / (4.0 * n_brute)
    assert abs(B[0] / predicted - 1.0) < 1e-4  # measured ~5e-8 (float32 floor)


def test_n_tri_positive_and_counts():
    # The estimator's mode-triplet count is a positive integer for a closing
    # config and zero for one that violates the triangle inequality at bin level.
    kf = BOX.k_fundamental
    closing = (2 * kf, 8 * kf, 8 * kf)
    open_tri = (1 * kf, 2 * kf, 8 * kf)  # 1 + 2 < 8: cannot close
    delta = IC.linear_density(BOX, COSMO, seed=0)
    _, n_tri = F.bispectrum(delta, BOX, [closing, open_tri])
    assert n_tri[0] > 0
    # near-integer up to the float32 round-off of the J-product (rel ~1e-6)
    assert abs(n_tri[0] - round(n_tri[0])) < 1e-5 * n_tri[0]
    assert abs(n_tri[1]) < 1e-3  # non-closing: zero up to float32 round-off


def _squeezed_triangles():
    kf = BOX.k_fundamental
    k_short = round(8 * kf, 12)
    return [(round(n * kf, 12), k_short, k_short) for n in (2, 3, 4)]


def _antisym_signal(triangles, f_NL, nseed, seed0=2000):
    """[B(+f) - B(-f)]/2 averaged over matched-phase seed pairs. Cancels the
    Gaussian cosmic-variance term, isolating the tree signal."""
    acc = np.zeros(len(triangles))
    for s in range(nseed):
        dp = IC.linear_density(BOX, COSMO, seed=seed0 + s, f_NL=f_NL)
        dm = IC.linear_density(BOX, COSMO, seed=seed0 + s, f_NL=-f_NL)
        bp, _ = F.bispectrum(dp, BOX, triangles)
        bm, _ = F.bispectrum(dm, BOX, triangles)
        acc += 0.5 * (bp - bm)
    return acc / nseed


def test_squeezed_matches_binned_template():
    # The headline closure: the cosmic-variance-cancelled squeezed signal
    # recovers the bin-averaged template with unit calibration. Measured
    # c_cal ~ 1.006 (16 seeds); allow a measured band, do not relax further.
    tris = _squeezed_triangles()
    f_NL = 1000.0
    signal = _antisym_signal(tris, f_NL, nseed=16)
    tmpl = IC.local_bispectrum_binned(BOX, COSMO, tris, f_NL)
    c_cal = float(np.sum(signal * tmpl) / np.sum(tmpl**2))
    assert 0.85 < c_cal < 1.15
    assert np.all(np.abs(signal / tmpl - 1.0) < 0.30)  # per-bin (measured <0.15)


def test_squeezed_divergence():
    # The scale-dependent-bias signature: B rises as k_long shrinks (toward the
    # 1/M_long ~ 1/k_long^2 divergence). Check monotonic decrease with k_long in
    # both the signal and the binned template.
    tris = _squeezed_triangles()  # k_long increasing: 2,3,4 kf
    f_NL = 1000.0
    signal = _antisym_signal(tris, f_NL, nseed=16)
    tmpl = IC.local_bispectrum_binned(BOX, COSMO, tris, f_NL)
    assert signal[0] > signal[1] > signal[2] > 0
    assert tmpl[0] > tmpl[1] > tmpl[2] > 0


def test_gaussian_field_zero_bispectrum():
    # A Gaussian (f_NL = 0) field has zero tree bispectrum: the seed-mean B is
    # consistent with zero, i.e. small next to the f_NL = 1000 signal.
    tris = _squeezed_triangles()
    nseed = 16
    acc = np.zeros(len(tris))
    for s in range(nseed):
        d = IC.linear_density(BOX, COSMO, seed=2000 + s, f_NL=0.0)
        b, _ = F.bispectrum(d, BOX, tris)
        acc += b
    mean0 = acc / nseed
    signal = _antisym_signal(tris, 1000.0, nseed=nseed)
    assert np.all(np.abs(mean0) < 0.6 * np.abs(signal))  # measured <0.3


def test_template_odd_and_linear_in_fnl():
    # Both templates are exactly linear (hence odd) in f_NL: pure-theory check.
    tris = _squeezed_triangles()
    for tmpl in (
        lambda f: IC.local_bispectrum_template(tris, COSMO, f),
        lambda f: IC.local_bispectrum_binned(BOX, COSMO, tris, f),
    ):
        b1 = tmpl(1.0)
        assert np.allclose(tmpl(250.0), 250.0 * b1, rtol=1e-6)
        assert np.allclose(tmpl(-100.0), -100.0 * b1, rtol=1e-6)
        assert np.all(b1 > 0)  # f_NL > 0 gives positive squeezed B


def test_bispectrum_autodiff_matches_fd():
    # mx.grad flows through bispectrum_single -> linear_density -> f_NL and
    # matches a finite difference (measured rel.err ~5e-5).
    tri = _squeezed_triangles()[1]

    def Bfun(f):
        d = IC.linear_density(BOX, COSMO, seed=2000, f_NL=f)
        return F.bispectrum_single(d, BOX, tri)

    g = float(mx.grad(Bfun)(mx.array(150.0)))
    h = 5.0
    fd = (float(Bfun(mx.array(150.0 + h))) - float(Bfun(mx.array(150.0 - h)))) / (2 * h)
    assert abs(g - fd) / abs(fd) < 1e-2
