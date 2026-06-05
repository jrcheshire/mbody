"""Gaussian random fields on the mesh, and the P(k) estimator that measures them.

This is the first module that runs on the GPU (MLX/Metal, float32). It turns the
linear power spectrum P(k) from mbody.cosmology into an actual realization of a
random universe -- a 3D density-contrast field delta(x) on a periodic mesh --
and provides the estimator that measures P(k) back out of a field.

How the generation works (the standard recipe):

1. Draw white noise w(x) ~ N(0, 1) independently in each real-space cell.
2. Forward transform: w_k = rfftn(w). Because w is real, w_k automatically has
   the Hermitian symmetry a real field requires -- no manual bookkeeping.
3. Colour the noise: multiply each mode by sqrt(P(k) / V_cell), so the variance
   of mode k becomes the target power.
4. Inverse transform back to real space: delta(x) = irfftn(coloured w_k).

The k = 0 (mean) mode is set to zero, so the field has zero mean overdensity.

Conventions and precision: the k-grid and the per-mode amplitude are built in
float64 on the CPU (a precision island -- coordinates and P(k) want the
accuracy), then cast to float32 for the GPU FFT hot loop. The estimator
normalization V/N^6 is the inverse of the generator's, so a generated field
measures back the P(k) it was built from; this is checked in the tests.

The mesh is indexed [x, y, z] with rfftn taken over all three axes, so the last
axis has N//2 + 1 entries (the non-redundant half of a real transform).
"""

import mlx.core as mx
import numpy as np

from mbody import cosmology as C
from mbody import precision as P


def k_grid(box):
    """Fourier-space wavenumber grid for an rfftn of an n_mesh^3 box.

    Returns (k_1d, kz_1d, k_mag), all float64 in h/Mpc:
    - k_1d: the N full-axis wavenumbers (for the x and y axes), ordered as
      numpy's fftfreq: 0, +, ..., Nyquist, ..., -.
    - kz_1d: the N//2 + 1 non-negative wavenumbers of the rfft (last) axis.
    - k_mag: |k| on the (N, N, N//2 + 1) half-grid.
    """
    N, L = box.n_mesh, box.box_size
    d = L / N  # cell size, Mpc/h
    k_1d = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    kz_1d = 2.0 * np.pi * np.fft.rfftfreq(N, d=d)
    k_mag = np.sqrt(
        k_1d[:, None, None] ** 2 + k_1d[None, :, None] ** 2 + kz_1d[None, None, :] ** 2
    )
    return k_1d, kz_1d, k_mag


def gaussian_random_field(box, cosmo, seed=0, z=0.0, backend="camb"):
    """Generate a Gaussian density-contrast field delta(x) with power P(k).

    Returns a real float32 MLX array of shape (n_mesh, n_mesh, n_mesh) with zero
    mean and the linear power spectrum of `cosmo` at redshift `z`.
    """
    N, L = box.n_mesh, box.box_size
    _, _, k_mag = k_grid(box)

    # P(k) and per-mode amplitude in the float64 CPU island.
    Pk = C.linear_power(k_mag.ravel(), cosmo, z=z, backend=backend)
    Pk = Pk.reshape(k_mag.shape)
    Pk[0, 0, 0] = 0.0  # no mean overdensity (kill the DC mode)
    v_cell = (L / N) ** 3
    amp = np.sqrt(Pk / v_cell).astype(np.float32)

    # White noise -> colour -> real space, all float32 on the GPU.
    w = mx.random.normal((N, N, N), key=mx.random.key(seed))
    delta_k = mx.fft.rfftn(w) * mx.array(amp)
    delta = mx.fft.irfftn(delta_k, s=(N, N, N), axes=(0, 1, 2))
    return P.as_real(delta)


def power_spectrum(delta, box, dk=None, kmin=None, kmax=None):
    """Estimate the power spectrum P(k) of a real field delta(x).

    Bins |delta_k|^2 in spherical |k| shells with the estimator normalization
    P(k) = (V / N^6) |delta_k|^2, where V = L^3. Returns (k_centers, P_k,
    n_modes) as float64 numpy arrays, excluding the k = 0 mode.
    """
    N, L = box.n_mesh, box.box_size
    delta_k = mx.fft.rfftn(delta)
    power_modes = np.asarray(mx.abs(delta_k) ** 2, dtype=np.float64)
    _, _, k_mag = k_grid(box)

    power_modes *= L**3 / N**6  # estimator normalization (inverse of generator)

    kf = box.k_fundamental
    if dk is None:
        dk = kf
    if kmin is None:
        kmin = 0.5 * kf  # first bin straddles the fundamental; excludes k=0
    if kmax is None:
        kmax = box.k_nyquist
    edges = np.arange(kmin, kmax + dk, dk)

    km = k_mag.ravel()
    pm = power_modes.ravel()
    sum_p, _ = np.histogram(km, bins=edges, weights=pm)
    counts, _ = np.histogram(km, bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])

    good = counts > 0
    return centers[good], sum_p[good] / counts[good], counts[good]


def _shell_mask(k_mag, k_lo, k_hi):
    """Boolean mask selecting the |k| shell [k_lo, k_hi) on the rfftn half-grid.

    Half-open so adjacent shells of width dk tile without double-counting a mode
    on a bin edge. k_mag is the float64 (N, N, N//2 + 1) array from k_grid.
    """
    return (k_mag >= k_lo) & (k_mag < k_hi)


def _band_fields(delta, box, centers, dk):
    """Real-space band-filtered fields for the Scoccimarro bispectrum estimator.

    For each shell center kc in `centers` builds two real (N, N, N) MLX fields:

        I[kc](x) = irfftn(delta_k * Theta),   J[kc](x) = irfftn(Theta),

    where Theta is the indicator of the shell [kc - dk/2, kc + dk/2) and
    delta_k = rfftn(delta). I carries the data; J is the "field replaced by
    ones" field whose triple product counts closeable triangles. Returns
    (I_fields, J_fields) dicts keyed by the exact float center, so callers must
    look configs up with the same float objects they passed in. Differentiable
    in `delta` (the masks are constants), so it backs both the numeric estimator
    and the autodiff entry point below.
    """
    N = box.n_mesh
    _, _, k_mag = k_grid(box)
    delta_k = mx.fft.rfftn(delta)
    I_fields, J_fields = {}, {}
    for kc in centers:
        mask = _shell_mask(k_mag, kc - 0.5 * dk, kc + 0.5 * dk)
        theta = P.as_complex(mx.array(mask.astype(np.float32)))
        I_fields[kc] = mx.fft.irfftn(delta_k * theta, s=(N, N, N), axes=(0, 1, 2))
        J_fields[kc] = mx.fft.irfftn(theta, s=(N, N, N), axes=(0, 1, 2))
    return I_fields, J_fields


def bispectrum(delta, box, triangles, dk=None):
    """Binned bispectrum B(k1, k2, k3) via the Scoccimarro FFT estimator.

    For each triangle the estimator is

        B = (V^2 / N^9) * sum_x I1 I2 I3 / sum_x J1 J2 J3,   V = L^3,

    with I, J the band-filtered fields from `_band_fields`. The V^2/N^9 prefactor
    is exact in the same DFT convention that fixes power_spectrum's V/N^6 (the
    triangle-count cancels between data and the J normalization), so a correct
    field returns B with no free constant. Real-space sums are reduced in
    float64 on the CPU stream, because the triple product of zero-mean
    band-limited fields cancels heavily and float32 accumulation is not safe.

    Parameters
    ----------
    delta : real (N, N, N) field.
    triangles : sequence of (k1, k2, k3) shell-center wavenumbers (h/Mpc). Fully
        general -- squeezed, equilateral, anything that closes; configs whose
        bins cannot form a triangle return n_tri = 0 (B is then meaningless).
    dk : shell width (h/Mpc); defaults to the fundamental, matching
        power_spectrum.

    Returns (B, n_tri) as float64 numpy arrays, one entry per triangle. n_tri is
    the integer count of mode-triplets in the bin (a sampling diagnostic: small
    counts are noisy).
    """
    N, L = box.n_mesh, box.box_size
    if dk is None:
        dk = box.k_fundamental
    centers = sorted({float(k) for tri in triangles for k in tri})
    I_fields, J_fields = _band_fields(delta, box, centers, dk)

    alpha = L**6 / N**9
    B = np.empty(len(triangles), dtype=np.float64)
    n_tri = np.empty(len(triangles), dtype=np.float64)
    for t, (k1, k2, k3) in enumerate(triangles):
        k1, k2, k3 = float(k1), float(k2), float(k3)
        S = float(P.accurate_sum(I_fields[k1] * I_fields[k2] * I_fields[k3]))
        norm = float(P.accurate_sum(J_fields[k1] * J_fields[k2] * J_fields[k3]))
        B[t] = alpha * S / norm
        n_tri[t] = N**6 * norm  # mode-triplet count (J product = count / N^6)
    return B, n_tri


def bispectrum_single(delta, box, triangle, dk=None):
    """Single-triangle bispectrum as a differentiable MLX scalar.

    Same estimator as `bispectrum` for one (k1, k2, k3), but reduced with a
    float32 GPU sum and returned as a 0-d MLX array so mx.grad flows through to
    delta (and, via ic.linear_density, to f_NL). Use `bispectrum` for accurate
    numeric values; use this where a differentiable summary statistic is needed.
    """
    N, L = box.n_mesh, box.box_size
    if dk is None:
        dk = box.k_fundamental
    k1, k2, k3 = float(triangle[0]), float(triangle[1]), float(triangle[2])
    I_fields, J_fields = _band_fields(delta, box, sorted({k1, k2, k3}), dk)
    S = mx.sum(I_fields[k1] * I_fields[k2] * I_fields[k3])
    norm = mx.sum(J_fields[k1] * J_fields[k2] * J_fields[k3])
    return (L**6 / N**9) * S / norm


def _bin_masks(box, k_bins, dk):
    """Boolean shell masks (as float32 MLX arrays) and their mode counts for a
    list of bin centers, on the rfftn half-grid. Shared by the band powers."""
    _, _, k_mag = k_grid(box)
    masks, counts = [], []
    for kc in k_bins:
        m = _shell_mask(k_mag, kc - 0.5 * dk, kc + 0.5 * dk).astype(np.float32)
        masks.append(mx.array(m))
        counts.append(float(m.sum()))
    return masks, counts


def band_power(delta, box, k_bins, dk=None):
    """Differentiable band power P(k) of a real field over fixed |k| shells.

    P[b] = (V / N^6) sum_{shell b} |delta_k|^2 / count_b -- the same
    normalization as power_spectrum, but with fixed boolean masks (not a numpy
    histogram), so it returns an MLX vector and mx.grad / mx.jvp flow through to
    delta. `k_bins` are shell centers (h/Mpc); `dk` defaults to the fundamental.
    """
    N, L = box.n_mesh, box.box_size
    if dk is None:
        dk = box.k_fundamental
    masks, counts = _bin_masks(box, k_bins, dk)
    p_modes = mx.abs(mx.fft.rfftn(delta)) ** 2
    norm = L**3 / N**6
    bins = [norm * mx.sum(mk * p_modes) / c for mk, c in zip(masks, counts)]
    return mx.stack(bins)


def cross_power(delta_a, delta_b, box, k_bins, dk=None):
    """Differentiable cross power P_ab(k) of two real fields over |k| shells.

    P_ab[b] = (V / N^6) sum_{shell b} Re(a_k conj(b_k)) / count_b. Returns an MLX
    vector; cross_power(d, d) equals band_power(d). Same conventions as
    band_power.
    """
    N, L = box.n_mesh, box.box_size
    if dk is None:
        dk = box.k_fundamental
    masks, counts = _bin_masks(box, k_bins, dk)
    cross = mx.real(mx.fft.rfftn(delta_a) * mx.conj(mx.fft.rfftn(delta_b)))
    norm = L**3 / N**6
    bins = [norm * mx.sum(mk * cross) / c for mk, c in zip(masks, counts)]
    return mx.stack(bins)
