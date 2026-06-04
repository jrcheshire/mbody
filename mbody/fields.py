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
