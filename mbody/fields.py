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


def cic_window(box):
    """CIC mass-assignment window W(k) on the rfftn half-grid (float64).

    Painting particles with cloud-in-cell convolves the density with a triangular
    cloud, multiplying each Fourier mode by W(k) = prod_i sinc^2(k_i / (2 k_nyq))
    (the order-1 / CIC assignment window; sinc(x) = sin(pi x)/(pi x) = np.sinc).
    A particle-painted power spectrum is therefore suppressed by W(k)^2 relative to
    the true field -- negligible at low k, ~50% near k_nyq. Divide a measured
    particle power by W^2 to deconvolve it (power_spectrum's deconvolve_cic flag);
    a grid field (e.g. ic.linear_density) carries no such window. Returned shape
    matches k_grid's k_mag, (N, N, N//2 + 1).
    """
    k_1d, kz_1d, _ = k_grid(box)
    knyq = box.k_nyquist
    wx = np.sinc(k_1d / (2.0 * knyq)) ** 2
    wz = np.sinc(kz_1d / (2.0 * knyq)) ** 2
    return wx[:, None, None] * wx[None, :, None] * wz[None, None, :]


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


def power_spectrum(delta, box, dk=None, kmin=None, kmax=None, deconvolve_cic=False):
    """Estimate the power spectrum P(k) of a real field delta(x).

    Bins |delta_k|^2 in spherical |k| shells with the estimator normalization
    P(k) = (V / N^6) |delta_k|^2, where V = L^3. Returns (k_centers, P_k,
    n_modes) as float64 numpy arrays, excluding the k = 0 mode. With
    deconvolve_cic=True each mode is divided by the CIC window W(k)^2 (see
    cic_window) before binning -- use it for a particle-painted (CIC) field to
    remove the mass-assignment suppression; leave it off for a grid field.
    """
    N, L = box.n_mesh, box.box_size
    delta_k = mx.fft.rfftn(delta)
    power_modes = np.asarray(mx.abs(delta_k) ** 2, dtype=np.float64)
    _, _, k_mag = k_grid(box)

    power_modes *= L**3 / N**6  # estimator normalization (inverse of generator)
    if deconvolve_cic:
        power_modes /= cic_window(box) ** 2

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


# --- Redshift-space multipoles (anisotropic P(k)) ----------------------------
#
# In redshift space P(k) is anisotropic, P(k, mu) with mu = k_los/|k| the cosine
# to the line of sight, and is summarized by its Legendre multipoles P_ell(k).
# For a periodic box with a *fixed* Cartesian line of sight, the plane-parallel
# estimator (the Yamamoto pair-LOS form is needed only for survey geometry) is
#
#     P_ell(k) = (2 ell + 1) * mean_{shell}[ |delta_k|^2 L_ell(mu) ],
#
# the same V/N^6 normalization and rfftn half-grid shells as band_power, with an
# extra Legendre weight per mode. Even multipoles (ell = 0, 2, 4) depend only on
# |mu|, so the half-grid (which samples |mu| completely by Hermitian symmetry) is
# sufficient. ell = 0 reduces exactly to band_power.
#
# Two finite-mesh subtleties, both validated rather than assumed (probe_rsd.py):
#   * Line of sight MUST be a full fft axis (0 or 1), not the rfft axis (2): on
#     the half-grid the kz = 0 plane has mu = 0 for a whole plane of modes, which
#     skews the discrete shell-average of the Legendre weights badly.
#   * A thin shell samples solid angle non-uniformly, so the raw estimator leaks
#     power between multipoles (<L_ell>_shell != 0). multipole_decoupling inverts
#     that to recover the continuum multipoles for interpretation/validation; the
#     Fisher uses the raw multipoles (a forecast is invariant to that linear map).
#     At toy box sizes P0 and P2 recover Kaiser well; P4 is noise-dominated (its
#     signal is ~0.04 P0).


def _mu_grid(box, los_axis=0):
    """mu = k_los/|k| on the rfftn half-grid (float64, k = 0 -> 0).

    `los_axis` picks the line of sight. Use a FULL fft axis (0 or 1); the rfft
    (last) axis 2 is supported but skews the discrete multipole average (see the
    module note above). Returned shape matches k_grid's k_mag, (N, N, N//2 + 1).
    Only |mu| matters for even multipoles, and k_los is taken non-negative, so mu
    lands in [0, 1].
    """
    k_1d, kz_1d, k_mag = k_grid(box)
    if los_axis == 0:
        k_los = np.abs(k_1d)[:, None, None]
    elif los_axis == 1:
        k_los = np.abs(k_1d)[None, :, None]
    elif los_axis == 2:
        k_los = kz_1d[None, None, :]
    else:
        raise ValueError(f"los_axis must be 0, 1 or 2 (got {los_axis})")
    mu = np.zeros_like(k_mag)
    nz = k_mag > 0
    mu[nz] = np.broadcast_to(k_los, k_mag.shape)[nz] / k_mag[nz]
    return mu


def _legendre_weight(mu, ell):
    """Legendre polynomial L_ell(mu) on the grid (float64). ell in {0, 2, 4}."""
    mu2 = mu**2
    if ell == 0:
        return np.ones_like(mu)
    if ell == 2:
        return 0.5 * (3.0 * mu2 - 1.0)
    if ell == 4:
        return 0.125 * (35.0 * mu2**2 - 30.0 * mu2 + 3.0)
    raise ValueError(f"ell must be 0, 2 or 4 (got {ell})")


def band_power_multipole(delta, box, k_bins, ell, los_axis=0, dk=None):
    """Differentiable redshift-space multipole band power P_ell(k) (raw).

    P_ell[b] = (2 ell + 1) (V / N^6) sum_{shell b} L_ell(mu) |delta_k|^2 / count_b
    -- the anisotropic sibling of band_power, with mu = k_los/|k| (`los_axis`) and
    the closed-form Legendre weight. ell in {0, 2, 4}; ell = 0 is band_power.
    Differentiable in `delta` (the mu / Legendre grids are constants), so it backs
    the redshift-space Fisher data vector. This is the RAW estimator (it carries
    the discrete-shell multipole leakage); the Fisher is invariant to undoing it,
    so the leakage is corrected only in power_multipoles for interpretation.
    Returns an MLX vector.
    """
    N, L = box.n_mesh, box.box_size
    if dk is None:
        dk = box.k_fundamental
    masks, counts = _bin_masks(box, k_bins, dk)
    leg = mx.array(_legendre_weight(_mu_grid(box, los_axis), ell).astype(np.float32))
    weighted = leg * mx.abs(mx.fft.rfftn(delta)) ** 2
    norm = (2 * ell + 1) * L**3 / N**6
    bins = [norm * mx.sum(mk * weighted) / c for mk, c in zip(masks, counts)]
    return mx.stack(bins)


def cross_power_multipole(delta_a, delta_b, box, k_bins, ell, los_axis=0, dk=None):
    """Differentiable redshift-space cross multipole P_ell^{ab}(k) (raw).

    P_ell[b] = (2 ell + 1) (V / N^6) sum_{shell b} L_ell(mu) Re(a_k conj(b_k))
    / count_b -- band_power_multipole with the auto power |delta_k|^2 replaced by
    the cross power Re(a_k conj(b_k)) (the same substitution cross_power makes on
    band_power). Reduces to cross_power at ell = 0 and to band_power_multipole when
    delta_a is delta_b. Differentiable in both fields (the mu / Legendre grids are
    constants), so it backs the redshift-space multi-tracer cross-spectrum data
    vector. RAW estimator (carries the discrete-shell multipole leakage, which a
    Fisher is invariant to). Returns an MLX vector.
    """
    N, L = box.n_mesh, box.box_size
    if dk is None:
        dk = box.k_fundamental
    masks, counts = _bin_masks(box, k_bins, dk)
    leg = mx.array(_legendre_weight(_mu_grid(box, los_axis), ell).astype(np.float32))
    cross = mx.real(mx.fft.rfftn(delta_a) * mx.conj(mx.fft.rfftn(delta_b)))
    weighted = leg * cross
    norm = (2 * ell + 1) * L**3 / N**6
    bins = [norm * mx.sum(mk * weighted) / c for mk, c in zip(masks, counts)]
    return mx.stack(bins)


def multipole_decoupling(box, k_bins, ells=(0, 2), los_axis=0, dk=None):
    """Per-bin inverse of the discrete-shell multipole mode-coupling matrix.

    On a finite mesh a thin |k| shell samples solid angle non-uniformly, so the
    raw estimator mixes multipoles:

        raw_ell[b] = sum_L M_b[ell, L] P_L[b],
        M_b[ell, L] = (2 ell + 1) mean_{shell b}[ L_ell(mu) L_L(mu) ],

    which is the identity only in the continuum (<L_ell L_L> = delta/(2 ell+1)).
    Returns the list of inverse matrices M_b^{-1} (float64, one per bin) so the
    corrected multipoles are P[b] = M_b^{-1} raw[b]. Geometric (field-independent),
    so applying it to differentiable raw band powers preserves differentiability;
    and being a constant linear map of the data vector it leaves a Fisher forecast
    invariant -- it is needed only to interpret/validate the multipoles against
    continuum theory.
    """
    if dk is None:
        dk = box.k_fundamental
    _, _, k_mag = k_grid(box)
    mu = _mu_grid(box, los_axis)
    Ls = [_legendre_weight(mu, el) for el in ells]
    inv = []
    for kc in k_bins:
        m = _shell_mask(k_mag, kc - 0.5 * dk, kc + 0.5 * dk)
        M = np.array(
            [
                [
                    (2 * el + 1) * float((Ls[i][m] * Ls[j][m]).mean())
                    for j in range(len(ells))
                ]
                for i, el in enumerate(ells)
            ]
        )
        inv.append(np.linalg.inv(M))
    return inv


def power_multipoles(
    delta,
    box,
    ells=(0, 2),
    los_axis=0,
    dk=None,
    kmin=None,
    kmax=None,
    deconvolve_cic=False,
    decouple=True,
):
    """Estimate redshift-space multipoles P_ell(k) of a real field (diagnostic).

    Bins (2 ell + 1) L_ell(mu) |delta_k|^2 in |k| shells with the V/N^6
    normalization, then -- when decouple=True (default) -- inverts the
    discrete-shell mode-coupling (see multipole_decoupling) so the returned
    multipoles match continuum theory. Returns (k_centers, {ell: P_ell}, n_modes)
    as float64 numpy arrays, excluding k = 0. `deconvolve_cic` divides out the CIC
    window W(k)^2 (use for a particle-painted field). ells defaults to (0, 2); P4
    is available but noise-dominated at small box sizes (its signal is ~0.04 P0).
    Off the autodiff path; use band_power_multipole where a differentiable
    multipole is needed.
    """
    N, L = box.n_mesh, box.box_size
    power_modes = np.asarray(mx.abs(mx.fft.rfftn(delta)) ** 2, dtype=np.float64)
    _, _, k_mag = k_grid(box)
    power_modes *= L**3 / N**6
    if deconvolve_cic:
        power_modes /= cic_window(box) ** 2
    mu = _mu_grid(box, los_axis)

    kf = box.k_fundamental
    if dk is None:
        dk = kf
    if kmin is None:
        kmin = 0.5 * kf
    if kmax is None:
        kmax = box.k_nyquist
    edges = np.arange(kmin, kmax + dk, dk)

    km = k_mag.ravel()
    counts, _ = np.histogram(km, bins=edges)
    centers = 0.5 * (edges[1:] + edges[:-1])
    good = counts > 0
    kcen = centers[good]

    raw = {}
    for el in ells:
        w = _legendre_weight(mu, el).ravel() * power_modes.ravel()
        sum_p, _ = np.histogram(km, bins=edges, weights=w)
        raw[el] = (2 * el + 1) * sum_p[good] / counts[good]

    if not decouple:
        return kcen, raw, counts[good]

    inv = multipole_decoupling(box, kcen, ells=ells, los_axis=los_axis, dk=dk)
    raw_mat = np.array([raw[el] for el in ells])  # (n_ell, n_bin)
    dec = np.array([inv[b] @ raw_mat[:, b] for b in range(raw_mat.shape[1])]).T
    return kcen, {el: dec[i] for i, el in enumerate(ells)}, counts[good]


def interlaced_density_contrast(positions, box):
    """CIC density contrast with interlacing, to suppress mass-assignment aliasing.

    Paints the particles on the mesh and on a second copy shifted by half a cell
    in every dimension, then averages the two in Fourier space after realigning
    the shifted one by its phase exp(i k . s), s = (d/2, d/2, d/2), d = L/N. This
    cancels the leading (odd) aliasing images of the CIC assignment (Sefusatti et
    al. 2016, arXiv:1512.07295), sharpening the multipoles at intermediate k where
    aliasing contaminates the anisotropic signal most. A drop-in replacement for
    painting.density_contrast: returns a real (N, N, N) field, differentiable in
    positions. The CIC window W(k) suppression remains -- deconvolve it in the
    estimator (deconvolve_cic) as usual. This is the standard measurement painter
    for all P(k) / band-power diagnostics and Jacobians; the force solve
    (forces.forces_on_particles) keeps plain CIC.
    """
    from mbody import painting as PA  # local import: painting must not import fields

    N, L = box.n_mesh, box.box_size
    d = L / N
    mean = positions.shape[0] / (N**3)
    delta1 = PA.cic_paint(positions, box) / mean - 1.0
    delta2 = PA.cic_paint(positions + 0.5 * d, box) / mean - 1.0

    k_1d, kz_1d, _ = k_grid(box)
    kdots = 0.5 * d * (k_1d[:, None, None] + k_1d[None, :, None] + kz_1d[None, None, :])
    phase = mx.array(np.exp(1j * kdots).astype(np.complex64))
    delta_k = 0.5 * (mx.fft.rfftn(delta1) + phase * mx.fft.rfftn(delta2))
    return mx.fft.irfftn(delta_k, s=(N, N, N), axes=(0, 1, 2))
