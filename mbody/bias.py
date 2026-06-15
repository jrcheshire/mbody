"""A minimal differentiable biased tracer, and the Dalal scale-dependent-bias
reference shape.

The matter density itself has no power-spectrum response to local f_NL at linear
order: dlnP_matter/df_NL = 0 (the correction is 2 Re<phi_G^* phi_G^2>, a Gaussian
three-point, which vanishes). The Dalal et al. (2008) scale-dependent bias
Delta b(k) ~ f_NL / k^2 is a property of *biased tracers*: an object whose
abundance responds to the local small-scale density variance feels the
long-wavelength potential phi that local f_NL couples to that variance, and phi =
delta / M(k) with M(k) proportional to k^2 T(k) D(z), hence the ~1/k^2 at large
scales (where the transfer T -> 1).

The smallest differentiable tracer that exhibits this is an Eulerian local
quadratic bias

    delta_h(x) = b1 delta(x) + (b2/2) [delta(x)^2 - <delta^2>],

where the b2 (delta^2) term couples to the squeezed bispectrum of the f_NL field
(the same physics validated in mbody.fields.bispectrum), producing a
scale-dependent bias dlnP_h/df_NL ~ 1/M(k). It is FFT-free, elementwise, and so
trivially differentiable -- mx.grad / mx.jvp flow through to f_NL.
"""

import numpy as np
import mlx.core as mx

from mbody import cosmology as C
from mbody import fields as F
from mbody import ic as IC
from mbody import precision as P


def local_bias_tracer(delta, b1, b2):
    """Eulerian local quadratic-bias tracer.

    delta_h = b1 delta + (b2/2) (delta^2 - <delta^2>), a real field the same
    shape as delta. The mean subtraction only shifts the (unobserved) k = 0 mode,
    so it is taken as a constant -- it does not affect any band power or its
    f_NL derivative. Differentiable in delta; pass b1, b2 as plain floats.
    """
    mean2 = float(P.accurate_mean(delta**2))
    return b1 * delta + 0.5 * b2 * (delta**2 - mean2)


def scale_dependent_shape(k, cosmo, z=0.0, backend="camb"):
    """The Dalal scale-dependent-bias reference shape, 1 / M(k) ~ 1 / (k^2 T(k)).

    M(k) = ic.poisson_M(k, z) is the Poisson/transfer factor (delta = M phi), so
    1/M(k) is the k-shape of the local-f_NL bias correction the tracer's
    dlnP/df_NL should follow at large scales. Returns a float64 numpy array (the
    amplitude is set by the tracer's b_phi; overlay this normalized to the data).
    For the *absolute* amplitude (no free normalization) use
    scale_dependent_bias_response.
    """
    return 1.0 / IC.poisson_M(k, cosmo, z=z, backend=backend)


def mesh_variance(box, cosmo, z=0.0, backend="camb"):
    """Expected linear-density variance <delta^2> on the box's mesh, at amplitude 1.

    The variance of a Gaussian (or f_NL = 0) linear field generated on the mesh,
    in expectation: <delta^2> = (1/V) sum_{k != 0} P_lin(|k|, z) over ALL N^3
    Fourier modes (V = L^3). Computed on the full fft grid (not the rfft half-grid)
    so every mode counts once with no Hermitian double-count bookkeeping. This is a
    *mesh* quantity -- it depends on the Nyquist cutoff n_mesh/L, exactly as the b2
    delta^2 operator does -- so it is the right sigma^2 for the scale-dependent-bias
    amplitude below (a continuum integral would carry a UV systematic). Returns a
    float; equals the realized field's <delta^2> up to per-seed cosmic variance.
    """
    N, L = box.n_mesh, box.box_size
    k1d = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    k_mag = np.sqrt(
        k1d[:, None, None] ** 2 + k1d[None, :, None] ** 2 + k1d[None, None, :] ** 2
    )
    Pk = C.linear_power(k_mag.ravel(), cosmo, z=z, backend=backend).reshape(k_mag.shape)
    Pk[0, 0, 0] = 0.0  # the DC mode carries no variance (zero-mean field)
    return float(Pk.sum() / L**3)


def scale_dependent_bias_response(
    k, box, cosmo, b1, b2, amplitude=1.0, z=0.0, backend="camb"
):
    """Absolute large-scale dlnP_h/df_NL(k) of the local quadratic-bias tracer.

    For delta_h = b1 delta + (b2/2)(delta^2 - <delta^2>) with delta = A * M phi and
    local f_NL (phi = phi_G + f_NL(phi_G^2 - <phi_G^2>)), a long-wavelength
    potential phi_L modulates the small-scale variance as (1 + 4 f_NL phi_L), so the
    tracer's delta^2 term acquires a long-mode piece 2 b2 A sigma^2 f_NL phi_L. With
    phi_L = delta_L / (A M(k)) this is an effective scale-dependent bias
    Delta b(k) = 2 b2 A sigma^2 f_NL / M(k), hence

        dlnP_h/df_NL(k) = 4 b2 A sigma^2 / (b1 M(k)),     (large-scale leading term)

    where sigma^2 = mesh_variance(box) is the f_NL = 0, A = 1 mesh variance (the
    amplitude A scales it back out linearly). This is the squeezed-limit of the
    integrated tree-level bispectrum (I(k)/P(k) -> 4 sigma^2/M(k)); it holds at low
    k where Delta b ~ 1/M(k) dominates, and is the *absolute* amplitude behind the
    shape-only scale_dependent_shape. Returns a float64 numpy array (k can be a
    scalar or array, h/Mpc).
    """
    sigma2 = mesh_variance(box, cosmo, z=z, backend=backend)
    M = IC.poisson_M(k, cosmo, z=z, backend=backend)
    return 4.0 * b2 * amplitude * sigma2 / (b1 * M)


def scale_dependent_bias_response_binned(
    box, cosmo, k_bins, b1, b2, amplitude=1.0, dk=None, z=0.0, backend="camb"
):
    """Bin-averaged dlnP_h/df_NL over |k| shells -- the exact prediction for the
    band-power log-derivative measured on the linear field.

    fields.band_power measures dlnP_b/df_NL = d ln(sum_shell P_h) / df_NL =
    (sum_shell dP_h/df) / (sum_shell P_h). With the leading scale-dependent bias
    dP_h(q)/df = 4 b1 b2 A sigma^2 P_lin(q)/M(q) and P_h(q) = b1^2 P_lin(q), the
    shell ratio is

        dlnP_b/df_NL = (4 b2 A sigma^2 / b1) * sum_shell[P_lin/M] / sum_shell[P_lin],

    a P_lin-weighted shell average of 1/M(q). Evaluating it on the same rfftn
    half-grid shells band_power sums over removes the binning systematic that
    biases the steep low-k bins when scale_dependent_bias_response is read at the
    bin centre (the rfft plane double-count cancels in the ratio) -- the same
    shell-average fix as ic.local_bispectrum_binned. Returns a float64 numpy array,
    one entry per bin. This is the absolute low-k anchor; high-k bins still carry
    the sub-leading (non-squeezed + b2^2) terms the leading 1/M(q) form omits.
    """
    sigma2 = mesh_variance(box, cosmo, z=z, backend=backend)
    _, _, k_mag = F.k_grid(box)
    km = k_mag.ravel()
    P_lin = C.linear_power(km, cosmo, z=z, backend=backend)
    M = IC.poisson_M(km, cosmo, z=z, backend=backend)
    M_safe = np.where(km > 0, M, 1.0)  # M -> 0 at k=0; avoid 0/0 (masked out below)
    inv_M = np.where(km > 0, P_lin / M_safe, 0.0)
    P_weight = np.where(km > 0, P_lin, 0.0)

    if dk is None:
        dk = box.k_fundamental
    out = np.empty(len(k_bins), dtype=np.float64)
    for b, kc in enumerate(k_bins):
        mask = (km >= kc - 0.5 * dk) & (km < kc + 0.5 * dk)
        num = inv_M[mask].sum()
        den = P_weight[mask].sum()
        out[b] = 4.0 * b2 * amplitude * sigma2 / b1 * (num / den)
    return out


def scale_dependent_bias_tracer(
    delta, box, cosmo, b1, b_phi, f_NL, z=0.0, backend="camb", invM=None
):
    """Tracer with an EXPLICIT local-f_NL scale-dependent bias (Fourier space):

        delta_h(k) = [b1 + b_phi f_NL / M(k)] delta(k),

    M(k) = ic.poisson_M (delta = M phi), so 1/M(k) ~ 1/(k^2 T(k)) is the Dalal
    scale-dependent-bias kernel. Unlike local_bias_tracer -- whose f_NL response
    is EMERGENT from b2 and so carries a broadband b2 self-calibration handle that
    partially breaks the b_phi-f_NL degeneracy at high k -- here b_phi is a FREE
    parameter entering ONLY through the k^-2 term. So d/df_NL and d/db_phi share
    the same 1/M(k) shape and are PERFECTLY degenerate (the product f_NL*b_phi) at
    all k, the faithful Barreira/Dalal degeneracy. Pass `delta` a Gaussian
    (f_NL=0) field -- the matter field has no O(f_NL) power, so f_NL is a pure
    bias parameter here. b1, b_phi, f_NL may be floats or mx scalars
    (differentiable). `invM` (the precomputed 1/M(k) on the rfft half-grid) is
    optional caching. Returns a real (N, N, N) field.
    """
    N = box.n_mesh
    if invM is None:
        _, _, k_mag = F.k_grid(box)
        M = IC.poisson_M(np.where(k_mag > 0, k_mag, 1.0), cosmo, z=z, backend=backend)
        invM = np.where(k_mag > 0, 1.0 / M, 0.0).astype(np.float32)
    factor = P.as_complex(b1 + b_phi * f_NL * mx.array(invM))
    return mx.fft.irfftn(factor * mx.fft.rfftn(delta), s=(N, N, N), axes=(0, 1, 2))


# --- Lagrangian bias (operators on the initial field; advected as weights) ----
#
# The Eulerian local quadratic tracer above paints onto the EVOLVED field, so its
# f_NL response is the EMERGENT b_phi = 2 b2 sigma^2 with sigma^2 the UV-sensitive
# *evolved* mesh variance -- diluted ~4x through the PM and needing nonperturbative
# b2 to reach a physical b_phi at high bias. The principled fix is a Lagrangian
# bias: build the bias operators on the INITIAL (linear) field, carry 1 + delta_g^L
# as per-particle weights, and advect them by the displacement (the standard
# CLEFT / LPT-bias / BORG-LEFTfield construction). Then b_phi enters DIRECTLY as a
# Lagrangian coefficient -- exactly the separate-universe / Barreira definition --
# with no sigma^2 bottleneck and no through-PM dilution (the Eulerian-vs-Lagrangian
# mismatch that caused the dilution is removed by construction). These functions
# build the operator fields; integrate/catalog do the advect-and-paint.


def tidal_sq(delta, box):
    """Traceless tidal shear squared s^2(q) = sum_ij s_ij s_ij of a field.

    The second-order tidal (shear) bias operator. s_ij(k) = (k_i k_j / |k|^2 -
    delta_ij / 3) delta-hat(k) is the traceless tidal tensor; s^2 sums its 6 unique
    components (the tensor is symmetric, so the 3 off-diagonals count twice), each
    formed with one inverse FFT. Returns a real (N, N, N) field, NOT mean-subtracted
    (the caller subtracts <s^2> to make it a zero-mean operator). Differentiable in
    `delta` (the kernels are constants).
    """
    N = box.n_mesh
    k_1d, kz_1d, k_mag = F.k_grid(box)
    kx = k_1d[:, None, None]
    ky = k_1d[None, :, None]
    kz = kz_1d[None, None, :]
    inv_k2 = np.where(k_mag > 0, 1.0 / np.where(k_mag > 0, k_mag**2, 1.0), 0.0)
    dk = mx.fft.rfftn(delta)
    # (k_a, k_b, multiplicity, is_diagonal); off-diagonal pairs count twice.
    comps = [
        (kx, kx, 1.0, True),
        (ky, ky, 1.0, True),
        (kz, kz, 1.0, True),
        (kx, ky, 2.0, False),
        (kx, kz, 2.0, False),
        (ky, kz, 2.0, False),
    ]
    s2 = mx.zeros((N, N, N), dtype=P.REAL)
    for ka, kb, mult, diag in comps:
        kernel = ka * kb * inv_k2  # broadcasts to the (N, N, N//2+1) half-grid
        if diag:
            kernel = kernel - 1.0 / 3.0  # remove the trace (k=0 has delta-hat=0)
        ker = P.as_complex(mx.array(kernel.astype(np.float32)))
        s_ij = P.as_real(mx.fft.irfftn(ker * dk, s=(N, N, N), axes=(0, 1, 2)))
        s2 = s2 + mult * (s_ij * s_ij)
    return s2


def laplacian_field(delta, box):
    """Laplacian nabla^2 delta = -|k|^2 delta-hat, back in real space.

    The leading higher-derivative bias operator (captures the finite nonlocality
    scale of galaxy formation). Returns a real (N, N, N) field; differentiable in
    `delta` (the -k^2 kernel is a constant). The sign is the physics Laplacian
    convention; an overall sign and the k-grid units are absorbed by the bias
    coefficient bnabla2.
    """
    N = box.n_mesh
    _, _, k_mag = F.k_grid(box)
    neg_k2 = mx.array((-(k_mag**2)).astype(np.float32))
    dk = mx.fft.rfftn(delta)
    return P.as_real(
        mx.fft.irfftn(P.as_complex(neg_k2) * dk, s=(N, N, N), axes=(0, 1, 2))
    )


def _coeff(c):
    """Normalize a bias coefficient: returns (is_zero_scalar, value).

    A scalar passes through (and is flagged when exactly 0 so its operator can be
    skipped); a per-cell (N, N, N) numpy array is cast to a float32 mx field so the
    coefficient can vary spatially -- e.g. a radially-evolving b1(r) / b_phi(r) along
    the line of sight (no single effective redshift). An mx array passes through.
    """
    if isinstance(c, np.ndarray):
        return False, mx.array(c.astype(np.float32))
    if isinstance(c, mx.array):
        return False, c
    return (c == 0.0), c


def lagrangian_bias_field(
    delta,
    box,
    b1,
    b2=0.0,
    bs2=0.0,
    bnabla2=0.0,
    b_phi=0.0,
    f_NL=0.0,
    phi_G=None,
    cosmo=None,
    seed=0,
    z=0.0,
    backend="camb",
):
    """Second-order Lagrangian bias field delta_g^L(q) from a linear field.

        delta_g^L = b1 delta + (b2/2)(delta^2 - <delta^2>)
                    + bs2 (s^2 - <s^2>) + bnabla2 nabla^2 delta
                    + b_phi f_NL phi_G,

    the standard local-plus-tidal Lagrangian bias expansion with an EXPLICIT local
    primordial-non-Gaussianity term. `delta` is the LINEAR (Lagrangian) density
    field (e.g. ic.linear_density); b1..bnabla2 are the LAGRANGIAN bias coefficients
    (after advection the Eulerian linear bias is 1 + b1, since the uniform unit
    weight advects to the matter field). `b_phi` enters the f_NL scale-dependent
    bias DIRECTLY -- the separate-universe / Barreira definition b_phi = 2 delta_c
    (b1_E - p) -- through the Gaussian primordial potential phi_G = delta_G / M
    (ic.primordial_potential; built from `cosmo`/`seed`/`z` if not passed). In
    Fourier the b_phi term is b_phi f_NL delta_G / M(k), so it imprints the Dalal
    scale-dependent bias Delta_b(k) = b_phi f_NL / M(k) at the Lagrangian level with
    no b2*sigma^2 conversion.

    Each coefficient may be a scalar OR a per-cell (N, N, N) array, so the bias can
    evolve along the line of sight -- b1(r), b_phi(r) as functions of comoving
    distance from the observer (the continuous-bias treatment, no effective
    redshift) -- via radial coefficient fields the caller builds (e.g. from
    catalog.radial_nbar_field with a b(r) profile).

    Returns a real (N, N, N) field; carry 1 + delta_g^L(q) as per-particle weights
    (q in lpt.lagrangian_grid order = raveled field order) and CIC-paint the
    advected particles. Differentiable in `delta`, `f_NL` (pass an mx scalar) and
    scalar coefficients.
    """
    delta = P.as_real(delta)
    zb1, b1 = _coeff(b1)
    zb2, b2 = _coeff(b2)
    zbs2, bs2 = _coeff(bs2)
    znb, bnabla2 = _coeff(bnabla2)
    zphi, b_phi = _coeff(b_phi)

    field = b1 * delta
    if not zb2:
        mean_d2 = float(P.accurate_mean(delta * delta))
        field = field + 0.5 * b2 * (delta * delta - mean_d2)
    if not zbs2:
        s2 = tidal_sq(delta, box)
        field = field + bs2 * (s2 - float(P.accurate_mean(s2)))
    if not znb:
        field = field + bnabla2 * laplacian_field(delta, box)
    if not zphi:
        if phi_G is None:
            if cosmo is None:
                raise ValueError("the b_phi term needs phi_G, or cosmo to build it")
            phi_G = IC.primordial_potential(box, cosmo, seed=seed, z=z, backend=backend)
        field = field + b_phi * f_NL * P.as_real(phi_G)
    return field
