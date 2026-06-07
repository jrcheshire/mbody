"""A minimal differentiable biased tracer, and the Dalal scale-dependent-bias
reference shape.

The matter density itself has no power-spectrum response to local f_NL at linear
order: dlnP_matter/df_NL = 0 (the correction is 2 Re<phi_G^* phi_G^2>, a Gaussian
three-point, which vanishes). The Dalal et al. (2008) scale-dependent bias
Delta b(k) ~ f_NL / k^2 is a property of *biased tracers*: an object whose
abundance responds to the local small-scale density variance feels the
long-wavelength potential phi that local f_NL couples to that variance, and phi =
delta / M(k) with M(k) ~ k^2, hence the 1/k^2.

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
