"""Initial conditions: Gaussian, and local primordial non-Gaussianity (f_NL).

The Gaussian field code (mbody.fields) draws the linear density delta directly
from P(k). To inject *local* primordial non-Gaussianity we have to step back to
the primordial gravitational potential, because the local model is defined
there:

    phi(x) = phi_G(x) + f_NL [phi_G(x)^2 - <phi_G^2>],

with phi_G a Gaussian potential and the constant subtracted so <phi> = 0. The
quadratic term couples scales -- a long-wavelength phi_G modulates the local
small-scale power -- which is the squeezed-limit bispectrum and, through galaxy
formation, the f_NL scale-dependent bias b_phi that is the real SPHEREx lever.

We never need to calibrate phi against the CMB directly. The density and the
potential are tied by the Poisson/transfer relation

    delta(k, z) = M(k, z) phi(k),
    M(k, z) = (2/3) (c/H0)^2 k^2 T(k) D_md(z) / Omega_m,

where T(k) is the transfer function and D_md is the growth normalized to D = a
in matter domination (cosmology.growth_factor_md). So we generate delta_G
exactly as before (already sigma8-normalized), divide by M to get the physical
phi_G (which comes out at the ~1e-5 COBE scale, a good check on the
normalization), apply the local transform in real space, and multiply by M to
return to delta. At f_NL = 0 this round-trips to the Gaussian field exactly.

Everything is FFTs and a real-space square, so it is differentiable in f_NL
(linearly -- f_NL enters as a single multiplicative term) and, unlike the CIC
path, kink-free and bit-reproducible. This is the parameter the project
ultimately differentiates the power spectrum with respect to.
"""

import mlx.core as mx
import numpy as np

from mbody import cosmology as C
from mbody import fields as F
from mbody import precision as P

# c / H0 in Mpc/h: c = 299792.458 km/s, H0 = 100 h km/s/Mpc, and the h cancels
# when lengths are measured in Mpc/h, so this is just c[km/s] / 100.
C_OVER_H0 = 299792.458 / 100.0


def poisson_factor(box, cosmo, z=0.0, backend="camb"):
    """M(k, z) on the rfftn half-grid: delta_lin(k, z) = M(k, z) phi(k).

    M = (2/3) (c/H0)^2 k^2 T(k) D_md(z) / Omega_m. Returned as a float64 numpy
    array of shape (N, N, N//2 + 1); the k = 0 entry is set to 1 as a safe
    placeholder (callers keep the density's DC mode at zero, so it never
    matters). M -> 0 as k -> 0, so phi = delta/M reddens the field, as it should.
    """
    _, _, k_mag = F.k_grid(box)
    k_flat = k_mag.ravel()
    k_safe = np.where(k_flat > 0, k_flat, 1.0)
    T = C.transfer(k_safe, cosmo, backend).reshape(k_mag.shape)
    D_md = C.growth_factor_md(z, cosmo)
    M = (2.0 / 3.0) * C_OVER_H0**2 * k_mag**2 * T * D_md / cosmo.Omega_m
    return np.where(k_mag > 0, M, 1.0)


def _M_mx(box, cosmo, z, backend):
    """Poisson factor as a float32 MLX array for the GPU FFT path."""
    return mx.array(poisson_factor(box, cosmo, z=z, backend=backend).astype(np.float32))


def primordial_potential(box, cosmo, seed=0, z=0.0, backend="camb"):
    """Gaussian primordial potential phi_G(x), with delta_G = M phi_G.

    Returns a real float32 (N, N, N) field, at the ~1e-5 scale of the physical
    primordial potential -- a sanity check that M is normalized correctly.
    """
    N = box.n_mesh
    delta_G = F.gaussian_random_field(box, cosmo, seed=seed, z=z, backend=backend)
    M = _M_mx(box, cosmo, z, backend)
    phi_k = mx.fft.rfftn(delta_G) / M
    return P.as_real(mx.fft.irfftn(phi_k, s=(N, N, N), axes=(0, 1, 2)))


def linear_density(box, cosmo, seed=0, z=0.0, f_NL=0.0, backend="camb"):
    """Linear density field with optional local primordial non-Gaussianity.

    delta_G -> phi_G = delta_G/M -> phi = phi_G + f_NL (phi_G^2 - <phi_G^2>) ->
    delta = M phi. Returns a real float32 (N, N, N) field; at f_NL = 0 it equals
    mbody.fields.gaussian_random_field (to FFT round-off). Differentiable in
    f_NL (pass f_NL as an mx scalar for mx.grad).
    """
    N = box.n_mesh
    delta_G = F.gaussian_random_field(box, cosmo, seed=seed, z=z, backend=backend)
    M = _M_mx(box, cosmo, z, backend)
    phi_G = P.as_real(
        mx.fft.irfftn(mx.fft.rfftn(delta_G) / M, s=(N, N, N), axes=(0, 1, 2))
    )
    mean_phi2 = float(P.accurate_mean(phi_G**2))  # fp64 mean; constant in f_NL
    phi_NG = phi_G + f_NL * (phi_G**2 - mean_phi2)
    delta_k = mx.fft.rfftn(phi_NG) * M
    return P.as_real(mx.fft.irfftn(delta_k, s=(N, N, N), axes=(0, 1, 2)))
