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


def poisson_M(k, cosmo, z=0.0, backend="camb"):
    """M(k, z) = (2/3) (c/H0)^2 k^2 T(k) D_md(z) / Omega_m, for arbitrary k.

    The Poisson/transfer factor relating potential and density,
    delta_lin(k, z) = M(k, z) phi(k). Accepts scalar or array k (h/Mpc) and
    preserves its shape; the transfer function is evaluated at |k| with k = 0
    mapped to a safe placeholder (M itself still -> 0 there via the k^2). The
    shared building block of poisson_factor (on the mesh) and
    local_bispectrum_template (at a handful of triangle wavenumbers).
    """
    k = np.asarray(k, dtype=np.float64)
    k_safe = np.where(k > 0, k, 1.0)
    T = C.transfer(k_safe.ravel(), cosmo, backend).reshape(k.shape)
    D_md = C.growth_factor_md(z, cosmo)
    return (2.0 / 3.0) * C_OVER_H0**2 * k**2 * T * D_md / cosmo.Omega_m


def poisson_factor(box, cosmo, z=0.0, backend="camb"):
    """M(k, z) on the rfftn half-grid: delta_lin(k, z) = M(k, z) phi(k).

    M = (2/3) (c/H0)^2 k^2 T(k) D_md(z) / Omega_m (see poisson_M). Returned as a
    float64 numpy array of shape (N, N, N//2 + 1); the k = 0 entry is set to 1
    as a safe placeholder (callers keep the density's DC mode at zero, so it
    never matters). M -> 0 as k -> 0, so phi = delta/M reddens the field.
    """
    _, _, k_mag = F.k_grid(box)
    M = poisson_M(k_mag, cosmo, z=z, backend=backend)
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


def local_bispectrum_template(triangles, cosmo, f_NL, z=0.0, backend="camb"):
    """Tree-level local-f_NL density bispectrum at the given triangles.

    For delta(k) = M(k) phi(k) with phi = phi_G + f_NL (phi_G^2 - <phi_G^2>),

        B(k1,k2,k3) = 2 f_NL [ M3/(M1 M2) P1 P2 + M2/(M1 M3) P1 P3
                                                + M1/(M2 M3) P2 P3 ],

    where M_i = poisson_M(k_i, z) and P_i = cosmology.linear_power(k_i, z) are
    the same factors linear_density is built from, so this is the exact tree
    prediction for that field (the growth D_md(z) cancels between M and P_phi).
    Squeezed k1 -> 0 gives 1/M1 ~ 1/k1^2, the scale-dependent-bias divergence;
    equilateral is finite. This is what fields.bispectrum is validated against.

    `triangles` is a sequence of (k1, k2, k3) wavenumbers (h/Mpc). Returns a
    float64 numpy array, one entry per triangle.
    """
    tris = np.atleast_2d(np.asarray(triangles, dtype=np.float64))
    ks = np.unique(tris)
    M = dict(zip(ks, poisson_M(ks, cosmo, z=z, backend=backend)))
    Pk = dict(zip(ks, C.linear_power(ks, cosmo, z=z, backend=backend)))
    out = np.empty(len(tris), dtype=np.float64)
    for t, (k1, k2, k3) in enumerate(tris):
        M1, M2, M3 = M[k1], M[k2], M[k3]
        P1, P2, P3 = Pk[k1], Pk[k2], Pk[k3]
        out[t] = (
            2.0
            * f_NL
            * (
                M3 / (M1 * M2) * P1 * P2
                + M2 / (M1 * M3) * P1 * P3
                + M1 / (M2 * M3) * P2 * P3
            )
        )
    return out


def local_bispectrum_binned(
    box, cosmo, triangles, f_NL, z=0.0, backend="camb", dk=None
):
    """Bin-averaged tree-level local-f_NL bispectrum -- the exact prediction for
    fields.bispectrum.

    fields.bispectrum returns the mean of B_tree over all mode-triplets in each
    (b1, b2, b3) bin, so comparing to the continuum B_tree at the bin centre
    (local_bispectrum_template) carries a binning systematic: the shell mode
    density ~ k^2 shifts the effective k above centre, which biases the steep
    squeezed long side low. This evaluates the same bin average exactly, with no
    centre approximation, so the estimator should match it with unit
    calibration.

    B_tree = 2 f_NL sum_perm g(k_a) f(k_b) f(k_c), with g = M, f = P_lin / M, so
    the bin average is

        2 f_NL sum_x (F1 F2 G3 + G1 F2 F3 + F1 G2 F3) / sum_x (J1 J2 J3),
        G_i = irfftn(M Theta_i), F_i = irfftn((P/M) Theta_i), J_i = irfftn(Theta_i),

    the same Scoccimarro shell-product identity fields.bispectrum uses (the N^6
    factors cancel against the triangle count). `dk` defaults to the
    fundamental, matching the estimator. Returns a float64 numpy array.
    """
    N = box.n_mesh
    if dk is None:
        dk = box.k_fundamental
    _, _, k_mag = F.k_grid(box)
    M = poisson_M(k_mag, cosmo, z=z, backend=backend)
    P_lin = C.linear_power(k_mag.ravel(), cosmo, z=z, backend=backend).reshape(M.shape)
    f_wt = np.where(k_mag > 0, P_lin / np.where(k_mag > 0, M, 1.0), 0.0)

    M32 = mx.array(M.astype(np.float32))
    f32 = mx.array(f_wt.astype(np.float32))

    def shell_field(weight32, kc):
        mask = F._shell_mask(k_mag, kc - 0.5 * dk, kc + 0.5 * dk).astype(np.float32)
        band = P.as_complex(weight32 * mx.array(mask))
        return mx.fft.irfftn(band, s=(N, N, N), axes=(0, 1, 2))

    centers = sorted({float(k) for tri in triangles for k in tri})
    ones = mx.ones(M32.shape, dtype=P.REAL)
    G = {kc: shell_field(M32, kc) for kc in centers}
    Fk = {kc: shell_field(f32, kc) for kc in centers}
    J = {kc: shell_field(ones, kc) for kc in centers}

    out = np.empty(len(triangles), dtype=np.float64)
    for t, (k1, k2, k3) in enumerate(triangles):
        k1, k2, k3 = float(k1), float(k2), float(k3)
        num = P.accurate_sum(
            Fk[k1] * Fk[k2] * G[k3] + G[k1] * Fk[k2] * Fk[k3] + Fk[k1] * G[k2] * Fk[k3]
        )
        den = P.accurate_sum(J[k1] * J[k2] * J[k3])
        out[t] = 2.0 * f_NL * float(num) / float(den)
    return out
