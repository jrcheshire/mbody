"""Lagrangian perturbation theory: turn a density field into moving particles.

This is the bridge from "field" to "N-body." Instead of tracking the density at
fixed points in space (Eulerian), Lagrangian perturbation theory (LPT) follows
each fluid element: a particle starts at its grid position q and moves to

    x(q, t) = q + Psi(q, t),

where Psi is the displacement field. First-order LPT -- the **Zel'dovich
approximation** (Zel'dovich 1970) -- keeps the leading displacement

    Psi_1(q) = grad (inverse-laplacian) delta_0(q),    so   div Psi_1 = -delta_0,

i.e. each particle streams in a straight line along the gradient of the initial
gravitational potential, toward overdensities. The amplitude grows with the
linear growth factor: Psi(q, z) = D(z) Psi_1(q). It is "linear in the
displacement" but not in the density -- moving particles and seeing where they
pile up already produces real nonlinear structure (the Zel'dovich "pancakes").

In Fourier space the displacement is simply

    Psi_1,j(k) = i k_j / k^2 * delta_0(k),

with the k = 0 mode set to zero. We compute it with the same rfftn machinery as
the field code. Psi_1 is normalized to z = 0 (D = 1); positions at any redshift
follow by scaling with D(z), which is exactly what makes a structure-formation
animation fall out for free.

2LPT (the next correction) is a TODO. Velocities are deferred to the integrator,
which will fix the time variable and velocity convention. For now LPT assumes
n_particles == n_mesh (one particle per cell).
"""

import mlx.core as mx
import numpy as np

from mbody import cosmology as C
from mbody import ic as IC
from mbody import precision as P


def _k_components(box):
    """i k_j (complex) and 1/k^2 (real) on the rfftn half-grid, k=0 safe.

    Returns (ikx, iky, ikz, inv_k2) as MLX arrays. The ik_j are kept low-rank
    (broadcastable) to save memory; inv_k2 is the full (N, N, N//2+1) grid.
    """
    N, L = box.n_mesh, box.box_size
    d = L / N
    kx = 2.0 * np.pi * np.fft.fftfreq(N, d=d)
    kz = 2.0 * np.pi * np.fft.rfftfreq(N, d=d)
    M = N // 2 + 1
    KX = kx.reshape(N, 1, 1)
    KY = kx.reshape(1, N, 1)
    KZ = kz.reshape(1, 1, M)
    k2 = KX**2 + KY**2 + KZ**2
    k2[0, 0, 0] = 1.0  # avoid 0/0; the ik_j are zero at k=0 anyway
    inv_k2 = (1.0 / k2).astype(np.float32)
    ikx = mx.array((1j * KX).astype(np.complex64))
    iky = mx.array((1j * KY).astype(np.complex64))
    ikz = mx.array((1j * KZ).astype(np.complex64))
    return ikx, iky, ikz, mx.array(inv_k2)


def zeldovich_displacement(box, cosmo, seed=0, f_NL=0.0, backend="camb"):
    """First-order (Zel'dovich) displacement field, normalized to z = 0.

    Returns (psi_x, psi_y, psi_z), each a real float32 MLX array of shape
    (n_mesh, n_mesh, n_mesh), in Mpc/h. Multiply by D(z) for the displacement
    at redshift z. The source density is ic.linear_density, so passing `f_NL`
    (as an mx scalar) injects local non-Gaussianity and keeps the whole
    displacement -- and anything built on it -- differentiable in f_NL; f_NL = 0
    recovers the Gaussian field (to FFT round-off).
    """
    N = box.n_mesh
    delta0 = IC.linear_density(box, cosmo, seed=seed, z=0.0, f_NL=f_NL, backend=backend)
    dk = mx.fft.rfftn(delta0)
    ikx, iky, ikz, inv_k2 = _k_components(box)
    # i k_j / k^2 colours the noise into a displacement. This is exact for all
    # resolved modes; the Nyquist plane (where a real field's spectral gradient
    # is ill-defined) is handled by irfftn's real projection -- a
    # resolution-scale ambiguity, confirmed in test_lpt to be the only residual
    # in the div(Psi) = -delta identity.
    psi_x = mx.fft.irfftn(dk * ikx * inv_k2, s=(N, N, N), axes=(0, 1, 2))
    psi_y = mx.fft.irfftn(dk * iky * inv_k2, s=(N, N, N), axes=(0, 1, 2))
    psi_z = mx.fft.irfftn(dk * ikz * inv_k2, s=(N, N, N), axes=(0, 1, 2))
    return P.as_real(psi_x), P.as_real(psi_y), P.as_real(psi_z)


def divergence(box, vx, vy, vz):
    """Divergence of a vector field on the mesh, via FFT. A reusable diagnostic
    (e.g. div Psi_1 should equal -delta_0). Returns a real float32 MLX array.
    """
    N = box.n_mesh
    ikx, iky, ikz, _ = _k_components(box)
    div_k = mx.fft.rfftn(vx) * ikx + mx.fft.rfftn(vy) * iky + mx.fft.rfftn(vz) * ikz
    return P.as_real(mx.fft.irfftn(div_k, s=(N, N, N), axes=(0, 1, 2)))


def lagrangian_grid(box):
    """Unperturbed particle positions: a uniform grid, shape (n_mesh^3, 3)."""
    N, L = box.n_mesh, box.box_size
    d = L / N
    coords = mx.arange(N, dtype=mx.float32) * d
    qx, qy, qz = mx.meshgrid(coords, coords, coords, indexing="ij")
    return mx.stack([qx.reshape(-1), qy.reshape(-1), qz.reshape(-1)], axis=1)


def displace(box, psi, growth):
    """Displace the uniform grid by `growth` * psi, with periodic wrapping.

    Returns particle positions of shape (n_mesh^3, 3) in [0, L), float32.
    """
    N, L = box.n_mesh, box.box_size
    d = L / N
    coords = mx.arange(N, dtype=mx.float32) * d
    qx = coords.reshape(N, 1, 1)
    qy = coords.reshape(1, N, 1)
    qz = coords.reshape(1, 1, N)

    def wrap(a):
        return a - L * mx.floor(a / L)

    x = wrap(qx + growth * psi[0])
    y = wrap(qy + growth * psi[1])
    z = wrap(qz + growth * psi[2])
    return mx.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], axis=1)


def lpt_positions(box, cosmo, seed=0, z=0.0, f_NL=0.0, backend="camb"):
    """Particle positions at redshift z under the Zel'dovich approximation.

    Convenience wrapper: build the z=0 displacement and scale it by D(z).
    """
    if box.n_particles != box.n_mesh:
        raise ValueError("LPT currently assumes n_particles == n_mesh")
    psi = zeldovich_displacement(box, cosmo, seed=seed, f_NL=f_NL, backend=backend)
    return displace(box, psi, C.growth_factor(z, cosmo))
