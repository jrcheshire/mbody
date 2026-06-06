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

2LPT (the second-order correction) is implemented in `second_order_displacement`
below: the displacement gains a term D2 Psi2 sourced by the quadratic
`lpt2_source`, which curbs the Zel'dovich approximation's spurious early-time
transient. Velocities are deferred to the integrator, which fixes the time
variable and velocity convention. LPT assumes n_particles == n_mesh (one
particle per cell).
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


def _second_derivatives(delta, box):
    """The six unique second derivatives phi,ij(x) of the linear potential.

    phi = lap^-1 delta is the linear potential whose gradient is the Zel'dovich
    displacement (phi,i = Psi1_i), so phi_k = delta_k / k^2 and
    phi,ij(k) = -k_i k_j phi_k = (i k_i)(i k_j) phi_k. Returns a dict keyed by
    the index pairs (0,0),(1,1),(2,2),(0,1),(0,2),(1,2), each a real (N,N,N) MLX
    field. Differentiable in delta (the k-kernels are constants).
    """
    N = box.n_mesh
    ikx, iky, ikz, inv_k2 = _k_components(box)
    ik = (ikx, iky, ikz)
    phi_k = mx.fft.rfftn(delta) * inv_k2  # phi_k = delta_k / k^2
    out = {}
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        phi_ij_k = ik[i] * ik[j] * phi_k
        out[(i, j)] = P.as_real(mx.fft.irfftn(phi_ij_k, s=(N, N, N), axes=(0, 1, 2)))
    return out


def lpt2_source(delta, box):
    """Second-order (2LPT) density source from the linear density delta.

    delta2(x) = sum_{i<j} [ phi,ii phi,jj - (phi,ij)^2 ], the standard 2LPT
    source (Bouchet et al. 1995; Scoccimarro 1998), built from the second
    derivatives of the linear potential. Returns a real (N,N,N) MLX field;
    differentiable in delta, hence in f_NL (delta2 is quadratic in delta).
    """
    d = _second_derivatives(delta, box)
    pxx, pyy, pzz = d[(0, 0)], d[(1, 1)], d[(2, 2)]
    pxy, pxz, pyz = d[(0, 1)], d[(0, 2)], d[(1, 2)]
    return pxx * pyy + pxx * pzz + pyy * pzz - pxy**2 - pxz**2 - pyz**2


def second_order_displacement(box, cosmo, seed=0, f_NL=0.0, backend="camb"):
    """Second-order (2LPT) displacement Psi2 = grad lap^-1 delta2, z=0 normalized.

    delta2 = lpt2_source(ic.linear_density). Uses the same i k_j / k^2 operator
    as zeldovich_displacement, so div Psi2 = -delta2. Returns (psi2_x, psi2_y,
    psi2_z), each a real float32 (N,N,N) MLX field in Mpc/h; multiply by D2(z)
    for the second-order displacement at redshift z. Differentiable in f_NL.
    """
    N = box.n_mesh
    delta0 = IC.linear_density(box, cosmo, seed=seed, z=0.0, f_NL=f_NL, backend=backend)
    d2k = mx.fft.rfftn(lpt2_source(delta0, box))
    ikx, iky, ikz, inv_k2 = _k_components(box)
    psi_x = mx.fft.irfftn(d2k * ikx * inv_k2, s=(N, N, N), axes=(0, 1, 2))
    psi_y = mx.fft.irfftn(d2k * iky * inv_k2, s=(N, N, N), axes=(0, 1, 2))
    psi_z = mx.fft.irfftn(d2k * ikz * inv_k2, s=(N, N, N), axes=(0, 1, 2))
    return P.as_real(psi_x), P.as_real(psi_y), P.as_real(psi_z)


def displacement(box, cosmo, order=1, seed=0, f_NL=0.0, backend="camb"):
    """Lagrangian displacement field(s) normalized to z=0.

    order=1 returns (Psi1,); order=2 returns (Psi1, Psi2), where Psi1 is the
    Zel'dovich displacement and Psi2 the 2LPT correction. Each Psi is a tuple
    (psi_x, psi_y, psi_z) of real float32 (N,N,N) fields. The integrator scales
    them by the first- and second-order growth factors. Differentiable in f_NL.
    """
    psi1 = zeldovich_displacement(box, cosmo, seed=seed, f_NL=f_NL, backend=backend)
    if order == 1:
        return (psi1,)
    if order == 2:
        psi2 = second_order_displacement(
            box, cosmo, seed=seed, f_NL=f_NL, backend=backend
        )
        return (psi1, psi2)
    raise ValueError(f"order must be 1 or 2 (got {order})")


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
