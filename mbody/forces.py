"""The gravitational force: solving Poisson's equation on the mesh.

This is the dynamical heart of a particle-mesh code. The particles carry the
mass; gravity tells them how to move. In an expanding universe the peculiar
gravitational potential phi obeys the comoving Poisson equation

    laplacian phi = (3/2) Omega_m H0^2 a^-1 delta,

i.e. the source of gravity is the density contrast delta = rho/rho_mean - 1
(an overdensity pulls, an underdensity pushes). A particle feels the
acceleration g = -grad phi. Solving a differential equation like this on a
periodic mesh is exactly what the FFT is for: under a Fourier transform the
Laplacian is just multiplication by -k^2, so the differential equation collapses
to algebra,

    phi_k = -(source)_k / k^2,      g_k = -i k phi_k = i k (source)_k / k^2,

with the k = 0 (mean) mode set to zero -- there is no net force on a periodic
box, which is also momentum conservation.

This module deliberately splits the *geometry* from the *cosmology*:

- Here we solve the dimensionless Poisson equation  laplacian phi = delta  and
  return the geometric acceleration  g = -grad phi. In Fourier space that is
  g_j(k) = i k_j delta_k / k^2 -- which is *exactly* the Zel'dovich displacement
  kernel from mbody.lpt. The linear gravitational force and the Zel'dovich
  displacement are the same Green's-function operation: this is why first-order
  LPT works at all, and it gives an exact machine-precision cross-check
  (acceleration_field on a realization equals its Zel'dovich displacement).

- The cosmological prefactor (3/2) Omega_m H0^2 a^-1 and the velocity/time
  convention live in mbody.integrate (next), where the leapfrog kick uses them.
  Keeping forces.py purely geometric makes it trivially testable and decouples
  it from the choice of time variable.

The full chain is: particles -> CIC paint -> delta on the mesh -> rfftn ->
multiply by the i k / k^2 kernel -> irfftn -> acceleration mesh -> CIC read
back to each particle. Every step is differentiable (the kernel is linear; CIC
detaches only the integer cell index), so mx.grad reaches from a particle's
final position all the way back through the force -- the property the whole
project is built on.

Refinements left for later (documented, not yet implemented): a
finite-difference gradient kernel (i sin(k d)/d) for exact momentum
conservation in place of the spectral i k, and deconvolution of the CIC
assignment window. The spectral kernel used here is the exact analogue of the
LPT module and is more than accurate enough for a toy.
"""

import functools

import mlx.core as mx

from mbody import painting as PA
from mbody import precision as P
from mbody.lpt import _k_components  # the i k / k^2 kernel, shared with LPT


def potential(delta, box):
    """Peculiar potential phi solving the dimensionless Poisson eq lap phi = delta.

    In Fourier space phi_k = -delta_k / k^2 (the k = 0 mode set to zero).
    Returns a real float32 (N, N, N) mesh. This is mainly a diagnostic /
    visualization field -- the dynamics only needs its gradient, which
    acceleration_field computes directly.
    """
    N = box.n_mesh
    _, _, _, inv_k2 = _k_components(box)
    dk = mx.fft.rfftn(delta)
    phi_k = -dk * inv_k2
    return P.as_real(mx.fft.irfftn(phi_k, s=(N, N, N), axes=(0, 1, 2)))


def acceleration_field(delta, box):
    """Geometric acceleration g = -grad phi on the mesh, for lap phi = delta.

    In Fourier space g_j(k) = i k_j delta_k / k^2 -- identical to the Zel'dovich
    displacement kernel. Returns (gx, gy, gz), each a real float32 (N, N, N)
    mesh, in unit-source (geometric) units. The integrator multiplies by the
    cosmological prefactor (3/2) Omega_m H0^2 / a to get a physical
    acceleration.
    """
    N = box.n_mesh
    ikx, iky, ikz, inv_k2 = _k_components(box)
    dk = mx.fft.rfftn(delta)
    src = dk * inv_k2  # delta_k / k^2 (k = 0 already neutralized in inv_k2)
    gx = mx.fft.irfftn(src * ikx, s=(N, N, N), axes=(0, 1, 2))
    gy = mx.fft.irfftn(src * iky, s=(N, N, N), axes=(0, 1, 2))
    gz = mx.fft.irfftn(src * ikz, s=(N, N, N), axes=(0, 1, 2))
    return P.as_real(gx), P.as_real(gy), P.as_real(gz)


def forces_on_particles(positions, box, delta=None):
    """Acceleration (force per unit mass) on each particle.

    Paints the particles to a CIC density contrast (unless `delta` is supplied),
    solves Poisson on the mesh, and reads the acceleration field back to each
    particle by CIC interpolation -- the same trilinear weights as paint, so the
    paint/solve/read chain is a consistent operator pair. Returns an
    (n_particles, 3) float32 array, differentiable in `positions`. Geometric
    (unit-source) units; the integrator applies the cosmological amplitude.
    """
    if delta is None:
        delta = PA.density_contrast(positions, box)
    gx, gy, gz = acceleration_field(delta, box)
    ax = PA.cic_read(gx, positions, box)
    ay = PA.cic_read(gy, positions, box)
    az = PA.cic_read(gz, positions, box)
    return mx.stack([ax, ay, az], axis=1)


@functools.lru_cache(maxsize=None)
def _compiled_force_fn(box):
    """An mx.compile of the paint -> solve -> read force chain for a fixed box.

    Kernel-fuses the whole solve into one graph. The box-derived Fourier kernels
    (the i k / k^2 arrays from _k_components) become baked-in constants of the
    compiled graph, so a separate compiled function must be cached per box --
    hence the lru_cache keyed on the (frozen, hashable) BoxConfig. A change in
    particle count re-traces automatically, because mx.compile keys its own
    cache on input shape; only the box-constant kernels need the manual cache.
    """

    def _force(positions):
        return forces_on_particles(positions, box)

    return mx.compile(_force)


def make_force_fn(box, compiled=False, checkpoint=False):
    """Build the per-step force callable f(positions) -> (n_particles, 3) accel.

    This is what the leapfrog calls once per step. Two orthogonal wrappers:

    - `compiled`: return the mx.compile'd (kernel-fused) solve from
      `_compiled_force_fn` -- numerically identical to the eager solve up to
      float32 reassociation (the same round-off floor as the CIC scatter-add,
      well below any dP/df_NL signal). Measured ~1x on this FFT-bound model: the
      Metal FFTs dominate, so there is little elementwise work to fuse. Kept
      because it is correct, harmless, and may help as the elementwise share
      grows (2LPT, richer bias).
    - `checkpoint`: wrap in mx.checkpoint so the solve's intermediates are
      recomputed in the backward pass. Grad is identical to replay, but -- MEASURED
      -- this does NOT reduce reverse-mode peak memory here, at any granularity.
      The PM solve is near-linear (FFT, kernel multiply, scatter/gather), and the
      VJP of a linear op needs no saved forward activation, so MLX's lazy graph
      already keeps per-step memory minimal: there is nothing to recompute-away.
      The real O(1)-in-steps memory lever is the reversible adjoint
      (integrate.adjoint_grad_fnl), not checkpointing. Kept as correct plumbing
      tied to TimeStepping.memory_mode. Compose order is checkpoint-of-compiled.

    With both flags False this is exactly the eager `forces_on_particles`, so the
    default leapfrog behaviour is unchanged.
    """
    if compiled:
        fn = _compiled_force_fn(box)
    else:

        def fn(positions):
            return forces_on_particles(positions, box)

    if checkpoint:
        fn = mx.checkpoint(fn)
    return fn
