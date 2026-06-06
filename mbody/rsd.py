"""Redshift-space distortions: map real-space positions to redshift space.

What we observe in a galaxy survey is not the true comoving position but the
redshift, which mixes the Hubble flow with the line-of-sight peculiar velocity.
In the plane-parallel (distant-observer) approximation, a particle at comoving
position x with physical peculiar velocity v_pec is *seen* at

    s = x + (v_pec . n_hat) / (a H(a)) n_hat,

i.e. its line-of-sight coordinate is shifted by its peculiar velocity (Kaiser
1987). Coherent infall onto overdensities (the velocities pointing inward) makes
structure look squashed along the line of sight on large scales -- the Kaiser
effect -- which is exactly the anisotropy the multipole estimator in
mbody.fields picks up.

The momentum -> shift conversion (pinned to mbody's units). The integrator works
in scale-factor time with the momentum rescaled so H0 = 1 (see mbody.integrate),
with the drift dx/da = p / (a^3 E). Pushing the Kaiser shift through that gives a
conversion with no stray h factor (the box is already in Mpc/h):

    v_pec = p / a,      Delta_s_los = v_pec / (a H) = p_los / (a^2 E(a)),

where E(a) = H(a)/H0. So the redshift-space displacement is simply
p_los / (a^2 E) in Mpc/h -- the h factor only appears if one detours through
physical km/s, which we never do. A `f_growth` multiplier (default 1, the
simulation's own growth) scales the shift so it can be treated as a
differentiable growth-rate amplitude parameter in the Fisher forecast.

Differentiability. The shift is linear in the positions, the momenta and
f_growth, and the periodic wrap uses floor (zero gradient), so mx.grad flows to
all three -- and, because the shift depends on the *final momenta*, the f_NL / A
gradients must carry a cotangent on p_final through the reversible adjoint (see
mbody.integrate.adjoint_grad_ic).
"""

import mlx.core as mx
import numpy as np

from mbody import cosmology as C
from mbody import fields as F
from mbody import precision as P


def redshift_space_positions(x, p, box, cosmo, z, los_axis=0, f_growth=1.0):
    """Map real-space positions to redshift space along a fixed line of sight.

    Shifts the `los_axis` component by Delta_s = f_growth * p_los / (a^2 E(a)),
    then wraps periodically into [0, box_size). `x`, `p` are (n_particles, 3) MLX
    arrays (the final state from the integrator); `z` is the snapshot redshift;
    `los_axis` selects the line of sight -- use a full fft axis (0 or 1), matching
    the multipole estimator (see fields._mu_grid). Differentiable in `x`, `p` and
    `f_growth` (pass f_growth as an mx scalar to differentiate it). Returns a new
    (n_particles, 3) array; the transverse components are unchanged.
    """
    a = 1.0 / (1.0 + z)
    E = float(C.E(z, cosmo))
    coef = f_growth / (a**2 * E)  # p_los -> comoving Mpc/h, H0 = 1 units
    L = box.box_size
    cols = [x[:, i] for i in range(3)]
    shifted = cols[los_axis] + coef * p[:, los_axis]
    cols[los_axis] = shifted - L * mx.floor(shifted / L)  # periodic, grad-safe
    return mx.stack(cols, axis=1)


def apply_linear_kaiser(delta, box, b1, f, los_axis=0):
    """Linear redshift-space density field (b1 + f mu^2) delta in real space.

    The Fourier-space Kaiser operator delta_s(k) = (b1 + f mu^2) delta(k) -- the
    analytic large-scale RSD on a *field* (no particles), used as the closed-form
    validation target and as the linear-field milestone in the Fisher (there the
    velocity-divergence term f mu^2 is set by f = f_growth * f_linear). `b1` and
    `f` may be plain floats or mx scalars (differentiable). Returns a real
    (N, N, N) field; differentiable in `delta`, `b1`, `f`.
    """
    N = box.n_mesh
    mu2 = mx.array((F._mu_grid(box, los_axis) ** 2).astype(np.float32))
    factor = P.as_complex(b1 + f * mu2)
    return mx.fft.irfftn(factor * mx.fft.rfftn(delta), s=(N, N, N), axes=(0, 1, 2))
