"""The PM time-stepper: a leapfrog in scale-factor time.

This is what finally makes the universe move. The force module hands us the
acceleration g = -grad phi of the current density field; the integrator marches
the particles under it from the initial redshift to z = 0, and the cosmic web
sharpens out of the Zel'dovich sheet.

Time variable. We integrate in the scale factor a (not cosmic time t), the
standard choice for cosmological PM. Writing the comoving equations of motion in
a, and rescaling the momentum so the Hubble constant H0 drops out (units where
H0 = 1), gives a strikingly clean pair:

    dx/da = p / (a^3 E(a)),                  (drift)
    dp/da = (3/2) Omega_m g(x) / (a^2 E(a)), (kick)

with E(a) = H(a)/H0 = sqrt(Omega_m a^-3 + Omega_Lambda) and g the *geometric*
acceleration from mbody.forces (the solution of laplacian phi = delta). Every
piece of cosmology -- the (3/2) Omega_m and all of the a-dependence -- sits in
the kick and drift coefficients, so forces.py stays purely geometric.

Leapfrog (kick-drift-kick). Over one step a_i -> a_{i+1} with midpoint a_c we do
a half kick, a full drift, then a half kick, where each coefficient is the exact
integral of the equation-of-motion factor over the sub-interval:

    K(a, a')  = integral_a^a' (3/2) Omega_m / (a^2 E(a)) da    (kick factor)
    Dr(a, a') = integral_a^a'           1 / (a^3 E(a)) da      (drift factor)

These background integrals are evaluated in float64 on the CPU (a precision
island); they are constants of the step, so they never enter the autodiff graph.
Consecutive half kicks share the same force evaluation, so the cost is one force
solve (one FFT Poisson solve) per step -- which is also what sets the
reverse-mode memory, since the graph unrolls over steps.

Initial velocity. The particles start on the Zel'dovich growing mode, so the
momentum is set to a^2 E D f Psi_0 (with Psi_0 the z=0-normalized displacement,
D the growth factor and f = dlnD/dlna the growth rate). This is the velocity
that makes a linear mode grow as pure D(a) with no decaying-mode transient -- a
sharp check that the IC and the stepper agree.

Differentiability. The stepping core (evolve_state) is a pure function of the
initial (x, p); mx.grad flows through every kick, drift, force solve and CIC
operation across all steps. The optional `snapshot` callback is for diagnostics
and the structure-formation animation only -- it is off the autodiff path (its
return value is ignored), so pass snapshot=None when differentiating.

This is the exact-background leapfrog (stage 1). The FastPM growth-corrected
kick/drift kernels (Feng et al. 2016), which reproduce linear growth at very low
step count, are a documented next layer to be checked against this baseline.
"""

import mlx.core as mx
import numpy as np
from scipy.integrate import quad

from mbody import cosmology as C
from mbody import forces as FO
from mbody import lpt as L


def _E_of_a(a, cosmo):
    """Dimensionless Hubble rate E(a) = H(a)/H0 for flat LCDM (no radiation)."""
    return np.sqrt(cosmo.Omega_m / a**3 + cosmo.Omega_Lambda)


def kick_factor(a0, a1, cosmo):
    """Leapfrog kick coefficient, integral of (3/2) Omega_m / (a^2 E) da.

    Multiplies the geometric acceleration g to advance the momentum. Computed
    in float64 on the CPU; a constant of the step (not part of the AD graph).
    """
    val, _ = quad(lambda a: 1.5 * cosmo.Omega_m / (a**2 * _E_of_a(a, cosmo)), a0, a1)
    return val


def drift_factor(a0, a1, cosmo):
    """Leapfrog drift coefficient, integral of 1 / (a^3 E) da.

    Multiplies the momentum to advance the position. Float64 CPU; constant of
    the step.
    """
    val, _ = quad(lambda a: 1.0 / (a**3 * _E_of_a(a, cosmo)), a0, a1)
    return val


def a_grid(time, spacing="linear"):
    """Step boundaries in scale factor a, from z_init to z_final.

    `spacing` is "linear" (equal steps in a) or "log" (equal steps in ln a,
    finer at early times). Returns an (n_steps + 1,) float64 numpy array.
    """
    a_i = 1.0 / (1.0 + time.z_init)
    a_f = 1.0 / (1.0 + time.z_final)
    if spacing == "linear":
        return np.linspace(a_i, a_f, time.n_steps + 1)
    if spacing == "log":
        return np.exp(np.linspace(np.log(a_i), np.log(a_f), time.n_steps + 1))
    raise ValueError(f"spacing must be 'linear' or 'log' (got {spacing!r})")


def _wrap(x, box_size):
    """Periodic wrap of positions into [0, box_size)."""
    return x - box_size * mx.floor(x / box_size)


def initial_state(box, cosmo, time, seed=0, backend="camb"):
    """Zel'dovich initial conditions (positions and momenta) at z_init.

    Positions x = q + D(a_i) Psi_0; momenta p = a_i^2 E(a_i) D(a_i) f(a_i) Psi_0
    (the growing-mode velocity). Both are (n_particles^3, 3) float32 arrays;
    momenta are in the H0 = 1 units used by the stepper.
    """
    psi = L.zeldovich_displacement(box, cosmo, seed=seed, backend=backend)
    a_i = 1.0 / (1.0 + time.z_init)
    D = C.growth_factor(time.z_init, cosmo)
    f = C.growth_rate(time.z_init, cosmo)
    E = float(_E_of_a(a_i, cosmo))

    x = L.displace(box, psi, D)
    p_coef = a_i**2 * E * D * f
    px = (p_coef * psi[0]).reshape(-1)
    py = (p_coef * psi[1]).reshape(-1)
    pz = (p_coef * psi[2]).reshape(-1)
    p = mx.stack([px, py, pz], axis=1)
    return x, p


def evolve_state(x, p, box, cosmo, a_steps, snapshot=None):
    """Leapfrog (KDK) from a_steps[0] to a_steps[-1], given initial (x, p).

    Pure function of the initial state -- mx.grad flows through it. One force
    solve per step (consecutive half kicks reuse the same force). The optional
    `snapshot(step, a, x, p)` callback runs after each step (and once at the
    start, step 0); it is off the AD path, so use it only for forward runs.
    Returns the final (x, p).
    """
    g = FO.forces_on_particles(x, box)  # geometric acceleration at the start
    if snapshot is not None:
        snapshot(0, float(a_steps[0]), x, p)

    for i in range(len(a_steps) - 1):
        a0, a1 = float(a_steps[i]), float(a_steps[i + 1])
        a_c = 0.5 * (a0 + a1)
        p = p + kick_factor(a0, a_c, cosmo) * g  # half kick (force at x_i)
        x = _wrap(x + drift_factor(a0, a1, cosmo) * p, box.box_size)  # drift
        g = FO.forces_on_particles(x, box)  # force at x_{i+1}
        p = p + kick_factor(a_c, a1, cosmo) * g  # half kick
        if snapshot is not None:
            snapshot(i + 1, a1, x, p)
    return x, p


def leapfrog(box, cosmo, time, seed=0, backend="camb", spacing="linear", snapshot=None):
    """Evolve Zel'dovich initial conditions to z_final with the PM leapfrog.

    Convenience wrapper: build the initial state and step it. Returns the final
    (positions, momenta), each (n_particles^3, 3) float32. Pass `snapshot` to
    capture the trajectory for diagnostics / animation.
    """
    x0, p0 = initial_state(box, cosmo, time, seed=seed, backend=backend)
    steps = a_grid(time, spacing)
    return evolve_state(x0, p0, box, cosmo, steps, snapshot=snapshot)
