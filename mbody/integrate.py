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

Two integrators share this KDK structure, selected by `integrator`: "exact"
integrates the background factors literally (a ~2% growth deficit at low step
count), and "fastpm" uses the growth-corrected kick/drift kernels (Feng et al.
2016, fastpm_kick_factor / fastpm_drift_factor) that make a single linear mode
grow exactly as D(a) at any step count. Both are constants of the step, so the
gradient (mx.grad and the reversible adjoint) is unaffected by the choice.
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


# --- FastPM growth-corrected kick/drift (Feng et al. 2016) -------------------
#
# The "exact" kick/drift above integrate the background factors literally and
# pick up a ~2% growth deficit at low step count. FastPM instead chooses the
# coefficients so a single linear (Zel'dovich) mode is integrated *exactly* at
# any step size. mbody's momentum convention already matches FastPM's
# (p = a^3 E D' s = G_f(a) s), so the modified factors translate directly; they
# remain constants of the step (float64 CPU, not in the AD graph), so mx.grad
# and the reversible adjoint are unaffected.


def _D_and_deriv(a, cosmo):
    """Growth factor D(a) and its scale-factor derivative D'(a) = D f / a."""
    z = 1.0 / a - 1.0
    D = C.growth_factor(z, cosmo)
    f = C.growth_rate(z, cosmo)
    return D, D * f / a


def _G_f(a, cosmo):
    """FastPM auxiliary G_f(a) = a^3 E(a) D'(a) -- the momentum of a unit
    Zel'dovich mode at a (equals mbody's IC momentum coefficient a^2 E D f)."""
    _, Dp = _D_and_deriv(a, cosmo)
    return a**3 * float(_E_of_a(a, cosmo)) * Dp


def _g_f(a, cosmo, rel=1e-5):
    """dG_f/da via a central difference (a smooth float64 background quantity)."""
    h = rel * a
    return (_G_f(a + h, cosmo) - _G_f(a - h, cosmo)) / (2.0 * h)


def fastpm_drift_factor(a0, a1, a_r, cosmo):
    """FastPM drift coefficient (Feng et al. 2016, Eq. 24).

    [D(a1) - D(a0)] / [a_r^3 E(a_r) D'(a_r)], with the momentum defined at the
    reference scale a_r. A Zel'dovich mode (momentum a_r^3 E D'(a_r) s) then
    drifts by exactly [D(a1) - D(a0)] s. Reduces to the exact drift integrand
    1/(a^3 E) as a1 -> a0.
    """
    D0, _ = _D_and_deriv(a0, cosmo)
    D1, _ = _D_and_deriv(a1, cosmo)
    _, Dpr = _D_and_deriv(a_r, cosmo)
    return (D1 - D0) / (a_r**3 * float(_E_of_a(a_r, cosmo)) * Dpr)


def fastpm_kick_factor(a0, a1, a_r, cosmo):
    """FastPM kick coefficient (Feng et al. 2016, Eq. 25).

    (3/2) Omega_m [G_f(a1) - G_f(a0)] / [a_r^2 E(a_r) g_f(a_r)], with the force
    evaluated at a_r. Applied to mbody's geometric acceleration g, it advances a
    Zel'dovich mode's momentum exactly from G_f(a0) s to G_f(a1) s. The (3/2)
    Omega_m converts the geometric g into the FastPM force; reduces to the exact
    kick integrand (3/2) Omega_m / (a^2 E) as a1 -> a0.
    """
    num = _G_f(a1, cosmo) - _G_f(a0, cosmo)
    den = a_r**2 * float(_E_of_a(a_r, cosmo)) * _g_f(a_r, cosmo)
    return 1.5 * cosmo.Omega_m * num / den


# --- BullFrog: 2LPT-accurate drift-kick-drift (Rampf, List & Hahn 2024) --------
#
# BullFrog (arXiv:2409.19049, JCAP 2025) is a leapfrog-family integrator whose
# single step is *2LPT-accurate* -- where FastPM's step is only 1LPT/Zel'dovich-
# accurate -- so it converges to the exact solution with far fewer steps. It steps
# in growth-factor time D with velocity v = dx/dD, as a drift-kick-drift with an
# AFFINE kick (the alpha != 1 velocity rescale is what absorbs the 2LPT growth):
#
#     x_mid = x + (dD/2) v
#     v'    = alpha v + (beta / D_mid) g(x_mid)
#     x'    = x_mid + (dD/2) v'
#
# with dD = D(a1) - D(a0), D_mid = D(a0) + dD/2, and g mbody's geometric force
# (= the paper's acceleration A, div A = -delta; for a linear mode g = +D Psi1, so
# it enters the kick with no extra sign). The weights are (paper Eqs 2.3-2.4)
#
#     F_mid = (E0 + E0' dD/2) / D_mid - D_mid
#     alpha = (E1' - F_mid) / (E0' - F_mid),    beta = 1 - alpha,
#
# with E the second-order growth and E' = dE/dD. We use the EdS relation
# E = -(3/7) D^2, E' = -(6/7) D (consistent with the EdS 2LPT IC, cosmology.
# growth_factor_2) evaluated on the EXACT LCDM growth D = cosmology.growth_factor.
# With D propto a this reproduces the paper's published EdS closed form
# alpha = [4n(4n+1)-5]/[4n(4n+7)+7], beta = [24n+12]/[4n(4n+7)+7], n = D0/dD
# (verified algebraically; pinned by test_bullfrog_weights_match_eds_closed_form).
#
# alpha, beta, dD, D_mid are background-only float64 constants of the step, so --
# like FastPM -- mx.grad and the reversible adjoint are unaffected by them; the
# affine step is exactly invertible (alpha != 0), so it is time-reversible. mbody's
# a-time momentum p maps to BullFrog's D-time velocity by v = p / G_f(a) (G_f =
# a^3 E D', _G_f), so the BullFrog path runs on (x, v) internally and converts
# p <-> v at the IC and the output only.


def _bullfrog_weights(D0, D1):
    """BullFrog (alpha, beta, dD, D_mid) for a step with linear growth D0 -> D1.

    EdS second-order growth E = -(3/7)D^2, E' = -(6/7)D; F_mid and the weights are
    paper Eqs 2.4 and 2.3. Pure function of the two growth values, so it is tested
    directly against the published EdS closed form (D propto a) without a cosmology.
    """
    dD = D1 - D0
    D_mid = D0 + 0.5 * dD
    E0 = -(3.0 / 7.0) * D0 * D0
    E0p = -(6.0 / 7.0) * D0
    E1p = -(6.0 / 7.0) * D1
    F_mid = (E0 + E0p * 0.5 * dD) / D_mid - D_mid
    alpha = (E1p - F_mid) / (E0p - F_mid)
    return alpha, 1.0 - alpha, dD, D_mid


def bullfrog_coeffs(a_steps, cosmo):
    """Per-step BullFrog (dD/2, alpha, beta/D_mid) coefficients over the a-grid.

    Background-only float64 constants of the step (so off the AD graph, like the
    FastPM kernels). The drift coefficient is dD/2 (half the growth change), and the
    affine kick is v -> alpha v + (beta/D_mid) g. Returns a list of
    (dD_half, alpha, beta_over_Dmid), one per sub-interval.
    """
    co = []
    for i in range(len(a_steps) - 1):
        D0 = C.growth_factor(1.0 / float(a_steps[i]) - 1.0, cosmo)
        D1 = C.growth_factor(1.0 / float(a_steps[i + 1]) - 1.0, cosmo)
        alpha, beta, dD, D_mid = _bullfrog_weights(D0, D1)
        co.append((0.5 * dD, alpha, beta / D_mid))
    return co


def _bullfrog_forward(x, v, coeff, force_fn, box_size):
    """One BullFrog drift-kick-drift step on (x, v) in D-time. Differentiable."""
    dD_half, alpha, bcoef = coeff
    x = _wrap(x + dD_half * v, box_size)
    g = force_fn(x)  # geometric force at the half-drifted (midpoint) position
    v = alpha * v + bcoef * g
    x = _wrap(x + dD_half * v, box_size)
    return x, v


def _bullfrog_reverse(x, v, coeff, force_fn, box_size):
    """Exact inverse of _bullfrog_forward (reconstructs the previous (x, v))."""
    dD_half, alpha, bcoef = coeff
    x = _wrap(x - dD_half * v, box_size)  # undo the 2nd half-drift -> midpoint
    g = force_fn(x)  # same midpoint position -> same force as the forward step
    v = (v - bcoef * g) / alpha  # invert the affine kick (alpha != 0)
    x = _wrap(x - dD_half * v, box_size)  # undo the 1st half-drift -> previous x
    return x, v


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


def initial_state(
    box, cosmo, time, seed=0, f_NL=0.0, backend="camb", lpt_order=2, amplitude=1.0
):
    """LPT initial conditions (positions and momenta) at z_init.

    Second order (2LPT, the default): x = q + D1 Psi1 - D2 Psi2, with the
    growing-mode velocity p = a_i^2 E (D1 f1 Psi1 - D2 f2 Psi2); the D2 Psi2 term
    curbs the Zel'dovich early-time transient. First order (Zel'dovich,
    lpt_order=1): just x = q + D1 Psi1, p = a_i^2 E D1 f1 Psi1. The default
    matches the SimConfig default (the loose and config APIs agree). Both returns
    are (n_particles^3, 3) float32 arrays; momenta are in the H0 = 1 units used by
    the stepper. `f_NL` (an mx scalar when differentiating) flows into the
    displacement via ic.linear_density, so the whole evolved state is
    differentiable in f_NL. `amplitude` (a differentiable sigma8 / A_s proxy,
    default 1) scales the linear IC density, so it propagates through LPT and the
    PM evolution as a primordial amplitude (Psi1 ~ amplitude, Psi2 ~ amplitude^2).
    """
    a_i = 1.0 / (1.0 + time.z_init)
    E = float(_E_of_a(a_i, cosmo))
    D = C.growth_factor(time.z_init, cosmo)
    f = C.growth_rate(time.z_init, cosmo)

    if lpt_order == 1:
        psi = L.zeldovich_displacement(
            box, cosmo, seed=seed, f_NL=f_NL, backend=backend, amplitude=amplitude
        )
        x = L.displace(box, psi, D)
        p_coef = a_i**2 * E * D * f
        p = mx.stack([(p_coef * psi[c]).reshape(-1) for c in range(3)], axis=1)
        return x, p
    if lpt_order == 2:
        psi1, psi2 = L.displacement(
            box,
            cosmo,
            order=2,
            seed=seed,
            f_NL=f_NL,
            backend=backend,
            amplitude=amplitude,
        )
        D2 = C.growth_factor_2(time.z_init, cosmo)
        f2 = C.growth_rate_2(time.z_init, cosmo)
        # mbody's Psi is +grad lap^-1 delta (so div Psi = -delta), the opposite
        # potential sign to the textbook 2LPT x = q - D1 grad psi1 + D2 grad psi2.
        # That flips the sign of the second-order term: the coefficient of Psi2
        # is -D2 = +(3/7) D1^2. Validated by the skewness test (sign = collapse).
        dx = [D * psi1[c] - D2 * psi2[c] for c in range(3)]
        x = L.displace(box, dx, 1.0)
        p_coef = a_i**2 * E
        p = mx.stack(
            [
                (p_coef * (D * f * psi1[c] - D2 * f2 * psi2[c])).reshape(-1)
                for c in range(3)
            ],
            axis=1,
        )
        return x, p
    raise ValueError(f"lpt_order must be 1 or 2 (got {lpt_order})")


def evolve_state(
    x, p, box, cosmo, a_steps, snapshot=None, force_fn=None, integrator="exact"
):
    """Leapfrog (KDK) from a_steps[0] to a_steps[-1], given initial (x, p).

    Pure function of the initial state -- mx.grad flows through it. One force
    solve per step (consecutive half kicks reuse the same force). `integrator`
    selects the kick/drift coefficients ("exact" or growth-corrected "fastpm").
    `force_fn` is the callable that maps positions to the geometric acceleration;
    pass one from `forces.make_force_fn` to opt into the compiled and/or
    gradient-checkpointed solve. It defaults to the plain eager force, so the
    kick/drift scalars, the periodic wrap, and the snapshot callback all stay in
    eager Python -- only the force solve is ever fused/checkpointed. The optional
    `snapshot(step, a, x, p)` callback runs after each step (and once at the
    start, step 0); it is off the AD path, so use it only for forward runs.
    Returns the final (x, p).
    """
    if force_fn is None:
        force_fn = FO.make_force_fn(box)
    if integrator == "bullfrog":
        return _evolve_bullfrog(x, p, box, cosmo, a_steps, snapshot, force_fn)
    g = force_fn(x)  # geometric acceleration at the start
    if snapshot is not None:
        snapshot(0, float(a_steps[0]), x, p)

    for i, (k1, dr, k2) in enumerate(_step_coeffs(a_steps, cosmo, integrator)):
        p = p + k1 * g  # half kick (force at x_i)
        x = _wrap(x + dr * p, box.box_size)  # drift
        g = force_fn(x)  # force at x_{i+1}
        p = p + k2 * g  # half kick (force at x_{i+1}, reused next step)
        if snapshot is not None:
            snapshot(i + 1, float(a_steps[i + 1]), x, p)
    return x, p


def _evolve_bullfrog(x, p, box, cosmo, a_steps, snapshot, force_fn):
    """BullFrog (drift-kick-drift) forward pass; differentiable like evolve_state.

    Runs on the D-time velocity v = p / G_f(a), converting the mbody a-time momentum
    in at the start and back out at the end (and for each snapshot). One force solve
    per step (at the midpoint); there is no half-kick to share across steps.
    """
    a_i = float(a_steps[0])
    v = p / _G_f(a_i, cosmo)  # a-time momentum -> D-time velocity
    if snapshot is not None:
        snapshot(0, a_i, x, p)
    for i, coeff in enumerate(bullfrog_coeffs(a_steps, cosmo)):
        x, v = _bullfrog_forward(x, v, coeff, force_fn, box.box_size)
        if snapshot is not None:
            a1 = float(a_steps[i + 1])
            snapshot(i + 1, a1, x, v * _G_f(a1, cosmo))
    return x, v * _G_f(float(a_steps[-1]), cosmo)


def leapfrog(
    box,
    cosmo,
    time,
    seed=0,
    f_NL=0.0,
    backend="camb",
    spacing="linear",
    snapshot=None,
    compiled=False,
    memory_mode=None,
    integrator=None,
    lpt_order=2,
    amplitude=1.0,
):
    """Evolve LPT initial conditions to z_final with the PM leapfrog.

    Convenience wrapper: build the initial state and step it. Returns the final
    (positions, momenta), each (n_particles^3, 3) float32. Pass `f_NL` to inject
    local non-Gaussianity (differentiable in f_NL); pass `snapshot` to capture
    the trajectory for diagnostics / animation.

    Performance knobs (both off by default, so behaviour is unchanged):
    - `compiled`: kernel-fuse the force solve via mx.compile. Numerically
      identical to the eager solve; ~1x on this FFT-bound model (the Metal FFTs
      dominate, leaving little elementwise work to fuse), but harmless and the
      right plumbing.
    - `memory_mode`: how a *subsequent* mx.grad over this call manages memory;
      defaults to `time.memory_mode`. "replay" keeps the full unrolled graph
      (memory ~ grid x steps). "checkpoint" wraps the force in mx.checkpoint;
      grad is identical to replay but, measured, it does NOT reduce peak memory
      here (the solve is near-linear, so reverse-mode retains little to
      recompute-away). For an O(1)-in-steps gradient use the reversible adjoint,
      `integrate.adjoint_grad_fnl` -- a separate eager code path, not mx.grad.
    """
    if integrator is None:
        integrator = time.integrator
    x0, p0 = initial_state(
        box,
        cosmo,
        time,
        seed=seed,
        f_NL=f_NL,
        backend=backend,
        lpt_order=lpt_order,
        amplitude=amplitude,
    )
    steps = a_grid(time, spacing)
    if memory_mode is None:
        memory_mode = time.memory_mode
    if memory_mode == "adjoint":
        raise ValueError(
            "memory_mode='adjoint' is a gradient strategy, not a forward mode: "
            "compute adjoint gradients with integrate.adjoint_grad_fnl(...), and "
            "run the forward itself with memory_mode='replay'."
        )
    force_fn = FO.make_force_fn(
        box, compiled=compiled, checkpoint=(memory_mode == "checkpoint")
    )
    return evolve_state(
        x0,
        p0,
        box,
        cosmo,
        steps,
        snapshot=snapshot,
        force_fn=force_fn,
        integrator=integrator,
    )


def _step_coeffs(a_steps, cosmo, integrator="exact"):
    """Per-step KDK coefficients (k1, drift, k2) for each sub-interval.

    k1 kicks over [a0, a_mid], drift translates over [a0, a1], k2 kicks over
    [a_mid, a1] -- float64-CPU constants of the step. Precomputed once so the
    forward stepper and the adjoint use identical numbers. `integrator` selects
    "exact" (literal background integrals) or "fastpm" (growth-corrected kernels
    that make a linear mode exact at any step count). For FastPM the reference
    scale a_r is the force-evaluation time for the kicks (a0 for the first half
    kick, a1 for the second) and the midpoint a_c for the drift.
    """
    co = []
    for i in range(len(a_steps) - 1):
        a0, a1 = float(a_steps[i]), float(a_steps[i + 1])
        a_c = 0.5 * (a0 + a1)
        if integrator == "exact":
            co.append(
                (
                    kick_factor(a0, a_c, cosmo),
                    drift_factor(a0, a1, cosmo),
                    kick_factor(a_c, a1, cosmo),
                )
            )
        elif integrator == "fastpm":
            co.append(
                (
                    fastpm_kick_factor(a0, a_c, a0, cosmo),
                    fastpm_drift_factor(a0, a1, a_c, cosmo),
                    fastpm_kick_factor(a_c, a1, a1, cosmo),
                )
            )
        else:
            raise ValueError(f"unknown integrator {integrator!r}")
    return co


def _make_steppers(box, force_fn):
    """A self-contained KDK step and its exact reverse, for the adjoint.

    Unlike evolve_state (which reuses one force solve across the shared half
    kicks of adjacent steps), each step here recomputes the force, so a step is
    an invertible map of (x, p) alone: reverse_step reconstructs the previous
    (x, p) from the current one. The leapfrog is time-reversible up to the
    float32 CIC scatter-add floor (reconstruction drift measured ~1e-4 cells at
    n_mesh=64, ~flat in step count). Returns (one_step, reverse_step).
    """
    box_size = box.box_size

    def one_step(x, p, k1, dr, k2):
        p = p + k1 * force_fn(x)
        x = _wrap(x + dr * p, box_size)
        p = p + k2 * force_fn(x)
        return x, p

    def reverse_step(x, p, k1, dr, k2):
        p = p - k2 * force_fn(x)
        x = _wrap(x - dr * p, box_size)
        p = p - k1 * force_fn(x)
        return x, p

    return one_step, reverse_step


def evolve_eager(x, p, box, cosmo, a_steps, force_fn=None, integrator="exact"):
    """Forward leapfrog evaluated step by step, keeping only the final state.

    Materializes (mx.eval) and releases each step, so it runs in O(1) memory in
    the step count -- where evolve_state under mx.grad unrolls the graph. The
    force is recomputed each step (no half-kick reuse) so a step is exactly
    reversible; this is the forward pass the adjoint gradient builds on. Returns
    the final (x, p). For a differentiable replay forward, use evolve_state.
    """
    if force_fn is None:
        force_fn = FO.make_force_fn(box)
    if integrator == "bullfrog":
        a_i = float(a_steps[0])
        v = p / _G_f(a_i, cosmo)
        mx.eval(x, v)
        for coeff in bullfrog_coeffs(a_steps, cosmo):
            x, v = _bullfrog_forward(x, v, coeff, force_fn, box.box_size)
            mx.eval(x, v)
        return x, v * _G_f(float(a_steps[-1]), cosmo)
    one_step, _ = _make_steppers(box, force_fn)
    for k1, dr, k2 in _step_coeffs(a_steps, cosmo, integrator):
        x, p = one_step(x, p, k1, dr, k2)
        mx.eval(x, p)
    return x, p


def _reversible_ic_grad(
    loss_field,
    ic_fn,
    theta,
    box,
    cosmo,
    time,
    spacing,
    compiled,
    integrator,
    loss_uses_momentum=False,
    recompute_cic=True,
):
    """Reversible-leapfrog adjoint: d loss_field(x_final[, p_final]) / d theta.

    `ic_fn(theta) -> [x0, p0]` builds the initial state from the IC parameter(s)
    theta. The four eager, O(grid)-memory stages are: build the IC, run the
    leapfrog forward keeping only the final state, seed the backward sweep with
    the loss cotangent at the final state, then walk the leapfrog backward
    (reconstructing each state by reverse-stepping) and push the initial-state
    cotangent through ic_fn back to theta. Peak memory is independent of the step
    count. The reverse sweep does not depend on theta, so a vector theta gets
    every gradient from the single final IC vjp -- shared by adjoint_grad_fnl and
    adjoint_grad_ic.

    With loss_uses_momentum=True the loss is `loss_field(x_final, p_final)` (the
    a-time momentum), needed for redshift-space summaries that depend on the final
    velocities: the momentum cotangent is then seeded from the loss instead of
    zero, so the f_NL / amplitude gradient correctly carries the velocity's
    dependence on the initial conditions through the trajectory. (Both the KDK and
    BullFrog reverse sweeps already propagate the momentum cotangent.)
    """
    if integrator is None:
        integrator = time.integrator
    a_steps = a_grid(time, spacing)
    force_fn = FO.make_force_fn(box, compiled=compiled, recompute_cic=recompute_cic)

    # Per-integrator steppers and the IC->state map. For BullFrog the trajectory
    # state is (x, v) in D-time, so ic_mom converts the IC momentum p0 -> v0 =
    # p0/G_f(a_i) (a differentiable scalar, handled by the final IC vjp); for KDK
    # the state is (x, p) and ic_mom is the IC unchanged. one_step/reverse_step take
    # (x, mom, coeff) so the sweep below is integrator-agnostic.
    if integrator == "bullfrog":
        co = bullfrog_coeffs(a_steps, cosmo)
        gf_i = _G_f(float(a_steps[0]), cosmo)

        def ic_mom(th):
            x0, p0 = ic_fn(th)
            return [x0, p0 / gf_i]

        def one_step(x, m, c):
            return _bullfrog_forward(x, m, c, force_fn, box.box_size)

        def reverse_step(x, m, c):
            return _bullfrog_reverse(x, m, c, force_fn, box.box_size)

    else:
        co = _step_coeffs(a_steps, cosmo, integrator)
        ic_mom = ic_fn
        kdk_one, kdk_rev = _make_steppers(box, force_fn)

        def one_step(x, m, c):
            return kdk_one(x, m, *c)

        def reverse_step(x, m, c):
            return kdk_rev(x, m, *c)

    # 1-2. IC, then eager forward keeping only the final state.
    x, m = ic_mom(theta)
    mx.eval(x, m)
    for c in co:
        x, m = one_step(x, m, c)
        mx.eval(x, m)

    # 3. cotangent of the loss at the final state. For a positional loss the
    # momentum cotangent is zero; for a momentum-dependent (redshift-space) loss
    # seed it too, converting the trajectory momentum m to the a-time momentum p
    # (m is p for KDK, the D-time velocity v = p/G_f for BullFrog, so p = m*G_f).
    if loss_uses_momentum:
        gf_final = _G_f(float(a_steps[-1]), cosmo) if integrator == "bullfrog" else 1.0

        def loss_xm(xx, mm):
            return loss_field(xx, mm * gf_final)

        _, (gx, gm) = mx.vjp(loss_xm, [x, m], [mx.array(1.0)])
    else:
        _, (gx,) = mx.vjp(loss_field, [x], [mx.array(1.0)])
        gm = mx.zeros_like(m)
    mx.eval(gx, gm)

    # 4. reverse adjoint sweep -- O(1) memory in the step count.
    for c in reversed(co):
        x_prev, m_prev = reverse_step(x, m, c)
        _, (gx, gm) = mx.vjp(
            lambda a, b, c=c: one_step(a, b, c), [x_prev, m_prev], [gx, gm]
        )
        x, m = x_prev, m_prev
        mx.eval(x, m, gx, gm)

    # 5. push the (x0, mom0) cotangent through the IC map back to the parameters.
    _, (gtheta,) = mx.vjp(ic_mom, [theta], [gx, gm])
    return gtheta


def adjoint_grad_fnl(
    loss_field,
    box,
    cosmo,
    time,
    seed=0,
    f_NL=0.0,
    backend="camb",
    spacing="linear",
    compiled=False,
    integrator=None,
    lpt_order=2,
    recompute_cic=True,
):
    """Gradient d loss_field(x_final) / d f_NL via the reversible-leapfrog adjoint.

    `loss_field(x)` maps the final particle positions (n_particles^3, 3) to a
    scalar mx.array (e.g. one band power of the CIC-painted local-bias tracer).
    The gradient is assembled from eager, O(grid)-memory pieces:

      1. build the Zel'dovich IC (x0, p0) from f_NL;
      2. run the leapfrog forward step by step, keeping only the final (x, p);
      3. seed the backward sweep with the cotangent of loss_field at x_final
         (final momenta do not enter a positional summary, so their cotangent
         is zero);
      4. walk the leapfrog *backward*: reconstruct each (x, p) by reverse
         stepping and apply the single-step VJP, then push the resulting
         (x0, p0) cotangent through the IC back to f_NL.

    Every piece is materialized and released as it goes, so peak memory is
    independent of the step count: the adjoint trades the unrolled autodiff graph
    (memory ~ grid x steps) for ~2x the compute (memory ~ grid). This is the
    lever for many-step / high-resolution gradients; mx.grad over `leapfrog`
    (replay) is simpler but its memory grows with step x grid. By default the CIC
    paint/read are checkpointed (recompute_cic=True): ~18% less peak memory for an
    exact gradient; pass recompute_cic=False to disable.

    Returns the gradient as a 0-d mx.array. For a vector statistic (a P(k) over
    bins) call once per component, with `loss_field` selecting that component;
    each call is an independent O(grid)-memory sweep.

    Caveat: reversibility is exact only up to the float32 CIC scatter-add floor;
    the adjoint gradient matches the replay mx.grad to ~1e-6 relative (measured),
    far below any dP/df_NL signal, but it is not bit-identical. Unlike mx.grad
    this entry point is eager and not composable inside an outer transform.
    """
    if not isinstance(f_NL, mx.array):
        f_NL = mx.array(f_NL)

    def ic_fn(f):
        x0, p0 = initial_state(
            box, cosmo, time, seed=seed, f_NL=f, backend=backend, lpt_order=lpt_order
        )
        return [x0, p0]

    return _reversible_ic_grad(
        loss_field,
        ic_fn,
        f_NL,
        box,
        cosmo,
        time,
        spacing,
        compiled,
        integrator,
        recompute_cic=recompute_cic,
    )


def adjoint_grad_ic(
    loss_field,
    box,
    cosmo,
    time,
    seed=0,
    f_NL=0.0,
    amplitude=1.0,
    backend="camb",
    spacing="linear",
    compiled=False,
    integrator=None,
    lpt_order=2,
    loss_uses_momentum=False,
    recompute_cic=True,
):
    """Gradient of loss_field(x_final[, p_final]) w.r.t. the IC params (f_NL, A).

    The multi-parameter generalization of adjoint_grad_fnl. Both parameters are
    IC-stage -- they shape the initial conditions the leapfrog then evolves -- so
    they SHARE the trajectory adjoint: one O(grid)-memory reverse sweep yields the
    initial-state cotangent, and a single IC vjp returns both gradients at once.
    The cost is therefore the same one sweep as adjoint_grad_fnl, NOT 2x (the
    trajectory reverse-stepping is independent of how many IC parameters there
    are; only the final, cheap IC vjp sees both).

    With loss_uses_momentum=True the loss is `loss_field(x_final, p_final)` (the
    final a-time momenta), for a redshift-space summary whose value depends on the
    peculiar velocities -- the velocity's dependence on f_NL / A then flows through
    the trajectory too (see _reversible_ic_grad).

    Returns a length-2 mx.array [d loss / d f_NL, d loss / d amplitude]. For a
    vector statistic (a P(k) over bins) call once per component. The memory and
    reversibility caveats are those of adjoint_grad_fnl: eager, O(grid) memory,
    ~2x compute, grad matching replay mx.grad to ~1e-6, not mx.grad-composable.
    """
    theta = mx.array([float(f_NL), float(amplitude)])

    def ic_fn(t):
        x0, p0 = initial_state(
            box,
            cosmo,
            time,
            seed=seed,
            f_NL=t[0],
            backend=backend,
            lpt_order=lpt_order,
            amplitude=t[1],
        )
        return [x0, p0]

    return _reversible_ic_grad(
        loss_field,
        ic_fn,
        theta,
        box,
        cosmo,
        time,
        spacing,
        compiled,
        integrator,
        loss_uses_momentum=loss_uses_momentum,
        recompute_cic=recompute_cic,
    )
