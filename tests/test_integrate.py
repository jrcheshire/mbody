"""Tests for mbody.integrate: the scale-factor leapfrog PM stepper.

The load-bearing check is linear growth: on the largest scales the evolved field
must grow as the linear growth factor D(a), and the plain leapfrog must converge
toward it as the step count rises (the residual at low step count is the
expected discretization error that FastPM kernels later remove). Plus the
background kick/drift integrals, the Zel'dovich growing-mode velocity IC,
determinism, the snapshot callback, and that mx.grad flows through the whole
unrolled trajectory.
"""

import numpy as np
import mlx.core as mx

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import cosmology as C
from mbody import fields as F
from mbody import forces as FO
from mbody import lpt as L
from mbody import painting as PA
from mbody import integrate as IG

COSMO = Cosmology()
SMALL = BoxConfig(box_size=128.0, n_mesh=16, n_particles=16)


def test_a_grid_endpoints_and_spacing():
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10)
    for spacing in ("linear", "log"):
        ag = IG.a_grid(t, spacing=spacing)
        assert ag.shape == (11,)
        assert abs(ag[0] - 0.1) < 1e-12  # a = 1/(1+9)
        assert abs(ag[-1] - 1.0) < 1e-12  # a = 1/(1+0)
        assert np.all(np.diff(ag) > 0)
    lin = IG.a_grid(t, "linear")
    assert np.allclose(np.diff(lin), lin[1] - lin[0])  # equal steps in a


def test_kick_drift_factors_positive_and_additive():
    a0, ac, a1 = 0.2, 0.35, 0.5
    for fac in (IG.kick_factor, IG.drift_factor):
        assert fac(a0, a1, COSMO) > 0
        whole = fac(a0, a1, COSMO)
        split = fac(a0, ac, COSMO) + fac(ac, a1, COSMO)
        assert abs(split - whole) / whole < 1e-6  # integral is additive


def test_initial_state_positions_and_growing_mode_velocity():
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10)
    x, p = IG.initial_state(SMALL, COSMO, t, seed=0)
    npart = SMALL.n_particles**3
    assert tuple(x.shape) == (npart, 3) and tuple(p.shape) == (npart, 3)
    assert x.dtype == mx.float32 and p.dtype == mx.float32
    assert float(mx.min(x)) >= 0.0 and float(mx.max(x)) < SMALL.box_size

    # Positions are exactly the z_init Zel'dovich layout.
    x_lpt = L.lpt_positions(SMALL, COSMO, seed=0, z=9.0)
    assert float(mx.max(mx.abs(x - x_lpt))) == 0.0

    # Velocity is the growing mode p = a^2 E f * (displacement), with the
    # displacement = x - q (minimum image). This is the IC that grows as D(a).
    a_i = 0.1
    q = np.asarray(L.lagrangian_grid(SMALL), np.float64)
    xn = np.asarray(x, np.float64)
    Lh = SMALL.box_size
    disp = (xn - q + Lh / 2) % Lh - Lh / 2
    coef = a_i**2 * float(C.E(9.0, COSMO)) * float(C.growth_rate(9.0, COSMO))
    p_expected = coef * disp
    rel = np.max(np.abs(np.asarray(p, np.float64) - p_expected)) / np.max(
        np.abs(p_expected)
    )
    assert rel < 1e-5


def _painted_skewness(x, box):
    f = np.asarray(PA.density_contrast(x, box), np.float64).ravel()
    f = f - f.mean()
    return np.mean(f**3) / np.mean(f**2) ** 1.5


def test_initial_state_2lpt_valid_and_differs():
    box = BoxConfig(box_size=200.0, n_mesh=32, n_particles=32)
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10)
    x1, _ = IG.initial_state(box, COSMO, t, seed=1, backend="eh98", lpt_order=1)
    x2, p2 = IG.initial_state(box, COSMO, t, seed=1, backend="eh98", lpt_order=2)
    npart = box.n_particles**3
    assert tuple(x2.shape) == (npart, 3) and tuple(p2.shape) == (npart, 3)
    assert x2.dtype == mx.float32 and p2.dtype == mx.float32
    assert float(mx.min(x2)) >= 0.0 and float(mx.max(x2)) < box.box_size
    assert bool(mx.all(mx.isfinite(x2))) and bool(mx.all(mx.isfinite(p2)))
    # 2LPT adds a real second-order term, so it must differ from Zel'dovich.
    assert float(mx.max(mx.abs(x2 - x1))) > 0.0


def test_initial_state_2lpt_increases_skewness():
    # The 2LPT correction sources gravitational collapse, so its IC density is
    # MORE positively skewed than the Zel'dovich IC. This pins the sign of the
    # second-order term: a flipped sign would push the skewness below ZA.
    box = BoxConfig(box_size=200.0, n_mesh=32, n_particles=32)
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10)
    x1, _ = IG.initial_state(box, COSMO, t, seed=1, backend="eh98", lpt_order=1)
    x2, _ = IG.initial_state(box, COSMO, t, seed=1, backend="eh98", lpt_order=2)
    s1 = _painted_skewness(x1, box)
    s2 = _painted_skewness(x2, box)
    assert s1 > 0.0
    assert s2 > s1


def test_initial_state_invalid_lpt_order():
    import pytest

    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=4)
    with pytest.raises(ValueError):
        IG.initial_state(SMALL, COSMO, t, seed=0, lpt_order=3)


def test_determinism():
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
    xa, _ = IG.leapfrog(SMALL, COSMO, t, seed=4)
    xb, _ = IG.leapfrog(SMALL, COSMO, t, seed=4)
    xc, _ = IG.leapfrog(SMALL, COSMO, t, seed=5)
    mx.eval(xa, xb, xc)
    # Same seed agrees up to GPU scatter-add round-off, not bit-exactly: CIC's
    # atomic accumulation order is not reproducible on Metal (~1e-6 per paint,
    # growing mildly over steps), so this is a tiny tolerance, not == 0.
    assert float(mx.max(mx.abs(xa - xb))) < 1e-3
    assert float(mx.max(mx.abs(xa - xc))) > 1.0


def test_snapshot_callback():
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=4)
    calls = []

    def snap(step, a, x, p):
        calls.append((step, a, tuple(x.shape), tuple(p.shape)))

    IG.leapfrog(SMALL, COSMO, t, seed=0, snapshot=snap)
    assert [c[0] for c in calls] == [0, 1, 2, 3, 4]  # step 0 (IC) + n_steps
    a_vals = [c[1] for c in calls]
    assert abs(a_vals[0] - 0.1) < 1e-12 and abs(a_vals[-1] - 1.0) < 1e-12
    assert all(np.diff(a_vals) > 0)
    npart = SMALL.n_particles**3
    assert all(c[2] == (npart, 3) and c[3] == (npart, 3) for c in calls)


def test_linear_growth_converges_to_D():
    # Largest-scale modes must grow as the linear growth factor D, and the plain
    # leapfrog must converge toward it as the step count rises. The cross-
    # correlation estimator of the per-mode growth is measured on k < 0.04 h/Mpc
    # (deeply linear) of a large box; the residual at high step count is the
    # expected leapfrog discretization (plus a little CIC-resolution) error.
    box = BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
    ratio = C.growth_factor(0.0, COSMO) / C.growth_factor(9.0, COSMO)
    _, _, kmag = F.k_grid(box)
    mask = (kmag > 0) & (kmag < 0.04)

    def growth_estimate(n_steps):
        # Explicitly the exact-background integrator: this test is about ITS
        # convergence to D as steps rise (the deficit FastPM removes). FastPM is
        # checked to be exact at any step count in test_fastpm_linear_mode_*.
        t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=n_steps, integrator="exact")
        x0, _ = IG.initial_state(box, COSMO, t, seed=0)
        d_i = np.asarray(mx.fft.rfftn(PA.density_contrast(x0, box)))
        xf, _ = IG.leapfrog(box, COSMO, t, seed=0)
        d_f = np.asarray(mx.fft.rfftn(PA.density_contrast(xf, box)))
        return np.real(
            np.sum(d_f[mask] * np.conj(d_i[mask])) / np.sum(np.abs(d_i[mask]) ** 2)
        )

    err_coarse = abs(growth_estimate(6) / ratio - 1.0)
    err_fine = abs(growth_estimate(24) / ratio - 1.0)
    assert err_fine < err_coarse  # converging toward linear growth
    assert err_fine < 0.03  # within 3% of D on the largest scales


def test_differentiable_through_leapfrog():
    # The architecture headline: mx.grad flows through the full unrolled
    # trajectory (every kick, drift, force solve and CIC op over all steps).
    # Checked against a random directional finite difference; the ~few-percent
    # floor is the FD reference straddling CIC kinks across steps (the AD itself
    # is exact), and (as for the force) smaller h is the accurate end here.
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
    x0, p0 = IG.initial_state(SMALL, COSMO, t, seed=0)
    ag = IG.a_grid(t)

    def loss(x):
        xf, _ = IG.evolve_state(x, p0, SMALL, COSMO, ag)
        return mx.sum(PA.density_contrast(xf, SMALL) ** 2)

    g = mx.grad(loss)(x0)
    mx.eval(g)
    assert bool(mx.all(mx.isfinite(g)))
    assert float(mx.max(mx.abs(g))) > 0.0

    def loss64(x):
        xf, _ = IG.evolve_state(x, p0, SMALL, COSMO, ag)
        return float(
            np.sum(np.asarray(PA.density_contrast(xf, SMALL), np.float64) ** 2)
        )

    rng = np.random.default_rng(0)
    v = rng.standard_normal(tuple(x0.shape)).astype(np.float32)
    v /= np.linalg.norm(v)
    vmx = mx.array(v)
    h = 0.02 * SMALL.cell_size
    fd = (loss64(x0 + h * vmx) - loss64(x0 - h * vmx)) / (2.0 * h)
    ad = float(mx.sum(g * vmx))
    assert abs(fd - ad) / abs(ad) < 0.10


# --- Performance wiring: compiled force solve + gradient checkpointing ---
# These must change cost (wall time / AD memory) WITHOUT changing the numbers:
# the compiled solve and the checkpointed solve are the same computation, so the
# forward field and its gradient must agree with the eager/replay path to the
# float32 + CIC-scatter round-off floor (~1e-6), not to some looser tolerance.


def test_compiled_force_matches_eager():
    # mx.compile fuses the paint -> Poisson solve -> read chain; the result must
    # equal the eager force up to float32 reassociation (no semantic change).
    x0, _ = IG.initial_state(SMALL, COSMO, TimeStepping(n_steps=4), seed=0)
    g_eager = FO.make_force_fn(SMALL)(x0)
    g_comp = FO.make_force_fn(SMALL, compiled=True)(x0)
    mx.eval(g_eager, g_comp)
    rel = float(mx.max(mx.abs(g_eager - g_comp))) / float(mx.max(mx.abs(g_eager)))
    assert rel < 1e-5


def _traj_grad(force_fn, x0, p0, ag):
    def loss(x):
        xf, _ = IG.evolve_state(x, p0, SMALL, COSMO, ag, force_fn=force_fn)
        return mx.sum(PA.density_contrast(xf, SMALL) ** 2)

    return mx.grad(loss)(x0)


def test_checkpoint_grad_matches_replay():
    # Gradient checkpointing recomputes each step's force in the backward pass
    # to save memory; the gradient must be identical to the full-replay graph
    # (and to the compiled+checkpointed path) up to the ~1e-6 scatter-add floor.
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=4)
    x0, p0 = IG.initial_state(SMALL, COSMO, t, seed=0)
    ag = IG.a_grid(t)
    g_replay = _traj_grad(FO.make_force_fn(SMALL), x0, p0, ag)
    g_ckpt = _traj_grad(FO.make_force_fn(SMALL, checkpoint=True), x0, p0, ag)
    g_cc = _traj_grad(
        FO.make_force_fn(SMALL, compiled=True, checkpoint=True), x0, p0, ag
    )
    mx.eval(g_replay, g_ckpt, g_cc)
    den = float(mx.max(mx.abs(g_replay)))
    assert float(mx.max(mx.abs(g_ckpt - g_replay))) / den < 1e-4
    assert float(mx.max(mx.abs(g_cc - g_replay))) / den < 1e-4


def test_leapfrog_memory_mode_and_compiled_paths():
    # The leapfrog knobs run end-to-end and agree with the default path; the
    # 'adjoint' mode is a gradient strategy, not a forward mode, so leapfrog
    # rejects it (pointing at adjoint_grad_fnl) rather than silently mis-stepping.
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
    x_def, _ = IG.leapfrog(SMALL, COSMO, t, seed=1)
    x_ck, _ = IG.leapfrog(SMALL, COSMO, t, seed=1, memory_mode="checkpoint")
    x_cc, _ = IG.leapfrog(
        SMALL, COSMO, t, seed=1, compiled=True, memory_mode="checkpoint"
    )
    mx.eval(x_def, x_ck, x_cc)
    assert float(mx.max(mx.abs(x_ck - x_def))) < 1e-3  # scatter-add floor
    assert float(mx.max(mx.abs(x_cc - x_def))) < 1e-3
    try:
        IG.leapfrog(SMALL, COSMO, t, seed=1, memory_mode="adjoint")
        raise AssertionError("adjoint must not run as a forward memory_mode")
    except ValueError:
        pass


# --- Reversible adjoint integrator: O(1)-in-steps gradient of f_NL ---
# The adjoint reconstructs each state by reverse-stepping the (time-reversible)
# leapfrog instead of storing the trajectory, so its gradient must match the
# replay mx.grad to the float32 reversibility floor while its memory is flat in
# the step count. These check correctness; the memory win is measured in
# scripts/bench_pm.py.


def _adjoint_setup():
    box = BoxConfig(box_size=256.0, n_mesh=16, n_particles=16)
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=6)
    kb = np.arange(1, 4) * box.k_fundamental

    def loss_field(xf):  # one scalar summary of the final positions
        d = PA.density_contrast(xf, box)
        return mx.sum(F.band_power(d, box, kb))

    return box, t, kb, loss_field


def test_evolve_eager_matches_replay():
    # The eager (O(1)-memory) forward must reach the same final field as the
    # replay stepper, up to the scatter-add floor. (It recomputes the force each
    # step instead of reusing it, so this also checks the two KDK forms agree.)
    box, t, _, _ = _adjoint_setup()
    x0, p0 = IG.initial_state(box, COSMO, t, seed=0)
    ag = IG.a_grid(t)
    xr, _ = IG.evolve_state(x0, p0, box, COSMO, ag)
    xe, _ = IG.evolve_eager(x0, p0, box, COSMO, ag)
    mx.eval(xr, xe)
    cell = box.cell_size
    assert float(mx.max(mx.abs(xr - xe))) < 1e-3 * cell


def test_adjoint_grad_matches_replay():
    # The reversible-adjoint d loss/d f_NL must match the replay mx.grad to the
    # measured ~1e-6 reversibility floor (far below any dP/df_NL signal).
    box, t, _, loss_field = _adjoint_setup()

    def replay_loss(f):
        x, _ = IG.leapfrog(box, COSMO, t, seed=0, f_NL=f, backend="eh98")
        return loss_field(x)

    g_replay = float(mx.grad(replay_loss)(mx.array(50.0)))
    g_adjoint = float(
        IG.adjoint_grad_fnl(
            loss_field, box, COSMO, t, seed=0, f_NL=50.0, backend="eh98"
        )
    )
    assert abs(g_replay) > 0.0
    assert abs(g_adjoint - g_replay) / abs(g_replay) < 1e-3


# --- FastPM growth-corrected integrator ---------------------------------------


def test_fastpm_reduces_to_exact_small_step():
    # In the continuum (tiny-step) limit the FastPM kernels must reproduce the
    # literal background integrals -- a sanity check on the kernels and the
    # numerical g_f derivative.
    a0, a1 = 0.30, 0.30 + 1e-5
    a_c = 0.5 * (a0 + a1)
    assert (
        abs(
            IG.fastpm_kick_factor(a0, a1, a_c, COSMO) / IG.kick_factor(a0, a1, COSMO)
            - 1.0
        )
        < 1e-6
    )
    assert (
        abs(
            IG.fastpm_drift_factor(a0, a1, a_c, COSMO) / IG.drift_factor(a0, a1, COSMO)
            - 1.0
        )
        < 1e-6
    )


def _grow_linear_mode(integrator, n_steps, z_init=9.0):
    # Toy single-mode integrator: a linear mode has geometric acceleration equal
    # to its displacement amplitude (both track D(a)), so the KDK coefficients
    # alone determine the growth. Returns the final displacement amplitude.
    a = IG.a_grid(TimeStepping(z_init=z_init, z_final=0.0, n_steps=n_steps))
    ai = float(a[0])
    Di = C.growth_factor(z_init, COSMO)
    x = Di
    p = ai**2 * float(IG._E_of_a(ai, COSMO)) * Di * C.growth_rate(z_init, COSMO)
    for k1, dr, k2 in IG._step_coeffs(a, COSMO, integrator):
        p = p + k1 * x  # g = x (force at the start position)
        x = x + dr * p
        p = p + k2 * x  # g = x (force at the end position)
    return x


def test_fastpm_linear_mode_exact_growth():
    # The defining FastPM property: a single linear mode grows as D(a) EXACTLY at
    # any step count, while the exact-background leapfrog has a step-count deficit.
    Di = C.growth_factor(9.0, COSMO)
    target = C.growth_factor(0.0, COSMO) / Di  # D(z=0) / D(z_init)
    for n in (2, 4, 8):
        assert abs(_grow_linear_mode("fastpm", n) / Di / target - 1.0) < 1e-4
    # the exact integrator is off by > 2% at low step count
    assert abs(_grow_linear_mode("exact", 2) / Di / target - 1.0) > 0.02


def test_exact_path_unchanged_by_refactor():
    # The "exact" integrator must still produce the literal kick_factor /
    # drift_factor KDK, unchanged by the integrator refactor (to the scatter-add
    # floor, since the two paths re-evaluate the force independently).
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=4)
    x0, p0 = IG.initial_state(SMALL, COSMO, t, seed=0)
    ag = IG.a_grid(t)
    xr, _ = IG.evolve_state(x0, p0, SMALL, COSMO, ag, integrator="exact")

    ff = FO.make_force_fn(SMALL)
    x, p = x0, p0
    g = ff(x)
    for i in range(len(ag) - 1):
        a0, a1 = float(ag[i]), float(ag[i + 1])
        a_c = 0.5 * (a0 + a1)
        p = p + IG.kick_factor(a0, a_c, COSMO) * g
        x = IG._wrap(x + IG.drift_factor(a0, a1, COSMO) * p, SMALL.box_size)
        g = ff(x)
        p = p + IG.kick_factor(a_c, a1, COSMO) * g
    mx.eval(xr, x)
    assert float(mx.max(mx.abs(xr - x))) < 1e-3 * SMALL.cell_size


def test_fastpm_adjoint_matches_replay():
    # FastPM coefficients are constants of the step, so mx.grad (replay) and the
    # reversible adjoint must agree under integrator="fastpm" too.
    box, t, _, loss_field = _adjoint_setup()

    def replay_loss(f):
        x, _ = IG.leapfrog(
            box, COSMO, t, seed=0, f_NL=f, backend="eh98", integrator="fastpm"
        )
        return loss_field(x)

    g_replay = float(mx.grad(replay_loss)(mx.array(50.0)))
    g_adjoint = float(
        IG.adjoint_grad_fnl(
            loss_field,
            box,
            COSMO,
            t,
            seed=0,
            f_NL=50.0,
            backend="eh98",
            integrator="fastpm",
        )
    )
    assert abs(g_replay) > 0.0
    assert abs(g_adjoint - g_replay) / abs(g_replay) < 1e-3
