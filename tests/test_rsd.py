"""Redshift-space distortion tests: the velocity->shift conversion, the multipole
estimator vs linear Kaiser, interlacing, the differentiable Jacobian, the
momentum-seeded adjoint, and the multipole Fisher. Tolerances are those measured
in scripts/probe_rsd.py (measure first, then bake in -- never guessed).
"""

import numpy as np
import mlx.core as mx
import pytest

from mbody import bias as B
from mbody import config
from mbody import cosmology as C
from mbody import fields as F
from mbody import fisher as FI
from mbody import integrate as IN
from mbody import lpt as L
from mbody import painting as PA
from mbody import rsd

BOX = config.BoxConfig(box_size=512.0, n_mesh=48, n_particles=48)
COSMO = config.Cosmology()
LOS = 0


def _kbins(box, kmax_frac=0.35):
    kf = box.k_fundamental
    return np.arange(1.5 * kf, kmax_frac * box.k_nyquist, kf)


def test_conversion_reproduces_growth_rate():
    """Delta_s = p_los/(a^2 E) makes a Zeldovich field's RSD shift equal f*disp."""
    zt = 0.05
    time = config.TimeStepping(z_init=zt, z_final=0.0)
    a, E = 1.0 / (1.0 + zt), float(C.E(zt, COSMO))
    D, f0 = C.growth_factor(zt, COSMO), C.growth_rate(zt, COSMO)
    amp, seed = 0.03, 1
    x, p = IN.initial_state(
        BOX, COSMO, time, seed=seed, backend="eh98", lpt_order=1, amplitude=amp
    )
    psi_los = np.asarray(
        L.zeldovich_displacement(BOX, COSMO, seed=seed, backend="eh98", amplitude=amp)[
            LOS
        ]
    ).reshape(-1)
    real_disp = D * psi_los
    extra = (1.0 / (a**2 * E)) * np.asarray(p[:, LOS])
    m = np.abs(real_disp) > 1e-4
    ratio = float(np.mean(extra[m] / real_disp[m]))
    assert abs(ratio / f0 - 1.0) < 1e-4  # measured ~2e-7


def test_redshift_space_positions_periodic_and_transverse():
    """The map shifts only the LOS axis and wraps into [0, L)."""
    time = config.TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
    x, p = IN.initial_state(BOX, COSMO, time, seed=0, backend="eh98", lpt_order=1)
    s = rsd.redshift_space_positions(x, p, BOX, COSMO, z=0.0, los_axis=LOS)
    s_np = np.asarray(s)
    assert s_np[:, LOS].min() >= 0.0 and s_np[:, LOS].max() < BOX.box_size
    for ax in (1, 2):  # transverse axes unchanged
        assert np.allclose(np.asarray(x[:, ax]), s_np[:, ax], atol=1e-4)


def test_estimator_recovers_kaiser():
    """Imposed-Kaiser field recovers the continuum P2/P0 ratio (decoupled)."""
    beta = 0.5
    n_seed = 12
    acc = None
    for seed in range(n_seed):
        delta = F.gaussian_random_field(BOX, COSMO, seed=seed, z=0.0, backend="eh98")
        ds = rsd.apply_linear_kaiser(delta, BOX, 1.0, beta, los_axis=LOS)
        _, P, _ = F.power_multipoles(
            ds, BOX, ells=(0, 2), los_axis=LOS, kmax=0.45 * BOX.k_nyquist
        )
        acc = {el: P[el] if acc is None else acc[el] + P[el] for el in P}
    P = {el: acc[el] / n_seed for el in acc}
    lo = slice(0, len(P[0]) // 2)
    r2 = float(P[2][lo].sum() / P[0][lo].sum())
    c0 = 1 + 2 / 3 * beta + 1 / 5 * beta**2
    c2 = 4 / 3 * beta + 4 / 7 * beta**2
    assert abs(r2 - c2 / c0) < 0.08  # measured ~0.02


def test_estimator_isotropic_quadrupole_vanishes():
    """An isotropic (beta=0) field has ~zero quadrupole after decoupling."""
    n_seed = 12
    acc = None
    for seed in range(n_seed):
        delta = F.gaussian_random_field(BOX, COSMO, seed=seed, z=0.0, backend="eh98")
        _, P, _ = F.power_multipoles(
            delta, BOX, ells=(0, 2), los_axis=LOS, kmax=0.45 * BOX.k_nyquist
        )
        acc = {el: P[el] if acc is None else acc[el] + P[el] for el in P}
    P = {el: acc[el] / n_seed for el in acc}
    lo = slice(0, len(P[0]) // 2)
    assert abs(float(P[2][lo].sum() / P[0][lo].sum())) < 0.06  # measured ~0.01


def test_cross_power_multipole_reductions():
    """cross_power_multipole(a, b, ell=0) == cross_power(a, b), and the auto case
    cross_power_multipole(a, a, ell) == band_power_multipole(a, ell). Geometric, so
    measured ~1e-7 (the ell=0 cross is exact; the auto match is the float32
    abs**2-vs-Re(z conj z) floor)."""
    from mbody import ic as IC

    kbins = _kbins(BOX)
    a = IC.linear_density(BOX, COSMO, seed=0, f_NL=0.0, backend="eh98")
    b = IC.linear_density(BOX, COSMO, seed=1, f_NL=0.0, backend="eh98")
    c0 = np.asarray(F.cross_power_multipole(a, b, BOX, kbins, 0, los_axis=LOS))
    xp = np.asarray(F.cross_power(a, b, BOX, kbins))
    assert np.allclose(c0, xp, rtol=1e-6, atol=0.0)
    for el in (0, 2, 4):
        auto = np.asarray(F.cross_power_multipole(a, a, BOX, kbins, el, los_axis=LOS))
        bpm = np.asarray(F.band_power_multipole(a, BOX, kbins, el, los_axis=LOS))
        rel = np.abs(auto / bpm - 1.0).max()
        assert rel < 1e-5, f"ell={el} reduction rel {rel:.2e}"


def test_interlacing_reduces_aliasing():
    """Interlacing removes the CIC power upturn near Nyquist."""
    box = config.BoxConfig(box_size=256.0, n_mesh=48, n_particles=48)
    time = config.TimeStepping(z_init=0.05, z_final=0.0)
    x, _ = IN.initial_state(box, COSMO, time, seed=3, backend="eh98", lpt_order=2)
    k, Pp, _ = F.power_spectrum(PA.density_contrast(x, box), box, deconvolve_cic=True)
    _, Pi, _ = F.power_spectrum(
        F.interlaced_density_contrast(x, box), box, deconvolve_cic=True
    )
    Plin = C.linear_power(k, COSMO, z=0.05, backend="eh98")
    hi = k > 0.85 * box.k_nyquist
    plain_up = float(np.max(Pp[hi] / Plin[hi]))
    inter_up = float(np.max(Pi[hi] / Plin[hi]))
    assert plain_up > 1.05  # plain aliasing upturn present
    assert inter_up < plain_up - 0.15  # interlacing suppresses it


def test_interlacing_matches_plain_cic_at_low_k():
    """At low k interlacing is identity to plain CIC (measured ~6e-5). This is why
    making interlacing the standard measurement painter leaves the f_NL signal and
    the autodiff d ln P/d theta unchanged -- both live at low k, and the CIC window
    cancels in the log-derivative regardless."""
    box = config.BoxConfig(box_size=256.0, n_mesh=48, n_particles=48)
    time = config.TimeStepping(z_init=0.05, z_final=0.0)
    x, _ = IN.initial_state(box, COSMO, time, seed=3, backend="eh98", lpt_order=2)
    k, Pp, _ = F.power_spectrum(PA.density_contrast(x, box), box, deconvolve_cic=True)
    _, Pi, _ = F.power_spectrum(
        F.interlaced_density_contrast(x, box), box, deconvolve_cic=True
    )
    lo = k < 0.1 * box.k_nyquist
    assert np.all(np.abs(Pi[lo] / Pp[lo] - 1.0) < 1e-3)  # measured ~6e-5


def test_linear_multipole_jacobian_matches_fd():
    """dP_ell/dtheta (autodiff) matches a matched-phase finite difference."""
    kbins = _kbins(BOX)
    theta = {"f_NL": 0.0, "b1": 2.0, "b2": 1.0, "A": 1.0, "f_growth": 1.0}
    ells = (0, 2)
    seed = 7
    J, _ = FI.linear_multipole_jacobian(
        BOX, COSMO, theta, kbins, ells=ells, los_axis=LOS, seed=seed, backend="eh98"
    )
    f_lin = C.growth_rate(0.0, COSMO)

    def model(thd):
        from mbody import ic as IC

        d = thd["A"] * IC.linear_density(
            BOX, COSMO, seed=seed, f_NL=thd["f_NL"], backend="eh98"
        )
        tr = FI._redshift_tracer_linear(
            d, BOX, thd["b1"], thd["b2"], thd["f_growth"] * f_lin, LOS
        )
        return np.concatenate(
            [
                np.asarray(F.band_power_multipole(tr, BOX, kbins, el, los_axis=LOS))
                for el in ells
            ]
        )

    for name, h in (("f_NL", 1.0), ("b1", 0.02), ("f_growth", 0.02)):
        tp, tm = dict(theta), dict(theta)
        tp[name] += h
        tm[name] -= h
        fd = (model(tp) - model(tm)) / (2 * h)
        ci = FI.PARAM_NAMES_RSD.index(name)
        rel = np.abs(J[:, ci] - fd) / (np.abs(fd).max() + 1e-30)
        assert rel.max() < 2e-3  # measured ~1e-4


def test_redshift_adjoint_matches_replay():
    """Momentum-seeded adjoint (f_NL, A) grad of a redshift-space band power
    matches replay mx.grad to the reversibility floor."""
    box = config.BoxConfig(box_size=256.0, n_mesh=24, n_particles=24)
    time = config.TimeStepping(z_init=9.0, z_final=0.0, n_steps=4)
    a_steps = IN.a_grid(time, "linear")
    kf = box.k_fundamental
    kbins = np.arange(1.5 * kf, 6.5 * kf, kf)
    b1, b2, ell, bin_i, seed = 2.0, 1.0, 2, 2, 2

    def redshift_bp(x, p):
        s = rsd.redshift_space_positions(x, p, box, COSMO, z=time.z_final, los_axis=LOS)
        tr = B.local_bias_tracer(F.interlaced_density_contrast(s, box), b1, b2)
        return F.band_power_multipole(tr, box, kbins, ell, los_axis=LOS)[bin_i]

    def loss_theta(th):
        x0, p0 = IN.initial_state(
            box,
            COSMO,
            time,
            seed=seed,
            f_NL=th[0],
            backend="eh98",
            lpt_order=2,
            amplitude=th[1],
        )
        x, p = IN.evolve_state(x0, p0, box, COSMO, a_steps, integrator="fastpm")
        return redshift_bp(x, p)

    g_replay = np.asarray(mx.grad(loss_theta)(mx.array([5.0, 1.0])))
    g_adj = np.asarray(
        IN.adjoint_grad_ic(
            redshift_bp,
            box,
            COSMO,
            time,
            seed=seed,
            f_NL=5.0,
            amplitude=1.0,
            backend="eh98",
            integrator="fastpm",
            lpt_order=2,
            loss_uses_momentum=True,
        )
    )
    rel = np.abs(g_adj - g_replay) / (np.abs(g_replay) + 1e-30)
    assert rel.max() < 1e-4  # measured ~2e-6


def test_fisher_quadrupole_tightens_constraints():
    """Adding the quadrupole tightens sigma(f_NL) and sharply sigma(f_growth)."""
    kbins = _kbins(BOX)
    theta = {"f_NL": 0.0, "b1": 2.0, "b2": 1.0, "A": 1.0, "f_growth": 1.0}
    priors = {"A": 0.1}

    def forecast(ells):
        J = np.zeros((len(kbins) * len(ells), len(FI.PARAM_NAMES_RSD)))
        for s in range(4):
            J += FI.linear_multipole_jacobian(
                BOX,
                COSMO,
                theta,
                kbins,
                ells=ells,
                los_axis=LOS,
                seed=s,
                backend="eh98",
            )[0]
        J /= 4
        cov = FI.multipole_gaussian_covariance(
            BOX,
            COSMO,
            theta,
            kbins,
            ells=ells,
            los_axis=LOS,
            n_mock=200,
            backend="eh98",
        )
        return FI.FisherForecast(
            J,
            covariance=cov,
            fiducial_params=theta,
            param_names=FI.PARAM_NAMES_RSD,
            priors=priors,
        )

    mono = forecast((0,))
    quad = forecast((0, 2))
    assert quad.sigma("f_NL") < mono.sigma("f_NL")
    assert quad.sigma("f_growth") < 0.3 * mono.sigma("f_growth")


def test_fisher_forecast_covariance_validation():
    """FisherForecast requires exactly one of variances / covariance, and a
    diagonal covariance reproduces the variances path."""
    J = np.array([[1.0, 0.0], [0.5, 2.0], [0.0, 1.0]])
    var = np.array([1.0, 4.0, 0.25])
    fid = {"a": 0.0, "b": 0.0}
    names = ("a", "b")
    with pytest.raises(ValueError):
        FI.FisherForecast(
            J,
            variances=var,
            covariance=np.diag(var),
            fiducial_params=fid,
            param_names=names,
        )
    with pytest.raises(ValueError):
        FI.FisherForecast(J, fiducial_params=fid, param_names=names)
    f_diag = FI.FisherForecast(J, variances=var, fiducial_params=fid, param_names=names)
    f_cov = FI.FisherForecast(
        J, covariance=np.diag(var), fiducial_params=fid, param_names=names
    )
    assert np.allclose(f_diag.compute(), f_cov.compute())


def test_config_redshift_space_defaults_and_validation():
    rs = config.RedshiftSpace()
    assert rs.enabled is False and rs.los_axis == 0 and rs.f_growth == 1.0
    with pytest.raises(ValueError):
        config.RedshiftSpace(los_axis=3)
    with pytest.raises(ValueError):
        config.RedshiftSpace(f_growth=-1.0)


def test_driver_power_multipoles_requires_rsd():
    """run() exposes power_multipoles only when rsd is enabled."""
    box = config.BoxConfig(box_size=400.0, n_mesh=32, n_particles=32)
    time = config.TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
    base = dict(cosmology=COSMO, box=box, time=time, tracer=config.Tracer())
    from mbody import run

    res_off = run(config.SimConfig(**base), backend="eh98")
    with pytest.raises(ValueError):
        res_off.power_multipoles()

    res_on = run(
        config.SimConfig(rsd=config.RedshiftSpace(enabled=True, los_axis=0), **base),
        backend="eh98",
    )
    k, P, n = res_on.power_multipoles(ells=(0, 2))
    assert 0 in P and 2 in P and len(k) == len(P[0])
