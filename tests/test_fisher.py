"""Tests for the autodiff Fisher matrix (mbody.fisher), milestone 1: the
linear-field forecast over theta = {f_NL, b1, b2, A}.

What is checked, and how the tolerances were set (measured with
scripts/probe_fisher.py, not guessed):

  * each Jacobian column d ln P_b / d theta from reverse-mode mx.grad matches a
    matched-phase finite difference (worst column, b2, measured ~1e-3);
  * the Gaussian band-power variance band_power_log_variance matches the
    empirical white-noise ensemble variance -- the check that pins the half-grid
    plane double-counting (the naive 1/count_b is ~1.3-1.5x low at low k);
  * the FisherForecast linear algebra (compute / inverse / sigma / priors /
    marginalized_2d / fixed) matches a hand-built analytic case, with no FFT;
  * sigma(f_NL) scales as 1/sqrt(volume) -- i.e. as 1/sqrt(info) when the
    per-bin variance is halved;
  * the assembled Fisher is symmetric and positive-definite (with a weak prior
    on the near-degenerate amplitude direction).

Forward-mode mx.jvp is NOT used (wrong through the FFT); all gradients are
reverse-mode mx.grad. Tolerances are not relaxed without a measured basis.
"""

import numpy as np
import mlx.core as mx

from mbody.config import BoxConfig, Cosmology, TimeStepping, Tracer
from mbody import fields as F
from mbody import ic as IC
from mbody import bias as B
from mbody import integrate as IN
from mbody import fisher as FI

COSMO = Cosmology()
BOX = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
TIME_PM = TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
TRACER = Tracer()  # b1=2, b2=1, A=1
THETA_FID = {"f_NL": 0.0, "b1": TRACER.b1, "b2": TRACER.b2, "A": TRACER.A}
H = {"f_NL": 50.0, "b1": 1e-2, "b2": 1e-2, "A": 1e-2}


def _kb():
    return np.array([1, 2, 3, 4]) * BOX.k_fundamental


def _logP(theta, seed, kb):
    f_NL = mx.array(theta["f_NL"])
    delta = theta["A"] * IC.linear_density(BOX, COSMO, seed=seed, f_NL=f_NL)
    tracer = B.local_bias_tracer(delta, theta["b1"], theta["b2"])
    return np.asarray(mx.log(F.band_power(tracer, BOX, kb)), np.float64)


def _fd_column(name, seed, kb):
    hi = dict(THETA_FID, **{name: THETA_FID[name] + H[name]})
    lo = dict(THETA_FID, **{name: THETA_FID[name] - H[name]})
    return (_logP(hi, seed, kb) - _logP(lo, seed, kb)) / (2.0 * H[name])


def test_jacobian_columns_vs_matched_phase_fd():
    # Each autodiff column matches the matched-phase central FD. Measured worst
    # column (b2) ~1e-3; the loosest is the tolerance.
    kb = _kb()
    J, _ = FI.linear_logP_jacobian(BOX, COSMO, THETA_FID, kb, seed=0)
    for i, name in enumerate(FI.PARAM_NAMES):
        fd = _fd_column(name, 0, kb)
        rel = np.abs(J[:, i] / fd - 1.0)
        assert np.all(rel < 2e-3), f"{name}: max rel.err {rel.max():.2e}"


def test_band_power_variance_matches_gaussian():
    # The Gaussian formula (count_b + n_plane)/count_b^2 matches the empirical
    # white-noise band-power variance for well-sampled bins. A pure mode-count
    # 1/count_b would be ~1.3-1.5x low at low k (the plane double-counting), so
    # this test has teeth. nseed=300 -> sampling error ~ sqrt(2/300) ~ 0.082.
    N = BOX.n_mesh
    kb = np.array([3, 4, 5, 6, 8]) * BOX.k_fundamental
    pred = FI.band_power_log_variance(BOX, kb)
    _, counts = F._bin_masks(BOX, kb, BOX.k_fundamental)
    counts = np.array(counts)

    nseed = 300
    P = np.array(
        [
            np.asarray(
                F.band_power(
                    mx.random.normal((N, N, N), key=mx.random.key(7000 + s)), BOX, kb
                ),
                np.float64,
            )
            for s in range(nseed)
        ]
    )
    emp = np.log(P).var(axis=0, ddof=1)  # Var[ln P_b]
    well = counts >= 30
    ratio = emp[well] / pred[well]
    assert np.all(np.abs(ratio - 1.0) < 0.25), f"variance ratios {ratio}"


def test_gaussian_toy_analytic_assembly():
    # The FisherForecast linear algebra, with no FFT/cosmology: a hand-built
    # Jacobian and variances whose Fisher is known in closed form.
    J = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    var = np.array([1.0, 4.0, 0.5])  # weights 1/var = [1, 0.25, 2]
    names = ("p", "q")
    fid = {"p": 0.0, "q": 0.0}

    fc = FI.FisherForecast(J, var, fid, param_names=names)
    F_exp = np.array([[3.0, 2.0], [2.0, 3.0]])  # sum_b J_b outer J_b / var_b
    assert np.allclose(fc.compute(), F_exp)

    # inverse is a true inverse; sigma = sqrt(diag F^-1); cond = 1/sqrt(diag F).
    assert np.allclose(fc.inverse @ fc.fisher_matrix, np.eye(2), atol=1e-12)
    Finv = np.linalg.inv(F_exp)
    assert np.isclose(fc.sigma("p"), np.sqrt(Finv[0, 0]))
    assert np.isclose(fc.sigma_conditional("q"), 1.0 / np.sqrt(F_exp[1, 1]))

    # A Gaussian prior adds exactly 1/sigma^2 to the diagonal.
    fc_p = FI.FisherForecast(J, var, fid, param_names=names, priors={"p": 1.0})
    assert np.allclose(fc_p.compute(), F_exp + np.array([[1.0, 0.0], [0.0, 0.0]]))

    # marginalized_2d returns the sub-covariance, correlation, and ellipse angle.
    m = fc.marginalized_2d("p", "q")
    assert np.isclose(m["rho"], Finv[0, 1] / np.sqrt(Finv[0, 0] * Finv[1, 1]))
    ang = 0.5 * np.degrees(np.arctan2(2 * Finv[0, 1], Finv[0, 0] - Finv[1, 1]))
    assert np.isclose(m["angle_deg"], ang)

    # Fixing q drops its row/column: F reduces to the 1x1 [F_pp].
    fc_fix = FI.FisherForecast(J, var, fid, param_names=names, fixed_params=["q"])
    assert fc_fix.compute().shape == (1, 1)
    assert np.isclose(fc_fix.sigma("p"), 1.0 / np.sqrt(F_exp[0, 0]))


def test_sigma_fnl_scales_with_volume():
    # A Gaussian forecast's sigma scales as 1/sqrt(volume): halving the per-bin
    # variance (twice the survey volume / twice the modes) tightens sigma(f_NL)
    # by exactly sqrt(2). Hold the nuisances fixed so the 1x1 f_NL block isolates
    # the scaling cleanly.
    kb = _kb()
    J, _ = FI.linear_logP_jacobian(BOX, COSMO, THETA_FID, kb, seed=0)
    var = FI.band_power_log_variance(BOX, kb)
    fixed = ["b1", "b2", "A"]
    fc1 = FI.FisherForecast(J, var, THETA_FID, fixed_params=fixed)
    fc2 = FI.FisherForecast(J, var / 2.0, THETA_FID, fixed_params=fixed)
    assert np.isclose(fc2.sigma("f_NL"), fc1.sigma("f_NL") / np.sqrt(2.0))


# --- Milestone 2: the PM-evolved forecast ---


def _pm_logP(theta, seed, kb):
    x, _ = IN.leapfrog(
        BOX,
        COSMO,
        TIME_PM,
        seed=seed,
        f_NL=mx.array(theta["f_NL"]),
        amplitude=mx.array(theta["A"]),
        lpt_order=2,
    )
    delta = F.interlaced_density_contrast(x, BOX)
    tracer = B.local_bias_tracer(delta, theta["b1"], theta["b2"])
    return np.asarray(mx.log(F.band_power(tracer, BOX, kb)), np.float64)


def _pm_fd_column(name, seed, kb):
    hi = dict(THETA_FID, **{name: THETA_FID[name] + H[name]})
    lo = dict(THETA_FID, **{name: THETA_FID[name] - H[name]})
    return (_pm_logP(hi, seed, kb) - _pm_logP(lo, seed, kb)) / (2.0 * H[name])


def test_pm_jacobian_columns_vs_matched_phase_fd():
    # The PM Jacobian (f_NL, A via the reversible adjoint; b1, b2 via the cheap
    # fixed-field downstream grad) matches matched-phase FD. CIC scatter-add sets
    # the floor; measured worst column (f_NL) ~3e-3.
    kb = _kb()
    J, _ = FI.pm_logP_jacobian(BOX, COSMO, TIME_PM, THETA_FID, kb, seed=0)
    for i, name in enumerate(FI.PARAM_NAMES):
        fd = _pm_fd_column(name, 0, kb)
        rel = np.abs(J[:, i] / fd - 1.0)
        assert np.all(rel < 1e-2), f"{name}: max rel.err {rel.max():.2e}"


def test_adjoint_grad_ic_matches_replay():
    # The reversible adjoint (adjoint_grad_ic, used for the f_NL and A columns)
    # agrees with replay mx.grad through the leapfrog -- both autodiff, differing
    # only at the float32 CIC reversibility floor (measured ~1e-6, far below any
    # signal). This validates the adjoint independently of the FD ground truth.
    kb = _kb()

    def loss_at(x, b):
        tracer = B.local_bias_tracer(
            F.interlaced_density_contrast(x, BOX), THETA_FID["b1"], THETA_FID["b2"]
        )
        return mx.log(F.band_power(tracer, BOX, kb))[b]

    def replay_loss(tv, b):
        x, _ = IN.leapfrog(
            BOX, COSMO, TIME_PM, seed=0, f_NL=tv[0], amplitude=tv[1], lpt_order=2
        )
        return loss_at(x, b)

    tv0 = mx.array([THETA_FID["f_NL"], THETA_FID["A"]])
    for b in range(len(kb)):
        g_adj = np.asarray(
            IN.adjoint_grad_ic(
                lambda x, b=b: loss_at(x, b),
                BOX,
                COSMO,
                TIME_PM,
                seed=0,
                f_NL=THETA_FID["f_NL"],
                amplitude=THETA_FID["A"],
                lpt_order=2,
            )
        )
        g_rep = np.asarray(mx.grad(lambda tv, b=b: replay_loss(tv, b))(tv0))
        assert np.all(np.abs(g_adj / g_rep - 1.0) < 1e-4)


def test_fisher_symmetry_and_positive_definite():
    # With a weak prior on the near-degenerate amplitude A, the assembled Fisher
    # is symmetric and positive-definite (Cholesky succeeds).
    kb = _kb()
    J = np.zeros((len(kb), len(FI.PARAM_NAMES)))
    nseed = 4
    for s in range(nseed):
        Js, _ = FI.linear_logP_jacobian(BOX, COSMO, THETA_FID, kb, seed=s)
        J += Js
    J /= nseed
    var = FI.band_power_log_variance(BOX, kb)
    fc = FI.FisherForecast(J, var, THETA_FID, priors={"A": 0.1})
    Fm = fc.fisher_matrix
    assert np.allclose(Fm, Fm.T)
    np.linalg.cholesky(Fm)  # raises if not positive-definite
    assert np.all(np.linalg.eigvalsh(Fm) > 0)
