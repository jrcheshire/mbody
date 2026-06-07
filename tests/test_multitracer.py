"""Tests for the multi-tracer Fisher (mbody.fisher), the b_phi-f_NL capstone.

Two tracers painted from the same field break part of the b_phi-f_NL degeneracy
via sample-variance cancellation (Seljak 2009; Barreira & Krause 2023). Two
tracer models are tested:

  NATIVE (b_phi emergent from b2), DETECTION regime (fiducial f_NL=0):
    * the linear {P_AA,P_AB,P_BB} Jacobian matches matched-phase FD (worst ~2e-4);
    * the PM Jacobian matches FD (worst ~6e-4), and adjoint_grad_ic for the cross
      spectrum matches replay mx.grad (~1e-7);
    * sample-variance cancellation: 2 tracers tighten sigma(f_NL) vs 1.

  EXPLICIT-b_phi (b_phi a free k^-2 parameter), DEGENERACY regime (f_NL != 0):
    * the Jacobian matches FD (worst ~1e-4);
    * FREE -> sigma(f_NL) is degenerate (the Fisher is singular in (f_NL,b_phi));
      TIED (universality) -> f_NL recovered and 2 tracers are much tighter than 1.

  Covariance:
    * the analytic Gaussian block matches the hand-written formula (no FFT) and a
      white-noise mock ensemble (where the Gaussian assumption is exact);
    * per-tracer shot noise has band power 1/n (the unit anchor).

All tolerances were measured with scripts/probe_multitracer.py first, not guessed,
and are not relaxed without a measured basis. Gradients are reverse-mode mx.grad.
"""

import numpy as np
import mlx.core as mx

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import fields as F
from mbody import ic as IC
from mbody import bias as B
from mbody import integrate as IN
from mbody import fisher as FI

COSMO = Cosmology()
BACKEND = "eh98"
DELTA_C = 1.686
BOX = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
TIME_PM = TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
N_A = N_B = 1.0e-3

# Native fiducial at f_NL=0 (b2 at the universality value).
THETA = FI.universality_fiducial(0.0, 1.0, 1.5, 2.5, BOX, COSMO, backend=BACKEND)
H = {"f_NL": 50.0, "A": 1e-2, "b1_A": 1e-2, "b2_A": 1e-2, "b1_B": 1e-2, "b2_B": 1e-2}

# Explicit-b_phi fiducial at f_NL != 0 (the degeneracy regime), bigger box.
BOX_DEG = BoxConfig(box_size=1024.0, n_mesh=32, n_particles=32)
THETA_BPHI = FI.universality_bphi_fiducial(100.0, 1.0, 1.5, 2.5, delta_c=DELTA_C)
H_BPHI = {
    "f_NL": 5.0,
    "A": 1e-2,
    "b1_A": 1e-2,
    "bphi_A": 1e-2,
    "b1_B": 1e-2,
    "bphi_B": 1e-2,
}


def _kb(box, ns=(1, 2, 3, 4)):
    return np.array([n * box.k_fundamental for n in ns])


def _col_err(jcol, fd):
    denom = max(float(np.abs(fd).max()), 1e-300)
    return float(np.abs(np.asarray(jcol) - fd).max() / denom)


# --- native tracer: Jacobians ---------------------------------------------------


def _mt_vec(theta, seed, kb):
    f_NL = mx.array(theta["f_NL"])
    delta = theta["A"] * IC.linear_density(
        BOX, COSMO, seed=seed, f_NL=f_NL, backend=BACKEND
    )
    hA = B.local_bias_tracer(delta, theta["b1_A"], theta["b2_A"])
    hB = B.local_bias_tracer(delta, theta["b1_B"], theta["b2_B"])
    return np.asarray(FI._multitracer_vector(hA, hB, BOX, kb, None), np.float64)


def test_linear_multitracer_jacobian_vs_fd():
    # Each native linear column matches matched-phase FD; measured worst ~2e-4.
    kb = _kb(BOX)
    J, _ = FI.linear_multitracer_jacobian(
        BOX, COSMO, THETA, kb, seed=0, backend=BACKEND
    )
    for i, p in enumerate(FI.PARAM_NAMES_MT):
        hi = dict(THETA, **{p: THETA[p] + H[p]})
        lo = dict(THETA, **{p: THETA[p] - H[p]})
        fd = (_mt_vec(hi, 0, kb) - _mt_vec(lo, 0, kb)) / (2 * H[p])
        assert _col_err(J[:, i], fd) < 1e-3, f"{p}"


def test_pm_multitracer_downstream_jacobian_vs_fd():
    # The downstream bias columns (b1, b2 per tracer) are an mx.grad at the FIXED
    # final field. Validate them by FD on the SAME field (evolve once, so no
    # trajectory CIC-scatter nondeterminism between the +/- evaluations) -- tight.
    kb = _kb(BOX)
    x, _ = IN.leapfrog(
        BOX,
        COSMO,
        TIME_PM,
        seed=0,
        f_NL=mx.array(THETA["f_NL"]),
        amplitude=mx.array(THETA["A"]),
        lpt_order=2,
    )
    field = mx.stop_gradient(F.interlaced_density_contrast(x, BOX))

    def vec(b):
        hA = B.local_bias_tracer(field, b[0], b[1])
        hB = B.local_bias_tracer(field, b[2], b[3])
        return FI._multitracer_vector(hA, hB, BOX, kb, None)

    b0 = mx.array([THETA["b1_A"], THETA["b2_A"], THETA["b1_B"], THETA["b2_B"]])
    n_data = 3 * len(kb)
    Jd = np.array(
        [
            np.asarray(mx.grad(lambda b, d=d: vec(b)[d])(b0), np.float64)
            for d in range(n_data)
        ]
    )
    eye = np.eye(4)
    for j in range(4):
        hi = mx.array(np.asarray(b0) + 1e-2 * eye[j])
        lo = mx.array(np.asarray(b0) - 1e-2 * eye[j])
        fd = (np.asarray(vec(hi)) - np.asarray(vec(lo))) / 2e-2
        assert _col_err(Jd[:, j], fd) < 1e-3, f"downstream col {j}"


def test_pm_adjoint_ic_columns_match_replay():
    # The IC columns (f_NL, A) come from adjoint_grad_ic (one shared sweep). Validate
    # them against replay mx.grad through leapfrog for the AA, AB AND BB spectra (the
    # novel multi-tracer cross path included) -- both autodiff, agreeing to the
    # float32 CIC floor (measured ~1e-7).
    kb = _kb(BOX)
    nb = len(kb)

    def loss_at(x, d):
        field = F.interlaced_density_contrast(x, BOX)
        hA = B.local_bias_tracer(field, THETA["b1_A"], THETA["b2_A"])
        hB = B.local_bias_tracer(field, THETA["b1_B"], THETA["b2_B"])
        return FI._multitracer_vector(hA, hB, BOX, kb, None)[d]

    def replay(tv, d):
        x, _ = IN.leapfrog(
            BOX, COSMO, TIME_PM, seed=0, f_NL=tv[0], amplitude=tv[1], lpt_order=2
        )
        return loss_at(x, d)

    tv0 = mx.array([THETA["f_NL"], THETA["A"]])
    for d in (0, nb, 2 * nb):  # AA, AB, BB pivot bins
        g_adj = np.asarray(
            IN.adjoint_grad_ic(
                lambda x, d=d: loss_at(x, d),
                BOX,
                COSMO,
                TIME_PM,
                seed=0,
                f_NL=THETA["f_NL"],
                amplitude=THETA["A"],
            )
        )
        g_rep = np.asarray(mx.grad(lambda tv, d=d: replay(tv, d))(tv0))
        assert np.all(np.abs(g_adj / g_rep - 1.0) < 1e-4), f"d={d}"


# --- covariance: analytic block, white-noise mock, shot noise -------------------


def test_analytic_covariance_block_formula():
    # The 3x3 per-k block matches the hand-written Gaussian formula (no FFT), and
    # off-diagonal-in-k blocks are zero.
    box = BoxConfig(box_size=100.0, n_mesh=16, n_particles=16)
    kb = _kb(box, ns=(2, 3, 4))
    nb = len(kb)
    PAA = np.array([10.0, 8.0, 6.0])
    PAB = np.array([7.0, 5.0, 3.0])
    PBB = np.array([12.0, 9.0, 5.0])
    nA, nB = 1e-2, 2e-2
    cov = FI.multitracer_analytic_covariance(box, kb, PAA, PAB, PBB, nA, nB)
    nmodes = 2.0 / FI.band_power_log_variance(box, kb)
    for b in range(nb):
        tA, tB, x = PAA[b] + 1 / nA, PBB[b] + 1 / nB, PAB[b]
        exp = (
            np.array(
                [
                    [2 * tA**2, 2 * tA * x, 2 * x**2],
                    [2 * tA * x, tA * tB + x**2, 2 * tB * x],
                    [2 * x**2, 2 * tB * x, 2 * tB**2],
                ]
            )
            / nmodes[b]
        )
        idx = [b, nb + b, 2 * nb + b]
        assert np.allclose(cov[np.ix_(idx, idx)], exp)
    assert cov[0, 1] == 0.0  # different k-bins are uncorrelated


def test_analytic_covariance_vs_white_noise_mock():
    # On a WHITE field (no within-shell P(k) variation) the Gaussian block is
    # exact, so it matches the mock covariance to the sampling floor. Two
    # deterministic-bias tracers hA=b1A w, hB=b1B w + independent shot noise.
    box = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
    kb = _kb(box, ns=(3, 4, 5, 6))
    nb = len(kb)
    b1A, b1B = 1.5, 2.5
    nA, nB = 1e-3, 1e-3
    N, V = box.n_mesh, box.box_size**3
    sigA, sigB = float(np.sqrt(N**3 / (nA * V))), float(np.sqrt(N**3 / (nB * V)))
    n_mock = 1000
    data = np.empty((n_mock, 3 * nb))
    for m in range(n_mock):
        w = mx.random.normal((N, N, N), key=mx.random.key(11000 + m))
        kA, kB = mx.random.split(mx.random.key(22000 + m))
        hA = b1A * w + sigA * mx.random.normal((N, N, N), key=kA)
        hB = b1B * w + sigB * mx.random.normal((N, N, N), key=kB)
        data[m] = np.asarray(FI._multitracer_vector(hA, hB, box, kb, None), np.float64)
    cov_mc = np.cov(data, rowvar=False)
    mean = data.mean(0)
    PAA = mean[:nb] - 1 / nA  # subtract shot to get the signal the analytic re-adds
    PAB = mean[nb : 2 * nb]
    PBB = mean[2 * nb :] - 1 / nB
    cov_an = FI.multitracer_analytic_covariance(box, kb, PAA, PAB, PBB, nA, nB)
    ratio = np.diag(cov_mc) / np.diag(cov_an)
    assert np.all(np.abs(ratio - 1.0) < 0.2), f"diag ratios {ratio}"


def test_shot_noise_band_power_units():
    # The white shot field has band power 1/n (the unit anchor for the covariance).
    box = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
    kb = _kb(box, ns=(2, 3, 4, 5))
    n = 1e-3
    N, V = box.n_mesh, box.box_size**3
    sig = float(np.sqrt(N**3 / (n * V)))
    P = np.mean(
        [
            np.asarray(
                F.band_power(
                    sig * mx.random.normal((N, N, N), key=mx.random.key(500 + s)),
                    box,
                    kb,
                ),
                np.float64,
            )
            for s in range(80)
        ],
        axis=0,
    )
    assert np.all(np.abs(P * n - 1.0) < 0.1)


# --- universality tie -----------------------------------------------------------


def test_universality_b2_gives_universal_bphi():
    # universality_b2 makes the toy's emergent b_phi = 2 b2 A sigma^2 equal the
    # universality value 2 delta_c (b1-1).
    b1, A = 2.0, 1.0
    sigma2 = B.mesh_variance(BOX, COSMO, backend=BACKEND)
    b2 = FI.universality_b2(b1, BOX, COSMO, A=A, delta_c=DELTA_C, backend=BACKEND)
    b_phi = 2.0 * b2 * A * sigma2
    assert np.isclose(b_phi, 2.0 * DELTA_C * (b1 - 1.0))


def test_universality_bphi_tie_matches_direct_jacobian():
    # The chain-rule pushforward J_tied = J_free @ T equals directly
    # differentiating the universality-tied explicit-b_phi model. Validates that
    # the tie correctly folds the b_phi columns into b1.
    box = BoxConfig(box_size=512.0, n_mesh=32, n_particles=32)
    kb = _kb(box, ns=(1, 2, 3, 4, 6))
    th = FI.universality_bphi_fiducial(80.0, 1.0, 1.5, 2.5, delta_c=DELTA_C)
    Jf, _ = FI.linear_multitracer_bphi_jacobian(
        box, COSMO, th, kb, seed=0, backend=BACKEND
    )
    T, names, _ = FI.universality_bphi_tie_matrix(th, delta_c=DELTA_C)
    Jt = Jf @ T

    # direct: differentiate the tied model (b_phi_i = 2 delta_c (b1_i - 1)) wrt the
    # tied params via matched-phase FD.
    invM = FI._inv_M_grid(box, COSMO, backend=BACKEND)

    def tied_vec(f_NL, A, b1A, b1B, seed):
        delta = A * IC.linear_density(box, COSMO, seed=seed, f_NL=0.0, backend=BACKEND)
        bpA, bpB = 2 * DELTA_C * (b1A - 1), 2 * DELTA_C * (b1B - 1)
        hA = B.scale_dependent_bias_tracer(
            delta, box, COSMO, b1A, bpA, mx.array(f_NL), invM=invM
        )
        hB = B.scale_dependent_bias_tracer(
            delta, box, COSMO, b1B, bpB, mx.array(f_NL), invM=invM
        )
        return np.asarray(FI._multitracer_vector(hA, hB, box, kb, None), np.float64)

    base = [th["f_NL"], th["A"], th["b1_A"], th["b1_B"]]
    steps = [5.0, 1e-2, 1e-2, 1e-2]
    for k, (nm, st) in enumerate(zip(names, steps)):
        hi = list(base)
        hi[k] += st
        lo = list(base)
        lo[k] -= st
        fd = (tied_vec(*hi, 0) - tied_vec(*lo, 0)) / (2 * st)
        assert _col_err(Jt[:, k], fd) < 5e-3, f"tied col {nm}"


# --- explicit-b_phi: Jacobian and the degeneracy structure ----------------------


def _bphi_vec(theta, seed, kb):
    f_NL = mx.array(theta["f_NL"])
    delta = theta["A"] * IC.linear_density(
        BOX_DEG, COSMO, seed=seed, f_NL=0.0, backend=BACKEND
    )
    invM = FI._inv_M_grid(BOX_DEG, COSMO, backend=BACKEND)
    hA = B.scale_dependent_bias_tracer(
        delta, BOX_DEG, COSMO, theta["b1_A"], theta["bphi_A"], f_NL, invM=invM
    )
    hB = B.scale_dependent_bias_tracer(
        delta, BOX_DEG, COSMO, theta["b1_B"], theta["bphi_B"], f_NL, invM=invM
    )
    return np.asarray(FI._multitracer_vector(hA, hB, BOX_DEG, kb, None), np.float64)


def test_bphi_jacobian_vs_fd():
    kb = _kb(BOX_DEG, ns=(1, 2, 3, 4, 6, 8))
    J, _ = FI.linear_multitracer_bphi_jacobian(
        BOX_DEG, COSMO, THETA_BPHI, kb, seed=0, backend=BACKEND
    )
    for i, p in enumerate(FI.PARAM_NAMES_MT_BPHI):
        hi = dict(THETA_BPHI, **{p: THETA_BPHI[p] + H_BPHI[p]})
        lo = dict(THETA_BPHI, **{p: THETA_BPHI[p] - H_BPHI[p]})
        fd = (_bphi_vec(hi, 0, kb) - _bphi_vec(lo, 0, kb)) / (2 * H_BPHI[p])
        assert _col_err(J[:, i], fd) < 2e-3, f"{p}"


def test_bphi_degeneracy_free_singular_tied_recovered():
    # The headline: explicit-b_phi at f_NL != 0. FREE -> f_NL is degenerate with
    # b_phi (the Fisher is singular: a huge condition number). TIED (universality)
    # -> f_NL is recovered (finite), and 2 tracers are much tighter than 1.
    kb = _kb(BOX_DEG, ns=(1, 2, 3, 4, 6, 8, 12, 16))
    nb = len(kb)
    J = np.zeros((3 * nb, 6))
    for s in range(4):
        J += FI.linear_multitracer_bphi_jacobian(
            BOX_DEG, COSMO, THETA_BPHI, kb, seed=s, backend=BACKEND
        )[0]
    J /= 4
    cov = FI.multitracer_bphi_gaussian_covariance(
        BOX_DEG, COSMO, THETA_BPHI, kb, N_A, N_B, n_mock=400, backend=BACKEND
    )
    priors = {"A": 0.1}
    tie = FI.universality_bphi_tie_matrix(THETA_BPHI, delta_c=DELTA_C)
    fc2_free = FI.multitracer_forecast(
        J, cov, THETA_BPHI, FI.PARAM_NAMES_MT_BPHI, priors=priors
    )
    fc2_tied = FI.multitracer_forecast(
        J, cov, THETA_BPHI, FI.PARAM_NAMES_MT_BPHI, priors=priors, tie=tie
    )

    # free: degenerate in (f_NL, b_phi) -> the Fisher is hugely ill-conditioned.
    assert fc2_free.condition_number() > 1e12

    # 1-tracer (AA only), tied: a finite f_NL constraint.
    JA = J[:nb][:, [0, 1, 2, 3]]
    covA = cov[:nb, :nb]
    T1 = np.zeros((4, 3))
    T1[0, 0] = T1[1, 1] = T1[2, 2] = 1.0
    T1[3, 2] = 2 * DELTA_C
    fc1_tied = FI.FisherForecast(
        JA @ T1,
        covariance=covA,
        fiducial_params={k: THETA_BPHI[k] for k in ("f_NL", "A", "b1_A")},
        param_names=("f_NL", "A", "b1_A"),
        priors=priors,
    )

    s1, s2 = fc1_tied.sigma("f_NL"), fc2_tied.sigma("f_NL")
    assert np.isfinite(s1) and np.isfinite(s2)
    assert s2 < s1  # multi-tracer is tighter (the |b1_A - b1_B| differential gain)
    assert s1 / s2 > 1.5  # measured ~2.7x; a comfortable floor
