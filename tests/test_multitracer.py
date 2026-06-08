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
import pytest

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import fields as F
from mbody import ic as IC
from mbody import bias as B
from mbody import integrate as IN
from mbody import fisher as FI
from mbody import cosmology as C
from mbody import rsd as RS

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


# --- redshift-space multi-tracer (the composition capstone) ---------------------
#
# Two local-bias tracers painted from the SAME redshift-space field, summarized by
# their auto/cross multipoles: the quadrupole's f_growth handle stacks on top of the
# multi-tracer cancellation, and the f_NL/A gradients flow through the momentum-seeded
# adjoint (the loss depends on the final velocities). Native tracer only; tolerances
# measured in scripts/probe_rsd_multitracer.py.

LOS = 0
ELLS = (0, 2)
THETA_RSD = FI.universality_rsd_fiducial(
    0.0, 1.0, 1.0, 1.5, 2.5, BOX, COSMO, backend=BACKEND
)
H_RSD = {
    "f_NL": 50.0,
    "A": 1e-2,
    "f_growth": 1e-2,
    "b1_A": 1e-2,
    "b2_A": 1e-2,
    "b1_B": 1e-2,
    "b2_B": 1e-2,
}


def _rsd_mt_vec(theta, seed, kb):
    f_NL = mx.array(theta["f_NL"])
    delta = theta["A"] * IC.linear_density(
        BOX, COSMO, seed=seed, f_NL=f_NL, backend=BACKEND
    )
    feff = theta["f_growth"] * C.growth_rate(0.0, COSMO)
    hA = FI._redshift_tracer_linear(delta, BOX, theta["b1_A"], theta["b2_A"], feff, LOS)
    hB = FI._redshift_tracer_linear(delta, BOX, theta["b1_B"], theta["b2_B"], feff, LOS)
    return np.asarray(
        FI._mt_multipole_vector(hA, hB, BOX, kb, ELLS, LOS, None), np.float64
    )


def test_linear_rsd_multitracer_jacobian_vs_fd():
    # Each linear RSD-MT column matches matched-phase FD; measured worst ~2e-4 (b2).
    kb = _kb(BOX)
    J, _ = FI.linear_multitracer_multipole_jacobian(
        BOX, COSMO, THETA_RSD, kb, ells=ELLS, los_axis=LOS, seed=0, backend=BACKEND
    )
    for i, p in enumerate(FI.PARAM_NAMES_MT_RSD):
        hi = dict(THETA_RSD, **{p: THETA_RSD[p] + H_RSD[p]})
        lo = dict(THETA_RSD, **{p: THETA_RSD[p] - H_RSD[p]})
        fd = (_rsd_mt_vec(hi, 0, kb) - _rsd_mt_vec(lo, 0, kb)) / (2 * H_RSD[p])
        assert _col_err(J[:, i], fd) < 1e-3, f"{p}"


def test_pm_rsd_multitracer_downstream_vs_fd():
    # The downstream columns (f_growth + the four biases) are an mx.grad at the FIXED
    # final (x, p). Validate by FD on that same fixed state. f_growth's FD is limited
    # by the periodic-wrap floor of the redshift map (measured ~2e-3); the biases ~1e-5.
    kb = _kb(BOX)
    xf, pf = IN.leapfrog(
        BOX,
        COSMO,
        TIME_PM,
        seed=0,
        f_NL=mx.array(THETA_RSD["f_NL"]),
        amplitude=mx.array(THETA_RSD["A"]),
        lpt_order=2,
    )
    xf, pf = mx.stop_gradient(xf), mx.stop_gradient(pf)

    def vec(t):
        s = RS.redshift_space_positions(
            xf, pf, BOX, COSMO, z=TIME_PM.z_final, los_axis=LOS, f_growth=t[0]
        )
        field = F.interlaced_density_contrast(s, BOX)
        hA = B.local_bias_tracer(field, t[1], t[2])
        hB = B.local_bias_tracer(field, t[3], t[4])
        return FI._mt_multipole_vector(hA, hB, BOX, kb, ELLS, LOS, None)

    t0 = mx.array(
        [
            THETA_RSD["f_growth"],
            THETA_RSD["b1_A"],
            THETA_RSD["b2_A"],
            THETA_RSD["b1_B"],
            THETA_RSD["b2_B"],
        ]
    )
    n_data = 3 * len(ELLS) * len(kb)
    Jd = np.array(
        [
            np.asarray(mx.grad(lambda t, d=d: vec(t)[d])(t0), np.float64)
            for d in range(n_data)
        ]
    )
    eye = np.eye(5)
    names = ["f_growth", "b1_A", "b2_A", "b1_B", "b2_B"]
    for j, nm in enumerate(names):
        hi = mx.array(np.asarray(t0) + H_RSD[nm] * eye[j])
        lo = mx.array(np.asarray(t0) - H_RSD[nm] * eye[j])
        fd = (np.asarray(vec(hi)) - np.asarray(vec(lo))) / (2 * H_RSD[nm])
        assert _col_err(Jd[:, j], fd) < 5e-3, f"downstream {nm}"


def test_pm_rsd_multitracer_adjoint_matches_replay():
    # The IC columns (f_NL, A) come from the momentum-seeded adjoint (one shared sweep
    # per component). Validate against replay mx.grad for the AA, AB AND BB quadrupole
    # pivot bins (the novel two-tracer cross-multipole path) -- both autodiff, agreeing
    # to the float32 CIC floor (measured ~3e-7).
    kb = _kb(BOX)
    nb = len(kb)

    def loss_at(x, p, d):
        s = RS.redshift_space_positions(
            x,
            p,
            BOX,
            COSMO,
            z=TIME_PM.z_final,
            los_axis=LOS,
            f_growth=THETA_RSD["f_growth"],
        )
        field = F.interlaced_density_contrast(s, BOX)
        hA = B.local_bias_tracer(field, THETA_RSD["b1_A"], THETA_RSD["b2_A"])
        hB = B.local_bias_tracer(field, THETA_RSD["b1_B"], THETA_RSD["b2_B"])
        return FI._mt_multipole_vector(hA, hB, BOX, kb, ELLS, LOS, None)[d]

    def replay(tv, d):
        x, p = IN.leapfrog(
            BOX, COSMO, TIME_PM, seed=0, f_NL=tv[0], amplitude=tv[1], lpt_order=2
        )
        return loss_at(x, p, d)

    tv0 = mx.array([THETA_RSD["f_NL"], THETA_RSD["A"]])
    for d in (nb, 3 * nb, 5 * nb):  # AA, AB, BB quadrupole pivot bins
        g_adj = np.asarray(
            IN.adjoint_grad_ic(
                lambda x, p, d=d: loss_at(x, p, d),
                BOX,
                COSMO,
                TIME_PM,
                seed=0,
                f_NL=THETA_RSD["f_NL"],
                amplitude=THETA_RSD["A"],
                loss_uses_momentum=True,
            )
        )
        g_rep = np.asarray(mx.grad(lambda tv, d=d: replay(tv, d))(tv0))
        assert np.all(np.abs(g_adj / g_rep - 1.0) < 1e-4), f"d={d}"


def test_rsd_shot_noise_is_monopole():
    # A white shot field is pure-monopole with band power 1/n (the unit anchor); the
    # raw quadrupole carries only the discrete-shell leakage (<< the monopole). So shot
    # enters the data-vector MEAN in the monopole auto power only.
    kb = _kb(BOX, ns=(2, 3, 4, 5))
    n = 1e-3
    N, V = BOX.n_mesh, BOX.box_size**3
    sig = float(np.sqrt(N**3 / (n * V)))
    P0s, P2s = [], []
    for s in range(80):
        w = sig * mx.random.normal((N, N, N), key=mx.random.key(700 + s))
        P0s.append(
            np.asarray(F.band_power_multipole(w, BOX, kb, 0, los_axis=LOS), np.float64)
        )
        P2s.append(
            np.asarray(F.band_power_multipole(w, BOX, kb, 2, los_axis=LOS), np.float64)
        )
    P0 = np.mean(P0s, axis=0) * n
    P2 = np.mean(P2s, axis=0) * n
    assert np.all(np.abs(P0 - 1.0) < 0.1)  # monopole band power = 1/n
    assert np.all(np.abs(P2) < 0.5 * P0)  # quadrupole is raw-shell leakage only


def _rsd_head(box, theta, ells, nseed=3):
    """1- and 2-tracer tied RSD-MT forecasts at a fixed ells set (headline helper)."""
    kb = np.arange(1.5, 8.0, 1.0) * box.k_fundamental
    ne, nb = len(ells), len(kb)
    J = np.zeros((3 * ne * nb, 7))
    for s in range(nseed):
        J += FI.linear_multitracer_multipole_jacobian(
            box, COSMO, theta, kb, ells=ells, los_axis=LOS, seed=s, backend=BACKEND
        )[0]
    J /= nseed
    cov = FI.multitracer_multipole_gaussian_covariance(
        box,
        COSMO,
        theta,
        kb,
        5e-4,
        5e-4,
        ells=ells,
        los_axis=LOS,
        n_mock=250,
        backend=BACKEND,
    )
    priors = {"A": 0.1}
    tie = FI.universality_rsd_tie_matrix(theta, box, COSMO, backend=BACKEND)
    fc2 = FI.multitracer_forecast(
        J, cov, theta, FI.PARAM_NAMES_MT_RSD, priors=priors, tie=tie
    )
    naa = ne * nb
    JA = J[:naa][:, [0, 1, 2, 3, 4]]
    covA = cov[:naa, :naa]
    s2 = B.mesh_variance(box, COSMO, backend=BACKEND)
    A, b1 = theta["A"], theta["b1_A"]
    T1 = np.zeros((5, 4))
    T1[0, 0] = T1[1, 1] = T1[2, 2] = T1[3, 3] = 1.0
    T1[4, 1] = -DELTA_C * (b1 - 1) / (A**2 * s2)
    T1[4, 3] = DELTA_C / (A * s2)
    fc1 = FI.FisherForecast(
        JA @ T1,
        covariance=covA,
        fiducial_params={n: theta[n] for n in ("f_NL", "A", "f_growth", "b1_A")},
        param_names=("f_NL", "A", "f_growth", "b1_A"),
        priors=priors,
    )
    return fc1, fc2


def test_rsd_multitracer_headline_cancellation_and_quadrupole():
    # The composition headline: the second tracer's sample-variance cancellation
    # tightens sigma(f_NL) (measured ~1.8-2.1x), and the quadrupole sharply pins
    # sigma(f_growth) (measured >5x on 1-tracer). Both stack. Tied universality basis.
    box = BoxConfig(box_size=1024.0, n_mesh=32, n_particles=32)
    theta = FI.universality_rsd_fiducial(
        0.0, 1.0, 1.0, 1.5, 2.5, box, COSMO, backend=BACKEND
    )
    fc1_m, fc2_m = _rsd_head(box, theta, (0,))
    fc1_q, _ = _rsd_head(box, theta, (0, 2))
    assert fc2_m.sigma("f_NL") < fc1_m.sigma("f_NL")
    assert fc1_m.sigma("f_NL") / fc2_m.sigma("f_NL") > 1.4  # cancellation gain
    assert fc1_q.sigma("f_growth") < 0.5 * fc1_m.sigma(
        "f_growth"
    )  # quadrupole pins f_growth


# --- coverage closures (Stage-2 review) -----------------------------------------


def test_mock_covariance_rejects_too_few_mocks():
    # The Hartlap factor h = (n_mock - n_data - 2)/(n_mock - 1) flips sign when
    # n_mock is too small; the builder must raise rather than silently return a
    # sign-flipped (Fisher-corrupting) covariance.
    kb = _kb(BOX, ns=(1, 2, 3))  # n_data = 3 * 3 = 9 for the multi-tracer vector
    with pytest.raises(ValueError):
        FI.multitracer_gaussian_covariance(
            BOX, COSMO, THETA, kb, N_A, N_B, n_mock=8, backend=BACKEND
        )


def test_universality_native_tie_matches_direct_jacobian():
    # J_tied = J_free @ T equals directly differentiating the universality-tied
    # NATIVE (b2-sourced) model. Unlike the explicit-b_phi tie, the native (and
    # RSD) ties carry the A-coupled cross-term db2/dA, so they get their own check.
    box = BoxConfig(box_size=512.0, n_mesh=32, n_particles=32)
    kb = _kb(box, ns=(1, 2, 3, 4, 6))
    th = FI.universality_fiducial(30.0, 1.0, 1.5, 2.5, box, COSMO, backend=BACKEND)
    Jf, _ = FI.linear_multitracer_jacobian(box, COSMO, th, kb, seed=0, backend=BACKEND)
    T, names, _ = FI.universality_tie_matrix(th, box, COSMO, backend=BACKEND)
    Jt = Jf @ T

    def tied_vec(f_NL, A, b1A, b1B, seed):
        delta = A * IC.linear_density(box, COSMO, seed=seed, f_NL=f_NL, backend=BACKEND)
        b2A = FI.universality_b2(b1A, box, COSMO, A=A, backend=BACKEND)
        b2B = FI.universality_b2(b1B, box, COSMO, A=A, backend=BACKEND)
        hA = B.local_bias_tracer(delta, b1A, b2A)
        hB = B.local_bias_tracer(delta, b1B, b2B)
        return np.asarray(FI._multitracer_vector(hA, hB, box, kb, None), np.float64)

    base = [th["f_NL"], th["A"], th["b1_A"], th["b1_B"]]
    steps = [5.0, 1e-2, 1e-2, 1e-2]
    for k, (nm, st) in enumerate(zip(names, steps)):
        hi, lo = list(base), list(base)
        hi[k] += st
        lo[k] -= st
        fd = (tied_vec(*hi, 0) - tied_vec(*lo, 0)) / (2 * st)
        assert _col_err(Jt[:, k], fd) < 5e-3, f"tied col {nm}"


def test_universality_rsd_tie_matches_direct_jacobian():
    # The redshift-space sibling of the native tie check: the 7x5 tie with the
    # f_growth passthrough. J_tied = J_free @ T vs direct differentiation of the
    # tied redshift-space multipole model.
    box = BoxConfig(box_size=512.0, n_mesh=32, n_particles=32)
    kb = _kb(box, ns=(1, 2, 3, 4, 6))
    th = FI.universality_rsd_fiducial(
        30.0, 1.0, 1.0, 1.5, 2.5, box, COSMO, backend=BACKEND
    )
    Jf, _ = FI.linear_multitracer_multipole_jacobian(
        box, COSMO, th, kb, ells=ELLS, los_axis=LOS, seed=0, backend=BACKEND
    )
    T, names, _ = FI.universality_rsd_tie_matrix(th, box, COSMO, backend=BACKEND)
    Jt = Jf @ T
    f_lin = C.growth_rate(0.0, COSMO)

    def tied_vec(f_NL, A, fg, b1A, b1B, seed):
        delta = A * IC.linear_density(box, COSMO, seed=seed, f_NL=f_NL, backend=BACKEND)
        b2A = FI.universality_b2(b1A, box, COSMO, A=A, backend=BACKEND)
        b2B = FI.universality_b2(b1B, box, COSMO, A=A, backend=BACKEND)
        hA = FI._redshift_tracer_linear(delta, box, b1A, b2A, fg * f_lin, LOS)
        hB = FI._redshift_tracer_linear(delta, box, b1B, b2B, fg * f_lin, LOS)
        return np.asarray(
            FI._mt_multipole_vector(hA, hB, box, kb, ELLS, LOS, None), np.float64
        )

    base = [th["f_NL"], th["A"], th["f_growth"], th["b1_A"], th["b1_B"]]
    steps = [5.0, 1e-2, 1e-2, 1e-2, 1e-2]
    for k, (nm, st) in enumerate(zip(names, steps)):
        hi, lo = list(base), list(base)
        hi[k] += st
        lo[k] -= st
        fd = (tied_vec(*hi, 0) - tied_vec(*lo, 0)) / (2 * st)
        assert _col_err(Jt[:, k], fd) < 5e-3, f"tied col {nm}"


def test_pm_multitracer_multipole_jacobian_ic_columns_match_replay():
    # The assembled PM redshift-space multi-tracer multipole Jacobian (the
    # composition-capstone function) is otherwise uncalled: exercise its IC
    # (f_NL, A) column routing end to end against replay mx.grad, for the AA/AB/BB
    # quadrupole pivot bins (both autodiff -> agree to the CIC scatter floor).
    kb = _kb(BOX)
    nb = len(kb)
    J, _ = FI.pm_multitracer_multipole_jacobian(
        BOX,
        COSMO,
        TIME_PM,
        THETA_RSD,
        kb,
        ells=ELLS,
        los_axis=LOS,
        seed=0,
        backend=BACKEND,
    )

    def loss_at(x, p, d):
        s = RS.redshift_space_positions(
            x,
            p,
            BOX,
            COSMO,
            z=TIME_PM.z_final,
            los_axis=LOS,
            f_growth=THETA_RSD["f_growth"],
        )
        field = F.interlaced_density_contrast(s, BOX)
        hA = B.local_bias_tracer(field, THETA_RSD["b1_A"], THETA_RSD["b2_A"])
        hB = B.local_bias_tracer(field, THETA_RSD["b1_B"], THETA_RSD["b2_B"])
        return FI._mt_multipole_vector(hA, hB, BOX, kb, ELLS, LOS, None)[d]

    def replay(tv, d):
        x, p = IN.leapfrog(
            BOX,
            COSMO,
            TIME_PM,
            seed=0,
            f_NL=tv[0],
            amplitude=tv[1],
            backend=BACKEND,
            lpt_order=2,
        )
        return loss_at(x, p, d)

    tv0 = mx.array([THETA_RSD["f_NL"], THETA_RSD["A"]])
    for d in (nb, 3 * nb, 5 * nb):  # AA, AB, BB quadrupole pivot bins
        g_rep = np.asarray(mx.grad(lambda tv, d=d: replay(tv, d))(tv0))
        assert _col_err(J[d, 0:2], g_rep) < 1e-3, f"IC cols d={d}"
