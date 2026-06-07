"""Probe the multi-tracer Fisher pieces before fixing any test tolerances.

Two local-bias tracers A, B painted from the SAME field break part of the
b_phi-f_NL degeneracy via sample-variance cancellation (Seljak 2009; Barreira &
Krause 2023). This probe MEASURES every number tests/test_multitracer.py later
asserts (the standing "measure first, never relax a tolerance" rule), across the
two scientific regimes:

  NATIVE tracer (b_phi emergent from b2), the DETECTION regime (fiducial f_NL=0):
    V1  linear {P_AA,P_AB,P_BB} Jacobian columns vs matched-phase central FD
    V2  PM adjoint_grad_ic (f_NL, A) vs replay mx.grad, for AA / AB / BB
    V3  full PM multi-tracer Jacobian vs matched-phase FD
    V4  analytic Gaussian block covariance vs the mock ensemble (+ shot noise)
    V5  shot-noise unit anchor: band power of a 1/n white field (target 1.0)
    V6  cross-correlation r(k) + the identical-tracer cancellation limit
    H1  sample-variance-cancellation headline: 1-tracer vs 2-tracer sigma(f_NL)

  EXPLICIT-b_phi tracer (b_phi a free k^-2 parameter), the DEGENERACY regime
  (fiducial f_NL != 0 -- the product degeneracy is invisible at f_NL=0):
    V7  linear explicit-b_phi Jacobian columns vs matched-phase FD
    H2  the b_phi-f_NL degeneracy: free -> only the product f_NL*b_phi is
        constrained (sigma(f_NL) degenerate); universality-tied -> f_NL recovered,
        2-tracer much tighter than 1-tracer (the |b1_A-b1_B| gain).

Run: pixi run python scripts/probe_multitracer.py
Nothing is written; the printed numbers set tolerances in tests/test_multitracer.py.
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import fields as F
from mbody import ic as IC
from mbody import bias as B
from mbody import integrate as IN
from mbody import fisher as FI

COSMO = Cosmology()
BOX = BoxConfig(box_size=256.0, n_mesh=64, n_particles=64)
BOX_PM = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
TIME_PM = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5)
BACKEND = "eh98"
INTEG = "fastpm"
DELTA_C = 1.686

# Two tracers with different linear bias (hence different b_phi). Number densities
# n_i (per (Mpc/h)^3) set the shot noise 1/n_i; chosen so P*n is order a few.
B1_A, B1_B, A_FID = 1.5, 2.5, 1.0
N_A = N_B = 5.0e-4
PRIORS = {"A": 0.1}

# Native (b2-sourced) fiducial at f_NL=0, with b2_i at the universality value so
# free and tied compare at the same physical point.
THETA = FI.universality_fiducial(0.0, A_FID, B1_A, B1_B, BOX, COSMO, backend=BACKEND)
THETA_PM = FI.universality_fiducial(
    0.0, A_FID, B1_A, B1_B, BOX_PM, COSMO, backend=BACKEND
)

# Explicit-b_phi degeneracy regime: a non-zero f_NL on a big (cheap) linear box.
# Box size is free in this toy, so use a large volume for a sensitive forecast;
# f_NL chosen so the scale-dependent bias Delta_b/b1 ~ 0.2 (a marginal detection).
BOX_DEG = BoxConfig(box_size=1024.0, n_mesh=64, n_particles=64)
FNL_DEG = 100.0
THETA_BPHI = FI.universality_bphi_fiducial(FNL_DEG, A_FID, B1_A, B1_B, delta_c=DELTA_C)

# Matched-phase FD steps per parameter (f_NL tiny per-unit response -> big step).
H = {"f_NL": 50.0, "A": 1e-2, "b1_A": 1e-2, "b2_A": 1e-2, "b1_B": 1e-2, "b2_B": 1e-2}
H_BPHI = {
    "f_NL": 5.0,
    "A": 1e-2,
    "b1_A": 1e-2,
    "bphi_A": 1e-2,
    "b1_B": 1e-2,
    "bphi_B": 1e-2,
}


def k_bins(box, ns=(1, 2, 3, 4, 6, 8)):
    return np.array([n * box.k_fundamental for n in ns])


def _col_err(jcol, fd):
    """Max |J - FD| residual normalized by the column's largest FD entry."""
    denom = max(float(np.abs(fd).max()), 1e-300)
    return float(np.abs(np.asarray(jcol) - fd).max() / denom)


# --- native tracer, linear field -----------------------------------------------


def _mt_vec(theta, seed, kb):
    f_NL = mx.array(theta["f_NL"])
    delta = theta["A"] * IC.linear_density(
        BOX, COSMO, seed=seed, f_NL=f_NL, backend=BACKEND
    )
    hA = B.local_bias_tracer(delta, theta["b1_A"], theta["b2_A"])
    hB = B.local_bias_tracer(delta, theta["b1_B"], theta["b2_B"])
    return np.asarray(FI._multitracer_vector(hA, hB, BOX, kb, None), np.float64)


def _fd_col(name, seed, kb, theta0, step):
    hi = dict(theta0, **{name: theta0[name] + step[name]})
    lo = dict(theta0, **{name: theta0[name] - step[name]})
    return (_mt_vec(hi, seed, kb) - _mt_vec(lo, seed, kb)) / (2.0 * step[name])


def probe_linear_jacobian():
    kb = k_bins(BOX)
    print("=== V1 native linear Jacobian: mx.grad vs matched-phase FD (4 seeds) ===")
    print("  max |J - FD| / max|FD| per column:")
    worst = {p: 0.0 for p in FI.PARAM_NAMES_MT}
    for s in range(4):
        J, _ = FI.linear_multitracer_jacobian(
            BOX, COSMO, THETA, kb, seed=2000 + s, backend=BACKEND
        )
        for i, p in enumerate(FI.PARAM_NAMES_MT):
            worst[p] = max(
                worst[p], _col_err(J[:, i], _fd_col(p, 2000 + s, kb, THETA, H))
            )
    for p in FI.PARAM_NAMES_MT:
        print(f"  {p:<6s}  {worst[p]:.2e}")


def _signal_spectra(theta, kb, nseed=8):
    Psum = np.zeros(3 * len(kb))
    for s in range(nseed):
        _, Pf = FI.linear_multitracer_jacobian(
            BOX, COSMO, theta, kb, seed=5000 + s, backend=BACKEND
        )
        Psum += Pf
    Pf = Psum / nseed
    nb = len(kb)
    return Pf[:nb], Pf[nb : 2 * nb], Pf[2 * nb :]


def probe_covariance():
    kb = k_bins(BOX)
    nb = len(kb)
    print("\n=== V4 covariance: analytic Gaussian block vs mock ensemble ===")
    PAA, PAB, PBB = _signal_spectra(THETA, kb)
    print(
        f"  P_AA*n_A (pivot bin) = {PAA[0] * N_A:.2f}   P_BB*n_B = {PBB[0] * N_B:.2f}"
    )
    cov_an = FI.multitracer_analytic_covariance(BOX, kb, PAA, PAB, PBB, N_A, N_B)
    cov_mc = FI.multitracer_gaussian_covariance(
        BOX, COSMO, THETA, kb, N_A, N_B, n_mock=800, backend=BACKEND
    )
    ratio = np.diag(cov_mc) / np.diag(cov_an)
    labels = ["AA"] * nb + ["AB"] * nb + ["BB"] * nb
    print("  diag(mock)/diag(analytic) (target 1; sampling ~%.2f):" % np.sqrt(2 / 799))
    print("  (the colored field inflates the steep low-k bins -- mock is ground truth)")
    for lab, r in zip(labels, ratio):
        print(f"   {lab}  {r:.3f}")
    print(f"  worst |ratio-1| = {np.abs(ratio - 1).max():.3f}")


def probe_shot():
    kb = k_bins(BOX)
    N, V = BOX.n_mesh, BOX.box_size**3
    sig = float(np.sqrt(N**3 / (N_A * V)))
    print("\n=== V5 shot-noise unit anchor: band_power(1/n field) * n (target 1) ===")
    Ps = [
        np.asarray(
            F.band_power(
                sig * mx.random.normal((N, N, N), key=mx.random.key(9000 + s)), BOX, kb
            ),
            np.float64,
        )
        for s in range(60)
    ]
    print("  per bin:", np.array2string(np.mean(Ps, axis=0) * N_A, precision=3))


def probe_cancellation():
    kb = k_bins(BOX)
    print("\n=== V6 cross-correlation r(k) & identical-tracer limit ===")
    PAA, PAB, PBB = _signal_spectra(THETA, kb)
    tA, tB = PAA + 1.0 / N_A, PBB + 1.0 / N_B
    print("  r(k)=P_AB/sqrt(Ptot_AA Ptot_BB):")
    print(
        "   deterministic (no shot):",
        np.array2string(PAB / np.sqrt(PAA * PBB), precision=3),
    )
    print(
        "   with shot noise        :",
        np.array2string(PAB / np.sqrt(tA * tB), precision=3),
    )
    theta_id = dict(THETA, b1_B=THETA["b1_A"], b2_B=THETA["b2_A"])
    cov_id = FI.multitracer_gaussian_covariance(
        BOX, COSMO, theta_id, kb, N_A, N_B, n_mock=800, backend=BACKEND
    )
    cov_diff = FI.multitracer_gaussian_covariance(
        BOX, COSMO, THETA, kb, N_A, N_B, n_mock=800, backend=BACKEND
    )
    print(
        f"  cond(cov): identical tracers {np.linalg.cond(cov_id):.2e}  "
        f"different {np.linalg.cond(cov_diff):.2e}"
    )


def _one_tracer(J, cov, fiducial, nb, names4, names3=None, tie=None):
    """Slice the AA-only (single-tracer) forecast from the MT J/cov. names4 are
    the 4 free params (f_NL, A, b1_X, second_X); tie/names3 give the tied version.
    """
    JA = J[:nb][:, [0, 1, 2, 3]]
    covA = cov[:nb, :nb]
    fid4 = {n: fiducial[n] for n in names4}
    if tie is None:
        return FI.FisherForecast(
            JA, covariance=covA, fiducial_params=fid4, param_names=names4, priors=PRIORS
        )
    return FI.FisherForecast(
        JA @ tie,
        covariance=covA,
        fiducial_params={n: fiducial[n] for n in names3},
        param_names=names3,
        priors=PRIORS,
    )


def probe_headline_cancellation():
    kb = k_bins(BOX)
    nb = len(kb)
    print("\n=== H1 cancellation headline (native, linear, f_NL=0) ===")
    J = np.zeros((3 * nb, 6))
    for s in range(8):
        J += FI.linear_multitracer_jacobian(
            BOX, COSMO, THETA, kb, seed=s, backend=BACKEND
        )[0]
    J /= 8
    cov = FI.multitracer_gaussian_covariance(
        BOX, COSMO, THETA, kb, N_A, N_B, n_mock=800, backend=BACKEND
    )
    tie = FI.universality_tie_matrix(THETA, BOX, COSMO, backend=BACKEND)
    fc2 = FI.multitracer_forecast(
        J, cov, THETA, FI.PARAM_NAMES_MT, priors=PRIORS, tie=tie
    )
    A, b1 = A_FID, B1_A
    s2 = B.mesh_variance(BOX, COSMO, backend=BACKEND)
    T1 = np.zeros((4, 3))
    T1[0, 0] = T1[1, 1] = T1[2, 2] = 1.0
    T1[3, 1] = -DELTA_C * (b1 - 1) / (A**2 * s2)
    T1[3, 2] = DELTA_C / (A * s2)
    fc1 = _one_tracer(
        J, cov, THETA, nb, ["f_NL", "A", "b1_A", "b2_A"], ["f_NL", "A", "b1_A"], tie=T1
    )
    print(
        f"  tied sigma(f_NL): 1-tracer {fc1.sigma('f_NL'):.4g}  2-tracer {fc2.sigma('f_NL'):.4g}"
    )
    print(
        f"  --> sample-variance-cancellation gain (1t/2t) = {fc1.sigma('f_NL')/fc2.sigma('f_NL'):.2f}x"
    )


# --- native tracer, PM field ---------------------------------------------------


def _pm_mt_vec(theta, seed, kb):
    x, _ = IN.leapfrog(
        BOX_PM,
        COSMO,
        TIME_PM,
        seed=seed,
        f_NL=mx.array(theta["f_NL"]),
        amplitude=mx.array(theta["A"]),
        backend=BACKEND,
        integrator=INTEG,
        lpt_order=2,
    )
    field = F.interlaced_density_contrast(x, BOX_PM)
    hA = B.local_bias_tracer(field, theta["b1_A"], theta["b2_A"])
    hB = B.local_bias_tracer(field, theta["b1_B"], theta["b2_B"])
    return np.asarray(FI._multitracer_vector(hA, hB, BOX_PM, kb, None), np.float64)


def _pm_fd_col(name, seed, kb, theta0):
    hi = dict(theta0, **{name: theta0[name] + H[name]})
    lo = dict(theta0, **{name: theta0[name] - H[name]})
    return (_pm_mt_vec(hi, seed, kb) - _pm_mt_vec(lo, seed, kb)) / (2.0 * H[name])


def probe_pm():
    kb = k_bins(BOX_PM, ns=(1, 2, 3, 4))
    nb = len(kb)
    print(
        f"\n=== V3 PM native Jacobian (N={BOX_PM.n_mesh}, {TIME_PM.n_steps} steps) vs FD ==="
    )
    J, _ = FI.pm_multitracer_jacobian(
        BOX_PM, COSMO, TIME_PM, THETA_PM, kb, seed=0, backend=BACKEND, integrator=INTEG
    )
    for i, p in enumerate(FI.PARAM_NAMES_MT):
        print(f"  {p:<6s}  {_col_err(J[:, i], _pm_fd_col(p, 0, kb, THETA_PM)):.2e}")

    print("=== V2 adjoint_grad_ic vs replay mx.grad (f_NL, A), AA/AB/BB pivot bins ===")

    def loss_at(x, d):
        field = F.interlaced_density_contrast(x, BOX_PM)
        hA = B.local_bias_tracer(field, THETA_PM["b1_A"], THETA_PM["b2_A"])
        hB = B.local_bias_tracer(field, THETA_PM["b1_B"], THETA_PM["b2_B"])
        return FI._multitracer_vector(hA, hB, BOX_PM, kb, None)[d]

    def replay(tv, d):
        x, _ = IN.leapfrog(
            BOX_PM,
            COSMO,
            TIME_PM,
            seed=0,
            f_NL=tv[0],
            amplitude=tv[1],
            backend=BACKEND,
            integrator=INTEG,
            lpt_order=2,
        )
        return loss_at(x, d)

    tv0 = mx.array([THETA_PM["f_NL"], THETA_PM["A"]])
    for lab, d in [("AA", 0), ("AB", nb), ("BB", 2 * nb)]:
        g_adj = np.asarray(
            IN.adjoint_grad_ic(
                lambda x, d=d: loss_at(x, d),
                BOX_PM,
                COSMO,
                TIME_PM,
                seed=0,
                f_NL=THETA_PM["f_NL"],
                amplitude=THETA_PM["A"],
                backend=BACKEND,
                integrator=INTEG,
                lpt_order=2,
            )
        )
        g_rep = np.asarray(mx.grad(lambda tv, d=d: replay(tv, d))(tv0))
        rel = np.abs(g_adj / g_rep - 1.0)
        print(f"  {lab}: f_NL {rel[0]:.2e}  A {rel[1]:.2e}")


def probe_headline_cancellation_pm():
    kb = k_bins(BOX_PM, ns=(1, 2, 3, 4))
    nb = len(kb)
    nseed = 3
    print(f"\n=== H1 cancellation headline (native, PM-evolved, {nseed}-seed J) ===")
    J = np.zeros((3 * nb, 6))
    for s in range(nseed):
        J += FI.pm_multitracer_jacobian(
            BOX_PM,
            COSMO,
            TIME_PM,
            THETA_PM,
            kb,
            seed=s,
            backend=BACKEND,
            integrator=INTEG,
        )[0]
    J /= nseed
    cov = FI.multitracer_gaussian_covariance(
        BOX_PM, COSMO, THETA_PM, kb, N_A, N_B, n_mock=800, backend=BACKEND
    )
    tie = FI.universality_tie_matrix(THETA_PM, BOX_PM, COSMO, backend=BACKEND)
    fc2 = FI.multitracer_forecast(
        J, cov, THETA_PM, FI.PARAM_NAMES_MT, priors=PRIORS, tie=tie
    )
    A, b1 = A_FID, B1_A
    s2 = B.mesh_variance(BOX_PM, COSMO, backend=BACKEND)
    T1 = np.zeros((4, 3))
    T1[0, 0] = T1[1, 1] = T1[2, 2] = 1.0
    T1[3, 1] = -DELTA_C * (b1 - 1) / (A**2 * s2)
    T1[3, 2] = DELTA_C / (A * s2)
    fc1 = _one_tracer(
        J,
        cov,
        THETA_PM,
        nb,
        ["f_NL", "A", "b1_A", "b2_A"],
        ["f_NL", "A", "b1_A"],
        tie=T1,
    )
    print(
        f"  tied sigma(f_NL): 1-tracer {fc1.sigma('f_NL'):.4g}  2-tracer {fc2.sigma('f_NL'):.4g}  "
        f"--> gain {fc1.sigma('f_NL')/fc2.sigma('f_NL'):.2f}x"
    )


# --- explicit-b_phi tracer (the clean degeneracy) ------------------------------


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


def probe_bphi_jacobian():
    kb = k_bins(BOX_DEG, ns=(1, 2, 3, 4, 6, 8, 12, 16))
    print("\n=== V7 explicit-b_phi linear Jacobian: mx.grad vs matched-phase FD ===")
    worst = {p: 0.0 for p in FI.PARAM_NAMES_MT_BPHI}
    for s in range(4):
        J, _ = FI.linear_multitracer_bphi_jacobian(
            BOX_DEG, COSMO, THETA_BPHI, kb, seed=2000 + s, backend=BACKEND
        )
        for i, p in enumerate(FI.PARAM_NAMES_MT_BPHI):
            hi = dict(THETA_BPHI, **{p: THETA_BPHI[p] + H_BPHI[p]})
            lo = dict(THETA_BPHI, **{p: THETA_BPHI[p] - H_BPHI[p]})
            fd = (_bphi_vec(hi, 2000 + s, kb) - _bphi_vec(lo, 2000 + s, kb)) / (
                2 * H_BPHI[p]
            )
            worst[p] = max(worst[p], _col_err(J[:, i], fd))
    for p in FI.PARAM_NAMES_MT_BPHI:
        print(f"  {p:<7s}  {worst[p]:.2e}")


def _sigma_product(fc, p_a, p_b):
    """sigma of the derived product p_a*p_b via the Fisher pseudo-inverse (the
    product is the constrained direction even when f_NL alone is degenerate)."""
    free = fc._free_names
    g = np.zeros(len(free))
    g[free.index(p_a)] = fc.fiducial[p_b]
    g[free.index(p_b)] = fc.fiducial[p_a]
    return float(np.sqrt(g @ np.linalg.pinv(fc.fisher_matrix) @ g))


def _safe_sigma(fc, p):
    try:
        return fc.sigma(p)
    except Exception:
        return float("inf")


def probe_headline_degeneracy():
    kb = k_bins(BOX_DEG, ns=(1, 2, 3, 4, 6, 8, 12, 16))
    nb = len(kb)
    invM = FI._inv_M_grid(BOX_DEG, COSMO, backend=BACKEND)
    db_b1 = THETA_BPHI["bphi_A"] * FNL_DEG * invM[invM > 0].max() / THETA_BPHI["b1_A"]
    prod_fid = FNL_DEG * THETA_BPHI["bphi_A"]
    print(
        f"\n=== H2 b_phi-f_NL degeneracy (explicit-b_phi, L={BOX_DEG.box_size:.0f}, f_NL={FNL_DEG:.0f}) ==="
    )
    print(f"  scale-dependent bias at fiducial: Delta_b/b1 (k_min) ~ {db_b1:.2f}")
    J = np.zeros((3 * nb, 6))
    for s in range(8):
        J += FI.linear_multitracer_bphi_jacobian(
            BOX_DEG, COSMO, THETA_BPHI, kb, seed=s, backend=BACKEND
        )[0]
    J /= 8
    cov = FI.multitracer_bphi_gaussian_covariance(
        BOX_DEG, COSMO, THETA_BPHI, kb, N_A, N_B, n_mock=800, backend=BACKEND
    )
    tie = FI.universality_bphi_tie_matrix(THETA_BPHI, delta_c=DELTA_C)
    fc2_free = FI.multitracer_forecast(
        J, cov, THETA_BPHI, FI.PARAM_NAMES_MT_BPHI, priors=PRIORS
    )
    fc2_tied = FI.multitracer_forecast(
        J, cov, THETA_BPHI, FI.PARAM_NAMES_MT_BPHI, priors=PRIORS, tie=tie
    )
    T1 = np.zeros((4, 3))
    T1[0, 0] = T1[1, 1] = T1[2, 2] = 1.0
    T1[3, 2] = 2 * DELTA_C
    n4 = ["f_NL", "A", "b1_A", "bphi_A"]
    n3 = ["f_NL", "A", "b1_A"]
    fc1_free = _one_tracer(J, cov, THETA_BPHI, nb, n4)
    fc1_tied = _one_tracer(J, cov, THETA_BPHI, nb, n4, n3, tie=T1)
    print(
        "  FREE (b_phi marginalized): f_NL ALONE is degenerate; only the product f_NL*b_phi is constrained"
    )
    print(
        f"    cond(F) (huge = degenerate): 1-tracer {fc1_free.condition_number():.1e}  2-tracer {fc2_free.condition_number():.1e}"
    )
    print(
        f"    sigma(f_NL) free (degenerate): 1-tracer {_safe_sigma(fc1_free, 'f_NL'):.2g}  2-tracer {_safe_sigma(fc2_free, 'f_NL'):.2g}"
    )
    print(
        f"    sigma(f_NL*b_phi_A) (the constrained product; fiducial {prod_fid:.0f}): "
        f"1-tracer {_sigma_product(fc1_free, 'f_NL', 'bphi_A'):.0f}  2-tracer {_sigma_product(fc2_free, 'f_NL', 'bphi_A'):.0f}"
    )
    print("  TIED (universality b_phi=2 delta_c(b1-1)): f_NL recovered")
    print(
        f"    sigma(f_NL): 1-tracer {fc1_tied.sigma('f_NL'):.4g}  2-tracer {fc2_tied.sigma('f_NL'):.4g}  "
        f"--> multi-tracer gain {fc1_tied.sigma('f_NL')/fc2_tied.sigma('f_NL'):.2f}x"
    )


def main():
    probe_linear_jacobian()
    probe_covariance()
    probe_shot()
    probe_cancellation()
    probe_headline_cancellation()
    probe_pm()
    probe_headline_cancellation_pm()
    probe_bphi_jacobian()
    probe_headline_degeneracy()


if __name__ == "__main__":
    main()
