"""Probe the autodiff Fisher pieces before fixing any test tolerances.

Measures, on the linear f_NL field (milestone 1):

  * each Jacobian column d ln P_b / d theta (theta in {f_NL, b1, b2, A}) from
    reverse-mode mx.grad vs a matched-phase central finite difference -- the
    autodiff-vs-FD agreement that tests/test_fisher.py asserts;
  * the empirical band-power variance Var[ln P_b] over a seed ensemble of
    Gaussian fields vs the mode-count prediction 2 / N_modes,b = 1/count_b --
    the check that pins the Hermitian factor in independent_mode_count;
  * the condition number of the {f_NL, b1, b2, A} Fisher (the A-b1 amplitude
    degeneracy).

Run: pixi run python scripts/probe_fisher.py
Nothing is written; the printed numbers set tolerances in tests/test_fisher.py.
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology, TimeStepping, Tracer
from mbody import fields as F
from mbody import ic as IC
from mbody import bias as B
from mbody import painting as PA
from mbody import integrate as IN
from mbody import fisher as FI

COSMO = Cosmology()
BOX = BoxConfig(box_size=256.0, n_mesh=64, n_particles=64)
BOX_PM = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
TIME_PM = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5)
TRACER = Tracer()  # b1=2, b2=1, A=1
THETA_FID = {"f_NL": 0.0, "b1": TRACER.b1, "b2": TRACER.b2, "A": TRACER.A}

# Matched-phase finite-difference steps, per parameter. f_NL needs a large step
# (its per-unit response is tiny next to the CIC floor); the others are O(1).
H = {"f_NL": 50.0, "b1": 1e-2, "b2": 1e-2, "A": 1e-2}


def k_bins(box):
    kf = box.k_fundamental
    return np.array([n * kf for n in (1, 2, 3, 4, 6, 8, 12)])


def _logP(theta, seed, kb):
    """ln P_b of the linear-field tracer at parameter dict `theta`, one seed."""
    f_NL = mx.array(theta["f_NL"])
    delta = theta["A"] * IC.linear_density(BOX, COSMO, seed=seed, f_NL=f_NL)
    tracer = B.local_bias_tracer(delta, theta["b1"], theta["b2"])
    return np.asarray(mx.log(F.band_power(tracer, BOX, kb)), np.float64)


def fd_column(name, seed, kb):
    """Matched-phase central FD of ln P_b w.r.t. parameter `name`."""
    hi = dict(THETA_FID, **{name: THETA_FID[name] + H[name]})
    lo = dict(THETA_FID, **{name: THETA_FID[name] - H[name]})
    return (_logP(hi, seed, kb) - _logP(lo, seed, kb)) / (2.0 * H[name])


def probe_jacobian():
    kb = k_bins(BOX)
    print("=== Jacobian columns: mx.grad vs matched-phase FD (4 seeds) ===")
    print("  worst per-bin relative error per column:")
    rel = {p: [] for p in FI.PARAM_NAMES}
    for s in range(4):
        J, _ = FI.linear_logP_jacobian(BOX, COSMO, THETA_FID, kb, seed=1000 + s)
        for i, p in enumerate(FI.PARAM_NAMES):
            fd = fd_column(p, 1000 + s, kb)
            rel[p].append(np.abs(J[:, i] / fd - 1.0))
    for p in FI.PARAM_NAMES:
        worst = np.array(rel[p]).max()
        print(f"  {p:<6s}  max rel.err = {worst:.2e}")


def _white_noise(box, seed):
    """A flat-power (white-noise) real field: P(k) = const, so the band-power
    variance is set purely by the mode count, with no within-shell P(k) spread."""
    N = box.n_mesh
    return mx.random.normal((N, N, N), key=mx.random.key(seed))


def _plane_count(box, kb):
    """n_plane,b: half-grid shell cells in the kz in {0, N/2} planes (the cells
    band_power double-counts because each pairs with its in-plane conjugate)."""
    N = box.n_mesh
    masks, _ = F._bin_masks(box, kb, box.k_fundamental)
    out = []
    for m in masks:
        mm = np.asarray(m, np.float64)
        n = mm[:, :, 0].sum() + (mm[:, :, N // 2].sum() if N % 2 == 0 else 0.0)
        out.append(n)
    return np.array(out)


def probe_variance():
    kb = k_bins(BOX)
    nseed = 1000
    print(f"\n=== band-power variance over {nseed} WHITE-NOISE fields ===")
    print("  (flat power isolates the mode count; a colored field's Var is")
    print("   inflated by within-shell P(k) variation, an expected effect)")
    _, counts = F._bin_masks(BOX, kb, BOX.k_fundamental)
    counts = np.array(counts)
    n_plane = _plane_count(BOX, kb)
    pred = (counts + n_plane) / counts**2  # exact Gaussian relVar of band_power
    P = np.array(
        [
            np.asarray(F.band_power(_white_noise(BOX, 3000 + s), BOX, kb), np.float64)
            for s in range(nseed)
        ]
    )
    rel_var = P.var(axis=0, ddof=1) / P.mean(axis=0) ** 2  # Var[P_b]/<P_b>^2
    lnP_var = np.log(P).var(axis=0, ddof=1)  # Var[ln P_b]
    print("  k         count_b  n_plane  relVar_emp  pred(Gauss)  ratio  Var[lnP]/pred")
    for i, k in enumerate(kb):
        print(
            f"  {k:7.4f}  {counts[i]:7.0f}  {n_plane[i]:7.0f}  {rel_var[i]:.3e}  "
            f"{pred[i]:.3e}  {rel_var[i]/pred[i]:6.3f}  {lnP_var[i]/pred[i]:6.3f}"
        )
    sampling = np.sqrt(2.0 / (nseed - 1))
    print(f"  (sampling error on each variance ratio ~ {sampling:.3f})")


def probe_fisher():
    kb = k_bins(BOX)
    nseed = 16
    J = np.zeros((len(kb), len(FI.PARAM_NAMES)))
    for s in range(nseed):
        Js, _ = FI.linear_logP_jacobian(BOX, COSMO, THETA_FID, kb, seed=s)
        J += Js
    J /= nseed
    var = FI.band_power_log_variance(BOX, kb)
    print(f"\n=== Fisher over {{f_NL, b1, b2, A}} ({nseed}-seed mean J) ===")
    fc = FI.FisherForecast(J, var, THETA_FID)
    print("  no prior:")
    print("   cond(F) = %.3e" % fc.condition_number())
    for p in FI.PARAM_NAMES:
        print(f"   sigma({p}) = {fc.sigma(p):.4g}")
    fc_p = FI.FisherForecast(J, var, THETA_FID, priors={"A": 0.1})
    print("  with sigma(A)=0.1 prior:")
    print("   cond(F) = %.3e" % fc_p.condition_number())
    for p in FI.PARAM_NAMES:
        print(f"   sigma({p}) = {fc_p.sigma(p):.4g}")
    m = fc_p.marginalized_2d("f_NL", "b1")
    print(f"   rho(f_NL, b1) = {m['rho']:+.3f}")
    m = fc_p.marginalized_2d("A", "b1")
    print(f"   rho(A, b1)    = {m['rho']:+.3f}")


def _pm_logP(theta, seed, kb):
    x, _ = IN.leapfrog(
        BOX_PM,
        COSMO,
        TIME_PM,
        seed=seed,
        f_NL=mx.array(theta["f_NL"]),
        amplitude=mx.array(theta["A"]),
        lpt_order=2,
    )
    delta = PA.density_contrast(x, BOX_PM)
    tracer = B.local_bias_tracer(delta, theta["b1"], theta["b2"])
    return np.asarray(mx.log(F.band_power(tracer, BOX_PM, kb)), np.float64)


def _pm_fd_column(name, seed, kb):
    hi = dict(THETA_FID, **{name: THETA_FID[name] + H[name]})
    lo = dict(THETA_FID, **{name: THETA_FID[name] - H[name]})
    return (_pm_logP(hi, seed, kb) - _pm_logP(lo, seed, kb)) / (2.0 * H[name])


def probe_pm():
    kb = np.array([1, 2, 3, 4]) * BOX_PM.k_fundamental
    print(f"\n=== PM Jacobian (N={BOX_PM.n_mesh}, {TIME_PM.n_steps} steps) ===")
    print("  adjoint (f_NL, A) + downstream (b1, b2) vs matched-phase FD:")
    J, _ = FI.pm_logP_jacobian(BOX_PM, COSMO, TIME_PM, THETA_FID, kb, seed=0)
    for i, p in enumerate(FI.PARAM_NAMES):
        fd = _pm_fd_column(p, 0, kb)
        rel = np.abs(J[:, i] / fd - 1.0)
        print(f"  {p:<6s}  max rel.err = {rel.max():.2e}")

    print("  adjoint_grad_ic vs replay mx.grad (f_NL, A), per bin:")

    def _loss_at(x, b):
        tracer = B.local_bias_tracer(
            PA.density_contrast(x, BOX_PM), THETA_FID["b1"], THETA_FID["b2"]
        )
        return mx.log(F.band_power(tracer, BOX_PM, kb))[b]

    def _replay_loss(tv, b):
        x, _ = IN.leapfrog(
            BOX_PM, COSMO, TIME_PM, seed=0, f_NL=tv[0], amplitude=tv[1], lpt_order=2
        )
        return _loss_at(x, b)

    tv0 = mx.array([THETA_FID["f_NL"], THETA_FID["A"]])
    for b in range(len(kb)):
        g_adj = IN.adjoint_grad_ic(
            lambda x, b=b: _loss_at(x, b),
            BOX_PM,
            COSMO,
            TIME_PM,
            seed=0,
            f_NL=THETA_FID["f_NL"],
            amplitude=THETA_FID["A"],
            lpt_order=2,
        )
        g_rep = mx.grad(lambda tv, b=b: _replay_loss(tv, b))(tv0)
        rel = np.abs(np.asarray(g_adj) / np.asarray(g_rep) - 1.0)
        print(f"   bin {b}: f_NL {rel[0]:.2e}  A {rel[1]:.2e}")

    nseed = 6
    Jm = np.zeros_like(J)
    for s in range(nseed):
        Js, _ = FI.pm_logP_jacobian(BOX_PM, COSMO, TIME_PM, THETA_FID, kb, seed=s)
        Jm += Js
    Jm /= nseed
    var = FI.band_power_log_variance(BOX_PM, kb)
    fc = FI.FisherForecast(Jm, var, THETA_FID, priors={"A": 0.1})
    print(f"  PM Fisher ({nseed}-seed mean J, sigma(A)=0.1 prior):")
    print("   cond(F) = %.3e" % fc.condition_number())
    for p in FI.PARAM_NAMES:
        print(f"   sigma({p}) = {fc.sigma(p):.4g}")


def main():
    probe_jacobian()
    probe_variance()
    probe_fisher()
    probe_pm()


if __name__ == "__main__":
    main()
