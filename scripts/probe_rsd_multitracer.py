"""Probe the redshift-space multi-tracer Fisher before fixing any test tolerance.

Composes the two subsystems already validated separately: the redshift-space
multipole Fisher and the multi-tracer Fisher. Two local-bias tracers A, B are
painted from the SAME redshift-space field and summarized by their auto/cross
multipoles -- so the quadrupole's f_growth handle stacks on top of the multi-tracer
sample-variance cancellation. The genuinely new autodiff path is the two-tracer
cross-spectrum gradient seeded on the FINAL VELOCITIES (the momentum-seeded
reversible adjoint). Native tracer only (b_phi emergent from b2); the scientific
conclusion is unchanged from the isotropic case -- this is the composition piece.

This probe MEASURES every number tests/test_multitracer.py (and test_rsd.py for
V10) later asserts (the standing "measure first, never relax a tolerance" rule):

  V10 cross_power_multipole reductions: (d,d,0)==cross_power;
      (d,d,ell)==band_power_multipole
  V11 linear RSD-MT Jacobian columns vs matched-phase central FD
  V12 PM RSD-MT: adjoint_grad_ic (f_NL, A; momentum-seeded) vs replay mx.grad for
      AA/AB/BB; downstream (f_growth, b's) vs fixed-state FD
  V13 shot noise lands in the MONOPOLE only (the quadrupole auto-cov is unchanged);
      cond(cov)
  H3  headline: tied sigma(f_NL), sigma(f_growth) in the 2x2 grid
      {monopole only, mono+quad} x {1-tracer, 2-tracer}

Run: pixi run probe-rsd-multitracer
Nothing is written; the printed numbers set tolerances in tests/test_multitracer.py.
"""

import numpy as np
import mlx.core as mx

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
INTEG = "fastpm"
DELTA_C = 1.686
LOS = 0
ELLS = (0, 2)

# Linear-field box for the Jacobian / covariance / shot checks.
BOX = BoxConfig(box_size=512.0, n_mesh=48, n_particles=48)
# Big cheap linear box for the sensitivity headline (box size is free in this toy).
BOX_HEAD = BoxConfig(box_size=1024.0, n_mesh=48, n_particles=48)
# Small PM box for the adjoint-vs-replay / downstream checks.
BOX_PM = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
TIME_PM = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5)

# Two tracers with different linear bias (hence different b_phi); shared growth.
B1_A, B1_B, A_FID, FG_FID = 1.5, 2.5, 1.0, 1.0
N_A = N_B = 5.0e-4
PRIORS = {"A": 0.1}

# Native fiducial at f_NL=0 (the detection regime), b2_i at the universality value.
THETA = FI.universality_rsd_fiducial(
    0.0, A_FID, FG_FID, B1_A, B1_B, BOX, COSMO, backend=BACKEND
)
THETA_HEAD = FI.universality_rsd_fiducial(
    0.0, A_FID, FG_FID, B1_A, B1_B, BOX_HEAD, COSMO, backend=BACKEND
)
THETA_PM = FI.universality_rsd_fiducial(
    0.0, A_FID, FG_FID, B1_A, B1_B, BOX_PM, COSMO, backend=BACKEND
)

# Matched-phase FD steps (f_NL has a tiny per-unit response -> big step).
H = {
    "f_NL": 50.0,
    "A": 1e-2,
    "f_growth": 1e-2,
    "b1_A": 1e-2,
    "b2_A": 1e-2,
    "b1_B": 1e-2,
    "b2_B": 1e-2,
}


def _kb(box, ns):
    return np.array([n * box.k_fundamental for n in ns])


def _col_err(jcol, fd):
    denom = max(float(np.abs(fd).max()), 1e-300)
    return float(np.abs(np.asarray(jcol) - fd).max() / denom)


# --- V10: cross_power_multipole reductions -------------------------------------


def probe_reductions():
    kb = _kb(BOX, (2, 3, 4, 6))
    d = IC.linear_density(BOX, COSMO, seed=0, f_NL=0.0, backend=BACKEND)
    e = IC.linear_density(BOX, COSMO, seed=1, f_NL=0.0, backend=BACKEND)
    print("=== V10 cross_power_multipole reductions ===")
    c0 = np.asarray(F.cross_power_multipole(d, e, BOX, kb, 0, los_axis=LOS), np.float64)
    xp = np.asarray(F.cross_power(d, e, BOX, kb), np.float64)
    print(
        f"  (a,b,ell=0) vs cross_power:          max rel {np.abs(c0 / xp - 1).max():.2e}"
    )
    for el in (0, 2, 4):
        auto = np.asarray(
            F.cross_power_multipole(d, d, BOX, kb, el, los_axis=LOS), np.float64
        )
        bpm = np.asarray(
            F.band_power_multipole(d, BOX, kb, el, los_axis=LOS), np.float64
        )
        rel = np.abs(auto / bpm - 1.0).max()
        print(f"  (a,a,ell={el}) vs band_power_multipole: max rel {rel:.2e}")


# --- V11: linear Jacobian vs matched-phase FD ----------------------------------


def _rsd_mt_vec(theta, box, seed, kb):
    f_NL = mx.array(theta["f_NL"])
    delta = theta["A"] * IC.linear_density(
        box, COSMO, seed=seed, f_NL=f_NL, backend=BACKEND
    )
    feff = theta["f_growth"] * C.growth_rate(0.0, COSMO)
    hA = FI._redshift_tracer_linear(delta, box, theta["b1_A"], theta["b2_A"], feff, LOS)
    hB = FI._redshift_tracer_linear(delta, box, theta["b1_B"], theta["b2_B"], feff, LOS)
    return np.asarray(
        FI._mt_multipole_vector(hA, hB, box, kb, ELLS, LOS, None), np.float64
    )


def _fd_col(name, box, seed, kb, theta0):
    hi = dict(theta0, **{name: theta0[name] + H[name]})
    lo = dict(theta0, **{name: theta0[name] - H[name]})
    return (_rsd_mt_vec(hi, box, seed, kb) - _rsd_mt_vec(lo, box, seed, kb)) / (
        2 * H[name]
    )


def probe_linear_jacobian():
    kb = _kb(BOX, (1, 2, 3, 4, 6))
    print("\n=== V11 linear RSD-MT Jacobian: mx.grad vs matched-phase FD (4 seeds) ===")
    print("  max |J - FD| / max|FD| per column:")
    worst = {p: 0.0 for p in FI.PARAM_NAMES_MT_RSD}
    for s in range(4):
        J, _ = FI.linear_multitracer_multipole_jacobian(
            BOX,
            COSMO,
            THETA,
            kb,
            ells=ELLS,
            los_axis=LOS,
            seed=2000 + s,
            backend=BACKEND,
        )
        for i, p in enumerate(FI.PARAM_NAMES_MT_RSD):
            worst[p] = max(
                worst[p], _col_err(J[:, i], _fd_col(p, BOX, 2000 + s, kb, THETA))
            )
    for p in FI.PARAM_NAMES_MT_RSD:
        print(f"  {p:<8s}  {worst[p]:.2e}")


# --- V12: PM adjoint-vs-replay and downstream FD -------------------------------


def probe_pm():
    kb = _kb(BOX_PM, (1, 2, 3, 4))
    nb = len(kb)
    n_ell = len(ELLS)
    print(f"\n=== V12 PM RSD-MT (N={BOX_PM.n_mesh}, {TIME_PM.n_steps} steps) ===")

    # Evolve once; downstream (f_growth, b's) at the fixed final (x, p).
    xf, pf = IN.leapfrog(
        BOX_PM,
        COSMO,
        TIME_PM,
        seed=0,
        f_NL=mx.array(THETA_PM["f_NL"]),
        amplitude=mx.array(THETA_PM["A"]),
        backend=BACKEND,
        integrator=INTEG,
        lpt_order=2,
    )
    xf, pf = mx.stop_gradient(xf), mx.stop_gradient(pf)

    def vec_down(t):
        s = RS.redshift_space_positions(
            xf, pf, BOX_PM, COSMO, z=TIME_PM.z_final, los_axis=LOS, f_growth=t[0]
        )
        field = F.interlaced_density_contrast(s, BOX_PM)
        hA = B.local_bias_tracer(field, t[1], t[2])
        hB = B.local_bias_tracer(field, t[3], t[4])
        return FI._mt_multipole_vector(hA, hB, BOX_PM, kb, ELLS, LOS, None)

    t0 = mx.array(
        [
            THETA_PM["f_growth"],
            THETA_PM["b1_A"],
            THETA_PM["b2_A"],
            THETA_PM["b1_B"],
            THETA_PM["b2_B"],
        ]
    )
    n_data = 3 * n_ell * nb
    Jd = np.array(
        [
            np.asarray(mx.grad(lambda t, d=d: vec_down(t)[d])(t0), np.float64)
            for d in range(n_data)
        ]
    )
    names_d = ["f_growth", "b1_A", "b2_A", "b1_B", "b2_B"]
    print("  downstream mx.grad vs fixed-state FD (worst over data vector):")
    eye = np.eye(5)
    for j, nm in enumerate(names_d):
        hi = mx.array(np.asarray(t0) + H[nm] * eye[j])
        lo = mx.array(np.asarray(t0) - H[nm] * eye[j])
        fd = (np.asarray(vec_down(hi)) - np.asarray(vec_down(lo))) / (2 * H[nm])
        print(f"    {nm:<9s}  {_col_err(Jd[:, j], fd):.2e}")

    # IC columns f_NL, A: momentum-seeded adjoint vs replay mx.grad.
    print("  adjoint_grad_ic vs replay (f_NL, A), pivot bins of AA/AB/BB at ell=2:")

    def loss_at(x, p, d):
        s = RS.redshift_space_positions(
            x,
            p,
            BOX_PM,
            COSMO,
            z=TIME_PM.z_final,
            los_axis=LOS,
            f_growth=THETA_PM["f_growth"],
        )
        field = F.interlaced_density_contrast(s, BOX_PM)
        hA = B.local_bias_tracer(field, THETA_PM["b1_A"], THETA_PM["b2_A"])
        hB = B.local_bias_tracer(field, THETA_PM["b1_B"], THETA_PM["b2_B"])
        return FI._mt_multipole_vector(hA, hB, BOX_PM, kb, ELLS, LOS, None)[d]

    def replay(tv, d):
        x, p = IN.leapfrog(
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
        return loss_at(x, p, d)

    tv0 = mx.array([THETA_PM["f_NL"], THETA_PM["A"]])
    # quadrupole pivot bin (index nb) of each spectrum: AA-ell2, AB-ell2, BB-ell2.
    for lab, d in [("AA", nb), ("AB", 3 * nb), ("BB", 5 * nb)]:
        g_adj = np.asarray(
            IN.adjoint_grad_ic(
                lambda x, p, d=d: loss_at(x, p, d),
                BOX_PM,
                COSMO,
                TIME_PM,
                seed=0,
                f_NL=THETA_PM["f_NL"],
                amplitude=THETA_PM["A"],
                backend=BACKEND,
                integrator=INTEG,
                lpt_order=2,
                loss_uses_momentum=True,
            )
        )
        g_rep = np.asarray(mx.grad(lambda tv, d=d: replay(tv, d))(tv0))
        rel = np.abs(g_adj / g_rep - 1.0)
        print(f"    {lab}-ell2: f_NL {rel[0]:.2e}  A {rel[1]:.2e}")


# --- V13: shot noise lands in the monopole only --------------------------------


def probe_shot():
    kb = _kb(BOX, (1, 2, 3, 4, 6))
    N, V = BOX.n_mesh, BOX.box_size**3
    sig = float(np.sqrt(N**3 / (N_A * V)))
    print(
        "\n=== V13 shot noise: a white shot field is pure-monopole, band power 1/n ==="
    )
    # The Poisson shot field is white (isotropic): its SIGNAL is the monopole 1/n,
    # the quadrupole is ~0 (only the raw discrete-shell leakage). So shot enters the
    # data-vector MEAN in the monopole auto power only. (It still raises the COVARIANCE
    # of all multipoles, like any Gaussian term -- that is the conditioning gain below.)
    P0s, P2s = [], []
    for s in range(60):
        w = sig * mx.random.normal((N, N, N), key=mx.random.key(9000 + s))
        P0s.append(
            np.asarray(F.band_power_multipole(w, BOX, kb, 0, los_axis=LOS), np.float64)
        )
        P2s.append(
            np.asarray(F.band_power_multipole(w, BOX, kb, 2, los_axis=LOS), np.float64)
        )
    P0 = np.mean(P0s, axis=0) * N_A
    P2 = np.mean(P2s, axis=0) * N_A
    print(f"  monopole   P0 * n (target 1): {np.array2string(P0, precision=3)}")
    print(
        f"  quadrupole P2 * n (target ~0, raw-shell leakage only): {np.array2string(P2, precision=3)}"
    )
    cov_on = FI.multitracer_multipole_gaussian_covariance(
        BOX,
        COSMO,
        THETA,
        kb,
        N_A,
        N_B,
        ells=ELLS,
        los_axis=LOS,
        n_mock=800,
        backend=BACKEND,
        shot=True,
    )
    cov_off = FI.multitracer_multipole_gaussian_covariance(
        BOX,
        COSMO,
        THETA,
        kb,
        N_A,
        N_B,
        ells=ELLS,
        los_axis=LOS,
        n_mock=800,
        backend=BACKEND,
        shot=False,
    )
    print(
        f"  cond(cov): shot regularizes {np.linalg.cond(cov_off):.2e} -> {np.linalg.cond(cov_on):.2e}"
    )


# --- H3: the 2x2 headline ------------------------------------------------------


def _tie1_native(box, theta):
    """1-tracer (AA) native universality tie T1: free (f_NL, A, f_growth, b1_A, b2_A)
    -> tied (f_NL, A, f_growth, b1_A)."""
    s2 = B.mesh_variance(box, COSMO, backend=BACKEND)
    A, b1 = theta["A"], theta["b1_A"]
    T1 = np.zeros((5, 4))
    T1[0, 0] = T1[1, 1] = T1[2, 2] = T1[3, 3] = 1.0
    T1[4, 1] = -DELTA_C * (b1 - 1) / (A**2 * s2)
    T1[4, 3] = DELTA_C / (A * s2)
    return T1


def _headline(box, theta, ells, nseed=8):
    kb = np.arange(1.5, 12.0, 1.0) * box.k_fundamental
    n_ell, nb = len(ells), len(kb)
    J = np.zeros((3 * n_ell * nb, 7))
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
        N_A,
        N_B,
        ells=ells,
        los_axis=LOS,
        n_mock=600,
        backend=BACKEND,
    )
    tie = FI.universality_rsd_tie_matrix(theta, box, COSMO, backend=BACKEND)
    fc2 = FI.multitracer_forecast(
        J, cov, theta, FI.PARAM_NAMES_MT_RSD, priors=PRIORS, tie=tie
    )

    # 1-tracer (AA only): the first n_ell*nb rows, columns [f_NL,A,f_growth,b1_A,b2_A].
    naa = n_ell * nb
    JA = J[:naa][:, [0, 1, 2, 3, 4]]
    covA = cov[:naa, :naa]
    T1 = _tie1_native(box, theta)
    fc1 = FI.FisherForecast(
        JA @ T1,
        covariance=covA,
        fiducial_params={n: theta[n] for n in ("f_NL", "A", "f_growth", "b1_A")},
        param_names=("f_NL", "A", "f_growth", "b1_A"),
        priors=PRIORS,
    )
    return fc1, fc2


def probe_headline():
    print(f"\n=== H3 headline (native, linear, f_NL=0, L={BOX_HEAD.box_size:.0f}) ===")
    print("  tied sigma(f_NL) / sigma(f_growth), 1-tracer vs 2-tracer:")
    for tag, ells in (("monopole only", (0,)), ("mono + quad ", (0, 2))):
        fc1, fc2 = _headline(BOX_HEAD, THETA_HEAD, ells)
        g = fc1.sigma("f_NL") / fc2.sigma("f_NL")
        print(
            f"  {tag}: sigma(f_NL) 1t {fc1.sigma('f_NL'):8.1f}  2t {fc2.sigma('f_NL'):8.1f} "
            f"(cancellation {g:.2f}x) | sigma(f_growth) 1t {fc1.sigma('f_growth'):.3f} "
            f"2t {fc2.sigma('f_growth'):.3f}"
        )


def main():
    probe_reductions()
    probe_linear_jacobian()
    probe_pm()
    probe_shot()
    probe_headline()


if __name__ == "__main__":
    main()
