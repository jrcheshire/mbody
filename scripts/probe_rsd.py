"""Probe the redshift-space distortion pieces, measuring every tolerance before
any test bakes one in (the standing measure-don't-guess discipline).

Run: pixi run rsd   (or: pixi run python scripts/probe_rsd.py)

Checks, in order of how foundational they are:
  1. Momentum -> shift conversion: the redshift-space displacement of a
     low-amplitude Zeldovich field is exactly f * (real-space displacement), so
     Delta_s = p_los/(a^2 E) reproduces the linear growth rate f (the Kaiser beta
     for matter). Pinned directly on the displacement arrays (no estimator).
  2. Multipole estimator: a field with an imposed Kaiser factor (b1 + f mu^2)
     recovers the closed-form continuum multipole ratios P2/P0, P4/P0 after the
     discrete-shell mode-coupling decoupling. P0/P2 are tight; P4 is reported but
     noise-dominated at toy box sizes.
  3. Interlacing: plain CIC power turns up toward Nyquist (aliasing); the
     interlaced estimator stays flat -- the leading aliasing image is cancelled.

The Jacobian / adjoint-with-p-seed / covariance checks live with the Fisher
pieces (probe_fisher.py covers the isotropic ones; the redshift-space additions
are exercised in tests/test_rsd.py).
"""

import numpy as np
import mlx.core as mx

from mbody import bias as B, config, fields as F, fisher as FI
from mbody import integrate as IN, lpt as L, painting as PA, rsd
from mbody import cosmology as C


def _kaiser_coeffs(beta):
    """Continuum linear-Kaiser multipole ratios for bias 1 (Hamilton 1992)."""
    c0 = 1.0 + 2.0 / 3.0 * beta + 1.0 / 5.0 * beta**2
    c2 = 4.0 / 3.0 * beta + 4.0 / 7.0 * beta**2
    c4 = 8.0 / 35.0 * beta**2
    return c2 / c0, c4 / c0


def probe_conversion(box, cosmo, seed=1, zt=0.05, amplitude=0.03, los=0):
    """Delta_s = p_los/(a^2 E) reproduces f on a Zeldovich field (direct)."""
    time = config.TimeStepping(z_init=zt, z_final=0.0)
    a = 1.0 / (1.0 + zt)
    E = float(C.E(zt, cosmo))
    D = C.growth_factor(zt, cosmo)
    f0 = C.growth_rate(zt, cosmo)

    x, p = IN.initial_state(
        box, cosmo, time, seed=seed, backend="eh98", lpt_order=1, amplitude=amplitude
    )
    psi_los = np.asarray(
        L.zeldovich_displacement(
            box, cosmo, seed=seed, backend="eh98", amplitude=amplitude
        )[los]
    ).reshape(-1)

    real_disp = D * psi_los  # x - q along los
    extra = (1.0 / (a**2 * E)) * np.asarray(p[:, los])  # the RSD shift
    m = np.abs(real_disp) > 1e-4
    ratio = float(np.mean(extra[m] / real_disp[m]))
    print(
        f"[conversion] mean (s-x)/(x-q) along los = {ratio:.6f}  (f = {f0:.6f})"
        f"  rel err {abs(ratio / f0 - 1):.2e}"
    )
    return ratio, f0


def probe_estimator(box, cosmo, betas=(0.0, 0.5, 0.8), n_seed=24, los=0):
    """Imposed-Kaiser field recovers the continuum multipole ratios."""
    kmax = 0.45 * box.k_nyquist
    print(
        f"[estimator] continuum Kaiser ratios recovered (n_seed={n_seed}, los={los}):"
    )
    out = {}
    for beta in betas:
        acc = None
        for seed in range(n_seed):
            delta = F.gaussian_random_field(
                box, cosmo, seed=seed, z=0.0, backend="eh98"
            )
            ds = rsd.apply_linear_kaiser(delta, box, 1.0, beta, los_axis=los)
            kcen, P, _ = F.power_multipoles(
                ds, box, ells=(0, 2, 4), los_axis=los, kmax=kmax, decouple=True
            )
            acc = {el: P[el] if acc is None else acc[el] + P[el] for el in P}
        P = {el: acc[el] / n_seed for el in acc}
        lo = slice(0, len(kcen) // 2)  # low-k where the field is linear
        r2 = float(P[2][lo].sum() / P[0][lo].sum())
        r4 = float(P[4][lo].sum() / P[0][lo].sum())
        c2, c4 = _kaiser_coeffs(beta)
        print(
            f"  beta={beta:.2f}: P2/P0 {r2:+.4f} (Kaiser {c2:+.4f}) | "
            f"P4/P0 {r4:+.4f} (Kaiser {c4:+.4f}, noise-limited)"
        )
        out[beta] = (r2, c2, r4, c4)
    return out


def probe_interlacing(box, cosmo, seed=3, los=0):
    """Interlacing cancels the CIC aliasing upturn near Nyquist."""
    time = config.TimeStepping(z_init=0.05, z_final=0.0)
    x, _ = IN.initial_state(
        box, cosmo, time, seed=seed, backend="eh98", lpt_order=2, amplitude=1.0
    )
    k, Pp, _ = F.power_spectrum(PA.density_contrast(x, box), box, deconvolve_cic=True)
    _, Pi, _ = F.power_spectrum(
        F.interlaced_density_contrast(x, box), box, deconvolve_cic=True
    )
    Plin = C.linear_power(k, cosmo, z=0.05, backend="eh98")
    knyq = box.k_nyquist
    hi = k > 0.85 * knyq
    plain_upturn = float(np.max(Pp[hi] / Plin[hi]))
    inter_max = float(np.max(Pi[hi] / Plin[hi]))
    print(
        f"[interlacing] near Nyquist (k>0.85 k_nyq): plain P/Plin up to "
        f"{plain_upturn:.3f}, interlaced {inter_max:.3f} (aliasing removed)"
    )
    return plain_upturn, inter_max


def probe_jacobian(box, cosmo, seed=7, los=0):
    """Linear multipole Jacobian dP_ell/dtheta vs matched-phase finite difference."""
    kf = box.k_fundamental
    kbins = np.arange(1.5 * kf, 0.35 * box.k_nyquist, kf)
    theta = {"f_NL": 0.0, "b1": 2.0, "b2": 1.0, "A": 1.0, "f_growth": 1.0}
    ells = (0, 2)
    J, _ = FI.linear_multipole_jacobian(
        box, cosmo, theta, kbins, ells=ells, los_axis=los, seed=seed, backend="eh98"
    )
    f_lin = C.growth_rate(0.0, cosmo)

    def model(thd):
        from mbody import ic as IC

        d = thd["A"] * IC.linear_density(
            box, cosmo, seed=seed, f_NL=thd["f_NL"], backend="eh98"
        )
        tr = FI._redshift_tracer_linear(
            d, box, thd["b1"], thd["b2"], thd["f_growth"] * f_lin, los
        )
        return np.concatenate(
            [
                np.asarray(F.band_power_multipole(tr, box, kbins, el, los_axis=los))
                for el in ells
            ]
        )

    print("[jacobian] linear multipole dP_ell/dtheta vs finite difference:")
    for name, h in (("f_NL", 1.0), ("b1", 0.02), ("f_growth", 0.02)):
        tp, tm = dict(theta), dict(theta)
        tp[name] += h
        tm[name] -= h
        fd = (model(tp) - model(tm)) / (2 * h)
        ci = FI.PARAM_NAMES_RSD.index(name)
        rel = np.abs(J[:, ci] - fd) / (np.abs(fd).max() + 1e-30)
        print(f"  d/d{name:8s}: max rel err {rel.max():.2e}")


def probe_adjoint(box, cosmo, time, seed=2, los=0):
    """Redshift-space adjoint (momentum-seeded) f_NL/A gradient vs replay mx.grad."""
    a_steps = IN.a_grid(time, "linear")
    kf = box.k_fundamental
    kbins = np.arange(1.5 * kf, 6.5 * kf, kf)
    b1, b2, ell, bin_i = 2.0, 1.0, 2, 2

    def redshift_bp(x, p):
        s = rsd.redshift_space_positions(x, p, box, cosmo, z=time.z_final, los_axis=los)
        tr = B.local_bias_tracer(PA.density_contrast(s, box), b1, b2)
        return F.band_power_multipole(tr, box, kbins, ell, los_axis=los)[bin_i]

    def loss_theta(th):
        x0, p0 = IN.initial_state(
            box,
            cosmo,
            time,
            seed=seed,
            f_NL=th[0],
            backend="eh98",
            lpt_order=2,
            amplitude=th[1],
        )
        x, p = IN.evolve_state(x0, p0, box, cosmo, a_steps, integrator="fastpm")
        return redshift_bp(x, p)

    g_replay = np.asarray(mx.grad(loss_theta)(mx.array([5.0, 1.0])))
    g_adj = np.asarray(
        IN.adjoint_grad_ic(
            redshift_bp,
            box,
            cosmo,
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
    print(
        f"[adjoint] redshift-space (f_NL, A) grad: adjoint vs replay rel err "
        f"{rel[0]:.2e}, {rel[1]:.2e}"
    )


def probe_fisher(box, cosmo, seed=7, los=0):
    """Headline: adding the quadrupole tightens sigma(f_NL) and sigma(f_growth)."""
    kf = box.k_fundamental
    kbins = np.arange(1.5 * kf, 0.35 * box.k_nyquist, kf)
    theta = {"f_NL": 0.0, "b1": 2.0, "b2": 1.0, "A": 1.0, "f_growth": 1.0}
    priors = {"A": 0.1}
    print("[fisher] linear-field redshift-space forecast (prior sigma(A)=0.1):")
    for tag, ells in (("monopole only", (0,)), ("mono+quad ", (0, 2))):
        J, _ = FI.linear_multipole_jacobian(
            box, cosmo, theta, kbins, ells=ells, los_axis=los, seed=seed, backend="eh98"
        )
        cov = FI.multipole_gaussian_covariance(
            box,
            cosmo,
            theta,
            kbins,
            ells=ells,
            los_axis=los,
            n_mock=400,
            backend="eh98",
        )
        fc = FI.FisherForecast(
            J,
            covariance=cov,
            fiducial_params=theta,
            param_names=FI.PARAM_NAMES_RSD,
            priors=priors,
        )
        print(
            f"  {tag}: sigma(f_NL)={fc.sigma('f_NL'):8.1f}  "
            f"sigma(f_growth)={fc.sigma('f_growth'):.3f}"
        )


def main():
    box = config.BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
    cosmo = config.Cosmology()
    print("M-body redshift-space distortion probe")
    print(f"box L={box.box_size} Mpc/h, N={box.n_mesh}\n")
    probe_conversion(box, cosmo)
    print()
    probe_estimator(box, cosmo)
    print()
    probe_interlacing(
        config.BoxConfig(box_size=256.0, n_mesh=64, n_particles=64), cosmo
    )
    print()
    fbox = config.BoxConfig(box_size=512.0, n_mesh=48, n_particles=48)
    probe_jacobian(fbox, cosmo)
    print()
    probe_adjoint(
        config.BoxConfig(box_size=256.0, n_mesh=24, n_particles=24),
        cosmo,
        config.TimeStepping(z_init=9.0, z_final=0.0, n_steps=4),
    )
    print()
    probe_fisher(fbox, cosmo)


if __name__ == "__main__":
    main()
