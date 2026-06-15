"""Lagrangian-bias tracer validation -- gates 1 (linear field) and 2 (through PM).

The Lagrangian tracer builds the bias on the INITIAL linear field and sets the
f_NL scale-dependent bias DIRECTLY via an explicit b_phi term (separate-universe /
Barreira), carries 1 + delta_g^L as per-particle weights, and advects them by the
PM displacement. This isolates two checks:

Gate 1 (linear field, no advection). For delta_g^L = b1 delta_G + b_phi f_NL phi_G
with phi_G = delta_G / M(k), the cross-over-auto with the matter field is, per |k|
shell, b1 + b_phi f_NL <1/M(k)>_shell. So the measured excess bias should equal
b_phi f_NL times the P-weighted shell average of the Dalal kernel 1/M(k) -- exactly
(same realization on both sides). Validates the b_phi wiring + M(k) consistency.

Gate 2 (through the PM) -- THE trust proof. Advect the weighted particles through
BullFrog and measure d ln P_g / df_NL by matched-phase finite difference (same IC
phases at +/-eps cancel cosmic variance, seed-averaged). For an explicit-b_phi
tracer the linear-theory Dalal prediction is

    d ln P_g / df_NL = 2 b_phi / (b1_E M(k)),   b1_E = 1 + b1_L,

with NO suppression. The Eulerian native-b2 tracer gives ~25% of its (emergent)
Dalal value through the PM (the documented dilution); the question here is whether
the explicit Lagrangian b_phi survives advection undiluted -> ratio ~1 at low k.
The matter field (weights = 1) is the null control (matter has no O(f_NL) power).

Run: ``pixi run python scripts/probe_lagrangian_bias.py``.
"""

import numpy as np

from mbody import bias as B
from mbody import cosmology as C
from mbody import fields as F
from mbody import ic as IC
from mbody import integrate as IG
from mbody import painting as PA
from mbody.config import BoxConfig, Cosmology, TimeStepping

BACKEND = "eh98"  # closed-form transfer; no CAMB call needed for these checks
DELTA_C = 1.686


def pweighted_inv_M_shell(box, cosmo, k_bins, dk, z, backend):
    """P_lin-weighted shell average of 1/M(k) on the rfftn half-grid, per bin.

    The cross/auto ratio (and the band-power derivative) weight each mode by
    |delta_k|^2 ~ P_lin(k), so the predicted scale-dependent bias is the P-weighted
    shell average of 1/M -- not the bin centre (1/M is steep at low k).
    """
    _, _, k_mag = F.k_grid(box)
    km = k_mag.ravel()
    P_lin = C.linear_power(km, cosmo, z=z, backend=backend)
    M = IC.poisson_M(np.where(km > 0, km, 1.0), cosmo, z=z, backend=backend)
    inv_M = np.where(km > 0, P_lin / M, 0.0)
    P_wt = np.where(km > 0, P_lin, 0.0)
    out = np.empty(len(k_bins))
    for b, kc in enumerate(k_bins):
        m = (km >= kc - 0.5 * dk) & (km < kc + 0.5 * dk)
        out[b] = inv_M[m].sum() / P_wt[m].sum()
    return out


def lagrangian_tracer_delta(box, cosmo, time, seed, f_NL, coeffs, backend=BACKEND):
    """Tracer overdensity from the advected Lagrangian-biased weights.

    Builds the linear field + Gaussian potential at the output redshift, forms the
    Lagrangian bias weights 1 + delta_g^L(q), advects the particles through the PM
    (BullFrog), and CIC-paints the WEIGHTED particles to the Eulerian tracer field.
    Forward-only (off the AD path); f_NL is a plain float (finite-difference axis).
    """
    z_out = time.z_final
    delta1 = IC.linear_density(
        box, cosmo, seed=seed, z=z_out, f_NL=f_NL, backend=backend
    )
    phi_G = IC.primordial_potential(box, cosmo, seed=seed, z=z_out, backend=backend)
    dgL = B.lagrangian_bias_field(delta1, box, f_NL=f_NL, phi_G=phi_G, **coeffs)
    w = (1.0 + dgL).reshape(-1)

    x0, p0 = IG.initial_state(
        box, cosmo, time, seed=seed, f_NL=f_NL, backend=backend, lpt_order=2
    )
    a_steps = IG.a_grid(time)
    x_final, _ = IG.evolve_state(
        x0, p0, box, cosmo, a_steps, integrator=time.integrator
    )
    n_g = PA.cic_paint(x_final, box, weights=w)
    mean_w = float(w.sum()) / (box.n_mesh**3)
    return n_g / mean_w - 1.0


def dlnP_df_matched(box, cosmo, time, seeds, eps, k_bins, dk, coeffs):
    """Matched-phase finite-difference d ln P_g / df_NL, averaged over seeds."""
    derivs = []
    for seed in seeds:
        dgp = lagrangian_tracer_delta(box, cosmo, time, seed, +eps, coeffs)
        dgm = lagrangian_tracer_delta(box, cosmo, time, seed, -eps, coeffs)
        Pp = np.asarray(F.band_power(dgp, box, k_bins, dk=dk))
        Pm = np.asarray(F.band_power(dgm, box, k_bins, dk=dk))
        derivs.append((np.log(Pp) - np.log(Pm)) / (2.0 * eps))
    return np.mean(derivs, axis=0), np.std(derivs, axis=0)


def gate1():
    box = BoxConfig(box_size=2048.0, n_mesh=64)  # large L -> low-k 1/M shape
    cosmo = Cosmology()
    z, seed = 0.0, 0
    b1L, b_phi, f_NL = 1.0, 5.0, 100.0

    delta_G = IC.linear_density(box, cosmo, seed=seed, z=z, f_NL=0.0, backend=BACKEND)
    phi_G = IC.primordial_potential(box, cosmo, seed=seed, z=z, backend=BACKEND)
    dgL = B.lagrangian_bias_field(
        delta_G, box, b1=b1L, b_phi=b_phi, f_NL=f_NL, phi_G=phi_G
    )

    dk = box.k_fundamental
    k_bins = dk * np.arange(1, 13)
    P_cross = np.asarray(F.cross_power(dgL, delta_G, box, k_bins, dk=dk))
    P_auto = np.asarray(F.band_power(delta_G, box, k_bins, dk=dk))
    measured = P_cross / P_auto - b1L

    pred_shell = (
        b_phi * f_NL * pweighted_inv_M_shell(box, cosmo, k_bins, dk, z, BACKEND)
    )
    pred_centre = b_phi * f_NL / IC.poisson_M(k_bins, cosmo, z=z, backend=BACKEND)

    print("=== Gate 1: linear-field b_phi response ===")
    print(f"  L={box.box_size} N={box.n_mesh}  b1={b1L} b_phi={b_phi} f_NL={f_NL}")
    print(
        f"  {'k':>8} {'measured':>10} {'pred(shell)':>11} {'meas/shell':>11}"
        f" {'pred(centre)':>12}"
    )
    for i, kc in enumerate(k_bins):
        print(
            f"  {kc:8.4f} {measured[i]:10.3f} {pred_shell[i]:11.3f}"
            f" {measured[i]/pred_shell[i]:11.5f} {pred_centre[i]:12.3f}"
        )
    r = measured / pred_shell
    print(f"  -> meas/shell mean={r.mean():.5f}  max|dev|={np.max(np.abs(r-1)):.2e}\n")


def barreira_bphi(b1E, p=0.55):
    """Barreira non-universal b_phi = 2 delta_c (b1_E - p)."""
    return 2.0 * DELTA_C * (b1E - p)


def gate2_data(box, cosmo, time, seeds, eps, coeffs, n_bins=8):
    """Through-PM d ln P_g/df_NL (matched-phase FD) and the linear Dalal prediction."""
    dk = box.k_fundamental
    k_bins = dk * np.arange(1, n_bins + 1)
    meas, err = dlnP_df_matched(box, cosmo, time, seeds, eps, k_bins, dk, coeffs)
    b1E = 1.0 + coeffs["b1"]
    pred = (
        2.0
        * coeffs["b_phi"]
        / b1E
        * pweighted_inv_M_shell(box, cosmo, k_bins, dk, time.z_final, BACKEND)
    )
    return k_bins, meas, err, pred


def gate2():
    box = BoxConfig(box_size=2048.0, n_mesh=64)
    cosmo = Cosmology()
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10, integrator="bullfrog")
    seeds = [0, 1, 2]
    eps = 10.0

    b1L = 1.0
    b1E = 1.0 + b1L
    coeffs = dict(b1=b1L, b_phi=barreira_bphi(b1E))  # isolate b1 + explicit b_phi
    null_coeffs = dict(b1=0.0, b_phi=0.0)  # weights = 1 -> matter null

    k_bins, meas, err, pred = gate2_data(box, cosmo, time, seeds, eps, coeffs)
    null, _ = dlnP_df_matched(
        box, cosmo, time, seeds, eps, k_bins, box.k_fundamental, null_coeffs
    )

    print("=== Gate 2: through-PM d ln P_g / df_NL  (THE trust proof) ===")
    print(
        f"  L={box.box_size} N={box.n_mesh} n_steps={time.n_steps} "
        f"bullfrog  seeds={seeds} eps={eps}"
    )
    print(f"  b1_L={b1L} -> b1_E={b1E}  b_phi={coeffs['b_phi']:.3f} (Barreira p=0.55)")
    print("  prediction = 2 b_phi/(b1_E M(k)) [linear Dalal, no dilution]")
    print(
        f"  {'k':>8} {'measured':>11} {'+/-':>9} {'pred':>11} {'meas/pred':>10}"
        f" {'matter null':>12}"
    )
    for i, kc in enumerate(k_bins):
        print(
            f"  {kc:8.4f} {meas[i]:11.4f} {err[i]:9.4f} {pred[i]:11.4f}"
            f" {meas[i]/pred[i]:10.4f} {null[i]:12.4f}"
        )
    r = meas / pred
    print(f"  -> meas/pred (low-k bins 1-4): {r[:4]}")
    print("     Eulerian native-b2 tracer is ~0.25 here; Lagrangian target ~1.0\n")


def gate4_resolution():
    """Gate 4: the explicit-b_phi response is resolution-stable (flat in N).

    The Eulerian native-b2 dilution worsens with N (sigma^2 is the UV-sensitive
    evolved mesh variance). The explicit Lagrangian b_phi is a large-scale linear
    term, so meas/pred should stay ~1 at both N=64 and N=128 (same L).
    """
    cosmo = Cosmology()
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=10, integrator="bullfrog")
    seeds = [0, 1, 2]
    eps = 10.0
    b1L = 1.0
    coeffs = dict(b1=b1L, b_phi=barreira_bphi(1.0 + b1L))

    print("=== Gate 4: resolution stability of the through-PM b_phi response ===")
    print(f"  L=2048  b1_E={1.0+b1L}  b_phi={coeffs['b_phi']:.3f}  seeds={seeds}")
    print(f"  {'N':>5} {'k':>8} {'meas/pred':>10}")
    for N in (64, 128):
        box = BoxConfig(box_size=2048.0, n_mesh=N)
        k_bins, meas, err, pred = gate2_data(box, cosmo, time, seeds, eps, coeffs)
        r = meas / pred
        for i in range(4):  # low-k bins, where f_NL lives
            print(f"  {N:5d} {k_bins[i]:8.4f} {r[i]:10.4f}")
        print(f"        -> N={N} low-k mean meas/pred = {r[:4].mean():.4f}")
    print()


if __name__ == "__main__":
    gate1()
    gate2()
    gate4_resolution()
