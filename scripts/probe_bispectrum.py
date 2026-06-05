"""Probe the squeezed-bispectrum estimator before fixing any test tolerances.

This is the "measure first" step (per the project's standing rule that
tolerances are measured, not guessed). It runs fields.bispectrum on the linear
local-f_NL field (ic.linear_density) over many seeds and reports:

  * per-triangle mean(B) / template and the seed-to-seed scatter,
  * the single calibration constant c_cal from a least-squares
    B_mean = c_cal * B_template (should come out ~ 1 if the V^2/N^9
    normalization is exact),
  * the f_NL = 0 residual relative to the f_NL signal (the noise floor for the
    "B(0) ~ 0" test),
  * a linearity check of B across a few f_NL values,
  * an autodiff smoke check: mx.grad of bispectrum_single w.r.t. f_NL vs a
    finite difference.

Run: pixi run python scripts/probe_bispectrum.py
Nothing is written; the printed numbers set the tolerances in
tests/test_bispectrum.py and get recorded in the work log.
"""

import time

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import fields as F
from mbody import ic as IC

COSMO = Cosmology()
# Light default so the probe reruns in a few seconds; bump box/NSEED for a
# tighter calibration (the 256/128, NSEED=40 regime gave squeezed c_cal ~1.01).
BOX = BoxConfig(box_size=128.0, n_mesh=64, n_particles=64)
F_NL = 1000.0
NSEED = 24
SEED0 = 1000


def make_triangles(box):
    """A handful of squeezed triangles + an equilateral control.

    Squeezed: two short sides fixed at k_short, long side swept over the first
    several bins. Equilateral: (k, k, k) over a few k. All centers are multiples
    of the fundamental so the same floats feed estimator and template.
    """
    kf = box.k_fundamental
    k_short = round(6 * kf, 12)  # ~0.29 h/Mpc, well sampled
    long_mult = [1, 2, 3, 4, 5]
    squeezed = [(round(n * kf, 12), k_short, k_short) for n in long_mult]
    eq_mult = [3, 5, 8]
    equilateral = [(round(n * kf, 12),) * 3 for n in eq_mult]
    return squeezed, equilateral


def measure(triangles, f_NL, nseed):
    """Return (B_all, n_tri) where B_all is (nseed, ntri)."""
    B_all = np.empty((nseed, len(triangles)), dtype=np.float64)
    n_tri = None
    for i in range(nseed):
        delta = IC.linear_density(BOX, COSMO, seed=SEED0 + i, f_NL=f_NL)
        B, n_tri = F.bispectrum(delta, BOX, triangles)
        B_all[i] = B
    return B_all, n_tri


def report_block(name, triangles, template):
    t0 = time.time()
    B_all, n_tri = measure(triangles, F_NL, NSEED)
    B0_all, _ = measure(triangles, 0.0, NSEED)
    dt = time.time() - t0

    mean = B_all.mean(axis=0)
    std = B_all.std(axis=0, ddof=1)
    sem = std / np.sqrt(NSEED)
    mean0 = B0_all.mean(axis=0)
    std0 = B0_all.std(axis=0, ddof=1)

    print(f"\n=== {name}  (f_NL={F_NL}, nseed={NSEED}, {dt:.1f}s) ===")
    hdr = "  k1      k2      k3     n_tri   B_meas/B_tmpl  +/-   |B0|/|B|"
    print(hdr)
    for j, tri in enumerate(triangles):
        ratio = mean[j] / template[j]
        ratio_err = sem[j] / abs(template[j])
        floor = abs(mean0[j]) / abs(mean[j]) if mean[j] != 0 else np.nan
        k1, k2, k3 = (tri + (tri[-1],) * (3 - len(tri)))[:3]
        print(
            f"  {k1:6.4f}  {k2:6.4f}  {k3:6.4f}  {n_tri[j]:6.0f}  "
            f"{ratio:8.3f}    {ratio_err:5.3f}  {floor:7.4f}"
        )

    # Single calibration constant: least squares B_mean = c_cal * B_template.
    c_cal = float(np.sum(mean * template) / np.sum(template**2))
    # Same fit dropping the lowest-k_long (most bin-width-biased) squeezed bin.
    print(f"  c_cal (all triangles)      = {c_cal:.4f}")
    return B_all, mean0, std0


def linearity(triangles, template):
    print("\n=== linearity in f_NL (squeezed, seed-averaged over 8) ===")
    fnls = [-200.0, -100.0, 0.0, 100.0, 200.0]
    ns = 8
    means = []
    for f in fnls:
        acc = np.zeros(len(triangles))
        for i in range(ns):
            d = IC.linear_density(BOX, COSMO, seed=SEED0 + i, f_NL=f)
            B, _ = F.bispectrum(d, BOX, triangles)
            acc += B
        means.append(acc / ns)
    means = np.array(means)  # (nf, ntri)
    # Fit B = a*f + b per triangle; report slope/template-per-fNL and intercept.
    fnls_arr = np.array(fnls)
    for j, tri in enumerate(triangles):
        a, b = np.polyfit(fnls_arr, means[:, j], 1)
        slope_ratio = a / (template[j] / F_NL)  # template is linear in f_NL
        print(
            f"  tri {tri[0]:.4f}: slope/expected = {slope_ratio:6.3f}, "
            f"intercept/|B(200)| = {b / abs(means[-1, j]):+.3f}"
        )


def ad_smoke(triangle):
    print("\n=== autodiff smoke: dB/df_NL vs finite difference ===")

    def Bfun(f):
        d = IC.linear_density(BOX, COSMO, seed=SEED0, f_NL=f)
        return F.bispectrum_single(d, BOX, triangle)

    f0 = mx.array(150.0)
    g = mx.grad(Bfun)(f0)
    h = 5.0
    fd = (float(Bfun(mx.array(150.0 + h))) - float(Bfun(mx.array(150.0 - h)))) / (2 * h)
    g = float(g)
    rel = abs(g - fd) / abs(fd)
    print(f"  grad = {g:.6e}   fd = {fd:.6e}   rel.err = {rel:.3e}")


def deterministic_check():
    """Noise-free normalization check via a closed-triangle plane-wave field.

    delta(x) = a[cos(q1.x) + cos(q2.x) + cos(q3.x)] with integer wavevectors
    (3,4,5)*kf summing to zero. Each shell then holds exactly one populated mode
    pair, so I_i = a cos(qi.x) and sum_x I1 I2 I3 = a^3 N^3 / 4 analytically.
    Hence the estimator must return B = L^6 a^3 / (4 n_tri) exactly (no cosmic
    variance) -- a direct test of the V^2/N^9 prefactor and the data path.
    """
    box = BoxConfig(box_size=200.0, n_mesh=32, n_particles=32)
    N, L = box.n_mesh, box.box_size
    kf = box.k_fundamental
    a = 0.5
    ax = (2.0 * np.pi / N) * np.arange(N)  # phase per cell along an axis

    def cos(m):
        return np.cos(
            m[0] * ax[:, None, None]
            + m[1] * ax[None, :, None]
            + m[2] * ax[None, None, :]
        )

    field = a * (cos((3, 0, 0)) + cos((0, 4, 0)) + cos((-3, -4, 0)))
    delta = mx.array(field.astype(np.float32))
    B, n_tri = F.bispectrum(delta, box, [(3 * kf, 4 * kf, 5 * kf)])
    predicted = L**6 * a**3 / (4.0 * n_tri[0])
    print("\n=== deterministic plane-wave normalization ===")
    print(
        f"  B = {B[0]:.6e}  predicted L^6 a^3/(4 n_tri) = {predicted:.6e}  "
        f"rel.err = {abs(B[0] / predicted - 1):.3e}  (n_tri={n_tri[0]:.0f})"
    )


def main():
    deterministic_check()
    squeezed, equilateral = make_triangles(BOX)
    # Compare against the bin-averaged template (removes the binning systematic);
    # the continuum template is what the naive bin-centre comparison would use.
    tmpl_sq = IC.local_bispectrum_binned(BOX, COSMO, squeezed, F_NL)
    tmpl_eq = IC.local_bispectrum_binned(BOX, COSMO, equilateral, F_NL)
    cont_sq = IC.local_bispectrum_template(squeezed, COSMO, F_NL)
    print("squeezed binned/continuum template ratio:")
    print("  " + "  ".join(f"{b / c:.3f}" for b, c in zip(tmpl_sq, cont_sq)))

    report_block("squeezed", squeezed, tmpl_sq)
    report_block("equilateral", equilateral, tmpl_eq)
    linearity(squeezed, tmpl_sq)
    ad_smoke(squeezed[2])


if __name__ == "__main__":
    main()
