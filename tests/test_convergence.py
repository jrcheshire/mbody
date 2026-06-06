"""External convergence cross-check: the PM forward model vs analytic linear theory.

Layer A of the cross-check (mbody-internal, no external code). On matched phases
(same seed -> cosmic variance cancels in ratios) the evolved field is compared to
linear theory with absolute normalization:

  - the CIC mass-assignment window and its deconvolution (deterministic);
  - the matched transfer T(k) = P_PM/P_lin -> 1 at the largest scale;
  - the propagator r(k) = <delta_PM delta_lin>/sqrt(P_PM P_lin) -> 1 at low k;
  - growth convergence: fastpm is step-count-independent, exact converges to it.

Tolerances were measured first with scripts/probe_pk_convergence.py (the matched
ratios are cosmic-variance-cancelled, so the fundamental-mode checks are tight even
at this small box; deeper bins droop because a coarse PM under-resolves small
scales -- the expected limitation, not asserted away). The CCL community-code
anchor (Layer B) lives in tests/test_external_ccl.py.
"""

import numpy as np

from mbody.config import (
    BoxConfig,
    Cosmology,
    TimeStepping,
    SimConfig,
    InitialConditions,
)
from mbody import fields as F
from mbody import ic as IC
from mbody import diagnostics as D
from mbody import driver

COSMO = Cosmology()
BOX = BoxConfig(box_size=256.0, n_mesh=32, n_particles=32)
TIME = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5)
NSEED = 4


def test_cic_window_and_deconvolution():
    # The CIC window is 1 at the DC mode, < 1 elsewhere, and falls toward Nyquist;
    # deconvolving a painted field's power divides by W^2 <= 1, so it never lowers
    # any mode's power (and raises high-k modes).
    W = F.cic_window(BOX)
    assert W[0, 0, 0] == 1.0
    assert np.all(W <= 1.0 + 1e-12)
    assert W.min() < 0.7  # substantial suppression near Nyquist
    res = driver.run(SimConfig(box=BOX, time=TIME, ic=InitialConditions(seed=0)))
    _, P_raw, _ = res.power()
    _, P_dec, _ = res.power(deconvolve_cic=True)
    assert np.all(P_dec >= P_raw - 1e-12)
    assert P_dec[-1] > 1.2 * P_raw[-1]  # the near-Nyquist bin is deconvolved up


def test_matched_transfer_fundamental():
    # Matched (same-seed) transfer T(k) = P_PM/P_lin at z=0: cosmic variance
    # cancels, so the fundamental mode -> 1 (absolute growth + normalization).
    # Measured T_dec[0] ~ 0.99 +/- 0.02 over seeds; deeper bins droop (PM
    # under-resolution), so only the cleanest low-k mode is asserted.
    knyq = BOX.k_nyquist
    Tsum = None
    for s in range(NSEED):
        res = driver.run(SimConfig(box=BOX, time=TIME, ic=InitialConditions(seed=s)))
        k, P_pm, _ = res.power(deconvolve_cic=True)
        _, P_lin, _ = F.power_spectrum(
            IC.linear_density(BOX, COSMO, seed=s, z=0.0), BOX
        )
        Tsum = (P_pm / P_lin) if Tsum is None else Tsum + P_pm / P_lin
    T = Tsum / NSEED
    assert abs(T[0] - 1.0) < 0.08
    assert k[0] < 0.1 * knyq  # the asserted bin really is the large-scale one


def test_propagator_tracks_linear_ic():
    # The propagator r(k) -> 1 at the fundamental (phases track linear theory) and
    # decoheres toward high k (the standard PM diagnostic). r is window-independent.
    res = driver.run(SimConfig(box=BOX, time=TIME, ic=InitialConditions(seed=0)))
    dlin = IC.linear_density(BOX, COSMO, seed=0, z=0.0)
    k, r, _ = D.cross_correlation(res.final_field, dlin, BOX)
    assert r[0] > 0.99
    assert r[-1] < r[0]  # decoheres at small scales
    assert r[-1] < 0.5


def _growth_ratio(integrator, n_steps):
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=n_steps, integrator=integrator)
    res = driver.run(SimConfig(box=BOX, time=t), record=True)
    a_arr, R = res.growth_history()
    Dref = D.linear_growth_reference(a_arr, COSMO)
    return R[-1] / Dref[-1]


def test_growth_convergence_vs_steps():
    # FastPM gets linear growth right at any step count (flat in n_steps), where the
    # exact-background leapfrog has a low-step deficit that converges upward.
    f2, f8 = _growth_ratio("fastpm", 2), _growth_ratio("fastpm", 8)
    e2, e8 = _growth_ratio("exact", 2), _growth_ratio("exact", 8)
    assert abs(f8 - f2) < 0.02  # fastpm ~ step-independent
    assert e2 < f2 - 0.02  # exact under-grows at low step count
    assert e8 > e2  # and converges upward with more steps
    assert e8 > 0.95 * f8  # toward the fastpm value
