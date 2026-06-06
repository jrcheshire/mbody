"""External cross-check Layer B: mbody's cosmology layer vs CCL (pyccl).

An independent community-code anchor for the linear theory mbody's whole forward
model is built on. mbody's CAMB backend is the same CAMB it would self-check
against, and the internal EH98-vs-CAMB agreement could hide a shared convention
bug; CCL reimplements the growth factor, the transfer function, and the sigma8
normalization independently. The h-unit conversions (CCL is physical Mpc, mbody is
Mpc/h) are the thing this check really pins -- see scripts/probe_external_ccl.py,
where the tolerances below were measured.

Skipped (not failed) if pyccl is unavailable, so the core suite never depends on
the optional anchor.
"""

import numpy as np
import pytest

ccl = pytest.importorskip("pyccl")

from mbody.config import Cosmology  # noqa: E402
from mbody import cosmology as C  # noqa: E402

CM = Cosmology()


def _ccl(transfer_function):
    return ccl.Cosmology(
        Omega_c=CM.Omega_cdm,
        Omega_b=CM.Omega_b,
        h=CM.h,
        n_s=CM.n_s,
        sigma8=CM.sigma8,
        T_CMB=CM.T_cmb_K,
        m_nu=0.0,
        Omega_k=0.0,
        transfer_function=transfer_function,
    )


def test_growth_factor_matches_ccl():
    # D(z) agrees to < 0.3% out to z = 9 (the small high-z drift is mbody
    # neglecting radiation in E(z), an intentional toy simplification).
    cc = _ccl("eisenstein_hu")
    for z in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 9.0):
        Dm = C.growth_factor(z, CM)
        Dc = ccl.growth_factor(cc, 1.0 / (1.0 + z))
        assert abs(Dm / Dc - 1.0) < 3e-3


def test_linear_power_eh98_matches_ccl():
    # mbody's EH98 port and the CCL EH98 transfer are the same formula: P(k) agrees
    # to ~1e-6 across k in [1e-3, 10] h/Mpc (a strong check of the sigma8 norm too).
    cc = _ccl("eisenstein_hu")
    k = np.logspace(-3, 1, 60)
    P_ccl = ccl.linear_matter_power(cc, k * CM.h, 1.0) * CM.h**3
    P_mb = C.linear_power(k, CM, z=0.0, backend="eh98")
    assert np.max(np.abs(P_mb / P_ccl - 1.0)) < 1e-3


def test_linear_power_camb_matches_ccl():
    # mbody's CAMB pipeline (fiducial A_s, rescaled to sigma8, log-log interpolated)
    # vs CCL's CAMB: two independent Boltzmann pipelines agree to < 0.5%.
    cc = _ccl("boltzmann_camb")
    k = np.logspace(-3, 1, 60)
    P_ccl = ccl.linear_matter_power(cc, k * CM.h, 1.0) * CM.h**3
    P_mb = C.linear_power(k, CM, z=0.0, backend="camb")
    assert np.max(np.abs(P_mb / P_ccl - 1.0)) < 5e-3


def test_sigmaR_matches_ccl():
    # sigma_R(R) (R in Mpc/h -> CCL R/h in Mpc) agrees to < 1e-3; sigma8 (R=8 Mpc/h)
    # recovers the input to machine precision in both.
    cc = _ccl("eisenstein_hu")
    assert abs(ccl.sigma8(cc) - CM.sigma8) < 1e-3
    for R in (4.0, 8.0, 16.0, 32.0):
        sm = C.sigma_R(R, CM, backend="eh98")
        sc = ccl.sigmaR(cc, R / CM.h, 1.0)
        assert abs(sm / sc - 1.0) < 1e-3
