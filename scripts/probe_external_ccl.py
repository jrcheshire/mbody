"""External cross-check Layer B: mbody's cosmology layer vs CCL (pyccl).

mbody's CAMB backend is the same CAMB it would be compared against, and the
internal EH98-vs-CAMB check could share a convention bug; CCL (the LSST DESC Core
Cosmology Library) is a fully independent implementation of the growth factor,
transfer function, and sigma8 normalization. This probe builds a CCL cosmology
matching mbody's defaults and compares, with the h-unit conversions made explicit
(CCL works in physical Mpc; mbody in Mpc/h):

    k_ccl [1/Mpc]   = k_mbody [h/Mpc] * h
    P_mbody [(Mpc/h)^3] = P_ccl [Mpc^3] * h^3
    R_ccl [Mpc]     = R_mbody [Mpc/h] / h

  B1. growth D(z) (both normalized to D(z=0)=1);
  B2. linear P(k): mbody "eh98" vs CCL "eisenstein_hu", and "camb" vs CCL
      "boltzmann_camb";
  B3. sigma8 and sigma_R(R).

Run: pixi run python scripts/probe_external_ccl.py
The printed numbers set the tolerances in tests/test_external_ccl.py.
"""

import numpy as np
import pyccl as ccl

from mbody.config import Cosmology
from mbody import cosmology as C


def ccl_cosmology(cosmo, transfer_function):
    """A CCL cosmology matching an mbody Cosmology (flat LCDM, massless nu)."""
    return ccl.Cosmology(
        Omega_c=cosmo.Omega_cdm,
        Omega_b=cosmo.Omega_b,
        h=cosmo.h,
        n_s=cosmo.n_s,
        sigma8=cosmo.sigma8,
        T_CMB=cosmo.T_cmb_K,
        m_nu=0.0,
        Omega_k=0.0,
        transfer_function=transfer_function,
    )


def main():
    cm = Cosmology()
    h = cm.h

    print("B1. growth factor D(z), normalized D(0)=1:")
    cc = ccl_cosmology(cm, "eisenstein_hu")
    print("  z       mbody       ccl         rel.err")
    g_rel = []
    for z in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 9.0):
        a = 1.0 / (1.0 + z)
        Dm = C.growth_factor(z, cm)
        Dc = ccl.growth_factor(cc, a)
        g_rel.append(abs(Dm / Dc - 1.0))
        print(f"  {z:4.1f}   {Dm:.6f}   {Dc:.6f}   {abs(Dm/Dc-1):.2e}")
    print(f"  max growth rel.err = {max(g_rel):.2e}")

    k = np.logspace(-3, 1, 60)  # h/Mpc
    print("\nB2. linear P(k, z=0) vs CCL (ratio mbody/ccl over k in [1e-3, 10]):")
    for mb_backend, ccl_tf in (("eh98", "eisenstein_hu"), ("camb", "boltzmann_camb")):
        cc = ccl_cosmology(cm, ccl_tf)
        P_ccl = ccl.linear_matter_power(cc, k * h, 1.0) * h**3
        P_mb = C.linear_power(k, cm, z=0.0, backend=mb_backend)
        ratio = P_mb / P_ccl
        print(
            f"  {mb_backend:5s} vs {ccl_tf:16s}: ratio "
            f"min={ratio.min():.4f} max={ratio.max():.4f} "
            f"median={np.median(ratio):.4f} max|.-1|={np.max(np.abs(ratio-1)):.2e}"
        )

    print("\nB3. sigma8 and sigma_R(R):")
    cc = ccl_cosmology(cm, "eisenstein_hu")
    print(
        f"  sigma8: mbody sigma_R(8)={C.sigma_R(8.0, cm):.5f}  "
        f"ccl.sigma8={ccl.sigma8(cc):.5f}  target={cm.sigma8}"
    )
    print("  R[Mpc/h]   mbody      ccl        rel.err")
    for R in (4.0, 8.0, 16.0, 32.0):
        sm = C.sigma_R(R, cm, backend="eh98")
        sc = ccl.sigmaR(cc, R / h, 1.0)
        print(f"  {R:6.1f}    {sm:.5f}   {sc:.5f}   {abs(sm/sc-1):.2e}")


if __name__ == "__main__":
    main()
