"""f_NL injection-recovery probe for the galaxy mock (mbody.catalog) -- Stage C.

Injects a known local f_NL in the initial conditions, evolves through the PM, and
checks that the mock galaxy field carries the Dalal scale-dependent bias
Delta_b(k) ~ f_NL / M(k) ~ f_NL / k^2 at large scales. The signal is a matched-phase
finite difference of dlnP/df_NL (same IC phases at +/-eps cancel cosmic variance),
seed-averaged, on:

  * the noiseless clipped galaxy field (the mock's expected field) -- the headline:
    does the f_NL response survive the clip non-negativity map?
  * the unclipped local-bias field, for the clip's effect on the amplitude;
  * the matter field (b2 = 0) -- the null (matter has no O(f_NL) power);
  * the full Poisson catalog, recovered via the shot-free cross spectrum with the
    matter field (the placement window is f_NL-independent -> cancels in dlnP).

Two documented forward-model facts (diagnostics, not errors):
  - amplitude SUPPRESSION: the through-PM dlnP/df_NL is ~1/4 of the linear-theory
    Dalal prediction (scale_dependent_bias_response_binned) -- nonlinear evolution
    adds non-primordial small-scale power that dilutes the b2-sourced response;
  - shape FLATTENING: the ratio to 1/M(k) rises with k (the clean 1/k^2 is a
    linear-theory result).

Prints the numbers behind the locked Stage-C test tolerance; nothing here asserts.
Run: ``pixi run python scripts/probe_mock_fnl.py``.
"""

import numpy as np
import mlx.core as mx

from mbody import bias as B
from mbody import catalog as CAT
from mbody import fields as F
from mbody import painting as PA
from mbody.config import (
    BoxConfig,
    CatalogSampling,
    Cosmology,
    InitialConditions,
    SimConfig,
    TimeStepping,
    Tracer,
)
from mbody.driver import run

COSMO = Cosmology()
BOX = BoxConfig(box_size=1024.0, n_mesh=64, n_particles=64)
TIME = TimeStepping(z_init=9.0, z_final=0.0, n_steps=5, integrator="fastpm")
B1, B2 = 2.0, 1.0
NBAR = 2.0e-2
EPS = 100.0
NSEED = 16
BACKEND = "eh98"


def _kbins(n_low=6):
    return np.arange(1, n_low + 1) * BOX.k_fundamental


def _evolve(f_NL, seed):
    """Run the PM at this f_NL (matched phase via seed); return measured fields."""
    cfg = SimConfig(
        cosmology=COSMO,
        box=BOX,
        time=TIME,
        ic=InitialConditions(f_NL=f_NL, kind="local_fnl", seed=seed),
        tracer=Tracer(b1=B1, b2=B2, A=1.0),
        catalog=CatalogSampling(enabled=True, nbar=NBAR, draw_seed=seed),
    )
    res = run(cfg, backend=BACKEND)
    delta_m = PA.density_contrast(res.x, BOX)
    delta_g = B.local_bias_tracer(delta_m, B1, B2)  # unclipped
    lam = CAT.intensity_field(delta_g, NBAR, BOX)
    delta_e = mx.array(np.asarray(lam / lam.mean() - 1.0, np.float32))  # clipped
    gxyz = mx.array(np.ascontiguousarray(res.catalog.xyz, np.float32))
    gfield = PA.density_contrast(gxyz, BOX)
    return delta_m, delta_g, delta_e, gfield


def _auto(fp, fm, kb):
    lp = np.log(np.asarray(F.band_power(fp, BOX, kb), np.float64))
    lm = np.log(np.asarray(F.band_power(fm, BOX, kb), np.float64))
    return (lp - lm) / (2.0 * EPS)


def _cross(ap, am, mp, mm, kb):
    lp = np.log(np.asarray(F.cross_power(ap, mp, BOX, kb), np.float64))
    lm = np.log(np.asarray(F.cross_power(am, mm, BOX, kb), np.float64))
    return (lp - lm) / (2.0 * EPS)


def main():
    kb = _kbins()
    shape = np.asarray(B.scale_dependent_shape(kb, COSMO, backend=BACKEND))  # 1/M(k)
    analytic = B.scale_dependent_bias_response_binned(
        BOX, COSMO, kb, B1, B2, backend=BACKEND
    )
    print(
        f"box L={BOX.box_size} n_mesh={BOX.n_mesh} steps={TIME.n_steps} "
        f"b1={B1} b2={B2} nbar={NBAR} eps={EPS} nseed={NSEED} backend={BACKEND}"
    )
    print(f"k bins (h/Mpc): {np.array2string(kb, precision=4)}\n")

    keys = ("clip", "unclip", "matter", "cat", "clip_x")
    acc = {k: np.zeros(len(kb)) for k in keys}
    for s in range(NSEED):
        dm_p, dg_p, de_p, gf_p = _evolve(EPS, s)
        dm_m, dg_m, de_m, gf_m = _evolve(-EPS, s)
        acc["clip"] += _auto(de_p, de_m, kb)
        acc["unclip"] += _auto(dg_p, dg_m, kb)
        acc["matter"] += _auto(dm_p, dm_m, kb)
        acc["cat"] += _cross(gf_p, gf_m, dm_p, dm_m, kb)
        acc["clip_x"] += _cross(de_p, de_m, dm_p, dm_m, kb)
    for k in acc:
        acc[k] /= NSEED

    ref = shape * (acc["clip"][0] / shape[0])  # 1/M(k) normalized at the lowest bin
    print("=== dlnP/df_NL: clipped field vs 1/M(k) and linear theory ===")
    print("   bin    k        clipped     1/M*norm   ratio   PM/linear   matter")
    for i, k in enumerate(kb):
        print(
            f"   {i+1:>3}  {k:7.4f}  {acc['clip'][i]:10.3e}  {ref[i]:10.3e}  "
            f"{acc['clip'][i]/ref[i]:5.2f}   {acc['clip'][i]/analytic[i]:6.3f}    "
            f"{acc['matter'][i]:10.3e}"
        )

    null = np.max(np.abs(acc["matter"][:3]) / np.abs(acc["clip"][:3]))
    print(
        f"\n  scale-dependence clipped bin1/bin3 : {acc['clip'][0]/acc['clip'][2]:.2f}"
    )
    print(
        f"  clip preserves signal (clip/unclip, bin1): {acc['clip'][0]/acc['unclip'][0]:.3f}"
    )
    print(f"  matter null |dlnP_m|/|dlnP_clip| (max, bins1-3): {null:.4f}")
    print(
        f"  through-PM / linear theory (bins1-3): "
        f"{np.array2string(acc['clip'][:3]/analytic[:3], precision=3)}  (SUPPRESSED)"
    )

    print("\n=== Poisson catalog recovery (shot-free cross spectrum) ===")
    print("   bin    k        cat_cross   clip_cross  cat/clip")
    for i, k in enumerate(kb):
        print(
            f"   {i+1:>3}  {k:7.4f}  {acc['cat'][i]:10.3e}  {acc['clip_x'][i]:10.3e}  "
            f"{acc['cat'][i]/acc['clip_x'][i]:6.3f}"
        )


if __name__ == "__main__":
    main()
