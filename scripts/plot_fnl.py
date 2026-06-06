"""The local-f_NL payoff figure: see the non-Gaussianity, then measure it.

Three panels:
  1. a density slice of the linear f_NL field (structure to look at);
  2. the one-point PDF at f_NL = -F, 0, +F on a log axis -- local f_NL skews the
     field, so the tails become asymmetric (annotated with the measured
     skewness, which flips sign with f_NL);
  3. the squeezed bispectrum B(k_long; k_short, k_short): the matched-phase
     cosmic-variance-cancelled measurement (points, with seed scatter) against
     the bin-averaged template (the estimator's exact expectation) and the
     continuum tree template (smooth curve), showing the 1/k_long^2
     scale-dependent-bias divergence.

Writes outputs/fnl_bispectrum.png (gitignored). Run:
    pixi run python scripts/plot_fnl.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody.config import BoxConfig, Cosmology  # noqa: E402
from mbody import diagnostics as D  # noqa: E402
from mbody import fields as F  # noqa: E402
from mbody import ic as IC  # noqa: E402

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
COSMO = Cosmology()
BOX = BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)
F_NL = 1000.0
NSEED = 24


def squeezed_sequence():
    kf = BOX.k_fundamental
    k_short = round(8 * kf, 12)
    longs = [round(n * kf, 12) for n in (1, 2, 3, 4, 6, 8)]
    return [(kl, k_short, k_short) for kl in longs], np.array(longs), k_short


def antisym_measure(triangles, nseed, seed0=3000):
    """Matched-phase [B(+F) - B(-F)]/2 per seed; return (mean, sem)."""
    per = np.empty((nseed, len(triangles)))
    for s in range(nseed):
        dp = IC.linear_density(BOX, COSMO, seed=seed0 + s, f_NL=F_NL)
        dm = IC.linear_density(BOX, COSMO, seed=seed0 + s, f_NL=-F_NL)
        bp, _ = F.bispectrum(dp, BOX, triangles)
        bm, _ = F.bispectrum(dm, BOX, triangles)
        per[s] = 0.5 * (bp - bm)
    return per.mean(axis=0), per.std(axis=0, ddof=1) / np.sqrt(nseed)


def main():
    triangles, k_long, k_short = squeezed_sequence()

    print(f"measuring squeezed bispectrum over {NSEED} matched seed pairs ...")
    meas, sem = antisym_measure(triangles, NSEED)
    tmpl_binned = IC.local_bispectrum_binned(BOX, COSMO, triangles, F_NL)
    fine_long = np.geomspace(k_long.min() * 0.9, k_long.max() * 1.1, 60)
    fine_tris = [(kl, k_short, k_short) for kl in fine_long]
    tmpl_cont = IC.local_bispectrum_template(fine_tris, COSMO, F_NL)
    c_cal = float(np.sum(meas * tmpl_binned) / np.sum(tmpl_binned**2))

    # Fields for the slice and the PDF.
    fields = {
        f: IC.linear_density(BOX, COSMO, seed=42, f_NL=f) for f in (-F_NL, 0.0, F_NL)
    }

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel 1: density slice of the +f_NL field.
    slab = np.asarray(fields[F_NL][:, :, BOX.n_mesh // 2])
    vlim = 3.0 * slab.std()
    im = ax0.imshow(
        slab, cmap="RdBu_r", vmin=-vlim, vmax=vlim, extent=[0, BOX.box_size] * 2
    )
    ax0.set_title(f"linear density slice (f_NL = {F_NL:.0f})")
    ax0.set_xlabel("Mpc/h")
    ax0.set_ylabel("Mpc/h")
    fig.colorbar(im, ax=ax0, fraction=0.046, label="delta")

    # Panel 2: one-point PDF, log-y, with skewness annotation.
    colors = {-F_NL: "C0", 0.0: "k", F_NL: "C3"}
    for f, d in fields.items():
        arr = np.asarray(d).ravel()
        sk = D.skewness(arr)
        ax1.hist(
            arr / arr.std(),
            bins=120,
            range=(-6, 6),
            histtype="step",
            density=True,
            color=colors[f],
            label=f"f_NL = {f:+.0f}  (skew {sk:+.3f})",
        )
    ax1.set_yscale("log")
    ax1.set_xlabel("delta / sigma")
    ax1.set_ylabel("PDF")
    ax1.set_title("one-point PDF: local f_NL skews the tails")
    ax1.legend(fontsize=8)

    # Panel 3: squeezed bispectrum, measured vs template.
    ax2.plot(fine_long, tmpl_cont, "-", color="0.6", label="continuum tree template")
    ax2.plot(
        k_long, tmpl_binned, "s", color="C1", mfc="none", label="bin-averaged template"
    )
    ax2.errorbar(
        k_long,
        meas,
        yerr=sem,
        fmt="o",
        color="C3",
        label=f"measured (c_cal = {c_cal:.3f})",
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("k_long  [h/Mpc]")
    ax2.set_ylabel("B(k_long, k_short, k_short)  [(Mpc/h)^6]")
    ax2.set_title(f"squeezed divergence (k_short = {k_short:.3f})")
    ax2.legend(fontsize=8)

    fig.suptitle(
        f"M-body local-f_NL bispectrum  (L={BOX.box_size:.0f}, "
        f"N={BOX.n_mesh}, {NSEED} matched seed pairs)"
    )
    fig.tight_layout()
    path = OUT / "fnl_bispectrum.png"
    fig.savefig(path, dpi=130)
    print(f"c_cal = {c_cal:.4f}; wrote {path}")


if __name__ == "__main__":
    main()
