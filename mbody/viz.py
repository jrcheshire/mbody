"""Optional visualization layer: density slices, the structure-formation
animation, and the standard run dashboard.

This is the only mbody module that depends on matplotlib, and mbody.__init__
deliberately does not import it -- so ``import mbody`` stays free of plotting
dependencies and off the autodiff path. Import it explicitly where you render::

    from mbody import viz

Everything here consumes plain numpy arrays / the diagnostics summaries and
writes figures or animations under a caller-chosen path (outputs/ is
gitignored). None of it is differentiable or performance-critical.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mbody import cosmology as C  # noqa: E402
from mbody import diagnostics as D  # noqa: E402
from mbody import fields as F  # noqa: E402


def _slab_image(delta, box, thickness=None, axis=2):
    """Mean of (1 + delta) over a thin slab along `axis`, as a 2D numpy array.

    The projected density 1 + delta is averaged over `thickness` cells (default
    a few) so a slice is not pure shot noise. Returns the (N, N) projection on
    the two axes orthogonal to `axis`.
    """
    d = np.asarray(delta, dtype=np.float64)
    N = box.n_mesh
    if thickness is None:
        thickness = max(1, N // 16)
    sl = [slice(None)] * 3
    sl[axis] = slice(0, thickness)
    return (1.0 + d[tuple(sl)]).mean(axis=axis)


def density_slice(
    delta, box, ax=None, thickness=None, axis=2, cmap="magma", smooth=None
):
    """Render a log-density slab projection of a real field. Returns (fig, ax).

    Shows log10 of the slab-averaged 1 + delta, clipped at a small floor so empty
    regions do not blow up the log. Pass an existing `ax` to draw into a panel.
    `smooth` (sigma in cells), if set, applies a periodic Gaussian smoothing to
    the projected slab before the log -- a display-only convenience that tames
    the ~1-particle-per-cell discreteness texture without altering any physics.
    """
    img = _slab_image(delta, box, thickness=thickness, axis=axis)
    if smooth:
        from scipy.ndimage import gaussian_filter

        img = gaussian_filter(img, sigma=smooth, mode="wrap")
    img = np.log10(np.clip(img, 1e-2, None))
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    L = box.box_size
    im = ax.imshow(
        img.T, origin="lower", extent=[0, L, 0, L], cmap=cmap, aspect="equal"
    )
    ax.set_xlabel("[Mpc/h]")
    ax.set_ylabel("[Mpc/h]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\log_{10}(1+\delta)$")
    return fig, ax


def animate_slab(slabs, a_arr, box, out, fps=15, point_size=0.3, color="k", alpha=0.5):
    """Animate a sequence of 2D particle-slab projections to a gif.

    `slabs` is a list of (M, 2) arrays (e.g. SnapshotRecorder.slabs) and `a_arr`
    the matching scale factors. Watch the Zel'dovich sheet collapse into the
    cosmic web. Writes `out` (a .gif path) and returns it.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter([], [], s=point_size, c=color, alpha=alpha, linewidths=0)
    ax.set_xlim(0, box.box_size)
    ax.set_ylim(0, box.box_size)
    ax.set_aspect("equal")
    ax.set_xlabel("[Mpc/h]")
    ax.set_ylabel("[Mpc/h]")
    title = ax.set_title("")

    def draw(i):
        a = a_arr[i]
        scat.set_offsets(np.asarray(slabs[i]))
        title.set_text("PM leapfrog   z = %.2f   a = %.3f" % (1.0 / a - 1.0, a))
        return scat, title

    anim = animation.FuncAnimation(fig, draw, frames=len(slabs), blit=False)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim.save(out, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return out


def _panel_power(ax, box, cosmo, epochs, backend):
    """P(k) measured vs linear at one or more epochs (list of (label, z, field))."""
    for i, (label, z, delta) in enumerate(epochs):
        k, pk, _ = F.power_spectrum(delta, box)
        plin = C.linear_power(k, cosmo, z=z, backend=backend)
        c = "C%d" % i
        ax.loglog(k, pk, "o", ms=4, color=c, label="measured %s" % label)
        ax.loglog(k, plin, "-", color=c, lw=1.1, label="linear %s" % label)
    ax.axvline(box.k_nyquist, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel(r"$P(k)$ [$(\mathrm{Mpc}/h)^3$]")
    ax.set_title("power vs linear theory")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.2)


def _panel_growth(ax, box, cosmo, recorder):
    """Measured large-scale growth R(a) vs linear D(a)/D(a_init)."""
    a_arr, R = recorder.growth_history()
    Dref = D.linear_growth_reference(a_arr, cosmo)
    ax.plot(a_arr, Dref, "-", color="k", lw=1.5, label="linear D(a)/D(a_i)")
    ax.plot(a_arr, R, "o", ms=4, color="C3", label="measured (PM, large scales)")
    ax.set_xlabel("scale factor a")
    ax.set_ylabel("large-scale growth")
    ax.set_title("growth tracks linear theory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)


def _panel_cross(ax, box, ic_field, final_field):
    """Cross-correlation r(k) of the evolved field with the linear IC."""
    k, r, _ = D.cross_correlation(final_field, ic_field, box)
    ax.semilogx(k, r, "o-", ms=4, color="C0")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.axvline(box.k_nyquist, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel(r"$r(k)$ vs linear IC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("phase correlation with the IC")
    ax.grid(True, which="both", alpha=0.2)


def _panel_pdf(ax, fields):
    """One-point PDF(s) of delta/sigma on a log axis, skewness in the legend.

    `fields` is a list of (label, color, field). Local f_NL and nonlinear
    gravity both skew the tails, so the asymmetry between the IC and the evolved
    field is the visible non-Gaussianity.
    """
    for label, color, fld in fields:
        centers, density, _, sk = D.one_point_pdf(fld)
        ax.step(
            centers,
            density,
            where="mid",
            color=color,
            label="%s (skew %+.3f)" % (label, sk),
        )
    ax.set_yscale("log")
    ax.set_xlabel(r"$\delta / \sigma$")
    ax.set_ylabel("PDF")
    ax.set_title("one-point PDF: tails and skewness")
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.2)


def _panel_summary(ax, summary):
    """A monospace text panel for the run's config / metadata. `summary` is a
    preformatted string (e.g. SimConfig.summary())."""
    ax.set_axis_off()
    ax.text(
        0.0,
        1.0,
        summary,
        family="monospace",
        fontsize=7.5,
        va="top",
        ha="left",
        transform=ax.transAxes,
    )


def dashboard(
    box,
    cosmo,
    ic_field,
    final_field,
    recorder=None,
    z_init=9.0,
    z_final=0.0,
    out="outputs/dashboard.png",
    title=None,
    backend="camb",
    summary=None,
):
    """The standard six-panel run diagnostic, written to `out` (a .png path).

    Panels (2 x 3):
      (1) a log-density slice of the final field;
      (2) measured vs linear P(k) at z_init and z_final;
      (3) the one-point PDF of the IC and final fields, with skewness (the
          visible non-Gaussianity from gravity and any local f_NL);
      (4) the large-scale growth history vs linear D(a) (if a recording
          `recorder` is given, else a placeholder);
      (5) the cross-correlation r(k) of the final field with the linear IC;
      (6) a text summary of the run (the `summary` string, e.g.
          SimConfig.summary()), or empty if none is given.

    Each is a reusable diagnostic from mbody.diagnostics -- the same estimators
    the tests pin -- so the figure just composes them. Returns `out`.
    """
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    density_slice(final_field, box, ax=axes[0, 0])
    axes[0, 0].set_title("final density (z = %g)" % z_final)
    _panel_power(
        axes[0, 1],
        box,
        cosmo,
        [("z=%g" % z_init, z_init, ic_field), ("z=%g" % z_final, z_final, final_field)],
        backend,
    )
    _panel_pdf(
        axes[0, 2],
        [
            ("z=%g IC" % z_init, "C0", ic_field),
            ("z=%g final" % z_final, "C3", final_field),
        ],
    )
    if recorder is not None and recorder.record_growth:
        _panel_growth(axes[1, 0], box, cosmo, recorder)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "no growth history\n(run with a recorder)",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_axis_off()
    _panel_cross(axes[1, 1], box, ic_field, final_field)
    if summary:
        _panel_summary(axes[1, 2], summary)
    else:
        axes[1, 2].set_axis_off()
    if title:
        fig.suptitle(title, fontsize=12)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
