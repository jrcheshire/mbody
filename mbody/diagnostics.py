"""Shared diagnostics for M-body runs: power, cross-correlation, growth history.

Everything here is read-only *measurement* -- it consumes fields and particle
states and returns numpy / plain-Python summaries, reusing the estimators in
mbody.fields rather than re-deriving them per script. None of it is on the hot
or autodiff path.

The SnapshotRecorder is the optional, memory-gated seam that the integrator's
``snapshot(step, a, x, p)`` callback feeds. It stores only bounded *reductions*
of each step -- a 2D slab projection for animation and a handful of low-|k|
Fourier modes for the growth history -- never the full (n_particles, 3) state,
so a long, high-resolution run stays cheap. Build one of these on a forward run,
then hand it (or a final field) to mbody.viz to render a dashboard or animation.
"""

import mlx.core as mx
import numpy as np

from mbody import cosmology as C
from mbody import fields as F
from mbody import painting as PA


def _bin_edges(box, dk, kmin, kmax):
    """|k| bin edges matching fields.power_spectrum (first bin excludes k=0)."""
    kf = box.k_fundamental
    if dk is None:
        dk = kf
    if kmin is None:
        kmin = 0.5 * kf
    if kmax is None:
        kmax = box.k_nyquist
    return np.arange(kmin, kmax + dk, dk)


def _bin(weights, km, edges):
    """Sum `weights` and count modes in each |k| shell. Returns (sums, counts)."""
    sums, _ = np.histogram(km, bins=edges, weights=weights)
    counts, _ = np.histogram(km, bins=edges)
    return sums, counts


def cross_correlation(delta_a, delta_b, box, dk=None, kmin=None, kmax=None):
    """Cross-correlation coefficient r(k) of two real fields.

    r(k) = P_ab(k) / sqrt(P_aa(k) P_bb(k)), binned in spherical |k| shells with
    the same edges as fields.power_spectrum. r lies in [-1, 1]; it is 1 for two
    fields that differ only by a (k-independent) amplitude -- the right way to
    ask "do the evolved phases still track the linear initial conditions?",
    independent of how much the amplitude has grown. Returns (k_centers, r,
    n_modes) as float64 numpy arrays, excluding the k = 0 mode.
    """
    N, L = box.n_mesh, box.box_size
    ak = mx.fft.rfftn(delta_a)
    bk = mx.fft.rfftn(delta_b)
    norm = L**3 / N**6
    paa = norm * np.asarray(mx.abs(ak) ** 2, dtype=np.float64).ravel()
    pbb = norm * np.asarray(mx.abs(bk) ** 2, dtype=np.float64).ravel()
    pab = norm * np.asarray(mx.real(ak * mx.conj(bk)), dtype=np.float64).ravel()

    _, _, k_mag = F.k_grid(box)
    km = k_mag.ravel()
    edges = _bin_edges(box, dk, kmin, kmax)
    saa, counts = _bin(paa, km, edges)
    sbb, _ = _bin(pbb, km, edges)
    sab, _ = _bin(pab, km, edges)

    centers = 0.5 * (edges[1:] + edges[:-1])
    good = counts > 0
    r = sab[good] / np.sqrt(saa[good] * sbb[good])
    return centers[good], r, counts[good]


def particle_power(positions, box, interlace=True, **kwargs):
    """P(k) of a particle set: CIC-paint to a density contrast, then measure.

    Thin convenience over the measurement painter + fields.power_spectrum so a
    run's measured spectrum comes from one call. By default the field is painted
    with interlacing (fields.interlaced_density_contrast) to suppress the CIC
    mass-assignment aliasing near Nyquist; pass interlace=False for the plain CIC
    field. Extra keywords (dk, kmin, kmax, deconvolve_cic) pass through to
    power_spectrum. Returns (k_centers, P_k, n_modes).
    """
    delta = (
        F.interlaced_density_contrast(positions, box)
        if interlace
        else PA.density_contrast(positions, box)
    )
    return F.power_spectrum(delta, box, **kwargs)


def linear_growth_reference(a_arr, cosmo):
    """Linear growth D(a)/D(a[0]) over a sequence of scale factors.

    The curve a snapshot growth history should track on large scales. Returns a
    float64 numpy array normalized to 1 at the first entry.
    """
    a_arr = np.asarray(a_arr, dtype=np.float64)
    z = 1.0 / a_arr - 1.0
    D = np.array([C.growth_factor(float(zi), cosmo) for zi in z])
    return D / D[0]


def growth_amplitude(modes_seq):
    """Large-scale growth ratio of a sequence of Fourier-mode snapshots.

    R[i] = Re<m_i m_0*> / <|m_0|^2>, the amplitude of snapshot i projected onto
    the initial low-k modes m_0. For a pure rescaling m_i = s_i m_0 this returns
    s_i exactly, so it isolates growth from the (fixed) phases. Returns a float64
    numpy array with R[0] = 1.
    """
    ref = modes_seq[0]
    denom = np.sum(np.abs(ref) ** 2)
    return np.array(
        [float(np.real(np.sum(m * np.conj(ref)) / denom)) for m in modes_seq]
    )


def skewness(field):
    """Reduced third moment <d^3> / <d^2>^1.5 of a real field.

    Zero for a Gaussian field; positive as gravity (or positive local f_NL)
    skews the density toward rare dense peaks. The single number behind the
    one-point PDF's asymmetry.
    """
    d = np.asarray(field, dtype=np.float64).ravel()
    d = d - d.mean()
    return float(np.mean(d**3) / np.mean(d**2) ** 1.5)


def one_point_pdf(field, bins=120, span=6.0):
    """One-point PDF of the standardized field delta / sigma.

    Histograms delta / sigma over [-span, span] as a normalized density, so
    fields of different amplitude overlay and the shape (tails, asymmetry) is
    what shows. Returns (centers, density, sigma, skew): the standardized PDF
    plus the unstandardized rms and skewness summaries.
    """
    d = np.asarray(field, dtype=np.float64).ravel()
    sigma = d.std()
    sk = skewness(d)
    density, edges = np.histogram(
        d / sigma, bins=bins, range=(-span, span), density=True
    )
    centers = 0.5 * (edges[1:] + edges[:-1])
    return centers, density, sigma, sk


class SnapshotRecorder:
    """Capture a PM trajectory off the autodiff path, as bounded reductions.

    A callable matching the integrator's ``snapshot(step, a, x, p)`` seam. It
    stores per step only:

    - `slabs`: a 2D projection of a thin Lagrangian slab of particles (for the
      structure-formation animation), of order n_mesh^2 * slab_thick points;
    - low-|k| density modes (for `growth_history`).

    It never keeps the full (n_particles, 3) state, so the memory cost is a small
    multiple of one mesh slice per frame -- cheap even for an 80-step, 128^3 run.
    Nothing here touches the forward physics or the gradient; attach it only to
    forward runs (pass snapshot=recorder to integrate.leapfrog).
    """

    def __init__(
        self,
        box,
        slab_thick=4,
        slab_axis=2,
        growth_kmax=0.05,
        record_slab=True,
        record_growth=True,
    ):
        if slab_axis not in (0, 1, 2):
            raise ValueError(f"slab_axis must be 0, 1 or 2 (got {slab_axis})")
        self.box = box
        self.slab_thick = slab_thick
        self.slab_axis = slab_axis
        self.record_slab = record_slab
        self.record_growth = record_growth
        self.steps = []
        self.a = []
        self.slabs = []
        self._modes = []
        if record_growth:
            _, _, k_mag = F.k_grid(box)
            self._low_k = (k_mag > 0) & (k_mag < growth_kmax)

    def __call__(self, step, a, x, p):
        N = self.box.n_mesh
        self.steps.append(int(step))
        self.a.append(float(a))
        if self.record_slab:
            xn = np.asarray(x, dtype=np.float32).reshape(N, N, N, 3)
            sl = [slice(None)] * 3
            sl[self.slab_axis] = slice(0, self.slab_thick)
            slab = xn[tuple(sl)].reshape(-1, 3)
            keep = [i for i in range(3) if i != self.slab_axis]
            self.slabs.append(slab[:, keep].copy())
        if self.record_growth:
            dk = np.asarray(mx.fft.rfftn(PA.density_contrast(x, self.box)))
            self._modes.append(dk[self._low_k].copy())

    def growth_history(self):
        """(a, R): large-scale density growth across the run.

        R(a) is the recorded field's amplitude projected onto the initial low-k
        modes (see growth_amplitude), to overlay on linear_growth_reference. R[0]
        is 1 by construction. Requires record_growth=True.
        """
        if not self.record_growth:
            raise ValueError("recorder was built with record_growth=False")
        return np.array(self.a), growth_amplitude(self._modes)
