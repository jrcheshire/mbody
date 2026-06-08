"""Forward-only galaxy mock catalogs by field-level Poisson sampling.

M-body's forward model evolves matter; this module turns the evolved field into a
discrete galaxy catalog the way approximate-mock pipelines (EZmock, PATCHY, ...)
do: apply a deterministic bias to the density field to get a mean galaxy
number-density, then Poisson-sample an integer count per mesh cell and scatter
galaxies inside the cells. There is no halo finding -- the bias is applied at the
field level (mbody.bias.local_bias_tracer), keeping the generator differentiable
in spirit and cheap. The draw itself is a forward-only Monte Carlo (numpy), so it
is off the autodiff path and not bound by the adjoint memory ceiling.

The intended use is a *systematics-validation* mock: a drop-in replacement for the
f_NL-free approximate mocks fed to a field-level null test. For f_NL the advantage
over a lognormal mock is that the scale-dependent bias arises from the dynamics
(local-f_NL initial conditions evolved forward), not from a hand-imposed ansatz.
Production f_NL *inference* is a separate code path (e.g. DISCO-DJ).

Non-negativity. A local quadratic bias delta_h = b1 delta + (b2/2)(delta^2 -
<delta^2>) can dip below -1 in deep voids, which would make the Poisson intensity
nbar (1 + delta_h) negative. We *clip* 1 + delta_h at 0 (the option that preserves
the perturbative bias and bispectrum that motivate using M-body over a lognormal
field). Clipping slightly inflates the realized mean density above the target;
that drift is reported (realized_nbar, clip_fraction), not renormalized, because
rescaling the intensity would distort the large-scale modes that carry the f_NL
signal.
"""

from dataclasses import dataclass

import numpy as np

from mbody import bias as B
from mbody import painting as PA


def intensity_field(delta_h, nbar, box):
    """Mean galaxy count per mesh cell from a biased density contrast.

    ``lam(x) = nbar * V_cell * max(1 + delta_h(x), 0)`` -- the clipped
    non-negativity map (deep voids where ``1 + delta_h < 0`` are floored to zero
    rather than a negative, unphysical intensity). ``delta_h`` is the tracer
    density contrast (e.g. from ``bias.local_bias_tracer``); it may be an mx or
    numpy array. ``V_cell = (L / n_mesh)^3``. Returns a float64 numpy array the
    same shape as ``delta_h``.
    """
    one_plus = 1.0 + np.asarray(delta_h, dtype=np.float64)
    cell_volume = (box.box_size / box.n_mesh) ** 3
    return nbar * cell_volume * np.maximum(one_plus, 0.0)


def poisson_counts(lam, rng):
    """Poisson-draw an integer galaxy count per cell from the intensity ``lam``.

    ``rng`` is a ``numpy.random.Generator`` (the forward-only, non-differentiable
    shot-noise draw). Returns an int64 array the same shape as ``lam``.
    """
    return rng.poisson(np.asarray(lam, dtype=np.float64))


def place_galaxies(counts, box, rng, origin=(0.0, 0.0, 0.0)):
    """Place galaxies uniformly inside each mesh cell given per-cell ``counts``.

    Each cell (i, j, k) gets ``counts[i, j, k]`` galaxies at its lower corner
    ``(i, j, k) * cell_size`` plus a uniform ``[0, cell_size)^3`` sub-cell offset;
    a constant observer ``origin`` is then added. Returns an (n_gal, 3) float64
    array. Vectorized over all galaxies; the draw order (one uniform block) is
    deterministic given ``rng``.
    """
    N = box.n_mesh
    d = box.box_size / N
    counts = np.asarray(counts).astype(np.int64).reshape(N, N, N)
    flat = counts.ravel()
    n_gal = int(flat.sum())
    if n_gal == 0:
        return np.empty((0, 3), dtype=np.float64)
    cell = np.repeat(np.arange(N * N * N), flat)
    ijk = np.stack(np.unravel_index(cell, (N, N, N)), axis=1).astype(np.float64)
    xyz = ijk * d + rng.uniform(0.0, d, size=(n_gal, 3))
    xyz += np.asarray(origin, dtype=np.float64)
    return xyz


@dataclass
class Catalog:
    """A sampled galaxy mock catalog: positions plus provenance.

    ``xyz`` is an (n_gal, 3) float64 array of comoving positions in Mpc/h
    (observer-centered if ``origin`` was nonzero). The remaining fields record how
    the catalog was made -- the box, the target and realized number density, the
    clip fraction, the redshift, and the IC/draw seeds -- so a catalog is
    reproducible and self-describing. ``write_parquet`` emits a simple parquet
    schema (x/y/z columns plus box-size metadata) for a downstream pipeline.
    """

    xyz: object
    box_size: float
    z_eff: float = 0.0
    nbar_target: float = 0.0
    realized_nbar: float = 0.0
    clip_fraction: float = 0.0
    bin_index: int = -1
    f_NL: float = 0.0
    ic_seed: int = 0
    draw_seed: int = 0

    @property
    def n_galaxies(self):
        """Number of galaxies in the catalog."""
        return int(np.asarray(self.xyz).shape[0])

    def write_parquet(self, path, overwrite=False):
        """Write the catalog to parquet (x/y/z columns + box-size metadata).

        Columns ``x, y, z`` (float64, Mpc/h); the box size lives in file-level
        metadata keys ``box_size_x/y/z`` (where the consumer reads it), alongside
        mbody provenance keys; an int8 ``bin`` column is added when
        ``bin_index >= 0``. Parent directories are created. Returns the path; with
        ``overwrite=False`` (default) an existing file is left untouched.
        """
        import os

        import pyarrow as pa
        import pyarrow.parquet as pq

        path = str(path)
        if os.path.exists(path) and not overwrite:
            return path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        xyz = np.ascontiguousarray(self.xyz, dtype=np.float64)
        table = pa.table({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})
        if self.bin_index >= 0:
            bin_col = pa.array(
                np.full(self.n_galaxies, self.bin_index, dtype=np.int8), pa.int8()
            )
            table = table.append_column("bin", bin_col)

        L = str(self.box_size)
        table = table.replace_schema_metadata(
            {
                "box_size_x": L,
                "box_size_y": L,
                "box_size_z": L,
                "generator": "mbody",
                "z_eff": str(self.z_eff),
                "nbar_target": str(self.nbar_target),
                "realized_nbar": str(self.realized_nbar),
                "clip_fraction": str(self.clip_fraction),
                "f_NL": str(self.f_NL),
                "ic_seed": str(self.ic_seed),
                "draw_seed": str(self.draw_seed),
                "bin_index": str(self.bin_index),
            }
        )
        pq.write_table(table, path)
        return path


def sample_from_positions(positions, box, tracer, sampling, z=0.0, f_NL=0.0, ic_seed=0):
    """Sample a galaxy catalog from final particle positions.

    Paints ``positions`` to a plain-CIC density contrast, applies the local
    quadratic bias (``tracer.b1``, ``tracer.b2``) to form the tracer field
    delta_h, builds the clipped Poisson intensity ``nbar (1 + delta_h)``, draws
    per-cell counts, and scatters galaxies uniformly within cells. Forward-only:
    the Poisson/placement draw uses ``numpy.random.default_rng(sampling.draw_seed)``
    -- a stochastic axis distinct from the IC phase seed. ``z``, ``f_NL`` and
    ``ic_seed`` are recorded as provenance only. Returns a ``Catalog``.
    """
    if sampling.sampler != "poisson":
        raise NotImplementedError(
            f"sampler={sampling.sampler!r} is reserved but not implemented"
        )
    if sampling.nonneg != "clip":
        raise NotImplementedError(
            f"nonneg={sampling.nonneg!r} is reserved but not implemented"
        )

    delta = PA.density_contrast(positions, box)
    delta_h = np.asarray(B.local_bias_tracer(delta, tracer.b1, tracer.b2), np.float64)
    lam = intensity_field(delta_h, sampling.nbar, box)
    clip_fraction = float(np.mean((1.0 + delta_h) < 0.0))
    realized_nbar = float(lam.sum() / box.box_size**3)

    rng = np.random.default_rng(sampling.draw_seed)
    counts = poisson_counts(lam, rng)
    xyz = place_galaxies(counts, box, rng, origin=sampling.origin)

    return Catalog(
        xyz=xyz,
        box_size=float(box.box_size),
        z_eff=float(z),
        nbar_target=float(sampling.nbar),
        realized_nbar=realized_nbar,
        clip_fraction=clip_fraction,
        bin_index=int(sampling.bin_index),
        f_NL=float(f_NL),
        ic_seed=int(ic_seed),
        draw_seed=int(sampling.draw_seed),
    )
