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
    numpy array. ``nbar`` is the target number density in (Mpc/h)^-3: a scalar
    for a uniform field, or a per-cell array the shape of ``delta_h`` (broadcast
    elementwise) for a position-dependent selection -- e.g. the radial selection
    function from ``radial_nbar_field``, which is zero outside the survey shell.
    ``V_cell = (L / n_mesh)^3``. Returns a float64 numpy array the same shape as
    ``delta_h``.
    """
    one_plus = 1.0 + np.asarray(delta_h, dtype=np.float64)
    cell_volume = (box.box_size / box.n_mesh) ** 3
    return np.asarray(nbar, dtype=np.float64) * cell_volume * np.maximum(one_plus, 0.0)


def radial_nbar_field(box, observer, profile, r_min=None, r_max=None):
    """Per-cell target number density for a radial selection about an observer.

    Builds an (n_mesh, n_mesh, n_mesh) array of the target ``nbar`` at each mesh
    cell, given an ``observer`` position in box coordinates (Mpc/h) and a radial
    selection ``profile`` n(r): the density of a survey shell that thins with
    comoving distance r = |cell_center - observer|. Pass it as the ``nbar``
    argument of ``intensity_field`` (or the ``nbar_field`` argument of the
    samplers) to draw a catalog with a realistic n(z)-driven radial selection
    instead of a uniform box.

    ``profile`` is either a scalar (constant density within the shell) or a
    callable ``n(r) -> density`` (e.g. a comoving-distance interpolation of a
    forecast n(z), built by the caller -- the survey-specific n(z) stays out of
    mbody). Cells with r outside ``[r_min, r_max]`` (when given) are set to zero,
    so the shell is carved out of the periodic box; choose the box so r_max is
    within its inscribed sphere (r_max < L/2) to avoid the periodic copies
    leaking into the shell. Returns a float64 array.
    """
    N = box.n_mesh
    d = box.box_size / N
    centers = (np.arange(N) + 0.5) * d
    ox, oy, oz = (float(o) for o in observer)
    # r via 1-D broadcasting (one N^3 array), not a full meshgrid stack.
    r = np.sqrt(
        ((centers - ox) ** 2)[:, None, None]
        + ((centers - oy) ** 2)[None, :, None]
        + ((centers - oz) ** 2)[None, None, :]
    )
    nbar = profile(r) if callable(profile) else np.full(r.shape, float(profile))
    nbar = np.asarray(nbar, dtype=np.float64)
    if r_min is not None:
        nbar[r < r_min] = 0.0
    if r_max is not None:
        nbar[r > r_max] = 0.0
    return nbar


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
    a constant observer ``origin`` is then added. Returns an (n_gal, 3) float32
    array of box-frame comoving positions (the inputs -- particle positions and
    the box -- are float32, and positions span only ``[0, L)``, so float32 carries
    the cell to sub-micro-Mpc/h; a large observer ``origin`` is the one case where
    its sub-cell offset would be lost, and the survey geometry adds that shift
    downstream from origin = 0).

    Memory- and allocation-lean for the survey-scale counts (~10^9 galaxies): a
    single per-galaxy index array is repeated only over *occupied* cells, the
    sub-cell offsets are drawn in place, and the integer cell corner is decoded
    in place per axis -- so the working set is ~28 bytes/galaxy rather than the
    ~80+ of building, stacking and up-casting separate (n_gal, 3) blocks. The
    draw order is deterministic given ``rng``.
    """
    N = box.n_mesh
    d = box.box_size / N
    counts = np.asarray(counts).astype(np.int64).reshape(-1)
    n_gal = int(counts.sum())
    if n_gal == 0:
        return np.empty((0, 3), dtype=np.float32)
    # The one unavoidable per-galaxy array: the flat cell index of each galaxy,
    # repeated over occupied cells only (a no-op when every cell is populated,
    # a big saving when the field is sparse, e.g. the low-nbar high-z shells).
    occ = np.flatnonzero(counts)
    flat_cell = np.repeat(occ, counts[occ])

    xyz = np.empty((n_gal, 3), dtype=np.float32)
    rng.random(dtype=np.float32, out=xyz)  # uniform [0, 1) sub-cell, in place
    _decode_cells_inplace(xyz, flat_cell, N)  # add the integer cell corner
    del flat_cell

    xyz *= d
    xyz += np.asarray(origin, dtype=np.float32)
    return xyz


def _decode_cells_inplace(xyz, flat_cell, N):
    """Add the integer cell corner (i, j, k) of each flat cell index to ``xyz``.

    ``xyz`` already holds the sub-cell offsets; this adds (i, j, k) per axis,
    decoding the flat index (k fastest) in place so only one int column-width
    temporary is ever live. Modifies ``xyz`` and consumes ``flat_cell``.
    """
    np.add(xyz[:, 2], flat_cell % N, out=xyz[:, 2], casting="unsafe")
    np.floor_divide(flat_cell, N, out=flat_cell)
    np.add(xyz[:, 1], flat_cell % N, out=xyz[:, 1], casting="unsafe")
    np.floor_divide(flat_cell, N, out=flat_cell)
    np.add(xyz[:, 0], flat_cell, out=xyz[:, 0], casting="unsafe")


def place_galaxies_chunked(counts, box, rng, origin=(0.0, 0.0, 0.0), chunk_cells=None):
    """Stream sub-cell galaxy placement in cell-slab chunks (bounded memory).

    A generator over contiguous blocks of ``chunk_cells`` mesh cells (flat,
    k-fastest order); each iteration yields the (n_chunk, 3) float32 positions of
    the galaxies in that block, placed exactly as ``place_galaxies`` does. Empty
    blocks are skipped without consuming the RNG, so iterating the whole grid and
    concatenating reproduces ``place_galaxies`` *bit for bit* (the uniform draw is
    consumed in the same galaxy order) -- the streamed catalog is identical to the
    in-memory one. This just caps the working set at one chunk rather than the full
    (n_gal, 3) block, which is ~10^9 rows for a survey shell.

    ``chunk_cells`` defaults to one mesh slab (n_mesh^2 cells, ~1/n_mesh of the
    galaxies per chunk).
    """
    N = box.n_mesh
    d = box.box_size / N
    counts = np.asarray(counts).astype(np.int64).reshape(-1)
    ncell = counts.size
    if chunk_cells is None:
        chunk_cells = N * N
    chunk_cells = max(1, int(chunk_cells))
    origin = np.asarray(origin, dtype=np.float32)
    for c0 in range(0, ncell, chunk_cells):
        sub = counts[c0 : c0 + chunk_cells]
        occ = np.flatnonzero(sub)
        if occ.size == 0:
            continue
        flat_cell = np.repeat(occ + c0, sub[occ])
        xyz = np.empty((flat_cell.size, 3), dtype=np.float32)
        rng.random(dtype=np.float32, out=xyz)  # uniform [0, 1) sub-cell, in place
        _decode_cells_inplace(xyz, flat_cell, N)
        del flat_cell
        xyz *= d
        xyz += origin
        yield xyz


@dataclass
class Catalog:
    """A sampled galaxy mock catalog: positions plus provenance.

    ``xyz`` is an (n_gal, 3) float32 array of comoving positions in Mpc/h
    (observer-centered if ``origin`` was nonzero); ``write_parquet`` up-casts to
    float64 on disk to match the consumer schema. The remaining fields record how
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


def _counts_from_delta(delta_h, box, sampling, nbar_field):
    """Clipped-Poisson per-cell counts from a precomputed tracer overdensity field.

    The shared back half of the samplers: given a tracer density contrast ``delta_h``
    (Eulerian local bias, or the advected Lagrangian-bias weighted paint), build the
    clipped intensity, draw per-cell Poisson counts, and report the realized density +
    clip fraction over the *selected* (nbar > 0) volume. ``nbar_field`` (an (N, N, N)
    array, e.g. from ``radial_nbar_field``) overrides the scalar ``sampling.nbar``.
    Returns ``(counts, rng, realized_nbar, clip_fraction)`` -- the ``rng`` is returned
    mid stream so the caller continues the same draw for placement (reproducibility).
    """
    if sampling.sampler != "poisson":
        raise NotImplementedError(
            f"sampler={sampling.sampler!r} is reserved but not implemented"
        )
    if sampling.nonneg != "clip":
        raise NotImplementedError(
            f"nonneg={sampling.nonneg!r} is reserved but not implemented"
        )

    delta_h = np.asarray(delta_h, dtype=np.float64)
    if nbar_field is None:
        nbar = sampling.nbar
        selected = None
        v_sel = box.box_size**3
    else:
        nbar = np.asarray(nbar_field, dtype=np.float64)
        selected = nbar > 0.0
        v_sel = float(np.count_nonzero(selected)) * (box.box_size / box.n_mesh) ** 3
    lam = intensity_field(delta_h, nbar, box)
    one_plus = 1.0 + delta_h
    clip_cells = one_plus < 0.0 if selected is None else (one_plus < 0.0) & selected
    denom = delta_h.size if selected is None else int(np.count_nonzero(selected))
    clip_fraction = float(np.count_nonzero(clip_cells) / denom) if denom else 0.0
    realized_nbar = float(lam.sum() / v_sel) if v_sel > 0 else 0.0

    rng = np.random.default_rng(sampling.draw_seed)
    counts = poisson_counts(lam, rng)
    return counts, rng, realized_nbar, clip_fraction


def _counts_from_positions(positions, box, tracer, sampling, nbar_field):
    """Paint -> Eulerian bias -> clipped Poisson counts (the position-sampling front).

    Paints ``positions`` to a plain-CIC density contrast and applies the Eulerian
    local quadratic bias, then defers to ``_counts_from_delta``. The Lagrangian-bias
    path skips this and feeds its weighted-paint field straight to ``_counts_from_delta``.
    """
    delta = PA.density_contrast(positions, box)
    delta_h = B.local_bias_tracer(delta, tracer.b1, tracer.b2)
    return _counts_from_delta(delta_h, box, sampling, nbar_field)


def sample_from_positions(
    positions, box, tracer, sampling, z=0.0, f_NL=0.0, ic_seed=0, nbar_field=None
):
    """Sample a galaxy catalog from final particle positions.

    Paints ``positions`` to a plain-CIC density contrast, applies the local
    quadratic bias (``tracer.b1``, ``tracer.b2``) to form the tracer field
    delta_h, builds the clipped Poisson intensity ``nbar (1 + delta_h)``, draws
    per-cell counts, and scatters galaxies uniformly within cells. Forward-only:
    the Poisson/placement draw uses ``numpy.random.default_rng(sampling.draw_seed)``
    -- a stochastic axis distinct from the IC phase seed. ``nbar_field`` (an
    (N, N, N) per-cell target density, e.g. from ``radial_nbar_field``) overrides
    the scalar ``sampling.nbar`` for a radial / n(z)-driven selection. ``z``,
    ``f_NL`` and ``ic_seed`` are recorded as provenance only. Returns a ``Catalog``.
    """
    counts, rng, realized_nbar, clip_fraction = _counts_from_positions(
        positions, box, tracer, sampling, nbar_field
    )
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


def sample_from_field(
    delta_g, box, sampling, z=0.0, f_NL=0.0, ic_seed=0, nbar_field=None
):
    """Sample a galaxy catalog from a precomputed tracer overdensity field.

    Like ``sample_from_positions`` but takes the tracer density contrast ``delta_g``
    directly -- e.g. the advected Lagrangian-bias weighted-paint field -- instead of
    re-deriving it from positions via plain CIC + the Eulerian local bias. Builds the
    clipped Poisson intensity ``nbar (1 + delta_g)``, draws per-cell counts, and
    scatters galaxies uniformly within cells. ``nbar_field`` (an (N, N, N) per-cell
    target density, e.g. from ``radial_nbar_field``) overrides the scalar
    ``sampling.nbar`` for a radial selection. ``z``, ``f_NL`` and ``ic_seed`` are
    recorded as provenance only. Returns a ``Catalog``.
    """
    counts, rng, realized_nbar, clip_fraction = _counts_from_delta(
        delta_g, box, sampling, nbar_field
    )
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


def sample_to_parquet(
    positions,
    box,
    tracer,
    sampling,
    path,
    chunk_cells=None,
    overwrite=False,
    z=0.0,
    f_NL=0.0,
    ic_seed=0,
    nbar_field=None,
):
    """Sample a galaxy catalog from positions and *stream* it to parquet.

    The memory-bounded sibling of ``sample_from_positions`` followed by
    ``Catalog.write_parquet``: it paints the positions, applies the bias, and
    draws the per-cell Poisson counts as usual, but then streams the placement
    straight to a parquet file in cell-slab chunks (``place_galaxies_chunked``) --
    so the full (n_gal, 3) catalog is never held in memory. For a survey shell
    (~10^9 galaxies) the in-memory path allocates tens of GB for the position
    block; here the working set is one chunk (~one mesh slab). The on-disk schema
    is identical to ``Catalog.write_parquet`` (x, y, z float64 + the same metadata
    + an optional int8 ``bin`` column), and the catalog is bit-identical to the
    in-memory path for the same ``sampling.draw_seed``. ``nbar_field`` (an
    (N, N, N) per-cell target density, e.g. from ``radial_nbar_field``) overrides
    the scalar ``sampling.nbar`` for a radial / n(z)-driven selection.

    Returns a provenance dict (no positions): ``path``, ``n_galaxies``,
    ``nbar_target``, ``realized_nbar``, ``clip_fraction``. With ``overwrite=False``
    (default) an existing file is left untouched and ``{"path", "skipped": True}``
    is returned.
    """
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    path = str(path)
    if os.path.exists(path) and not overwrite:
        return {"path": path, "skipped": True}

    counts, rng, realized_nbar, clip_fraction = _counts_from_positions(
        positions, box, tracer, sampling, nbar_field
    )

    metadata = {
        "box_size_x": str(box.box_size),
        "box_size_y": str(box.box_size),
        "box_size_z": str(box.box_size),
        "generator": "mbody",
        "z_eff": str(float(z)),
        "nbar_target": str(float(sampling.nbar)),
        "realized_nbar": str(realized_nbar),
        "clip_fraction": str(clip_fraction),
        "f_NL": str(float(f_NL)),
        "ic_seed": str(int(ic_seed)),
        "draw_seed": str(int(sampling.draw_seed)),
        "bin_index": str(int(sampling.bin_index)),
    }
    names = ["x", "y", "z"]
    fields = [pa.field(n, pa.float64()) for n in names]
    if sampling.bin_index >= 0:
        fields.append(pa.field("bin", pa.int8()))
    schema = pa.schema(fields, metadata=metadata)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    written = 0
    writer = pq.ParquetWriter(path, schema)
    try:
        for chunk in place_galaxies_chunked(
            counts, box, rng, origin=sampling.origin, chunk_cells=chunk_cells
        ):
            n = chunk.shape[0]
            arrays = [
                pa.array(np.ascontiguousarray(chunk[:, c], np.float64))
                for c in range(3)
            ]
            if sampling.bin_index >= 0:
                arrays.append(
                    pa.array(np.full(n, sampling.bin_index, np.int8), pa.int8())
                )
            writer.write_table(pa.table(arrays, schema=schema))
            written += n
    finally:
        writer.close()

    return {
        "path": path,
        "n_galaxies": written,
        "nbar_target": float(sampling.nbar),
        "realized_nbar": realized_nbar,
        "clip_fraction": clip_fraction,
    }


# --- Lightcone (radially-evolving bias + growth, observer-centered) -----------

DELTA_C = 1.686  # spherical-collapse threshold, for b_phi = 2 delta_c (b1 - p)


def lightcone_catalog(
    box,
    cosmo,
    time,
    observer,
    z_edges,
    bias_fn,
    nbar_profile,
    sampling,
    f_NL=0.0,
    seed=0,
    p=0.55,
    b2_fn=None,
    backend="camb",
    lpt_order=2,
):
    """Observer-centered lightcone galaxy catalog with radially-evolving bias + growth.

    Runs the PM forward ONCE, capturing a snapshot ladder, then builds an *onion*
    lightcone: each redshift shell [z_edges[s], z_edges[s+1]] is drawn from the
    snapshot whose scale factor is nearest a(z_mid), so the clustering carries the
    growth D(z(r)) of that shell rather than a single effective redshift. The tracer
    is the Lagrangian-bias field of that snapshot (bias.lagrangian_bias_field): the
    weights 1 + delta_g^L(q) are advected and CIC-painted, with the shell's bias
    b1 = bias_fn(z_mid) and b_phi = 2 delta_c (b1 - p) set DIRECTLY (separate-universe),
    so the injected f_NL scale-dependent bias is undiluted and EVOLVES along the line
    of sight. Each shell is carved by ``radial_nbar_field`` (the selection n(r)) and
    Poisson-sampled with an independent draw, then the shells are concatenated.

    Parameters
    ----------
    box, cosmo, time : the usual config objects; ``time`` runs from z_init to the
        lowest shell edge, and its n_steps sets the snapshot-ladder density.
    observer : (3,) position in box coordinates (Mpc/h); galaxies are output relative
        to it (so the comoving distance is r = |xyz|). Use the box centre for a
        full-sky shell, with the outer shell inside the inscribed sphere (chi < L/2).
    z_edges : increasing redshift shell edges; comoving via cosmology.comoving_distance.
    bias_fn : callable z -> Eulerian linear bias b1(z). The Lagrangian b1 is b1(z) - 1
        (advection of the uniform weight adds the +1).
    nbar_profile : the radial selection n(r) (callable of comoving distance, or scalar)
        passed to ``radial_nbar_field``.
    sampling : a ``CatalogSampling``; ``draw_seed`` is offset per shell for independent
        shot noise, and ``origin`` is set to -observer (output is observer-centered).
    b2_fn : optional callable z -> Lagrangian b2(z) (default 0; b1 + b_phi is the core).

    Returns a ``Catalog`` (concatenated over shells; per-shell realized nbar / clip are
    summarized). For survey-scale RAM, stream per shell instead (a future sibling).
    """
    import mlx.core as mx
    from dataclasses import replace

    from mbody import cosmology as C
    from mbody import ic as IC
    from mbody import integrate as IG

    z_edges = np.asarray(z_edges, dtype=np.float64)
    z_mid = 0.5 * (z_edges[1:] + z_edges[:-1])
    chi_edges = C.comoving_distance(z_edges, cosmo)
    observer = np.asarray(observer, dtype=np.float32)
    if float(chi_edges.max()) >= 0.5 * box.box_size:
        raise ValueError(
            f"outer shell chi={float(chi_edges.max()):.0f} exceeds the inscribed "
            f"sphere L/2={0.5 * box.box_size:.0f} (periodic copies would leak in)"
        )

    snaps = []  # (a, positions) captured along the single forward run

    def _record(step, a, x, p_):
        snaps.append((float(a), np.asarray(x, dtype=np.float32)))

    x0, p0 = IG.initial_state(
        box, cosmo, time, seed=seed, f_NL=f_NL, backend=backend, lpt_order=lpt_order
    )
    IG.evolve_state(
        x0,
        p0,
        box,
        cosmo,
        IG.a_grid(time),
        snapshot=_record,
        integrator=time.integrator,
    )
    a_snaps = np.array([a for a, _ in snaps])

    origin = tuple(float(-o) for o in observer)
    xyz_parts, nbar_real, clip_fracs = [], [], []
    for s in range(len(z_mid)):
        zc = float(z_mid[s])
        si = int(np.argmin(np.abs(a_snaps - 1.0 / (1.0 + zc))))
        b1 = float(bias_fn(zc))
        b_phi = 2.0 * DELTA_C * (b1 - p)
        b2 = float(b2_fn(zc)) if b2_fn is not None else 0.0
        delta1 = IC.linear_density(
            box, cosmo, seed=seed, z=zc, f_NL=f_NL, backend=backend
        )
        phi_G = IC.primordial_potential(box, cosmo, seed=seed, z=zc, backend=backend)
        dgL = B.lagrangian_bias_field(
            delta1, box, b1=b1 - 1.0, b2=b2, b_phi=b_phi, f_NL=f_NL, phi_G=phi_G
        )
        w = (1.0 + dgL).reshape(-1)
        n_g = PA.cic_paint(mx.array(snaps[si][1]), box, weights=w)
        mean_w = float(w.sum()) / box.n_mesh**3
        delta_g = np.asarray(n_g, dtype=np.float64) / mean_w - 1.0
        nbar_field = radial_nbar_field(
            box,
            observer,
            nbar_profile,
            r_min=float(chi_edges[s]),
            r_max=float(chi_edges[s + 1]),
        )
        shell_sampling = replace(sampling, draw_seed=sampling.draw_seed + s)
        counts, rng, rn, cf = _counts_from_delta(
            delta_g, box, shell_sampling, nbar_field
        )
        xyz_parts.append(place_galaxies(counts, box, rng, origin=origin))
        nbar_real.append(rn)
        clip_fracs.append(cf)

    xyz = np.concatenate(xyz_parts) if xyz_parts else np.empty((0, 3), dtype=np.float32)
    return Catalog(
        xyz=xyz,
        box_size=float(box.box_size),
        z_eff=float(z_mid.mean()),
        nbar_target=float(sampling.nbar),
        realized_nbar=float(np.mean(nbar_real)) if nbar_real else 0.0,
        clip_fraction=float(np.max(clip_fracs)) if clip_fracs else 0.0,
        bin_index=-1,
        f_NL=float(f_NL),
        ic_seed=int(seed),
        draw_seed=int(sampling.draw_seed),
    )
