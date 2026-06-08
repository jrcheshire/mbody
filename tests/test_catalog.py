"""Tests for the forward-only galaxy mock catalog (mbody.catalog) -- Stage A.

Unit-level: the clip non-negativity map keeps the Poisson intensity >= 0; the
draw places exactly the right number of galaxies, each inside its mesh cell (mass
conservation); the draw is reproducible from its seed; and the parquet writer
emits a simple schema (x, y, z columns plus box-size metadata, read here the same
way a downstream null-test pipeline does).

These are unit/contract tests; the statistical validation (nbar / b1 / f_NL
recovery, with tolerances measured from a probe) is Stage B/C.
"""

import dataclasses

import numpy as np
import mlx.core as mx
import pyarrow.parquet as pq
import pytest

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

SMALL_BOX = BoxConfig(box_size=128.0, n_mesh=8, n_particles=8)


def test_intensity_field_clip_nonneg():
    """lam = nbar V_cell max(1 + delta_h, 0): clipped where 1 + delta_h < 0."""
    box = BoxConfig(box_size=64.0, n_mesh=4, n_particles=4)
    delta_h = np.zeros((4, 4, 4))
    delta_h[0, 0, 0] = -2.5  # a deep void: 1 + delta_h = -1.5 -> clipped to 0
    delta_h[1, 1, 1] = 3.0  # an overdensity
    lam = CAT.intensity_field(delta_h, nbar=1.0, box=box)
    v_cell = (64.0 / 4.0) ** 3
    assert lam.shape == (4, 4, 4)
    assert np.all(lam >= 0.0)
    assert lam[0, 0, 0] == 0.0
    assert np.isclose(lam[1, 1, 1], 1.0 * v_cell * 4.0)
    assert np.isclose(lam[2, 2, 2], 1.0 * v_cell * 1.0)  # delta_h = 0 -> 1


def test_place_galaxies_mass_conservation():
    """Every galaxy lands in its own cell; counts are exactly conserved."""
    box = SMALL_BOX
    N = box.n_mesh
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 4, size=(N, N, N))
    xyz = CAT.place_galaxies(counts, box, rng)
    assert xyz.shape == (int(counts.sum()), 3)
    assert xyz.dtype == np.float64
    d = box.box_size / N
    idx = np.floor(xyz / d).astype(int)
    assert np.all(idx >= 0) and np.all(idx < N)  # in [0, L)
    flat = (idx[:, 0] * N + idx[:, 1]) * N + idx[:, 2]
    recovered = np.bincount(flat, minlength=N**3)
    assert np.array_equal(recovered, counts.ravel())


def test_place_galaxies_origin_and_empty():
    """The observer origin shifts every position; zero counts give an empty array."""
    box = SMALL_BOX
    rng = np.random.default_rng(1)
    counts = np.ones((box.n_mesh,) * 3, dtype=np.int64)
    origin = (10.0, -5.0, 1.0e5)
    xyz = CAT.place_galaxies(counts, box, rng, origin=origin)
    assert np.all(xyz[:, 2] >= 1.0e5)  # shifted down the +z line of sight
    assert np.all(xyz[:, 2] < 1.0e5 + box.box_size)
    empty = CAT.place_galaxies(np.zeros((box.n_mesh,) * 3, dtype=np.int64), box, rng)
    assert empty.shape == (0, 3)


def test_sampling_reproducible():
    """Same draw seed -> identical catalog; a different seed -> a different one."""
    box = SMALL_BOX
    lam = np.full((box.n_mesh,) * 3, 2.0)

    def draw(seed):
        rng = np.random.default_rng(seed)
        counts = CAT.poisson_counts(lam, rng)
        return CAT.place_galaxies(counts, box, rng)

    a, b, c = draw(42), draw(42), draw(43)
    assert np.array_equal(a, b)
    assert a.shape != c.shape or not np.array_equal(a, c)


def test_sample_from_positions_recovers_nbar():
    """A small forward run: realized nbar is close to target; clip stays small."""
    cfg = SimConfig(
        cosmology=Cosmology(),
        box=BoxConfig(box_size=256.0, n_mesh=16, n_particles=16),
        time=TimeStepping(z_init=9.0, z_final=0.0, n_steps=2),
        ic=InitialConditions(f_NL=0.0, kind="gaussian", seed=3),
        tracer=Tracer(b1=1.5, b2=0.0, A=1.0),
        catalog=CatalogSampling(enabled=True, nbar=5e-3, draw_seed=7),
    )
    cat = run(cfg, backend="eh98").catalog
    assert cat.n_galaxies > 0
    # realized nbar drifts slightly above target from clipping; small here.
    assert cat.realized_nbar >= cfg.catalog.nbar
    assert abs(cat.realized_nbar / cfg.catalog.nbar - 1.0) < 0.05
    # measured ~0.003 at this config; assert a loose bound (clip is negligible).
    assert cat.clip_fraction < 0.05
    # Ngal / V is the realized nbar to Poisson precision.
    nbar_meas = cat.n_galaxies / cfg.box.box_size**3
    assert abs(nbar_meas / cat.realized_nbar - 1.0) < 0.05


def test_write_parquet_schema(tmp_path):
    """The parquet matches the consumer schema: x, y, z + box-size metadata."""
    box = SMALL_BOX
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 3, size=(box.n_mesh,) * 3)
    xyz = CAT.place_galaxies(counts, box, rng)
    cat = CAT.Catalog(xyz=xyz, box_size=box.box_size, bin_index=5, z_eff=0.3)
    p = tmp_path / "realization_00001" / "catalog.parq"
    cat.write_parquet(str(p))

    t = pq.read_table(str(p))
    assert t.column_names == ["x", "y", "z", "bin"]
    assert str(t.schema.field("x").type) == "double"
    assert str(t.schema.field("bin").type) == "int8"
    md = {k.decode(): v.decode() for k, v in t.schema.metadata.items()}
    assert float(md["box_size_x"]) == box.box_size
    assert md["generator"] == "mbody"
    assert float(md["z_eff"]) == 0.3
    assert int(t.num_rows) == int(counts.sum())


def test_write_parquet_no_bin_when_unset(tmp_path):
    """bin_index = -1 (default) omits the bin column."""
    box = SMALL_BOX
    rng = np.random.default_rng(0)
    counts = np.ones((box.n_mesh,) * 3, dtype=np.int64)
    xyz = CAT.place_galaxies(counts, box, rng)
    cat = CAT.Catalog(xyz=xyz, box_size=box.box_size)
    p = tmp_path / "catalog.parq"
    cat.write_parquet(str(p))
    assert pq.read_table(str(p)).column_names == ["x", "y", "z"]


def test_parquet_consumer_read(tmp_path):
    """Read x/y/z from the parquet the way a downstream null-test pipeline does."""
    box = SMALL_BOX
    rng = np.random.default_rng(2)
    counts = rng.integers(0, 3, size=(box.n_mesh,) * 3)
    # a distant-observer offset so r is well away from 0 (matches survey geometry)
    xyz = CAT.place_galaxies(counts, box, rng, origin=(0.0, 0.0, 1.0e5))
    cat = CAT.Catalog(xyz=xyz, box_size=box.box_size, bin_index=1)
    p = tmp_path / "realization_00001" / "catalog.parq"
    cat.write_parquet(str(p))

    pf = pq.ParquetFile(str(p))
    cols = set(pf.schema.names)
    assert "r" not in cols  # the (x, y, z) consumer branch
    batch = next(pf.iter_batches(columns=["x", "y", "z"]))
    x = batch.column("x").to_numpy()
    y = batch.column("y").to_numpy()
    z = batch.column("z").to_numpy()
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x) % (2.0 * np.pi)
    assert len(r) == int(counts.sum())
    assert np.all(r > 0.0)
    assert np.all(np.isfinite(theta)) and np.all(np.isfinite(phi))
    assert np.all((theta >= 0.0) & (theta <= np.pi))
    assert np.all((phi >= 0.0) & (phi < 2.0 * np.pi))


def test_painting_round_trip_count_density():
    """The sampled positions paint back to a sensible mean density (sanity)."""
    box = SMALL_BOX
    rng = np.random.default_rng(4)
    counts = rng.poisson(3.0, size=(box.n_mesh,) * 3)
    xyz = CAT.place_galaxies(counts, box, rng)
    import mlx.core as mx

    painted = np.asarray(PA.cic_paint(mx.array(xyz, dtype=mx.float32), box))
    # CIC conserves mass: the painted total equals the galaxy count.
    assert np.isclose(painted.sum(), counts.sum(), rtol=1e-4)


def test_catalogsampling_validation():
    """CatalogSampling rejects nonsensical knobs."""
    CatalogSampling()  # defaults are valid
    with pytest.raises(ValueError):
        CatalogSampling(nbar=0.0)
    with pytest.raises(ValueError):
        CatalogSampling(sampler="negbinom")
    with pytest.raises(ValueError):
        CatalogSampling(nonneg="exp")  # reserved, not yet implemented
    with pytest.raises(ValueError):
        CatalogSampling(origin=(0.0, 1.0))


# --- Stage B: statistical validation (tolerances measured + user-approved via
# scripts/probe_mock_validation.py; see the worklog for the decomposition) -------


def test_nbar_recovery():
    """N_gal / V recovers the realized (clipped) intensity nbar to ~Poisson level.

    The realized nbar sits slightly ABOVE the target from clipping (reported, not
    renormalized); the *catalog* faithfully samples the realized intensity, so
    N_gal / V matches the realized nbar to the Poisson floor (~0.2% here).
    """
    cfg = SimConfig(
        box=BoxConfig(box_size=384.0, n_mesh=32, n_particles=32),
        time=TimeStepping(z_init=9.0, z_final=0.0, n_steps=3),
        ic=InitialConditions(f_NL=0.0, kind="gaussian", seed=5),
        tracer=Tracer(b1=1.5, b2=0.0, A=1.0),
        catalog=CatalogSampling(enabled=True, nbar=5.0e-3, draw_seed=2),
    )
    cat = run(cfg, backend="eh98").catalog
    nbar_meas = cat.n_galaxies / cfg.box.box_size**3
    assert abs(nbar_meas / cat.realized_nbar - 1.0) < 0.01  # measured ~0.002
    assert cat.realized_nbar >= cfg.catalog.nbar  # clip inflates above target


def test_sampler_unbiased_low_k():
    """The Poisson + placement draw is unbiased: the catalog reproduces the
    noiseless clipped intensity field's low-k cross-bias with the matter field.

    Averaging the catalog cross spectrum over draw seeds at a FIXED field removes
    cosmic variance, isolating the sampler. cross(catalog, m) / cross(noiseless, m)
    -> 1 at low k (the small high-k droop is the placement window). This is the
    clip-independent fidelity check; the clip's effect on the *effective* bias is a
    separate, documented physical shift (see probe_mock_validation.py).
    """
    # n_mesh = 64: the placement-window droop at the lowest 2 bins is ~0.3% / 1.3%
    # (it grows on a coarser mesh, where the cell -- the placement top-hat -- is
    # bigger; this is where the < 2% tolerance was measured).
    box = BoxConfig(box_size=512.0, n_mesh=64, n_particles=64)
    cfg = SimConfig(
        box=box,
        time=TimeStepping(z_init=9.0, z_final=0.0, n_steps=3),
        ic=InitialConditions(f_NL=0.0, kind="gaussian", seed=4),
        tracer=Tracer(b1=1.5, b2=0.0, A=1.0),
        catalog=CatalogSampling(enabled=True, nbar=5.0e-3, draw_seed=0),
    )
    res = run(cfg, backend="eh98")
    delta_m = PA.density_contrast(res.x, box)

    # The noiseless clipped intensity field -- the unbiased sampler target.
    delta_g = B.local_bias_tracer(delta_m, cfg.tracer.b1, cfg.tracer.b2)
    lam = CAT.intensity_field(delta_g, cfg.catalog.nbar, box)
    delta_e = mx.array(np.asarray(lam / lam.mean() - 1.0, np.float32))

    kb = np.arange(1, 3) * box.k_fundamental  # the lowest two |k| shells
    x_em = np.asarray(F.cross_power(delta_e, delta_m, box, kb), np.float64)

    ratios = []
    for ds in range(12):
        sampling = dataclasses.replace(cfg.catalog, draw_seed=ds)
        cat = CAT.sample_from_positions(res.x, box, cfg.tracer, sampling)
        gxyz = mx.array(np.ascontiguousarray(cat.xyz, np.float32))
        gfield = PA.density_contrast(gxyz, box)
        x_cm = np.asarray(F.cross_power(gfield, delta_m, box, kb), np.float64)
        ratios.append(x_cm / x_em)
    mean_ratio = np.mean(ratios, axis=0)
    assert np.all(np.abs(mean_ratio - 1.0) < 0.02)  # measured ~0.003 / ~0.013


# --- Stage C: f_NL injection-recovery (tolerances measured + user-approved via
# scripts/probe_mock_fnl.py; the through-PM amplitude suppression and the 1/k^2
# flattening are documented there as forward-model diagnostics) ------------------


def test_fnl_injection_survives_clip_and_poisson():
    """Local f_NL injected in the ICs gives the mock a scale-dependent bias that
    survives the clip non-negativity map and the Poisson draw.

    Matched-phase finite difference of dlnP/df_NL through the PM (same IC phases at
    +/-eps cancel cosmic variance), seed-averaged. Asserts the mock-specific claims
    only: the signal is scale-dependent, a tracer effect (matter null), the clip
    preserves it, and the Poisson catalog recovers it via the shot-free cross
    spectrum. The through-PM amplitude (~25% of linear theory) and the 1/k^2
    flattening are real forward-model effects, documented in the probe.
    """
    cosmo = Cosmology()
    # L = 1024 gives the low-k reach for the 1/k^2 scale-dependent bias.
    box = BoxConfig(box_size=1024.0, n_mesh=48, n_particles=48)
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=3)
    # nbar = 2e-2 keeps the (shot-free in the mean, but shot-variance-limited)
    # cross-spectrum recovery clean at the lowest bin with only 4 seeds.
    b1, b2, nbar, eps = 2.0, 1.0, 2.0e-2, 100.0
    kb = np.arange(1, 4) * box.k_fundamental

    def evolve(f_nl, seed):
        cfg = SimConfig(
            cosmology=cosmo,
            box=box,
            time=time,
            ic=InitialConditions(f_NL=f_nl, kind="local_fnl", seed=seed),
            tracer=Tracer(b1=b1, b2=b2, A=1.0),
            catalog=CatalogSampling(enabled=True, nbar=nbar, draw_seed=seed),
        )
        res = run(cfg, backend="eh98")
        delta_m = PA.density_contrast(res.x, box)
        delta_g = B.local_bias_tracer(delta_m, b1, b2)  # unclipped tracer
        lam = CAT.intensity_field(delta_g, nbar, box)
        delta_e = mx.array(np.asarray(lam / lam.mean() - 1.0, np.float32))  # clipped
        gxyz = mx.array(np.ascontiguousarray(res.catalog.xyz, np.float32))
        gfield = PA.density_contrast(gxyz, box)  # the Poisson catalog
        return delta_m, delta_g, delta_e, gfield

    def auto_dlnp(field_p, field_m):
        lp = np.log(np.asarray(F.band_power(field_p, box, kb), np.float64))
        lm = np.log(np.asarray(F.band_power(field_m, box, kb), np.float64))
        return (lp - lm) / (2.0 * eps)

    def cross_dlnp(a_p, a_m, m_p, m_m):
        lp = np.log(np.asarray(F.cross_power(a_p, m_p, box, kb), np.float64))
        lm = np.log(np.asarray(F.cross_power(a_m, m_m, box, kb), np.float64))
        return (lp - lm) / (2.0 * eps)

    clip, unclip, matter, recov = [], [], [], []
    for s in range(4):
        dmp, dgp, dep, gfp = evolve(eps, s)
        dmm, dgm, dem, gfm = evolve(-eps, s)
        clip.append(auto_dlnp(dep, dem))
        unclip.append(auto_dlnp(dgp, dgm))
        matter.append(auto_dlnp(dmp, dmm))
        recov.append(cross_dlnp(gfp, gfm, dmp, dmm) / cross_dlnp(dep, dem, dmp, dmm))
    clip = np.mean(clip, axis=0)
    unclip = np.mean(unclip, axis=0)
    matter = np.mean(matter, axis=0)
    recov = np.mean(recov, axis=0)

    # (1) scale-dependent bias: the response falls with k (Delta_b ~ 1/k^2-ish).
    assert clip[0] / clip[2] > 2.5  # measured ~4.7
    # (2) the clip preserves the f_NL response (does not kill it).
    assert 0.8 < clip[0] / unclip[0] < 1.3  # measured ~1.01
    # (3) a tracer effect: the matter field has ~no f_NL power.
    assert abs(matter[0] / clip[0]) < 0.15  # measured ~0.04
    # (4) the Poisson catalog recovers the field signal (shot-free cross spectrum).
    assert abs(recov[0] - 1.0) < 0.15  # measured ~0.04
