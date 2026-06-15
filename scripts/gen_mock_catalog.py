"""Generate a forward-only M-body galaxy mock catalog (single shell).

Runs the PM forward model once and Poisson-samples a galaxy catalog from the
biased final field (mbody.catalog), writing it as
``<out_dir>/realization_{idx:05d}/catalog.parq`` in a simple parquet schema
(x, y, z in Mpc/h, box size in file-level metadata, an optional int8 ``bin``
column) that a field-level null test can read directly.

This is the single-box, single-redshift generator. The IC phase seed and the
Poisson draw seed are independent axes -- vary ``--ic-seed`` for cosmic variance,
or hold it and vary ``--draw-seed`` to resample shot noise alone.

Example::

    pixi run python scripts/gen_mock_catalog.py \\
        --box 1500 --n-mesh 192 --z 0.1 --nbar 7.347e-2 --b1 1.03 \\
        --f-nl 0 --realization 1 --bin-index 1 --out outputs/mocks
"""

import argparse

from mbody import (
    BoxConfig,
    CatalogSampling,
    Cosmology,
    InitialConditions,
    RedshiftSpace,
    SimConfig,
    TimeStepping,
    Tracer,
    run,
)
from mbody import catalog as CAT
from mbody import integrate as IG
from mbody import rsd as RS


def build_config(args):
    """Assemble a SimConfig for one shell from parsed CLI arguments."""
    kind = "local_fnl" if args.f_nl != 0.0 else "gaussian"
    return SimConfig(
        cosmology=Cosmology(),
        box=BoxConfig(box_size=args.box, n_mesh=args.n_mesh, n_particles=args.n_mesh),
        time=TimeStepping(
            z_init=args.z_init,
            z_final=args.z,
            n_steps=args.n_steps,
            integrator=args.integrator,
        ),
        ic=InitialConditions(
            f_NL=args.f_nl, seed=args.ic_seed, lpt_order=args.lpt_order, kind=kind
        ),
        tracer=Tracer(b1=args.b1, b2=args.b2, A=args.amplitude),
        rsd=RedshiftSpace(enabled=args.rsd, los_axis=args.los_axis),
        catalog=CatalogSampling(
            enabled=True,
            nbar=args.nbar,
            draw_seed=args.draw_seed,
            origin=(0.0, 0.0, args.observer_distance),
            bin_index=args.bin_index,
        ),
    )


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--box", type=float, default=1500.0, help="box size, Mpc/h")
    p.add_argument("--n-mesh", type=int, default=128, help="mesh/particles per side")
    p.add_argument("--z", type=float, default=0.0, help="shell redshift (z_final)")
    p.add_argument("--z-init", type=float, default=9.0, help="initial redshift")
    p.add_argument("--n-steps", type=int, default=10, help="PM steps")
    p.add_argument("--integrator", default="bullfrog", help="exact|fastpm|bullfrog")
    p.add_argument("--lpt-order", type=int, default=2, help="1 or 2")
    p.add_argument("--nbar", type=float, default=1e-3, help="number density (Mpc/h)^-3")
    p.add_argument("--b1", type=float, default=1.5, help="linear bias")
    p.add_argument("--b2", type=float, default=0.0, help="quadratic bias")
    p.add_argument("--amplitude", type=float, default=1.0, help="linear amplitude A")
    p.add_argument("--f-nl", type=float, default=0.0, help="local f_NL")
    p.add_argument("--ic-seed", type=int, default=0, help="IC phase seed")
    p.add_argument("--draw-seed", type=int, default=0, help="Poisson/placement seed")
    p.add_argument("--rsd", action="store_true", help="sample in redshift space")
    p.add_argument("--los-axis", type=int, default=0, help="RSD line-of-sight axis")
    p.add_argument(
        "--observer-distance",
        type=float,
        default=0.0,
        help="offset added to z so the box is observer-centered (Mpc/h)",
    )
    p.add_argument("--realization", type=int, default=1, help="realization index")
    p.add_argument("--bin-index", type=int, default=-1, help="int8 bin label (-1=none)")
    p.add_argument("--backend", default="camb", help="linear-theory backend")
    p.add_argument("--out", default="outputs/mocks", help="output directory")
    p.add_argument(
        "--stream",
        action="store_true",
        help="stream the catalog straight to parquet in cell-slab chunks; the "
        "full (n_gal, 3) block is never held in memory (bounded RAM for the "
        "~10^9-galaxy survey shells). Skips the diagnostic field paints.",
    )
    p.add_argument(
        "--chunk-cells",
        type=int,
        default=0,
        help="cells per streamed chunk (0 = one mesh slab, n_mesh^2)",
    )
    return p.parse_args(argv)


def stream_to_parquet(args, cfg, path):
    """Lean forward run + streamed catalog write (bounded memory).

    Runs the PM forward model directly (skipping the driver's diagnostic field
    paints, which a mock does not need) and streams the catalog to parquet without
    ever materializing the full position block. Returns the path.
    """
    box, cosmo, time, ic = cfg.box, cfg.cosmology, cfg.time, cfg.ic
    x0, p0 = IG.initial_state(
        box,
        cosmo,
        time,
        seed=ic.seed,
        f_NL=ic.f_NL,
        backend=args.backend,
        lpt_order=ic.lpt_order,
    )
    xf, pf = IG.evolve_state(
        x0, p0, box, cosmo, IG.a_grid(time), integrator=time.integrator
    )
    if cfg.rsd.enabled:
        pos = RS.redshift_space_positions(
            xf,
            pf,
            box,
            cosmo,
            z=time.z_final,
            los_axis=cfg.rsd.los_axis,
            f_growth=cfg.rsd.f_growth,
        )
    else:
        pos = xf
    stats = CAT.sample_to_parquet(
        pos,
        box,
        cfg.tracer,
        cfg.catalog,
        path,
        chunk_cells=args.chunk_cells or None,
        overwrite=True,
        z=time.z_final,
        f_NL=ic.f_NL,
        ic_seed=ic.seed,
    )
    print(
        f"\nstreamed {stats['n_galaxies']} galaxies -> {path}\n"
        f"  nbar target={stats['nbar_target']:.4e} "
        f"realized={stats['realized_nbar']:.4e} "
        f"clip_fraction={stats['clip_fraction']:.4f}"
    )
    return path


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args)
    print(cfg.summary())
    path = f"{args.out}/realization_{args.realization:05d}/catalog.parq"
    if args.stream:
        return stream_to_parquet(args, cfg, path)
    result = run(cfg, backend=args.backend)
    cat = result.catalog
    cat.write_parquet(path, overwrite=True)
    print(
        f"\nwrote {cat.n_galaxies} galaxies -> {path}\n"
        f"  nbar target={cat.nbar_target:.4e} realized={cat.realized_nbar:.4e} "
        f"clip_fraction={cat.clip_fraction:.4f}"
    )
    return path


if __name__ == "__main__":
    main()
