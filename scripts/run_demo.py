"""Headline demo: one SimConfig in, a dashboard + animation + saved run out.

This is the "watch it and believe it" entry point. It builds a single SimConfig,
runs the forward model through the driver, and writes:

  outputs/demo_run/dashboard.png  -- density slice, P(k) vs linear theory at
                                     z_init and z_final, large-scale growth vs
                                     D(a), and r(k) phase-correlation with the IC
  outputs/demo_run/*.npy          -- the final state and density fields
  outputs/demo_run/config.json    -- the exact config that produced it
  outputs/run_anim.gif            -- the Zel'dovich sheet collapsing into the web

Uses the EH98 backend so it runs in seconds with no CAMB startup; pass
backend="camb" for the accurate transfer function.

Run: pixi run python scripts/run_demo.py
"""

import mbody
from mbody import viz
from mbody.config import BoxConfig, InitialConditions, SimConfig, TimeStepping


def main():
    cfg = SimConfig(
        box=BoxConfig(box_size=300.0, n_mesh=64, n_particles=64),
        time=TimeStepping(z_init=9.0, z_final=0.0, n_steps=40),
        ic=InitialConditions(seed=0, kind="gaussian"),
    )
    print(cfg.summary())

    res = mbody.run(cfg, backend="eh98", record=True)
    out_dir = res.save("outputs/demo_run")
    gif = viz.animate_slab(
        res.recorder.slabs, res.recorder.a, cfg.box, "outputs/run_anim.gif"
    )
    print("wrote %s/ (dashboard.png, *.npy, config.json)" % out_dir)
    print("wrote %s (%d frames)" % (gif, len(res.recorder.slabs)))


if __name__ == "__main__":
    main()
