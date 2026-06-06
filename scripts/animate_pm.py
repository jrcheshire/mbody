"""Animate the PM leapfrog: real gravitational structure formation.

This is the Zel'dovich animation's grown-up sibling. Instead of streaming
particles along a fixed LPT displacement, it runs the full particle-mesh
leapfrog -- solving for gravity at every step -- and films the result. Watch the
Zel'dovich sheet collapse into a sharp cosmic web: filaments thin, knots
brighten, voids empty. That sharpening (relative to the LPT version, which just
smears) is gravity doing its work.

The trajectory is captured by mbody.diagnostics.SnapshotRecorder, the optional,
off-the-autodiff-path seam that the integrator's `snapshot` callback feeds, and
rendered by mbody.viz.animate_slab -- the same shared diagnostics the dashboard
and tests use, not a one-off here. Growth recording is turned off (record_growth
=False) since the animation only needs the slab projections.

Run: pixi run python scripts/animate_pm.py
Saves: outputs/structure_formation.gif
"""

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import diagnostics as D
from mbody import integrate as IG
from mbody import viz


def main():
    cosmo = Cosmology()
    box = BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=80)

    rec = D.SnapshotRecorder(box, slab_thick=4, record_growth=False)
    IG.leapfrog(box, cosmo, time, seed=0, snapshot=rec)

    out = viz.animate_slab(rec.slabs, rec.a, box, "outputs/structure_formation.gif")
    print("wrote %s (%d frames)" % (out, len(rec.slabs)))


if __name__ == "__main__":
    main()
