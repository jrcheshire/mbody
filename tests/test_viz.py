"""Smoke tests for the optional visualization layer (mbody.viz).

These only check that the renderers run end to end and write non-empty files;
the underlying estimators are validated in test_diagnostics. Everything uses a
tiny 16^3 EH98 run and writes into a tmp directory, so the suite stays fast and
leaves no artifacts.
"""

import os

import matplotlib.pyplot as plt

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import diagnostics as D
from mbody import fields as F
from mbody import integrate as IG
from mbody import painting as PA
from mbody import viz


def _tiny():
    box = BoxConfig(box_size=200.0, n_mesh=16, n_particles=16)
    cosmo = Cosmology()
    time = TimeStepping(z_init=9.0, z_final=0.0, n_steps=4)
    return box, cosmo, time


def test_dashboard_and_animation_write_files(tmp_path):
    box, cosmo, time = _tiny()
    rec = D.SnapshotRecorder(box, slab_thick=2)
    x0, _ = IG.initial_state(box, cosmo, time, seed=0, backend="eh98")
    xf, _ = IG.leapfrog(box, cosmo, time, seed=0, backend="eh98", snapshot=rec)
    ic_field = PA.density_contrast(x0, box)
    final_field = PA.density_contrast(xf, box)

    out_png = str(tmp_path / "dash.png")
    p = viz.dashboard(
        box,
        cosmo,
        ic_field,
        final_field,
        recorder=rec,
        z_init=time.z_init,
        z_final=time.z_final,
        out=out_png,
        backend="eh98",
    )
    assert os.path.exists(p) and os.path.getsize(p) > 0

    out_gif = str(tmp_path / "anim.gif")
    g = viz.animate_slab(rec.slabs, rec.a, box, out_gif, fps=5)
    assert os.path.exists(g) and os.path.getsize(g) > 0


def test_dashboard_without_recorder(tmp_path):
    # The growth panel is optional: dashboard must still render with recorder=None.
    box, cosmo, time = _tiny()
    xf, _ = IG.leapfrog(box, cosmo, time, seed=0, backend="eh98")
    ic_field = PA.density_contrast(
        IG.initial_state(box, cosmo, time, seed=0, backend="eh98")[0], box
    )
    final_field = PA.density_contrast(xf, box)
    out_png = str(tmp_path / "dash_norec.png")
    p = viz.dashboard(box, cosmo, ic_field, final_field, out=out_png, backend="eh98")
    assert os.path.exists(p) and os.path.getsize(p) > 0


def test_density_slice_returns_axes():
    box, cosmo, _ = _tiny()
    d = F.gaussian_random_field(box, cosmo, seed=1, backend="eh98")
    fig, ax = viz.density_slice(d, box)
    assert fig is not None and ax is not None
    plt.close(fig)
