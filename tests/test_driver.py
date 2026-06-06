"""Tests for the single run driver (mbody.driver) and the honest-config defaults.

The driver is checked to: run a SimConfig end to end, equal the loose-call
result it wraps, reject the reserved-but-unimplemented config options, carry
f_NL through, record a trajectory on request, and serialize a run. Uses the
EH98 backend to stay fast and CAMB-free.
"""

import json
import os

import numpy as np
import pytest

import mbody
from mbody.config import (
    BoxConfig,
    Cosmology,
    InitialConditions,
    SimConfig,
    TimeStepping,
)
from mbody import integrate as IG
from mbody import painting as PA


def _cfg(**ic_kwargs):
    return SimConfig(
        cosmology=Cosmology(),
        box=BoxConfig(box_size=200.0, n_mesh=32, n_particles=32),
        time=TimeStepping(z_init=9.0, z_final=0.0, n_steps=6),
        ic=InitialConditions(**ic_kwargs),
    )


def test_config_defaults_are_implemented():
    # The honest-config invariant: SimConfig() defaults name only built physics.
    cfg = SimConfig()
    assert cfg.time.integrator == "exact"
    assert cfg.ic.lpt_order == 1


def test_run_matches_loose_leapfrog():
    cfg = _cfg(seed=7)
    res = mbody.run(cfg, backend="eh98")
    # Same seed/box/time through the loose API must give the same final state.
    xf, _ = IG.leapfrog(cfg.box, cfg.cosmology, cfg.time, seed=7, backend="eh98")
    assert np.allclose(np.asarray(res.x), np.asarray(xf), atol=1e-4)
    # The stored final field is the CIC density of those positions.
    fld = PA.density_contrast(xf, cfg.box)
    assert np.allclose(np.asarray(res.final_field), np.asarray(fld), atol=1e-4)


def test_run_rejects_unimplemented_integrator():
    cfg = SimConfig(time=TimeStepping(integrator="fastpm"))
    with pytest.raises(NotImplementedError):
        mbody.run(cfg, backend="eh98")


def test_run_supports_2lpt():
    # lpt_order=2 is now implemented: the driver runs it and the 2LPT IC differs
    # from the Zel'dovich one at the same seed.
    res1 = mbody.run(_cfg(seed=8, lpt_order=1), backend="eh98")
    res2 = mbody.run(_cfg(seed=8, lpt_order=2), backend="eh98")
    assert not np.allclose(np.asarray(res1.ic_field), np.asarray(res2.ic_field))


def test_run_threads_fnl():
    # f_NL from the config must reach the field: a nonzero f_NL changes the IC.
    res0 = mbody.run(_cfg(seed=1, f_NL=0.0), backend="eh98")
    resf = mbody.run(_cfg(seed=1, f_NL=200.0, kind="local_fnl"), backend="eh98")
    d0 = np.asarray(res0.ic_field)
    df = np.asarray(resf.ic_field)
    assert not np.allclose(d0, df)


def test_run_records_growth_history():
    res = mbody.run(_cfg(seed=2), backend="eh98", record=True)
    a_arr, R = res.growth_history()
    assert len(a_arr) == res.config.time.n_steps + 1
    assert np.isclose(R[0], 1.0)
    assert R[-1] > R[0]


def test_run_without_record_has_no_history():
    res = mbody.run(_cfg(seed=3), backend="eh98")
    assert res.recorder is None
    with pytest.raises(ValueError):
        res.growth_history()


def test_run_diagnostic_methods():
    res = mbody.run(_cfg(seed=4), backend="eh98")
    k, pk, n = res.power()
    assert np.all(pk > 0) and np.all(n > 0)
    kc, r, nc = res.cross_with_ic()
    assert np.all(r <= 1.0 + 1e-6)


def test_run_save_writes_artifacts(tmp_path):
    res = mbody.run(_cfg(seed=5), backend="eh98", record=True)
    out = str(tmp_path / "run0")
    res.save(out)
    for name in ("config.json", "x_final.npy", "final_field.npy", "dashboard.png"):
        p = os.path.join(out, name)
        assert os.path.exists(p) and os.path.getsize(p) > 0
    # config.json round-trips the box size.
    with open(os.path.join(out, "config.json")) as fh:
        saved = json.load(fh)
    assert saved["box"]["box_size"] == res.config.box.box_size
