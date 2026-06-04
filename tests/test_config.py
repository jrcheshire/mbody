"""Unit tests for mbody.config: derived quantities and validation.

config carries no physics, so these are exact-value checks -- a good guard
against a typo in a derived property silently shifting every downstream scale.
"""

import math

import pytest

from mbody.config import (
    BoxConfig,
    Cosmology,
    InitialConditions,
    SimConfig,
    TimeStepping,
)


def test_cosmology_defaults_are_flat():
    c = Cosmology()
    assert math.isclose(c.Omega_m + c.Omega_Lambda, 1.0)
    assert math.isclose(c.Omega_cdm, c.Omega_m - c.Omega_b)
    assert c.H0 == 100.0 * c.h


def test_cosmology_rejects_baryons_above_matter():
    with pytest.raises(ValueError):
        Cosmology(Omega_b=0.5, Omega_m=0.3)


def test_cosmology_rejects_nonpositive_amplitude():
    with pytest.raises(ValueError):
        Cosmology(sigma8=0.0)


def test_box_derived_quantities():
    b = BoxConfig(box_size=256.0, n_mesh=128, n_particles=128)
    assert math.isclose(b.cell_size, 2.0)
    assert b.n_mesh_total == 128**3
    assert b.n_particles_total == 128**3
    assert math.isclose(b.k_fundamental, 2.0 * math.pi / 256.0)
    assert math.isclose(b.k_nyquist, math.pi * 128 / 256.0)
    # Nyquist is exactly n_mesh/2 fundamental modes out.
    assert math.isclose(b.k_nyquist, (b.n_mesh / 2) * b.k_fundamental)


def test_box_rejects_bad_sizes():
    with pytest.raises(ValueError):
        BoxConfig(n_mesh=0)
    with pytest.raises(ValueError):
        BoxConfig(box_size=-1.0)


def test_timestepping_validation():
    with pytest.raises(ValueError):
        TimeStepping(integrator="leapfrog")
    with pytest.raises(ValueError):
        TimeStepping(memory_mode="rematerialize")
    with pytest.raises(ValueError):
        TimeStepping(z_init=0.0, z_final=1.0)
    with pytest.raises(ValueError):
        TimeStepping(n_steps=0)


def test_initial_conditions_validation():
    with pytest.raises(ValueError):
        InitialConditions(lpt_order=3)
    with pytest.raises(ValueError):
        InitialConditions(kind="equilateral_fnl")
    # f_NL = 0 with the default Gaussian kind is fine.
    assert InitialConditions().f_NL == 0.0


def test_simconfig_composes_and_summarizes():
    cfg = SimConfig(
        box=BoxConfig(box_size=512.0, n_mesh=256),
        ic=InitialConditions(f_NL=100.0, kind="local_fnl"),
    )
    assert isinstance(cfg.cosmology, Cosmology)
    assert cfg.box.box_size == 512.0
    assert cfg.ic.f_NL == 100.0
    text = cfg.summary()
    assert "M-body SimConfig" in text
    assert "f_NL=100.0" in text
