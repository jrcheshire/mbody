"""Tests for mbody.painting: CIC paint/read correctness and differentiability.

Covers mass conservation, the trilinear weights, the paint/read adjoint
identity, and -- the load-bearing one for M-body -- that mx.grad flows through
the scatter-add to particle positions (checked against a float64 finite
difference, since a naive float32 FD of the large sum-of-squares loss is pure
round-off).
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology
from mbody import lpt as L
from mbody import painting as PA

COSMO = Cosmology()
BOX = BoxConfig(box_size=64.0, n_mesh=32, n_particles=32)


def test_mass_conservation():
    pos = L.lpt_positions(BOX, COSMO, seed=0, z=0.0)
    rho = PA.cic_paint(pos, BOX)
    total = float(mx.sum(rho))
    assert abs(total - BOX.n_particles**3) / BOX.n_particles**3 < 1e-4


def test_uniform_grid_zero_contrast():
    grid = L.lagrangian_grid(BOX)
    delta = PA.density_contrast(grid, BOX)
    mx.eval(delta)
    assert float(mx.max(mx.abs(delta))) < 1e-4


def test_single_particle_weights():
    box = BoxConfig(box_size=8.0, n_mesh=8, n_particles=8)  # cell size = 1
    fx, fy, fz = 0.25, 0.10, 0.40
    p = mx.array([[2 + fx, 3 + fy, 4 + fz]])  # base cell (2, 3, 4)
    m = PA.cic_paint(p, box)
    mx.eval(m)
    assert abs(float(m[2, 3, 4]) - (1 - fx) * (1 - fy) * (1 - fz)) < 1e-5
    assert abs(float(m[3, 4, 5]) - fx * fy * fz) < 1e-5
    assert abs(float(mx.sum(m)) - 1.0) < 1e-5


def test_paint_read_adjoint():
    pos = L.lpt_positions(BOX, COSMO, seed=1, z=0.0)
    rng = np.random.default_rng(0)
    w = mx.array(rng.standard_normal(BOX.n_particles**3).astype(np.float32))
    f = mx.array(rng.standard_normal((BOX.n_mesh,) * 3).astype(np.float32))
    lhs = float(mx.sum(PA.cic_paint(pos, BOX, weights=w) * f))
    rhs = float(mx.sum(w * PA.cic_read(f, pos, BOX)))
    assert abs(lhs - rhs) / abs(lhs) < 1e-4


def test_read_constant_field_is_partition_of_unity():
    pos = L.lpt_positions(BOX, COSMO, seed=1, z=0.0)
    f = mx.ones((BOX.n_mesh,) * 3) * 3.5
    vals = PA.cic_read(f, pos, BOX)
    mx.eval(vals)
    assert float(mx.max(mx.abs(vals - 3.5))) < 1e-4


def test_cic_read_vector_equals_three_reads():
    # The shared-stencil 3-component read must match stacking three cic_reads --
    # both the forward values AND the gradient w.r.t. positions (the whole point
    # is that it is the same computation with the CIC index arithmetic shared,
    # for ~40% less reverse-mode memory in the force solve). Tested with a random
    # cotangent: an all-ones one would be a momentum-conservation null.
    pos = L.lpt_positions(BOX, COSMO, seed=3, z=0.0)
    rng = np.random.default_rng(0)
    fx, fy, fz = (
        mx.array(rng.standard_normal((BOX.n_mesh,) * 3).astype(np.float32))
        for _ in range(3)
    )
    stacked = mx.stack(
        [
            PA.cic_read(fx, pos, BOX),
            PA.cic_read(fy, pos, BOX),
            PA.cic_read(fz, pos, BOX),
        ],
        axis=1,
    )
    shared = PA.cic_read_vector(fx, fy, fz, pos, BOX)
    mx.eval(stacked, shared)
    assert float(mx.max(mx.abs(shared - stacked))) < 1e-5

    cot = mx.array(rng.standard_normal((BOX.n_particles**3, 3)).astype(np.float32))

    def g(fn):
        _, (gx,) = mx.vjp(lambda p: fn(p), [pos], [cot])
        return gx

    g_stack = g(
        lambda p: mx.stack(
            [PA.cic_read(fx, p, BOX), PA.cic_read(fy, p, BOX), PA.cic_read(fz, p, BOX)],
            axis=1,
        )
    )
    g_shared = g(lambda p: PA.cic_read_vector(fx, fy, fz, p, BOX))
    mx.eval(g_stack, g_shared)
    denom = float(mx.max(mx.abs(g_stack)))
    assert float(mx.max(mx.abs(g_shared - g_stack))) / denom < 1e-4


def test_paint_gradient_matches_finite_difference():
    pos = L.lpt_positions(BOX, COSMO, seed=2, z=0.0)
    g = mx.grad(lambda p: mx.sum(PA.cic_paint(p, BOX) ** 2))(pos)
    mx.eval(g)

    def loss64(p):  # float64 reduction avoids fp32 cancellation in the FD
        return float(np.sum(np.asarray(PA.cic_paint(p, BOX), np.float64) ** 2))

    h = 0.02 * BOX.cell_size
    grads, fds = [], []
    for idx in (123, 5000, 12345):
        for ax in range(3):
            e = np.zeros((BOX.n_particles**3, 3), np.float32)
            e[idx, ax] = 1.0
            e = mx.array(e)
            fds.append((loss64(pos + h * e) - loss64(pos - h * e)) / (2 * h))
            grads.append(float(g[idx, ax]))
    assert np.allclose(grads, fds, atol=0.03, rtol=0.05)
