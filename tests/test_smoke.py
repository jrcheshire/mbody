"""Smoke tests: the package imports and the MLX autodiff foundation works.

These exercise no physics. They confirm the pixi environment is wired
correctly (MLX present, arrays evaluate, mx.grad runs) so that a fresh
`pixi run test` is a meaningful green/red signal before any PM code exists.
"""

import mlx.core as mx

import mbody


def test_version():
    assert mbody.__version__


def test_mlx_array_roundtrip():
    x = mx.arange(5, dtype=mx.float32)
    assert float(mx.sum(x)) == 10.0


def test_mlx_grad_polynomial():
    # d/dx (x^3) = 3 x^2; at x = 2 this is 12.
    grad_fn = mx.grad(lambda x: x**3)
    assert abs(float(grad_fn(mx.array(2.0))) - 12.0) < 1e-5
