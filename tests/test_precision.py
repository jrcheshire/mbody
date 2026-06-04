"""Unit tests for mbody.precision: the float32/float64 policy.

These encode the verified MLX behavior the policy is built on, so if a future
MLX version changes it (e.g. starts supporting float64 on Metal) the tests flag
it instead of silently changing our accuracy.
"""

import mlx.core as mx
import pytest

from mbody import precision as P


def test_dtype_constants():
    assert P.REAL == mx.float32
    assert P.COMPLEX == mx.complex64
    assert P.FP64 == mx.float64


def test_fp64_cpu_island_is_true_double():
    # In float32, (1 + 1e-9) - 1 collapses to 0; in float64 it survives.
    with P.fp64_cpu():
        one = mx.array(1.0, dtype=P.FP64)
        val = one + 1e-9 - one
        mx.eval(val)
    assert float(val) > 1e-12


def test_float64_op_on_gpu_raises():
    # The load-bearing fact: float64 math on Metal is a hard error, not a
    # silent downcast. We saw it raise ValueError("float64 is not supported on
    # the GPU"); if this ever stops raising, revisit precision.py.
    with pytest.raises(ValueError, match="float64"):
        with mx.stream(mx.gpu):
            a = mx.array(2.0, dtype=P.FP64)
            mx.eval(a * a)


def test_ensure_gpu_safe_flags_float64():
    with pytest.raises(TypeError):
        P.ensure_gpu_safe(mx.array(1.0, dtype=P.FP64), name="test")
    # A float32 array passes through unchanged.
    x = mx.array(1.0, dtype=P.REAL)
    assert P.ensure_gpu_safe(x) is x


def test_accurate_sum_returns_float64_and_is_accurate():
    n = 5_000_000
    x = mx.full((n,), 0.1, dtype=mx.float32)
    # The exact sum of n copies of the float32 value of 0.1, in python double.
    ref = n * float(mx.array(0.1, dtype=mx.float32))
    acc = P.accurate_sum(x)
    assert acc.dtype == P.FP64
    acc = float(acc)
    naive = float(mx.sum(x))  # float32 accumulation on the GPU
    # The fp64 island is at least as accurate as the naive float32 reduction,
    # and essentially exact.
    assert abs(acc - ref) <= abs(naive - ref)
    assert abs(acc - ref) / ref < 1e-9


def test_subtract_mean_centers_field():
    x = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    y = P.subtract_mean(x)
    assert y.dtype == P.REAL
    assert abs(float(P.accurate_mean(y))) < 1e-6
