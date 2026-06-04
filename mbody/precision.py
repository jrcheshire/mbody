"""Precision policy: where M-body uses float32 vs float64, and why.

MLX on Apple Silicon runs the GPU (Metal) in **float32 only**. We verified this
directly (see docs/architecture-plan.md, "Local verification"): any float64
*operation* dispatched to the GPU raises

    ValueError: float64 is not supported on the GPU

It does NOT silently fall back to float32 -- it is a hard, loud error. True
float64 arithmetic works only on the CPU stream (`with mx.stream(mx.cpu)`),
which on a unified-memory machine shares the same physical RAM, so moving an
array there is cheap.

That single fact sets the whole policy:

- The PM "hot loop" -- painting particles to the mesh, the FFT force solve,
  reading forces back, the leapfrog kick/drift -- stays in float32 on the GPU.
  These are the big arrays (a 512^3 mesh is ~0.5 GB in float32) and the speed
  that makes the unrolled autodiff graph affordable.
- A handful of *small but precision-critical* calculations run as float64
  "islands" on the CPU: the growth-factor ODE, transfer-function tabulation,
  k-grid coordinates, and global reductions where float32 round-off or
  catastrophic cancellation would bite (the mean of a near-zero density field,
  the <phi_G^2> subtraction in f_NL initial conditions, shot-noise removal in
  P(k)). These touch O(N) or fewer elements once, not every step, so the cost
  is negligible and the correctness is not.

The rule of thumb: float64 is a deliberate, fenced-off CPU island. A stray
float64 array reaching a GPU kernel is a bug that crashes immediately -- which
is good, because the alternative (silent precision loss) is the kind of error
that quietly corrupts a gradient.
"""

from contextlib import contextmanager

import mlx.core as mx

# GPU hot-loop dtypes. Metal supports single precision only.
REAL = mx.float32
COMPLEX = mx.complex64

# CPU-only "island" dtype for precision-critical work.
FP64 = mx.float64


@contextmanager
def fp64_cpu():
    """Run the enclosed MLX ops on the CPU stream, where float64 is allowed.

    Usage:

        with fp64_cpu():
            growth = solve_growth_ode(...)   # float64, exact
        growth32 = as_real(growth)           # back to the GPU world

    Anything float64 must live inside such a block; outside it (default GPU
    stream) a float64 op raises.
    """
    with mx.stream(mx.cpu):
        yield


def as_real(x):
    """Cast an array to the GPU real dtype (float32)."""
    return x.astype(REAL)


def as_complex(x):
    """Cast an array to the GPU complex dtype (complex64)."""
    return x.astype(COMPLEX)


def ensure_gpu_safe(x, name="array"):
    """Raise if x is float64, before it can reach a Metal kernel and crash.

    A guard for module boundaries: call it on arrays returned from an fp64
    island that are about to re-enter the GPU hot loop, so the failure points
    at the real culprit instead of some downstream op.
    """
    if x.dtype == FP64:
        raise TypeError(
            f"{name} is float64; Metal/GPU ops raise on float64. Keep float64 "
            "inside fp64_cpu() islands and cast back with as_real() first."
        )
    return x


def accurate_sum(x, axis=None):
    """Sum reduced in float64 on the CPU stream; returns a float64 array.

    Use for one-shot *global* reductions where float32 accumulation error or
    cancellation matters (field mean, <phi_G^2>, shot-noise subtraction). Do
    not call this inside the per-step hot loop -- it forces a CPU reduction.
    """
    with fp64_cpu():
        return mx.sum(x.astype(FP64), axis=axis)


def accurate_mean(x, axis=None):
    """Mean reduced in float64 on the CPU stream; returns a float64 array."""
    with fp64_cpu():
        return mx.mean(x.astype(FP64), axis=axis)


def subtract_mean(field):
    """Return ``field`` with its float64-accurate mean removed, in float32.

    The mean is computed exactly (float64), then subtracted on the GPU in
    float32. This is the right way to enforce <delta> = 0 on a density
    contrast: a naive float32 mean of a large near-zero field carries enough
    round-off to leave a spurious k = 0 mode.
    """
    mean = float(accurate_mean(field))
    return field - mean
