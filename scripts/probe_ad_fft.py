"""Probe: does MLX support reverse-mode autodiff through the FFT?

This is the single largest technical risk for M-body (ROADMAP step 0): both
the PM force solve and the P(k) estimator route through mx.fft, so a usable
dlnP/df_NL needs gradients to flow through the transform. The check compares
mx.grad of an FFT-built scalar against the exact analytic gradient.

Run: pixi run probe
"""

from importlib.metadata import version

import mlx.core as mx


def total_fourier_power(x):
    # Total Fourier power of a 1D real signal, built through the FFT.
    xk = mx.fft.fft(x)
    return mx.sum(mx.abs(xk) ** 2)


def main():
    print("mlx version:", version("mlx"))
    print("default device:", mx.default_device())

    n = 16
    mx.random.seed(0)
    x = mx.random.normal((n,))

    try:
        grad = mx.grad(total_fourier_power)(x)
    except Exception as exc:
        print("FAIL: mx.grad through mx.fft raised:", repr(exc))
        return

    # By Parseval, sum_k |X_k|^2 = n * sum_i x_i^2, so d/dx_i = 2 * n * x_i.
    grad_exact = 2.0 * n * x
    scale = float(mx.max(mx.abs(grad_exact))) + 1e-12
    rel_err = float(mx.max(mx.abs(grad - grad_exact))) / scale
    print("max relative error vs analytic 2*n*x: {:.3e}".format(rel_err))

    if rel_err < 1e-3:
        print("PASS: autodiff flows through mx.fft and matches the analytic gradient.")
    else:
        print("WARN: gradient mismatch -- AD-through-FFT may be incomplete here.")


if __name__ == "__main__":
    main()
