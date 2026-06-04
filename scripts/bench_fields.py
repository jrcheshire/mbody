"""Benchmark Gaussian-random-field generation + P(k) estimation vs mesh size.

Reports wall time and peak GPU memory at a fixed 2 Mpc/h spacing, to map the
performance/memory scaling before committing to a production resolution. The
generator is GPU FFT (fast); the estimator currently copies |delta_k|^2 to the
CPU and histograms there, so it dominates at large N -- a known optimization
target.

Run: pixi run python scripts/bench_fields.py
"""

import time

import mlx.core as mx

from mbody.config import BoxConfig, Cosmology
from mbody import fields as F


def main():
    cosmo = Cosmology()
    sizes = (64, 128, 256, 512)
    get_peak = getattr(mx, "get_peak_memory", None)
    reset_peak = getattr(mx, "reset_peak_memory", None)

    print("N      L[Mpc/h]   gen[s]   estimate[s]   peak_mem[MB]")
    for N in sizes:
        box = BoxConfig(box_size=2.0 * N, n_mesh=N, n_particles=N)
        if reset_peak:
            reset_peak()
        t0 = time.perf_counter()
        d = F.gaussian_random_field(box, cosmo, seed=0)
        mx.eval(d)
        t1 = time.perf_counter()
        F.power_spectrum(d, box)
        t2 = time.perf_counter()
        mem = get_peak() / 1e6 if get_peak else float("nan")
        print(
            f"{N:4d}   {2.0 * N:8.0f}   {t1 - t0:6.3f}   "
            f"{t2 - t1:9.3f}   {mem:11.1f}"
        )


if __name__ == "__main__":
    main()
