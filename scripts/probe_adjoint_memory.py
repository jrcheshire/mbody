"""Attribute the reversible-adjoint per-cell memory and run the two crux
experiments that pick the memory-reduction lever for pushing up N.

Memory of the adjoint dlnP/df_NL is ~970 B/cell * N^3 (measured), and it is the
working set of a SINGLE force-solve VJP (the reversible adjoint re-runs one
mx.vjp(one_step) per reverse step). This probe:

  1. ATTRIBUTION -- the VJP peak of each force-solve piece in isolation
     (full / CIC paint / CIC read / FFT Poisson), to split the 970 B/cell.
  2. CRUX A (recompute) -- wrap the CIC paint+read in mx.checkpoint (recompute
     the nonlinear stencil in the backward pass instead of storing it) and see
     whether the full force-solve VJP peak drops. If yes, recompute is the
     lever; if ~no-op, slab-tiling is.
  3. CRUX B (bf16 fallback viability) -- does MLX accumulate a bf16 scatter-add
     in fp32, or in bf16 (the ~256 saturation wall)? Decides if bf16 mesh
     storage is even safe later.

Budget-capped: max N=288 -> ~23 GB transient peak (safe alongside other use).
The decision ratios are N-independent, so small N suffices. Memory peaks are
read from mx.get_peak_memory() (reset before each measured call). A RANDOM
cotangent is used on the forces -- an all-ones cotangent is a momentum-
conservation null (sum of forces ~0), which would zero the gradient.

Run: pixi run python scripts/probe_adjoint_memory.py
"""

import mlx.core as mx

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import integrate as IG, painting as PA, forces as FO
from mbody import precision as P

SEED = 0
SIZES = (160, 224, 288)


def vjp_peak(fn, primal):
    """Peak GPU memory (bytes) of a reverse-mode VJP of fn at `primal`.

    The cotangent matches the forward output shape and is random (so a force
    output is not seeded by the momentum-conserving all-ones null).
    """
    fwd = fn(primal)
    mx.eval(fwd)
    cot = mx.random.normal(fwd.shape, dtype=P.REAL)
    del fwd
    mx.clear_cache()
    mx.reset_peak_memory()
    _, vj = mx.vjp(fn, [primal], [cot])
    mx.eval(vj[0])
    peak = mx.get_peak_memory()
    del vj
    mx.clear_cache()
    return peak


def _checkpointed_force(box):
    """forces_on_particles with the CIC paint and read wrapped in mx.checkpoint,
    so their (recomputable, nonlinear-in-position) stencil/weight activations are
    regenerated in the backward pass instead of stored."""
    paint_ck = mx.checkpoint(lambda xx: PA.cic_paint(xx, box))
    read_ck = mx.checkpoint(
        lambda fx, fy, fz, xx: PA.cic_read_vector(fx, fy, fz, xx, box)
    )
    n_cell = box.n_mesh**3

    def force(xx):
        rho = paint_ck(xx)
        delta = rho / (xx.shape[0] / n_cell) - 1.0
        gx, gy, gz = FO.acceleration_field(delta, box)
        return read_ck(gx, gy, gz, xx)

    return force


def attribution_and_crux_a():
    cosmo = Cosmology()
    print("== Force-solve VJP memory attribution + crux A (checkpoint stencil) ==")
    print("  (peak GB; B/cell = peak / N^3; random force cotangent)")
    hdr = f"{'N':>5} {'full':>7} {'paint':>7} {'read':>7} {'fft':>7} {'ckpt':>7}"
    print(hdr + f"   {'full B/c':>9} {'ckpt B/c':>9} {'ckpt/full':>9}")
    for N in SIZES:
        box = BoxConfig(box_size=2.0 * N, n_mesh=N, n_particles=N)
        t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=2)
        x, _ = IG.initial_state(box, cosmo, t, seed=SEED, backend="eh98")
        mx.eval(x)
        # fixed inputs for the isolated FFT and read pieces
        delta = PA.density_contrast(x, box)
        gx, gy, gz = FO.acceleration_field(delta, box)
        mx.eval(delta, gx, gy, gz)

        # Default-arg binding captures the loop variables (avoids late binding).
        def f_full(xx, b=box):
            return FO.forces_on_particles(xx, b)

        def f_paint(xx, b=box):
            return PA.density_contrast(xx, b)

        def f_read(xx, fx=gx, fy=gy, fz=gz, b=box):
            return PA.cic_read_vector(fx, fy, fz, xx, b)

        def f_fft(dd, b=box):
            return mx.stack(FO.acceleration_field(dd, b))

        full = vjp_peak(f_full, x)
        paint = vjp_peak(f_paint, x)
        read = vjp_peak(f_read, x)
        fft = vjp_peak(f_fft, delta)
        ckpt = vjp_peak(_checkpointed_force(box), x)

        ncell = N**3
        g = 1e9
        print(
            f"{N:5d} {full/g:7.2f} {paint/g:7.2f} {read/g:7.2f} {fft/g:7.2f} "
            f"{ckpt/g:7.2f}   {full/ncell:9.1f} {ckpt/ncell:9.1f} "
            f"{ckpt/full:9.2f}"
        )


def crux_b_bf16_scatter():
    print("\n== Crux B: does MLX accumulate a bf16 scatter-add in fp32? ==")
    M = 100_000  # well under fp32's 2^24 (~16.7M) saturation
    idx = mx.zeros((M,), dtype=mx.int32)  # all deposits into cell 0
    ones_bf16 = mx.ones((M,), dtype=mx.bfloat16)
    acc_bf16 = mx.zeros((1,), dtype=mx.bfloat16).at[idx].add(ones_bf16)
    acc_fp32 = (
        mx.zeros((1,), dtype=mx.float32).at[idx].add(ones_bf16.astype(mx.float32))
    )
    mx.eval(acc_bf16, acc_fp32)
    b = float(acc_bf16[0])
    f = float(acc_fp32[0])
    print(f"  truth = {M}, bf16-accumulator = {b:.1f}, fp32-accumulator = {f:.1f}")
    if b < 0.5 * M:
        print("  -> bf16 scatter accumulates in BF16 (saturates): a bf16 MESH is")
        print("     UNSAFE for the scatter-add; keep the accumulator fp32.")
    else:
        print("  -> bf16 scatter accumulates in fp32 internally: bf16 mesh storage")
        print("     is viable for the scatter path.")


if __name__ == "__main__":
    attribution_and_crux_a()
    crux_b_bf16_scatter()
