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
  4. CRUX C (slab-tiling: tried, MEASURED, NOT merged) -- a slab-tiled CIC VJP
     (the per-particle paint/read sum split into S eager index slabs). It is EXACT
     (validate_tiling_exact) and shrinks an ISOLATED force-solve VJP ~2.2x
     (crux_c_tiling), BUT integrating it into the per-step adjoint added only ~6%
     over the committed recompute and did NOT scale with S (crux_c_full_adjoint:
     S=4..32 all plateau at 560 B/cell vs recompute 596 at N=256). The full-adjoint
     peak is FFT + adjoint-state bound, not CIC-transient bound, so tiling was
     dropped -- recompute (crux A) stays the lever.

Budget-capped: max N=288 -> ~23 GB transient peak (safe alongside other use).
The decision ratios are N-independent, so small N suffices. Memory peaks are
read from mx.get_peak_memory() (reset before each measured call). A RANDOM
cotangent is used on the forces -- an all-ones cotangent is a momentum-
conservation null (sum of forces ~0), which would zero the gradient.

Run: pixi run python scripts/probe_adjoint_memory.py
"""

import mlx.core as mx
import numpy as np

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import integrate as IG, painting as PA, forces as FO
from mbody import bias as B, fields as F
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


# --- Crux C: slab-tiled CIC VJP (tried, MEASURED, NOT merged) ----------------
#
# The CIC paint/read are a sum over particles, so they partition EXACTLY by
# particle index: cic_paint(x) = sum_slabs cic_paint(x[slab]). Evaluating the
# slabs EAGERLY (one mx.vjp + mx.eval each, then released) caps the reverse-mode
# CIC transient at ~1/S of the monolithic VJP while the O(N^3) mesh arrays stay
# whole. These helpers are LOCAL to the probe because tiling was NOT merged:
#   * In ISOLATION it works -- the slab-tiled force-solve VJP is ~2.2x below the
#     monolithic / recompute force VJP (crux_c_tiling), exact to the scatter floor.
#   * In the FULL reversible adjoint it does almost nothing: integrating it into
#     the per-step backward added only ~6% over the already-committed recompute
#     (crux A) and DID NOT scale with S -- S=4,8,16,32 all plateaued at 560 B/cell
#     vs recompute 596, raw 992 at N=256 (crux_c_full_adjoint). The full-adjoint
#     peak is set by the FFT Poisson VJP + the persistent adjoint state +
#     in-flight forward fields, NOT the CIC transient, so tiling cannot reach it.
# Conclusion: recompute (crux A) is the lever; the residual ~596 B/cell floor
# needs the FFT (multigrid, wrong-order, deferred), not CIC tiling. A 9-cell N_max
# gain (~617->626 @128GB) did not justify a second hand-assembled KDK backward in
# the hot path. These helpers reproduce the isolated win + exactness; crux_c_full_
# adjoint reproduces the floor tiling could not beat.


def _slab_slices(n_part, n_slabs):
    """Contiguous index ranges partitioning [0, n_part) into n_slabs groups."""
    b = [round(i * n_part / n_slabs) for i in range(n_slabs + 1)]
    return [(b[i], b[i + 1]) for i in range(n_slabs)]


def tiled_paint_vjp(positions, box, cot_mesh, n_slabs):
    """d/d positions of <cot_mesh, cic_paint(positions)>, over n_slabs eager
    particle-index slabs (exact partition; bit-identical to mx.vjp(cic_paint))."""
    grads = []
    for lo, hi in _slab_slices(positions.shape[0], n_slabs):
        xs = positions[lo:hi]
        _, (g,) = mx.vjp(lambda x: PA.cic_paint(x, box), [xs], [cot_mesh])
        mx.eval(g)
        grads.append(g)
    return mx.concatenate(grads, axis=0)


def tiled_read_vector_vjp(fx, fy, fz, positions, box, cot_out, n_slabs):
    """VJP of cic_read_vector over n_slabs eager slabs: (d_fx, d_fy, d_fz,
    d_positions). Field cotangents summed across slabs (a scatter-add, ~1e-6); the
    position gradient is slab-local (bit-identical)."""
    dfx, dfy, dfz = mx.zeros_like(fx), mx.zeros_like(fy), mx.zeros_like(fz)
    pgrads = []
    for lo, hi in _slab_slices(positions.shape[0], n_slabs):
        xs, cs = positions[lo:hi], cot_out[lo:hi]
        _, (gfx, gfy, gfz, gx) = mx.vjp(
            lambda a, b, c, x: PA.cic_read_vector(a, b, c, x, box),
            [fx, fy, fz, xs],
            [cs],
        )
        dfx, dfy, dfz = dfx + gfx, dfy + gfy, dfz + gfz
        mx.eval(dfx, dfy, dfz, gx)
        pgrads.append(gx)
    return dfx, dfy, dfz, mx.concatenate(pgrads, axis=0)


def tiled_force_vjp(positions, box, cot_out, n_slabs):
    """Slab-tiled VJP of forces_on_particles: recompute the forward solve, then
    the tiled read VJP, the linear FFT VJP, and the tiled paint VJP (recompute and
    tiling stacked). Exact to the scatter floor; ~2.2x below the monolithic force
    VJP in isolation -- but see the module note: it does NOT translate to the full
    adjoint, so it lives here, not in the library."""
    delta = PA.density_contrast(positions, box)
    gx, gy, gz = FO.acceleration_field(delta, box)
    mx.eval(delta, gx, gy, gz)
    dgx, dgy, dgz, dpos_read = tiled_read_vector_vjp(
        gx, gy, gz, positions, box, cot_out, n_slabs
    )
    _, (d_delta,) = mx.vjp(
        lambda dd: mx.stack(FO.acceleration_field(dd, box)),
        [delta],
        [mx.stack([dgx, dgy, dgz])],
    )
    mean = positions.shape[0] / box.n_mesh**3
    dpos_paint = tiled_paint_vjp(positions, box, d_delta / mean, n_slabs)
    mx.eval(dpos_read, dpos_paint)
    return dpos_read + dpos_paint


def _peak_of(thunk):
    """Peak GPU bytes of evaluating thunk(), after a warmup.

    The warmup absorbs first-touch allocation / compile-cache priming (an
    un-warmed cold call allocates differently), matching vjp_peak / bench_pm's
    _measure so the tiled and monolithic numbers are comparable. The peak counter
    is reset just before the timed call.
    """
    res = thunk()
    leaves = res if isinstance(res, (tuple, list)) else (res,)
    mx.eval(*leaves)
    del res, leaves
    mx.clear_cache()
    mx.reset_peak_memory()
    res = thunk()
    leaves = res if isinstance(res, (tuple, list)) else (res,)
    mx.eval(*leaves)
    peak = mx.get_peak_memory()
    del res
    mx.clear_cache()
    return peak


def _max_abs(a, b):
    return float(mx.max(mx.abs(a - b)))


def validate_tiling_exact(N=96, n_slabs=8):
    """Tiled paint/read VJP vs the monolithic mx.vjp: max abs gradient diff.

    Expected ~ the float32 CIC scatter-add floor (reassociating the sum in slab
    order), the same ~1e-6 floor as adjoint-vs-replay -- NOT bit-identical.
    """
    cosmo = Cosmology()
    box = BoxConfig(box_size=2.0 * N, n_mesh=N, n_particles=N)
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=2)
    x, _ = IG.initial_state(box, cosmo, t, seed=SEED, backend="eh98")
    mx.eval(x)
    delta = PA.density_contrast(x, box)
    gx_f, gy_f, gz_f = FO.acceleration_field(delta, box)
    mx.eval(gx_f, gy_f, gz_f)
    n_part = x.shape[0]
    cot_mesh = mx.random.normal((N, N, N), dtype=P.REAL)
    cot_out = mx.random.normal((n_part, 3), dtype=P.REAL)
    mx.eval(cot_mesh, cot_out)

    _, (gp_ref,) = mx.vjp(lambda xx: PA.cic_paint(xx, box), [x], [cot_mesh])
    gp_til = tiled_paint_vjp(x, box, cot_mesh, n_slabs)
    _, (rfx, rfy, rfz, rgx) = mx.vjp(
        lambda a, b, c, xx: PA.cic_read_vector(a, b, c, xx, box),
        [gx_f, gy_f, gz_f, x],
        [cot_out],
    )
    tfx, tfy, tfz, tgx = tiled_read_vector_vjp(
        gx_f, gy_f, gz_f, x, box, cot_out, n_slabs
    )
    _, (gf_ref,) = mx.vjp(lambda xx: FO.forces_on_particles(xx, box), [x], [cot_out])
    gf_til = tiled_force_vjp(x, box, cot_out, n_slabs)
    mx.eval(gp_ref, gp_til, rfx, rfy, rfz, rgx, tfx, tfy, tfz, tgx, gf_ref, gf_til)

    print(f"\n== Crux C exactness (N={N}, S={n_slabs}, vs monolithic mx.vjp) ==")
    print(f"  paint d_pos      max|diff| = {_max_abs(gp_ref, gp_til):.2e}")
    print(f"  read  d_pos      max|diff| = {_max_abs(rgx, tgx):.2e}")
    print(f"  read  d_field_x  max|diff| = {_max_abs(rfx, tfx):.2e}")
    print(f"  force d_pos      max|diff| = {_max_abs(gf_ref, gf_til):.2e}")


def crux_c_tiling(sizes=(224, 288), slabs=(4, 8)):
    """Peak GPU memory of the slab-tiled force-solve VJP vs the monolithic and
    recompute (crux A) baselines -- the ISOLATED win (~2.2x, til/rec ~0.45).

    tiled_force_vjp is the full solve with the two CIC pieces slab-tiled and the
    linear FFT VJP between them. This shows tiling DOES shrink an isolated force
    VJP; crux_c_full_adjoint shows it does NOT carry into the full adjoint (the
    floor is elsewhere), which is why it was not merged. All measured with the
    same warmed harness so the ratios are honest.
    """
    cosmo = Cosmology()
    print("\n== Crux C: slab-tiled force-solve VJP peak vs mono / recompute ==")
    print(
        "  (peak GB; mono = monolithic mx.vjp; rec = crux-A checkpoint; til = S slabs)"
    )
    print(
        f"{'N':>5} {'S':>3} {'mono':>7} {'rec':>7} {'tiled':>7} "
        f"{'til/mono':>8} {'til/rec':>8} {'B/cell':>7}"
    )
    g = 1e9
    for N in sizes:
        box = BoxConfig(box_size=2.0 * N, n_mesh=N, n_particles=N)
        t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=2)
        x, _ = IG.initial_state(box, cosmo, t, seed=SEED, backend="eh98")
        mx.eval(x)
        cot_out = mx.random.normal((x.shape[0], 3), dtype=P.REAL)
        mx.eval(cot_out)

        def mono_vjp(x=x, box=box, cot=cot_out):
            def fwd(xx, box=box):
                return FO.forces_on_particles(xx, box)

            return mx.vjp(fwd, [x], [cot])[1]

        def rec_vjp(x=x, box=box, cot=cot_out):
            return mx.vjp(FO._cic_checkpointed_force(box), [x], [cot])[1]

        mono = _peak_of(mono_vjp)
        rec = _peak_of(rec_vjp)
        for S in slabs:
            til = _peak_of(
                lambda S=S, x=x, box=box, cot=cot_out: tiled_force_vjp(x, box, cot, S)
            )
            print(
                f"{N:5d} {S:3d} {mono/g:7.2f} {rec/g:7.2f} {til/g:7.2f} "
                f"{til/mono:7.2f}x {til/rec:7.2f}x {til/N**3:7.1f}"
            )


def crux_c_full_adjoint(N=256, n_steps=8):
    """The DECISIVE measurement: the full reversible-adjoint peak with recompute
    off vs on. This is why slab-tiling was not merged.

    Recompute (crux A) takes raw -> recompute (992 -> 596 B/cell at N=256, ~1.7x);
    the residual 596 floor is the FFT Poisson VJP + the persistent adjoint state +
    in-flight forward fields, NOT the CIC reverse transient. Integrating the tiled
    force VJP into the per-step backward (measured during this thread) plateaued at
    560 B/cell for EVERY S in {4,8,16,32} -- a flat ~6% that does not scale with
    slabs -- so tiling was dropped and recompute kept as the lever. This function
    reproduces the recompute floor that tiling could not beat (the tiled-adjoint
    integration itself is not in the tree; the plateau numbers are recorded here).
    """
    cosmo = Cosmology()
    box = BoxConfig(box_size=2.0 * N, n_mesh=N, n_particles=N)
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=n_steps)
    kb = np.arange(1, 4) * box.k_fundamental

    def loss_field(xf):
        tracer = B.local_bias_tracer(PA.density_contrast(xf, box), 2.0, 1.0)
        return mx.sum(F.band_power(tracer, box, kb))

    def adj_peak(recompute):
        return _peak_of(
            lambda: IG.adjoint_grad_fnl(
                loss_field,
                box,
                cosmo,
                t,
                seed=SEED,
                f_NL=mx.array(50.0),
                backend="eh98",
                recompute_cic=recompute,
            )
        )

    g = 1e9
    print(
        f"\n== Crux C full adjoint (N={N}, {n_steps} steps): recompute IS the lever =="
    )
    raw = adj_peak(False)
    rec = adj_peak(True)
    print(f"  raw (no recompute)  {raw/g:6.2f} GB  {raw/N**3:6.1f} B/cell")
    print(
        f"  recompute (default) {rec/g:6.2f} GB  {rec/N**3:6.1f} B/cell  "
        f"({raw/rec:.2f}x off raw)"
    )
    print("  tiled S=4..32 (measured when integrated): all ~560 B/cell -> +6% only,")
    print("  flat in S; the floor is FFT+state, so slab-tiling was NOT merged.")


if __name__ == "__main__":
    attribution_and_crux_a()
    crux_b_bf16_scatter()
    validate_tiling_exact()
    crux_c_tiling()
    crux_c_full_adjoint()
