"""Benchmark the full differentiable forward model: time + peak RAM scaling.

Where bench_fields.py covers only GRF generation + P(k), this measures the thing
that actually gates a science run: the cost of an end-to-end forward pass and of
its gradient,

    f_NL -> linear_density -> Zel'dovich -> PM leapfrog -> CIC -> local-bias
    tracer -> band power -> scalar,

as a function of mesh resolution N and leapfrog step count, and it compares the
two reverse-mode strategies that matter: replay mx.grad (the unrolled graph) vs
the reversible adjoint (integrate.adjoint_grad_fnl). The gradient peak memory is
the headline -- reverse-mode unrolls the leapfrog, so the autodiff graph, not the
forward arrays, is the binding constraint.

What sets cost (confirmed by the sweeps below):
- Resolution N drives everything. Particles are tied to the mesh (the IC lays
  down one particle per cell, n_particles = N^3), so there is no separate
  "particle size" axis in this toy -- N is it. Memory ~ N^3.
- Box size L (Mpc/h) does NOT affect time or memory: it only rescales the
  k-grid, never the array shapes. The L-invariance check at the end shows this.
- Replay grad memory grows LINEARLY with step count (the graph unrolls). The
  reversible adjoint reconstructs each state by reverse-stepping instead of
  storing it, so its memory is FLAT in step count (~grid only) at ~2x compute --
  the lever for many-step / high-resolution gradients.
- mx.compile of the force solve is ~1x here (FFT-bound); gradient checkpointing
  does NOT reduce memory (the solve is near-linear) -- both measured, see the
  module docstrings. The adjoint is the real memory win.

The linear P(k) backend is EH98 (closed-form) so the CAMB call does not confound
the PM timings. Run: pixi run python scripts/bench_pm.py
"""

import time

import numpy as np
import mlx.core as mx

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import integrate as IG
from mbody import painting as PA
from mbody import bias as B
from mbody import fields as F

BACKEND = "eh98"  # closed-form P(k): isolate PM cost from the CAMB call
B1, B2 = 2.0, 1.0  # local-bias tracer (the b2 term carries the f_NL response)
SEED = 0
F0 = mx.array(50.0)  # f_NL we differentiate at


def _kbins(box, n=4):
    """The first n band-power bins (the large scales that carry the f_NL signal)."""
    return np.arange(1, n + 1) * box.k_fundamental


def _make_loss(box, cosmo, time_, compiled, memory_mode):
    """Scalar loss(f_NL) = sum of the tracer band power of the PM-evolved field."""
    kb = _kbins(box)

    def loss(f_NL):
        x, _ = IG.leapfrog(
            box,
            cosmo,
            time_,
            seed=SEED,
            f_NL=f_NL,
            backend=BACKEND,
            compiled=compiled,
            memory_mode=memory_mode,
        )
        delta = PA.density_contrast(x, box)
        tracer = B.local_bias_tracer(delta, B1, B2)
        return mx.sum(F.band_power(tracer, box, kb))

    return loss


def _measure(call):
    """Wall time (s) and peak GPU memory (MB) of one eval'd call, after a warmup.

    The warmup absorbs the mx.compile trace and the first-touch allocation; the
    peak counter is reset just before the timed call so it reflects that call's
    high-water working set, not the warmup's.
    """
    out = call()
    mx.eval(out)
    del out
    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    out = call()
    mx.eval(out)
    dt = time.perf_counter() - t0
    peak = mx.get_peak_memory() / 1e6
    del out
    mx.clear_cache()
    return dt, peak


def _grad(box, cosmo, time_, compiled, memory_mode):
    gfn = mx.grad(_make_loss(box, cosmo, time_, compiled, memory_mode))
    return _measure(lambda: gfn(F0))


def _fwd(box, cosmo, time_, compiled, memory_mode="replay"):
    loss = _make_loss(box, cosmo, time_, compiled, memory_mode)
    return _measure(lambda: loss(F0))


def _loss_field(box):
    """Positions-only scalar summary, for the adjoint (which seeds its backward
    sweep from d(summary)/d(x_final))."""
    kb = _kbins(box)

    def lf(xf):
        tracer = B.local_bias_tracer(PA.density_contrast(xf, box), B1, B2)
        return mx.sum(F.band_power(tracer, box, kb))

    return lf


def _adjoint(box, cosmo, time_, compiled=True):
    """Wall time + peak RAM of the reversible-adjoint d(summary)/df_NL."""
    lf = _loss_field(box)
    return _measure(
        lambda: IG.adjoint_grad_fnl(
            lf,
            box,
            cosmo,
            time_,
            seed=SEED,
            f_NL=F0,
            backend=BACKEND,
            compiled=compiled,
        )
    )


def sweep_resolution(cosmo, sizes, n_steps):
    """Forward + gradient time and peak RAM vs mesh size N (compiled, replay)."""
    print(f"\n== Resolution sweep (n_steps={n_steps}, compiled, replay) ==")
    print("   N   particles    fwd[s]   grad[s]   fwd_mem[MB]   grad_mem[MB]")
    rows = []
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=n_steps)
    for N in sizes:
        box = BoxConfig(box_size=2.0 * N, n_mesh=N, n_particles=N)
        try:
            fdt, fmem = _fwd(box, cosmo, t, compiled=True)
            gdt, gmem = _grad(box, cosmo, t, compiled=True, memory_mode="replay")
        except Exception as e:  # noqa: BLE001  (OOM / backend hiccup: keep going)
            print(f"{N:4d}   {N**3:9d}   -- {type(e).__name__}: {e}")
            continue
        print(
            f"{N:4d}   {N**3:9d}   {fdt:7.3f}   {gdt:7.3f}   "
            f"{fmem:11.1f}   {gmem:12.1f}"
        )
        rows.append((N, fdt, gdt, fmem, gmem))
    return rows


def sweep_steps(cosmo, N, step_counts):
    """Grad time + peak RAM vs step count: replay mx.grad vs reversible adjoint.

    This is the headline comparison: replay memory grows linearly with the step
    count (the autodiff graph unrolls), while the adjoint reconstructs each state
    by reverse-stepping, so its memory is flat (~grid only) at ~2x the compute.
    """
    print(f"\n== Step-count sweep (N={N}) -- replay mx.grad vs reversible adjoint ==")
    print("  steps   grad_s(rep)  grad_s(adj)   mem_MB(rep)  mem_MB(adj)   mem_save")
    box = BoxConfig(box_size=2.0 * N, n_mesh=N, n_particles=N)
    rows = []
    for ns in step_counts:
        t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=ns)
        try:
            rdt, rmem = _grad(box, cosmo, t, compiled=True, memory_mode="replay")
            adt, amem = _adjoint(box, cosmo, t, compiled=True)
        except Exception as e:  # noqa: BLE001
            print(f"{ns:5d}   -- {type(e).__name__}: {e}")
            continue
        save = rmem / amem if amem else float("nan")
        print(
            f"{ns:5d}   {rdt:10.3f}   {adt:10.3f}   {rmem:10.1f}   "
            f"{amem:10.1f}   {save:6.2f}x"
        )
        rows.append((ns, rdt, adt, rmem, amem))
    return rows


def sweep_compile(cosmo, sizes, n_steps):
    """mx.compile speedup: gradient wall time compiled vs eager (replay)."""
    print(f"\n== Compile speedup (n_steps={n_steps}, replay) ==")
    print("   N   grad_s(eager)  grad_s(compiled)   speedup")
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=n_steps)
    for N in sizes:
        box = BoxConfig(box_size=2.0 * N, n_mesh=N, n_particles=N)
        try:
            edt, _ = _grad(box, cosmo, t, compiled=False, memory_mode="replay")
            cdt, _ = _grad(box, cosmo, t, compiled=True, memory_mode="replay")
        except Exception as e:  # noqa: BLE001
            print(f"{N:4d}   -- {type(e).__name__}: {e}")
            continue
        print(f"{N:4d}   {edt:13.3f}   {cdt:16.3f}   {edt / cdt:6.2f}x")


def check_box_invariance(cosmo, N, n_steps):
    """Confirm cost is independent of box size L at fixed resolution N."""
    print(f"\n== Box-size invariance (N={N}, n_steps={n_steps}, forward) ==")
    print("   L[Mpc/h]   fwd[s]   fwd_mem[MB]")
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=n_steps)
    for L in (128.0, 512.0, 2048.0):
        box = BoxConfig(box_size=L, n_mesh=N, n_particles=N)
        dt, mem = _fwd(box, cosmo, t, compiled=True)
        print(f"{L:9.0f}   {dt:6.3f}   {mem:11.1f}")


def make_plot(res_rows, step_rows, path):
    """Two-panel scaling figure: grad memory vs N, and vs step count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

    if res_rows:
        N = np.array([r[0] for r in res_rows], float)
        gmem = np.array([r[4] for r in res_rows], float)
        ax0.loglog(N**3, gmem, "o-", label="grad peak")
        ref = gmem[0] * (N**3 / N[0] ** 3)  # linear-in-cells guide
        ax0.loglog(N**3, ref, "k--", alpha=0.5, label="$\\propto N^3$")
        ax0.set_xlabel("mesh cells $N^3$")
        ax0.set_ylabel("peak memory [MB]")
        ax0.set_title("Gradient memory vs resolution")
        ax0.legend()

    if step_rows:
        ns = np.array([r[0] for r in step_rows], float)
        rmem = np.array([r[3] for r in step_rows], float)
        amem = np.array([r[4] for r in step_rows], float)
        ax1.plot(ns, rmem, "o-", label="replay mx.grad")
        ax1.plot(ns, amem, "s-", label="reversible adjoint")
        ax1.set_xlabel("leapfrog steps")
        ax1.set_ylabel("grad peak memory [MB]")
        ax1.set_title("Autodiff memory vs step count")
        ax1.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nsaved {path}")


def main():
    cosmo = Cosmology()
    res_rows = sweep_resolution(cosmo, sizes=(16, 24, 32, 48, 64, 96, 128), n_steps=10)
    step_rows = sweep_steps(cosmo, N=64, step_counts=(2, 4, 8, 16, 32))
    sweep_compile(cosmo, sizes=(64, 128), n_steps=10)
    check_box_invariance(cosmo, N=64, n_steps=10)
    make_plot(res_rows, step_rows, "outputs/bench_pm.png")


if __name__ == "__main__":
    main()
