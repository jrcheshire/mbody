"""Demonstrate the differentiable-resolution ceiling for real.

We have only ever EXTRAPOLATED the reversible-adjoint dlnP/df_NL ceiling from a
B/cell fit (~596 B/cell -> N_max ~617 @128GB). This actually RUNS the headline
gradient -- the default forward model (2LPT IC -> FastPM leapfrog -> CIC -> local
-bias tracer -> band power) differentiated w.r.t. f_NL through the reversible
adjoint with recompute (the committed memory lever) -- at a large mesh N, and
reports the true peak GPU memory and B/cell.

Two facts make this cheap and honest:
- The reversible adjoint is O(1) in step count, so peak memory is INDEPENDENT of
  n_steps. This uses a small n_steps to stay fast; the peak equals a many-step run.
- Box size L does not affect memory either (only the k-grid scale), so a big box
  at this N is free.

Run ONE N per process so each gets a fresh allocator and an OOM is isolated, and
output is flushed per line -- a near-wall OOM at the top rung does NOT lose the
rungs below it. Ascending ladder, stop at the first failure:

    for N in 448 512 544 576 600 617; do
      pixi run python scripts/demo_adjoint_ceiling.py $N || break
    done

617^3 sits at ~135-140 GB (~575-596 B/cell), i.e. right at the 128 GB (137 GB)
physical wall -- so the top rungs are expected to be where it actually stops.
"""

import sys
import time

import threading

import numpy as np
import mlx.core as mx

from mbody.config import BoxConfig, Cosmology, TimeStepping
from mbody import integrate as IG, painting as PA, bias as B, fields as F


def _heartbeat(N, t0, stop, period=5.0):
    """Flush elapsed time + live peak GB every `period` s until `stop` is set.

    Near the macOS jetsam threshold a run is SIGKILL'd (uncatchable), so this
    leaves a trail: the last heartbeat before the kill shows how far it got and
    the peak memory it had reached -- distinguishing an early allocation refusal
    from a compressor-fill death right at the high-water mark.
    """
    while not stop.wait(period):
        el = time.perf_counter() - t0
        print(
            f"  [N={N}] t={el:5.0f}s  peak={mx.get_peak_memory()/1e9:6.1f} GB",
            flush=True,
        )


def main(N, n_steps=2):
    cosmo = Cosmology()
    # L = 2N (Mpc/h) is arbitrary -- memory is box-size-independent. One particle
    # per cell (n_particles = N), so N^3 particles, the toy's resolution knob.
    box = BoxConfig(box_size=2.0 * N, n_mesh=N, n_particles=N)
    t = TimeStepping(z_init=9.0, z_final=0.0, n_steps=n_steps)
    kb = np.arange(1, 4) * box.k_fundamental  # low-k bins carry the f_NL signal

    def loss_field(xf):
        tracer = B.local_bias_tracer(PA.density_contrast(xf, box), 2.0, 1.0)
        return mx.sum(F.band_power(tracer, box, kb))

    mx.clear_cache()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    stop = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(N, t0, stop), daemon=True)
    hb.start()
    # recompute_cic=True is the default (the committed memory lever); fastpm + 2LPT
    # are the default forward model. This is the headline gradient, at scale.
    try:
        g = IG.adjoint_grad_fnl(
            loss_field, box, cosmo, t, seed=0, f_NL=mx.array(50.0), backend="eh98"
        )
        mx.eval(g)
    finally:
        stop.set()
    dt = time.perf_counter() - t0
    peak = mx.get_peak_memory()
    npart = N**3
    print(
        f"N={N:4d}  particles={npart:>13,d}  peak={peak/1e9:7.2f} GB  "
        f"B/cell={peak/npart:6.1f}  grad={float(g):.6e}  {dt:6.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    N = int(sys.argv[1])
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    try:
        main(N, n_steps)
    except Exception as e:  # noqa: BLE001  (OOM at the wall: report and signal stop)
        print(f"N={N:4d}  FAILED: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
