"""Example 1 -- a whole differentiable PM run from a single config.

Builds Gaussian initial conditions, evolves them with the FastPM + 2LPT defaults
from z = 9 to z = 0, and measures the final field: its power spectrum, how well it
still correlates with the initial conditions, and a diagnostic dashboard figure.

Run:  pixi run python examples/01_quickstart.py
"""

import os

import mbody

os.makedirs("outputs", exist_ok=True)

# A small, fast configuration. SimConfig() defaults to FastPM + 2LPT, z 9 -> 0;
# here we shrink the mesh and step count so the example runs in seconds.
cfg = mbody.SimConfig(
    box=mbody.BoxConfig(box_size=256.0, n_mesh=64, n_particles=64),
    time=mbody.TimeStepping(n_steps=5),
)
print(cfg.summary())

# run() threads the whole config through ic -> LPT -> leapfrog and measures it.
# record=True keeps a (bounded) trajectory so the dashboard can show growth too.
result = mbody.run(cfg, backend="eh98", record=True)

# The measured nonlinear power spectrum of the z = 0 field.
k, Pk, n_modes = result.power()
print("\nP(k) of the z=0 field (first few bins):")
for ki, pi in zip(k[:5], Pk[:5]):
    print(f"  k = {ki:6.3f} h/Mpc   P = {pi:11.1f} (Mpc/h)^3")

# The propagator r(k): how well the evolved phases still track the linear ICs.
# It is ~1 on large scales and decoheres toward small scales.
kc, r, _ = result.cross_with_ic()
print(f"\npropagator r(k): {r[0]:.3f} (largest scale) -> {r[-1]:.3f} (smallest)")

# A multi-panel diagnostic dashboard (imports matplotlib lazily).
out = result.dashboard(out="outputs/quickstart.png")
print(f"\nwrote {out}")
