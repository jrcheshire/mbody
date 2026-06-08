"""Example 4 -- redshift-space multipoles from an opt-in RSD run.

Enable RedshiftSpace on the config and the driver maps the final particles to
redshift space (a line-of-sight peculiar-velocity shift) before measuring, so the
run exposes the anisotropic multipoles P_0 (monopole) and P_2 (quadrupole). A
positive large-scale quadrupole is the Kaiser linear-RSD boost.

Run:  pixi run python examples/04_redshift_space.py
"""

import mbody

# RSD is opt-in (the honest-config default measures the real-space field). Use a
# full FFT axis (0 or 1) for the line of sight, never the rfft axis 2.
cfg = mbody.SimConfig(
    box=mbody.BoxConfig(box_size=256.0, n_mesh=64, n_particles=64),
    time=mbody.TimeStepping(n_steps=5),
    rsd=mbody.RedshiftSpace(enabled=True, los_axis=0, f_growth=1.0),
)
print(cfg.summary())

result = mbody.run(cfg, backend="eh98")

# With RSD enabled the run exposes the multipoles (real-space P(k) is still
# available via result.power()). Pell is a dict {ell: P_ell}.
k, Pell, n_modes = result.power_multipoles(ells=(0, 2))
P0, P2 = Pell[0], Pell[2]

print("\nredshift-space multipoles of the z=0 field:")
print(f"{'k [h/Mpc]':>10} {'P0':>12} {'P2':>12} {'P2/P0':>8}")
for ki, p0, p2 in zip(k, P0, P2):
    print(f"{ki:10.3f} {p0:12.1f} {p2:12.1f} {p2 / p0:8.3f}")
print("\nThe large-scale quadrupole is the Kaiser boost; trust only")
print("k < ~0.1 h/Mpc (a PM has no fingers-of-god).")
