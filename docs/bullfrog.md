# The BullFrog integrator

`integrator="bullfrog"` adds the BullFrog scheme (Rampf, List & Hahn 2024,
[arXiv:2409.19049](https://arxiv.org/abs/2409.19049), JCAP 2025) alongside the
`exact` and `fastpm` leapfrogs. It is a leapfrog-family integrator whose *single
step is 2LPT-accurate* -- where FastPM's step is only 1LPT/Zel'dovich-accurate --
so it converges to the exact solution in fewer steps. Since reverse-mode AD memory
in mbody scales as grid x steps, an integrator that needs fewer steps for the same
accuracy is a direct lever for higher-resolution gradients.

`fastpm` remains the SimConfig default; BullFrog is a selectable option.

## The scheme

BullFrog steps in **growth-factor (D) time** with velocity `v = dx/dD`, as a
**drift-kick-drift** with an **affine** kick (the `alpha != 1` velocity rescale is
what absorbs the second-order growth):

```
x_mid = x + (dD/2) v
v'    = alpha * v + (beta / D_mid) * g(x_mid)
x'    = x_mid + (dD/2) v'
```

with `dD = D(a1) - D(a0)`, `D_mid = D(a0) + dD/2`, and `g` mbody's geometric force
(`div g = -delta`, equal to the paper's acceleration `A`; for a linear mode
`g = +D Psi1`, so it enters the kick with no extra sign). The weights are
(paper Eqs 2.3-2.4)

```
F_mid = (E0 + E0' dD/2) / D_mid - D_mid
alpha = (E1' - F_mid) / (E0' - F_mid)
beta  = 1 - alpha
```

with `E` the second-order growth and `E' = dE/dD`. We use the **EdS relation**
`E = -(3/7) D^2`, `E' = -(6/7) D` (consistent with mbody's EdS 2LPT initial
conditions, `cosmology.growth_factor_2`) evaluated on the **exact LCDM** growth
`D = cosmology.growth_factor`. With `D ∝ a` this reduces to the paper's published
EdS closed form

```
alpha = [4n(4n+1) - 5] / [4n(4n+7) + 7],   beta = [24n + 12] / [4n(4n+7) + 7],
n = D0 / dD,
```

which we verified algebraically and pin in a unit test to 1e-16.

### Fit with mbody

- **Background-only coefficients.** `alpha`, `beta`, `dD`, `D_mid` are float64 CPU
  constants of the step (off the AD graph), exactly like the FastPM kernels, so
  `mx.grad` and the reversible adjoint are unaffected by how they are computed.
- **Reversible.** The affine drift-kick-drift map is exactly invertible
  (`alpha != 0`), so it reuses the reversible-leapfrog adjoint
  (`integrate.adjoint_grad_fnl` / `adjoint_grad_ic`) with a BullFrog-specific
  inverse step.
- **Momentum.** mbody's a-time momentum maps to BullFrog's D-time velocity by
  `v = p / G_f(a)` with `G_f = a^3 E D'` (`integrate._G_f`); the BullFrog path runs
  on `(x, v)` internally and converts `p <-> v` at the IC and the output only. The
  `(3/2) Omega_m` of the kick is absorbed into the D-time formulation (verified: a
  single linear mode grows as `D(a)` exactly).

## Validation (`pixi run bullfrog`, `tests/test_integrate.py`)

| check | result |
|-------|--------|
| weights vs published EdS closed form | match to 1e-16 |
| single linear mode growth | exactly `D(a)` at any step count (1, 2, 4, 8 steps) |
| converges to the FastPM field (12 steps) | agree to the scatter-add floor |
| reversibility (forward then reverse) | reconstructs `x0` to ~1e-6 cells |
| adjoint vs replay `mx.grad` | agree to ~1e-6 (scatter-add floor) |

## The advantage is resolution-gated (the honest finding)

BullFrog's 2LPT accuracy relies on the midpoint force carrying the second-order
mode coupling its kick is calibrated for. In mbody's coarse CIC PM that only holds
when the force is resolved well enough. Self-convergence (`1 - r(k)` of an n-step
run vs the same scheme's high-step limit, large scales, 3 seeds, L=256):

**n_mesh = 64 -- BullFrog wins (~3x fewer steps):**

| n_steps | bullfrog | fastpm | exact |
|--------:|---------:|-------:|------:|
| 2 | 0.00030 | 0.00082 | 0.00142 |
| 3 | 0.00014 | 0.00031 | 0.00058 |
| 5 | 0.00002 | 0.00007 | 0.00014 |

**n_mesh = 32 -- coarse force, no advantage (still converges, still beats exact):**

| n_steps | bullfrog | fastpm | exact |
|--------:|---------:|-------:|------:|
| 2 | 0.00126 | 0.00067 | 0.00161 |
| 3 | 0.00045 | 0.00025 | 0.00069 |
| 5 | 0.00010 | 0.00007 | 0.00016 |

So at `n_mesh=64` BullFrog at 2 steps roughly matches FastPM at 3 and exact at 5;
at `n_mesh=32` the coarse force under-resolves the 2LPT coupling and BullFrog does
not beat FastPM. It always converges to the same field and always beats the
exact-background leapfrog. The takeaway for mbody: BullFrog pays off at the
resolutions where the force is good enough, which is also where the AD-memory
budget matters most.

Reproduce: `pixi run bullfrog` -> `outputs/bullfrog_convergence.png` and the tables
above.
