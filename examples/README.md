# Examples

Short, commented, runnable end-to-end scripts -- the friendly entry point to the
package. Each is self-contained and fast (small meshes, a few steps). Run any of
them with pixi:

```bash
pixi run python examples/01_quickstart.py
```

- **`01_quickstart.py`** -- one differentiable PM run from a single `SimConfig`:
  measure P(k), the propagator r(k), and write a diagnostic dashboard.
- **`02_dlnp_dfnl.py`** -- the headline: autodiff `dlnP/df_NL` of a biased tracer
  vs the absolute Dalal prediction, with the matter-field null.
- **`03_fisher_forecast.py`** -- a Fisher forecast over `{f_NL, b1, b2, A}`;
  marginalized vs conditional constraints (the f_NL-bias degeneracy).
- **`04_redshift_space.py`** -- redshift-space multipoles `P_0`, `P_2` from an
  opt-in RSD run.

For the full public API see [`../docs/api.md`](../docs/api.md); for the richer
figure-making experiments and benchmarks see [`../scripts/`](../scripts/).
