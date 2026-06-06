# Science path: b_phi non-universality (mbody -> HETDEX -> SPHEREx)

Status (2026-06-05): plan recorded; Phase 1 not yet started.

## Why this doc exists

`ROADMAP.md` is the engineering plan (what to build, smallest-runnable-thing
first). This doc is the science plan (why), and its job is to keep three threads
connected so the b_phi work does not drift into a pure toy milestone:

- **mbody** -- the differentiable forward model, whose named Stretch goal is a
  first-principles b_phi and the universality relation. This is the *engine*.
- **SPHEREx** -- where b_phi universality is the load-bearing *assumption*; its
  failure is the dominant systematic in the f_NL error budget.
- **HETDEX** -- the largest LAE sample ever assembled, a real high-z strongly
  selected tracer where b_phi has never been measured or computed. The *data*.

The spine is: build the b_phi engine, then work toward a HETDEX data test.

## The question

b_phi (the local-PNG / amplitude bias) is the response of a tracer's number
density to a long-wavelength modulation of the small-scale fluctuation amplitude.
In separate-universe language b_phi is proportional to d ln n_g / d ln sigma_8
(conventions differ by an O(1) factor). It enters clustering as a
scale-dependent bias

    b(k) = b1 + b_phi * f_NL / M(k),   M(k) proportional to k^2 T(k) D(z),

so the large-scale local-f_NL signal goes as 1/k^2. The **universality relation**

    b_phi = 2 * delta_c * (b1 - 1),   delta_c = 1.686,

holds *iff* the tracer abundance depends only on peak significance
nu = delta_c / sigma -- i.e. pure mass selection. It is known to fail for
realistically selected galaxies (see references): stellar-mass, color, and
sSFR selections all give b_phi = 2 delta_c (b1 - p) with p in ~[0.4, 0.7], and
assembly bias (concentration split at fixed b1 ~ 2) swings b_phi from ~ -1 to
~ +9.

**LAEs are the sharp case.** b1 ~ 1.7-2.1, host halos log(Mh) ~ 11.1-11.5, but
they occupy only a few percent of available halos -- a strong secondary
selection -- and the actual selection (Lyman-alpha escape) keys on dust, HI
column, and ISM kinematics, all correlated with assembly history and none
captured by halo mass. No one has computed b_phi for a Lyman-alpha-radiative-
transfer-selected sample. The question is therefore concrete and open:

> **Is LAE b_phi universal, and if not, what selection property drives the
> deviation?**

This is the same question SPHEREx's f_NL error budget hangs on, in an
unexplored corner of tracer-selection space.

## The crux (what is and is not measurable)

The clustering observable is the *product* b_phi * f_NL. Planck pins f_NL near
zero (sigma ~ 5), and HETDEX's effective volume at z ~ 2-3 (~90 deg^2, a
fraction of a Gpc^3/h^3) is far too small to detect a 1/k^2 upturn of that
amplitude. So:

- **b_phi cannot be read off HETDEX's own scale-dependent bias.** That route is
  dead.
- b_phi must instead be **modeled** (separate-universe response: sims / mbody)
  or constrained **indirectly** (does the clustering amplitude depend on a
  selection proxy?).

This is what dictates the phasing below.

## Where mbody stands (anchor to ROADMAP.md)

- ROADMAP **Step 5** (the headline dlnP/df_NL on a biased tracer) already
  produces the correct 1/k^2 shape; its amplitude is currently shape-only
  (normalized at the largest scale). That amplitude *is* b_phi up to
  calibration -- so the engine is ~80% built.
- The reversible-leapfrog adjoint (`integrate.adjoint_grad_fnl`) gives
  O(1)-in-steps gradients and is the natural vehicle to extend the response
  from d/df_NL to d/d ln sigma_8 (ROADMAP Stretch: "generalize the adjoint IC
  step to cosmology params").
- **Scope, honestly:** mbody is a PM toy with no halos (FoF is non-
  differentiable, a PM under-resolves). So mbody computes b_phi for a
  *field-level* bias model, not a resolved-halo population. That is the right
  altitude for "how does *selection* bend b_phi": non-universality is induced
  by making the tracer respond to a *second field* (an assembly / escape
  proxy), not just delta.

## Phased plan

### Phase 1 -- the b_phi engine (mbody, self-contained)  [A1 + A2]

The capstone. Builds directly on Step 5.

- Variance-modulation tracer: n_g(x) proportional to F(delta_smooth(x),
  sigma_local(x)), where sigma_local is the local RMS of small-scale modes.
- b_phi by **autodiff** as the separate-universe amplitude response,
  `mx.grad` of ln <n_g> with respect to ln sigma_8 -- no finite-difference SU
  pairs.
- **Validate universality:** a mass-proxy threshold tracer (soft threshold on
  smoothed delta) should recover b_phi = 2 delta_c (b1 - 1).
- **Break it:** add a secondary-field dependence (assembly / concentration /
  escape proxy) and watch b_phi deviate from universality.
- Deliverable: a differentiable b_phi plus the sensitivity d(b_phi)/d(selection
  params) -- "which selection property drives non-universality" -- which a
  finite-difference SU study cannot get as cleanly.
- Validation discipline: cross-check the b_phi normalization against the
  analytic peak-background-split limit, and later against a real SU sim
  (pmwd / Quijote-PNG) before any number leaves the toy.

### Phase 2 -- toward the HETDEX data test  [A3 bridge -> A4]

The spine bends to real data here. A4 is the goal; A3 is the bridge that
de-risks it (tells you what selection coupling to look for) and doubles as the
SPHEREx deliverable.

**A3 -- HETDEX-anchored semi-analytic b_phi (the SPHEREx bridge).**
Constrain the LAE HOD from HETDEX b1 (clustering) + abundance (luminosity
function) + dn/dz, then map to b_phi via a peak-background-split HOD response,
or via the Barreira 2025 relation between b_phi and the number-density redshift
evolution (HETDEX gives dn/dz directly; flag the relation's assumptions).
Deliverable: a b_phi *prior* (value + uncertainty) for high-z strongly selected
tracers, usable by SPHEREx.

**A4 -- HETDEX differential clustering (the data test).** The one idea that
uses what makes SC2 special -- the full 1D spectra -- and sidesteps the crux.
Split the LAEs by a Lyman-alpha escape proxy read off the 1036-channel spectra
(rest-frame EW, line-profile asymmetry, velocity offset), and measure the
*relative* clustering bias between subsamples. If b1 depends on the escape
proxy, the selection couples to halo assembly -- direct evidence that LAE b_phi
is non-universal, with a handle on sign and strength. This is the ordinary
clustering amplitude (no tiny-f_NL term), so it is detectable in HETDEX.

- **Open dependency:** A4 needs a HETDEX window / random catalog. SC2 does not
  appear to ship randoms -- to confirm against the readme and the HETDEX survey
  / catalog papers when we reach this phase. Fallback: build the selection
  function from the detinfo `flux_noise_1sigma` sensitivity + the IFU footprint.
- A4 is its own project (a clustering estimator + a correct window); gate it on
  Phase 1, informed by A3.

## How the phases compose

A4 measures, from data, how strongly LAE selection couples to halo properties
-> A1/A2 convert that coupling into a b_phi via the differentiable engine ->
A3 packages it as the b_phi prior SPHEREx needs. mbody is the through-line;
HETDEX supplies the real selection function and the target numbers.

## Scope and honesty

- mbody is a toy: field-level bias, no halos; the clean Dalal 1/k^2 is a
  linear-theory result. Its b_phi outputs are intuition + sensitivity, to be
  cross-checked against SU sims before any external use.
- HETDEX's own f_NL constraint is not competitive; the value here is b_phi
  *calibration*, not a new f_NL number.

## Numbers to verify before they leave this repo

These are literature-agent values gathered during the 2026-06-05 brainstorm;
verify against the primary sources before any goes into a paper, posting, or
caption:

- LAE b1 ~ 1.7-2.1, log(Mh) ~ 11.1-11.5 at z ~ 2.4-3.1 (ODIN, arXiv:2503.17824).
- LAE halo occupation a few percent (drives the "strong secondary selection"
  claim) -- verify.
- Barrera-Blanco et al. 2023 (arXiv:2303.10337): H-alpha ELGs show ~ no b_phi
  assembly bias at z = 1 -- verify scope (it is H-alpha at z=1, not Lya at
  z~2-3).
- HETDEX single-tracer sigma(f_NL) ~ 10-20 (estimate; no published HETDEX f_NL
  forecast exists).

## References

- Dalal et al. 2008 (arXiv:0710.4560) -- f_NL scale-dependent bias.
- Slosar et al. 2008 (arXiv:0805.3580) -- b_phi for general tracers.
- Seljak 2009 (arXiv:0807.1770) -- multitracer cosmic-variance cancellation.
- Barreira et al. 2020 (arXiv:2006.09368, 2009.06622), 2021 (arXiv:2107.06887),
  2022 (arXiv:2205.05673), 2023 (arXiv:2303.08901), 2025 (arXiv:2503.21736) --
  b_phi non-universality, tolerances, ML b_phi, dn/dz relation.
- Lazeyras et al. 2022 (arXiv:2209.07251), 2023 (arXiv:2311.10088) -- assembly
  bias in b_phi; mitigation.
- Barrera-Blanco et al. 2023 (arXiv:2303.10337) -- ELG b_phi assembly bias.
- ODIN LAE clustering (arXiv:2503.17824, 2406.01803) -- b1, halo masses.
- HETDEX Source Catalog 2 / PDR1 (arXiv:2606.04208) -- the dataset.
- SPHEREx: Dore et al. 2014 (arXiv:1412.4872), Karagiannis et al. 2023
  (arXiv:2311.13082), Bock et al. 2025 (arXiv:2511.02985).

## Pointers

- HETDEX data: `~/hetdex_source_catalog_2/`; exploration scripts + figures:
  `~/hetdex_explore/`.
- Synergy memory (SPHEREx project): `hetdex_sc2_dataset.md`.
- mbody engineering plan: `ROADMAP.md` (Step 5, Stretch); rationale in
  `CLAUDE.md`; mbody memory `project_mbody_overview`, `reference_fnl_bphi_target`.
- SPHEREx b_phi context (SPHEREx project memory):
  `project_sr_scale_dependent_bias`, `project_differentiable_fnl_sideproject`.
