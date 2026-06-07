"""The single run driver: one SimConfig in, one reproducible result out.

mbody's modules (cosmology, ic, lpt, forces, integrate, diagnostics) each take
their parameters loose. ``run(cfg)`` is the one place that threads a whole
SimConfig through them, so a run is reproducible from one object and a script
need only build a config and call this. The returned RunResult bundles the final
state, the initial and final density fields, and an optional trajectory recorder,
and knows how to render the standard dashboard and serialize itself.

Honest config. The SimConfig enums reserve names for physics that is planned but
not yet built -- integrator "fastpm"/"bullfrog" and lpt_order 2 (2LPT). The
implemented baseline is integrator "exact" (exact-background leapfrog) and
lpt_order 1 (Zel'dovich), which are the SimConfig defaults, so ``run(SimConfig())``
works out of the box. Asking for an unimplemented option raises
NotImplementedError rather than silently running something else.

This driver runs the forward model. Gradients in f_NL use the dedicated
integrate.adjoint_grad_fnl (the reversible-adjoint entry point), not this path.
"""

import dataclasses
import json
import os

import numpy as np

from mbody import fields as F
from mbody import integrate as IG
from mbody import rsd as RS
from mbody.diagnostics import SnapshotRecorder, cross_correlation, particle_power


def _check_supported(cfg):
    """Reject SimConfig options that are reserved in the enums but not built.

    All three integrators -- 'exact' (exact-background leapfrog), 'fastpm'
    (growth-corrected KDK), and 'bullfrog' (2LPT-accurate drift-kick-drift) -- are
    implemented; this guard remains so any future reserved enum value fails loudly
    rather than silently running something else.
    """
    if cfg.time.integrator not in ("exact", "fastpm", "bullfrog"):
        raise NotImplementedError(
            f"integrator={cfg.time.integrator!r} is reserved in the config but "
            "not implemented; choose 'exact', 'fastpm', or 'bullfrog'."
        )


@dataclasses.dataclass(eq=False)
class RunResult:
    """The output of a single forward run, with its diagnostics on tap.

    Holds the config that produced it, the final particle state (x, p), the
    initial and final CIC density contrasts, the backend used for linear theory,
    and -- if the run was asked to record -- the SnapshotRecorder with the
    trajectory. The methods are thin wrappers over mbody.diagnostics / mbody.viz
    so a run measures and renders itself from one object.
    """

    config: object
    x: object
    p: object
    ic_field: object
    final_field: object
    backend: str
    recorder: object = None
    redshift_field: object = None

    def power(self, **kwargs):
        """Measured P(k) of the final field (k_centers, P_k, n_modes)."""
        return particle_power(self.x, self.config.box, **kwargs)

    def power_multipoles(self, ells=(0, 2), **kwargs):
        """Redshift-space multipoles P_ell(k) of the final field.

        Requires the run to have been made with cfg.rsd.enabled (so the
        redshift-space, interlaced density field is available). Returns
        (k_centers, {ell: P_ell}, n_modes); the CIC window is deconvolved and the
        discrete-shell mode-coupling decoupled. Extra keywords pass through to
        fields.power_multipoles (dk, kmax, ...).
        """
        if self.redshift_field is None:
            raise ValueError(
                "no redshift-space field; run with cfg.rsd = RedshiftSpace(enabled=True)"
            )
        return F.power_multipoles(
            self.redshift_field,
            self.config.box,
            ells=ells,
            los_axis=self.config.rsd.los_axis,
            deconvolve_cic=True,
            **kwargs,
        )

    def cross_with_ic(self, **kwargs):
        """Cross-correlation r(k) of the final field with the linear IC."""
        return cross_correlation(
            self.final_field, self.ic_field, self.config.box, **kwargs
        )

    def growth_history(self):
        """(a, R) large-scale growth across the run (requires record=True)."""
        if self.recorder is None:
            raise ValueError("run was not recorded; call run(cfg, record=True)")
        return self.recorder.growth_history()

    def dashboard(self, out="outputs/dashboard.png", title=None):
        """Render the four-panel diagnostic dashboard to `out`. Returns `out`.

        Imports mbody.viz lazily so ``import mbody`` stays matplotlib-free.
        """
        from mbody import viz

        t = self.config.time
        return viz.dashboard(
            self.config.box,
            self.config.cosmology,
            self.ic_field,
            self.final_field,
            recorder=self.recorder,
            z_init=t.z_init,
            z_final=t.z_final,
            out=out,
            title=title if title is not None else self.config.ic.kind,
            backend=self.backend,
            summary=self.config.summary(),
        )

    def save(self, path, dashboard=True):
        """Write the run to a directory: config.json, fields/state as .npy, and
        (by default) the dashboard PNG. Returns `path`.
        """
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "config.json"), "w") as fh:
            json.dump(dataclasses.asdict(self.config), fh, indent=2)
        np.save(os.path.join(path, "x_final.npy"), np.asarray(self.x))
        np.save(os.path.join(path, "p_final.npy"), np.asarray(self.p))
        np.save(os.path.join(path, "ic_field.npy"), np.asarray(self.ic_field))
        np.save(os.path.join(path, "final_field.npy"), np.asarray(self.final_field))
        if self.redshift_field is not None:
            np.save(
                os.path.join(path, "redshift_field.npy"),
                np.asarray(self.redshift_field),
            )
        if dashboard:
            self.dashboard(out=os.path.join(path, "dashboard.png"))
        return path


def run(cfg, backend="camb", record=False, recorder=None, spacing="linear"):
    """Run the forward model specified by `cfg` (a SimConfig). Returns RunResult.

    Builds Zel'dovich initial conditions from cfg.ic (seed, f_NL), evolves them
    with the exact-background leapfrog over cfg.time, and measures the initial
    and final CIC density. `backend` selects the linear-theory transfer ("camb"
    or "eh98"). Set `record=True` (or pass a `recorder`) to capture the
    trajectory for the growth history and the structure-formation animation --
    this is off the autodiff path and costs a little memory per step, so it is
    off by default. If cfg.rsd.enabled, the final particles are also mapped to
    redshift space and painted (with interlacing) into RunResult.redshift_field,
    so result.power_multipoles() returns the anisotropic P_0/P_2.
    """
    _check_supported(cfg)
    box, cosmo, time, ic = cfg.box, cfg.cosmology, cfg.time, cfg.ic

    if recorder is None and record:
        recorder = SnapshotRecorder(box)

    x0, p0 = IG.initial_state(
        box,
        cosmo,
        time,
        seed=ic.seed,
        f_NL=ic.f_NL,
        backend=backend,
        lpt_order=ic.lpt_order,
    )
    steps = IG.a_grid(time, spacing)
    xf, pf = IG.evolve_state(
        x0, p0, box, cosmo, steps, snapshot=recorder, integrator=time.integrator
    )

    ic_field = F.interlaced_density_contrast(x0, box)
    final_field = F.interlaced_density_contrast(xf, box)

    redshift_field = None
    if cfg.rsd.enabled:
        s = RS.redshift_space_positions(
            xf,
            pf,
            box,
            cosmo,
            z=time.z_final,
            los_axis=cfg.rsd.los_axis,
            f_growth=cfg.rsd.f_growth,
        )
        redshift_field = F.interlaced_density_contrast(s, box)

    return RunResult(
        config=cfg,
        x=xf,
        p=pf,
        ic_field=ic_field,
        final_field=final_field,
        backend=backend,
        recorder=recorder,
        redshift_field=redshift_field,
    )
