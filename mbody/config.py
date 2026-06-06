"""Simulation configuration: the knobs that define a single M-body run.

These are plain frozen dataclasses -- pure data, no MLX, no physics -- so they
import anywhere and are trivial to test, serialize, and sweep over. Everything
downstream (initial conditions, the force solve, the integrator, the
statistics) reads its parameters from here, which keeps a run reproducible from
one object.

A note on units. Like most of large-scale-structure cosmology, M-body works in
"h-inverse" comoving units: lengths in Mpc/h and wavenumbers in h/Mpc, where
h = H0 / (100 km/s/Mpc). Carrying the h around this way makes the box size and
the power spectrum independent of the exact value of H0, which is convenient
and conventional.
"""

from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Cosmology:
    """Background + linear-theory cosmological parameters (flat LCDM).

    Defaults are a Planck-2018-flavoured flat LCDM. These fix the expansion
    history, the linear growth of structure, and the shape and amplitude of the
    linear matter power spectrum P(k).

    Parameters
    ----------
    Omega_m : total matter density today, in units of the critical density
        (cold dark matter + baryons). Sets how strongly gravity clusters.
    Omega_b : baryon density today. Smaller than Omega_m; controls the
        baryon-acoustic features in the transfer function.
    h : dimensionless Hubble constant, H0 = 100 h km/s/Mpc.
    n_s : scalar spectral index of the primordial power spectrum (~1 means
        scale-invariant; the measured value is slightly "red", < 1).
    sigma8 : rms linear matter fluctuation in 8 Mpc/h spheres at z = 0. This is
        the amplitude knob -- it normalizes P(k).
    T_cmb_K : CMB temperature today, in kelvin (sets the radiation density and
        enters the Eisenstein & Hu transfer function).
    """

    Omega_m: float = 0.31
    Omega_b: float = 0.049
    h: float = 0.677
    n_s: float = 0.965
    sigma8: float = 0.81
    T_cmb_K: float = 2.7255

    def __post_init__(self):
        if not 0.0 < self.Omega_b < self.Omega_m < 1.0:
            raise ValueError(
                "require 0 < Omega_b < Omega_m < 1 "
                f"(got Omega_b={self.Omega_b}, Omega_m={self.Omega_m})"
            )
        if self.h <= 0.0:
            raise ValueError(f"h must be positive (got {self.h})")
        if self.sigma8 <= 0.0:
            raise ValueError(f"sigma8 must be positive (got {self.sigma8})")

    @property
    def Omega_cdm(self):
        """Cold dark matter density today (Omega_m minus baryons)."""
        return self.Omega_m - self.Omega_b

    @property
    def Omega_Lambda(self):
        """Dark-energy density today, assuming spatial flatness."""
        return 1.0 - self.Omega_m

    @property
    def H0(self):
        """Hubble constant today in km/s/Mpc."""
        return 100.0 * self.h


@dataclass(frozen=True)
class BoxConfig:
    """The simulation volume and its meshes.

    M-body is a *particle-mesh* code: it represents matter both as particles
    that move and as a regular 3D grid (mesh) onto which their density is
    "painted" so the gravitational force can be solved with an FFT. This object
    sets the size of the periodic box and the resolution of both.

    Parameters
    ----------
    box_size : comoving side length of the cubic, periodic box, in Mpc/h.
    n_mesh : number of force-mesh cells per side; the density/force grid is
        n_mesh^3. This sets the smallest resolved scale (the Nyquist
        wavenumber) and dominates memory and FFT cost.
    n_particles : number of particles per side; there are n_particles^3 of
        them. Defaults to equal n_mesh (one particle per cell at the start).
    """

    box_size: float = 256.0
    n_mesh: int = 128
    n_particles: int = 128

    def __post_init__(self):
        if self.box_size <= 0.0:
            raise ValueError(f"box_size must be positive (got {self.box_size})")
        if self.n_mesh <= 0:
            raise ValueError(f"n_mesh must be positive (got {self.n_mesh})")
        if self.n_particles <= 0:
            raise ValueError(f"n_particles must be positive (got {self.n_particles})")

    @property
    def cell_size(self):
        """Comoving size of one force-mesh cell, in Mpc/h."""
        return self.box_size / self.n_mesh

    @property
    def n_mesh_total(self):
        """Total number of force-mesh cells, n_mesh^3."""
        return self.n_mesh**3

    @property
    def n_particles_total(self):
        """Total number of particles, n_particles^3."""
        return self.n_particles**3

    @property
    def k_fundamental(self):
        """Smallest (fundamental) wavenumber in the box, 2 pi / L (h/Mpc).

        The largest scale the box can represent -- the modes M-body most wants
        to get right for f_NL.
        """
        return 2.0 * math.pi / self.box_size

    @property
    def k_nyquist(self):
        """Mesh Nyquist wavenumber, pi * n_mesh / L (h/Mpc).

        The smallest scale the force mesh can represent; power near here is
        unreliable (aliasing), which later motivates higher-order mass
        assignment and interlacing.
        """
        return math.pi * self.n_mesh / self.box_size


@dataclass(frozen=True)
class TimeStepping:
    """How the PM solver marches from the initial redshift to the final one.

    Parameters
    ----------
    z_init : starting redshift, where initial conditions / LPT are set.
    z_final : ending redshift (z = 0 is today).
    n_steps : number of leapfrog PM steps between them. More steps = more
        accurate growth, but reverse-mode autodiff memory scales with this
        count because the graph unrolls over every step.
    integrator : time-stepping scheme. "exact" is the implemented baseline --
        a kick-drift-kick leapfrog whose kick/drift coefficients are exact
        background integrals. "fastpm" (modified kernels that get linear growth
        right at low step count) and "bullfrog" (a 2LPT-accurate, time-reversible
        alternative) are reserved in the enum but not yet implemented; run()
        raises NotImplementedError on them.
    memory_mode : autodiff memory strategy. "replay" keeps the full unrolled
        graph; "checkpoint" recomputes each step in the backward pass to save
        memory; "adjoint" reconstructs state by reverse-time integration
        (O(1) snapshots), which requires a reversible integrator.
    """

    z_init: float = 9.0
    z_final: float = 0.0
    n_steps: int = 10
    integrator: str = "exact"
    memory_mode: str = "replay"

    _INTEGRATORS = ("exact", "fastpm", "bullfrog")
    _MEMORY_MODES = ("replay", "checkpoint", "adjoint")

    def __post_init__(self):
        if self.z_init <= self.z_final:
            raise ValueError(
                f"z_init ({self.z_init}) must exceed z_final ({self.z_final})"
            )
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive (got {self.n_steps})")
        if self.integrator not in self._INTEGRATORS:
            raise ValueError(
                f"integrator must be one of {self._INTEGRATORS} "
                f"(got {self.integrator!r})"
            )
        if self.memory_mode not in self._MEMORY_MODES:
            raise ValueError(
                f"memory_mode must be one of {self._MEMORY_MODES} "
                f"(got {self.memory_mode!r})"
            )


@dataclass(frozen=True)
class InitialConditions:
    """How the initial density field is generated.

    Parameters
    ----------
    f_NL : amplitude of local primordial non-Gaussianity. f_NL = 0 is a
        Gaussian field; nonzero injects phi -> phi + f_NL (phi^2 - <phi^2>) in
        the primordial potential. This is the parameter M-body ultimately
        differentiates with respect to.
    seed : RNG seed for the Gaussian random phases (reproducibility).
    lpt_order : Lagrangian perturbation theory order for the initial
        displacement -- 1 (Zel'dovich, implemented) or 2 (2LPT, more accurate
        large-scale flows; reserved but not yet implemented, run() raises on it).
    kind : "gaussian" or "local_fnl". (Redundant with f_NL != 0, but explicit
        so a run's intent is unambiguous.)
    """

    f_NL: float = 0.0
    seed: int = 0
    lpt_order: int = 1
    kind: str = "gaussian"

    _KINDS = ("gaussian", "local_fnl")

    def __post_init__(self):
        if self.lpt_order not in (1, 2):
            raise ValueError(f"lpt_order must be 1 or 2 (got {self.lpt_order})")
        if self.kind not in self._KINDS:
            raise ValueError(f"kind must be one of {self._KINDS} (got {self.kind!r})")


@dataclass(frozen=True)
class SimConfig:
    """Top-level configuration aggregating the four sub-configs.

    One SimConfig fully specifies a run. Construct with all defaults via
    ``SimConfig()``, or override piecewise, e.g.::

        cfg = SimConfig(
            box=BoxConfig(box_size=512.0, n_mesh=256),
            ic=InitialConditions(f_NL=100.0, kind="local_fnl"),
        )
    """

    cosmology: Cosmology = field(default_factory=Cosmology)
    box: BoxConfig = field(default_factory=BoxConfig)
    time: TimeStepping = field(default_factory=TimeStepping)
    ic: InitialConditions = field(default_factory=InitialConditions)

    def summary(self):
        """Human-readable one-screen summary, handy in scripts and logs."""
        c, b, t = self.cosmology, self.box, self.time
        return (
            "M-body SimConfig\n"
            f"  cosmology: Omega_m={c.Omega_m} Omega_b={c.Omega_b} h={c.h} "
            f"n_s={c.n_s} sigma8={c.sigma8}\n"
            f"  box:       L={b.box_size} Mpc/h, mesh={b.n_mesh}^3, "
            f"particles={b.n_particles}^3\n"
            f"             cell={b.cell_size:.3f} Mpc/h, "
            f"k_f={b.k_fundamental:.4f} h/Mpc, "
            f"k_nyq={b.k_nyquist:.3f} h/Mpc\n"
            f"  time:      z {t.z_init} -> {t.z_final} in {t.n_steps} steps "
            f"({t.integrator}, {t.memory_mode})\n"
            f"  ic:        kind={self.ic.kind} f_NL={self.ic.f_NL} "
            f"lpt_order={self.ic.lpt_order} seed={self.ic.seed}"
        )
