"""M-body: a differentiable particle-mesh (PM) N-body toy built on MLX.

The scientific goal is a fully differentiable forward model of large-scale
structure -- Gaussian or local-f_NL initial conditions, LPT displacement, a
handful of leapfrog PM steps -- run on Apple Silicon via MLX's unified-memory
arrays and composable autodiff. The headline target is autodiff
dlnP(k)/df_NL, compared against the analytic Dalal et al. (2008)
scale-dependent bias Delta b(k) ~ f_NL / k^2.

See README.md for orientation and ROADMAP.md for the staged plan.
"""

from mbody import (
    bias,
    cosmology,
    fields,
    forces,
    ic,
    integrate,
    lpt,
    painting,
    precision,
)
from mbody.config import (
    BoxConfig,
    Cosmology,
    InitialConditions,
    SimConfig,
    TimeStepping,
)

__version__ = "0.1.0"

__all__ = [
    "precision",
    "cosmology",
    "fields",
    "lpt",
    "painting",
    "forces",
    "integrate",
    "ic",
    "bias",
    "Cosmology",
    "BoxConfig",
    "TimeStepping",
    "InitialConditions",
    "SimConfig",
]
