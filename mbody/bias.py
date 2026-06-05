"""A minimal differentiable biased tracer, and the Dalal scale-dependent-bias
reference shape.

The matter density itself has no power-spectrum response to local f_NL at linear
order: dlnP_matter/df_NL = 0 (the correction is 2 Re<phi_G^* phi_G^2>, a Gaussian
three-point, which vanishes). The Dalal et al. (2008) scale-dependent bias
Delta b(k) ~ f_NL / k^2 is a property of *biased tracers*: an object whose
abundance responds to the local small-scale density variance feels the
long-wavelength potential phi that local f_NL couples to that variance, and phi =
delta / M(k) with M(k) ~ k^2, hence the 1/k^2.

The smallest differentiable tracer that exhibits this is an Eulerian local
quadratic bias

    delta_h(x) = b1 delta(x) + (b2/2) [delta(x)^2 - <delta^2>],

where the b2 (delta^2) term couples to the squeezed bispectrum of the f_NL field
(the same physics validated in mbody.fields.bispectrum), producing a
scale-dependent bias dlnP_h/df_NL ~ 1/M(k). It is FFT-free, elementwise, and so
trivially differentiable -- mx.grad / mx.jvp flow through to f_NL.
"""

from mbody import ic as IC
from mbody import precision as P


def local_bias_tracer(delta, b1, b2):
    """Eulerian local quadratic-bias tracer.

    delta_h = b1 delta + (b2/2) (delta^2 - <delta^2>), a real field the same
    shape as delta. The mean subtraction only shifts the (unobserved) k = 0 mode,
    so it is taken as a constant -- it does not affect any band power or its
    f_NL derivative. Differentiable in delta; pass b1, b2 as plain floats.
    """
    mean2 = float(P.accurate_mean(delta**2))
    return b1 * delta + 0.5 * b2 * (delta**2 - mean2)


def scale_dependent_shape(k, cosmo, z=0.0, backend="camb"):
    """The Dalal scale-dependent-bias reference shape, 1 / M(k) ~ 1 / (k^2 T(k)).

    M(k) = ic.poisson_M(k, z) is the Poisson/transfer factor (delta = M phi), so
    1/M(k) is the k-shape of the local-f_NL bias correction the tracer's
    dlnP/df_NL should follow at large scales. Returns a float64 numpy array (the
    amplitude is set by the tracer's b_phi; overlay this normalized to the data).
    """
    return 1.0 / IC.poisson_M(k, cosmo, z=z, backend=backend)
