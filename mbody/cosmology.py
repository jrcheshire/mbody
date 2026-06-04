"""Linear-theory cosmology: expansion, growth, transfer function, and P(k).

This module turns a set of cosmological parameters (mbody.config.Cosmology)
into the *linear* description of structure: how fast the universe expands, how
density perturbations grow with time, and the linear matter power spectrum
P(k) -- the statistical blueprint of how lumpy the universe is on each scale.
Everything downstream (initial conditions, the PM solve) starts from P(k).

Two transfer-function / P(k) backends, behind one interface:

- **"camb"** (default): the full Boltzmann solver (Lewis & Challinor). It
  integrates the coupled Einstein-Boltzmann equations for photons, baryons,
  CDM and neutrinos from first principles, sub-percent accurate. This is the
  reference.
- **"eh98"** / **"eh98_nowiggle"**: the Eisenstein & Hu (1998) analytic fitting
  formula (ported verbatim from the authors' tf_fit.c). No differential
  equations -- a closed-form algebraic approximation to a Boltzmann solve,
  good to ~a few percent. Self-contained, instant, and handy as a smooth
  baseline and a sanity cross-check on CAMB.

Other ideas, briefly:

- The **transfer function** T(k) encodes how the early universe processed an
  initially scale-invariant spectrum: modes entering the horizon during
  radiation domination were suppressed, leaving a turnover near matter-
  radiation equality and (from baryons sloshing in the photon-baryon fluid)
  the baryon acoustic oscillations.
- The **growth factor** D(z) gives delta(k, z) = D(z) delta(k, 0), so
  P(k, z) = D(z)^2 P(k, 0). For flat LCDM we get it exactly from one integral.
- The amplitude is fixed so the rms fluctuation in 8 Mpc/h spheres is sigma8.

Precision note: this is a float64 CPU "island" (numpy/scipy/CAMB) per
mbody.precision -- accuracy-critical, computed once, then sampled onto the
float32 mesh by the field code. No MLX, no GPU here. CAMB results are cached
per cosmology so the Boltzmann solve runs only once.
"""

from functools import lru_cache
import math

import numpy as np
from scipy.integrate import quad, simpson

# Mean comoving critical density today, in (M_sun/h) per (Mpc/h)^3. Equivalently
# 2.775e11 h^2 M_sun / Mpc^3. Handy for turning a box + particle count into a
# particle mass.
RHO_CRIT = 2.77536627e11

# The constant the EH98 reference C uses in place of Euler's e. Kept verbatim so
# this port reproduces the original formula bit-for-bit.
_E = 2.718282


def mean_matter_density(cosmo):
    """Comoving mean matter density, in (M_sun/h) per (Mpc/h)^3."""
    return cosmo.Omega_m * RHO_CRIT


def _prep(k_hmpc):
    arr = np.atleast_1d(np.asarray(k_hmpc, dtype=np.float64))
    return arr, np.ndim(k_hmpc) == 0


# --------------------------------------------------------------------------
# Eisenstein & Hu (1998) transfer function (ported from tf_fit.c)
# --------------------------------------------------------------------------
# Internally the fit works in Mpc (no h). Our public API takes k in h/Mpc and
# converts with k_mpc = k_hmpc * h, exactly as the reference driver does.


class _EH98:
    """Scalar EH98 parameters for one cosmology (mirrors TFset_parameters)."""

    def __init__(self, cosmo):
        Om, Ob, h = cosmo.Omega_m, cosmo.Omega_b, cosmo.h
        f_baryon = Ob / Om
        omhh = Om * h * h
        obhh = omhh * f_baryon
        theta = cosmo.T_cmb_K / 2.7

        z_eq = 2.50e4 * omhh / theta**4  # really 1 + z_eq
        k_eq = 0.0746 * omhh / theta**2  # Mpc^-1

        b1 = 0.313 * omhh**-0.419 * (1 + 0.607 * omhh**0.674)
        b2 = 0.238 * omhh**0.223
        z_drag = 1291 * omhh**0.251 / (1 + 0.659 * omhh**0.828) * (1 + b1 * obhh**b2)

        R_drag = 31.5 * obhh / theta**4 * (1000.0 / (1 + z_drag))
        R_eq = 31.5 * obhh / theta**4 * (1000.0 / z_eq)

        sound_horizon = (
            2.0
            / (3.0 * k_eq)
            * math.sqrt(6.0 / R_eq)
            * math.log(
                (math.sqrt(1 + R_drag) + math.sqrt(R_drag + R_eq))
                / (1 + math.sqrt(R_eq))
            )
        )

        k_silk = 1.6 * obhh**0.52 * omhh**0.73 * (1 + (10.4 * omhh) ** -0.95)

        a1 = (46.9 * omhh) ** 0.670 * (1 + (32.1 * omhh) ** -0.532)
        a2 = (12.0 * omhh) ** 0.424 * (1 + (45.0 * omhh) ** -0.582)
        alpha_c = a1 ** (-f_baryon) * a2 ** (-(f_baryon**3))

        bb1 = 0.944 / (1 + (458.0 * omhh) ** -0.708)
        bb2 = (0.395 * omhh) ** -0.0266
        beta_c = 1.0 / (1 + bb1 * ((1 - f_baryon) ** bb2 - 1))

        y = z_eq / (1 + z_drag)
        sqy = math.sqrt(1 + y)
        alpha_b_G = y * (-6.0 * sqy + (2.0 + 3.0 * y) * math.log((sqy + 1) / (sqy - 1)))
        alpha_b = 2.07 * k_eq * sound_horizon * (1 + R_drag) ** -0.75 * alpha_b_G

        beta_node = 8.41 * omhh**0.435
        beta_b = (
            0.5 + f_baryon + (3.0 - 2.0 * f_baryon) * math.sqrt((17.2 * omhh) ** 2 + 1)
        )

        sound_horizon_fit = (
            44.5 * math.log(9.83 / omhh) / math.sqrt(1 + 10.0 * obhh**0.75)
        )
        alpha_gamma = (
            1
            - 0.328 * math.log(431.0 * omhh) * f_baryon
            + 0.38 * math.log(22.3 * omhh) * f_baryon**2
        )

        self.h = h
        self.omhh = omhh
        self.obhh = obhh
        self.k_eq = k_eq
        self.sound_horizon = sound_horizon
        self.k_silk = k_silk
        self.alpha_c = alpha_c
        self.beta_c = beta_c
        self.alpha_b = alpha_b
        self.beta_b = beta_b
        self.beta_node = beta_node
        self.sound_horizon_fit = sound_horizon_fit
        self.alpha_gamma = alpha_gamma


def transfer_eh98(k_hmpc, cosmo):
    """Full EH98 transfer function T(k) with baryon acoustic oscillations.

    k in h/Mpc; T -> 1 as k -> 0. (Eq. 16 of Eisenstein & Hu 1998; TFfit_onek.)
    """
    k_arr, scalar = _prep(k_hmpc)
    p = _EH98(cosmo)
    k = k_arr * p.h  # Mpc^-1
    out = np.ones_like(k)
    m = k > 0
    kk = k[m]

    q = kk / 13.41 / p.k_eq
    xx = kk * p.sound_horizon

    ln_beta = np.log(_E + 1.8 * p.beta_c * q)
    ln_nobeta = np.log(_E + 1.8 * q)
    C_alpha = 14.2 / p.alpha_c + 386.0 / (1 + 69.9 * q**1.08)
    C_noalpha = 14.2 + 386.0 / (1 + 69.9 * q**1.08)

    f = 1.0 / (1.0 + (xx / 5.4) ** 4)
    T_c = f * ln_beta / (ln_beta + C_noalpha * q**2) + (1 - f) * ln_beta / (
        ln_beta + C_alpha * q**2
    )

    s_tilde = p.sound_horizon * (1 + (p.beta_node / xx) ** 3) ** (-1.0 / 3.0)
    xx_tilde = kk * s_tilde
    T_b_T0 = ln_nobeta / (ln_nobeta + C_noalpha * q**2)
    T_b = (
        np.sin(xx_tilde)
        / xx_tilde
        * (
            T_b_T0 / (1 + (xx / 5.2) ** 2)
            + p.alpha_b / (1 + (p.beta_b / xx) ** 3) * np.exp(-((kk / p.k_silk) ** 1.4))
        )
    )

    f_baryon = p.obhh / p.omhh
    out[m] = f_baryon * T_b + (1 - f_baryon) * T_c
    return float(out[0]) if scalar else out


def transfer_eh98_nowiggle(k_hmpc, cosmo):
    """No-wiggle EH98 transfer function: the smooth, BAO-free shape.

    Captures the broadband suppression of small-scale power without the
    oscillations (eqs. 29-31; TFnowiggles). A smooth baseline the full T(k)
    oscillates around.
    """
    k_arr, scalar = _prep(k_hmpc)
    p = _EH98(cosmo)
    k = k_arr * p.h  # Mpc^-1
    q = k / 13.41 / p.k_eq
    xx = k * p.sound_horizon_fit
    gamma_eff = p.omhh * (p.alpha_gamma + (1 - p.alpha_gamma) / (1 + (0.43 * xx) ** 4))
    q_eff = q * p.omhh / gamma_eff
    L0 = np.log(2.0 * _E + 1.8 * q_eff)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
    out = L0 / (L0 + C0 * q_eff**2)
    return float(out[0]) if scalar else out


def transfer_eh98_zerobaryon(k_hmpc, cosmo):
    """Zero-baryon EH98 transfer function (eq. 29; TFzerobaryon).

    The pure-CDM shape; a third independent fitting form for cross-checks.
    """
    k_arr, scalar = _prep(k_hmpc)
    p = _EH98(cosmo)
    k = k_arr * p.h
    q = k / 13.41 / p.k_eq
    L0 = np.log(2.0 * _E + 1.8 * q)
    C0 = 14.2 + 731.0 / (1 + 62.5 * q)
    out = L0 / (L0 + C0 * q * q)
    return float(out[0]) if scalar else out


# --------------------------------------------------------------------------
# CAMB backend (Boltzmann solver), cached per cosmology
# --------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _camb_pk0(cosmo, maxkh=50.0, npoints=800):
    """Linear P(k, z=0) from CAMB, rescaled to the target sigma8.

    Returns (kh, pk0) arrays in h/Mpc and (Mpc/h)^3. CAMB is run with a
    fiducial A_s and the spectrum is rescaled by (sigma8_target/sigma8_camb)^2,
    which is exact for the linear amplitude.
    """
    import camb
    from camb import model

    h = cosmo.h
    pars = camb.set_params(
        H0=cosmo.H0,
        ombh2=cosmo.Omega_b * h * h,
        omch2=cosmo.Omega_cdm * h * h,
        ns=cosmo.n_s,
        As=2.0e-9,
        mnu=0.0,
        omk=0.0,
        TCMB=cosmo.T_cmb_K,
    )
    pars.set_matter_power(redshifts=[0.0], kmax=maxkh * h * 1.2)
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, _z, pk = results.get_matter_power_spectrum(
        minkh=1e-5, maxkh=maxkh, npoints=npoints
    )
    try:
        s8 = float(results.get_sigma8_0())
    except Exception:
        s8 = float(results.get_sigma8()[-1])
    pk0 = pk[0] * (cosmo.sigma8 / s8) ** 2
    return kh, pk0


@lru_cache(maxsize=None)
def _camb_logpk0(cosmo):
    kh, pk0 = _camb_pk0(cosmo)
    return np.log(kh), np.log(pk0)


def _camb_power0(k_hmpc, cosmo):
    # Log-log interpolation of the cached CAMB P(k, z=0). np.interp clamps at
    # the grid edges; our k always falls inside [1e-5, 50] h/Mpc. Non-positive
    # k (e.g. the DC mode of a mesh) returns P = 0 without a log(0) warning.
    lk, lp = _camb_logpk0(cosmo)
    k = np.asarray(k_hmpc, dtype=np.float64)
    safe = np.where(k > 0, k, 1.0)
    out = np.exp(np.interp(np.log(safe), lk, lp))
    return np.where(k > 0, out, 0.0)


def transfer_camb(k_hmpc, cosmo):
    """Transfer function inferred from the CAMB P(k): T(k) = sqrt(P/k^n_s),
    normalized to T -> 1 as k -> 0. For apples-to-apples comparison with EH98.
    """
    k_arr, scalar = _prep(k_hmpc)
    shape = _camb_power0(k_arr, cosmo) / k_arr**cosmo.n_s
    norm = float(_camb_power0(np.array([1e-4]), cosmo)[0]) / (1e-4**cosmo.n_s)
    out = np.sqrt(shape / norm)
    return float(out[0]) if scalar else out


# --------------------------------------------------------------------------
# Backend dispatch
# --------------------------------------------------------------------------

_TRANSFER = {
    "camb": transfer_camb,
    "eh98": transfer_eh98,
    "eh98_nowiggle": transfer_eh98_nowiggle,
    "eh98_zerobaryon": transfer_eh98_zerobaryon,
}


def transfer(k_hmpc, cosmo, backend="camb"):
    """Transfer function T(k) from the chosen backend (default CAMB)."""
    try:
        fn = _TRANSFER[backend]
    except KeyError:
        raise ValueError(
            f"unknown backend {backend!r}; choose from {sorted(_TRANSFER)}"
        )
    return fn(k_hmpc, cosmo)


# --------------------------------------------------------------------------
# Background expansion + linear growth (exact for flat LCDM)
# --------------------------------------------------------------------------


def E(z, cosmo):
    """Dimensionless Hubble rate H(z)/H0 for flat LCDM (radiation neglected)."""
    a = 1.0 / (1.0 + np.asarray(z, dtype=np.float64))
    return np.sqrt(cosmo.Omega_m / a**3 + cosmo.Omega_Lambda)


def _growth_integrand(a, Om, OL):
    e = math.sqrt(Om / a**3 + OL)
    return 1.0 / (a * e) ** 3


def _growth_unnorm(a, cosmo):
    # D(a) propto (5 Om / 2) E(a) * integral_0^a da' / (a' E(a'))^3.
    integral, _ = quad(
        _growth_integrand, 0.0, a, args=(cosmo.Omega_m, cosmo.Omega_Lambda)
    )
    e = math.sqrt(cosmo.Omega_m / a**3 + cosmo.Omega_Lambda)
    return 2.5 * cosmo.Omega_m * e * integral


@lru_cache(maxsize=None)
def _D0(cosmo):
    return _growth_unnorm(1.0, cosmo)


def growth_factor(z, cosmo):
    """Linear growth factor D(z), normalized to D(z=0) = 1.

    P(k, z) = D(z)^2 P(k, 0). At high redshift (matter domination) D -> a.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    a = 1.0 / (1.0 + z_arr.ravel())
    D = np.array([_growth_unnorm(ai, cosmo) for ai in a]) / _D0(cosmo)
    D = D.reshape(z_arr.shape)
    return float(D) if z_arr.ndim == 0 else D


def growth_factor_md(z, cosmo):
    """Growth factor normalized to D = a = 1/(1+z) in matter domination.

    This is the convention used in the local-f_NL relation between the
    primordial potential and the linear density, delta(k, z) = M(k, z) phi(k).
    It is just the unnormalized growth integral, which tends to a as a -> 0
    (whereas growth_factor is rescaled to D(z=0) = 1, so the two differ by the
    constant growth_factor_md(0) = 1 / lim_{a->0} [growth_factor(a)/a]).
    """
    z_arr = np.asarray(z, dtype=np.float64)
    a = 1.0 / (1.0 + z_arr.ravel())
    D = np.array([_growth_unnorm(ai, cosmo) for ai in a]).reshape(z_arr.shape)
    return float(D) if z_arr.ndim == 0 else D


def growth_rate(z, cosmo):
    """Linear growth rate f = dlnD/dlna (exact for flat LCDM).

    f(z=0) ~ Omega_m^0.55 and f -> 1 deep in matter domination.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    Om, OL = cosmo.Omega_m, cosmo.Omega_Lambda
    a = 1.0 / (1.0 + z_arr.ravel())
    out = np.empty_like(a)
    for i, ai in enumerate(a):
        G, _ = quad(_growth_integrand, 0.0, ai, args=(Om, OL))
        e = math.sqrt(Om / ai**3 + OL)
        out[i] = -1.5 * Om / (ai**3 * e**2) + 1.0 / (ai**2 * e**3 * G)
    out = out.reshape(z_arr.shape)
    return float(out) if z_arr.ndim == 0 else out


# --------------------------------------------------------------------------
# Linear power spectrum + sigma(R) normalization
# --------------------------------------------------------------------------


def _tophat_window(x):
    # Fourier transform of a real-space spherical top hat; -> 1 as x -> 0.
    return 3.0 * (np.sin(x) - x * np.cos(x)) / x**3


def _sigma2_unnorm(R, cosmo, backend):
    # sigma^2(R) for the unnormalized EH98 P(k) = k^n_s T(k)^2 (amplitude A=1),
    # in d ln k: sigma^2 = (1/2pi^2) int dlnk k^3 P(k) W(kR)^2.
    lnk = np.linspace(math.log(1e-4), math.log(1e2), 4000)
    k = np.exp(lnk)
    Pk = k**cosmo.n_s * _TRANSFER[backend](k, cosmo) ** 2
    W = _tophat_window(k * R)
    return simpson(k**3 * Pk * W**2 / (2.0 * math.pi**2), x=lnk)


@lru_cache(maxsize=None)
def _amplitude(cosmo, backend):
    # EH98 amplitude A such that sigma(8 Mpc/h, z=0) = sigma8. CAMB is already
    # sigma8-normalized in _camb_pk0, so it needs no separate amplitude.
    return cosmo.sigma8**2 / _sigma2_unnorm(8.0, cosmo, backend)


def linear_power(k_hmpc, cosmo, z=0.0, backend="camb"):
    """Linear matter power spectrum P(k, z) in (Mpc/h)^3.

    Normalized so sigma8 is recovered at z = 0, and scaled by D(z)^2. Backend
    is "camb" (default), "eh98" (full, with BAO), or "eh98_nowiggle" (smooth).
    """
    k_arr, scalar = _prep(k_hmpc)
    if backend == "camb":
        P0 = _camb_power0(k_arr, cosmo)
    elif backend in ("eh98", "eh98_nowiggle"):
        P0 = (
            _amplitude(cosmo, backend)
            * k_arr**cosmo.n_s
            * _TRANSFER[backend](k_arr, cosmo) ** 2
        )
    else:
        raise ValueError(
            "linear_power backend must be camb/eh98/eh98_nowiggle, " f"got {backend!r}"
        )
    Pk = P0 * growth_factor(z, cosmo) ** 2
    return float(Pk[0]) if scalar else Pk


def sigma_R(R, cosmo, z=0.0, backend="camb"):
    """Rms linear density fluctuation in spheres of radius R (Mpc/h) at z.

    By construction sigma_R(8, z=0) recovers cosmo.sigma8 (a normalization
    cross-check, especially for the CAMB backend).
    """
    lnk = np.linspace(math.log(1e-4), math.log(1e2), 4000)
    k = np.exp(lnk)
    Pk0 = linear_power(k, cosmo, z=0.0, backend=backend)
    W = _tophat_window(k * R)
    s2 = simpson(k**3 * Pk0 * W**2 / (2.0 * math.pi**2), x=lnk)
    return math.sqrt(s2) * growth_factor(z, cosmo)


def dimensionless_power(k_hmpc, cosmo, z=0.0, backend="camb"):
    """Delta^2(k) = k^3 P(k) / (2 pi^2), the variance per ln k."""
    k_arr, scalar = _prep(k_hmpc)
    d2 = (
        k_arr**3 * linear_power(k_arr, cosmo, z=z, backend=backend) / (2.0 * math.pi**2)
    )
    return float(d2[0]) if scalar else d2
