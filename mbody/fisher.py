"""Autodiff Fisher matrix for M-body, over theta = {f_NL, b1, b2, A}.

The project's headline is a single autodiff derivative, dlnP(k)/df_NL. A Fisher
forecast turns that into the full information matrix over the differentiable
parameters of the forward model: the primordial non-Gaussianity f_NL, the local
quadratic-bias parameters b1, b2 of the tracer, and a linear amplitude A (a
differentiable sigma8 / A_s proxy, since sigma8's own normalization is numpy and
off the autodiff graph). Cosmology *shape* parameters (Omega_m, n_s, h) enter
only through numpy/CAMB and are out of scope here.

Conventions, and why they make the forecast clean:

- **Data vector mu = ln P_b** (log band powers). Then the Jacobian
  J[b, i] = d ln P_b / d theta_i is exactly the dlnP/dtheta the rest of the
  project already validates, and the Gaussian covariance is parameter-free. For
  a real Gaussian field the band_power estimator P_b = (V/N^6) <|delta_k|^2> over
  the half-grid shell has

      Var[ln P_b] = (count_b + n_plane,b) / count_b^2,

  where count_b is the number of rfftn half-grid cells in the shell (the
  denominator band_power divides by) and n_plane,b is the subset lying in the
  kz in {0, N/2} planes. Those plane cells are *double-counted* by the
  equal-weight half-grid sum -- each pairs with its in-plane Hermitian conjugate
  at the same modulus -- which inflates the variance above the naive 1/count_b by
  up to ~50% at the fundamental, exactly where the f_NL signal lives. This is the
  exact Gaussian variance of the estimator (derived, then *validated* against a
  white-noise seed ensemble to the ~5% sampling floor in the tests, not
  asserted). For a flat (n_plane = 0) shell it reduces to the textbook
  Var[ln P_b] = 1/count_b.

- **Fisher F_ij = sum_b J_{b,i} J_{b,j} / Var[ln P_b]**, a diagonal-covariance
  weighted sum (modes in different k-bins are independent for a Gaussian field).
  Gaussian priors enter as 1/sigma^2 on the diagonal; fixed parameters drop their
  row/column. This is a forecast Fisher (J^T C^-1 J), not a Hessian-of-(-logP)
  at an MLE -- this toy has no likelihood/optimizer.

Gradients are built bin-by-bin with reverse-mode mx.grad (one sweep of the
scalar ln P_b returns the whole parameter row); forward-mode mx.jvp is WRONG
through the FFT here, so it is never used. The FisherForecast object itself is
pure float64 numpy linear algebra -- small (n_params x n_params), off the GPU
hot path, and separately testable from the forward model.
"""

import numpy as np
import mlx.core as mx

from mbody import bias as B
from mbody import cosmology as C
from mbody import fields as F
from mbody import ic as IC
from mbody import integrate as IN
from mbody import precision as P  # noqa: F401  (used by Jacobian builders)
from mbody import rsd as RS

# Canonical parameter order for the M-body autodiff Fisher.
PARAM_NAMES = ("f_NL", "b1", "b2", "A")

# Redshift-space Fisher adds a growth-rate amplitude f_growth (the RSD analog of
# A): it scales the line-of-sight velocity term, fiducial 1 = the simulation's
# own growth. See linear_multipole_jacobian / pm_multipole_jacobian.
PARAM_NAMES_RSD = ("f_NL", "b1", "b2", "A", "f_growth")


def band_power_log_variance(box, k_bins, dk=None):
    """Gaussian fractional variance Var[ln P_b] of band_power for each |k| shell.

        Var[ln P_b] = (count_b + n_plane,b) / count_b^2,

    where count_b is the number of rfftn half-grid cells in the shell (from
    fields._bin_masks, the same masks band_power sums over) and n_plane,b is the
    subset in the kz in {0, N/2} planes. band_power weights every half-grid cell
    equally, so each plane cell -- which pairs with its in-plane Hermitian
    conjugate at the same modulus -- is double-counted, inflating the variance by
    n_plane,b / count_b above the textbook 1/count_b (up to ~50% at the
    fundamental, where the f_NL signal lives). This is the exact Gaussian
    estimator variance, validated against a white-noise ensemble to ~5% (the
    sampling floor) in tests/test_fisher.py. Returns a float64 numpy array; this
    is the per-bin covariance the Fisher inverse-weights by.
    """
    if dk is None:
        dk = box.k_fundamental
    N = box.n_mesh
    masks, counts = F._bin_masks(box, k_bins, dk)
    out = np.empty(len(counts), dtype=np.float64)
    for b, (m, c) in enumerate(zip(masks, counts)):
        plane = np.asarray(m, dtype=np.float64)
        n_plane = plane[:, :, 0].sum()
        if N % 2 == 0:  # the kz = N/2 Nyquist plane exists only for even N
            n_plane += plane[:, :, N // 2].sum()
        out[b] = (c + n_plane) / c**2
    return out


def _theta_vec(theta_fid, names=PARAM_NAMES):
    """Pack a {name: value} fiducial dict into a float32 MLX vector."""
    return mx.array([float(theta_fid[n]) for n in names], dtype=P.REAL)


def linear_logP_jacobian(
    box, cosmo, theta_fid, k_bins, seed=0, dk=None, backend="camb"
):
    """d ln P_b / d theta for the tracer of the LINEAR f_NL field, one seed.

    Forward model (all MLX, differentiable):
        delta = A * ic.linear_density(f_NL)
        tracer = local_bias_tracer(delta, b1, b2)
        mu_b = ln band_power(tracer)_b
    The Jacobian is built per bin with reverse-mode mx.grad -- one sweep returns
    the full (f_NL, b1, b2, A) row. Returns (J, P_fid): J is (n_bins, 4) float64
    and P_fid is the (n_bins,) fiducial band power.
    """
    theta = _theta_vec(theta_fid)

    def model_logP(t):
        f_NL, b1, b2, A = t[0], t[1], t[2], t[3]
        delta = A * IC.linear_density(box, cosmo, seed=seed, f_NL=f_NL, backend=backend)
        tracer = B.local_bias_tracer(delta, b1, b2)
        return mx.log(F.band_power(tracer, box, k_bins, dk=dk))

    n_bins = len(k_bins)
    J = np.empty((n_bins, len(PARAM_NAMES)), dtype=np.float64)
    for b in range(n_bins):
        g = mx.grad(lambda t, b=b: model_logP(t)[b])(theta)
        J[b, :] = np.asarray(g, dtype=np.float64)
    P_fid = np.asarray(mx.exp(model_logP(theta)), dtype=np.float64)
    return J, P_fid


def pm_logP_jacobian(
    box,
    cosmo,
    time,
    theta_fid,
    k_bins,
    seed=0,
    dk=None,
    backend="camb",
    integrator=None,
    lpt_order=2,
):
    """d ln P_b / d theta for the tracer of the PM-EVOLVED f_NL field, one seed.

    Forward model: f_NL and A shape the primordial IC, which is evolved through
    LPT + the leapfrog PM steps; the local-bias tracer (b1, b2) is then painted on
    the final field and its band power measured. The Jacobian exploits where each
    parameter enters:

      * f_NL and A are UPSTREAM of the leapfrog -- their columns come from the
        reversible-leapfrog adjoint (integrate.adjoint_grad_ic), one O(grid)-memory
        sweep per bin that returns BOTH columns at once (they share the trajectory);
      * b1 and b2 are DOWNSTREAM -- the tracer acts on the *final* field, so their
        columns are a cheap plain mx.grad at the fixed (stop_gradient) final
        density, no adjoint, no trajectory.

    Returns (J, P_fid): J is (n_bins, 4) float64 ordered as PARAM_NAMES, P_fid the
    (n_bins,) fiducial band power. Cost ~ n_bins adjoint sweeps + n_bins cheap
    downstream gradients.
    """
    fid_fNL = float(theta_fid["f_NL"])
    fid_b1 = float(theta_fid["b1"])
    fid_b2 = float(theta_fid["b2"])
    fid_A = float(theta_fid["A"])
    n_bins = len(k_bins)
    J = np.empty((n_bins, len(PARAM_NAMES)), dtype=np.float64)

    # Evolve once at the fiducial; the final density is fixed for the b1/b2 columns.
    x_final, _ = IN.leapfrog(
        box,
        cosmo,
        time,
        seed=seed,
        f_NL=mx.array(fid_fNL),
        amplitude=mx.array(fid_A),
        backend=backend,
        integrator=integrator,
        lpt_order=lpt_order,
    )
    delta_final = mx.stop_gradient(F.interlaced_density_contrast(x_final, box))

    # Downstream b1, b2 columns: cheap mx.grad at the fixed final field.
    tb = mx.array([fid_b1, fid_b2])

    def downstream_logP(t):
        tracer = B.local_bias_tracer(delta_final, t[0], t[1])
        return mx.log(F.band_power(tracer, box, k_bins, dk=dk))

    for b in range(n_bins):
        g = mx.grad(lambda t, b=b: downstream_logP(t)[b])(tb)
        J[b, 1] = float(g[0])  # b1
        J[b, 2] = float(g[1])  # b2

    # IC columns f_NL, A: the shared trajectory adjoint, one sweep per bin.
    def make_loss(b):
        def loss_field(x):
            tracer = B.local_bias_tracer(
                F.interlaced_density_contrast(x, box), fid_b1, fid_b2
            )
            return mx.log(F.band_power(tracer, box, k_bins, dk=dk))[b]

        return loss_field

    for b in range(n_bins):
        g_ic = IN.adjoint_grad_ic(
            make_loss(b),
            box,
            cosmo,
            time,
            seed=seed,
            f_NL=fid_fNL,
            amplitude=fid_A,
            backend=backend,
            integrator=integrator,
            lpt_order=lpt_order,
        )
        J[b, 0] = float(g_ic[0])  # f_NL
        J[b, 3] = float(g_ic[1])  # A

    P_fid = np.asarray(mx.exp(downstream_logP(tb)), dtype=np.float64)
    return J, P_fid


def _prewhitened_inverse(M):
    """Inverse of a symmetric positive matrix via sqrt-diag prewhitening.

    D = sqrt(diag M) gives M_w = D^-1 M D^-1 a unit diagonal and off-diagonals
    bounded by 1, dropping the condition number by orders of magnitude before the
    inverse; the result is unwhitened back. Used for both the band-power
    covariance C^-1 and the Fisher F^-1.
    """
    d = np.sqrt(np.diag(M))
    if np.any(d <= 0):
        raise ValueError("non-positive diagonal in matrix to invert")
    d_inv = 1.0 / d
    M_w = M * np.outer(d_inv, d_inv)
    return np.linalg.inv(M_w) * np.outer(d_inv, d_inv)


class FisherForecast:
    """Fisher matrix and parameter constraints, diagonal or full covariance.

    F = J^T C^-1 J (+ prior 1/sigma^2 on the diagonal), with fixed parameters
    dropped. Pass EITHER `variances` (a diagonal data covariance, e.g. the
    plane-corrected Var[ln P_b] for the isotropic log-band-power Fisher) OR
    `covariance` (a full data covariance, e.g. the block covariance coupling
    redshift-space multipoles within a k-bin). The Jacobian assembly
    (linear_logP_jacobian / pm_logP_jacobian, or the multipole variants) is
    deliberately decoupled from this pure linear-algebra object.

    Parameters
    ----------
    jacobian : (n_data, n_params) array, J[d, i] = d mu_d / d theta_i.
    variances : (n_data,) diagonal data covariance (mutually exclusive with
        `covariance`).
    fiducial_params : {name: value} of every parameter (for summary/reference).
    param_names : ordered names matching the columns of `jacobian`.
    priors : optional {name: sigma_prior}; adds 1/sigma^2 to the diagonal.
    fixed_params : optional list of names to hold fixed (drop from F).
    covariance : (n_data, n_data) full data covariance (mutually exclusive with
        `variances`); inverse-weighted via a prewhitened solve.
    """

    def __init__(
        self,
        jacobian,
        variances=None,
        fiducial_params=None,
        param_names=PARAM_NAMES,
        priors=None,
        fixed_params=None,
        covariance=None,
    ):
        self.J = np.asarray(jacobian, dtype=np.float64)
        self.param_names = list(param_names)
        self.fiducial = dict(fiducial_params or {})
        self.priors = dict(priors or {})
        self.fixed = list(fixed_params or [])

        if (variances is None) == (covariance is None):
            raise ValueError("pass exactly one of `variances` or `covariance`")

        n_bins, n_params = self.J.shape
        if n_params != len(self.param_names):
            raise ValueError(
                f"jacobian has {n_params} columns but {len(self.param_names)} "
                "param_names"
            )
        if variances is not None:
            self.variances = np.asarray(variances, dtype=np.float64)
            self.covariance = None
            if self.variances.shape != (n_bins,):
                raise ValueError(
                    f"variances shape {self.variances.shape} != ({n_bins},)"
                )
            if np.any(self.variances <= 0):
                raise ValueError("variances must be positive")
        else:
            self.variances = None
            self.covariance = np.asarray(covariance, dtype=np.float64)
            if self.covariance.shape != (n_bins, n_bins):
                raise ValueError(
                    f"covariance shape {self.covariance.shape} != ({n_bins}, {n_bins})"
                )

        self._free_names = [n for n in self.param_names if n not in self.fixed]
        self._fisher = None
        self._inverse = None

    def compute(self):
        """The Fisher matrix over the free parameters, (n_free, n_free)."""
        if self.covariance is None:
            weight = 1.0 / self.variances  # diagonal inverse covariance
            F_full = (self.J.T * weight) @ self.J
        else:
            c_inv = _prewhitened_inverse(self.covariance)
            F_full = self.J.T @ c_inv @ self.J
        for name, sigma in self.priors.items():
            i = self.param_names.index(name)
            F_full[i, i] += 1.0 / float(sigma) ** 2
        keep = [i for i, n in enumerate(self.param_names) if n not in self.fixed]
        F = F_full[np.ix_(keep, keep)]
        self._fisher = 0.5 * (F + F.T)
        self._inverse = None
        return self._fisher

    @property
    def fisher_matrix(self):
        if self._fisher is None:
            self.compute()
        return self._fisher

    @property
    def inverse(self):
        """F^-1 via a sqrt-diag prewhitened solve (conditioning insurance).

        The {A, b1} amplitude direction is near-degenerate (both scale ln P by a
        constant), so F can be ill-conditioned. Prewhitening with D = sqrt(diag F)
        gives F_w = D^-1 F D^-1 a unit diagonal and off-diagonals bounded by 1,
        dropping the condition number by orders of magnitude before the inverse;
        the result is unwhitened back (a standard prewhitened-solve trick).
        """
        if self._inverse is None:
            if np.any(np.diag(self.fisher_matrix) <= 0):
                raise ValueError(
                    "non-positive Fisher diagonal; an unconstrained free "
                    "parameter needs a prior or to be fixed"
                )
            self._inverse = _prewhitened_inverse(self.fisher_matrix)
        return self._inverse

    def _free_index(self, param):
        if param not in self._free_names:
            raise KeyError(f"{param!r} is not a free parameter ({self._free_names})")
        return self._free_names.index(param)

    def sigma(self, param):
        """Marginalized 1-sigma constraint, sqrt((F^-1)_aa)."""
        i = self._free_index(param)
        return float(np.sqrt(self.inverse[i, i]))

    def sigma_conditional(self, param):
        """Conditional 1-sigma constraint, 1/sqrt(F_aa) (others fixed)."""
        i = self._free_index(param)
        return float(1.0 / np.sqrt(self.fisher_matrix[i, i]))

    def marginalized_2d(self, param_i, param_j):
        """2D marginalized sub-covariance and 1-sigma error-ellipse parameters.

        Returns a dict with cov_2d, sigma_i, sigma_j, the correlation rho, and
        the ellipse position angle angle_deg (degrees, from the param_i axis).
        """
        i = self._free_index(param_i)
        j = self._free_index(param_j)
        F_inv = self.inverse
        cov_2d = np.array([[F_inv[i, i], F_inv[i, j]], [F_inv[j, i], F_inv[j, j]]])
        sigma_i = float(np.sqrt(cov_2d[0, 0]))
        sigma_j = float(np.sqrt(cov_2d[1, 1]))
        rho = float(cov_2d[0, 1] / (sigma_i * sigma_j))
        angle = 0.5 * np.arctan2(2.0 * cov_2d[0, 1], cov_2d[0, 0] - cov_2d[1, 1])
        return {
            "cov_2d": cov_2d,
            "sigma_i": sigma_i,
            "sigma_j": sigma_j,
            "rho": rho,
            "angle_deg": float(np.degrees(angle)),
        }

    def conditional_2d(self, param_i, param_j):
        """2D *conditional* sub-covariance (all other free params held fixed).

        The inverse of the 2x2 Fisher sub-block, vs marginalized_2d which inverts
        the full Fisher first. Same return fields as marginalized_2d; the
        conditional ellipse is tighter than the marginalized one. Useful for
        showing how much marginalizing over the other parameters costs.
        """
        i = self._free_index(param_i)
        j = self._free_index(param_j)
        cov_2d = np.linalg.inv(self.fisher_matrix[np.ix_([i, j], [i, j])])
        sigma_i = float(np.sqrt(cov_2d[0, 0]))
        sigma_j = float(np.sqrt(cov_2d[1, 1]))
        rho = float(cov_2d[0, 1] / (sigma_i * sigma_j))
        angle = 0.5 * np.arctan2(2.0 * cov_2d[0, 1], cov_2d[0, 0] - cov_2d[1, 1])
        return {
            "cov_2d": cov_2d,
            "sigma_i": sigma_i,
            "sigma_j": sigma_j,
            "rho": rho,
            "angle_deg": float(np.degrees(angle)),
        }

    def ellipse_xy(self, param_i, param_j, n_sigma=1.0, n_points=200):
        """1-sigma (or n_sigma) error-ellipse points centred on the fiducials.

        Returns (x, y) numpy arrays tracing the ellipse for plotting, using the
        2D marginalized sub-covariance eigen-decomposition. Centre is the
        fiducial (param_i, param_j).
        """
        m = self.marginalized_2d(param_i, param_j)
        evals, evecs = np.linalg.eigh(m["cov_2d"])
        t = np.linspace(0.0, 2.0 * np.pi, n_points)
        circle = np.stack([np.cos(t), np.sin(t)], axis=0)
        axes = n_sigma * np.sqrt(np.maximum(evals, 0.0))
        pts = evecs @ (axes[:, None] * circle)
        x0 = float(self.fiducial[param_i])
        y0 = float(self.fiducial[param_j])
        return x0 + pts[0], y0 + pts[1]

    def condition_number(self):
        """Condition number of the (free) Fisher matrix -- large = degenerate."""
        ev = np.linalg.eigvalsh(self.fisher_matrix)
        if ev[0] <= 0.0:
            return float("inf")  # singular (e.g. an unbroken product degeneracy)
        return float(ev[-1] / ev[0])

    def summary(self, name=""):
        """Human-readable forecast report: sigmas, conditioning, degeneracies."""
        lines = []
        title = "M-body Fisher forecast" + (f": {name}" if name else "")
        lines.append(title)
        lines.append(
            "  fiducial: "
            + ", ".join(f"{k}={self.fiducial[k]:g}" for k in self.param_names)
        )
        if self.fixed:
            lines.append("  fixed: " + ", ".join(self.fixed))
        if self.priors:
            lines.append(
                "  priors: "
                + ", ".join(f"sigma({k})={v:g}" for k, v in self.priors.items())
            )
        cond = self.condition_number()
        lines.append(
            f"  cond(F) = {cond:.3e}" + ("  [degenerate]" if cond > 1e12 else "")
        )
        lines.append("  param      sigma(marg)    sigma(cond)")
        for p in self._free_names:
            lines.append(
                f"  {p:<8s}  {self.sigma(p):13.5g}  {self.sigma_conditional(p):13.5g}"
            )
        return "\n".join(lines)


# --- Redshift-space multipole Fisher -----------------------------------------
#
# The redshift-space forecast adds the line-of-sight growth information that
# helps pin the b1 / amplitude block and partially lift its degeneracy with
# f_NL. Two changes from the isotropic log-band-power Fisher above:
#
#   * Data vector = the LINEAR multipole band powers P_ell(k) (not ln P): the
#     quadrupole can be negative per realization, so a log is ill-defined. The
#     multipoles are stacked ell-major (all bins of ell=0, then all of ell=2).
#   * A full block COVARIANCE replaces the diagonal variance, because P_0 and
#     P_2 are correlated within a k-bin. It is estimated from a Gaussian mock
#     ensemble (the standard disconnected forecast covariance), which captures
#     the cross-multipole and discrete-grid mode counting exactly.
#
# The fifth parameter f_growth (the RSD growth-rate amplitude) is downstream of
# the trajectory -- it scales the final-state velocity shift -- so its column is
# a cheap fixed-field mx.grad alongside b1, b2; f_NL and A remain IC-stage and
# share the reversible adjoint, now seeded with the momentum cotangent because
# the redshift-space field depends on the final velocities (loss_uses_momentum).


def _redshift_tracer_linear(delta, box, b1, b2, f_eff, los_axis):
    """Linear redshift-space tracer field, (b1 + f_eff mu^2) delta + b2 term.

    The Kaiser operator carries the linear bias and the line-of-sight velocity
    term (f_eff = f_growth * f_linear); the (b2/2)(delta^2 - <delta^2>) quadratic
    bias is the local real-space addition (isotropic, so it lands in the monopole
    at leading order). Differentiable in delta, b1, b2, f_eff.
    """
    kaiser = RS.apply_linear_kaiser(delta, box, b1, f_eff, los_axis=los_axis)
    mean2 = float(P.accurate_mean(delta**2))
    return kaiser + 0.5 * b2 * (delta**2 - mean2)


def linear_multipole_jacobian(
    box,
    cosmo,
    theta_fid,
    k_bins,
    ells=(0, 2),
    los_axis=0,
    seed=0,
    dk=None,
    backend="camb",
):
    """dP_ell/dtheta for the LINEAR redshift-space tracer, theta = PARAM_NAMES_RSD.

    Forward model (all MLX, differentiable):
        delta = A * ic.linear_density(f_NL)
        tracer_s = (b1 + f_growth f_lin mu^2) delta + (b2/2)(delta^2 - <delta^2>)
        mu_d = P_ell(k_b)            # linear multipole band power (not log)
    The Jacobian is built per data component with reverse-mode mx.grad. Returns
    (J, P_fid): J is (n_ell*n_bins, 5) float64 ell-major, P_fid the fiducial
    multipole band powers (same layout).
    """
    theta = mx.array([float(theta_fid[n]) for n in PARAM_NAMES_RSD])
    f_lin = C.growth_rate(0.0, cosmo)
    n_bins, n_ell = len(k_bins), len(ells)
    n_data = n_bins * n_ell

    def model_P(t):
        f_NL, b1, b2, A, fg = t[0], t[1], t[2], t[3], t[4]
        delta = A * IC.linear_density(box, cosmo, seed=seed, f_NL=f_NL, backend=backend)
        tracer = _redshift_tracer_linear(delta, box, b1, b2, fg * f_lin, los_axis)
        parts = [
            F.band_power_multipole(tracer, box, k_bins, el, los_axis=los_axis, dk=dk)
            for el in ells
        ]
        return mx.concatenate(parts)

    J = np.empty((n_data, len(PARAM_NAMES_RSD)), dtype=np.float64)
    for d in range(n_data):
        g = mx.grad(lambda t, d=d: model_P(t)[d])(theta)
        J[d, :] = np.asarray(g, dtype=np.float64)
    P_fid = np.asarray(model_P(theta), dtype=np.float64)
    return J, P_fid


def pm_multipole_jacobian(
    box,
    cosmo,
    time,
    theta_fid,
    k_bins,
    ells=(0, 2),
    los_axis=0,
    seed=0,
    dk=None,
    backend="camb",
    integrator=None,
    lpt_order=2,
):
    """dP_ell/dtheta for the PM-EVOLVED redshift-space tracer, theta=PARAM_NAMES_RSD.

    f_NL and A shape the primordial IC (evolved through LPT + the leapfrog); the
    redshift-space map (mbody.rsd) uses the final positions AND velocities, then
    the local-bias tracer is painted and its multipoles measured. Columns by where
    each parameter enters:

      * f_NL, A: IC-stage -> the reversible adjoint (adjoint_grad_ic) with
        loss_uses_momentum=True, since the loss depends on the final velocities;
        one O(grid)-memory sweep per data component returns both columns.
      * b1, b2, f_growth: downstream -- they act on the fixed final (x, p), so a
        cheap mx.grad at the stop_gradient'd final state (f_growth scales the RSD
        shift).

    Returns (J, P_fid): J is (n_ell*n_bins, 5) float64 ell-major, P_fid the
    fiducial multipole band powers.
    """
    fid = {n: float(theta_fid[n]) for n in PARAM_NAMES_RSD}
    z_final = time.z_final
    n_bins, n_ell = len(k_bins), len(ells)
    n_data = n_bins * n_ell
    J = np.empty((n_data, len(PARAM_NAMES_RSD)), dtype=np.float64)

    # Evolve once at the fiducial; the final state is fixed for the downstream cols.
    x_final, p_final = IN.leapfrog(
        box,
        cosmo,
        time,
        seed=seed,
        f_NL=mx.array(fid["f_NL"]),
        amplitude=mx.array(fid["A"]),
        backend=backend,
        integrator=integrator,
        lpt_order=lpt_order,
    )
    xf = mx.stop_gradient(x_final)
    pf = mx.stop_gradient(p_final)

    # Downstream b1, b2, f_growth: cheap mx.grad at the fixed final (x, p).
    td = mx.array([fid["b1"], fid["b2"], fid["f_growth"]])

    def downstream_P(t):
        s = RS.redshift_space_positions(
            xf, pf, box, cosmo, z=z_final, los_axis=los_axis, f_growth=t[2]
        )
        tracer = B.local_bias_tracer(F.interlaced_density_contrast(s, box), t[0], t[1])
        parts = [
            F.band_power_multipole(tracer, box, k_bins, el, los_axis=los_axis, dk=dk)
            for el in ells
        ]
        return mx.concatenate(parts)

    for d in range(n_data):
        g = mx.grad(lambda t, d=d: downstream_P(t)[d])(td)
        J[d, 1] = float(g[0])  # b1
        J[d, 2] = float(g[1])  # b2
        J[d, 4] = float(g[2])  # f_growth
    P_fid = np.asarray(downstream_P(td), dtype=np.float64)

    # IC columns f_NL, A: the shared trajectory adjoint, one sweep per component.
    def make_loss(ell, b):
        def loss_field(x, p):
            s = RS.redshift_space_positions(
                x,
                p,
                box,
                cosmo,
                z=z_final,
                los_axis=los_axis,
                f_growth=fid["f_growth"],
            )
            tracer = B.local_bias_tracer(
                F.interlaced_density_contrast(s, box), fid["b1"], fid["b2"]
            )
            return F.band_power_multipole(
                tracer, box, k_bins, ell, los_axis=los_axis, dk=dk
            )[b]

        return loss_field

    for d in range(n_data):
        ell, b = ells[d // n_bins], d % n_bins
        g_ic = IN.adjoint_grad_ic(
            make_loss(ell, b),
            box,
            cosmo,
            time,
            seed=seed,
            f_NL=fid["f_NL"],
            amplitude=fid["A"],
            backend=backend,
            integrator=integrator,
            lpt_order=lpt_order,
            loss_uses_momentum=True,
        )
        J[d, 0] = float(g_ic[0])  # f_NL
        J[d, 3] = float(g_ic[1])  # A
    return J, P_fid


def multipole_gaussian_covariance(
    box,
    cosmo,
    theta_fid,
    k_bins,
    ells=(0, 2),
    los_axis=0,
    n_mock=400,
    seed0=1000,
    dk=None,
    backend="camb",
    hartlap=True,
):
    """Gaussian (disconnected) covariance of the multipole band powers, from mocks.

    Generates n_mock Gaussian realizations of the linear redshift-space tracer at
    the fiducial parameters, measures the raw multipole band powers, and returns
    their sample covariance (n_ell*n_bins, ell-major). This is the standard
    Gaussian forecast covariance: it captures the cross-multipole correlation and
    the discrete-grid / rfft-plane mode counting exactly, because it is the sample
    covariance of the very estimator the Jacobian differentiates. Used for both
    the linear and PM forecasts (the PM nonlinearity enters the Jacobian/signal,
    not this disconnected covariance -- documented as the forecast approximation).
    With hartlap=True the covariance is scaled by 1/h, h=(n_mock-n_data-2)/(n_mock-1),
    so a Fisher's inverse-covariance is the unbiased (Hartlap-corrected) precision.
    """
    f_lin = C.growth_rate(0.0, cosmo)
    b1, b2 = float(theta_fid["b1"]), float(theta_fid["b2"])
    A, fg, fnl = (
        float(theta_fid["A"]),
        float(theta_fid["f_growth"]),
        float(theta_fid["f_NL"]),
    )
    n_bins, n_ell = len(k_bins), len(ells)
    n_data = n_bins * n_ell

    data = np.empty((n_mock, n_data), dtype=np.float64)
    for m in range(n_mock):
        delta = A * IC.linear_density(
            box, cosmo, seed=seed0 + m, f_NL=fnl, backend=backend
        )
        tracer = _redshift_tracer_linear(delta, box, b1, b2, fg * f_lin, los_axis)
        parts = [
            np.asarray(
                F.band_power_multipole(
                    tracer, box, k_bins, el, los_axis=los_axis, dk=dk
                ),
                dtype=np.float64,
            )
            for el in ells
        ]
        data[m, :] = np.concatenate(parts)

    cov = np.cov(data, rowvar=False)
    if hartlap:
        if n_mock <= n_data + 2:
            raise ValueError(
                f"n_mock={n_mock} too small for n_data={n_data}: the Hartlap "
                "factor needs n_mock > n_data + 2 (else h <= 0). Increase n_mock."
            )
        h = (n_mock - n_data - 2) / (n_mock - 1)
        cov = cov / h
    return cov


# --- Multi-tracer Fisher (the b_phi-f_NL degeneracy capstone) -----------------
#
# Two local-bias tracers A and B painted from the SAME field sample the same
# modes, so their sample variance is shared: the cross spectrum P_AB and the
# autos {P_AA, P_BB} together carry differential-bias information cosmic variance
# cannot wash out (Seljak 2009, arXiv:0807.1770). The multi-tracer constraining
# power on the local-f_NL scale-dependent bias goes as |b1_B b_phi_A - b1_A
# b_phi_B| (Barreira & Krause 2023, arXiv:2302.09066): it constrains the PRODUCTS
# f_NL*b_phi per tracer with NO b_phi prior, and pins f_NL itself only once a
# b_phi(b1) relation is imposed -- multi-tracer relaxes/robustifies that prior,
# it does not by itself break the degeneracy.
#
# In this toy b_phi is emergent from b2: dlnP_h/df_NL = 4 b2 A sigma^2/(b1 M(k))
# (bias.scale_dependent_bias_response) equals 2 b_phi/(b1 M(k)), so b_phi =
# 2 b2 A sigma^2 -- independent of b1. Two regimes are demonstrated:
#   * FREE: marginalize (b1_i, b2_i) per tracer; only f_NL*b_phi_i is constrained
#     and sigma(f_NL) blows up (the degeneracy survives, sharpened by the
#     cancellation of the products).
#   * TIED: impose universality b_phi = 2 delta_c (b1-1), i.e.
#     b2_i = delta_c (b1_i-1)/(A sigma^2); now f_NL is constrained and the gain
#     goes as |b1_A - b1_B| (universality_b2 / universality_tie_matrix).
#
# Data vector mu = [P_AA(k), P_AB(k), P_BB(k)] (spectrum-major), RAW linear band
# powers (the cross P_AB is not positive-definite, so no log) -> a full block
# COVARIANCE with per-tracer Poisson shot noise (the gain is shot-noise-limited;
# with noiseless fields the cancellation is formally perfect and the forecast
# vacuous). The IC params (f_NL, A) share one reversible-adjoint sweep per data
# component; the four b's are cheap downstream fixed-field gradients.

PARAM_NAMES_MT = ("f_NL", "A", "b1_A", "b2_A", "b1_B", "b2_B")


def _multitracer_vector(field_a, field_b, box, k_bins, dk):
    """[P_AA, P_AB, P_BB] of two real tracer fields, spectrum-major MLX vector."""
    return mx.concatenate(
        [
            F.band_power(field_a, box, k_bins, dk=dk),
            F.cross_power(field_a, field_b, box, k_bins, dk=dk),
            F.band_power(field_b, box, k_bins, dk=dk),
        ]
    )


def linear_multitracer_jacobian(
    box, cosmo, theta_fid, k_bins, seed=0, dk=None, backend="camb"
):
    """d{P_AA,P_AB,P_BB}/dtheta for two LINEAR-field local-bias tracers.

    theta = PARAM_NAMES_MT = (f_NL, A, b1_A, b2_A, b1_B, b2_B). Both tracers are
    painted from the SAME A*linear_density(f_NL); the data vector is the raw
    auto/cross band powers (spectrum-major). Built per data component with
    reverse-mode mx.grad. Returns (J, P_fid): J (3*n_bins, 6) float64, P_fid the
    (3*n_bins,) fiducial spectra (same layout).
    """
    theta = mx.array([float(theta_fid[n]) for n in PARAM_NAMES_MT])
    n_bins = len(k_bins)
    n_data = 3 * n_bins

    def model_P(t):
        f_NL, A, b1A, b2A, b1B, b2B = t[0], t[1], t[2], t[3], t[4], t[5]
        delta = A * IC.linear_density(box, cosmo, seed=seed, f_NL=f_NL, backend=backend)
        hA = B.local_bias_tracer(delta, b1A, b2A)
        hB = B.local_bias_tracer(delta, b1B, b2B)
        return _multitracer_vector(hA, hB, box, k_bins, dk)

    J = np.empty((n_data, len(PARAM_NAMES_MT)), dtype=np.float64)
    for d in range(n_data):
        g = mx.grad(lambda t, d=d: model_P(t)[d])(theta)
        J[d, :] = np.asarray(g, dtype=np.float64)
    P_fid = np.asarray(model_P(theta), dtype=np.float64)
    return J, P_fid


def pm_multitracer_jacobian(
    box,
    cosmo,
    time,
    theta_fid,
    k_bins,
    seed=0,
    dk=None,
    backend="camb",
    integrator=None,
    lpt_order=2,
):
    """d{P_AA,P_AB,P_BB}/dtheta for two PM-EVOLVED local-bias tracers.

    Both tracers are painted from the SAME final field (one trajectory). Columns
    by where each parameter enters, exactly as pm_logP_jacobian:

      * f_NL, A: IC-stage -> the reversible adjoint (adjoint_grad_ic), ONE
        O(grid)-memory sweep per data component (the loss paints BOTH tracers
        from the shared final state, so two tracers cost the same one sweep as
        one). Cost = 3*n_bins sweeps (3x single-tracer; independent of #tracers).
      * b1_A, b2_A, b1_B, b2_B: downstream -- cheap mx.grad at the stop_gradient'd
        final density (P_AA depends only on A's bias, P_BB only on B's, P_AB on
        all four; the zeros fall out).

    Returns (J, P_fid): J (3*n_bins, 6) float64, P_fid the fiducial spectra.
    """
    fid = {n: float(theta_fid[n]) for n in PARAM_NAMES_MT}
    n_bins = len(k_bins)
    n_data = 3 * n_bins
    J = np.empty((n_data, len(PARAM_NAMES_MT)), dtype=np.float64)

    # Evolve once at the fiducial; the final density is fixed for the b columns.
    x_final, _ = IN.leapfrog(
        box,
        cosmo,
        time,
        seed=seed,
        f_NL=mx.array(fid["f_NL"]),
        amplitude=mx.array(fid["A"]),
        backend=backend,
        integrator=integrator,
        lpt_order=lpt_order,
    )
    delta_final = mx.stop_gradient(F.interlaced_density_contrast(x_final, box))

    # Downstream b columns: cheap mx.grad at the fixed final field.
    td = mx.array([fid["b1_A"], fid["b2_A"], fid["b1_B"], fid["b2_B"]])

    def downstream_P(t):
        hA = B.local_bias_tracer(delta_final, t[0], t[1])
        hB = B.local_bias_tracer(delta_final, t[2], t[3])
        return _multitracer_vector(hA, hB, box, k_bins, dk)

    for d in range(n_data):
        g = mx.grad(lambda t, d=d: downstream_P(t)[d])(td)
        J[d, 2] = float(g[0])  # b1_A
        J[d, 3] = float(g[1])  # b2_A
        J[d, 4] = float(g[2])  # b1_B
        J[d, 5] = float(g[3])  # b2_B
    P_fid = np.asarray(downstream_P(td), dtype=np.float64)

    # IC columns f_NL, A: the shared trajectory adjoint, one sweep per component.
    def make_loss(d):
        def loss_field(x):
            field = F.interlaced_density_contrast(x, box)
            hA = B.local_bias_tracer(field, fid["b1_A"], fid["b2_A"])
            hB = B.local_bias_tracer(field, fid["b1_B"], fid["b2_B"])
            return _multitracer_vector(hA, hB, box, k_bins, dk)[d]

        return loss_field

    for d in range(n_data):
        g_ic = IN.adjoint_grad_ic(
            make_loss(d),
            box,
            cosmo,
            time,
            seed=seed,
            f_NL=fid["f_NL"],
            amplitude=fid["A"],
            backend=backend,
            integrator=integrator,
            lpt_order=lpt_order,
        )
        J[d, 0] = float(g_ic[0])  # f_NL
        J[d, 1] = float(g_ic[1])  # A
    return J, P_fid


def multitracer_analytic_covariance(box, k_bins, P_AA, P_AB, P_BB, n_A, n_B, dk=None):
    """Analytic Gaussian block covariance of [P_AA, P_AB, P_BB] with shot noise.

    The disconnected (Gaussian) covariance is

        Cov(P_ij, P_kl) = (Ptot_ik Ptot_jl + Ptot_il Ptot_jk) / N_modes,

    with Ptot_ii = P_ii + 1/n_i (auto spectra carry Poisson shot noise; the cross
    does not, for independent populations) and the effective mode count
    N_modes = 2 / Var[ln P_b] (band_power_log_variance, which carries the rfft
    half-grid plane double-count). Returns (3*n_bins, 3*n_bins) float64,
    spectrum-major, block-diagonal in k (Gaussian modes in different shells are
    independent). P_AA/P_AB/P_BB are the fiducial signal band powers (n_bins,);
    n_A/n_B are number densities (per (Mpc/h)^3, so 1/n is a power). This is the
    clean Gaussian approximation; multitracer_gaussian_covariance is the mock
    ground truth it is validated against.
    """
    P_AA = np.asarray(P_AA, dtype=np.float64)
    P_AB = np.asarray(P_AB, dtype=np.float64)
    P_BB = np.asarray(P_BB, dtype=np.float64)
    n_bins = len(k_bins)
    nmodes = 2.0 / band_power_log_variance(box, k_bins, dk=dk)
    tA = P_AA + 1.0 / float(n_A)
    tB = P_BB + 1.0 / float(n_B)
    x = P_AB
    cov = np.zeros((3 * n_bins, 3 * n_bins), dtype=np.float64)
    for b in range(n_bins):
        blk = (
            np.array(
                [
                    [2 * tA[b] ** 2, 2 * tA[b] * x[b], 2 * x[b] ** 2],
                    [2 * tA[b] * x[b], tA[b] * tB[b] + x[b] ** 2, 2 * tB[b] * x[b]],
                    [2 * x[b] ** 2, 2 * tB[b] * x[b], 2 * tB[b] ** 2],
                ]
            )
            / nmodes[b]
        )
        idx = [b, n_bins + b, 2 * n_bins + b]
        cov[np.ix_(idx, idx)] = blk
    return cov


def _mt_mock_covariance(
    paint, box, k_bins, n_A, n_B, n_mock, seed0, dk, shot, hartlap, measure=None
):
    """Mock covariance of a tracer-pair data vector for a tracer-pair painter.

    paint(seed) -> (hA, hB), the two real tracer fields for one realization.
    Optionally adds independent white shot noise to each (band power 1/n_i, the
    Poisson level for number density n_i per (Mpc/h)^3 -- cross spectra get none).
    measure(hA, hB) -> the data-vector row (a 1D float64 numpy array); defaults to
    the isotropic [P_AA, P_AB, P_BB] (_multitracer_vector). Pass a multipole measure
    for the redshift-space multi-tracer covariance. Returns the sample covariance,
    Hartlap-corrected when hartlap=True.
    """
    if measure is None:

        def measure(hA, hB):
            return np.asarray(
                _multitracer_vector(hA, hB, box, k_bins, dk), dtype=np.float64
            )

    N, V = box.n_mesh, box.box_size**3
    # real-space white-noise rms whose band power is 1/n_i (Poisson shot level)
    sigA = float(np.sqrt(N**3 / (float(n_A) * V)))
    sigB = float(np.sqrt(N**3 / (float(n_B) * V)))
    rows = []
    for m in range(n_mock):
        hA, hB = paint(seed0 + m)
        if shot:
            kA, kB = mx.random.split(mx.random.key(7_000_003 + seed0 + m))
            hA = hA + sigA * mx.random.normal((N, N, N), key=kA)
            hB = hB + sigB * mx.random.normal((N, N, N), key=kB)
        rows.append(measure(hA, hB))
    data = np.asarray(rows, dtype=np.float64)
    n_data = data.shape[1]
    cov = np.cov(data, rowvar=False)
    if hartlap:
        if n_mock <= n_data + 2:
            raise ValueError(
                f"n_mock={n_mock} too small for n_data={n_data}: the Hartlap "
                "factor needs n_mock > n_data + 2 (else h <= 0). Increase n_mock."
            )
        h = (n_mock - n_data - 2) / (n_mock - 1)
        cov = cov / h
    return cov


def multitracer_gaussian_covariance(
    box,
    cosmo,
    theta_fid,
    k_bins,
    n_A,
    n_B,
    n_mock=400,
    seed0=2000,
    dk=None,
    backend="camb",
    hartlap=True,
    shot=True,
):
    """Mock Gaussian covariance of [P_AA, P_AB, P_BB] with per-tracer shot noise.

    Generates n_mock Gaussian realizations of the two linear-field local-bias
    tracers at the fiducial parameters, optionally adds independent white shot
    noise to each, measures the raw auto/cross band powers, and returns their
    sample covariance (3*n_bins, spectrum-major). This is the ground-truth
    forecast covariance -- it captures the cross-spectrum mode statistics and the
    weak b2 non-Gaussianity exactly; multitracer_analytic_covariance is the clean
    Gaussian approximation validated against it. Hartlap-corrected when
    hartlap=True.
    """
    fnl, A = float(theta_fid["f_NL"]), float(theta_fid["A"])
    b1A, b2A = float(theta_fid["b1_A"]), float(theta_fid["b2_A"])
    b1B, b2B = float(theta_fid["b1_B"]), float(theta_fid["b2_B"])

    def paint(seed):
        delta = A * IC.linear_density(box, cosmo, seed=seed, f_NL=fnl, backend=backend)
        return B.local_bias_tracer(delta, b1A, b2A), B.local_bias_tracer(
            delta, b1B, b2B
        )

    return _mt_mock_covariance(
        paint, box, k_bins, n_A, n_B, n_mock, seed0, dk, shot, hartlap
    )


def universality_b2(b1, box, cosmo, A=1.0, delta_c=1.686, z=0.0, backend="camb"):
    """b2 making the toy's b_phi follow the universality relation b_phi =
    2 delta_c (b1 - 1).

    The toy's emergent b_phi = 2 b2 A sigma^2 (from dlnP_h/df_NL = 4 b2 A sigma^2
    / (b1 M) = 2 b_phi/(b1 M)), so universality is b2 = delta_c (b1 - 1)/
    (A sigma^2), sigma^2 = bias.mesh_variance. Returns a float.
    """
    sigma2 = B.mesh_variance(box, cosmo, z=z, backend=backend)
    return float(delta_c * (float(b1) - 1.0) / (float(A) * sigma2))


def universality_fiducial(
    f_NL, A, b1_A, b1_B, box, cosmo, delta_c=1.686, z=0.0, backend="camb"
):
    """Free-basis fiducial dict with b2_i set to the universality value.

    A J_full built at this fiducial is consistent with the linearization in
    universality_tie_matrix, so J_tied = J_full @ T is exact at the fiducial.
    """
    sigma2 = B.mesh_variance(box, cosmo, z=z, backend=backend)
    c = delta_c / (float(A) * sigma2)
    return {
        "f_NL": f_NL,
        "A": A,
        "b1_A": b1_A,
        "b2_A": float(c * (float(b1_A) - 1.0)),
        "b1_B": b1_B,
        "b2_B": float(c * (float(b1_B) - 1.0)),
    }


def universality_tie_matrix(
    theta_fid, box, cosmo, delta_c=1.686, z=0.0, backend="camb"
):
    """Constant chain-rule map T (6x4) from the free MT basis to the tied basis.

    Enforces b2_i = delta_c (b1_i-1)/(A sigma^2), so b_phi_i = 2 delta_c (b1_i-1);
    the tied basis is (f_NL, A, b1_A, b1_B). J_tied = J_full @ T, exact only when
    J_full is built at a universality-consistent fiducial (universality_fiducial).
    Returns (T, tied_names, tied_fiducial).
    """
    A = float(theta_fid["A"])
    b1A = float(theta_fid["b1_A"])
    b1B = float(theta_fid["b1_B"])
    sigma2 = B.mesh_variance(box, cosmo, z=z, backend=backend)
    db2_db1 = delta_c / (A * sigma2)
    db2A_dA = -delta_c * (b1A - 1.0) / (A**2 * sigma2)
    db2B_dA = -delta_c * (b1B - 1.0) / (A**2 * sigma2)
    T = np.zeros((6, 4), dtype=np.float64)
    T[0, 0] = 1.0  # f_NL -> f_NL
    T[1, 1] = 1.0  # A -> A (direct)
    T[2, 2] = 1.0  # b1_A -> b1_A (direct)
    T[4, 3] = 1.0  # b1_B -> b1_B (direct)
    T[3, 1] = db2A_dA  # b2_A <- A
    T[3, 2] = db2_db1  # b2_A <- b1_A
    T[5, 1] = db2B_dA  # b2_B <- A
    T[5, 3] = db2_db1  # b2_B <- b1_B
    tied_names = ("f_NL", "A", "b1_A", "b1_B")
    tied_fid = {"f_NL": float(theta_fid["f_NL"]), "A": A, "b1_A": b1A, "b1_B": b1B}
    return T, tied_names, tied_fid


def multitracer_forecast(
    J, cov, fiducial, param_names=PARAM_NAMES_MT, priors=None, tie=None
):
    """Assemble a multi-tracer FisherForecast, optionally universality-tied.

    With tie=None the FREE forecast over `param_names` is returned. To impose the
    universality relation pass `tie` = the (T, tied_names, tied_fiducial) tuple
    from universality_tie_matrix (native b2 tracer) or universality_bphi_tie_matrix
    (explicit-b_phi tracer); then J_tied = J @ T and the forecast is over the tied
    basis. The covariance is identical free or tied (the tie reparametrizes the
    signal, not the data). J must have been built at a universality-consistent
    fiducial (universality_fiducial / universality_bphi_fiducial) for the tie to
    be exact.
    """
    if tie is None:
        return FisherForecast(
            J,
            covariance=cov,
            fiducial_params=fiducial,
            param_names=param_names,
            priors=priors,
        )
    T, tied_names, tied_fid = tie
    return FisherForecast(
        np.asarray(J, dtype=np.float64) @ T,
        covariance=cov,
        fiducial_params=tied_fid,
        param_names=tied_names,
        priors=priors,
    )


# --- Explicit-b_phi multi-tracer (the clean all-k degeneracy) ------------------
#
# The native (b2-sourced) tracer reproduces the b_phi-f_NL degeneracy only at low
# k -- at high k the b2 broadband loop self-calibrates b_phi (an artifact the real
# b_phi lacks). bias.scale_dependent_bias_tracer makes b_phi an EXPLICIT parameter
# entering ONLY as the k^-2 scale-dependent bias delta_h(k) = [b1 + b_phi f_NL/
# M(k)] delta(k), on a GAUSSIAN field. Then d/df_NL and d/db_phi share the 1/M(k)
# shape and are PERFECTLY degenerate (the product f_NL*b_phi) at all k -- the
# faithful Barreira/Dalal degeneracy, invisible at f_NL=0 (the cross-term vanishes)
# and sharpest at f_NL ~ sigma(f_NL). Universality is the clean b_phi=2 delta_c
# (b1-1) (no mesh sigma^2). The forecast is linear-field only: the degeneracy is a
# large-scale linear-bias statement that the PM evolution does not change (the PM
# autodiff showcase is the native model's job).

PARAM_NAMES_MT_BPHI = ("f_NL", "A", "b1_A", "bphi_A", "b1_B", "bphi_B")


def _inv_M_grid(box, cosmo, z=0.0, backend="camb"):
    """1/M(k) on the rfft half-grid (k=0 -> 0), the scale-dependent-bias kernel."""
    _, _, k_mag = F.k_grid(box)
    M = IC.poisson_M(np.where(k_mag > 0, k_mag, 1.0), cosmo, z=z, backend=backend)
    return np.where(k_mag > 0, 1.0 / M, 0.0).astype(np.float32)


def linear_multitracer_bphi_jacobian(
    box, cosmo, theta_fid, k_bins, seed=0, dk=None, backend="camb"
):
    """d{P_AA,P_AB,P_BB}/dtheta for two EXPLICIT-b_phi scale-dependent-bias tracers.

    theta = PARAM_NAMES_MT_BPHI = (f_NL, A, b1_A, bphi_A, b1_B, bphi_B). Both
    tracers are painted from the SAME Gaussian (f_NL=0) linear field with
    bias.scale_dependent_bias_tracer; f_NL enters ONLY through the k^-2 term, so
    f_NL and b_phi are perfectly degenerate (the product) at all k. Reverse-mode
    mx.grad per component. Returns (J, P_fid): J (3*n_bins, 6) float64.
    """
    theta = mx.array([float(theta_fid[n]) for n in PARAM_NAMES_MT_BPHI])
    invM = _inv_M_grid(box, cosmo, backend=backend)
    n_bins = len(k_bins)
    n_data = 3 * n_bins

    def model_P(t):
        f_NL, A, b1A, bpA, b1B, bpB = t[0], t[1], t[2], t[3], t[4], t[5]
        delta = A * IC.linear_density(box, cosmo, seed=seed, f_NL=0.0, backend=backend)
        hA = B.scale_dependent_bias_tracer(delta, box, cosmo, b1A, bpA, f_NL, invM=invM)
        hB = B.scale_dependent_bias_tracer(delta, box, cosmo, b1B, bpB, f_NL, invM=invM)
        return _multitracer_vector(hA, hB, box, k_bins, dk)

    J = np.empty((n_data, len(PARAM_NAMES_MT_BPHI)), dtype=np.float64)
    for d in range(n_data):
        g = mx.grad(lambda t, d=d: model_P(t)[d])(theta)
        J[d, :] = np.asarray(g, dtype=np.float64)
    P_fid = np.asarray(model_P(theta), dtype=np.float64)
    return J, P_fid


def multitracer_bphi_gaussian_covariance(
    box,
    cosmo,
    theta_fid,
    k_bins,
    n_A,
    n_B,
    n_mock=400,
    seed0=2000,
    dk=None,
    backend="camb",
    hartlap=True,
    shot=True,
):
    """Mock Gaussian covariance for the explicit-b_phi tracer pair (at the
    fiducial f_NL / b_phi). Same shot-noise treatment as
    multitracer_gaussian_covariance.
    """
    fnl, A = float(theta_fid["f_NL"]), float(theta_fid["A"])
    b1A, bpA = float(theta_fid["b1_A"]), float(theta_fid["bphi_A"])
    b1B, bpB = float(theta_fid["b1_B"]), float(theta_fid["bphi_B"])
    invM = _inv_M_grid(box, cosmo, backend=backend)

    def paint(seed):
        delta = A * IC.linear_density(box, cosmo, seed=seed, f_NL=0.0, backend=backend)
        hA = B.scale_dependent_bias_tracer(delta, box, cosmo, b1A, bpA, fnl, invM=invM)
        hB = B.scale_dependent_bias_tracer(delta, box, cosmo, b1B, bpB, fnl, invM=invM)
        return hA, hB

    return _mt_mock_covariance(
        paint, box, k_bins, n_A, n_B, n_mock, seed0, dk, shot, hartlap
    )


def universality_bphi_fiducial(f_NL, A, b1_A, b1_B, delta_c=1.686):
    """Explicit-b_phi fiducial with b_phi_i = 2 delta_c (b1_i - 1) (universality)."""
    return {
        "f_NL": f_NL,
        "A": A,
        "b1_A": b1_A,
        "bphi_A": 2.0 * delta_c * (float(b1_A) - 1.0),
        "b1_B": b1_B,
        "bphi_B": 2.0 * delta_c * (float(b1_B) - 1.0),
    }


def universality_bphi_tie_matrix(theta_fid, delta_c=1.686):
    """Chain-rule map T (6x4) for the explicit-b_phi model: free
    (f_NL, A, b1_A, bphi_A, b1_B, bphi_B) -> tied (f_NL, A, b1_A, b1_B), enforcing
    b_phi_i = 2 delta_c (b1_i - 1) (so db_phi/db1 = 2 delta_c, db_phi/dA = 0 --
    cleaner than the native tie, no mesh sigma^2). Returns (T, tied_names,
    tied_fiducial).
    """
    T = np.zeros((6, 4), dtype=np.float64)
    T[0, 0] = 1.0  # f_NL -> f_NL
    T[1, 1] = 1.0  # A -> A
    T[2, 2] = 1.0  # b1_A -> b1_A
    T[4, 3] = 1.0  # b1_B -> b1_B
    T[3, 2] = 2.0 * delta_c  # bphi_A <- b1_A
    T[5, 3] = 2.0 * delta_c  # bphi_B <- b1_B
    tied_names = ("f_NL", "A", "b1_A", "b1_B")
    tied_fid = {
        "f_NL": float(theta_fid["f_NL"]),
        "A": float(theta_fid["A"]),
        "b1_A": float(theta_fid["b1_A"]),
        "b1_B": float(theta_fid["b1_B"]),
    }
    return T, tied_names, tied_fid


# --- Redshift-space multi-tracer Fisher (the composition capstone) -------------
#
# Compose the two subsystems above: two local-bias tracers A, B painted from the
# SAME redshift-space field, summarized by their auto/cross multipoles. This adds
# the line-of-sight growth handle (the quadrupole pins f_growth) ON TOP OF the
# multi-tracer sample-variance cancellation, and exercises the genuinely new
# autodiff path -- the two-tracer cross-spectrum gradient seeded on the FINAL
# VELOCITIES (the momentum-seeded reversible adjoint). The scientific conclusion
# is unchanged from the isotropic case (RSD does not break b_phi*f_NL; multi-tracer
# constrains the products); this is the completeness/composition piece.
#
# Only the NATIVE tracer is built here (b_phi emergent from b2): the explicit-b_phi
# clean degeneracy is a large-scale linear, isotropic statement that the multipoles
# do not change. Data vector mu = the raw LINEAR multipole band powers, spectrum-
# major then ell-major: [P_AA^0, P_AA^2, P_AB^0, P_AB^2, P_BB^0, P_BB^2] (cross and
# quadrupole are not positive-definite -> no log -> a full block COVARIANCE).
# Shot noise is white -> it enters the data-vector MEAN in the monopole auto power
# only (it still raises the covariance of all multipoles, like any Gaussian term).

PARAM_NAMES_MT_RSD = ("f_NL", "A", "f_growth", "b1_A", "b2_A", "b1_B", "b2_B")


def _mt_multipole_vector(field_a, field_b, box, k_bins, ells, los_axis, dk):
    """[P_AA, P_AB, P_BB] multipoles of two real tracer fields, spectrum-major then
    ell-major: [AA(ell0..bins), AA(ell2..), AB(ell0..), AB(ell2..), BB(ell0..),
    BB(ell2..)]. AA/BB are band_power_multipole, AB is cross_power_multipole."""
    parts = []
    for el in ells:
        parts.append(
            F.band_power_multipole(field_a, box, k_bins, el, los_axis=los_axis, dk=dk)
        )
    for el in ells:
        parts.append(
            F.cross_power_multipole(
                field_a, field_b, box, k_bins, el, los_axis=los_axis, dk=dk
            )
        )
    for el in ells:
        parts.append(
            F.band_power_multipole(field_b, box, k_bins, el, los_axis=los_axis, dk=dk)
        )
    return mx.concatenate(parts)


def linear_multitracer_multipole_jacobian(
    box,
    cosmo,
    theta_fid,
    k_bins,
    ells=(0, 2),
    los_axis=0,
    seed=0,
    dk=None,
    backend="camb",
):
    """dP_ell^{AA,AB,BB}/dtheta for two LINEAR redshift-space local-bias tracers.

    theta = PARAM_NAMES_MT_RSD = (f_NL, A, f_growth, b1_A, b2_A, b1_B, b2_B). Both
    tracers are painted from the SAME A*linear_density(f_NL) with the shared Kaiser
    velocity term f_eff = f_growth * f_linear (one velocity field); the data vector
    is the raw auto/cross multipole band powers (spectrum-major then ell-major).
    Built per data component with reverse-mode mx.grad. Returns (J, P_fid): J
    (3*n_ell*n_bins, 7) float64, P_fid the fiducial spectra (same layout).
    """
    theta = mx.array([float(theta_fid[n]) for n in PARAM_NAMES_MT_RSD])
    f_lin = C.growth_rate(0.0, cosmo)
    n_bins, n_ell = len(k_bins), len(ells)
    n_data = 3 * n_ell * n_bins

    def model_P(t):
        f_NL, A, fg = t[0], t[1], t[2]
        b1A, b2A, b1B, b2B = t[3], t[4], t[5], t[6]
        delta = A * IC.linear_density(box, cosmo, seed=seed, f_NL=f_NL, backend=backend)
        hA = _redshift_tracer_linear(delta, box, b1A, b2A, fg * f_lin, los_axis)
        hB = _redshift_tracer_linear(delta, box, b1B, b2B, fg * f_lin, los_axis)
        return _mt_multipole_vector(hA, hB, box, k_bins, ells, los_axis, dk)

    J = np.empty((n_data, len(PARAM_NAMES_MT_RSD)), dtype=np.float64)
    for d in range(n_data):
        g = mx.grad(lambda t, d=d: model_P(t)[d])(theta)
        J[d, :] = np.asarray(g, dtype=np.float64)
    P_fid = np.asarray(model_P(theta), dtype=np.float64)
    return J, P_fid


def pm_multitracer_multipole_jacobian(
    box,
    cosmo,
    time,
    theta_fid,
    k_bins,
    ells=(0, 2),
    los_axis=0,
    seed=0,
    dk=None,
    backend="camb",
    integrator=None,
    lpt_order=2,
):
    """dP_ell^{AA,AB,BB}/dtheta for two PM-EVOLVED redshift-space local-bias tracers.

    The merge of pm_multipole_jacobian (RSD single-tracer) and
    pm_multitracer_jacobian: f_NL and A shape the primordial IC (evolved through
    LPT + the leapfrog); the redshift-space map (mbody.rsd) uses the final positions
    AND velocities, then BOTH tracers are painted from the same final field and
    their auto/cross multipoles measured. Columns by where each parameter enters:

      * f_NL, A: IC-stage -> the reversible adjoint (adjoint_grad_ic) with
        loss_uses_momentum=True (the loss depends on the final velocities); one
        O(grid)-memory sweep per data component returns both columns (the loss paints
        both tracers from the shared final state, so two tracers cost one sweep).
      * f_growth, b1_A, b2_A, b1_B, b2_B: downstream -- a cheap mx.grad at the fixed
        final (x, p); f_growth scales the single RSD shift feeding both tracers.

    Returns (J, P_fid): J (3*n_ell*n_bins, 7) float64, P_fid the fiducial spectra.
    """
    fid = {n: float(theta_fid[n]) for n in PARAM_NAMES_MT_RSD}
    z_final = time.z_final
    n_bins, n_ell = len(k_bins), len(ells)
    n_data = 3 * n_ell * n_bins
    J = np.empty((n_data, len(PARAM_NAMES_MT_RSD)), dtype=np.float64)

    # Evolve once at the fiducial; the final state is fixed for the downstream cols.
    x_final, p_final = IN.leapfrog(
        box,
        cosmo,
        time,
        seed=seed,
        f_NL=mx.array(fid["f_NL"]),
        amplitude=mx.array(fid["A"]),
        backend=backend,
        integrator=integrator,
        lpt_order=lpt_order,
    )
    xf = mx.stop_gradient(x_final)
    pf = mx.stop_gradient(p_final)

    # Downstream f_growth, b1_A, b2_A, b1_B, b2_B: cheap mx.grad at fixed (x, p).
    td = mx.array([fid["f_growth"], fid["b1_A"], fid["b2_A"], fid["b1_B"], fid["b2_B"]])

    def downstream_P(t):
        s = RS.redshift_space_positions(
            xf, pf, box, cosmo, z=z_final, los_axis=los_axis, f_growth=t[0]
        )
        field = F.interlaced_density_contrast(s, box)
        hA = B.local_bias_tracer(field, t[1], t[2])
        hB = B.local_bias_tracer(field, t[3], t[4])
        return _mt_multipole_vector(hA, hB, box, k_bins, ells, los_axis, dk)

    for d in range(n_data):
        g = mx.grad(lambda t, d=d: downstream_P(t)[d])(td)
        J[d, 2] = float(g[0])  # f_growth
        J[d, 3] = float(g[1])  # b1_A
        J[d, 4] = float(g[2])  # b2_A
        J[d, 5] = float(g[3])  # b1_B
        J[d, 6] = float(g[4])  # b2_B
    P_fid = np.asarray(downstream_P(td), dtype=np.float64)

    # IC columns f_NL, A: the shared momentum-seeded adjoint, one sweep per component.
    def make_loss(d):
        def loss_field(x, p):
            s = RS.redshift_space_positions(
                x,
                p,
                box,
                cosmo,
                z=z_final,
                los_axis=los_axis,
                f_growth=fid["f_growth"],
            )
            field = F.interlaced_density_contrast(s, box)
            hA = B.local_bias_tracer(field, fid["b1_A"], fid["b2_A"])
            hB = B.local_bias_tracer(field, fid["b1_B"], fid["b2_B"])
            return _mt_multipole_vector(hA, hB, box, k_bins, ells, los_axis, dk)[d]

        return loss_field

    for d in range(n_data):
        g_ic = IN.adjoint_grad_ic(
            make_loss(d),
            box,
            cosmo,
            time,
            seed=seed,
            f_NL=fid["f_NL"],
            amplitude=fid["A"],
            backend=backend,
            integrator=integrator,
            lpt_order=lpt_order,
            loss_uses_momentum=True,
        )
        J[d, 0] = float(g_ic[0])  # f_NL
        J[d, 1] = float(g_ic[1])  # A
    return J, P_fid


def multitracer_multipole_gaussian_covariance(
    box,
    cosmo,
    theta_fid,
    k_bins,
    n_A,
    n_B,
    ells=(0, 2),
    los_axis=0,
    n_mock=400,
    seed0=3000,
    dk=None,
    backend="camb",
    hartlap=True,
    shot=True,
):
    """Mock Gaussian covariance of the redshift-space multi-tracer multipoles.

    Generates n_mock Gaussian realizations of the two linear redshift-space
    local-bias tracers at the fiducial parameters, optionally adds independent
    white shot noise to each, measures the raw auto/cross multipole band powers,
    and returns their sample covariance (3*n_ell*n_bins, spectrum-major then
    ell-major). This is the ground-truth forecast covariance -- it captures the
    cross-multipole AND cross-spectrum coupling exactly (there is no analytic
    multipole block, as for the single-tracer RSD covariance). Hartlap-corrected
    when hartlap=True.

    Shot-noise physics: the Poisson field is white (isotropic), so its SIGNAL is
    the monopole 1/n_i only -- the quadrupole and the cross-spectrum means are
    unchanged (an isotropic field has no even multipole above ell = 0, up to the raw
    discrete-shell leakage). Adding it in real space reproduces this in the data-
    vector mean automatically. Note that, like any Gaussian term, shot still raises
    the COVARIANCE of ALL multipoles (the variance of a multipole estimator depends
    on the total power P + 1/n at every mu); it is not confined to the monopole there.
    """
    f_lin = C.growth_rate(0.0, cosmo)
    fnl, A, fg = (
        float(theta_fid["f_NL"]),
        float(theta_fid["A"]),
        float(theta_fid["f_growth"]),
    )
    b1A, b2A = float(theta_fid["b1_A"]), float(theta_fid["b2_A"])
    b1B, b2B = float(theta_fid["b1_B"]), float(theta_fid["b2_B"])

    def paint(seed):
        delta = A * IC.linear_density(box, cosmo, seed=seed, f_NL=fnl, backend=backend)
        hA = _redshift_tracer_linear(delta, box, b1A, b2A, fg * f_lin, los_axis)
        hB = _redshift_tracer_linear(delta, box, b1B, b2B, fg * f_lin, los_axis)
        return hA, hB

    def measure(hA, hB):
        return np.asarray(
            _mt_multipole_vector(hA, hB, box, k_bins, ells, los_axis, dk),
            dtype=np.float64,
        )

    return _mt_mock_covariance(
        paint, box, k_bins, n_A, n_B, n_mock, seed0, dk, shot, hartlap, measure=measure
    )


def universality_rsd_fiducial(
    f_NL, A, f_growth, b1_A, b1_B, box, cosmo, delta_c=1.686, z=0.0, backend="camb"
):
    """PARAM_NAMES_MT_RSD fiducial with b2_i at the universality value and f_growth
    inserted -- the redshift-space sibling of universality_fiducial. A J_full built
    here is consistent with universality_rsd_tie_matrix's linearization.
    """
    fid = universality_fiducial(
        f_NL, A, b1_A, b1_B, box, cosmo, delta_c=delta_c, z=z, backend=backend
    )
    return {
        "f_NL": fid["f_NL"],
        "A": fid["A"],
        "f_growth": f_growth,
        "b1_A": fid["b1_A"],
        "b2_A": fid["b2_A"],
        "b1_B": fid["b1_B"],
        "b2_B": fid["b2_B"],
    }


def universality_rsd_tie_matrix(
    theta_fid, box, cosmo, delta_c=1.686, z=0.0, backend="camb"
):
    """Constant chain-rule map T (7x5) from the free MT-RSD basis to the tied basis
    (f_NL, A, f_growth, b1_A, b1_B), enforcing b2_i = delta_c (b1_i-1)/(A sigma^2)
    (so b_phi_i = 2 delta_c (b1_i-1)). f_growth is a direct passthrough, left FREE
    (a distinct growth-rate parameter, NOT tied). J_tied = J_full @ T, exact only
    when J_full is built at universality_rsd_fiducial. Mirrors universality_tie_matrix
    with the f_growth column inserted. Returns (T, tied_names, tied_fiducial).
    """
    A = float(theta_fid["A"])
    b1A = float(theta_fid["b1_A"])
    b1B = float(theta_fid["b1_B"])
    sigma2 = B.mesh_variance(box, cosmo, z=z, backend=backend)
    db2_db1 = delta_c / (A * sigma2)
    db2A_dA = -delta_c * (b1A - 1.0) / (A**2 * sigma2)
    db2B_dA = -delta_c * (b1B - 1.0) / (A**2 * sigma2)
    # free rows:  0 f_NL, 1 A, 2 f_growth, 3 b1_A, 4 b2_A, 5 b1_B, 6 b2_B
    # tied cols:  0 f_NL, 1 A, 2 f_growth, 3 b1_A, 4 b1_B
    T = np.zeros((7, 5), dtype=np.float64)
    T[0, 0] = 1.0  # f_NL -> f_NL
    T[1, 1] = 1.0  # A -> A (direct)
    T[2, 2] = 1.0  # f_growth -> f_growth (passthrough)
    T[3, 3] = 1.0  # b1_A -> b1_A (direct)
    T[5, 4] = 1.0  # b1_B -> b1_B (direct)
    T[4, 1] = db2A_dA  # b2_A <- A
    T[4, 3] = db2_db1  # b2_A <- b1_A
    T[6, 1] = db2B_dA  # b2_B <- A
    T[6, 4] = db2_db1  # b2_B <- b1_B
    tied_names = ("f_NL", "A", "f_growth", "b1_A", "b1_B")
    tied_fid = {
        "f_NL": float(theta_fid["f_NL"]),
        "A": A,
        "f_growth": float(theta_fid["f_growth"]),
        "b1_A": b1A,
        "b1_B": b1B,
    }
    return T, tied_names, tied_fid
