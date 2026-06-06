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
  row/column. This mirrors the forecast Fisher in ~/cmb/cmb-augr/augr/fisher.py
  (J^T C^-1 J), not a Hessian-of-(-logP) at an MLE -- this toy has no
  likelihood/optimizer.

Gradients are built bin-by-bin with reverse-mode mx.grad (one sweep of the
scalar ln P_b returns the whole parameter row); forward-mode mx.jvp is WRONG
through the FFT here, so it is never used. The FisherForecast object itself is
pure float64 numpy linear algebra -- small (n_params x n_params), off the GPU
hot path, and separately testable from the forward model.
"""

import numpy as np
import mlx.core as mx

from mbody import bias as B
from mbody import fields as F
from mbody import ic as IC
from mbody import integrate as IN
from mbody import painting as PA
from mbody import precision as P  # noqa: F401  (used by Jacobian builders)

# Canonical parameter order for the M-body autodiff Fisher.
PARAM_NAMES = ("f_NL", "b1", "b2", "A")


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
    delta_final = mx.stop_gradient(PA.density_contrast(x_final, box))

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
            tracer = B.local_bias_tracer(PA.density_contrast(x, box), fid_b1, fid_b2)
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


class FisherForecast:
    """Diagonal-covariance Fisher matrix and parameter constraints.

    F_ij = sum_b J_{b,i} J_{b,j} / Var[ln P_b] (+ prior 1/sigma^2 on the
    diagonal), with fixed parameters dropped. The Jacobian and per-bin variances
    are passed in as plain arrays -- the forward-model Jacobian assembly
    (linear_logP_jacobian, pm_logP_jacobian) is deliberately decoupled from this
    pure linear-algebra object, which mirrors the ergonomics of
    ~/cmb/cmb-augr/augr/fisher.py.

    Parameters
    ----------
    jacobian : (n_bins, n_params) array, J[b, i] = d mu_b / d theta_i (mu = lnP).
    variances : (n_bins,) array, Var[ln P_b] (from band_power_log_variance).
    fiducial_params : {name: value} of every parameter (for summary/reference).
    param_names : ordered names matching the columns of `jacobian`.
    priors : optional {name: sigma_prior}; adds 1/sigma^2 to the diagonal.
    fixed_params : optional list of names to hold fixed (drop from F).
    """

    def __init__(
        self,
        jacobian,
        variances,
        fiducial_params,
        param_names=PARAM_NAMES,
        priors=None,
        fixed_params=None,
    ):
        self.J = np.asarray(jacobian, dtype=np.float64)
        self.variances = np.asarray(variances, dtype=np.float64)
        self.param_names = list(param_names)
        self.fiducial = dict(fiducial_params)
        self.priors = dict(priors or {})
        self.fixed = list(fixed_params or [])

        n_bins, n_params = self.J.shape
        if n_params != len(self.param_names):
            raise ValueError(
                f"jacobian has {n_params} columns but {len(self.param_names)} "
                "param_names"
            )
        if self.variances.shape != (n_bins,):
            raise ValueError(f"variances shape {self.variances.shape} != ({n_bins},)")
        if np.any(self.variances <= 0):
            raise ValueError("variances must be positive")
        self._free_names = [n for n in self.param_names if n not in self.fixed]
        self._fisher = None
        self._inverse = None

    def compute(self):
        """The Fisher matrix over the free parameters, (n_free, n_free)."""
        weight = 1.0 / self.variances  # inverse band-power covariance, diagonal
        F_full = (self.J.T * weight) @ self.J
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
        the result is unwhitened back. Same trick as cmb-augr's per-bin solve.
        """
        if self._inverse is None:
            F = self.fisher_matrix
            d = np.sqrt(np.diag(F))
            if np.any(d <= 0):
                raise ValueError(
                    "non-positive Fisher diagonal; an unconstrained free "
                    "parameter needs a prior or to be fixed"
                )
            d_inv = 1.0 / d
            F_w = F * np.outer(d_inv, d_inv)
            self._inverse = np.linalg.inv(F_w) * np.outer(d_inv, d_inv)
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
