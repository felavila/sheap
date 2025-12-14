r"""
Loss Function Builder
=====================

This module defines the construction of flexible loss functions used in *sheap*
for spectral fitting and optimization.

Contents
--------
- **build_loss_function**: Factory for JAX-compatible scalar loss functions
    combining residuals, penalties, and regularization.

Loss Components
---------------
The constructed loss may include the following terms:

1. **Data fidelity (log-cosh residuals)**

.. math::
    \mathcal{L}_\text{data} =
    \langle \log\cosh(r) \rangle + \alpha \, \max(\log\cosh(r)),
    \quad r = \frac{y_\text{pred} - y}{\sigma}

2. **Optional penalty on parameters**

.. math::
    \mathcal{L}_\text{penalty} =
    \beta \, \text{penalty\_function}(x, \theta)

3. **Curvature matching**

.. math::
    \mathcal{L}_\text{curvature} =
        \gamma \, \langle (f''_\text{pred} - f''_\text{true})^2 \rangle

4. **Residual smoothness**

.. math::
    \mathcal{L}_\text{smoothness} =
        \delta \, \langle (\nabla r)^2 \rangle

Notes
-----
- `penalty_function` can enforce additional physics or priors.
- `param_converter` allows transformation from raw to physical parameters.
- All terms are implemented using JAX and are fully differentiable.

Example
-------
.. code-block:: python

   from sheap.Minimizer.loss_builder import build_loss_function

   loss_fn = build_loss_function(model_fn, weighted=True, curvature_weight=1e4)
   loss_val = loss_fn(params, x_grid, flux, flux_err)
"""

__author__ = 'felavila'

__all__ = [
    "build_loss_function",
]

from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap,lax

def build_loss_function(
    func: Callable,
    weighted: bool = True,
    penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    penalty_weight: float = 0.01,
    param_converter: Optional["Parameters"] = None,
    curvature_weight: float = 1e3,      # γ: second-derivative match 1e5
    smoothness_weight: float = 1e5,     # δ: first-derivative smoothness 0.0
    max_weight: float = 0.1,            # α: weight on worst‐pixel term
) -> Callable:
    r"""
    Build a flexible JAX-compatible loss function for regression-style modeling tasks.

    This loss function combines several components:

    **1. Data term using log-cosh residuals**

    .. math::
    
        \text{data} = \operatorname{mean}(\log\cosh(r)) + \alpha \cdot \max(\log\cosh(r)),
        \quad \text{where } r = \frac{y_\text{pred} - y}{y_\text{err}}

    **2. Optional penalty term on parameters**

    .. math::
    
        \text{penalty} = \beta \cdot \text{penalty\_function}(x, \theta)

    **3. Optional curvature matching (second derivative difference)**

    .. math::
    
        \text{curvature} = \gamma \cdot \operatorname{mean}[(f''_\text{pred} - f''_\text{true})^2]

    **4. Optional smoothness penalty on the residuals**
    
    .. math::
    
        \text{smoothness} = \delta \cdot \operatorname{mean}[(\nabla r)^2]

    Parameters
    ----------
    func : Callable
        The prediction function, called as ``func(xs, phys_params)``, returning ``y_pred``.
    weighted : bool, default=True
        Whether to apply inverse error weighting to the residuals.
    penalty_function : Callable, optional
        A callable penalty term ``penalty(xs, params) → scalar loss``, scaled by ``penalty_weight``.
    penalty_weight : float, default=0.01
        Coefficient for the penalty function term.
    param_converter : Parameters, optional
        Object with a ``raw_to_phys`` method to convert raw to physical parameters.
    curvature_weight : float, default=1e3
        Coefficient for the second-derivative matching term.
    smoothness_weight : float, default=1e5
        Coefficient for smoothness of the residuals.
    max_weight : float, default=0.1
        Weight for the maximum log-cosh residual relative to the mean.

    Returns
    -------
    Callable
        A loss function with signature ``(params, xs, y, yerr) → scalar``,
        where ``params`` are raw parameters (optionally converted to physical).
    """

    #print("smoothness_weight =",smoothness_weight,"penalty_weight =",penalty_weight,"max_weight=",max_weight,"curvature_weight=",curvature_weight)
    def log_cosh(x):
        # numerically stable log(cosh(x))
        return jnp.logaddexp(x, -x) - jnp.log(2.0)

    def wrapped(xs, raw_params):
        phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
        return func(xs, phys)

    def curvature_term(y_pred, y):
        d2p = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
        d2o = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
        return jnp.nanmean((d2p - d2o)**2)

    def smoothness_term(y_pred, y):
        dr = y_pred - y
        dp = jnp.gradient(dr, axis=-1)
        return jnp.nanmean(dp**2)

    if weighted and penalty_function:
        def loss(params, xs, y, yerr):
            y_pred   = wrapped(xs, params)
            r        = (y_pred - y) / jnp.clip(yerr, 1e-8)

            # data term = mean + max
            Lmean    = jnp.nanmean(log_cosh(r))
            Lmax     = jnp.max   (log_cosh(r))
            data_term = Lmean + max_weight * Lmax

            # penalty on params
            reg_term = penalty_weight * penalty_function(xs, params) * 1e3

            # curvature & smoothness
            curv_term   = curvature_weight  * curvature_term(y_pred, y)
            smooth_term = smoothness_weight * smoothness_term(y_pred, y)

            return data_term + reg_term + curv_term + smooth_term

        return loss

    elif weighted:
        def loss(params, xs, y, yerr):
            y_pred   = wrapped(xs, params)
            r        = (y_pred - y) / jnp.clip(yerr, 1e-8)

            Lmean    = jnp.nanmean(log_cosh(r))
            Lmax     = jnp.max   (log_cosh(r))
            data_term = Lmean + max_weight * Lmax

            curv_term   = curvature_weight  * curvature_term(y_pred, y)
            smooth_term = smoothness_weight * smoothness_term(y_pred, y)

            return data_term + curv_term + smooth_term

        return loss

    elif penalty_function:
        def loss(params, xs, y, yerr):
            y_pred   = wrapped(xs, params)
            r        = (y_pred - y)

            Lmean    = jnp.nanmean(log_cosh(r))
            Lmax     = jnp.max   (log_cosh(r))
            data_term = Lmean + max_weight * Lmax

            reg_term    = penalty_weight * penalty_function(xs, params) * 1e3
            curv_term   = curvature_weight  * curvature_term(y_pred, y)
            smooth_term = smoothness_weight * smoothness_term(y_pred, y)

            return data_term + reg_term + curv_term + smooth_term

        return loss

    else:
        def loss(params, xs, y, yerr):
            y_pred   = wrapped(xs, params)
            r        = (y_pred - y)

            Lmean    = jnp.nanmean(log_cosh(r))
            Lmax     = jnp.max   (log_cosh(r))
            data_term = Lmean + max_weight * Lmax

            curv_term   = curvature_weight  * curvature_term(y_pred, y)
            smooth_term = smoothness_weight * smoothness_term(y_pred, y)

            return data_term + curv_term + smooth_term

        return loss
    
    
def _solve_weighted_linear_least_squares(
    A: jnp.ndarray,
    y: jnp.ndarray,
    yerr: jnp.ndarray,
    lambda_reg: float = 0.0,
    reg_matrix: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    r"""
    Solve the weighted linear least-squares problem for amplitudes.

    Minimizes

    .. math::

        \| W (y - A a) \|^2 + \lambda \| R a \|^2,

    where :math:`W = \mathrm{diag}(1/\sigma)`.

    Parameters
    ----------
    A : jnp.ndarray
        Design matrix, shape (n_pix, n_lin).
    y : jnp.ndarray
        Observed spectrum, shape (n_pix,).
    yerr : jnp.ndarray
        1-sigma uncertainties, shape (n_pix,).
    lambda_reg : float, default=0.0
        Regularization strength :math:`\lambda`.
    reg_matrix : jnp.ndarray, optional
        Regularization operator :math:`R`; if None, the identity is used.

    Returns
    -------
    jnp.ndarray
        Optimal amplitudes ``a_star``, shape (n_lin,).
    """
    w = 1.0 / jnp.clip(yerr, 1e-10)
    Aw = A * w[:, None]        # (n_pix, n_lin)
    yw = y * w                 # (n_pix,)

    ATA = Aw.T @ Aw            # (n_lin, n_lin)
    ATy = Aw.T @ yw            # (n_lin,)

    if lambda_reg > 0.0:
        n_lin = ATA.shape[0]
        if reg_matrix is None:
            R = jnp.eye(n_lin, dtype=A.dtype)
        else:
            R = reg_matrix
        ATA = ATA + lambda_reg * (R.T @ R)

    a_star = jnp.linalg.solve(ATA, ATy)
    return a_star


def build_varpro_loss_function(
    func: Callable,
    weighted: bool = True,
    penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    penalty_weight: float = 0.01,
    param_converter: Optional["Parameters"] = None,
    curvature_weight: float = 0.0,
    smoothness_weight: float = 0.0,
    max_weight: float = 0.0,
    lambda_reg: float = 0.0,
    reg_matrix: Optional[jnp.ndarray] = None,
) -> Callable:
    r"""
    Build a loss function using variable projection for linear amplitudes.

    This variant assumes that a subset of parameters in ``param_converter`` are
    linear-in-the-model (e.g., amplitudes, weights). Those are *not* optimized
    directly; instead, for each evaluation of the non-linear parameters, the
    optimal amplitudes are solved by weighted linear least squares.

    The optimizer operates only on the non-linear subset of the raw parameter
    vector, but the loss function signature remains compatible with
    :func:`build_loss_function`::

        loss(params, xs, y, yerr)

    where here ``params`` are the *non-linear* raw parameters.

    Parameters
    ----------
    func : Callable
        Model function called as ``func(xs, phys_params) -> y_pred``.
    weighted : bool, default=True
        Whether to use inverse-variance weighting via ``yerr``.
    penalty_function : Callable, optional
        Penalty term evaluated as ``penalty(xs, phys_full)`` and scaled by
        ``penalty_weight``.
    penalty_weight : float, default=0.01
        Global weight for the penalty term.
    param_converter : Parameters, optional
        Parameter container with ``raw_to_phys`` and linear/non-linear indices.
    curvature_weight : float, default=0.0
        Coefficient for curvature-matching term.
    smoothness_weight : float, default=0.0
        Coefficient for smoothness of residuals.
    max_weight : float, default=0.0
        Coefficient for the max-logcosh term relative to the mean.
    lambda_reg : float, default=0.0
        Regularization strength for the linear amplitudes.
    reg_matrix : jnp.ndarray, optional
        Regularization operator for amplitudes (defaults to identity if None
        and ``lambda_reg > 0``).

    Returns
    -------
    Callable
        A loss function with signature ``loss(raw_nl, xs, y, yerr) -> scalar``,
        where ``raw_nl`` are the non-linear raw parameters.
    """

    if param_converter is None:
        raise ValueError("build_varpro_loss_function requires a param_converter.")

    linear_phys_idx    = param_converter.linear_phys_indices
    nonlinear_raw_idx  = param_converter.nonlinear_raw_indices
    n_free             = len(param_converter._raw_list)

    def log_cosh(x):
        return jnp.logaddexp(x, -x) - jnp.log(2.0)

    def curvature_term(y_pred, y):
        d2p = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
        d2o = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
        return jnp.nanmean((d2p - d2o) ** 2)

    def smoothness_term(y_pred, y):
        dr = y_pred - y
        dp = jnp.gradient(dr, axis=-1)
        return jnp.nanmean(dp ** 2)

    def build_design_matrix_and_base(
        xs: jnp.ndarray,
        raw_nl: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Given non-linear raw params, build:

        - A: design matrix (n_pix, n_lin),
        - y_base: baseline spectrum with all linear amplitudes = 0,
        - phys_full: full physical parameter vector corresponding to raw_nl
                     (with linear entries set to 0 in phys space).
        """
        # 1) Build a full raw vector with only non-linear entries filled.
        full_raw = jnp.zeros((n_free,), dtype=raw_nl.dtype)
        full_raw = full_raw.at[nonlinear_raw_idx].set(raw_nl)

        # 2) Convert to physical parameters
        phys_full = param_converter.raw_to_phys(full_raw)  # (n_total,)

        # 3) Zero out physical linear amplitudes to define the baseline
        phys_base = phys_full.at[linear_phys_idx].set(0.0)

        # 4) Baseline spectrum with all linear amps = 0
        y_base = func(xs, phys_base)  # (n_pix,)

        n_lin = linear_phys_idx.shape[0]
        j_idx = jnp.arange(n_lin, dtype=int)

        # 5) For each linear parameter j, set phys_base[lin_phys_idx[j]] = 1
        #    and compute its basis contribution.
        def basis_col(j):
            phys_j = phys_base.at[linear_phys_idx[j]].set(1.0)
            y_j = func(xs, phys_j)
            return y_j - y_base  # contribution per unit amplitude

        cols = jax.vmap(basis_col)(j_idx)  # (n_lin, n_pix)
        A = cols.T                         # (n_pix, n_lin)

        return A, y_base, phys_base

    def loss(raw_nl: jnp.ndarray, xs: jnp.ndarray, y: jnp.ndarray, yerr: jnp.ndarray) -> jnp.ndarray:
        # 1) Build design matrix and baseline
        A, y_base, phys_base = build_design_matrix_and_base(xs, raw_nl)

        # 2) Solve for optimal amplitudes
        if weighted:
            a_star = _solve_weighted_linear_least_squares(
                A, y, yerr, lambda_reg=lambda_reg, reg_matrix=reg_matrix
            )
            y_pred = y_base + A @ a_star
            r = (y_pred - y) / jnp.clip(yerr, 1e-8)
        else:
            a_star = _solve_weighted_linear_least_squares(
                A, y, jnp.ones_like(yerr), lambda_reg=lambda_reg, reg_matrix=reg_matrix
            )
            y_pred = y_base + A @ a_star
            r = (y_pred - y)

        # 3) Data term = mean + max log-cosh
        Lmean = jnp.nanmean(log_cosh(r))
        Lmax  = jnp.max(log_cosh(r))
        data_term = Lmean + max_weight * Lmax

        # 4) Optional penalty on the physical parameters (using phys_base, i.e.
        #    with linear amps set to 0; you can adjust if you prefer otherwise)
        penalty_term = 0.0
        if penalty_function is not None and penalty_weight != 0.0:
            penalty_term = penalty_weight * penalty_function(xs, phys_base)

        # 5) Optional curvature & smoothness terms
        curv_term   = curvature_weight  * curvature_term(y_pred, y)
        smooth_term = smoothness_weight * smoothness_term(y_pred, y)

        return data_term + penalty_term + curv_term + smooth_term

    return loss
