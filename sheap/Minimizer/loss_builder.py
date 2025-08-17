"""This module handles basic operations."""
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


#####################################################################################################################
# curvature_weight=1e-3,
#     smoothness_weight=1e-4,
#     softmax_weight=0.1,
#     softmax_beta=0.07,

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     curvature_weight: float = 1e-3,      # γ: second-derivative match
#     smoothness_weight: float = 1e-4,     # δ: first-derivative smoothness
#     softmax_weight: float = 0.1,        # α: weight on the soft-max term
#     softmax_beta:   float = 0.07,        # β: sharpness of the soft-max
# ) -> Callable:
#     """
#     Build a loss(params, xs, y, yerr) with:
#       - log_cosh data term (mean + soft-max)
#       - optional parameter penalty
#       - curvature matching (d²)
#       - smoothness regularization (d¹)
#     """
#     print("γ,δ,α,β")
#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def pixel_losses(r):
#         # per-pixel log-cosh loss
#         return jnp.logaddexp(r, -r) - jnp.log(2.0)

#     def softmax_term(Li: jnp.ndarray) -> jnp.ndarray:
#         # differentiable approximation of max(Li)
#         return (1.0 / softmax_beta) * jnp.log(jnp.sum(jnp.exp(softmax_beta * Li)))

#     def curvature_term(y_pred, y):
#         d2p = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
#         d2o = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
#         return jnp.nanmean((d2p - d2o) ** 2)

#     def smoothness_term(y_pred, y):
#         dr = y_pred - y
#         dp = jnp.gradient(dr, axis=-1)
#         return jnp.nanmean(dp ** 2)

#     # -------------------------------------------------------------------
#     # Weighted + penalty
#     if weighted and penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             Li = pixel_losses(r)
#             Lmean = jnp.nanmean(Li)
#             Lsoft = softmax_term(Li)
#             data_term = Lmean + softmax_weight * Lsoft

#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term = smoothness_weight * smoothness_term(y_pred, y)

#             return data_term + reg_term + curv_term + smooth_term

#         return loss

#     # -------------------------------------------------------------------
#     # Weighted only
#     elif weighted:
#         def loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             Li = pixel_losses(r)
#             Lmean = jnp.nanmean(Li)
#             Lsoft = softmax_term(Li)
#             data_term = Lmean + softmax_weight * Lsoft

#             curv_term = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term = smoothness_weight * smoothness_term(y_pred, y)

#             return data_term + curv_term + smooth_term

#         return loss

#     # -------------------------------------------------------------------
#     # Unweighted + penalty
#     elif penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y)

#             Li = pixel_losses(r)
#             Lmean = jnp.nanmean(Li)
#             Lsoft = softmax_term(Li)
#             data_term = Lmean + softmax_weight * Lsoft

#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term = smoothness_weight * smoothness_term(y_pred, y)

#             return data_term + reg_term + curv_term + smooth_term

#         return loss

#     # -------------------------------------------------------------------
#     # Unweighted only
#     else:
#         def loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y)

#             Li = pixel_losses(r)
#             Lmean = jnp.nanmean(Li)
#             Lsoft = softmax_term(Li)
#             data_term = Lmean + softmax_weight * Lsoft

#             curv_term = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term = smoothness_weight * smoothness_term(y_pred, y)

#             return data_term + curv_term + smooth_term

#         return loss


# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     curvature_weight: float = 1e6,      # γ: second-derivative match
#     smoothness_weight: float = 1e6,     # δ: first-derivative smoothness
# ) -> Callable:
#     """
#     Build a loss function with:
#       - log_cosh data term
#       - optional parameter penalty
#       - curvature matching (d²)
#       - smoothness regularization (d¹)
#     """
#     print("loss smut curvature")
#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def curvature_term(y_pred, y):
#         d2p = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
#         d2o = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
#         return jnp.nanmean((d2p - d2o)**2)

#     def smoothness_term(y_pred, y):
#         dr  = y_pred - y
#         dp  = jnp.gradient(dr, axis=-1)
#         return jnp.nanmean(dp**2)

#     # Weighted + penalty
#     if weighted and penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred     = wrapped(xs, params)
#             r          = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term  = jnp.nanmean(log_cosh(r))
#             reg_term   = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term  = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term= smoothness_weight * smoothness_term(y_pred, y)
#             return data_term + reg_term + curv_term + smooth_term
#         return loss

#     # Weighted only
#     elif weighted:
#         def loss(params, xs, y, yerr):
#             y_pred     = wrapped(xs, params)
#             r          = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term  = jnp.nanmean(log_cosh(r))
#             curv_term  = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term= smoothness_weight * smoothness_term(y_pred, y)
#             return data_term + curv_term + smooth_term
#         return loss

#     # Unweighted + penalty
#     elif penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred     = wrapped(xs, params)
#             r          = (y_pred - y)
#             data_term  = jnp.nanmean(log_cosh(r))
#             reg_term   = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term  = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term= smoothness_weight * smoothness_term(y_pred, y)
#             return data_term + reg_term + curv_term + smooth_term
#         return loss

#     # Unweighted only
#     else:
#         def loss(params, xs, y, yerr):
#             y_pred     = wrapped(xs, params)
#             data_term  = jnp.nanmean(log_cosh(y_pred - y))
#             curv_term  = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term= smoothness_weight * smoothness_term(y_pred, y)
#             return data_term + curv_term + smooth_term
#         return loss
    
# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     curvature_weight: float = 1e6,      # new curvature weight γ
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function with optional curvature penalty.

#     Args:
#         func: The model function, called as func(xs, params)
#         weighted: If True, normalize residuals by yerr
#         penalty_function: Optional regularization function; called as penalty_function(xs, params)
#         penalty_weight: Multiplier for that regularization term
#         param_converter: Optional Parameters() object to transform raw → phys
#         curvature_weight: Weight for the curvature-matching term

#     Returns:
#         A loss function with signature (params, xs, y, yerr) -> scalar loss
#     """
#     print("loss_with_curvature_penalty")
#     def log_cosh(x):
#         # log(cosh(x)) in a numerically stable way
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     def curvature_term(y_pred, y):
#         # assumes last axis is the spectral pixel dimension
#         d2_pred = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
#         d2_obs  = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
#         return jnp.nanmean((d2_pred - d2_obs) ** 2)

#     # Weighted + penalty
#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + reg_term + curv_term
#         return weighted_with_penalty

#     # Weighted only
#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(log_cosh(r))
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + curv_term
#         return weighted_loss

#     # Unweighted + penalty
#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = y_pred - y
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + reg_term + curv_term
#         return unweighted_with_penalty

#     # Unweighted only
#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             data_term = jnp.nanmean(log_cosh(y_pred - y))
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + curv_term
#         return unweighted_loss


# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     huber_k: float = 2.0,   # Δ = huber_k * scale_estimate(r)
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using Adaptive Huber.
#     The worst one 
#     Args:
#         func:             Model function, called as func(xs, phys_params).
#         weighted:         If True, divide residuals by yerr.
#         penalty_function: Optional extra penalty, called as penalty_function(xs, params).
#         penalty_weight:   Multiplier for that penalty.
#         param_converter:  Optional raw→physical parameter converter.
#         huber_k:          Multiplier for adaptive Δ (default 1.0).

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar
#     """

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def adaptive_huber(r: jnp.ndarray) -> jnp.ndarray:
#         # 1) robust scale estimate via MAD
#         med  = jnp.nanmedian(r)
#         mad  = jnp.nanmedian(jnp.abs(r - med)) + 1e-8
#         scale = 1.4826 * mad

#         # 2) threshold Δ = k * scale
#         Δ = huber_k * scale

#         # 3) Huber formula
#         abs_r = jnp.abs(r)
#         return jnp.where(
#             abs_r <= Δ,
#             0.5 * r**2,
#             Δ * (abs_r - 0.5 * Δ)
#         )

#     def make_reg(xs, params):
#         if penalty_function:
#             return penalty_weight * penalty_function(xs, params) * 1e3
#         return 0.0

#     # Four‐branch logic, identical shape to your original:
#     if weighted and penalty_function:
#         def fn(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data     = jnp.nanmean(adaptive_huber(r))
#             return data + make_reg(xs, params)
#         return fn

#     elif weighted:
#         def fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(adaptive_huber(r))
#         return fn

#     elif penalty_function:
#         def fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y)
#             data   = jnp.nanmean(adaptive_huber(r))
#             return data + make_reg(xs, params)
#         return fn

#     else:
#         def fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y)
#             return jnp.nanmean(adaptive_huber(r))
#         return fn

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     cauchy_scale: float = 5.0,
#     curvature_weight: float = 10,    # new γ parameter
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using a Cauchy robust loss plus
#     an optional curvature penalty (second-derivative match).

#     Args:
#         func: model function, called as func(xs, params)
#         weighted: normalize residuals by yerr if True
#         penalty_function: optional regularizer func(xs, params)
#         penalty_weight: multiplier for that regularizer
#         param_converter: optional raw→phys parameter transformer
#         cauchy_scale: scale for robust Cauchy loss
#         curvature_weight: weight for curvature term
#     Returns:
#         loss(params, xs, y, yerr) → scalar
#     """

#     def cauchy_loss(r):
#         return jnp.log1p((r / cauchy_scale) ** 2)

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def curvature_term(y_pred, y):
#         # assumes last axis is the spectral pixel dimension
#         d2_pred = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
#         d2_obs  = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
#         return jnp.nanmean((d2_pred - d2_obs) ** 2)

#     # Weighted + penalty
#     if weighted and penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + reg_term + curv_term

#     # Weighted only
#     elif weighted:
#         def loss(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + curv_term

#     # Unweighted + penalty
#     elif penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = (y_pred - y)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + reg_term + curv_term

#     # Unweighted only
#     else:
#         def loss(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = (y_pred - y)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + curv_term

#     return loss

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     cauchy_scale: float = 100.0,  # heavier clipping for smaller scale
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using a Cauchy robust loss.

#     Args:
#         func: The model function, called as func(xs, params)
#         weighted: If True, normalize residuals by yerr
#         penalty_function: Optional regularization function; called as penalty_function(xs, params)
#         penalty_weight: Multiplier for that regularization term
#         param_converter: Optional Parameters() object to transform raw → phys
#         cauchy_scale: Scale parameter for the Cauchy loss:
#                       loss(r) = log(1 + (r / cauchy_scale)**2)
#     Returns:
#         A loss(params, xs, y, yerr) -> scalar loss
#     """
#     def cauchy_loss(r):
#         # elementwise: log(1 + (r/scale)^2)
#         return jnp.log1p((r / cauchy_scale) ** 2)

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(cauchy_loss(r))
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = y_pred - y
#             data_term = jnp.nanmean(cauchy_loss(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = y_pred - y
#             return jnp.nanmean(cauchy_loss(r))
#         return unweighted_loss

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using L2 (“p=2”) residuals.

#     Args:
#         func: The model function, called as func(xs, params)
#         weighted: If True, normalize residuals by yerr
#         penalty_function: Optional regularization function; called as penalty_function(xs, params)
#         penalty_weight: Multiplier for that regularization term
#         param_converter: Optional Parameters() object to transform raw → phys

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar loss
#     """
#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(jnp.abs(r)**2)                            # <-- L2 here
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(jnp.abs(r)**2)                                # <-- L2 here
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = y_pred - y
#             data_term = jnp.nanmean(jnp.abs(r)**2)                          # <-- L2 here
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = y_pred - y
#             return jnp.nanmean(jnp.abs(r)**2)                                # <-- L2 here
#         return unweighted_loss
    
# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     huber_delta: float = 30,         
# ) -> Callable:
#     """
#     Build a JIT‐compiled loss function using a Huber (“rubber”) loss.

#     Args:
#         func: The model function, called as func(xs, params)
#         weighted: whether to divide by yerr
#         penalty_function: optional regularization term
#         penalty_weight: its multiplier
#         param_converter: optional raw → phys transform
#         huber_delta: transition point between quadratic and linear

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar
#     """
#     def huber(r):
#         abs_r = jnp.abs(r)
#         # quadratic for |r|<=δ, linear beyond
#         return jnp.where(abs_r <= huber_delta,
#                          0.5 * r**2,
#                          huber_delta * (abs_r - 0.5 * huber_delta))

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(huber(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(huber(r))
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y
#             data_term = jnp.nanmean(huber(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             return jnp.nanmean(huber(y_pred - y))
#         return unweighted_loss
# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     residual_power: float = 10.0,    # Emphasis on large residuals
#     uncertainty_power: float = 1.0, # Emphasis on small uncertainties
#     huber_delta: float = 30.0,      # Transition point for Huber loss
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using Huber loss with hybrid adaptive weighting.

#     Args:
#         func: Model function, called as func(xs, params).
#         weighted: Whether to normalize residuals by yerr.
#         penalty_function: Optional regularization function.
#         penalty_weight: Multiplier for the regularization term.
#         param_converter: Optional raw → physical parameter converter.
#         residual_power: Controls emphasis on large residuals.
#         uncertainty_power: Controls emphasis on small uncertainties.
#         huber_delta: Threshold δ between quadratic and linear Huber regions.

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar loss
#     """

#     def huber(r):
#         abs_r = jnp.abs(r)
#         return jnp.where(
#             abs_r <= huber_delta,
#             0.5 * r**2,
#             huber_delta * (abs_r - 0.5 * huber_delta)
#         )

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def make_reg_term(params, xs):
#         # PPXF or any custom penalty can go here via penalty_function
#         if penalty_function:
#             return penalty_weight * penalty_function(xs, params) * 1e3
#         else:
#             return 0.0

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             rw = jnp.abs(residuals) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             w = rw * uw

#             data_term = jnp.nansum(huber(residuals) * w) / (jnp.nansum(w) + 1e-8)
#             return data_term + make_reg_term(params, xs)

#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             rw = jnp.abs(residuals) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             w = rw * uw

#             return jnp.nansum(huber(residuals) * w) / (jnp.nansum(w) + 1e-8)

#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = y_pred - y

#             rw = jnp.abs(residuals) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             w = rw * uw

#             data_term = jnp.nansum(huber(residuals) * w) / (jnp.nansum(w) + 1e-8)
#             return data_term + make_reg_term(params, xs)

#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = y_pred - y

#             rw = jnp.abs(residuals) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             w = rw * uw

#             return jnp.nansum(huber(residuals) * w) / (jnp.nansum(w) + 1e-8)

#         return unweighted_loss

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     residual_power: float = 2.0,      # emphasis on large residuals
#     uncertainty_power: float = 1.0,   # emphasis on small yerr
#     derivative_power: float = 2.0,    # emphasis on steep spectral gradients
#     huber_delta: float = 1e3,        # Huber transition
# ) -> Callable:
#     """
#     Build a JIT-compiled loss that
#       1) uses Huber for robust residuals,
#       2) adapts weights by residual magnitude,
#       3) adapts weights by 1/yerr,
#       4) adapts weights by |d(obs)/dx|,
#       5) optionally adds a penalty_function(xs, params).

#     Args:
#         func: model → flux, called func(xs, phys_params).
#         weighted: if True, normalize residuals by yerr.
#         penalty_function: extra reg term, signature (xs, params)->scalar.
#         penalty_weight: multiplier for penalty_function.
#         param_converter: raw→physical converter.
#         residual_power: power on |residual|.
#         uncertainty_power: power on 1/yerr.
#         derivative_power: power on |d(obs)/dx|.
#         huber_delta: δ for Huber loss.

#     Returns:
#         loss(params, xs, y, yerr) → scalar
#     """

#     def huber(r):
#         a = jnp.abs(r)
#         return jnp.where(a <= huber_delta,
#                          0.5 * r**2,
#                          huber_delta * (a - 0.5 * huber_delta))

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def make_penalty(xs, params):
#         if penalty_function:
#             return penalty_weight * penalty_function(xs, params) * 1e3
#         return 0.0

#     def feature_weight(y, xs):
#         # |d(obs)/dx|^derivative_power
#         # xs: wavelengths, y: observed flux (same shape)
#         grad = jnp.abs(jnp.gradient(y, xs))
#         return grad ** derivative_power

#     # four branches:
#     if weighted:
#         def loss_fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             rw = jnp.abs(r) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             dw = feature_weight(y, xs)

#             w = rw * uw * (1.0 + dw)   # add 1 so flat regions still count

#             data = jnp.nansum(huber(r) * w) / (jnp.nansum(w) + 1e-8)
#             return data + make_penalty(xs, params)

#     else:
#         def loss_fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y

#             rw = jnp.abs(r) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             dw = feature_weight(y, xs)

#             w = rw * uw * (1.0 + dw)

#             data = jnp.nansum(huber(r) * w) / (jnp.nansum(w) + 1e-8)
#             return data + make_penalty(xs, params)

#     return loss_fn

# def ppxf_smoothness_penalty(params: jnp.ndarray, lambda_reg: float = 1.0) -> jnp.ndarray:
#     """
#     Compute a PPXF-like smoothness penalty on `params`:
#       penalty = lambda_reg * sum( (params[i+2] - 2*params[i+1] + params[i])^2 ) over i
#     """
#     second_diff = jnp.diff(params, n=2)              # shape (N-2,)
#     penalty    = jnp.sum(second_diff**2)             # scalar
#     return lambda_reg * penalty
# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,  # ignore for now
#     penalty_weight: float = 0.01,   # multiplier for *that* penalty
#     param_converter: Optional["Parameters"] = None,
#     ppxf_lambda: float = 1.0,       # << new!
# ) -> Callable:
#     """
#     Build a JIT-compiled loss with optional PPXF smoothness penalty.
#     """
#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     # choose where to inject the PPXF penalty:
#     def make_reg_term(params):
#         # original penalty_function is still available if you want to combine both:
#         extra = 0.0
#         if penalty_function:
#             extra = penalty_weight * penalty_function(xs, params) * 1e3
#         # now add the PPXF smoothness:
#         smooth = ppxf_smoothness_penalty(params, lambda_reg=ppxf_lambda)
#         return smooth + extra

#     if weighted:
#         def weighted_with_ppxf(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data = jnp.nanmean(log_cosh(r))
#             return data + make_reg_term(params)
#         return weighted_with_ppxf

#     else:
#         def unweighted_with_ppxf(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             data = jnp.nanmean(log_cosh(y_pred - y))
#             return data + make_reg_term(params)
#         return unweighted_with_ppxf

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function.

#     Args:
#         func: The model function, called as func(xs, params)
#         param_converter: Optional Parameters() object to transform raw → phys

#     Returns:
#         A loss function with signature (params, xs, y, yerr) -> scalar loss
#     """
#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(log_cosh(r))
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             return jnp.nanmean(log_cosh(y_pred - y))
#         return unweighted_loss

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     residual_power: float = 2.0,       # Emphasis on large residuals
#     uncertainty_power: float = 1.0,    # Emphasis on small uncertainties
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using hybrid adaptive weighting (residual + uncertainty).

#     Args:
#         func: Model function, called as func(xs, params).
#         weighted: Whether to normalize residuals by uncertainties (yerr).
#         penalty_function: Optional regularization function.
#         penalty_weight: Multiplier for penalty term.
#         param_converter: Optional object to transform raw → physical parameters.
#         residual_power: Controls emphasis on large residuals (adaptive weighting).
#         uncertainty_power: Controls emphasis on uncertainties.

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar loss
#     """

#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             residual_weights = jnp.abs(residuals) ** residual_power
#             uncertainty_weights = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             combined_weights = residual_weights * uncertainty_weights

#             data_term = jnp.nansum(log_cosh(residuals) * combined_weights) / (jnp.nansum(combined_weights) + 1e-8)

#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term

#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             residual_weights = jnp.abs(residuals) ** residual_power
#             uncertainty_weights = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             combined_weights = residual_weights * uncertainty_weights

#             return jnp.nansum(log_cosh(residuals) * combined_weights) / (jnp.nansum(combined_weights) + 1e-8)

#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = y_pred - y

#             residual_weights = jnp.abs(residuals) ** residual_power
#             uncertainty_weights = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             combined_weights = residual_weights * uncertainty_weights

#             data_term = jnp.nansum(log_cosh(residuals) * combined_weights) / (jnp.nansum(combined_weights) + 1e-8)

#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term

#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = y_pred - y

#             residual_weights = jnp.abs(residuals) ** residual_power
#             uncertainty_weights = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             combined_weights = residual_weights * uncertainty_weights

#             return jnp.nansum(log_cosh(residuals) * combined_weights) / (jnp.nansum(combined_weights) + 1e-8)

#         return unweighted_loss


# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     residual_power: float = 2.0,      # emphasis on large residuals
#     uncertainty_power: float = 1.0,   # emphasis on small yerr
#     derivative_power: float = 1.0,    # emphasis on steep spectral gradients
#     huber_delta: float = 15.0,        # Huber transition
# ) -> Callable:
#     """
#     Build a JIT-compiled loss that
#       1) uses Huber for robust residuals,
#       2) adapts weights by residual magnitude,
#       3) adapts weights by 1/yerr,
#       4) adapts weights by |d(obs)/dx|,
#       5) optionally adds a penalty_function(xs, params).

#     Args:
#         func: model → flux, called func(xs, phys_params).
#         weighted: if True, normalize residuals by yerr.
#         penalty_function: extra reg term, signature (xs, params)->scalar.
#         penalty_weight: multiplier for penalty_function.
#         param_converter: raw→physical converter.
#         residual_power: power on |residual|.
#         uncertainty_power: power on 1/yerr.
#         derivative_power: power on |d(obs)/dx|.
#         huber_delta: δ for Huber loss.

#     Returns:
#         loss(params, xs, y, yerr) → scalar
#     """

#     def huber(r):
#         a = jnp.abs(r)
#         return jnp.where(a <= huber_delta,
#                          0.5 * r**2,
#                          huber_delta * (a - 0.5 * huber_delta))

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def make_penalty(xs, params):
#         if penalty_function:
#             return penalty_weight * penalty_function(xs, params) * 1e3
#         return 0.0

#     def feature_weight(y, xs):
#         # |d(obs)/dx|^derivative_power
#         # xs: wavelengths, y: observed flux (same shape)
#         grad = jnp.abs(jnp.gradient(y, xs))
#         return grad ** derivative_power

#     # four branches:
#     if weighted:
#         def loss_fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             rw = jnp.abs(r) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             dw = feature_weight(y, xs)

#             w = rw * uw * (1.0 + dw)   # add 1 so flat regions still count

#             data = jnp.nansum(huber(r) * w) / (jnp.nansum(w) + 1e-8)
#             return data + make_penalty(xs, params)

#     else:
#         def loss_fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y

#             rw = jnp.abs(r) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             dw = feature_weight(y, xs)

#             w = rw * uw * (1.0 + dw)

#             data = jnp.nansum(huber(r) * w) / (jnp.nansum(w) + 1e-8)
#             return data + make_penalty(xs, params)

#     return loss_fn


# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function.

#     Args:
#         func: The model function, called as func(xs, params)
#         param_converter: Optional Parameters() object to transform raw → phys

#     Returns:
#         A loss function with signature (params, xs, y, yerr) -> scalar loss
#     """
#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(log_cosh(r))
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             return jnp.nanmean(log_cosh(y_pred - y))
#         return unweighted_loss
