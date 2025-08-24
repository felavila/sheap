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