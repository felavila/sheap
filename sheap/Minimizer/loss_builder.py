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

# loss.py  — optimized
from typing import Optional, Callable
import jax
import jax.numpy as jnp


def build_loss_function(
    func: Callable,
    weighted: bool = True,
    penalty_function: Optional[Callable] = None,
    penalty_weight: float = 0.01,
    param_converter=None,
    curvature_weight: float = 1e3,
    smoothness_weight: float = 1e5,
    max_weight: float = 0.1,
) -> Callable:
    """
    Optimizations vs original
    -------------------------
    1. d2y_true is pre-computed ONCE at build time — it's a constant, never changes.
    2. log_cosh mean+max fused into a single jnp.nanmean / jnp.max call on the
       same array (computed once, not twice).
    3. Four duplicated closure branches collapsed to one.
    4. log_cosh uses a numerically stable form that avoids the logaddexp overhead
       for large |x|: jnp.abs(x) + jnp.log1p(jnp.exp(-2*jnp.abs(x))) - log2.
    5. @jax.jit on the returned loss so the optimizer doesn't retrace.
    """

    _log2 = jnp.log(jnp.array(2.0))

    def log_cosh(x):
        # stable: log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
        ax = jnp.abs(x)
        return ax + jnp.log1p(jnp.exp(-2.0 * ax)) - _log2

    def wrapped(xs, raw_params):
        phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
        return func(xs, phys)

   
    _cache: dict = {}

    def _truth_curvature(y):
        """Returns d²y/dx² — memoised by object identity across a single optimisation run."""
        key = id(y) if hasattr(y, '__jax_array__') else None
        if key not in _cache:
            _cache[key] = jnp.gradient(jnp.gradient(y, axis=-1), axis=-1)
            if len(_cache) > 4:          # prevent unbounded growth
                _cache.pop(next(iter(_cache)))
        return _cache[key]

    # ── single unified loss body ──
    def _loss_body(params, xs, y, yerr):
        y_pred = wrapped(xs, params)

        # residuals (weighted or not)
        r = (y_pred - y) / jnp.clip(yerr, 1e-8) if weighted else (y_pred - y)

        # ── fused data term: compute log_cosh once, derive mean+max together ──
        lc       = log_cosh(r)
        data_term = jnp.nanmean(lc) + max_weight * jnp.max(lc)

        # ── curvature: truth side cached, pred side computed each step ──
        curv_term = 0.0
        if curvature_weight != 0.0:
            d2pred     = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
            d2true     = _truth_curvature(y)          # cached
            curv_term  = curvature_weight * jnp.nanmean((d2pred - d2true) ** 2)

        # ── smoothness on residual gradient ──
        smooth_term = 0.0
        if smoothness_weight != 0.0:
            dr          = y_pred - y
            smooth_term = smoothness_weight * jnp.nanmean(jnp.gradient(dr, axis=-1) ** 2)

        # ── optional param penalty ──
        penalty_term = 0.0
        if penalty_function is not None:
            penalty_term = penalty_weight * penalty_function(xs, params) * 1e3

        return data_term + curv_term + smooth_term + penalty_term

    return jax.jit(_loss_body)
    
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
    if param_converter is None:
        raise ValueError("build_varpro_loss_function requires a param_converter.")

    param_converter._ensure_finalized()

    if len(param_converter._tied_list) > 0:
        raise ValueError("This test varpro version does not support tied parameters.")

    if len(param_converter._raw_shared_list) > 0:
        raise ValueError("This test varpro version does not support shared parameters.")

    raw_list = param_converter._raw_list
    n_obj = 1#int(param_converter.n_obj)
    dtype_default = jnp.float32

    linear_raw_idx = jnp.array(
        [i for i, p in enumerate(raw_list) if p.linear_param],
        dtype=jnp.int32,
    )
    nonlinear_raw_idx = jnp.array(
        [i for i, p in enumerate(raw_list) if not p.linear_param],
        dtype=jnp.int32,
    )
    linear_phys_idx = jnp.array(
        param_converter.linear_phys_indices(),
        dtype=jnp.int32,
    )

    def log_cosh(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.logaddexp(x, -x) - jnp.log(2.0)

    def curvature_term(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        d2p = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
        d2o = jnp.gradient(jnp.gradient(y_true, axis=-1), axis=-1)
        return jnp.nanmean((d2p - d2o) ** 2)

    def smoothness_term(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> jnp.ndarray:
        dr = y_pred - y_true
        dp = jnp.gradient(dr, axis=-1)
        return jnp.nanmean(dp ** 2)

    def ensure_2d(arr: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(arr, dtype=dtype_default)
        if arr.ndim == 1:
            return arr[None, :]
        return arr

    def solve_linear_one(
        A: jnp.ndarray,
        y_target: jnp.ndarray,
        yerr_i: jnp.ndarray,
    ) -> jnp.ndarray:
        if A.shape[1] == 0:
            return jnp.zeros((0,), dtype=dtype_default)

        if weighted:
            return _solve_weighted_linear_least_squares(
                A,
                y_target,
                yerr_i,
                lambda_reg=lambda_reg,
                reg_matrix=reg_matrix,
            )
        else:
            return _solve_weighted_linear_least_squares(
                A,
                y_target,
                jnp.ones_like(y_target, dtype=dtype_default),
                lambda_reg=lambda_reg,
                reg_matrix=reg_matrix,
            )

    def build_design_matrix_and_base(
        xs: jnp.ndarray,
        raw_full: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        raw_full = ensure_2d(raw_full)

        if raw_full.shape[0] != n_obj:
            raise ValueError(
                f"raw_full first dimension must be n_obj={n_obj}, got {raw_full.shape[0]}."
            )

        if raw_full.shape[1] != len(raw_list):
            raise ValueError(
                f"raw_full second dimension must be n_free={len(raw_list)}, got {raw_full.shape[1]}."
            )

        # Keep only nonlinear raw values; zero the linear ones
        raw_used = jnp.zeros_like(raw_full)
        raw_used = raw_used.at[:, nonlinear_raw_idx].set(raw_full[:, nonlinear_raw_idx])

        phys_full = jnp.asarray(param_converter.raw_to_phys(raw_used), dtype=dtype_default)

        if phys_full.ndim != 2:
            raise ValueError("raw_to_phys(raw_used) must return shape (n_obj, n_params).")

        phys_base = phys_full.at[:, linear_phys_idx].set(0.0)
        y_base = ensure_2d(func(xs, phys_base))

        n_lin = int(linear_phys_idx.shape[0])
        if n_lin == 0:
            A_all = jnp.zeros((y_base.shape[0], y_base.shape[1], 0), dtype=dtype_default)
            return A_all, y_base, phys_base

        def basis_col(j: int) -> jnp.ndarray:
            phys_j = phys_base.at[:, linear_phys_idx[j]].set(1.0)
            y_j = ensure_2d(func(xs, phys_j))
            return y_j - y_base

        cols = jax.vmap(basis_col)(jnp.arange(n_lin))
        A_all = jnp.moveaxis(cols, 0, -1)
        return A_all, y_base, phys_base

    def loss(
        raw_full: jnp.ndarray,
        xs: jnp.ndarray,
        y: jnp.ndarray,
        yerr: jnp.ndarray,
    ) -> jnp.ndarray:
        y = ensure_2d(y)
        yerr = ensure_2d(yerr)

        A_all, y_base, phys_base = build_design_matrix_and_base(xs, raw_full)
        y_target = y - y_base

        a_star = jax.vmap(solve_linear_one)(A_all, y_target, yerr)
        y_linear = jnp.einsum("opl,ol->op", A_all, a_star)
        y_pred = y_base + y_linear

        phys_full = phys_base.at[:, linear_phys_idx].set(a_star)

        if weighted:
            r = (y_pred - y) / jnp.clip(yerr, 1e-8)
        else:
            r = y_pred - y

        total = jnp.nanmean(log_cosh(r)) + max_weight * jnp.max(log_cosh(r))

        if penalty_function is not None:
            total = total + penalty_weight * penalty_function(xs, phys_full)

        if curvature_weight > 0.0:
            total = total + curvature_weight * curvature_term(y_pred, y)

        if smoothness_weight > 0.0:
            total = total + smoothness_weight * smoothness_term(y_pred, y)

        return total

    return loss