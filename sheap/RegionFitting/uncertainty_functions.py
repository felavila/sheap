from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap

# from jax import config
# config.update("jax_platform_name", "cpu")


def residuals(
    func: Callable,
    params: jnp.ndarray,
    xs: jnp.ndarray,
    y: jnp.ndarray,
    y_uncertainties: jnp.ndarray,
) -> jnp.ndarray:
    predictions = func(xs, params)
    return jnp.abs(y - predictions) / y_uncertainties


def error_covariance_matrix_single(
    func: Callable,
    params_i: jnp.ndarray,
    xs_i: jnp.ndarray,
    y_i: jnp.ndarray,
    yerr_i: jnp.ndarray,
    free_params: int,
) -> jnp.ndarray:
    residual = residuals(func, params_i, xs_i, y_i, yerr_i)
    jac_fn = lambda p: residuals(func, p, xs_i, y_i, yerr_i)
    jacobian = jax.jacobian(jac_fn)(params_i)
    JTJ = jacobian.T @ jacobian
    dof = residual.shape[0] - free_params
    s_sq = jnp.sum(residual**2) / dof
    cov = jnp.linalg.inv(JTJ + 1e-6 * jnp.eye(params_i.shape[0])) * s_sq
    return jnp.sqrt(jnp.diag(cov))


def error_for_loop(model, params, spectra, free_params=0):
    x, y, error = jnp.moveaxis(spectra, 0, 1)
    list = jnp.zeros_like(params)
    for n, (params_i, x_i, y_i, error_i) in enumerate(zip(params, x, y, error)):
        sigma = error_covariance_matrix_single(
            model, params_i, x_i, y_i, error_i, free_params=free_params
        )
        list = list.at[n].set(sigma)
        # list[n] = sigma
        # list.append(sigma)
    return list


def batch_error_covariance_in_chunks(
    func: Callable,
    params: jnp.ndarray,
    spectra: jnp.ndarray,
    batch_size: int = 30,
    free_params: int = 0,
) -> jnp.ndarray:
    """
    Estimate parameter uncertainties using Jacobian-based covariance matrix in batches.

    Parameters:
        func: Callable model function (xs, params) -> predictions
        params: (N, P) array of optimized parameters
        spectra: (3, N, X) array of [xs, y, yerr]
        batch_size: Maximum batch size to process in memory
        free_params: Number of free parameters in the model

    Returns:
        (N, P) array of standard deviations per parameter per spectrum
    """
    xs, y, yerr = jnp.moveaxis(spectra, 0, 1)
    n = params.shape[0]

    if n == 1:
        return error_covariance_matrix_single(
            func, params[0], xs[0], y[0], yerr[0], free_params
        )[
            None, :
        ]  # add batch dim

    results = []
    for i in range(0, n, batch_size):
        batch_fn = vmap(
            lambda p, x, y_, ye: error_covariance_matrix_single(
                func, p, x, y_, ye, free_params
            ),
            in_axes=(0, 0, 0, 0),
        )
        batch_res = batch_fn(
            params[i : i + batch_size],
            xs[i : i + batch_size],
            y[i : i + batch_size],
            yerr[i : i + batch_size],
        )
        results.append(batch_res)

    return jnp.concatenate(results, axis=0)




def error_covariance_matrix_singlev2(
    func: Callable,
    params_i: jnp.ndarray,
    xs_i: jnp.ndarray,
    y_i: jnp.ndarray,
    yerr_i: jnp.ndarray,
    full_params: int,
    return_full: bool = False,
    regularization: float = 1e-6,
    overboost_threshold: float = 1e10,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Estimate parameter uncertainties from the inverse of the Jacobian's Gram matrix.

    If residuals contain NaN/Inf, or model is ill-conditioned, returns fallback values.

    Args:
        func: Model function.
        params_i: Best-fit parameters.
        xs_i, y_i, yerr_i: Data and uncertainties.
        full_params: Total number of parameters (before constraint filtering).
        return_full: Whether to return the full covariance matrix.
        regularization: Small diagonal value added for stability.
        overboost_threshold: Any yerr > this value is considered 'masking'.

    Returns:
        std_error: Parameter uncertainties.
        cov_matrix (optional): Full covariance matrix.
    """

    # Mask out overboosted entries
    mask = yerr_i < overboost_threshold
    if mask.sum() == 0:
        # All points masked ⇒ return fallback
        fallback = jnp.abs(params_i) * 5.0 + 1.0
        return (fallback, jnp.diag(fallback**2)) if return_full else fallback

    xs_valid, y_valid, yerr_valid = xs_i[mask], y_i[mask], yerr_i[mask]

    def residual_fn(p):
        return residuals(func, p, xs_valid, y_valid, yerr_valid)

    residual = residual_fn(params_i)

    if jnp.any(jnp.isnan(residual)) or jnp.any(jnp.isinf(residual)):
        fallback = jnp.abs(params_i) * 5.0 + 1.0
        return (fallback, jnp.diag(fallback**2)) if return_full else fallback

    # Build Jacobian
    jacobian = jax.jacobian(residual_fn)(params_i)
    JTJ = jacobian.T @ jacobian

    # Infer number of free parameters from Jacobian rank
    param_mask = jnp.linalg.norm(jacobian, axis=0) > 1e-8
    free_params = int(param_mask.sum())

    dof = max(residual.size - free_params, 1)
    s_sq = jnp.sum(residual ** 2) / dof
    reg = regularization * jnp.eye(JTJ.shape[0])

    try:
        cov = jnp.linalg.inv(JTJ + reg) * s_sq
    except (jax.errors.ConcretizationTypeError, RuntimeError, ValueError):
        cov = jnp.linalg.pinv(JTJ) * s_sq

    diag_cov = jnp.clip(jnp.diag(cov), a_min=1e-20)
    std_error = jnp.sqrt(diag_cov)

    return (std_error, cov) if return_full else std_error
