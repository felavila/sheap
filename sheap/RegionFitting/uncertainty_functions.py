import jax
import jax.numpy as jnp
from jax import vmap
from typing import Callable
# from jax import config
# config.update("jax_platform_name", "cpu")

def residuals(func: Callable, params: jnp.ndarray, xs: jnp.ndarray, y: jnp.ndarray, y_uncertainties: jnp.ndarray) -> jnp.ndarray:
    predictions = func(xs, params)
    return jnp.abs(y - predictions) / y_uncertainties

def error_covariance_matrix_single(
    func: Callable,
    params_i: jnp.ndarray,
    xs_i: jnp.ndarray,
    y_i: jnp.ndarray,
    yerr_i: jnp.ndarray,
    free_params: int
) -> jnp.ndarray:
    residual = residuals(func, params_i, xs_i, y_i, yerr_i)
    jac_fn = lambda p: residuals(func, p, xs_i, y_i, yerr_i)
    jacobian = jax.jacobian(jac_fn)(params_i)
    JTJ = jacobian.T @ jacobian
    dof = residual.shape[0] - free_params
    s_sq = jnp.sum(residual ** 2) / dof
    cov = jnp.linalg.inv(JTJ + 1e-6 * jnp.eye(params_i.shape[0])) * s_sq
    return jnp.sqrt(jnp.diag(cov))

def error_for_loop(model,params,spectra,free_params=0):
    x,y,error = jnp.moveaxis(spectra,0,1)
    list = jnp.zeros_like(params)
    for n,(params_i,x_i,y_i,error_i) in enumerate(zip(params,x,y,error)):
        sigma = error_covariance_matrix_single(model,params_i,x_i,y_i,error_i,free_params=free_params)
        list = list.at[n].set(sigma)
        #list[n] = sigma
        #list.append(sigma)
    return list


def batch_error_covariance_in_chunks(
    func: Callable,
    params: jnp.ndarray,
    spectra: jnp.ndarray,
    batch_size: int = 30,
    free_params: int = 0
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
        return error_covariance_matrix_single(func, params[0], xs[0], y[0], yerr[0], free_params)[None, :]  # add batch dim

    results = []
    for i in range(0, n, batch_size):
        batch_fn = vmap(
            lambda p, x, y_, ye: error_covariance_matrix_single(func, p, x, y_, ye, free_params),
            in_axes=(0, 0, 0, 0)
        )
        batch_res = batch_fn(
            params[i:i+batch_size],
            xs[i:i+batch_size],
            y[i:i+batch_size],
            yerr[i:i+batch_size]
        )
        results.append(batch_res)

    return jnp.concatenate(results, axis=0)
