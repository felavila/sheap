import jax 
from jax import jit,vmap
import jax.numpy as jnp
from typing import Callable
from jax import value_and_grad

def residuals(func,params: jnp.ndarray, xs, y: jnp.ndarray, y_uncertainties: jnp.ndarray):
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
    
    # Jacobian w.r.t. params
    jac_fn = lambda p: residuals(func, p, xs_i, y_i, yerr_i)
    jacobian = jax.jacobian(jac_fn)(params_i)  # shape (n_data, n_params)
    
    JTJ = jacobian.T @ jacobian
    chi_square = jnp.sum(residual ** 2)
    dof = residual.shape[0] - free_params
    s_sq = chi_square / dof

    # Add small diagonal term to avoid singular matrix
    cov = jnp.linalg.inv(JTJ + 1e-6 * jnp.eye(params_i.shape[0])) * s_sq
    return jnp.sqrt(jnp.diag(cov))  # shape: (n_params,)

def batch_error_covariance_in_chunks(func,params,spectra, batch_size=30):
    xs, y, yerr = jnp.moveaxis(spectra,0,1)
    n = params.shape[0]
    results = []
    for i in range(0, n, batch_size):
        batch_fn = vmap(
            lambda p, x, y_, ye: error_covariance_matrix_single(func, p, x, y_, ye, 0),
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