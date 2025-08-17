"""This module handles ?."""
__author__ = 'felavila'

__all__ = [
    "Errorfromloop",
    "error_covariance_matrix",
    "error_for_loop_s",
    "make_residuals_free_fn",
    "residuals",
]

from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import random,vmap

from sheap.Assistants.parser_mapper import apply_tied_and_fixed_params

#TODO This requires major updates 

def residuals(
    func: Callable,
    params: jnp.ndarray,
    xs: jnp.ndarray,
    y: jnp.ndarray,
    y_uncertainties: jnp.ndarray,
) -> jnp.ndarray:
    predictions = func(xs, params)
    return (y - predictions) / y_uncertainties

def make_residuals_free_fn(
    model_func: Callable,
    xs: jnp.ndarray,
    y: jnp.ndarray,
    yerr: jnp.ndarray,
    template_params: jnp.ndarray,
    dependencies
) -> Callable:
    def residual_fn(free_params: jnp.ndarray) -> jnp.ndarray:
        full_params = apply_tied_and_fixed_params(free_params,template_params,dependencies)
        return residuals(model_func, full_params, xs, y, yerr)
    return residual_fn

def error_covariance_matrix(
    residual_fn: Callable,
    params_i: jnp.ndarray,
    xs_i: jnp.ndarray,
    y_i: jnp.ndarray,
    yerr_i: jnp.ndarray,
    free_params: int,
    return_full: bool = False,
    regularization: float = 1e-6,
    overboost_threshold: float = 1e10,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Estimate uncertainty for free parameters using JTJ approximation.
    TODO: CHECK IF THIS CAN BE UPGRADED 
    """

    mask = yerr_i < overboost_threshold
    if jnp.sum(mask) == 0:
        fallback = jnp.abs(params_i) * 5.0 + 1.0
        return (fallback, jnp.diag(fallback**2)) if return_full else fallback

    #xs_valid, y_valid, yerr_valid = xs_i[mask], y_i[mask], yerr_i[mask]
    residual = residual_fn(params_i)[mask]#.astype(jnp.float32)

    if jnp.any(jnp.isnan(residual)) or jnp.any(jnp.isinf(residual)):
        fallback = jnp.abs(params_i) * 5.0 + 1.0
        return (fallback, jnp.diag(fallback**2)) if return_full else fallback

    jacobian = jax.jacobian(residual_fn)(params_i)#.astype(jnp.float32)
    JTJ = jacobian.T @ jacobian
    dof = max(residual.size - free_params, 1) #to avoid fall back in negatives values 
    s_sq = jnp.sum(residual**2) / dof
    reg = regularization * jnp.eye(JTJ.shape[0])

    try:
        cov = jnp.linalg.inv(JTJ + reg) * s_sq
    except:
        cov = jnp.linalg.pinv(JTJ) * s_sq

    diag_cov = jnp.clip(jnp.diag(cov), a_min=1e-20)
    std_error = jnp.sqrt(diag_cov)

    return (std_error, cov) if return_full else std_error



def Errorfromloop(model, spectra, params, dependencies):
    spectra = jnp.asarray(spectra, dtype=jnp.float64)
    params   = jnp.asarray(params,   dtype=jnp.float64)

    # unpack: spectra has shape (batch, 3, pixels) after moveaxis
    wl, flux, yerr = jnp.moveaxis(spectra, 0, 1)

    # identify which params are free vs tied
    idx_target      = [i[1] for i in dependencies]
    idx_free_params = list(set(range(params.shape[-1])) - set(idx_target))

    # 2) accumulator in float32
    std = jnp.zeros_like(params)

    # 3) loop over each object
    for n, (p_i, wl_i, fl_i, err_i) in enumerate(zip(params, wl, flux, yerr)):
        # re-cast each slice for safety
        #p_i   = p_i.astype(jnp.float32)
        #wl_i  = wl_i.astype(jnp.float32)
        #fl_i  = fl_i.astype(jnp.float32)
        #err_i = err_i.astype(jnp.float32)

        # pick out the free params (already float32)
        free_p = p_i[jnp.array(idx_free_params)]

        # make your residual-fn; assume it handles float32 okay
        res_fn = make_residuals_free_fn(
            model_func      = model,
            xs              = wl_i,
            y               = fl_i,
            yerr            = err_i,
            template_params = p_i,
            dependencies    = dependencies
        )

        # compute covariance in float32
        std_errs, _ = error_covariance_matrix(
            residual_fn  = res_fn,
            params_i     = free_p,
            xs_i         = wl_i,
            y_i          = fl_i,
            yerr_i       = err_i,
            free_params  = free_p.shape[0],
            return_full  = True
        )

        # apply your ties/fixes and store back into the float32 array
        tied_full = apply_tied_and_fixed_params(std_errs, params[0], dependencies)
        std       = std.at[n].set(tied_full)

    return std


def error_for_loop_s(model,spectra,params,dependencies):
    "save the samples could increase the number of stuff."
    wl, flux, yerr = jnp.moveaxis(spectra, 0, 1)
    idx_target = [i[1] for i in dependencies]
    idx_free_params = list(set(range(len(params[0])))-set(idx_target))
    std = jnp.zeros_like(params).astype(jnp.float32)
    for n, (params_i, wl_i, flux_i, yerr_i) in enumerate(zip(params, wl, flux, yerr)):
        free_params = params_i[jnp.array(idx_free_params)]
        res_fn = make_residuals_free_fn(model_func=model,
                                        xs=wl_i,y=flux_i,
                                        yerr=yerr_i,
                                        template_params=params_i,
                                        dependencies=dependencies)
        
        std_errs, _ = error_covariance_matrix(residual_fn=res_fn,
                                                        params_i=free_params,
                                                        xs_i=wl_i,
                                                        y_i=flux_i,
                                                        yerr_i=yerr_i,
                                                        free_params=len(free_params),
                                                        return_full=True)
        std = std.at[n].set(apply_tied_and_fixed_params(std_errs,params[0],dependencies))
    return std

#@jax.jit
