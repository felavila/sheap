from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import vmap
from jax import random

#This requires major updates 

def residuals(
    func: Callable,
    params: jnp.ndarray,
    xs: jnp.ndarray,
    y: jnp.ndarray,
    y_uncertainties: jnp.ndarray,
) -> jnp.ndarray:
    predictions = func(xs, params)
    return (y - predictions) / y_uncertainties

def apply_arithmetic_ties(samples, ties):
    #this is a general function that have to be move soon
    #_, target, source, op, operand = dep
    tag,target_idx,src_idx, op, val = ties
    src = samples[src_idx]
    #print(src,src_idx)
    if op == '+':
            result = src + val
    elif op == '-':
        result = src - val
    elif op == '*':
        result = src * val
    elif op == '/':
        result = src / val
    else:
        raise ValueError(f"Unsupported operation: {op}")
    #print(op,val,result)
        #params[f"theta_{target_idx}"] = result
    return result


def apply_tied_and_fixed_params(free_params,template_params,dependencies):
    #this can be call just one time 
    idx_target = [i[1] for i in dependencies]
    #idx_source = [i[2] for i in dependencies]
    idx_free_params = list(set(range(len(template_params))) - set(idx_target))
    #free_params = params[jnp.array(idx_free_params)]
    #params_ = jnp.zeros_like(template_params)
    template_params = template_params.at[jnp.array(idx_free_params)].set(free_params)
    template_params = template_params.at[jnp.array(idx_target)].set([apply_arithmetic_ties(template_params,ties) for ties in dependencies])
    return template_params


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
    """

    mask = yerr_i < overboost_threshold
    if jnp.sum(mask) == 0:
        fallback = jnp.abs(params_i) * 5.0 + 1.0
        return (fallback, jnp.diag(fallback**2)) if return_full else fallback

    #xs_valid, y_valid, yerr_valid = xs_i[mask], y_i[mask], yerr_i[mask]
    residual = residual_fn(params_i)[mask]

    if jnp.any(jnp.isnan(residual)) or jnp.any(jnp.isinf(residual)):
        fallback = jnp.abs(params_i) * 5.0 + 1.0
        return (fallback, jnp.diag(fallback**2)) if return_full else fallback

    jacobian = jax.jacobian(residual_fn)(params_i)
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

#def postfit_rutine(model,spectra,params,max_flux,z,dependencies,N = 2_000):
# scaled = max_flux  # / (10**exp_factor)
#             idxs = mapping_params(
#                 self.params_dict, [["amplitude"], ["scale"]]
#             )  # check later on cont how it works
#             self.params = params.at[:, idxs].multiply(scaled[:, None])
#             self.uncertainty_params = uncertainty_params.at[:, idxs].multiply(scaled[:, None])
#             self.spec = norm_spec.at[:, [1, 2], :].multiply(
#                 jnp.moveaxis(jnp.tile(scaled, (2, 1)), 0, 1)[:, :, None]
#             )




def error_for_loop(model,spectra,params,dependencies,N = 2_000):
    "save the samples could increase the number of stuff."
    #print("the sampling will be N = ",N)
    wl, flux, yerr = jnp.moveaxis(spectra, 0, 1)
    idx_target = [i[1] for i in dependencies]
    idx_free_params = list(set(range(len(params[0])))-set(idx_target))
    std = jnp.zeros_like(params)
    #key = random.PRNGKey(0) # i should look for this?
    def apply_one_sample(free_sample):
        return apply_tied_and_fixed_params(free_sample,params[0],dependencies)
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
        #L = jnp.linalg.cholesky(cov_matrix + 1e-6 * jnp.eye(cov_matrix.shape[0]))
        #z = random.normal(key, shape=(N, len(free_params)))
        #samples_free = free_params + z @ L.T  # (N, n_free)
        #full_samples = jax.vmap(apply_one_sample)(samples_free)
        #here cames all the functions from posterior process 
        #means = jnp.mean(full_samples, axis=0)  # shape: (110,)
        #stds  = jnp.std(full_samples, axis=0)   # shape: (110,)
        std = std.at[n].set(apply_tied_and_fixed_params(std_errs,params[0],dependencies))
    return std
        