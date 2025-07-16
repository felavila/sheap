
# from typing import Callable, Tuple, Union

# import jax
# import jax.numpy as jnp
# from jax import vmap
# from jax import random


# from .uncertainty_functions import make_residuals_free_fn,error_covariance_matrix,apply_tied_and_fixed_params
# from sheap.Mappers.LineMapper import mapping_params

# ####?
# def AfterFit(model,spectra,max_flux,params,dependencies,params_dict,N = 2_000):
#     print("the sampling will be N = ",N)
#     idx_target = [i[1] for i in dependencies]
#     idx_free_params = list(set(range(len(params[0])))-set(idx_target))
#     std = jnp.zeros_like(params)
#     key = random.PRNGKey(0) # i should look for this?
    
#     def apply_one_sample(free_sample):
#         return apply_tied_and_fixed_params(free_sample,params[0],dependencies)
#     spectra_result = jnp.zeros_like(spectra)
#     for n, (params_i, spectra_i) in enumerate(zip(params,spectra)):
#         wl_i, flux_i, yerr_i = jnp.moveaxis(spectra, 0, 1)
#         free_params = params_i[jnp.array(idx_free_params)]
#         res_fn = make_residuals_free_fn(model_func=model,
#                                         xs=wl_i,y=flux_i,
#                                         yerr=yerr_i,
#                                         template_params=params_i,
#                                         dependencies=dependencies)
#         std_errs, cov_matrix = error_covariance_matrix(residual_fn=res_fn,
#                                                         params_i=free_params,
#                                                         xs_i=wl_i,
#                                                         y_i=flux_i,
#                                                         yerr_i=yerr_i,
#                                                         free_params=len(free_params),
#                                                         return_full=True)
#         L = jnp.linalg.cholesky(cov_matrix + 1e-6 * jnp.eye(cov_matrix.shape[0]))
#         z = random.normal(key, shape=(N, len(free_params)))
#         samples_free = free_params + z @ L.T  # (N, n_free)
#         full_samples = jax.vmap(apply_one_sample)(samples_free)
#         #here cames all the functions from posterior process 
#         #means = jnp.mean(full_samples, axis=0)  # shape: (110,)
#         #stds  = jnp.std(full_samples, axis=0)   # shape: (110,)
#         #std = std.at[n].set(stds)
#         max_flux_ = max_flux[0]
#         #come back to real units:
#         spectra_i = spectra_i.at[1, 2].multiply(jnp.moveaxis(jnp.tile(max_flux_, (2, 1)), 0, 1)[:,None])
#         spectra_result = spectra_result.at[n].set(spectra_i)
#         idxs = mapping_params(params_dict, [["amplitude"], ["scale"]])  # check later on cont how it works
#         #full_samples = params.at[:, idxs].multiply(max_flux_[:, None])
#         #uncertainty_params = uncertainty_params.at[:, idxs].multiply(max_flux_[:, None])
        
        
    
#     return spectra_result