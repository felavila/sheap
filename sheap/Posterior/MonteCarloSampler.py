from typing import Tuple, Dict, List

import jax.numpy as jnp
from jax import vmap, random
import numpy as np 
from tqdm import tqdm

# from .functions import calc_flux,calc_luminosity,calc_fwhm_kms,calc_monochromatic_luminosity,calc_bolometric_luminosity,calc_black_hole_mass
# from .constants import BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS
# from sheap.Mappers.LineMapper import LineMapper 
from sheap.Mappers.helpers import mapping_params
from .parameter_from_sampler import full_params_sampled_to_posterior_params
from .posterior_v2 import posterior_physical_parameters

# this have to be move outside 


#we have to add this. vmap and sum.

# from scipy.stats import skew, kurtosis

# sk = skew(emission_profiles, axis=1)
# print("sk",sk)
# kt = kurtosis(emission_profiles, axis=1, fisher=False)  
# print("kt",kt)
def safe_cholesky(
    cov: jnp.ndarray,
    initial_jitter: float = 1e-6,
    max_tries: int = 5
) -> jnp.ndarray:
    """
    Robust Cholesky decomposition: adds increasing diagonal jitter until SPD.
    """
    # ensure jitter matches matrix dtype
    dtype = cov.dtype
    I = jnp.eye(cov.shape[-1], dtype=dtype)
    jitter = jnp.array(initial_jitter, dtype=dtype)
    for _ in range(max_tries):
        try:
            return jnp.linalg.cholesky(cov + jitter * I)
        except Exception:
            jitter = jitter * 10
    raise RuntimeError(f"Cholesky failed after adding up to {float(jitter)} jitter.")



class MonteCarloSampler:
    """
    Monte Carlo sampler for spectral fit results and parameter uncertainties.
    BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS should came from ParameterEstimation
    """
    def __init__(self, estimator: "ParameterEstimation"):
        self.estimator = estimator  # ParameterEstimation instance
        self.model = estimator.model
        self.c = estimator.c
        self.dependencies = estimator.dependencies
        self.scale = estimator.scale
        self.fluxnorm = estimator.fluxnorm
        self.spec = estimator.spec
        self.mask = estimator.mask
        self.d = estimator.d
        self.params = estimator.params
        self.params_dict = estimator.params_dict
        self.BOL_CORRECTIONS = estimator.BOL_CORRECTIONS
        self.SINGLE_EPOCH_ESTIMATORS = estimator.SINGLE_EPOCH_ESTIMATORS
        self.names = estimator.names 
        self.ComplexRegion_class = estimator.ComplexRegion_class
    
    def sample_params(self, N: int = 2000, key_seed: int = 0,summarize=True,get_full_posterior=True) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        from sheap.RegionFitting.uncertainty_functions import (
            apply_tied_and_fixed_params, make_residuals_free_fn, error_covariance_matrix
        )
        scale = self.scale
        norm_spec = self.spec.at[:, [1, 2], :].divide(
            jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None]
        )
        norm_spec = norm_spec.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spec[:, 2, :]))
        norm_spec = norm_spec.astype(jnp.float64)
        idxs = mapping_params(self.params_dict, [["amplitude"], ["scale"]])
        params = self.params.at[:, idxs].divide(scale[:, None]).astype(jnp.float64)
        names = self.names 
        wl, flux, yerr = jnp.moveaxis(norm_spec, 0, 1)
        model = self.model
        dependencies = self.dependencies
        idx_target = [i[1] for i in dependencies]
        idx_free_params = list(set(range(len(params[0]))) - set(idx_target))
        key = random.PRNGKey(key_seed)
        dic_posterior_params = {}
        matrix_sample_params = jnp.zeros((norm_spec.shape[0],N,params.shape[1])) 
        if len(dependencies) == 0:
            print('No dependencies')
            dependencies = None 
        iterator =tqdm(zip(names,params, wl, flux, yerr,self.mask), total=len(params), desc="Sampling obj")
        for n, (name_i,params_i, wl_i, flux_i, yerr_i,mask_i) in enumerate(iterator):
            free_params = params_i[jnp.array(idx_free_params)]                 
            res_fn = make_residuals_free_fn(
                model_func=model, xs=wl_i, y=flux_i, yerr=yerr_i,
                template_params=params_i, dependencies=dependencies
            )
            
            
            _, cov_matrix = error_covariance_matrix(
                residual_fn=res_fn,
                params_i=free_params,
                xs_i=wl_i,
                y_i=flux_i,
                yerr_i=yerr_i,
                free_params=len(free_params),
                return_full=True
            )
            
            L = jnp.linalg.cholesky(cov_matrix + 1e-6 * jnp.eye(cov_matrix.shape[0]))
            z = random.normal(key, shape=(N, len(free_params)))
            samples_free = free_params + z @ L.T  # (N, n_free)

            def apply_one_sample(free_sample):
                return apply_tied_and_fixed_params(free_sample, params_i, dependencies)
        
            full_samples = vmap(apply_one_sample)(samples_free)
            full_samples = full_samples.at[:, idxs].multiply(scale[n])
            if get_full_posterior:
                dic_posterior_params[name_i] = posterior_physical_parameters(wl_i, flux_i, yerr_i,mask_i,full_samples,self.ComplexRegion_class
                                                                                ,np.full((N,), self.d[n],dtype=np.float64),
                                                                                c=self.c,
                                                                                BOL_CORRECTIONS=self.BOL_CORRECTIONS,
                                                                                SINGLE_EPOCH_ESTIMATORS=self.SINGLE_EPOCH_ESTIMATORS,summarize=summarize)
            
            matrix_sample_params = matrix_sample_params.at[n].set(full_samples)
        iterator.close()
        return matrix_sample_params,dic_posterior_params
    
    
    # def posterior_physical_parameters(
    # wl_i: np.ndarray,
    # flux_i: np.ndarray,
    # yerr_i: np.ndarray,
    # mask_i: np.ndarray,
    # full_samples: np.ndarray,
    # region_group: Any,
    # distances: np.ndarray,
    # BOL_CORRECTIONS: Dict[str, float],
    # SINGLE_EPOCH_ESTIMATORS: Dict[str, Dict[str, Any]],
    # c: float,
    # summarize: bool = False,