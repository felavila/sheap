"""This module handles basic operations."""
__author__ = 'felavila'

__all__ = [
    "MonteCarloSampler",
]

from typing import Tuple, Dict, List

import jax.numpy as jnp
from jax import jit , random
import jax.numpy as jnp

import numpy as np 
import time

from sheap.ComplexFitting.ComplexFitting import ComplexFitting
from sheap.Assistants.parser_mapper import descale_amp,scale_amp,apply_tied_and_fixed_params,make_get_param_coord_value,build_tied,parse_dependencies,flatten_tied_map
from sheap.ComplexAfterFit.AfterFitParams import AfterFitParams
from sheap.Assistants.Parameters import build_Parameters
from sheap.Minimizer.Minimizer import Minimizer

class MonteCarloSampler:
    """
    Approximate posterior sampling via local Gaussian (Laplace) expansion
    BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS should came from ParameterEstimation
    """
    
    def __init__(self, estimator: "ComplexAfterFit"):
        
        self.estimator = estimator  # ParameterEstimation instance
        self.afterfitparams = AfterFitParams(estimator)
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
        self.complex_class = estimator.complex_class
        self.fitkwargs = estimator.fitkwargs
        self.constraints = estimator.constraints
        self.initial_params  = estimator.initial_params
        self.get_param_coord_value = make_get_param_coord_value(self.params_dict, self.initial_params)  # important

    def sample_params(self, num_samples: int = 100, key_seed: int = 0, summarize=True, extra_products=True) -> jnp.ndarray:
        from tqdm import tqdm
        print("Running Monte Carlo with JAX.")
        
        model = jit(self.model)
        names = self.names

        # Normalize spectra
        scale = self.scale.astype(jnp.float32)
        spec = self.spec.astype(jnp.float32)
        norm_spec = spec.at[:, [1, 2], :].divide(jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None])
        norm_spec = norm_spec.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spec[:, 2, :]))
        norm_spec = norm_spec.astype(jnp.float32)

        param_min = jnp.array([c[0] for c in self.constraints], dtype=jnp.float32)
        param_max = jnp.array([c[1] for c in self.constraints], dtype=jnp.float32)

        print("num_samples =", num_samples)

        # JAX random sampling
        key = random.PRNGKey(key_seed)
        samples = random.uniform(
            key,
            shape=(num_samples, param_min.shape[0]),
            minval=param_min,
            maxval=param_max,
            dtype=jnp.float32,
        )

        
        # self.params_obj = Parameters()
        # for name, idx in self.params_dict.items():
        #     val = float(self.initial_params[idx])
        #     min_, max_ = self.constraints[idx]
        #     self.params_obj.add(name, val, min=min_, max=max_)
        
        list_dependencies = self._build_tied(self.fitkwargs[-1]["tied"])
        list_dependencies = parse_dependencies(self._build_tied(self.fitkwargs[-1]["tied"]))
        tied_map = {T[1]: T[2:] for  T in list_dependencies}
        tied_map = flatten_tied_map(tied_map)
        
        self.params_obj = build_Parameters(tied_map,self.params_dict,self.initial_params,self.constraints)
            
        iterator = tqdm(range(num_samples), total=num_samples, desc="Sampling obj")
        
        monte_params = []
        _minimizer = self.make_minimizer(model=model, **self.fitkwargs[-1])
        for n in iterator:    
            p = samples[n]
            p = jnp.tile(p, (norm_spec.shape[0], 1)).astype(jnp.float32)
            raw_init = self.params_obj.phys_to_raw(p)
            #start_time = time.time()
            raw_params, _ = _minimizer(raw_init, *norm_spec.transpose(1, 0, 2), self.constraints)
            #end_time = time.time()
            params_m = self.params_obj.raw_to_phys(raw_params)
            monte_params.append(params_m)
            #elapsed = end_time - start_time
            #print(f"Time elapsed for : {n}-{elapsed:.2f} seconds")
        _monte_params = np.stack(monte_params).reshape(norm_spec.shape[0],num_samples,-1)
        dic_posterior_params = {}
        for n,name_i in enumerate(self.names):
            dic_posterior_params[name_i] = self.afterfitparams.extract_basic_params(_monte_params[n],n)
        return dic_posterior_params
    
        
    def make_minimizer(self,model,non_optimize_in_axis,num_steps,learning_rate,
             method,penalty_weight,curvature_weight,smoothness_weight,max_weight,penalty_function=None,weighted=True):
        
        
        
        minimizer = Minimizer(model,non_optimize_in_axis=non_optimize_in_axis,num_steps=num_steps,weighted=weighted,
                              learning_rate=learning_rate,param_converter= self.params_obj,penalty_function = penalty_function,method=method,
                            penalty_weight= penalty_weight,curvature_weight= curvature_weight,smoothness_weight= smoothness_weight,max_weight= max_weight)
        
        
        #print(raw_params)
        return minimizer
        
        
        
    def _build_tied(self, tied_params):
        """
        Convert tied‑parameter specifications into dependency strings.

        Parameters
        ----------
        tied_params : list of list
            Each inner list is `[param_target, param_source, ..., optional_value]`.

        Returns
        -------
        list[str]
            Dependency expressions for the minimizer.
        """
        return build_tied(tied_params,self.get_param_coord_value)
    
    
    
    # for n,p in enumerate(iterator):
    #         start_time = time.time()  # 
    #         p = jnp.tile(p, (norm_spec.shape[0], 1))
    #         #result.configfittr?
    #         raw_init = self.params_obj.phys_to_raw(p)
    #         raw_params, _ = jit(_minimizer(raw_init, *norm_spec.transpose(1, 0, 2), self.constraints))
    #         params_m = self.params_obj.raw_to_phys(raw_params)
    #         #params_m, _ = self._fit(norm_spec=norm_spec,model = self.model,initial_params=p,**self.fitkwargs[-1])
    #         monte_params.append(params_m)
    #         end_time = time.time()  # 
    #         elapsed = end_time - start_time
    #         print(f"Time elapsed for : {n}-{elapsed:.2f} seconds")
