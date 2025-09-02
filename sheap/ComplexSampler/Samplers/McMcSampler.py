"""
MCMC Sampler (NumPyro)
======================

This module provides the :class:`McMcSampler`, a wrapper around
`numpyro.infer.MCMC` + NUTS for sampling posterior distributions of
spectral fit parameters.

Main Features
-------------
- Interfaces directly with a :class:`ComplexAfterFit` estimator
  (after a fit has been run).
- Prepares normalized spectra, constraints, and parameter dictionaries
  for NumPyro.
- Builds a model function via :func:`make_numpyro_model`.
- Runs Hamiltonian Monte Carlo (No-U-Turn Sampler).
- Reconstructs full parameter vectors from sampled free parameters,
  applying tied and fixed constraints.
- Rescales amplitude/log-amplitude parameters back into original units.
- Wraps posterior samples into physical quantities using
  :class:`ComplexParams`.

Public API
----------
- :class:`McMcSampler`:
    * :meth:`McMcSampler.sample_params` — run the sampler for one or more
      spectra, returning posterior parameter dictionaries.

Notes
-----
- Dependencies (ties/fixes) are enforced via
  :func:`sheap.Assistants.parser_mapper.apply_tied_and_fixed_params`.
- By default, each parameter is renamed to ``theta_N`` for NumPyro’s
  sampler to avoid issues with long names.
- Internally uses JAX PRNG keys; ``n_random`` and ``key_seed`` can be
  used to control reproducibility.
"""

__author__ = 'felavila'

__all__ = [
    "McMcSampler",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax 
from jax import grad, vmap,jit, random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
#

from sheap.Assistants.parser_mapper import descale_amp,scale_amp
from sheap.ComplexParams.ComplexParams import ComplexParams
from sheap.ComplexSampler.Samplers.Utils.numpyro_utils import make_numpyro_model



class McMcSampler:
    def __init__(self, estimator: "ComplexSampler"):
        
        self.estimator = estimator  
        self.complexparams = ComplexParams(estimator)
        self.model = estimator.model
        self.dependencies = estimator.dependencies
        self.scale = estimator.scale
        self.spectra = estimator.spectra
        self.mask = estimator.mask
        self.params = estimator.params
        self.params_dict = estimator.params_dict
        self.names = estimator.names 
        self.complex_class = estimator.complex_class
        self.constraints = estimator.constraints 
        
    def sample_params(self, num_samples: int = 2000, num_warmup:int = 500
                      ,summarize=True,n_random=1_000,
                      list_of_objects=None
                      ,key_seed: int = 0):
        from sheap.Assistants.parser_mapper import apply_tied_and_fixed_params
        
        scale = self.scale
        model = self.model
        names = self.names
        constraints = self.constraints
        dependencies = self.dependencies 
        norm_spectra = self.spectra.at[:, [1, 2], :].divide(jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None])
        norm_spectra = norm_spectra.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spectra[:, 2, :]))
        norm_spectra = norm_spectra.astype(jnp.float64)
        wl, flux, yerr = jnp.moveaxis(norm_spectra, 0, 1)
        params = descale_amp(self.params_dict,self.params,scale[:, None])
        constraints = [tuple(x) for x in jnp.asarray(constraints)] #constrains are ok they are still in space 0-2.
        theta_to_sheap = {f"theta_{i}":str(key) for i,key in enumerate(self.params_dict.keys())} #dictionary that creates "theta_n" params easier to work with them in numpyro.
        name_list =  list(theta_to_sheap.keys())
        fixed_params = {}
        if not list_of_objects:
            import numpy as np 
            print("The mcmc will run for all the objects")
            list_of_objects = np.arange(norm_spectra.shape[0])
        dic_posterior_params = {}
        if len(dependencies) == 0:
            print('No dependencies')
            dependencies = None
        #iterator =tqdm(zip(names,params, wl, flux, yerr,self.mask), total=len(params), desc="Sampling obj")
        for n, (name_i,params_i, wl_i, flux_i, yerr_i,mask_i) in enumerate(zip(names,params, wl, flux, yerr,self.mask)):
            print(f"Runing MCMC object {name_i}")
            if n not in list_of_objects:
                continue
            numpyro_model,init_value = make_numpyro_model(name_list,wl_i,flux_i,yerr_i,constraints,params_i,theta_to_sheap,fixed_params,dependencies,model)
            init_strategy = init_to_value(values=init_value)
            kernel = NUTS(numpyro_model, init_strategy=init_strategy)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
            mcmc.run(random.PRNGKey(n_random))
            get_samples = mcmc.get_samples()
            sorted_theta = sorted(get_samples.keys(), key=lambda x: int(x.split('_')[1]))  #How much info can be lost in this steep?
            samples_free = jnp.array([get_samples[i] for i in sorted_theta]).T             #collect_fields=("log_likelihood",)
            def apply_one_sample(free_sample):
                return apply_tied_and_fixed_params(free_sample, params_i, dependencies)
            full_samples = vmap(apply_one_sample)(samples_free)
            full_samples = scale_amp(self.params_dict,full_samples,self.scale[n])
            dic_posterior_params[name_i] = self.complexparams.extract_params(full_samples,n,summarize=summarize)
            dic_posterior_params[name_i].update({"full_samples":full_samples})
            #iterator.close()
        return dic_posterior_params
       
