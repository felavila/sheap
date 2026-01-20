"""
Monte Carlo Sampler
===================

This module implements the :class:`MonteCarloSampler`, a simple
posterior approximation for spectral fits based on randomized parameter
initialization and local re-optimization.

Main Features
-------------
- Generates random draws of parameter vectors within their constraints.
- Converts parameters to raw space and re-optimizes them with
  :class:`Minimizer`.
- Handles tied/fixed parameters through :func:`build_Parameters` and
  dependency flattening utilities.
- Reconstructs physical parameters from optimized raw vectors.
- Computes physical quantities (fluxes, FWHM, luminosities, etc.)
  for each draw using :class:`ComplexParams`.

Public API
----------
- :class:`MonteCarloSampler`
    * :meth:`MonteCarloSampler.sample_params` —
      run the Monte Carlo sampler and return posterior dictionaries.
    * :meth:`MonteCarloSampler.make_minimizer` —
      construct a :class:`Minimizer` configured with penalties/weights.
    * :meth:`MonteCarloSampler._build_tied` —
      convert tied-parameter specifications into dependency strings.

Notes
-----
- This method approximates the posterior distribution by repeatedly
  optimizing from random starts (sometimes called a “poor man’s MCMC”).
- Actual uncertainty propagation is performed by analyzing the
  distribution of optimized solutions.
- Dependencies are flattened so that all tied parameters ultimately
  reference free parameters only.
"""

__author__ = 'felavila'

__all__ = ["MonteCarloSampler",]

from typing import Tuple, Dict, List

import jax.numpy as jnp
from jax import jit , random
import jax.numpy as jnp

import numpy as np 
import time


from sheap.Assistants.parser_mapper import descale_amp,scale_amp,make_get_param_coord_value,build_tied,parse_dependencies,flatten_tied_map
from sheap.ComplexParams.ComplexParams import ComplexParams
from sheap.Assistants.Parameters import build_Parameters
from sheap.Minimizer.Minimizer import Minimizer
from sheap.ComplexSampler.Samplers.Utils.montecarlo_utils import phys_trust_region_inits,resample_spec_all 


class MonteCarloSampler:
	"""
	Montecarlo sampler 
	still under developmen.
	"""
    
	def __init__(self, estimator: "ComplexSampler"):
		self.estimator = estimator  # ParameterEstimation instance
		self.complexparams = ComplexParams(samplerclass=estimator)
		self.names = estimator.names 
		self.model = jit(estimator.model)
		#####norm_spectra####
		self.scale = estimator.scale
		self.spectra = estimator.spectra
		self.mask = estimator.mask
		self.norm_spectra = self._normalize_spectra()
		########
		self.params = estimator.params # are this in the normal scale
		########
		self.dependencies = estimator.dependencies
		self.params_dict = estimator.params_dict
		
		self.complex_class = estimator.complex_class
		self.fitkwargs = estimator.fitkwargs
		self.initial_params  = estimator.initial_params
		self.get_param_coord_value = make_get_param_coord_value(self.params_dict, self.initial_params)  # important
		self.tied_params = self.fitkwargs[-1]["tied"] #the tied params of the last iteration.
		self.constraints = jnp.asarray(estimator.constraints, dtype=jnp.float32) #this will be moved
		self.params_class = self._build_params_class()
		self.best_params = descale_amp(self.params_dict,self.params,self.scale).astype(jnp.float32) #thescaled

	
	def sample_params(self, num_samples: int = 100, key_seed: int = 0, summarize=True,**kwargs) -> jnp.ndarray:
		
		from tqdm import tqdm
		print(f"Running Monte Carlo with JAX.,sample over the spectra")
		norm_spectra = self.norm_spectra
		model = self.model 
		
		_minimizer = self.make_minimizer(model=model, **self.fitkwargs[-1])
		iterator = tqdm(range(num_samples), total=num_samples, desc="Sampling obj")
		key = random.PRNGKey(key_seed)
		monte_params = []
		for n in iterator:
			key, ki = random.split(key)
			norm_spectra_local = resample_spec_all(ki,norm_spectra)
			t0 = time.perf_counter()
			params_m, _ = _minimizer(self.best_params, *norm_spectra_local, self.constraints)
			t1 = time.perf_counter()
			monte_params.append(params_m)
			iterator.set_postfix({"it_s": f"{(t1 - t0):.4f}"})

		_monte_params = np.moveaxis(np.stack(monte_params),0,1)

  
		dic_posterior_params = {}
  
		iterator = tqdm(self.names, total=len(self.names), desc="Getting posterior-params")
		for n, name_i in enumerate(iterator):
			full_samples = scale_amp(self.params_dict,_monte_params[n],self.scale[n])
			dic_posterior_params[name_i] = {"samples_phys":full_samples}
			dic_posterior_params[name_i] = self.complexparams.extract_params(full_samples,n,summarize=summarize)
			dic_posterior_params[name_i].update({"samples_phys":full_samples})

		return dic_posterior_params

 
	def sample_params_experimental(self, num_samples: int = 100, key_seed: int = 0, summarize=True,return_only_draws=False,frac_box_sigma=0.5, k_sigma= 0.5 ) -> jnp.ndarray:
			#it looks like this only work for frac_box_sigma=0.02,k_sigma=0.3 limits 
			from tqdm import tqdm
			print(f"Running Monte Carlo with JAX.,frac_box_sigma={frac_box_sigma},k_sigma={k_sigma}")
			model = self.model 
			norm_spectra = self.norm_spectra
			best_params = self.best_params 

			_, draws_phys = phys_trust_region_inits(key_seed, params_class=self.params_class, best_params=best_params, phys_bounds=self.constraints, num_samples=num_samples, frac_box_sigma= frac_box_sigma,k_sigma=k_sigma )
			
			draws_phys = draws_phys.astype(jnp.float32)  # ensure consistent dtype
			if return_only_draws:
				iterator = tqdm(self.names, total=len(self.names), desc="Getting draws")
				_draws_phys = np.moveaxis(draws_phys,0,1)
				dic_posterior_params = {}
				for n, name_i in enumerate(iterator):
					draws_phys_n = scale_amp(self.params_dict,np.array(_draws_phys[n]),self.scale[n])
					dic_posterior_params[name_i]=({"draws_phys":draws_phys_n})
				return dic_posterior_params
   			
			_minimizer = self.make_minimizer(model=model, **self.fitkwargs[-1])


			iterator = tqdm(range(num_samples), total=num_samples, desc="Sampling obj")

			monte_params = []
			for n in iterator:
				draws_phys_local = draws_phys[n]  # already float32
				t0 = time.perf_counter()
				params_m, _ = _minimizer(draws_phys_local, *norm_spectra, self.constraints)
				t1 = time.perf_counter()

				monte_params.append(params_m)
				iterator.set_postfix({"it_s": f"{(t1 - t0):.4f}"})
	
			_monte_params = np.moveaxis(np.stack(monte_params),0,1)
			_draws_phys = np.moveaxis(draws_phys,0,1)
			dic_posterior_params = {}

			iterator = tqdm(self.names, total=len(self.names), desc="Getting posterior-params")
			
			for n, name_i in enumerate(iterator):
				full_samples = scale_amp(self.params_dict,_monte_params[n],self.scale[n])
				draws_phys_n = scale_amp(self.params_dict,np.array(_draws_phys[n]),self.scale[n])
				dic_posterior_params[name_i] = self.complexparams.extract_params(full_samples,n,summarize=summarize)
				dic_posterior_params[name_i].update({"samples_phys":full_samples,"draws_phys":draws_phys_n})

			return dic_posterior_params


	def make_minimizer(self,model,non_optimize_in_axis,num_steps,learning_rate,
					method,penalty_weight,curvature_weight,smoothness_weight,max_weight,penalty_function=None,weighted=True,**kwargs):
		minimizer = Minimizer(model,non_optimize_in_axis=non_optimize_in_axis,num_steps=num_steps,weighted=weighted,
							learning_rate=learning_rate,param_converter= self.params_class,penalty_function = penalty_function,method=method,
							penalty_weight= penalty_weight,curvature_weight= curvature_weight,smoothness_weight= smoothness_weight,max_weight= max_weight)
		
		
		return minimizer  
	def _normalize_spectra(self):
		"from the clasical shape to the one that is use during the fitting"
		scale = jnp.atleast_1d(self.scale.astype(jnp.float32))
		spectra = self.spectra.astype(jnp.float32)
		norm_spectra = spectra.at[:, [1, 2], :].divide(jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None])
		norm_spectra = norm_spectra.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spectra[:, 2, :]))
		return norm_spectra.astype(jnp.float32).transpose(1, 0, 2)

 
	def _build_params_class(self):
		dependencies = build_tied(self.tied_params,self.get_param_coord_value)
		list_dependencies = parse_dependencies(dependencies)
		tied_map = {T[1]: T[2:] for  T in list_dependencies}
		tied_map = flatten_tied_map(tied_map)
		params_class = build_Parameters(tied_map,self.params_dict,self.initial_params,self.constraints)
		return params_class
  
 		#list_dependencies = self.dependencies
        #tied_map = {T[1]: T[2:] for  T in list_dependencies}
        #tied_map = flatten_tied_map(tied_map)
        #self.tied_map = tied_map
        #self.params_obj = build_Parameters(tied_map,self.params_dict,initial_params,self.constraints) #this one should came from fitting or the clase itself.
        
    	# def _build_tied(self, tied_params):
	# 	"""
	# 	Convert tied‑parameter specifications into dependency strings.

	# 	Parameters
	# 	----------
	# 	tied_params : list of list
	# 		Each inner list is `[param_target, param_source, ..., optional_value]`.

	# 	Returns
	# 	-------
	# 	list[str]
	# 		Dependency expressions for the minimizer.
	# 	"""
	# 	return build_tied(tied_params,self.get_param_coord_value)
