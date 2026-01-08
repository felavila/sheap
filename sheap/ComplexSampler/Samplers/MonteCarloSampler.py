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

__all__ = [
    "MonteCarloSampler",
]

from typing import Tuple, Dict, List

import jax.numpy as jnp
from jax import jit , random
import jax.numpy as jnp

import numpy as np 
import time


from sheap.Assistants.parser_mapper import descale_amp,scale_amp,apply_tied_and_fixed_params,make_get_param_coord_value,build_tied,parse_dependencies,flatten_tied_map
from sheap.ComplexParams.ComplexParams import ComplexParams
from sheap.Assistants.Parameters import build_Parameters
from sheap.Minimizer.Minimizer import Minimizer


def phys_trust_region_inits(
    key, *,
    params_obj,          # has phys_to_raw / raw_to_phys and knows ties
    phys_map,            # MAP in physical space (vector of *free* params)
    phys_bounds,         # [(lo, hi), ...] in physical space (same shape)
    num_samples=100,
    sigma_phys=None,     # per-parameter std in physical space; if None use frac of box
    frac_box_sigma=0.5, # fallback noise size ~5% of (hi-lo)
    k_sigma= 0.5          # multiplier for sigma_phys
):
    key = random.PRNGKey(key) if isinstance(key, int) else key

    lo = jnp.array([b[0] for b in phys_bounds], dtype=jnp.float32)
    hi = jnp.array([b[1] for b in phys_bounds], dtype=jnp.float32)
    width = hi - lo

    if sigma_phys is None:
        # fallback: a fraction of the box width (handles fixed params when width==0)
        sigma_phys = jnp.where(width > 0, frac_box_sigma * width, 0.0)

    keys = random.split(key, num_samples)
    draws_phys = []
    for ki in keys:
        step = k_sigma * sigma_phys * random.normal(ki, shape=phys_map.shape)
        phys = phys_map + step
        # project back to physical bounds
        phys = jnp.clip(phys, lo, hi)
        draws_phys.append(phys)

    draws_phys = jnp.stack(draws_phys)  # (N, P)
    # map to raw (so your optimizer can work)
    draws_raw = jnp.stack([params_obj.phys_to_raw(p) for p in draws_phys])
    return draws_raw, draws_phys

class MonteCarloSampler:
	"""
	Montecarlo sampler 
	still under developmen.
	"""
    
	def __init__(self, estimator: "ComplexSampler"):
		self.estimator = estimator  # ParameterEstimation instance
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
		self.fitkwargs = estimator.fitkwargs
		self.constraints = estimator.constraints
		self.initial_params  = estimator.initial_params
		self.get_param_coord_value = make_get_param_coord_value(self.params_dict, self.initial_params)  # important
	
	def sample_params(self, num_samples: int = 100, key_seed: int = 0, summarize=True) -> jnp.ndarray:
			from tqdm import tqdm
			print("Running Monte Carlo with JAX.")
			model = jit(self.model)
			# Normalize spectratra
			scale = np.atleast_1d(self.scale.astype(jnp.float32))
			#print(scale.shape)
			spectra = self.spectra.astype(jnp.float32)

			norm_spectra = spectra.at[:, [1, 2], :].divide(jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None])
			norm_spectra = norm_spectra.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spectra[:, 2, :]))
			norm_spectra = norm_spectra.astype(jnp.float32)

			#print(self.params_dict,self.params,self.scale)
			phys_map = descale_amp(self.params_dict,self.params,self.scale)

			# param_min = jnp.array([c[0] for c in self.constraints], dtype=jnp.float32)
			# param_max = jnp.array([c[1] for c in self.constraints], dtype=jnp.float32)

			list_dependencies = self._build_tied(self.fitkwargs[-1]["tied"])
			list_dependencies = parse_dependencies(self._build_tied(self.fitkwargs[-1]["tied"]))
			tied_map = {T[1]: T[2:] for  T in list_dependencies}
			tied_map = flatten_tied_map(tied_map)

			norm_spectra_T = norm_spectra.transpose(1, 0, 2)

			self.params_obj = build_Parameters(tied_map,self.params_dict,self.initial_params,self.constraints)

			draws_raw, draws_phys = phys_trust_region_inits(
			key_seed,
			params_obj=self.params_obj,
			phys_map=phys_map,
			phys_bounds=self.constraints,
			num_samples=num_samples)

			draws_raw = draws_raw.astype(jnp.float32)  # ensure consistent dtype
			#print(self.fitkwargs)
			_minimizer = self.make_minimizer(model=model, **self.fitkwargs[-1])

			constraints_jnp = jnp.asarray(self.constraints, dtype=jnp.float32)

			#raw_init0 = draws_raw[0]
			#raw_params0, _ = _minimizer(raw_init0, *norm_spectra_T, constraints_jnp)
			# Force the computation to finish so compile time happens here:
			#raw_params0.block_until_ready()

			raw_init = self.params_obj.phys_to_raw(phys_map)

			iterator = tqdm(range(num_samples), total=num_samples, desc="Sampling obj")

			monte_params = []
			for n in iterator:
				raw_init = draws_raw[n]  # already float32
				t0 = time.perf_counter()
				params_m, _ = _minimizer(raw_init, *norm_spectra_T, constraints_jnp)
				t1 = time.perf_counter()

				monte_params.append(params_m)
				iterator.set_postfix({"it_s": f"{(t1 - t0):.4f}"})
	
			_monte_params = np.moveaxis(np.stack(monte_params),0,1)
			_draws_phys = np.moveaxis(draws_phys,0,1)
			
			#print(draws_raw[n])
			dic_posterior_params = {}

			iterator = tqdm(self.names, total=len(self.names), desc="Getting posterior-params")
			
			for n, name_i in enumerate(iterator):
				full_samples = scale_amp(self.params_dict,_monte_params[n],scale[n])
				draws_phys_n = scale_amp(self.params_dict,np.array(_draws_phys[n]),scale[n])
				dic_posterior_params[name_i] = self.complexparams.extract_params(full_samples,n,summarize=summarize)
				dic_posterior_params[name_i].update({"samples_phys":full_samples,"draws_phys":draws_phys_n})

			return dic_posterior_params


	def make_minimizer(self,model,non_optimize_in_axis,num_steps,learning_rate,
					method,penalty_weight,curvature_weight,smoothness_weight,max_weight,penalty_function=None,weighted=True,**kwargs):
		
		num_steps = 2_000
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
