from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax 
from jax import grad, vmap,jit, random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value
from tqdm import tqdm



from sheap.Mappers.helpers import mapping_params
from .tools.numpyro_helpers import make_numpyro_model,params_to_dict
from .parameter_from_sampler import posterior_physical_parameters


class McMcSampler:
    # TODO big how to combine distributions
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
        self.complex_class = estimator.complex_class
        self.constraints = estimator.constraints 
        
    def sample_params(self, num_samples: int = 2000, num_warmup:int = 500,summarize=True,get_full_posterior=True,n_random=1_000,
                      list_of_objects=None,key_seed: int = 0,extra_products=True) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        from sheap.RegionFitting.uncertainty_functions import apply_tied_and_fixed_params
        scale = self.scale
        model = self.model
        names = self.names
        constraints = self.constraints
        dependencies = self.dependencies 
        norm_spec = self.spec.at[:, [1, 2], :].divide(jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None])
        norm_spec = norm_spec.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spec[:, 2, :]))
        norm_spec = norm_spec.astype(jnp.float64)
        wl, flux, yerr = jnp.moveaxis(norm_spec, 0, 1)
        idxs = mapping_params(self.params_dict, [["amplitude"], ["scale"]])
        params = self.params.at[:, idxs].divide(scale[:, None]).astype(jnp.float64)
        #idx_target = [i[1] for i in self.dependencies] # already calculated
        #idx_free_params = list(set(range(len(params[0]))) - set(idx_target))
        constraints = [tuple(x) for x in jnp.asarray(constraints)] #constrains are ok they are still in space 0-2.
        theta_to_sheap = {f"theta_{i}":str(key) for i,key in enumerate(self.params_dict.keys())} #dictionary that creates "theta_n" params easier to work with them in numpyro.
        name_list =  list(theta_to_sheap.keys())
        #tied_targets = {target_idx for (_, _, target_idx, _, _) in  self.dependencies}
        fixed_params = {}
        if not list_of_objects:
            import numpy as np 
            print("The mcmc will be runend for all the sample")
            list_of_objects = np.arange(norm_spec.shape[0])
        dic_posterior_params = {}
        matrix_sample_params = jnp.zeros((norm_spec.shape[0],num_samples,params.shape[1])) 
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
            full_samples = full_samples.at[:, idxs].multiply(scale[n])
            matrix_sample_params = matrix_sample_params.at[n].set(full_samples)
            dic_posterior_params[name_i] = posterior_physical_parameters(wl_i, flux_i, yerr_i,mask_i,full_samples,self.complex_class
                                                                                ,np.full((num_samples,), self.d[n],dtype=np.float64),
                                                                                c=self.c,
                                                                                BOL_CORRECTIONS=self.BOL_CORRECTIONS,
                                                                                SINGLE_EPOCH_ESTIMATORS=self.SINGLE_EPOCH_ESTIMATORS,
                                                                                summarize=summarize,extra_products=extra_products)
            #iterator.close()
            return dic_posterior_params
       