from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

import jax 
from jax import grad, vmap,jit, random

from sheap.Mappers.helpers import mapping_params
from sheap.Posterior.numpyro_helpers import make_numpyro_model

class McMcSampler:
    # TODO big how to combine distributions
    def __init__(self, estimator: "ParameterEstimation"):
        
        self.estimator = estimator  # ParameterEstimation instance
        self.model_func = estimator.model
        self.c = estimator.c
        self.dependencies = estimator.dependencies
        self.kinds_map = estimator.kinds_map
        self.max_flux = estimator.max_flux
        self.fluxnorm = estimator.fluxnorm
        self.spec = estimator.spec
        self.mask = estimator.mask
        self.d = estimator.d #
        self.params = estimator.params #
        self.params_dict = estimator.params_dict
        
        ####part of the re-scale###
        scaled = self.max_flux
        norm_spec = self.spec.at[:, [1, 2], :].divide(
            jnp.moveaxis(jnp.tile(scaled, (2, 1)), 0, 1)[:, :, None]
        )
        self.norm_spec = norm_spec.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spec[:, 2, :]))
        self.idxs = mapping_params(self.params_dict, [["amplitude"], ["scale"]])
        self.matrix_params = self.params.at[:, self.idxs].divide(scaled[:, None])
        constraints = estimator.constraints #.at[self.idxs,:].divide(scaled)
        #print(constraints)
        #self.model_func = self.model #can  test if a model is already jited?
        dependencies = self.dependencies
        idx_target = [i[1] for i in dependencies] # already calculated
        self.idx_free_params = list(set(range(len(self.matrix_params[0]))) - set(idx_target))
        self.constraints = [tuple(x) for x in jnp.asarray(constraints)] # this is the way to let the contrains in a safe mode#
        self.theta_to_sheap = {f"theta_{i}":str(key) for i,key in enumerate(self.params_dict.keys())} #params that can be use in the mcmc because params_dict names is to large
        self.name_list =  list(self.theta_to_sheap.keys())
        self.tied_targets = {target_idx for (_, _, target_idx, _, _) in  self.dependencies}
        self.fixed_params = {}
        self.ties = None 
        self.scaled = scaled #change names require 
        #self.init_values = {key:self.matrix_params[n][_] for _,key in enumerate(self.theta_to_sheap.values())}
        #key = random.PRNGKey(key_seed)
    
    # def sample_params(self,n_random,num_warmup=500,num_samples=1000):
    #     from sheap.RegionFitting.uncertainty_functions import (
    #         apply_tied_and_fixed_params
    #     )
    #     name_list = self.name_list
    #     constraints = self.constraints
    #     theta_to_sheap = self.theta_to_sheap
    #     #tied_targets = self.tied_targets
    #     fixed_params = self.fixed_params
    #     dependencies = self.dependencies
    #     #print(dependencies)
    #     model_func = self.model_func
    #     dependencies = self.dependencies
    #     params_i = self.matrix_params[0]
    #     idxs = self.idxs #scale indx
    #     scaled = self.scaled
    #     for n in [0]:#range(self.max_flux.shape[0]): over all over this system option
    #         wl,flux,sigma = self.norm_spec[n]
    #         free_params = self.matrix_params[n][jnp.array(self.idx_free_params)]
    #         init_values = {key: self.matrix_params[n][_] for _,key in enumerate(self.theta_to_sheap.values())}
    #         numpyro_model = make_numpyro_model(name_list,wl,flux,sigma,constraints,init_values,theta_to_sheap,fixed_params,dependencies,model_func)
    #         init_strategy = init_to_value(values=free_params)
            
    #         kernel = NUTS(numpyro_model, init_strategy=init_strategy)
    #         mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
    #         mcmc.run(jax.random.PRNGKey(n_random))
    #         get_samples = mcmc.get_samples()
    #         sorted_theta = sorted(get_samples.keys(), key=lambda x: int(x.split('_')[1]))  #How much info can be lost in this steep?
    #         samples_free = jnp.array([get_samples[i] for i in sorted_theta]).T
    #         #full_samples = vmap(apply_one_sample)(samples_free)
    #         def apply_one_sample(free_sample):
    #             return apply_tied_and_fixed_params(free_sample, params_i, dependencies)
    #         full_samples = vmap(apply_one_sample)(samples_free)
    #         full_samples = full_samples.at[:, idxs].multiply(scaled[n])
            
    #         #collect_fields=("log_likelihood",)
    #         #samples = samples
        
    #     return full_samples
    
    
    def sample_params(self,n_random,num_warmup=500,num_samples=1000,list_of_objects=None):
            from sheap.RegionFitting.uncertainty_functions import (
                apply_tied_and_fixed_params
            )
            if list_of_objects is None:
                import numpy as np 
                print("The mcmc will be runend for all the sample")
                list_of_objects = np.arange(self.norm_spec.shape[0])

            name_list = self.name_list
            constraints = self.constraints
            theta_to_sheap = self.theta_to_sheap
            #tied_targets = self.tied_targets
            fixed_params = self.fixed_params
            dependencies = self.dependencies
            model_func = self.model_func
            dependencies = self.dependencies
            
            idxs = self.idxs #scale indx
            scaled = self.scaled
            #mmm 
            matrix_sample_params = jnp.zeros((len(list_of_objects),num_samples,self.matrix_params.shape[1]))
            for n in list_of_objects:#range(self.max_flux.shape[0]): over all over this system option
                wl,flux,sigma = self.norm_spec[n]
                params_i = self.matrix_params[n]
                free_params = self.matrix_params[n][jnp.array(self.idx_free_params)]
                init_values = {key: self.matrix_params[n][_] for _,key in enumerate(self.theta_to_sheap.values())}
                numpyro_model = make_numpyro_model(name_list,wl,flux,sigma,constraints,init_values,theta_to_sheap,fixed_params,dependencies,model_func)
                init_strategy = init_to_value(values=free_params)
                
                kernel = NUTS(numpyro_model, init_strategy=init_strategy)
                mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
                mcmc.run(jax.random.PRNGKey(n_random))
                get_samples = mcmc.get_samples()
                sorted_theta = sorted(get_samples.keys(), key=lambda x: int(x.split('_')[1]))  #How much info can be lost in this steep?
                samples_free = jnp.array([get_samples[i] for i in sorted_theta]).T             #collect_fields=("log_likelihood",)
                #full_samples = vmap(apply_one_sample)(samples_free)
                def apply_one_sample(free_sample):
                    return apply_tied_and_fixed_params(free_sample, params_i, dependencies)
                full_samples = vmap(apply_one_sample)(samples_free)
                full_samples = full_samples.at[:, idxs].multiply(scaled[n])
                matrix_sample_params = matrix_sample_params.at[n].set(full_samples)
                #from samples to full samples here.
                #then we choose the strategi 

                #samples = samples
            
            return matrix_sample_params