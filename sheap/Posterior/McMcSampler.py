from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

import jax 
from jax import grad, vmap,jit, random

from sheap.Mappers.helpers import mapping_params
from sheap.Posterior.numpyro_helpers import make_numpyro_model,params_to_dict
from .parameter_from_sampler import full_params_sampled_to_posterior_params
class McMcSampler:
    # TODO big how to combine distributions
    def __init__(self, estimator: "ParameterEstimation"):
        
        self.estimator = estimator  # ParameterEstimation instance
        self.model_func = estimator.model
        self.c = estimator.c
        self.dependencies = estimator.dependencies
        self.kinds_map = estimator.kinds_map
        self.scale = estimator.scale
        self.fluxnorm = estimator.fluxnorm
        self.spec = estimator.spec
        self.mask = estimator.mask
        self.d = estimator.d #
        self.params = estimator.params #
        self.params_dict = estimator.params_dict
        self.BOL_CORRECTIONS = estimator.BOL_CORRECTIONS
        self.SINGLE_EPOCH_ESTIMATORS = estimator.SINGLE_EPOCH_ESTIMATORS
        
        ####part of the re-scale###
        #scale = self.scale
        norm_spec = self.spec.at[:, [1, 2], :].divide(
            jnp.moveaxis(jnp.tile(self.scale, (2, 1)), 0, 1)[:, :, None]
        )
        self.norm_spec = norm_spec.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spec[:, 2, :]))
        self.idxs = mapping_params(self.params_dict, [["amplitude"], ["scale"]])
        self.matrix_params = self.params.at[:, self.idxs].divide(self.scale[:, None])
        constraints = estimator.constraints #.at[self.idxs,:].divide(scale)
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
        
    
    def sample_params(self,n_random,num_warmup=500,num_samples=1000,list_of_objects=None):
            from sheap.RegionFitting.uncertainty_functions import (
                apply_tied_and_fixed_params
            )
            if list_of_objects is None:
                import numpy as np 
                print("The mcmc will be runend for all the sample")
                list_of_objects = np.arange(self.norm_spec.shape[0])

            name_list = self.name_list
            #print(name_list)
            constraints = self.constraints
            theta_to_sheap = self.theta_to_sheap
            #tied_targets = self.tied_targets
            fixed_params = self.fixed_params
            dependencies = self.dependencies
            model_func = self.model_func
            ##
            idxs = self.idxs #scale indx
            scale = self.scale
            #mmm 
            matrix_sample_params = jnp.zeros((len(list_of_objects),num_samples,self.matrix_params.shape[1]))
            dic_posterior_params = {}
            for n in list_of_objects:#range(self.max_flux.shape[0]): over all over this system option
                print(f"Runing mcmc for object {n}")
                wl_i,flux_i,yerr_i = self.norm_spec[n]
                params_i,mask_i = self.matrix_params[n],self.mask[n]
                free_params = self.matrix_params[n][jnp.array(self.idx_free_params)]
                self.matrix_params[n][jnp.array(self.idx_free_params)]
                params_to_dict(self.matrix_params[n],dependencies)
                
                init_values = {key: self.matrix_params[n][_] for _,key in enumerate(self.theta_to_sheap.values())}
                numpyro_model = make_numpyro_model(name_list,wl_i,flux_i,yerr_i,constraints,init_values,theta_to_sheap,fixed_params,dependencies,model_func)
                init_value = params_to_dict(params_i,dependencies,constraints)
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
                
                dic_posterior_params[n] = full_params_sampled_to_posterior_params(wl_i, flux_i, yerr_i,mask_i,full_samples,
                                                                                self.kinds_map,
                                                                                self.d[n],
                                                                                c=self.c,
                                                                                BOL_CORRECTIONS=self.BOL_CORRECTIONS,
                                                                                SINGLE_EPOCH_ESTIMATORS=self.SINGLE_EPOCH_ESTIMATORS)
                
                
            return matrix_sample_params,dic_posterior_params
        
    #def check on samples?