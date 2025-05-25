from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax 
from jax import grad, vmap,jit, random

from sheap.FunctionsMinimize.utils import combine_auto
from sheap.LineMapper.LineMapper import mapping_params
from sheap.RegionFitting.utils import make_constraints, make_get_param_coord_value
from sheap.FunctionsMinimize.utils import parse_dependency,parse_dependencies

#Check again all the stuff abut this indexing 
#the official structure throughout the code is tag,target_idx,src_idx, op, val = ties
##_, target, source, op, operand = dep
#tag,target_idx,src_idx, op, val = ties
def apply_arithmetic_ties(params: Dict[str, float], ties: List[Tuple]) -> Dict[str, float]:
    #_, target, source, op, operand = dep
    #tag,target_idx,src_idx, op, val = ties
    for tag, src_idx, target_idx, op, val in ties:
        src = params[f"theta_{src_idx}"]
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
        params[f"theta_{target_idx}"] = result
    return params

def apply_arithmetic_ties_restore(samples, ties):
    #_, target, source, op, operand = dep
    #tag,target_idx,src_idx, op, val = ties
    tag, src_idx, target_idx, op, val = ties
    src = samples[f"theta_{src_idx}"]
    if op == '-':
        result = src + val
    elif op == '+':
        result = src - val
    elif op == '/':
        result = src * val
    elif op == '*':
        result = src / val
    else:
        raise ValueError(f"Unsupported operation: {op}")
        #params[f"theta_{target_idx}"] = result
    return result

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
        norm_spec = norm_spec.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spec[:, 2, :]))
        idxs = mapping_params(self.params_dict, [["amplitude"], ["scale"]])
        matrix_params = self.params.at[:, idxs].divide(scaled[:, None])
        constraints = estimator.constraints.at[idxs,:].divide(scaled)
        #self.model_func = self.model #can  test if a model is already jited?
        dependencies = self.dependencies
        idx_target = [i[1] for i in dependencies]
        idx_free_params = list(set(range(len(matrix_params[0]))) - set(idx_target))
        self.name_list =  list(self.theta_to_sheap.keys())
        self.constraints = [tuple(x) for x in jnp.asarray(constraints)] # this is the way to let the contrains in a safe mode#
        self.theta_to_sheap = {f"theta_{i}":str(key) for i,key in enumerate(self.params_dict.keys())} #params that can be use in the mcmc because params_dict names is to large
        self.tied_targets = {target_idx for (_, _, target_idx, _, _) in  self.dependencies}
        self.fixed_params = None
        self.ties = None 
        #self.init_values = {key:self.matrix_params[n][_] for _,key in enumerate(self.theta_to_sheap.values())}
        #key = random.PRNGKey(key_seed)
    
    def sample_params(self,n_random,num_warmup=500,num_samples=1000):
        name_list = self.name_list
        constraints = self.constraints
        theta_to_sheap = self.theta_to_sheap
        tied_targets = self.tied_targets
        fixed_params = self.fixed_params
        ties = self.ties
        model_func = self.model_func
        for n in range(self.max_flux.shape[0]):
            wl,flux,sigma = self.spec_scaled[n]
            init_values = {key: self.matrix_params[n][_] for _,key in enumerate(self.theta_to_sheap.values())}
            numpyro_model = self._make_numpyro_model(name_list,wl,flux,sigma,constraints,init_values,theta_to_sheap,tied_targets,fixed_params,ties,model_func)
            #here we can add the init strategy 
            kernel = NUTS(numpyro_model)#, init_strategy=init_strategy)
            mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
            mcmc.run(jax.random.PRNGKey(n_random))
            samples = mcmc.get_samples()
            # def apply_one_sample(free_sample):
            #     return apply_tied_and_fixed_params(free_sample, params_i, dependencies)

            # full_samples = vmap(apply_one_sample)(samples_free)
            # full_samples = full_samples.at[:, idxs].multiply(scaled[n])
            
            #from samples to full samples here.
            #then we choose the strategi 
            #collect_fields=("log_likelihood",)
            #samples = samples
        
        return 
    
    
    def _make_numpyro_model(name_list,wl,flux,sigma,constraints,init_values,theta_to_sheap,tied_targets,fixed_params,ties,model_func):
        def numpyro_model():
            params = {}    
            for i, (name, (low, high)) in enumerate(zip(name_list, constraints)):
                sheap_name = theta_to_sheap[name]
                if i in tied_targets:
                    continue  # skip tied targets; they'll be calculated later
                elif sheap_name in fixed_params.keys():
                    val = fixed_params[sheap_name]
                    if val is None:
                        val = init_values.get(sheap_name)
                        if val is None:
                            raise ValueError(f"Fixed param '{sheap_name}' is None and not found in init_values.")
                else:
                    val = numpyro.sample(name, dist.Uniform(low, high))
                params[name] = val
            params = apply_arithmetic_ties(params, ties)
            theta = jnp.array([params[name] for name in name_list])
            pred = model_func(wl, theta)
            numpyro.sample("obs", dist.Normal(pred, sigma), obs=flux)
        return numpyro_model  
    
    
    def _get_best_values(self,samples=None,get_best=True):
        import pandas as pd
        samples = samples or self.samples
        ties = self.ties
        dict_ ={f"theta_{i[2]}":i for i in ties}
        theta_to_sheap = self.theta_to_sheap
        summary_df = pd.DataFrame({
            name: samples[name] if name in samples.keys() 
            else apply_arithmetic_ties_restore(samples,dict_[name]) for name in list(theta_to_sheap.keys())
            })
        #summary_df["theta_{src_idx}"]
        summary_df = summary_df.rename(columns=theta_to_sheap)
        
        summary_stats = summary_df.describe(percentiles=[0.16, 0.5, 0.84]).T
        if get_best:
            params_bh = summary_stats._get_best_values()["mean"].values
            idxs = mapping_params(self.params_dict, [["amplitude"], ["scale"]])
            scaled = 10 ** self.spectra_exp[0]
            params_bh[idxs] = params_bh[idxs]*scaled
        return params_bh           
    
    # def _make_modelold(self,n,fixed_params = None,nrandom=10,num_warmup=500,num_samples=1000):
    #     fixed_params =  fixed_params if fixed_params is not None else self.fixed_params
    #     #wl,flux,sigma = self.spec_scaled[n]
    #     name_list =  list(self.theta_to_sheap.keys())
    #     self.init_values = {key:self.matrix_params[n][_] for _,key in enumerate(self.theta_to_sheap.values())}
    #     tied_targets = {target_idx for (_, _, target_idx, _, _) in self.ties}
    #     def numpyro_model():
    #         params = {}    
    #         for i, (name, (low, high)) in enumerate(zip(name_list, self.constraints)):
    #             sheap_name = self.theta_to_sheap[name]
    #             if i in tied_targets:
    #                 continue  # skip tied targets; they'll be calculated later
    #             elif sheap_name in self.fixed_params.keys():
    #                 val = self.fixed_params[sheap_name]
    #                 if val is None:
    #                     val = self.init_values.get(sheap_name)
    #                     if val is None:
    #                         raise ValueError(f"Fixed param '{sheap_name}' is None and not found in init_values.")
    #             else:
    #                 val = numpyro.sample(name, dist.Uniform(low, high))
    #             params[name] = val
    #         params = apply_arithmetic_ties(params, self.ties)
    #         theta = jnp.array([params[name] for name in name_list])
    #         pred = self.model_func(wl, theta)
    #         numpyro.sample("obs", dist.Normal(pred, sigma), obs=flux)
        
    #     kernel = NUTS(numpyro_model)#, init_strategy=init_strategy)
    #     mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
    #     mcmc.run(jax.random.PRNGKey(nrandom))
    #     samples = mcmc.get_samples()
    #     #collect_fields=("log_likelihood",)
    #     self.samples = samples