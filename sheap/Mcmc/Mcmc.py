from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax 
from jax import grad, vmap,jit

from sheap.FunctionsMinimize.utils import combine_auto
from sheap.LineMapper.LineMapper import mapping_params
from sheap.RegionFitting.utils import make_constraints, make_get_param_coord_value
from sheap.FunctionsMinimize.utils import parse_dependency,parse_dependencies

def apply_arithmetic_ties(params: Dict[str, float], ties: List[Tuple]) -> Dict[str, float]:
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

class ShpMcmc:
    # TODO big how to combine distributions
    def __init__(
        self,
        sheap: Optional["Sheapectral"] = None,fixed_params={}):
        if sheap is not None:
            self._from_sheap(sheap)
        #elif fit_result is not None and spectra is not None:
         #   self._from_fit_result(fit_result, spectra,z)
        else:
            raise ValueError("Provide either `sheap` or (?).")
        self.fixed_params = fixed_params
        self.get_param_coord_value = make_get_param_coord_value(
            self.params_dict, 
            self.initial_params # this is only here to give the get_param_coord_ a kind of structure but in reality make no sense  
        )
        self.ties = parse_dependencies(self._build_tied())
        
    def _from_sheap(self, sheap):
        #self.z = sheap.z
        self.spectra_exp = sheap.spectra_exp
        #self.max_flux = sheap.max_flux
        #self.result = sheap.result  # keep reference if needed
        spec_scaled = jnp.array(sheap.spectra.at[:, [1, 2], :].multiply(10 ** (-1 * sheap.spectra_exp[:, None, None])))
        result = sheap.result  # for convenience
        
        idxs = mapping_params(result.params_dict, [["amplitude"], ["scale"]])
        scaled = 10 ** -self.spectra_exp
        self.profile_functions = result.profile_functions
        self.matrix_params = result.params.at[:, idxs].multiply(scaled[:, None])
        self.matrix_uncertainty_params = result.uncertainty_params.at[:, idxs].multiply(scaled[:, None])
        constraints = result.constraints.at[idxs,:].multiply(scaled)
        self.spec_scaled = spec_scaled.at[:,2,:].set(jnp.where(result.mask, 1e31,spec_scaled[:,2,:]))
        self.params_dict  = result.params_dict
        self.theta_to_sheap = {f"theta_{i}":str(key) for i,key in enumerate(self.params_dict.keys())} #params that can be use in the mcmc because params_dict names is to large
        self.constraints = [tuple(x) for x in jnp.asarray(constraints)] # this is the way to let the contrains in a safe mode#
        self.model_func = jit(combine_auto(self.profile_functions))
        #for sure exist more clear ways to do this 
        last_step = list(result.fitting_rutine.keys())[-1]
        self.tied_params = result.fitting_rutine[last_step]["tied"]
        self.initial_params = result.initial_params
    
    
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

    
    
    #kernel = NUTS(numpyro_model)
    #mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, progress_bar=True, 
     #       collect_fields=("log_likelihood",))
    #mcmc.run(jax.random.PRNGKey(2))
    
    def _runmcmc(self,n,nrandom=10,num_warmup=500,num_samples=1000):
        
        wl,flux,sigma = self.spec_scaled[n]
        name_list =  list(self.mcmc_dict.keys())
        def numpyro_model():
            params = []
            for nameparam, (low, high) in zip(name_list,self.constraints):
                #print(i,low,high)
                p = numpyro.sample(nameparam, dist.Uniform(low, high))
                params.append(p)
            #return params
            theta = jnp.array(params)
            pred = self.model_func(wl, theta)
            numpyro.sample("obs", dist.Normal(pred, sigma), obs=flux)

        # Run MCMC
        #init_strategy = init_to_value(values=init_values)
        kernel = NUTS(numpyro_model)#, init_strategy=init_strategy)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=1000, progress_bar=True)
        mcmc.run(jax.random.PRNGKey(nrandom))
        samples = mcmc.get_samples()
        #collect_fields=("log_likelihood",)
        self.samples = samples
        print("I ended")
    #n,nrandom=10,num_warmup=500,num_samples=1000
    
    def make_modelv2(self,n,fixed_params = None,nrandom=10,num_warmup=500,num_samples=1000):
        fixed_params =  fixed_params if fixed_params is not None else self.fixed_params
        wl,flux,sigma = self.spec_scaled[n]
        name_list =  list(self.theta_to_sheap.keys())
        self.init_values = {key:self.matrix_params[n][_] for _,key in enumerate(self.theta_to_sheap.values())}
        tied_targets = {target_idx for (_, _, target_idx, _, _) in self.ties}
        def numpyro_model():
            params = {}    
            for i, (name, (low, high)) in enumerate(zip(name_list, self.constraints)):
                sheap_name = self.theta_to_sheap[name]
                if i in tied_targets:
                    continue  # skip tied targets; they'll be calculated later
                elif sheap_name in self.fixed_params.keys():
                    val = self.fixed_params[sheap_name]
                    if val is None:
                        val = self.init_values.get(sheap_name)
                        if val is None:
                            raise ValueError(f"Fixed param '{sheap_name}' is None and not found in init_values.")
                else:
                    val = numpyro.sample(name, dist.Uniform(low, high))
                params[name] = val
            params = apply_arithmetic_ties(params, self.ties)
            theta = jnp.array([params[name] for name in name_list])
            pred = self.model_func(wl, theta)
            numpyro.sample("obs", dist.Normal(pred, sigma), obs=flux)
        
        kernel = NUTS(numpyro_model)#, init_strategy=init_strategy)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
        mcmc.run(jax.random.PRNGKey(nrandom))
        samples = mcmc.get_samples()
        #collect_fields=("log_likelihood",)
        self.samples = samples
        print("I ended")
        
        
        #return 
    
    


        #return init_values
        # for name, (low, high) in zip(name_list, self.constraints):
        #     sheap_name = self.theta_to_sheap[name]
        #     if sheap_name in self.fixed_params:
        #         val = self.fixed_params[sheap_name]
        #         if val is None:
        #             val = init_values.get(sheap_name)
        # def numpyro_model():
        #     theta_vals = {}
        #     for name, (low, high) in zip(name_list, self.constraints):
        #         sheap_name = self.theta_to_sheap[name]
        #         if sheap_name in self.fixed_params:
        #             val = self.fixed_params[sheap_name]
        #             if val is None:
        #                 val = self.init_values.get(sheap_name)
        #                 if val is None:
        #                     raise ValueError(f"Fixed param '{sheap_name}' is None and not found in init_values.")
                
        #         elif sheap_name in self.tied_params:
        #             ref_name, delta = self.tied_params[sheap_name]
        #             ref_theta = self.sheap_to_theta[ref_name]
        #             val = theta_vals[ref_theta] + delta
        #         else:
        #             val = numpyro.sample(name, dist.Uniform(low, high))

        #         theta_vals[name] = val

        #     theta = jnp.array([theta_vals[name] for name in self.name_list])
        #     pred = self.model_func(wl, theta)
        #     numpyro.sample("obs", dist.Normal(pred, sigma), obs=flux)

        # return numpyro_model
    
    
    
    
    
    
    
    def _build_tied(self):
        "this is a copy from RegionFtitting given both do the same maybe is moment to think about make this its own function"
        list_tied_params = []
        if len(self.tied_params) > 0:
            for tied in self.tied_params:
                param1, param2 = tied[:2]
                pos_param1, val_param1, param_1 = self.get_param_coord_value(
                    *param1.split("_")
                )
                pos_param2, val_param2, param_2 = self.get_param_coord_value(
                    *param2.split("_")
                )
                if len(tied) == 2:
                    if param_1 == param_2 == "center" and len(tied):
                        delta = val_param1 - val_param2
                        tied_val = "+" + str(delta) if delta > 0 else "-" + str(abs(delta))
                        # if log_mode:
                    elif param_1 == param_2:
                        tied_val = "*1"
                    else:
                        print(f"Define constraints properly. {self.tied_params}")
                else:
                    tied_val = tied[-1]
                if isinstance(tied_val, str):
                    list_tied_params.append(f"{pos_param1} {pos_param2} {tied_val}")
                else:
                    print("Define constraints properly.")
        else:
            list_tied_params = []
        return list_tied_params