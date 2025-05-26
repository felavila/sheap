from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def apply_arithmetic_ties(params: Dict[str, float], ties: List[Tuple]) -> Dict[str, float]:
    #_, target, source, op, operand = dep
    #tag,target_idx,src_idx, op, val = ties
    for tag, target_idx,src_idx, op, val in ties:
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

def make_numpyro_model(name_list,wl,flux,sigma,constraints,init_values,theta_to_sheap,fixed_params,dependencies,model_func):
    def numpyro_model():
        params = {}    
        idx_targets = [i[1] for i in dependencies]
        for i, (name, (low, high)) in enumerate(zip(name_list, constraints)):
            sheap_name = theta_to_sheap[name]
            if i in idx_targets:
                #print(i,idx_targets,dependencies)
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
        params = apply_arithmetic_ties(params, dependencies)
        theta = jnp.array([params[name] for name in name_list])
        pred = model_func(wl, theta)
        numpyro.sample("obs", dist.Normal(pred, sigma), obs=flux)
    return numpyro_model  