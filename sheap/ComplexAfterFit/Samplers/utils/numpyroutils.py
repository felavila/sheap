"""This module handles basic operations."""
__author__ = 'felavila'

__all__ = [
    "apply_arithmetic_ties",
    "make_numpyro_model",
    "params_to_dict",
    "params_to_dict_old",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def apply_arithmetic_ties(params: Dict[str, float], dependencies: List[Tuple]) -> Dict[str, float]:
    #_, target, source, op, operand = dep
    #tag,target_idx,src_idx, op, val = ties
    for tag, target_idx,src_idx, op, val in dependencies:
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

def params_to_dict_old(params,dependencies):
    #tag,target_idx,src_idx, op, val = dependencies 
    target_idx_list = [d[1] for d in dependencies]
    init_value = {}
    for idx,val in enumerate(params):
        if idx in target_idx_list:
            continue
        else:
            init_value[f"theta_{idx}"] = val
    return init_value



def params_to_dict(params, dependencies, constraints=None, eps=1e-2):
    """
    Convert a list of parameter values into a dict for init_to_value,
    excluding tied (dependent) parameters and applying jitter near bounds.

    Args:
        params: list or array of parameter values (e.g. from minimization)
        dependencies: list of (tag, target_idx, src_idx, op, val)
        constraints: optional dict {f"theta_{i}": (low, high)} or list of (low, high)
        eps: minimum distance from bounds to avoid edge initialization

    Returns:
        init_dict: dict of {param_name: jittered_value}
    """
    if not dependencies:
        dependencies = ()
    target_idx_list = [d[1] for d in dependencies]
    init_dict = {}

    for idx, val in enumerate(params):
        if idx in target_idx_list:
            continue  # skip dependent (tied) parameters
        name = f"theta_{idx}"
        val = float(val)

        # Apply jitter if constraints are available
        if constraints is not None:
            if isinstance(constraints, dict):
                low, high = constraints.get(name, (-jnp.inf, jnp.inf))
            else:  # assume list aligned with params
                low, high = constraints[idx]

            if not jnp.isfinite(val) or val <= low or val >= high:
                val = 0.5 * (low + high) if jnp.isfinite(low) and jnp.isfinite(high) else val
            if val - low < eps:
                val = low + eps
            elif high - val < eps:
                val = high - eps

        # Optional: prevent exact zero
        if val == 0:
            val = eps

        init_dict[name] = val

    return init_dict

def make_numpyro_model(name_list,wl,flux,sigma,constraints,params_i,theta_to_sheap,fixed_params,dependencies,model_func):
    
    init_values = {key: params_i[_] for _,key in enumerate(theta_to_sheap.values())}
    init_value = params_to_dict(params_i,dependencies,constraints)
    if not dependencies:
        dependencies = ()
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
    
    
    return numpyro_model,init_value

