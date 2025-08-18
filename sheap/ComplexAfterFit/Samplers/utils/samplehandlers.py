"""This module handles basic operations."""
__author__ = 'felavila'

__all__ = [
    "flatten_mass_dict",
    "flatten_mass_samples_to_df",
    "flatten_param_dict",
    "flatten_scalar_dict",
    "pivot_and_split",
    "summarize_nested_samples",
    "summarize_samples",
]

from typing import Dict, Any
import pandas as pd
import warnings
from auto_uncertainties.uncertainty.uncertainty_containers import VectorUncertainty
import numpy as np 
import jax.numpy as jnp
#?

def flatten_mass_samples_to_df(dict_samples: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract 'masses' from nested sample dictionaries and return a flat pandas DataFrame.
    
    Parameters:
    - dict_samples: Dictionary of objects, each containing a 'masses' dictionary.

    Returns:
    - A pandas DataFrame with columns: object, line, quantity, median, err_minus, err_plus
    """
    records = []
    
    for object_key, item in dict_samples.items():
        if not isinstance(item, Dict) or "masses" not in item:
            continue
        for line_name, stats in item["masses"].items():
            for stat_name, values in stats.items():
                records.append({
                    "object": object_key,
                    "line": line_name,
                    "quantity": stat_name,
                    "median": values["median"].item(),
                    "err_minus": values["err_minus"].item(),
                    "err_plus": values["err_plus"].item()
                })
    
    return pd.DataFrame(records)


def flatten_param_dict(dict_basic_params):
    rows = []
    for kind, values in dict_basic_params.items():
        lines = values["lines"]
        components = values["component"]
        for param_name, param_values in values.items():
            if param_name in ["lines", "component"]:
                continue
            medians = param_values["median"]
            err_minus = param_values.get("err_minus", [None]*len(medians))
            err_plus = param_values.get("err_plus", [None]*len(medians))

            for _, (line, comp, med, err_m, err_p) in enumerate(zip(lines, components, medians, err_minus, err_plus)):
                rows.append({
                    "line_name": line,
                    "component": comp,
                    "kind": kind,
                    "parameter": param_name,
                    "median": med,
                    "err_minus": err_m,
                    "err_plus": err_p
                })
    return pd.DataFrame(rows)

def flatten_scalar_dict(name, scalar_dict):
    rows = []
    for key, stats in scalar_dict.items():
        rows.append({
            "quantity": name,
            "wavelength_or_line": key,
            "median": stats["median"].item(),
            "err_minus": stats["err_minus"].item(),
            "err_plus": stats["err_plus"].item()
        })
    return pd.DataFrame(rows)


def flatten_mass_dict(masses):
    rows = []
    for line, metrics in masses.items():
        #print(line)
        for stat_name, stats in metrics.items():
            rows.append({
                "line_name": line,
                "quantity": stat_name,
                "median": stats["median"].item(),
                "err_minus": stats["err_minus"].item(),
                "err_plus": stats["err_plus"].item()
            })
    return pd.DataFrame(rows)



def pivot_and_split(obj_names, result):
    """
    Turn `result` (a nested dict of dicts whose leaves are either:
       - VectorUncertainty of length N,
       - indexable arrays/lists of length N, or
       - scalars
    ) into a dict keyed by each obj_name, where each leaf becomes either:
       - {'value': ..., 'error': ...} for VectorUncertainty
       - the single element node[obj_idx] for other indexables
       - the original scalar for non-indexables
    """
    def _recurse(node, idx):
        # 1) if it's a dict, recurse on each item
        if isinstance(node, dict):
            return {k: _recurse(v, idx) for k, v in node.items()}
        
        elif isinstance(node, (str, float, int)):
            return node
                
        # 2) if it's a VectorUncertainty, split into value & error
        elif isinstance(node, VectorUncertainty):
            return {
                'value': node.value[idx].squeeze(),
                'error': node.error[idx].squeeze()
            }
        elif isinstance(node, np.ndarray) and node.shape[0] == len(obj_names):
            return {'value': node[idx].squeeze(),'error':0}
        # 3) array/list/tuple → index        
        elif isinstance(node, (np.ndarray, list, tuple)):
            # if isinstance(node, list) and all(isinstance(x, dict) for x in node):
            #     return [_recurse(n, idx) for n in node]
            return node
        
        warnings.warn(f"Unhandled node type {type(node).__name__} for value: {node}")
    return {
        obj_name: _recurse(result, obj_idx)
        for obj_idx, obj_name in enumerate(obj_names)
    }


def summarize_samples(samples) -> Dict[str, np.ndarray]:
    """Compute 16/50/84 percentiles and return a summary dict using NumPy."""
    if isinstance(samples, jnp.ndarray):
        samples = np.asarray(samples)
    samples = np.atleast_2d(samples).T
    if np.isnan(samples).sum() / samples.size > 0.2:
        warnings.warn("High fraction of NaNs; uncertainty estimates may be biased.")
    if samples.shape[1]<=1:
        q = np.nanpercentile(samples, [16, 50, 84], axis=0)
    else:
        q = np.nanpercentile(samples, [16, 50, 84], axis=1)
    #else:
    
    return {
        "median": q[1],
        "err_minus": q[1] - q[0],
        "err_plus": q[2] - q[1]
    }
    
    
def summarize_nested_samples(d: dict) -> dict:
    """
    Recursively walk through a dictionary and apply summarize_samples_numpy
    to any array-like values.
    """
    summarized = {}
    for k, v in d.items():
        if isinstance(v, dict):
            summarized[k] = summarize_nested_samples(v)
        elif isinstance(v, (np.ndarray, jnp.ndarray)) and np.ndim(v) >= 1 and k!='component':
            summarized[k] = summarize_samples(v)
        else:
            summarized[k] = v
    return summarized



#TODO the functions to handle the after sapling methods are still in developed process like take the best "parameters" and teh 1sigma plot 
# import numpy as np 

# fwhm_kms_all = broad_params.get("fwhm_kms")
# lum_all = broad_params.get("luminosity")
# line_list = np.array(broad_params.get("lines", []))
# component_list = np.array(broad_params.get("component", []))

# def ensure_column_matrix(x):
#     #to utils.
#     x = np.asarray(x)
#     if x.ndim == 1:
#         return x.reshape(-1, 1)  # Convert to (N, 1)
#     return x


# #print(estimators.keys())
# for line_name  in ['Halpha']:
   
#     idxs = np.where(line_list == line_name)[0]

#     compt = component_list[idxs]
#     fkm = ensure_column_matrix(fwhm_kms_all)[:, idxs].squeeze()
#     lum_custom = ensure_column_matrix(lum_all)[:, idxs].squeeze()
