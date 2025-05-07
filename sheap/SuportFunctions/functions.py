
from typing import List, Union, Dict, Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from jax import jit, vmap

from sheap.FunctionsMinimize.utils import combine_auto
from sheap.DataClass.DataClass import SpectralLine,FittingLimits


def mapping_params(params_dict,params,verbose=False):
        """
        params is a str or list
        [["width","broad"],"cont"]
        if verbose you can check if the mapping of parameters was correctly done
        """
        if isinstance(params,str):
            params = [params]
        match_list = []
        for param in params:
            if isinstance(param,str):
                param = [param]
            #print(self.params_dict.keys())
            #print([[self.params_dict[key],key] for key in self.params_dict.keys() if all([p in key for p in param])])
            
            match_list += ([params_dict[key] for key in params_dict.keys() if all([p in key for p in param])])
        
        match_list = jnp.array(match_list)
        unique_arr = jnp.unique(match_list)
        if verbose:
            print(np.array(list(params_dict.keys()))[unique_arr])#[])
        return unique_arr



class LineMapper:
    """
    Filters and maps spectral line entries based on attribute conditions.
    Also supports combining selected profile functions using JAX for efficient evaluation.
    
    Attributes:
        region_defs: List of Line or SpectralLine entries.
        profile_functions: List of associated profile functions.
        initial_params: Initial parameters for each line.
        profile_params_index_list: Index mapping for profile parameters.
    """

    def __init__(
        self,
        complex_region: List[SpectralLine],
        profile_functions: List[Any],
        params: List[Any], #array 
        profile_params_index_list: List[List[int]],
        params_dict,
        profile_names
    ):
        self.complex_region = complex_region
        self.profile_functions = profile_functions
        self.params = params
        self.profile_params_index_list = profile_params_index_list
        self.params_dict = params_dict
        self.profile_names = profile_names
        self._last_filtered = {}  # cache for later use in combine_profiles()
    
    def _get(self,where: Union[str, List[str]],what: Union[str, List[str]],
    logic: str = "and",  # Can be "or" or "and"
    super_param = None
    ) -> Dict[str, Any]:
        entries = self.complex_region
        idx, regions, centers, kinds, components, line_names = zip(*[
            (i, e.region, e.center, e.kind, e.component, e.line_name) for i, e in enumerate(entries)
        ])

        dic_ = {
            "region": np.array(regions),
            "center": np.array(centers),
            "kind": np.array(kinds),
            "component": np.array(components),
            "line_name": np.array(line_names),
        }

        if isinstance(where, str):
            where = [where]
        if isinstance(what, str):
            what = [what]
        assert len(what)==len(where), "where and what have to have the same lenght"
        mask = np.ones(len(entries), dtype=bool) if logic == "and" else np.zeros(len(entries), dtype=bool)
        #m = []
        for w,v in zip(where,what):
            current_mask = np.char.find(dic_[w].astype(str), v) >= 0
            if logic == "or":
                mask |= current_mask
            elif logic == "and":
                mask &= current_mask
            else:
                raise ValueError(f"Invalid logic '{logic}'. Choose 'or' or 'and'.")
            if isinstance(super_param,dict):
                mask &= np.char.find(dic_[super_param.get("where")].astype(str), super_param.get("what")) >= 0        
        
        #self.m = m
        mask_idx = np.where(mask)[0]
        idx = np.array(idx)[mask_idx].astype(int)

        filtered_profile_functions = np.array(self.profile_functions)[mask_idx]
        filtered_profile_names =  np.array(self.profile_names)[mask_idx]
        filtered_profile_params_index_list = np.array(self.profile_params_index_list, dtype=object)[mask_idx]
        profile_params_index_flat = np.concatenate(filtered_profile_params_index_list)
        filtered_params = np.array(self.params)[:, profile_params_index_flat]
        
        result = {
            "idx": idx.tolist(),
            "line_name": dic_["line_name"][mask_idx],
            "region": dic_["region"][mask_idx].tolist(),
            "center": dic_["center"][mask_idx].astype(float).tolist(),
            "kind": dic_["kind"][mask_idx].tolist(),
            "original_centers":np.array(centers)[mask_idx],
            "component": dic_["component"][mask_idx].tolist(),
            "entries": np.array(entries)[mask_idx].tolist(),
            "profile_functions": filtered_profile_functions,
            "profile_names": filtered_profile_names,
            "profile_params_index_flat": profile_params_index_flat,
            "profile_params_index_list": filtered_profile_params_index_list,
            "params_names": np.array(list(self.params_dict.keys()))[profile_params_index_flat.astype(int)],
            "params": filtered_params,
        }

        self._last_filtered = result
        return result

    
    def combine_profiles(self, spec: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Applies JAX-compiled and vectorized profile combination using last filtered selection.

        Args:
            spec: Input spectrum array of shape (N, λ) or (N, M) where N is batch.
            params: Full parameter array of shape (N, P), from which relevant slices are taken.

        Returns:
            Array of combined profiles per sample.
        """
        if not self._last_filtered:
            raise RuntimeError("No filtered result found. Call the instance with `where` and `what` first.")

        profile_funcs = self._last_filtered["profile_functions"]
        param_ids = self._last_filtered["profile_params_index_list"]

        combine_fn = jit(combine_auto(profile_funcs))
        combine_vectorized = vmap(combine_fn, in_axes=(0, 0))

        # Slice parameters accordingly
        param_subset = params[:, param_ids]
        return combine_vectorized(spec, param_subset)


# def mapping_lines(region_defs,profile_functions,initial_params,profile_params_index_list,where, what):
#         "this should be a class and where and what the call of the class"
#         entries = region_defs  # list of Lines or list of SpectralLine
#         idx, regions, centers, kinds, component, line_names = np.array([
#             [i, e.region, e.center, e.kind, e.component, e.line_name] for i, e in enumerate(entries)
#         ]).T

#         dic_ = {"region": regions, "center": centers, "kind": kinds, "component": component,"line_names":line_names}

#         # Normalize where and what to lists
#         if isinstance(where, str):
#             where = [where]
#         if isinstance(what, str):
#             what = [what]

#         assert len(where) == len(what), "where and what must have the same length."

#         # Build the mask
#         mask = np.ones(len(entries), dtype=bool)  # Start with everything True
#         for w, v in zip(where, what):
#             mask &= np.char.find(dic_[w], v) >= 0

#         # Apply mask
#         mask_idx = np.where(mask)[0]

#         idx = idx[mask_idx].astype(int)
#         regions = regions[mask_idx]
#         centers = centers[mask_idx].astype(float)
#         kinds = kinds[mask_idx]
#         components = component[mask_idx]
#         entries = np.array(entries)[mask_idx]
#         profile_functions = np.array(profile_functions)[mask_idx]
#         initial_params = np.array(initial_params)[mask_idx]
#         profile_params_index = np.array(profile_params_index)[mask_idx].astype(int)
#         profile_params_index_list = [item for id in idx for item in profile_params_index_list[id]]
#         line_names = line_names[mask_idx]

#         # Return as a dictionary
#         result = {
#             "idx": idx,
#             "region": regions,
#             "center": centers,
#             "kind": kinds,
#             "component": components,
#             "entries": entries,
#             "profile_functions": profile_functions,
#             "initial_params": initial_params,
#             "profile_params_index": profile_params_index,
#             "line_name": line_names,
#             "profile_params_index_list":profile_params_index_list
#         }

#         return result
    
    
    
    
#     from jax import jit,vmap
# from sheap.Fitting.utils import combine_auto
# ff = aja.mapping_lines("region","continuum")
# profile_functions = list(ff["profile_functions"])
# id_params = list(ff["profile_params_index_list"])
# combine = jit(combine_auto(profile_functions))
# comb_v = vmap(jit(combine_auto(profile_functions)),in_axes=(0,0))
# Fe_2 = comb_v(aja.spec[:,0,:],aja.params[:,id_params])