
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
        region_defs: List[SpectralLine],
        profile_functions: List[Any],
        initial_params: List[Any],
        profile_params_index_list: List[List[int]]
    ):
        self.region_defs = region_defs
        self.profile_functions = profile_functions
        self.initial_params = initial_params
        self.profile_params_index_list = profile_params_index_list
        self._last_filtered = {}  # cache for later use in combine_profiles()

    def __call__(self, where: Union[str, List[str]], what: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Filters lines where specified attributes contain given substrings.

        Args:
            where: Attribute name(s) to filter on (e.g., 'region', 'kind').
            what: Substring(s) to match against each attribute.

        Returns:
            Dictionary of filtered line data, stored internally for reuse.
        """
        entries = self.region_defs
        idx, regions, centers, kinds, components, line_names = np.array([
            [i, e.region, e.center, e.kind, e.component, e.line_name] for i, e in enumerate(entries)
        ]).T

        dic_ = {
            "region": regions,
            "center": centers,
            "kind": kinds,
            "component": components,
            "line_names": line_names
        }

        if isinstance(where, str):
            where = [where]
        if isinstance(what, str):
            what = [what]
        assert len(where) == len(what), "where and what must have the same length."

        mask = np.ones(len(entries), dtype=bool)
        for w, v in zip(where, what):
            mask &= np.char.find(dic_[w], v) >= 0

        mask_idx = np.where(mask)[0]

        # Apply filtered indices
        idx = idx[mask_idx].astype(int)
        filtered_profile_functions = np.array(self.profile_functions)[mask_idx]
        filtered_initial_params = np.array(self.initial_params)[mask_idx]
        filtered_profile_params_index = np.array(self.profile_params_index_list)[mask_idx].astype(int)
        profile_params_index_flat = [
            item for id_ in idx for item in self.profile_params_index_list[id_]
        ]

        result = {
            "idx": idx,
            "region": regions[mask_idx],
            "center": centers[mask_idx].astype(float),
            "kind": kinds[mask_idx],
            "component": components[mask_idx],
            "entries": np.array(entries)[mask_idx],
            "profile_functions": filtered_profile_functions,
            "initial_params": filtered_initial_params,
            "profile_params_index": filtered_profile_params_index,
            "line_name": line_names[mask_idx],
            "profile_params_index_list": profile_params_index_flat
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