from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from jax import jit, vmap

from sheap.DataClass.DataClass import FittingLimits, SpectralLine
from sheap.DataClass.utils import is_list_of
from sheap.FunctionsMinimize.utils import combine_auto


def mapping_params(params_dict, params, verbose=False):
    """
    params is a str or list
    [["width","broad"],"cont"]
    if verbose you can check if the mapping of parameters was correctly done
    """
    if isinstance(params_dict, np.ndarray):
        params_dict = {str(key): n for n, key in enumerate(params_dict)}
    if isinstance(params, str):
        params = [params]
    match_list = []
    for param in params:
        if isinstance(param, str):
            param = [param]
        # print(self.params_dict.keys())
        # print([[self.params_dict[key],key] for key in self.params_dict.keys() if all([p in key for p in param])])

        match_list += [
            params_dict[key] for key in params_dict.keys() if all([p in key for p in param])
        ]

    match_list = jnp.array(match_list)
    unique_arr = jnp.unique(match_list)
    if verbose:
        print(np.array(list(params_dict.keys()))[unique_arr])  # [])
    return unique_arr


@dataclass
class LineSelectionResult:
    idx: List[int]
    line_name: np.ndarray
    region: List[str]
    center: List[float]
    kind: List[str]
    original_centers: np.ndarray
    component: List[Union[int, str]]
    lines: List[Any]
    profile_functions: np.ndarray
    profile_names: np.ndarray
    profile_params_index_flat: np.ndarray
    profile_params_index_list: np.ndarray
    params_names: np.ndarray
    params: np.ndarray
    uncertainty_params: np.ndarray
    profile_functions_combine: Callable[[np.ndarray, jnp.ndarray], jnp.ndarray]


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
        params: List[Any],
        uncertainty_params: List[Any],
        profile_params_index_list: List[List[int]],
        params_dict,
        profile_names,
        kind_list=None,
    ):

        if is_list_of(complex_region, SpectralLine):
            self.complex_region = complex_region
        elif is_list_of(complex_region, dict):
            self.complex_region = [SpectralLine(**i) for i in complex_region]
        else:
            raise TypeError(
                "complex_region must be a list of SpectralLine instances or a list of dicts that can be unpacked into SpectralLine."
            )

        # self.complex_region = complex_region
        self.profile_functions = profile_functions
        self.params = params
        self.profile_params_index_list = profile_params_index_list
        self.params_dict = params_dict
        self.profile_names = profile_names
        self.uncertainty_params = uncertainty_params
        self._last_filtered = {}  # cache for later use in combine_profiles()

    def _get(
        self,
        where: Union[str, List[str]],
        what: Union[str, List[str]],
        logic: str = "and",
        super_param: Optional[Dict[str, str]] = None,
    ) -> LineSelectionResult:

        entries = self.complex_region
        n_entries = len(entries)
        # Extract attributes into arrays
        attributes_dict = {
            key: np.array([getattr(e, key) for e in entries])
            for key in ["region", "center", "kind", "component", "line_name"]
        }

        # Build logic mask
        def make_mask(key: str, value: str):
            return np.char.find(attributes_dict[key].astype(str), value) >= 0

        if isinstance(where, str):
            where = [where]
        if isinstance(what, str):
            what = [what]
        assert len(where) == len(what), "`where` and `what` must have same length."

        mask = (
            np.ones(n_entries, dtype=bool)
            if logic == "and"
            else np.zeros(n_entries, dtype=bool)
        )
        for w, v in zip(where, what):
            current_mask = make_mask(w, v)
            mask = mask & current_mask if logic == "and" else mask | current_mask

        if isinstance(super_param, dict):
            mask &= make_mask(super_param["where"], super_param["what"])

        mask_idx = np.where(mask)[0]

        # Filter and flatten indices
        idx = mask_idx.tolist()
        filtered_profile_functions = np.array(self.profile_functions)[mask_idx]
        filtered_profile_names = np.array(self.profile_names)[mask_idx]
        filtered_profile_params_index_list = np.array(
            self.profile_params_index_list, dtype=object
        )[mask_idx]
        profile_params_index_flat = np.concatenate(filtered_profile_params_index_list)

        # Convert parameter arrays once
        params_arr = np.asarray(self.params)
        filtered_params = params_arr[:, profile_params_index_flat]
        filtered_u_params = np.asarray(self.uncertainty_params)[:, profile_params_index_flat]

        # if hasattr(self, "uncertainty_params"):
        #
        # else:
        #     filtered_u_params = None

        combined_profile_func = combine_auto(filtered_profile_functions)

        result = LineSelectionResult(
            idx=idx,
            line_name=attributes_dict["line_name"][mask_idx],
            region=attributes_dict["region"][mask_idx].tolist(),
            center=attributes_dict["center"][mask_idx].astype(float).tolist(),
            kind=attributes_dict["kind"][mask_idx].tolist(),
            original_centers=np.array([e.center for e in entries])[mask_idx],
            component=attributes_dict["component"][mask_idx].tolist(),
            lines=np.array(entries)[mask_idx].tolist(),
            profile_functions=filtered_profile_functions,
            profile_names=filtered_profile_names,
            profile_params_index_flat=profile_params_index_flat,
            profile_params_index_list=filtered_profile_params_index_list,
            params_names=np.array(list(self.params_dict.keys()))[
                profile_params_index_flat.astype(int)
            ],
            params=filtered_params,
            uncertainty_params=filtered_u_params,
            profile_functions_combine=combined_profile_func,
        )

        self._last_filtered = result
        return result

