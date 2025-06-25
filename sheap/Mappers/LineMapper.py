# from dataclasses import dataclass
# from typing import Any, Callable, Dict, List, Optional, Union

# import jax.numpy as jnp
# import numpy as np
# import pandas as pd
# import yaml
# from jax import jit, vmap

# from sheap.DataClass.DataClass import SpectralLine
# from sheap.DataClass.test_class import LineSelectionResult
# from sheap.DataClass.utils import is_list_of
# from sheap.Functions.utils import combine_auto


# from dataclasses import dataclass, field
# from typing import (
#     Any, Callable, Dict, List, Optional, Union
# )

# import numpy as np
# import pandas as pd

# from sheap.DataClass.DataClass import SpectralLine
# from sheap.Functions.utils import combine_auto


# @dataclass
# class ComplexRegion:
#     """
#     Holds a list of SpectralLine objects and, optionally, all the associated
#     profile functions and parameter data.  Supports slicing, filtering,
#     grouping, and metadata introspection.
#     """

#     # --- required at init ---
#     lines: List[SpectralLine]

#     # --- optional fit machinery (attach later via attach_profiles) ---
#     profile_functions: List[Callable]              = field(default_factory=list)
#     profile_names:     List[str]                   = field(default_factory=list)
#     params_dict:       Dict[str, Any]              = field(default_factory=dict)
#     profile_params_index_list: List[List[int]]     = field(default_factory=list)
#     params:            Optional[np.ndarray]        = None
#     uncertainty_params: Optional[np.ndarray]       = None

#     # --- internals (auto‐built) ---
#     _df:               pd.DataFrame                = field(init=False, repr=False)
#     _combined_func:    Optional[Callable]          = field(init=False, repr=False)

#     def __post_init__(self):
#         # if user didn't supply any profile_names, use each line.profile
#         fallback = [ln.profile for ln in self.lines]
#         prof_names = self.profile_names or fallback

#         rows = []
#         for i, ln in enumerate(self.lines):
#             rows.append({
#                 "idx":          i,
#                 "line_name":    ln.line_name,
#                 "region":       ln.region,
#                 "kind":         ln.kind,
#                 "component":    ln.component,
#                 "profile_name": prof_names[i],
#             })
#         self._df = pd.DataFrame(rows)

#     def attach_profiles(
#         self,
#         profile_functions: List[Callable],
#         profile_names: List[str],
#         params: np.ndarray,
#         uncertainty_params: np.ndarray,
#         profile_params_index_list: List[List[int]],
#         params_dict: Dict[str, Any],
#     ) -> None:
#         """
#         Supply the profile functions, names, parameter arrays, and index lists.
#         Must have one function/name per line in self.lines.
#         """
#         N = len(self.lines)
#         if not (len(profile_functions) == len(profile_names) == N):
#             raise ValueError("Need exactly one profile per line")

#         self.profile_functions         = profile_functions
#         self.profile_names             = profile_names
#         self.params                    = params
#         self.uncertainty_params        = uncertainty_params
#         self.profile_params_index_list = profile_params_index_list
#         self.params_dict               = params_dict
#         self._combined_func            = combine_auto(self.profile_functions)

#         # update DataFrame's profile_name column
#         self._df["profile_name"] = self.profile_names

#     @property
#     def combined_profile(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
#         """Return f(x, all_params) summing every profile in this region."""
#         if self._combined_func is None:
#             raise RuntimeError("No profiles attached")
#         return self._combined_func

#     def as_df(self) -> pd.DataFrame:
#         """Return a copy of the metadata DataFrame."""
#         return self._df.copy()

#     def filter(self, **conds) -> "ComplexRegion":
#         """
#         Return a new ComplexRegion with only lines matching all conds.
#         E.g. region.filter(kind='broad', component=1).
#         """
#         mask = np.ones(len(self.lines), bool)
#         for key, val in conds.items():
#             if key not in self._df.columns:
#                 raise KeyError(f"No metadata column {key!r}")
#             col = self._df[key].values
#             if isinstance(val, (list, tuple, np.ndarray)):
#                 mask &= np.isin(col, val)
#             else:
#                 mask &= (col == val)
#         return self._subset(mask)

#     def _subset(self, mask: np.ndarray) -> "ComplexRegion":
#         """
#         Internal: slice lines, profiles, names, params, etc. by mask,
#         rebuild metadata DF and combined_func in the new object.
#         """
#         lines2 = [ln for ln, keep in zip(self.lines, mask) if keep]
#         df2    = self._df[mask].reset_index(drop=True)

#         # if no profiles attached, return minimal region
#         if not self.profile_functions:
#             new = ComplexRegion(lines=lines2)
#             new._df = df2
#             return new

#         funcs2 = [f for f, keep in zip(self.profile_functions, mask) if keep]
#         names2 = [n for n, keep in zip(self.profile_names,   mask) if keep]
#         idxs2  = [self.profile_params_index_list[i]
#                   for i, keep in enumerate(mask) if keep]

#         flat2 = np.concatenate(idxs2).astype(int)
#         params2 = self.params[:, flat2]
#         u2      = self.uncertainty_params[:, flat2]

#         # rebuild param‐name → column‐index mapping
#         all_names     = np.array(list(self.params_dict.keys()))
#         names2_params = all_names[flat2]
#         filtered_dict2= { nm: new_i for new_i, nm in enumerate(names2_params) }

        
#         new = ComplexRegion(
#             lines=lines2,
#             profile_functions=funcs2,
#             profile_names=names2,
#             params_dict= filtered_dict2,
#             profile_params_index_list=idxs2,
#             params=params2,
#             uncertainty_params=u2,
#         )
#         new._df = df2
#         new._combined_func = combine_auto(funcs2)
#         return new
    
    
#     def _subset_v2(self, mask: np.ndarray) -> "ComplexRegion":
#         """
#         Internal: slice lines, profiles, names, params, etc. by mask,
#         rebuild metadata DF and combined_func in the new object.
#         """
        
#         lines2 = [ln for ln, keep in zip(self.lines, mask) if keep]
#         df2    = self._df[mask].reset_index(drop=True)


#         orig_lists = [
#             self.profile_params_index_list[i]
#             for i, keep in enumerate(mask) if keep
#         ]

#         # flatten them to get the columns we’ll keep:
#         flat_global = np.concatenate(orig_lists).astype(int)

#         # slice your big params → small params2
#         params2 = self.params[:, flat_global]
#         u2      = self.uncertainty_params[:, flat_global]

#         if not self.profile_functions:
#             new = ComplexRegion(lines=lines2)
#             new._df = df2
#             return new
#         funcs2 = [f for f, keep in zip(self.profile_functions, mask) if keep]
#         names2 = [n for n, keep in zip(self.profile_names,   mask) if keep]
#         idxs2  = [self.profile_params_index_list[i]
#                   for i, keep in enumerate(mask) if keep]
#         # now build a mapping from each global index → its new local index
#         local_map = { g_idx: new_i for new_i, g_idx in enumerate(flat_global) }

#         # rebuild the per‐line index lists in local coordinates:
#         new_idx_lists = [
#             [ local_map[g_idx] for g_idx in line_list ]
#             for line_list in orig_lists
#         ]

#         # rebuild param‐name → local‐column mapping
#         all_names    = np.array(list(self.params_dict.keys()))
#         names_global = all_names[flat_global]        # e.g. ['amp0_broad', 'shift_broad', …]
#         filtered_dict2 = { nm: i for i, nm in enumerate(names_global) }

#         new = ComplexRegion(
#             lines=lines2,
#             profile_functions=funcs2,
#             profile_names=names2,
#             params_dict=filtered_dict2,
#             profile_params_index_list=new_idx_lists,
#             params=params2,
#             uncertainty_params=u2,
#         )
#         # copy internals…
#         new._df            = df2
#         new._combined_func = combine_auto(funcs2)
#         return new

#     def __getitem__(self, key: Union[int, slice, np.ndarray, List[int]]) -> "ComplexRegion":
#         """NumPy‐style slicing: region[2:5], region[mask], or region[3]."""
#         if isinstance(key, int):
#             mask = np.zeros(len(self.lines), bool)
#             mask[key] = True
#         else:
#             mask = np.zeros(len(self.lines), bool)
#             mask[key] = True
#         return self._subset(mask)

#     def group_by(self, field: str) -> Dict[Any, "ComplexRegion"]:
#         """
#         Partition into sub-regions by unique values of `field`,
#         e.g. region.group_by('component')[1].
#         """
        
#         if field not in self._df.columns:
#             raise KeyError(f"No metadata column {field!r}")
#         out: Dict[Any, ComplexRegion] = {}
#         for val in np.unique(self._df[field].values):
#             out[val] = self.filter(**{field: val})
#         return out

#     def param_subdict(self) -> Dict[str, np.ndarray]:
#         """
#         Map each parameter name to its column of self.params,
#         based on the current profile_params_index_list.
#         """
#         flat = np.concatenate(self.profile_params_index_list).astype(int)
#         names = np.array(list(self.params_dict.keys()))[flat]
#         return {nm: self.params[:, i] for i, nm in enumerate(names)}

#     # ─── Metadata introspection ───────────────────────────────────────────────────

#     def unique(self, field: str) -> List[Any]:
#         """
#         Return sorted unique non‐None values in metadata column `field`.
#         """
#         if field not in self._df.columns:
#             raise KeyError(f"No metadata column {field!r}")
#         # drop None, then get unique via pandas
#         vals = pd.unique(self._df[field].dropna())
#         return sorted(vals.tolist())

#     @property
#     def kinds(self) -> List[Any]:
#         """All unique `kind` values."""
#         return self.unique("kind")

#     @property
#     def components(self) -> List[Any]:
#         """All unique `component` values."""
#         return self.unique("component")

#     @property
#     def regions(self) -> List[Any]:
#         """All unique `region` values."""
#         return self.unique("region")

#     @property
#     def profile_names_list(self) -> List[Any]:
#         """All unique `profile_name` values."""
#         return self.unique("profile_name")
#     @property
#     def flat_param_indices(self) -> np.ndarray:
#         """
#         A 1-D numpy array of every parameter-column index for this region,
#         in the same order you’d slice self.params.
#         """
#         if not self.profile_params_index_list:
#             return np.array([], dtype=int)
#         return np.concatenate(self.profile_params_index_list).astype(int)
    
#     def characteristics(self) -> Dict[str, List[Any]]:
#         """
#         Bundle of all important metadata → their unique values.
#         """
#         return {
#             "kinds":          self.kinds,
#             "components":     self.components,
#             "regions":        self.regions,
#             "profile_names":  self.profile_names_list,
#         }


# class LineMapper:
#     """
#     Filters and maps spectral line entries based on attribute conditions.
#     Returns a sliceable LineSelectionResult that you can further .filter() or slice.
    
#     """

#     def __init__(
#         self,
#         complex_region: List[Union[SpectralLine, dict]],
#         profile_functions: List[Any],
#         params: List[Any],
#         uncertainty_params: List[Any],
#         profile_params_index_list: List[List[int]],
#         params_dict: Dict[str, Any],
#         profile_names: List[str],
#         kinds_list:List[str]= None,
#     ):
#         # normalize to SpectralLine objects
#         if is_list_of(complex_region, SpectralLine):
#             self.complex_region = complex_region
#         elif is_list_of(complex_region, dict):
#             self.complex_region = [SpectralLine(**d) for d in complex_region]
#         else:
#             raise TypeError("complex_region must be list of SpectralLine or dicts")
#         self.kinds_list = kinds_list
#         self.profile_functions = profile_functions
#         self.params = np.asarray(params)
#         self.uncertainty_params = np.asarray(uncertainty_params)
#         self.profile_params_index_list = profile_params_index_list
#         self.params_dict = params_dict
#         self.profile_names = profile_names
#         if self.kinds_list:
#             self.make_kinds_dict = self.make_kinds_dict()
#         self._last_filtered: Optional[LineSelectionResult] = None

#     def select(
#         self,
#         where: Union[str, List[str]],
#         what: Union[str, List[str]],
#         logic: str = "and",
#         super_param: Optional[Dict[str, str]] = None,
#     ) -> LineSelectionResult:
#         """
#         Core selection: just like your old _get, but returns our enhanced LineSelectionResult.
#         """
#         entries = self.complex_region
#         n = len(entries)

#         # collect attributes
#         attrs = {k: [getattr(e, k) for e in entries] for k in ("region", "center", "kind", "component", "line_name")}

#         def mask_for(key: str, val: str):
#             return np.array([val in str(x) for x in attrs[key]])

#         if isinstance(where, str):
#             where = [where]
#             what = [what]  # mypy happy
        
#         assert len(where) == len(what), "`where` and `what` lengths differ"

#         if logic == "and":
#             mask = np.ones(n, bool)
#         else:
#             mask = np.zeros(n, bool)

#         for w, v in zip(where, what):
#             m = mask_for(w, v)
#             mask = mask & m if logic == "and" else mask | m

#         if super_param:
#             mask &= mask_for(super_param["where"], super_param["what"])

#         idxs = np.nonzero(mask)[0].tolist()

#         # pull out everything
#         prof_funcs = [self.profile_functions[i] for i in idxs]
#         prof_names = [self.profile_names[i] for i in idxs]
#         idx_lists = [self.profile_params_index_list[i] for i in idxs]
#         flat_idx = np.concatenate(idx_lists).astype(int)

#         filtered_params = self.params[:, flat_idx]
#         filtered_uparams = self.uncertainty_params[:, flat_idx]
#         combined = combine_auto(prof_funcs)
#         #print(prof_names)
#         # build the dataclass
        
#         result = LineSelectionResult(
#             idx= idxs,
#             line_name= np.atleast_1d([attrs["line_name"][i] for i in idxs]),
#             region= np.atleast_1d([attrs["region"][i] for i in idxs]),
#             kind=[attrs["kind"][i] for i in idxs],
#             component=[attrs["component"][i] for i in idxs],
#             params_idx_lists = idx_lists, #list of the params separated by profile
#             #flat_idx = flat_idx
#             profile_names=prof_names,
#             complex=np.array(entries)[mask],
#          )
#         self._last_filtered = result
#         return result

#     # --- convenience wrappers ---

#     def by_kind(self, kind: Union[str, List[str]], logic: str = "or") -> LineSelectionResult:
#         return self.select("kind", kind, logic)

#     def by_region(self, region: Union[str, List[str]], logic: str = "or") -> LineSelectionResult:
#         return self.select("region", region, logic)

#     def by_component(self, component: Union[int, str, List[Union[int, str]]], logic: str = "or") -> LineSelectionResult:
#         # convert everything to str so mask works
#         comps = [str(component)] if not isinstance(component, (list, tuple)) else [str(c) for c in component]
#         return self.select("component", comps, logic)

#     def by_profile_name(self, name: Union[str, List[str]], logic: str = "or") -> LineSelectionResult:
#         return self.select("line_name", name, logic)

#     def last(self) -> Optional[LineSelectionResult]:
#         """Return the last selection, if any."""
#         return self._last_filtered
    
#     def make_kinds_dict(self):
#         self.kinds_dict = {}
#         for k in self.kinds_list:
#             self.kinds_dict[k] = self.select(where="kind", what=k)


# #dont fully agree with the name 
# class LineMapper_old_stable:
#     """
#     Filters and maps spectral line entries based on attribute conditions.
#     Also supports combining selected profile functions using JAX for efficient evaluation.

#     Attributes:
#         region_defs: List of Line or SpectralLine entries.
#         profile_functions: List of associated profile functions.
#         initial_params: Initial parameters for each line.
#         profile_params_index_list: Index mapping for profile parameters.
#     """

#     def __init__(
#         self,
#         complex_region: List[SpectralLine],
#         profile_functions: List[Any],
#         params: List[Any],
#         uncertainty_params: List[Any],
#         profile_params_index_list: List[List[int]],
#         params_dict,
#         profile_names,

#     ):

#         if is_list_of(complex_region, SpectralLine):
#             self.complex_region = complex_region
#         elif is_list_of(complex_region, dict):
#             self.complex_region = [SpectralLine(**i) for i in complex_region]
#         else:
#             raise TypeError(
#                 "complex_region must be a list of SpectralLine instances or a list of dicts that can be unpacked into SpectralLine."
#             )

#         # self.complex_region = complex_region
#         self.profile_functions = profile_functions
#         self.params = params
#         self.profile_params_index_list = profile_params_index_list
#         self.params_dict = params_dict
#         self.profile_names = profile_names
#         self.uncertainty_params = uncertainty_params
#         self._last_filtered = {}  # cache for later use in combine_profiles()

#     def _get(
#         self,
#         where: Union[str, List[str]],
#         what: Union[str, List[str]],
#         logic: str = "and",
#         super_param: Optional[Dict[str, str]] = None,
#     ) -> LineSelectionResult:

#         entries = self.complex_region
#         n_entries = len(entries)

#         attributes_dict = {
#             key: [getattr(e, key) for e in entries]
#             for key in ["region", "center", "kind", "component", "line_name"]
#         }

#         # Build logic mask using lists
#         def make_mask(key: str, value: str):
#             return np.array([value in str(item) for item in attributes_dict[key]])

#         if isinstance(where, str):
#             where = [where]
#         if isinstance(what, str):
#             what = [what]
#         assert len(where) == len(what), "`where` and `what` must have same length."

#         mask = np.ones(n_entries, dtype=bool) if logic == "and" else np.zeros(n_entries, dtype=bool)
#         for w, v in zip(where, what):
#             current_mask = make_mask(w, v)
#             mask = mask & current_mask if logic == "and" else mask | current_mask

#         if isinstance(super_param, dict):
#             mask &= make_mask(super_param["where"], super_param["what"])

#         mask_idx = np.where(mask)[0]

#         # Filter and flatten indices
#         idx = mask_idx.tolist()
#         filtered_profile_functions = [self.profile_functions[i] for i in mask_idx]
#         filtered_profile_names = [self.profile_names[i] for i in mask_idx]
#         filtered_profile_params_index_list = [self.profile_params_index_list[i] for i in mask_idx]

#         # Flatten params index safely
#         profile_params_index_flat = np.concatenate(filtered_profile_params_index_list)

#         # Convert parameter arrays once
#         params_arr = np.asarray(self.params)
#         filtered_params = params_arr[:, profile_params_index_flat]
#         filtered_u_params = np.asarray(self.uncertainty_params)[:, profile_params_index_flat]

#         combined_profile_func = combine_auto(filtered_profile_functions)

#         result = LineSelectionResult(
#             idx=idx,
#             line_name= [attributes_dict["line_name"][i] for i in mask_idx],
#             region= [attributes_dict["region"][i] for i in mask_idx],
#             center= [attributes_dict["center"][i] for i in mask_idx],
#             kind= [attributes_dict["kind"][i] for i in mask_idx],
#             original_centers=[entries[i].center for i in mask_idx],
#             component=[attributes_dict["component"][i] for i in mask_idx],
#             lines=[entries[i] for i in mask_idx],
#             profile_functions=filtered_profile_functions,
#             profile_names=filtered_profile_names,
#             profile_params_index_flat=profile_params_index_flat,
#             profile_params_index_list=filtered_profile_params_index_list,
#             params_names=np.array(list(self.params_dict.keys()))[
#                 profile_params_index_flat.astype(int)
#             ],
#             params=filtered_params,
#             uncertainty_params=filtered_u_params,
#             profile_functions_combine=combined_profile_func,
#             filtered_dict = {keys:i for keys,i in zip(np.array(list(self.params_dict.keys()))[
#                 profile_params_index_flat.astype(int)],profile_params_index_flat)},
#             complex = np.asarray(entries)[mask]
#             )
            
#         self._last_filtered = result
#         return result
