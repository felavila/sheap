
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd 

from sheap.DataClass.DataClass import SpectralLine
from sheap.Functions.utils import combine_auto


@dataclass
class ComplexRegion:
    """
    Holds SpectralLines + (optionally) their profile functions & parameters.
    You can slice/filter/group arbitrarily, and still recover both the
    original (“global”) and per‐subset (“local”) parameter mappings.
    """

    # --- required at init ---
    lines: List[SpectralLine]

    # --- optional; supply via attach_profiles() ---
    profile_functions:         List[Callable]            = field(default_factory=list)
    profile_names:             List[str]                 = field(default_factory=list)
    params_dict:               Dict[str, int]            = field(default_factory=dict)
    profile_params_index_list: List[List[int]]           = field(default_factory=list)
    params:                    Optional[np.ndarray]      = None
    uncertainty_params:        Optional[np.ndarray]      = None

    # --- internals (auto‐built) ---
    original_idx:              List[int]                 = field(init=False)
    _df:                       pd.DataFrame              = field(init=False, repr=False)
    _combined_func:            Optional[Callable]        = field(init=False, repr=False)

    # --- NEW: master list of all param‐names in global order ---
    _master_param_names:       List[str]                 = field(init=False, default_factory=list)
    # --- track the *original* index‐lists, never overwritten by subsets ---
    global_profile_params_index_list: List[List[int]]    = field(init=False, default_factory=list)

    def __post_init__(self):
        # 1) record each line's original position
        self.original_idx = list(range(len(self.lines)))

        # 2) if we already have a params_dict, stash its keys in order
        if self.params_dict:
            self._master_param_names = list(self.params_dict.keys())
            # and record the original full index‐lists
            self.global_profile_params_index_list = [
                lst.copy() for lst in self.profile_params_index_list
            ]

        # 3) build metadata DF: local index = .index, orig_idx column
        fallback = [ln.profile for ln in self.lines]
        prof_names = self.profile_names or fallback
        rows = []
        for i, ln in enumerate(self.lines):
            rows.append({
                "orig_idx":     self.original_idx[i],
                "line_name":    ln.line_name,
                "region":       ln.region,
                "kind":         ln.kind,
                "component":    ln.component,
                "profile_name": prof_names[i],
            })
        self._df = pd.DataFrame(rows)

        # 4) pre‐combine if profiles exist
        self._combined_func = (
            combine_auto(self.profile_functions)
            if self.profile_functions else None
        )

    def attach_profiles(
        self,
        profile_functions: List[Callable],
        profile_names:     List[str],
        params:            np.ndarray,
        uncertainty_params: np.ndarray,
        profile_params_index_list: List[List[int]],
        params_dict:       Dict[str, int],
    ) -> None:
        """
        Supply the full fit‐machinery.  Must provide exactly one profile
        & name per line, and a params_dict mapping each param_name->col.
        """
        N = len(self.lines)
        if not (len(profile_functions) == len(profile_names) == N):
            raise ValueError("Need exactly one profile per line")

        self.profile_functions               = profile_functions
        self.profile_names                   = profile_names
        self.params                          = params
        self.uncertainty_params              = uncertainty_params
        self.profile_params_index_list       = [lst.copy() for lst in profile_params_index_list]
        self.params_dict                     = params_dict

        # record master list of all param‐names in the order of params_dict.keys()
        self._master_param_names             = list(params_dict.keys())

        # record the original global index-lists once and for all
        self.global_profile_params_index_list = [lst.copy() for lst in profile_params_index_list]

        # rebuild combined profile
        self._combined_func = combine_auto(self.profile_functions)

        # update DF’s profile_name column
        self._df["profile_name"] = self.profile_names

    @property
    def combined_profile(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if self._combined_func is None:
            raise RuntimeError("No profiles attached")
        return self._combined_func

    @property
    def flat_param_indices_global(self) -> np.ndarray:
        """
        All the *global* parameter columns (original indices), in order.
        """
        if not self.global_profile_params_index_list:
            return np.array([], dtype=int)
        return np.concatenate(self.global_profile_params_index_list).astype(int)

    @property
    def flat_param_indices_local(self) -> np.ndarray:
        """
        All the *local* parameter columns (subset indices), in order.
        """
        if not self.profile_params_index_list:
            return np.array([], dtype=int)
        return np.concatenate(self.profile_params_index_list).astype(int)

    def as_df(self) -> pd.DataFrame:
        """Local‐index DataFrame with columns including orig_idx, kind, component, etc."""
        return self._df.copy()

    def filter(self, **conds) -> "ComplexRegion":
        mask = np.ones(len(self.lines), dtype=bool)
        for k, v in conds.items():
            if k not in self._df.columns:
                raise KeyError(f"No metadata column {k!r}")
            col = self._df[k].values
            mask &= np.isin(col, v) if isinstance(v, (list,tuple,np.ndarray)) else (col == v)
        return self._subset(mask)

    def _subset(self, mask: np.ndarray) -> "ComplexRegion":
        # slice the lines + original indices
        lines2    = [ln for ln, keep in zip(self.lines, mask) if keep]
        orig2     = [oi for oi, keep in zip(self.original_idx, mask) if keep]

        # slice the DF & reset local index
        df2 = self._df[mask].reset_index(drop=False)
        df2.rename(columns={"index": "local_idx"}, inplace=True)
        df2["orig_idx"] = orig2

        # if no profiles attached, return minimal
        if not self.profile_functions:
            new = ComplexRegion(lines=lines2)
            new.original_idx = orig2
            new._df = df2
            return new

        # slice profiles + names
        funcs2 = [f for f, keep in zip(self.profile_functions,   mask) if keep]
        names2 = [n for n, keep in zip(self.profile_names,       mask) if keep]

        # build subset of the *global* index‐lists
        glob_lists2 = [
            self.global_profile_params_index_list[i]
            for i, keep in enumerate(mask) if keep
        ]
        flat_global = np.concatenate(glob_lists2).astype(int)

        # slice the *original* params by global indices
        params2 = self.params[:, flat_global]
        u2      = self.uncertainty_params[:, flat_global]

        # build local map: global→new‐local
        local_map = { g: i for i, g in enumerate(flat_global) }
        local_lists2 = [[ local_map[g] for g in lst ] for lst in glob_lists2]

        # rebuild the subsetted params_dict from master names
        master = np.array(self._master_param_names)
        names_global = master[flat_global]
        filtered_dict2 = { nm: i for i, nm in enumerate(names_global) }

        # assemble the child
        new = ComplexRegion(
            lines=lines2,
            profile_functions=funcs2,
            profile_names=names2,
            params_dict=filtered_dict2,
            profile_params_index_list=local_lists2,
            params=params2,
            uncertainty_params=u2,
        )
        new._master_param_names             = self._master_param_names
        new.global_profile_params_index_list = glob_lists2
        new.original_idx                     = orig2
        new._df                              = df2
        new._combined_func                   = combine_auto(funcs2)
        return new

    def __getitem__(self, key: Union[int, slice, np.ndarray, List[int]]) -> "ComplexRegion":
        mask = (np.zeros(len(self.lines), bool) if not isinstance(key,int)
                else np.zeros(len(self.lines), bool))
        mask[key] = True
        return self._subset(mask)

    def group_by(self, field: str) -> Dict[Any, "ComplexRegion"]:
        if field not in self._df.columns:
            raise KeyError(f"No metadata column {field!r}")
        return {
            val: self.filter(**{field: val})
            for val in np.unique(self._df[field].values)
        }

    def param_subdict(self) -> Dict[str, np.ndarray]:
        """
        Map each param name → its column in this instance’s params,
        using the *local* flattened indices.
        """
        names = np.array(list(self.params_dict.keys()))
        return {nm: self.params[:, idx] for nm, idx in self.params_dict.items()}

    def unique(self, field: str) -> List[Any]:
        if field not in self._df.columns:
            raise KeyError(f"No metadata column {field!r}")
        return sorted(pd.unique(self._df[field].dropna()).tolist())

    @property
    def kinds(self) -> List[Any]:
        return self.unique("kind")

    @property
    def components(self) -> List[Any]:
        return self.unique("component")

    @property
    def regions(self) -> List[Any]:
        return self.unique("region")

    @property
    def profile_names_list(self) -> List[Any]:
        return self.unique("profile_name")

    def characteristics(self) -> Dict[str, List[Any]]:
        return {
            "kinds":         self.kinds,
            "components":    self.components,
            "regions":       self.regions,
            "profile_names": self.profile_names_list,
        }
