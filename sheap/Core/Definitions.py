"""This module handles ?."""
from __future__ import annotations
__author__ = 'felavila'


__all__ = [
    "ComplexRegion",
    "ComplexResult",
    "FittingLimits",
    "ProfileConstraintSet",
    "SpectralLine",
]

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import jax.numpy as jnp
import numpy as np
import pandas as pd 

from sheap.Profiles.utils import make_fused_profiles


@dataclass
class SpectralLine:
    """
    Represents a single spectral emission or absorption line component.

    Parameters
    ----------
    line_name : str or list of str
        Identifier(s) for the spectral line (e.g., 'Halpha'), or for a composite region
        the region name plus component number.
    center : float or list of float, optional
        Central wavelength(s) of the line in Angstroms.
    region : str, optional
        Spatial region of the line, one of 'narrow', 'broad', 'outflow', or 'fe'.
    component : int, optional
        Integer identifier for this component within its region.
    subregion : str, optional
        Element + spatial subregion tag, useful for complex templates (e.g. FeII sub‐regions).
    amplitude : float or list of float, optional
        Initial or fixed amplitude(s) for the line(s).
    element : str, optional
        Chemical identifier of the line (e.g., 'H', 'FeII').
    profile : str, optional
        Name of the profile function to use ('gaussian', 'lorentzian', etc.).
    region_lines : list of str, optional
        Explicit list of line identifiers included in a composite region.
    amplitude_relations : list of list, optional
        Parameter‐tying definitions (e.g. fixed ratios) among amplitudes.
    subprofile : str, optional
        Sub‐profile name for compound models (e.g. a secondary kernel).
    rarity : str or list of str, optional
        Qualitative frequency label for the line (e.g. 'common', 'rare').
    template_info : dict, optional
        Additional template metadata (e.g. for 'hostmiles' or 'fetemplate' profiles).

    Attributes
    ----------
    (all parameters become attributes of this dataclass)

    Methods
    -------
    to_dict()
        Convert the SpectralLine instance into a plain dictionary via `asdict`.

    Examples
    --------
    >>> line = SpectralLine(
    ...     line_name='Halpha',
    ...     center=6563.0,
    ...     region='narrow',
    ...     component=0,
    ...     profile='gaussian',
    ...     amplitude=1.0
    ... )
    >>> d = line.to_dict()
    >>> print(d['center'])
    6563.0
    """
    line_name: Union[str, List[str]]
    center: Optional[Union[float, List[float]]] = None 
    region: Optional[str] = None
    component: Optional[int] = None
    subregion: Optional[str] = None
    amplitude: Optional[Union[float, List[float]]] = None
    element: Optional[str] = None
    profile: Optional[str] = None
    region_lines: Optional[List[str]] = None
    amplitude_relations: Optional[List[List]] = None
    subprofile: Optional[str] = None  
    rarity: Optional[Union[str, List[str]]] = None
    template_info: Optional[Dict] = None

    def to_dict(self) -> dict:
        """
        Convert the SpectralLine to a dictionary.

        Returns
        -------
        dict
            A dict representation of all fields of the dataclass.
        """
        return asdict(self)


@dataclass
class ComplexRegion:
    
    """
    Holds SpectralLines + (optionally) their profile functions & parameters.
    You can slice/filter/group arbitrarily, and still recover both the
    original (“global”) and per‐subset (“local”) parameter mappings.
    """

    lines: List[SpectralLine]

    profile_functions:         List[Callable]            = field(default_factory=list)
    profile_names:             List[str]                 = field(default_factory=list)
    params_dict:               Dict[str, int]            = field(default_factory=dict)
    profile_params_index_list: List[List[int]]           = field(default_factory=list)
    params:                    Optional[np.ndarray]      = None
    uncertainty_params:        Optional[np.ndarray]      = None

    original_idx:              List[int]                 = field(init=False)
    _df:                       pd.DataFrame              = field(init=False, repr=False)
    _combined_func:            Optional[Callable]        = field(init=False, repr=False)


    _master_param_names:       List[str]                 = field(init=False, default_factory=list)
    
    global_profile_params_index_list: List[List[int]]    = field(init=False, default_factory=list)

    def __post_init__(self):
       
        self.original_idx = list(range(len(self.lines)))

        
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
                "subregion":         ln.subregion,
                "element":  ln.element,
                "component":    ln.component,
                "profile_name": prof_names[i],
                })
        self._df = pd.DataFrame(rows)

        # 4) pre‐combine if profiles exist
        self._combined_func = (
            make_fused_profiles(self.profile_functions)
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
        self._combined_func = make_fused_profiles(self.profile_functions)

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
        new._combined_func                   = make_fused_profiles(funcs2)
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
    def regions(self) -> List[Any]:
        return self.unique("region")
    @property
    def components(self) -> List[Any]:
        return self.unique("component")
    @property
    def subregions(self) -> List[Any]:
        return self.unique("subregion")
    @property
    def elements(self) -> List[Any]:
        return self.unique("element")

    def characteristics(self) -> Dict[str, Any]:
        by_region_component = (
            self._df.groupby("region")["component"]
            .nunique()
            .sort_index()
            .to_dict()
        )

        return {
            "components": self.components,
            "regions": self.regions,
            #"profile_names": self.profile_names_list,
            "elements": self.elements,
            "subregions": self.subregions,
            "n_components_per_region": by_region_component,
        }
        
#still useffull? 
@dataclass
class ComplexResult:
    """
    Data class to store results from spectral region fitting.

    Attributes:
        complex_region (List[SpectralLine]): List of spectral line configurations.
        params (Optional[jnp.ndarray]): Optimized parameters from fitting.
        uncertainty_params (Optional[jnp.ndarray]): Estimated uncertainties for each parameter.
        mask (Optional[jnp.ndarray]): Mask used during the fitting process.
        profile_functions (Optional[List[Callable]]): Functions describing each spectral profile.
        profile_names (Optional[List[str]]): Names of spectral profiles used in fitting.
        loss (Optional[List]): Values of the loss function during optimization.
        profile_params_index_list (Optional[List]): Indices mapping profile parameters.
        initial_params (Optional[jnp.ndarray]): Initial guess parameters before fitting.
        scale (Optional[jnp.ndarray]): scale used for normalization.
        params_dict (Optional[Dict[str, int]]): Mapping from parameter names to indices.
        outer_limits (Optional[List]): Outer wavelength limits of the fitting region.
        inner_limits (Optional[List]): Inner wavelength limits defining the region of interest.
        model_keywords (Optional[dict]): Additional keywords for model configuration.
        kind_list (List[str]): Unique types of spectral lines (computed post-init).
        constraints same as constrains from fit 
    """
    complex_region: List[SpectralLine] # can be mode to complex_class at the moment after reading.
    fitting_routine: Optional[dict] = None
    params: Optional[jnp.ndarray] = None
    uncertainty_params: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    constraints: Optional[jnp.ndarray] = None
    profile_functions: Optional[List[Callable]] = None
    profile_names: Optional[List[str]] = None
    loss: Optional[List] = None
    profile_params_index_list: Optional[List] = None
    initial_params: Optional[jnp.ndarray] = None
    scale: Optional[jnp.ndarray] = None
    params_dict: Optional[Dict[str, int]] = None
    outer_limits: Optional[List] = None
    inner_limits: Optional[List] = None
    model_keywords: Optional[dict] = None
    source:Optional[dict] = None
    dependencies:Optional[List] = None 
    free_params:Optional[jnp.ndarray] = None 
    residuals:Optional[jnp.ndarray] = None 
    chi2_red:Optional[jnp.ndarray] = None 
    posterior:Optional[dict] = None 
    fitkwargs:Optional[List[Dict]] = None 
    # list tuple in reality
    #kind_list: List[str] = field(init=False)
    def __post_init__(self):
        self.complex_class = ComplexRegion(self.complex_region)
        self.complex_class.attach_profiles(self.profile_functions,self.profile_names,self.params,self.uncertainty_params
                                    ,self.profile_params_index_list,self.params_dict)

    def to_dict(self) -> dict:
        return asdict(self)




@dataclass
class ProfileConstraintSet:
    init: List[float]
    upper: List[float]
    lower: List[float]
    profile: str
    param_names: List[str]
    profile_fn: Optional[Callable] = None 

    def __post_init__(self):
        # Skip length check for SPAF profiles
        if self.profile.startswith("SPAF"):
            return

        n = len(self.init)
        if not (len(self.upper) == len(self.lower) == len(self.param_names) == n):
            raise ValueError(
                f"ConstraintSet mismatch: "
                f"got init[{n}], upper[{len(self.upper)}], "
                f"lower[{len(self.lower)}], param_names[{len(self.param_names)}]"
            )

#This have to be a more flexible 
@dataclass
class FittingLimits:
    """
    Stores FWHM and shift limits for a line component kind.

    Attributes:
        upper_fwhm (float): Maximum velocity FWHM (km/s).
        lower_fwhm (float): Minimum velocity FWHM (km/s).
        v_shift (float): Maximum velocity shift (km/s).
        max_amplitude (float): Maximum allowed amplitude.
    """

    upper_fwhm: float
    lower_fwhm: float
    v_shift: Optional[float] = None
    max_amplitude:  Optional[float] = None
    canonical_wavelengths: Optional[float] = None 
    references: Optional[list] = None 
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "FittingLimits":
        """
        Create FittingLimits from a dictionary with keys matching the attributes.

        Args:
            d (Dict[str, float]): Dictionary with keys:
                'upper_fwhm', 'lower_fwhm', 'center_shift', 'v_shift', 'max_amplitude'.

        Returns:
            FittingLimits: Instance created from the dictionary.

        Raises:
            ValueError: If any required key is missing from the dictionary.
        """
        required_keys = {'upper_fwhm', 'lower_fwhm', 'v_shift', 'max_amplitude'}
        missing = required_keys - d.keys()
        if missing:
            raise ValueError(f"Missing keys for FittingLimits: {missing}")

        return cls(
            upper_fwhm=d['upper_fwhm'],
            lower_fwhm=d['lower_fwhm'],
            v_shift=d['v_shift'],
            max_amplitude=d['max_amplitude'],
            canonical_wavelengths = d["canonical_wavelengths"]
        )

