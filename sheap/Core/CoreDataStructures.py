"""
Core Data Structures
====================

This module defines the core data classes used across **sheap** to describe
spectral lines, grouped regions, fitting outputs, per‑profile constraints,
and per‑kind fitting limits.

Exposed classes
---------------
- :class:`SpectralLine` — a single (or composite) emission/absorption component.
- :class:`SheapModel` — a container of lines with profile functions, parameter
  maps, and convenient subsetting/grouping utilities.
- :class:`SheapResult` — a structured record of a completed fit (parameters,
  uncertainties, residuals, χ², etc.).
- :class:`ProfileConstraintSet` — per‑profile initial values and bounds.
- :class:`FittingLimits` — canonical velocity/shift/amplitude limits by kind.

Main Features
-------------
- Dataclass APIs with typed fields and `.to_dict()` helpers.
- Region‑level table view (`as_df()`), filtering, grouping, and safe subsetting
  that preserves global↔local parameter index mappings.
- Lazy assembly of fused profile functions for fast evaluation with JAX.
- Seamless attachment of fitted parameter matrices and uncertainties.

Notes
-----
- Arrays may be NumPy or JAX arrays; fused profile evaluation is JAX‑friendly.
- Global parameter indices are preserved when subsetting regions so results can
  be traced back to the original packed parameter vector.
"""

from __future__ import annotations
__author__ = 'felavila'


__all__ = [
    "SheapModel",
    "SheapResult",
    "FittingLimits",
    "ProfileConstraintSet",
    "SpectralLine",
]

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import jax.numpy as jnp
import numpy as np
import pandas as pd 

from sheap.Profiles.Utils import make_fused_profiles


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

#this still require a few changes 
@dataclass
class SheapModel:
    
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

    def filter(self, **conds) -> "SheapModel":
        mask = np.ones(len(self.lines), dtype=bool)
        for k, v in conds.items():
            if k not in self._df.columns:
                raise KeyError(f"No metadata column {k!r}")
            col = self._df[k].values
            mask &= np.isin(col, v) if isinstance(v, (list,tuple,np.ndarray)) else (col == v)
        return self._subset(mask)

    def _subset(self, mask: np.ndarray) -> "SheapModel":

        mask = np.asarray(mask, dtype=bool)

        if mask.shape != (len(self.lines),):
            raise ValueError(f"Expected mask shape {(len(self.lines),)}, "
                f"got {mask.shape}")

        selected = np.flatnonzero(mask)

        # 1. Slice lines and preserve original indices
        lines2 = [self.lines[i] for i in selected]
        orig2 = [self.original_idx[i] for i in selected]

        # 2. Slice metadata and reset local indices
        df2 = self._df.iloc[selected].copy().reset_index(drop=True)
        df2["local_idx"] = np.arange(len(selected))
        df2["orig_idx"] = orig2

        # 3. Handle models without attached profiles
        if not self.profile_functions or len(selected) == 0:
            new = SheapModel(lines=lines2)
            new.original_idx = orig2
            new._df = df2
            new._master_param_names = self._master_param_names.copy()
            return new

        # 4. Slice profiles and profile names
        funcs2 = [self.profile_functions[i] for i in selected]
        names2 = [self.profile_names[i] for i in selected]

        # 5. Preserve GLOBAL parameter indices
        glob_lists2 = [
            np.asarray(
                self.global_profile_params_index_list[i],
                dtype=int
            ).copy()
            for i in selected
        ]

        # 6. Retrieve LOCAL parameter indices of the parent model
        parent_local_lists2 = [np.asarray(self.profile_params_index_list[i],dtype=int) for i in selected]

        flat_global = np.concatenate(glob_lists2).astype(int)
        flat_local = np.concatenate(parent_local_lists2).astype(int)

        # 7. Validate indices before slicing
        # Important because JAX may silently clip invalid indices
        for name, array in (("params", self.params), ("uncertainty_params", self.uncertainty_params),):
            if array is not None:
                if np.any(flat_local < 0) or np.any(flat_local >= array.shape[1]):
                    raise IndexError(
                        f"Invalid local indices for {name}: "
                        f"array shape={array.shape}, "
                        f"indices={flat_local}"
                    )

        # 8. Slice using LOCAL indices
        # Compatible with both NumPy and JAX
        params2 = (self.params[:, flat_local] if self.params is not None else None)

        u2 = (self.uncertainty_params[:, flat_local] if self.uncertainty_params is not None else None)

        # 9. Build new local parameter indices
        local_lists2 = []
        offset = 0

        for indices in glob_lists2:
            n_params = len(indices)

            local_lists2.append(
                list(range(offset, offset + n_params))
            )

            offset += n_params

        # 10. Rebuild parameter dictionary using GLOBAL names
        master = np.asarray(self._master_param_names)
        names_global = master[flat_global]

        filtered_dict2 = {
            nm: i for i, nm in enumerate(names_global)
        }

        # 11. Assemble child model
        new = SheapModel(lines=lines2,
            profile_functions=funcs2,
            profile_names=names2,
            params_dict=filtered_dict2,
            profile_params_index_list=local_lists2,
            params=params2,
            uncertainty_params=u2,)

        # 12. Preserve global metadata
        new._master_param_names = self._master_param_names.copy()
        new.global_profile_params_index_list = glob_lists2
        new.original_idx = orig2
        new._df = df2

        return new

    def __getitem__(self, key: Union[int, slice, np.ndarray, List[int]]) -> "SheapModel":
        mask = (np.zeros(len(self.lines), bool) if not isinstance(key,int)
                else np.zeros(len(self.lines), bool))
        mask[key] = True
        return self._subset(mask)

    def group_by(self, field: str) -> Dict[Any, "SheapModel"]:
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
class SheapResult:
    """
    Container for the results of a SHEAP spectral-region fit.

    This class stores the optimized fit parameters, associated metadata, model
    configuration, diagnostics, posterior information, and the reconstructed
    :class:`SheapModel` object associated with a fitted spectral region.

    Parameters
    ----------
    region_list : list of SpectralLine
        List of spectral-line configurations used to define the fitted region.
        This is used to initialize the internal :class:`SheapModel`.

    fitting_routine : dict or None, optional
        Dictionary describing the fitting routine used to obtain the result.

    params : jax.numpy.ndarray or None, optional
        Optimized model parameters.

    uncertainty_params : jax.numpy.ndarray or None, optional
        Estimated uncertainties associated with ``params``.

    mask : jax.numpy.ndarray or None, optional
        Mask used during the fitting process. The convention should match the
        one used by the fitting routine.

    constraints : jax.numpy.ndarray or None, optional
        Constraint array used during the fit. This should correspond to the
        constraints generated by the fitting setup.

    profile_functions : list of callable or None, optional
        Profile functions used to evaluate the spectral model.

    profile_names : list of str or None, optional
        Names of the profile functions used in the model.

    loss : list or None, optional
        Loss values recorded during the optimization.

    profile_params_index_list : list or None, optional
        Index mapping between spectral profiles and their corresponding
        parameter positions.

    initial_params : jax.numpy.ndarray or None, optional
        Initial parameter values before optimization.

    scale : jax.numpy.ndarray or None, optional
        Normalization scale applied to the data or model during fitting.

    params_dict : dict of str to int or None, optional
        Mapping between parameter names and their index positions in the
        parameter array.

    outer_limits : list or None, optional
        Outer wavelength limits used to define the full fitting region.

    inner_limits : list or None, optional
        Inner wavelength limits used to define the primary region of interest.

    model_keywords : dict or None, optional
        Additional keyword arguments used to configure the spectral model.

    source : dict or None, optional
        Metadata describing the input source or spectrum.

    dependencies : list or None, optional
        List describing parameter dependencies, ties, or linked relations.

    free_params : jax.numpy.ndarray or None, optional
        Boolean or index array identifying the free parameters in the fit.

    residuals : jax.numpy.ndarray or None, optional
        Residuals between the observed spectrum and the fitted model.

    chi2_red : jax.numpy.ndarray or None, optional
        Reduced chi-squared value or values associated with the fit.

    posterior : dict or None, optional
        Posterior samples or posterior-summary information, when available.

    fitkwargs : list of dict or None, optional
        Keyword arguments used for one or more fitting steps.

    elapsed_time : list of dict or None, optional
        Timing information for the different fitting stages.

    params_class : object or None, optional
        Optional parameter-class object associated with the fit.

    Attributes
    ----------
    sheapmodel : SheapModel
        Model object initialized from ``region_list`` and populated with the
        fitted profiles, parameters, uncertainties, profile names, profile
        parameter indices, and parameter dictionary.

    Notes
    -----
    The :class:`SheapModel` instance is created automatically during
    ``__post_init__``. Profile information is attached using
    ``SheapModel.attach_profiles``.

    The ``to_dict`` method uses :func:`dataclasses.asdict`, so nested dataclass
    fields are recursively converted to dictionaries.
    """

    region_list: List[SpectralLine]
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
    source: Optional[dict] = None
    dependencies: Optional[List] = None
    free_params: Optional[jnp.ndarray] = None
    residuals: Optional[jnp.ndarray] = None
    chi2_red: Optional[jnp.ndarray] = None
    posterior: Optional[dict] = None
    fitkwargs: Optional[List[Dict]] = None
    elapsed_time: Optional[List[Dict]] = None
    params_class: Optional = None

    def __post_init__(self):
        self.sheapmodel = SheapModel(self.region_list)
        self.sheapmodel.attach_profiles(
            self.profile_functions,
            self.profile_names,
            self.params,
            self.uncertainty_params,
            self.profile_params_index_list,
            self.params_dict,
        )

    def to_dict(self) -> dict:
        """
        Convert the result object into a dictionary.

        Returns
        -------
        dict
            Dictionary representation of the dataclass fields.
        """
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


@dataclass
class FittingLimits:
    """
    Stores FWHM and shift limits for a line component kind.

    Attributes:
        upper_fwhm_kms (float): Maximum velocity FWHM (km/s).
        lower_fwhm_kms (float): Minimum velocity FWHM (km/s).
        vshift_kms (float): Maximum velocity shift (km/s).
        max_amplitude (float): Maximum allowed amplitude.
    """

    upper_fwhm_kms: float
    lower_fwhm_kms: float
    init_fwhm_kms:Optional[float] = None
    vshift_kms: Optional[float] = None
    max_amplitude:  Optional[float] = None
    references: Optional[list] = None 
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "FittingLimits":
        """
        Create FittingLimits from a dictionary with keys matching the attributes.

        Args:
            d (Dict[str, float]): Dictionary with keys:
                'upper_fwhm_kms', 'lower_fwhm_kms', 'vshift_kms', 'max_amplitude'.

        Returns:
            FittingLimits: Instance created from the dictionary.

        Raises:
            ValueError: If any required key is missing from the dictionary.
        """
        required_keys = {'upper_fwhm_kms', 'lower_fwhm_kms', 'vshift_kms', 'max_amplitude'}
        missing = required_keys - d.keys()
        if missing:
            raise ValueError(f"Missing keys for FittingLimits: {missing}")

        return cls(
            upper_fwhm_kms=d['upper_fwhm_kms']
            ,lower_fwhm_kms=d['lower_fwhm_kms'],
            init_fwhm_kms = d.get('init_fwhm_kms'),
            vshift_kms =d['vshift_kms'],
            max_amplitude=d['max_amplitude'],
            references=d.get('references'),
        )

