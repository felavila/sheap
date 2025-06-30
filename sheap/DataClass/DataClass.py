"""
Classes in this module:

- SpectralLine: Describes a spectral line with optional metadata.
- FitResult: Stores results from spectral region fitting.
- LineSelectionResult: Output of a spectral line filtering routine.
- ConstraintSet: Encodes parameter constraints and profile metadata.
- FittingLimits: Constraint configuration for component fitting.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd 


@dataclass
class SpectralLine:
    """
    Represents a single spectral emission or absorption line component.

    Attributes:
        center (float or list of floats): Central wavelength(s) of the line in Angstroms.
        line_name (str or list of str): Identifier(s) for the spectral line (e.g., 'Halpha') or the case of composite spectral line the name ofthe region + comp number.
        region (str): spacial region of the line 'narrow', 'broad', 'outflow', or 'fe'.
        subregion (str): element and spacial region combination in general usefulll for fe. components in models 
        component (int): Integer identifier for the component number within its kind.
        amplitude (float or list of floats, default=1.0): Initial or fixed amplitude for the line.
        how (Optional[str]): Method to handle the line (e.g., 'template', 'sum'). Usually used for Fe templates.
        element (Optional[str]): quimical stuff of the line.
        profile (Optional[str]): Profile function name (e.g., 'gaussian', 'lorentzian').
        which_template (Optional[str]): Sub-template or subtype for complex profiles (e.g., 'OP' or 'UV' for FeII templates).
        region_lines (Optional[List[str]]): Explicit list of line names used in a sum or composite region.
        amplitude_relations (Optional[List[List]]): Parameter tying or scaling definitions, typically for ratios.
        subprofile (str):  Sub-profile function to use within compound models like.
        rarity the line is common? or uncommon 
    """
    line_name: Union[str, List[str]]
    center: Optional[Union[float, List[float]]] = None 
    region: Optional[str] = None
    component: Optional[int] = None
    subregion: Optional[str] = None
    amplitude: Union[float, List[float]] = None
    element: Optional[str] = None
    how: Optional[str] = None #this maybe can be remove.

    profile: Optional[str] = None
    which_template: Optional[str] = None
    region_lines: Optional[List[str]] = None
    amplitude_relations: Optional[List[List]] = None
    subprofile: Optional[str] = None  
    rarity: Union[str, List[str]] =  None
    def to_dict(self) -> dict:
        """Convert the SpectralLine to a dictionary."""
        return asdict(self)



#still useffull? 
@dataclass
class FitResult:
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
    complex_region: List[SpectralLine]
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
    dependencies:Optional[List] = None # list tuple in reality
    #kind_list: List[str] = field(init=False)
    #def __post_init__(self):
     #   self.kind_list = list({line.kind for line in self.complex_region})

    def to_dict(self) -> dict:
        return asdict(self)

 

#This will go to the fitting part in particular to some helper.py stuff

@dataclass
class ConstraintSet:
    init: List[float]
    upper: List[float]
    lower: List[float]
    profile: str
    param_names: List[str]

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
    Stores fwhm and shift limits for a line component kind.

    Attributes:
        upper_fwhm (float): Maximum velocity fwhm (km/s).
        lower_fwhm (float): Minimum velocity fwhm (km/s).
        center_shift (float): Maximum center shift (km/s).
        max_amplitude (float): Maximum allowed amplitude.
    """

    upper_fwhm: float
    lower_fwhm: float
    center_shift: float
    max_amplitude: float

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "FittingLimits":
        """
        Create FittingLimits from a dictionary with keys matching the attributes.

        Args:
            d (Dict[str, float]): Dictionary with keys:
                'upper_fwhm', 'lower_fwhm', 'center_shift', 'max_amplitude'.

        Returns:
            FittingLimits: Instance created from the dictionary.

        Raises:
            ValueError: If any required key is missing from the dictionary.
        """
        required_keys = {'upper_fwhm', 'lower_fwhm', 'center_shift', 'max_amplitude'}
        missing = required_keys - d.keys()
        if missing:
            raise ValueError(f"Missing keys for FittingLimits: {missing}")

        return cls(
            upper_fwhm=d['upper_fwhm'],
            lower_fwhm=d['lower_fwhm'],
            center_shift=d['center_shift'],
            max_amplitude=d['max_amplitude'],
        )
