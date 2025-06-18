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

@dataclass
class SpectralLine:
    """
    Represents a single spectral emission or absorption line component.

    Attributes:
        center (float or list of floats): Central wavelength(s) of the line in Angstroms.
        line_name (str or list of str): Identifier(s) for the spectral line (e.g., 'Halpha').
        kind (str): Component type, such as 'narrow', 'broad', 'outflow', or 'fe'.
        component (int): Integer identifier for the component number within its kind.
        amplitude (float or list of floats, default=1.0): Initial or fixed amplitude for the line.
        how (Optional[str]): Method to handle the line (e.g., 'template', 'sum'). Usually used for Fe templates.
        region (Optional[str]): Region label from the YAML template file or source.
        profile (Optional[str]): Profile function name (e.g., 'gaussian', 'lorentzian').
        which (Optional[str]): Sub-template or subtype for complex profiles (e.g., 'OP' or 'UV' for FeII templates).
        region_lines (Optional[List[str]]): Explicit list of line names used in a sum or composite region.
        amplitude_relations (Optional[List[List]]): Parameter tying or scaling definitions, typically for ratios.
        subprofile (str):  Sub-profile function to use within compound models like.
    """

    center: Union[float, List[float]]
    line_name: Union[str, List[str]]
    kind: str
    component: int
    amplitude: Union[float, List[float]] = 1.0
    how: Optional[str] = None
    region: Optional[str] = None
    profile: Optional[str] = None
    which: Optional[str] = None
    region_lines: Optional[List[str]] = None
    amplitude_relations: Optional[List[List]] = None
    subprofile: None = None  # not currently used or typed

    def to_dict(self) -> dict:
        """Convert the SpectralLine to a dictionary."""
        return asdict(self)
    
#     complex_region: List[SpectralLine]
#     profile_functions: List[Callable]
#     params: np.ndarray
#     uncertainty_params: np.ndarray
#     profile_params_index_list: np.ndarray
#     params_dict: Dict
#     profile_names: List[str]

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
    kind_list: List[str] = field(init=False)
    def __post_init__(self):
        self.kind_list = list({line.kind for line in self.complex_region})

    def to_dict(self) -> dict:
        return asdict(self)
    
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
    filtered_dict: Dict

@dataclass
class ConstraintSet:
    init: List[float]
    upper: List[float]
    lower: List[float]
    profile: str
    param_names: List[str]


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
