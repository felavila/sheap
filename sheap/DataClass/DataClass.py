from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np


@dataclass
class SpectralLine:
    center: Union[float, List[float]]
    line_name: Union[str, List[str]]
    kind: str
    component: int
    amplitude: Union[float, List[float]] = 1.0
    how: Optional[str] = None
    region: Optional[str] = None
    profile: Optional[str] = None
    which: Optional[str] = None

    def to_dict(self) -> dict:
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
        max_flux (Optional[jnp.ndarray]): Maximum flux used for normalization.
        params_dict (Optional[Dict[str, int]]): Mapping from parameter names to indices.
        outer_limits (Optional[List]): Outer wavelength limits of the fitting region.
        inner_limits (Optional[List]): Inner wavelength limits defining the region of interest.
        model_keywords (Optional[dict]): Additional keywords for model configuration.
        kind_list (List[str]): Unique types of spectral lines (computed post-init).
        constraints same as constrains from fit 
    """
    complex_region: List[SpectralLine]
    fitting_rutine: Optional[dict] = None
    params: Optional[jnp.ndarray] = None
    uncertainty_params: Optional[jnp.ndarray] = None
    mask: Optional[jnp.ndarray] = None
    constraints: Optional[jnp.ndarray] = None
    profile_functions: Optional[List[Callable]] = None
    profile_names: Optional[List[str]] = None
    loss: Optional[List] = None
    profile_params_index_list: Optional[List] = None
    initial_params: Optional[jnp.ndarray] = None
    max_flux: Optional[jnp.ndarray] = None
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
    

# @dataclass
# class FitResult:
#     params: jnp.ndarray
#     uncertainty_params: jnp.ndarray
#     mask: jnp.ndarray
#     profile_functions: List[Callable]
#     profile_names: List[str]
#     loss: List 
#     profile_params_index_list: List
#     initial_params:jnp.ndarray
#     max_flux: jnp.ndarray
#     params_dict: Dict[str, int]
#     complex_region: List[SpectralLine]
#     outer_limits: List
#     inner_limits: List
#     model_keywords: Optional[dict] = None
#     kind_list: List[str] = field(init=False)  # will be computed post-init

#     def __post_init__(self):
#         self.kind_list = list({line.kind for line in self.complex_region})

#     def to_dict(self) -> dict:
#         return asdict(self)
    

# @dataclass
# class ComplexRegion:
#     complex_region: List[SpectralLine]
#     profile_functions: List[Callable]
#     params: np.ndarray
#     uncertainty_params: np.ndarray
#     profile_params_index_list: np.ndarray
#     params_dict: Dict
#     profile_names: List[str]

#     kind_list: List[str] = field(init=False)  # will be computed post-init

#     def __post_init__(self):
#         self.kind_list = list({line.kind for line in self.complex_region})

#     def to_dict(self) -> dict:
#         return asdict(self)


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
