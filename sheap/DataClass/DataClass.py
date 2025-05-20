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

@dataclass
class FitResult:
    params: jnp.ndarray
    uncertainty_params: jnp.ndarray
    mask: jnp.ndarray
    profile_functions: List[Callable]
    profile_names: List[str]
    loss: List 
    profile_params_index_list: List
    initial_params:jnp.ndarray
    max_flux: jnp.ndarray
    params_dict: Dict[str, int]
    complex_region: List[SpectralLine]
    outer_limits: List
    inner_limits: List
    model_keywords: Optional[dict] = None
    
    

@dataclass
class ComplexRegion:
    complex_region: List[SpectralLine]
    profile_functions: List[Callable]
    params: np.ndarray
    uncertainty_params: np.ndarray
    profile_params_index_list: np.ndarray
    params_dict: Dict
    profile_names: List[str]

    kind_list: List[str] = field(init=False)  # will be computed post-init

    def __post_init__(self):
        self.kind_list = list({line.kind for line in self.complex_region})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConstraintSet:
    init: List[float]
    upper: List[float]
    lower: List[float]
    profile: str
    param_names: List[str]


@dataclass
class FittingLimits:
    """
    Stores width and shift limits for a line component kind.

    Attributes:
        upper_width (float): Maximum velocity width (km/s).
        lower_width (float): Minimum velocity width (km/s).
        center_shift (float): Maximum center shift (km/s).
        max_amplitude (float): Maximum allowed amplitude.
    """

    upper_width: float
    lower_width: float
    center_shift: float
    max_amplitude: float

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "FittingLimits":
        """
        Create FittingLimits from a dictionary with keys matching the attributes.

        Args:
            d (Dict[str, float]): Dictionary with keys:
                'upper_width', 'lower_width', 'center_shift', 'max_amplitude'.

        Returns:
            FittingLimits: Instance created from the dictionary.

        Raises:
            ValueError: If any required key is missing from the dictionary.
        """
        required_keys = {'upper_width', 'lower_width', 'center_shift', 'max_amplitude'}
        missing = required_keys - d.keys()
        if missing:
            raise ValueError(f"Missing keys for FittingLimits: {missing}")

        return cls(
            upper_width=d['upper_width'],
            lower_width=d['lower_width'],
            center_shift=d['center_shift'],
            max_amplitude=d['max_amplitude'],
        )
