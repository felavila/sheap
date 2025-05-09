from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union



@dataclass
class ConstraintSet:
    init: List[float]
    upper: List[float]
    lower: List[float]
    profile: str
    param_names: List[str]
    

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
            max_amplitude=d['max_amplitude']
        )