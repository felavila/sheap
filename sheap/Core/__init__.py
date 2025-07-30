"""Core data structures and typing definitions for spectral modeling."""

__version__ = '0.1.0'
__author__ = 'Felipe Avila-Vera'

from sheap.Core.Definitions import (SpectralLine,ComplexRegion,ComplexResult,ProfileConstraintSet,FittingLimits)
from sheap.Core.Signatures import (ArrayLike,ProfileFunc,SpectralLineList,)

__all__ = [
    "SpectralLine",
    "ComplexRegion",
    "ComplexResult",
    "ProfileConstraintSet",
    "FittingLimits",
    "ArrayLike",
    "ProfileFunc",
    "SpectralLineList",
]
