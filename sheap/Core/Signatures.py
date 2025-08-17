"""
Core type aliases and callable signatures used across sheap.

This module defines reusable typing primitives and function signatures
to support consistent and type-safe modeling across the codebase.

Attributes
----------
ArrayLike : type
    A generic array type that can be either a NumPy or JAX array.

ProfileFunc : Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    Signature for spectral profile functions. These functions take:
        - `x`: Wavelength grid (1D array)
        - `params`: Profile parameters (1D array)
    and return the evaluated model flux over `x`.

SpectralLineList : List[SpectralLine]
    Shorthand for a list of `SpectralLine` dataclass instances.
"""

__author__ = 'felavila'

# Auto-generated __all__
__all__ = [
    "ArrayLike",
    "ProfileFunc",
    "SpectralLineList",
]

from typing import Callable, List, Union
import numpy as np
import jax.numpy as jnp

from sheap.Core import SpectralLine

ArrayLike = Union[np.ndarray, jnp.ndarray]
"""Generic type alias for arrays that may be either NumPy or JAX ndarrays."""

ProfileFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
"""Callable signature for profile functions: (x, params) → model(x)."""

SpectralLineList = List[SpectralLine]
"""Convenient alias for a list of SpectralLine instances."""
