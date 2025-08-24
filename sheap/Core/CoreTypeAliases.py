r"""
Core Type Aliases
=================

Defines reusable typing primitives and callable signatures
used across *sheap* for consistency and type safety.

Contents
--------
- **ArrayLike** : Generic type alias for arrays (NumPy or JAX).
- **ProfileFunc** : Callable signature for spectral profile functions.
- **SpectralLineList** : Shorthand for a list of `SpectralLine`.

Notes
-----
- Ensures consistent typing across profiles, regions, and fitting routines.
- Profile functions follow the convention:

  .. math::
     f(x) = \mathrm{model}(x, \theta)

  where `x` is the wavelength grid and `θ` the parameter vector.
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
