from typing import Callable, Dict, Optional, Tuple, Union,List

import numpy as np
import jax.numpy as jnp

from sheap.Core import SpectralLine

ArrayLike = Union[np.ndarray, jnp.ndarray]

# Signature: (x, params) -> profile output
ProfileFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]


SpectralLineList = List[SpectralLine]