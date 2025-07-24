from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import jax.numpy as jnp



ArrayLike = Union[np.ndarray, jnp.ndarray]


# TODO Add multiple models to the reading.
def pad_error_channel(spectra: ArrayLike, frac: float = 0.01) -> ArrayLike:
    """Ensure *spectra* has a third channel (error) by padding with *frac* × signal."""
    if spectra.shape[1] != 2:
        return spectra  # already 3‑channel
    signal = spectra[:, 1, :]
    error = jnp.expand_dims(signal * frac, axis=1)
    return jnp.concatenate((spectra, error), axis=1)

