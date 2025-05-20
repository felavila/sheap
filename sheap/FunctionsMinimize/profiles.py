from typing import Callable, Dict
import jax.numpy as jnp

# Define the signature for profile functions: (x, params) -> output
ProfileFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

# === Import your actual model functions here ===
from .functions import (
    gaussian_func,
    lorentzian_func,
    powerlaw,
    fitFeOP,
    fitFeUV,
    linear,
    balmerconti,
    brokenpowerlaw,Gsum_model
)

# === Dictionary mapping profile names to functions ===
PROFILE_FUNC_MAP: Dict[str, ProfileFunc] = {
    'gaussian': gaussian_func,
    'lorentzian': lorentzian_func,
    'powerlaw': powerlaw,
    'fitFeOP': fitFeOP,
    'fitFeUV': fitFeUV,
    'linear': linear,
    'balmerconti': balmerconti,
    'brokenpowerlaw': brokenpowerlaw,
    "Gsum_model":Gsum_model
}
