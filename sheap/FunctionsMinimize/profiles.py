from typing import Callable, Dict
import jax.numpy as jnp

# Define the signature for profile functions: (x, params) -> output
ProfileFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

from .functions.continiumm_profiles import (linear,balmerconti,powerlaw,brokenpowerlaw)
from .functions.lines_profiles import (gaussian_fwhm,lorentzian_fwhm,Gsum_model)
from .functions.template_func import (fitFeOP,fitFeUV)


PROFILE_FUNC_MAP: Dict[str, ProfileFunc] = {
    'linear': linear,
    'powerlaw': powerlaw,
    'balmerconti': balmerconti,
    'brokenpowerlaw': brokenpowerlaw,
    
    'gaussian': gaussian_fwhm,
    'lorentzian': lorentzian_fwhm,
    "Gsum_model":Gsum_model,
   
    'fitFeOP': fitFeOP,
    'fitFeUV': fitFeUV,
    
    
    
}
