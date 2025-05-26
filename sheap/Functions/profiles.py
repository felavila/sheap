from typing import Callable, Dict
import jax.numpy as jnp



from sheap.Functions.continuum_profiles import (linear,balmerconti,powerlaw,brokenpowerlaw)
from sheap.Functions.lines_profiles import (gaussian_fwhm,lorentzian_fwhm,Gsum_model)
from sheap.Functions.template_func import (fitFeOP,fitFeUV)



# Define the signature for profile functions: (x, params) -> output
ProfileFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

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

def make_g(list):
    amplitudes, centers = list.amplitude, list.center
    return PROFILE_FUNC_MAP["Gsum_model"](centers, amplitudes)