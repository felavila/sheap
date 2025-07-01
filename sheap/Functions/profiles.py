from typing import Callable, Dict, List, Tuple
import jax.numpy as jnp

from sheap.Functions.continuum_profiles import (linear, balmercontinuum, powerlaw, brokenpowerlaw,logparabola,exp_cutoff,polynomial)
from sheap.Functions.lines_profiles import (gaussian_fwhm, lorentzian_fwhm, skewed_gaussian,emg_fwhm, top_hat, voigt_pseudo, Gsum_model,sum_gaussian_amplitude_free)
from sheap.Functions.template_func import fitFeOP, fitFeUV
from sheap.Functions.utils import param_count, with_param_names

# Signature: (x, params) -> profile output
ProfileFunc = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]

# Low-level line profiles (require center+amplitude inside param vector)
PROFILE_LINE_FUNC_MAP: Dict[str, ProfileFunc] = {
    'gaussian': gaussian_fwhm,
    'lorentzian': lorentzian_fwhm,
    'voigt_pseudo': voigt_pseudo,
    'skewed_gaussian': skewed_gaussian,
    'emg_fwhm': emg_fwhm,
    'top_hat': top_hat
}
PROFILE_CONTINUUM_FUNC_MAP: Dict[str, ProfileFunc] = {
    'linear': linear,
    'powerlaw': powerlaw,
    'brokenpowerlaw': brokenpowerlaw,
    'logparabola': logparabola,
    'exp_cutoff': exp_cutoff,
    'polynomial': polynomial
}

def wrap_profile_with_center_override(profile_func: Callable) -> Callable:
    """
    JIT-compatible version that inserts `center` into correct param position using `jnp.insert`.
    """
    param_names = profile_func.param_names
    if "center" not in param_names:
        raise ValueError(f"Profile '{profile_func.__name__}' has no 'center' parameter.")

    center_idx = param_names.index("center")

    def wrapped(x, params, override_center):
        full_params = jnp.insert(params, center_idx, override_center)
        return profile_func(x, full_params)

    return wrapped

def SPAF(centers: List[float], amplitude_rules: List[Tuple[int, float, int]], profile_name: str) -> ProfileFunc:
    """
    SPAF = Sum Profiles Amplitude Free with normalization of amplitude indices.

    Args:
        centers: Rest-frame line centers.
        amplitude_rules: List of (line_idx, coefficient, free_amp_idx).
                         free_amp_idx may be arbitrary ints; they will be remapped.
        profile_name: Base profile to use (must exist in PROFILE_LINE_FUNC_MAP).

    Returns:
        ProfileFunc: Callable G(x, params) with:
            - free amplitudes [N]
            - shared shift [1]
            - shared profile params [M]
    """
    centers = jnp.array(centers)

    # 1) Normalize free_amp_idx into a contiguous, 0-based Python int sequence
    raw_idxs = [rule[2] for rule in amplitude_rules]
    uniq_idxs = sorted({int(i) for i in raw_idxs})
    idx_map = {orig: new for new, orig in enumerate(uniq_idxs)}
    # Rebuild amplitude_rules with mapped indices
    amplitude_rules = [
        (line_i, coef, idx_map[int(free_i)])
        for line_i, coef, free_i in amplitude_rules
    ]

    # 2) Determine number of free amplitude parameters
    n_free_amps = len(uniq_idxs)

    # 3) Retrieve and wrap the base profile
    base_func = PROFILE_LINE_FUNC_MAP.get(profile_name)
    if base_func is None:
        raise ValueError(f"Profile '{profile_name}' not found in PROFILE_LINE_FUNC_MAP.")
    wrapped_profile = wrap_profile_with_center_override(base_func)

    # 4) Build parameter names: amplitudes, shift, then other profile params
    param_names = [f"amplitude{n}" for n in range(n_free_amps)] + ["shift"] + base_func.param_names[2:]

    @with_param_names(param_names)
    def G(x, params):
        free_amps = params[:n_free_amps]
        delta = params[n_free_amps]
        extras = params[n_free_amps + 1:]

        result = 0.0
        for idx, coef, free_idx in amplitude_rules:
            i = int(free_idx)  # safe Python int index
            amp = coef * free_amps[i]
            center = centers[idx] + delta
            full_params = jnp.concatenate([jnp.array([amp]), extras])
            result += wrapped_profile(x, full_params, center)
        return result

    return G


# Full profile registry (for spectral modeling)
PROFILE_FUNC_MAP: Dict[str, ProfileFunc] = {
    'balmercontinuum': balmercontinuum,
    'fitFeOP': fitFeOP,
    'fitFeUV': fitFeUV,
    'SPAF': SPAF}

PROFILE_FUNC_MAP.update(PROFILE_LINE_FUNC_MAP)
PROFILE_FUNC_MAP.update(PROFILE_CONTINUUM_FUNC_MAP)