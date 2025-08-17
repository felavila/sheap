"""This module ?."""
__author__ = 'felavila'


__all__ = [
    "SPAF",
    "wrap_profile_with_center_override",
]

from typing import Callable, Dict, List, Tuple
import jax.numpy as jnp


from sheap.Core import ProfileFunc

from sheap.Profiles.profiles_continuum import (linear, balmercontinuum, powerlaw, brokenpowerlaw,logparabola,exp_cutoff,polynomial)
from sheap.Profiles.profile_lines import (gaussian_fwhm, lorentzian_fwhm, skewed_gaussian,emg_fwhm, top_hat, voigt_pseudo)
from sheap.Profiles.profiles_templates import make_feii_template_function,make_host_function
from sheap.Profiles.utils import with_param_names



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
    Wrap a spectral profile function to allow external override of the center parameter.

    This wrapper ensures compatibility with JIT and automatically handles conversion
    from linear amplitude to logarithmic amplitude if the underlying profile expects
    a `logamp` as its first parameter.

    The wrapped function preserves the following signature:
        wrapped(x, params, override_center)

    Parameters
    ----------
    profile_func : Callable
        A spectral line profile function that accepts a parameter vector, where
        one of the parameters is named "center".

    Returns
    -------
    Callable
        A new profile function with the same input/output signature as `profile_func`,
        but with:
            - the ability to override the center value at call time, and
            - automatic conversion from linear to log-amplitude if needed.

    Raises
    ------
    ValueError
        If the provided profile function does not include a 'center' parameter.
    """
    param_names = profile_func.param_names
    if "center" not in param_names:
        raise ValueError(f"Profile '{profile_func.__name__}' has no 'center' parameter.")
    center_idx = param_names.index("center")

    # Will we need to convert linear→log10?
    expects_log = (param_names[0] == "logamp")

    def wrapped(x, params, override_center):
        # params[0] is always a linear amp coming in…
        if expects_log:
            # Convert to log10 space for the base profile
            # (we clip at a tiny positive floor to avoid log10(0))
            amp = params[0]
            safe_amp = jnp.maximum(amp, 1e-30)
            logamp = jnp.log10(safe_amp)
            # reconstruct the param vector that profile_func expects
            params_for_profile = jnp.concatenate([jnp.array([logamp]), params[1:]])
        else:
            params_for_profile = params

        # now insert the overridden center
        full_params = jnp.insert(params_for_profile, center_idx, override_center)
        return profile_func(x, full_params)

    return wrapped

def SPAF(centers: List[float], amplitude_rules: List[Tuple[int, float, int]], profile_name: str) -> ProfileFunc:
    """
    Create a SPAF (Sum Profiles Amplitude Free) profile composed of multiple shifted lines
    with tied amplitude coefficients and shared shape parameters.

    Parameters
    ----------
    centers : list of float
        Rest-frame centers for each individual line in the group.
    amplitude_rules : list of tuple
        Each rule is (line_idx, coefficient, free_amp_idx), specifying how each line’s
        amplitude is computed as `coefficient × free_amplitude[free_amp_idx]`.
    profile_name : str
        Name of the base profile to use for each line (e.g., "gaussian", "lorentzian").

    Returns
    -------
    ProfileFunc
        A callable G(x, params) that evaluates the composite profile.

        The parameter vector `params` contains:
            - amplitudes: either log10(amplitude) or linear [N], depending on profile
            - shift: a shared wavelength shift [1]
            - extras: remaining shape parameters from the base profile
    """
    centers = jnp.array(centers)

    raw_idxs = [rule[2] for rule in amplitude_rules]
    uniq_idxs = sorted({int(i) for i in raw_idxs})
    idx_map = {orig: new for new, orig in enumerate(uniq_idxs)}

    amplitude_rules = [
        (line_i, coef, idx_map[int(free_i)])
        for line_i, coef, free_i in amplitude_rules
    ]
    n_free_amps = len(uniq_idxs)

    base_func = PROFILE_LINE_FUNC_MAP.get(profile_name)
    if base_func is None:
        raise ValueError(f"Profile '{profile_name}' not found in PROFILE_LINE_FUNC_MAP.")

    wrapped_profile = wrap_profile_with_center_override(base_func)
    expects_log = (base_func.param_names[0] == "logamp")
    amp_names = [f"logamp{n}" if expects_log else f"amplitude{n}" for n in range(n_free_amps)]
    param_names = amp_names + ["shift"] + base_func.param_names[2:]

    @with_param_names(param_names)
    def G(x, params):
        amps = params[:n_free_amps]
        if expects_log:
            amps = 10 ** amps #jnp.sign(amps) *
            #amps = jnp.sign(amps) *10 ** jnp.abs(amps)
        delta = params[n_free_amps]
        extras = params[n_free_amps + 1:]

        result = 0.0
        for idx, coef, free_idx in amplitude_rules:
            amp = coef * amps[free_idx]
            center = centers[idx] + delta
            full_params = jnp.concatenate([jnp.array([amp]), extras])
            result += wrapped_profile(x, full_params, center)
        return result

    return G


# Full profile registry (for spectral modeling)
PROFILE_FUNC_MAP: Dict[str, ProfileFunc] = {
    'balmercontinuum': balmercontinuum,
    'fetemplate': make_feii_template_function,
    'SPAF': SPAF,
    "hostmiles": make_host_function} #

PROFILE_FUNC_MAP.update(PROFILE_LINE_FUNC_MAP)
PROFILE_FUNC_MAP.update(PROFILE_CONTINUUM_FUNC_MAP)
