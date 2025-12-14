


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sheap.Core import ProfileFunc
import jax.numpy as jnp
from typing import List, Tuple, Callable
from sheap.Profiles.profiles_lines_loglambda import (gaussian_fwhm_loglambda)
from sheap.Profiles.Utils import with_param_names,trapz_jax


PROFILE_LINE_FUNC_MAP_loglambda: Dict[str, ProfileFunc] = {
    'gaussian': gaussian_fwhm_loglambda}


def SPAF_loglambda(
    centers: List[float],
    amplitude_rules: List[Tuple[int, float, int]],
    profile_name: str,
):
    """
    SPAF (Sum Profiles Amplitude Free) for log-lambda profiles.

    Parameters
    ----------
    centers : list[float]
        Per-line rest wavelengths λ0 (Å). These are *required* and injected
        as the last parameter of the base profile.
    amplitude_rules : list[(line_idx, coefficient, free_amp_idx)]
        For each line: amp_line = coefficient * free_amplitudes[free_amp_idx].
        Example for a doublet with fixed 2:1 ratio sharing the same free amp 0:
            [(0, 1.0, 0), (1, 0.5, 0)]
    base_func : Callable
        A profile with param_names == ["amp","vshift_kms","fwhm_v_kms","lambda0"].

    Returns
    -------
    ProfileFunc G(x, params)
        params layout:
          [ amplitude0, amplitude1, ..., amplitude_{Nfree-1},
            shift_kms,            # shared Δv for the whole group
            fwhm_v_kms ]          # shared FWHM in km/s
    """
    centers = jnp.asarray(centers, dtype=jnp.float32)
    base_func = PROFILE_LINE_FUNC_MAP_loglambda.get(profile_name)
    if base_func is None:
        raise ValueError(f"Profile '{profile_name}' not found in PROFILE_LINE_FUNC_MAP_loglambda.")
    # normalize/compact free amplitude indices
    raw_free = [r[2] for r in amplitude_rules]
    uniq = sorted({int(i) for i in raw_free})
    idx_map = {orig: new for new, orig in enumerate(uniq)}
    rules = [(li, coef, idx_map[int(fi)]) for li, coef, fi in amplitude_rules]
    n_free = len(uniq)

    # Public param names (self-documenting)
    param_names = [f"amplitude{k}" for k in range(n_free)] + ["vshift_kms", "fwhm_v_kms"]

    @with_param_names(param_names)
    def G(x, params):
        amps_linear = params[:n_free]                  # linear amplitudes
        vshift      = params[n_free + 0]               # shared Δv [km/s]
        #fwhm_vkms   = 10**params[n_free + 1]               # shared FWHM_v [km/s]
        fwhm_vkms  = params[n_free + 1]      # stored as log10(FWHM [km/s])
        #fwhm_vkms = jnp.maximum(jnp.power(10.0, log10_fwhm), jnp.finfo(x.dtype).tiny)
        total = 0.0
        for line_idx, coef, free_idx in rules:
            amp_line  = coef * amps_linear[free_idx]
            lambda0_i = centers[line_idx]
            # base expects [amp, vshift_kms, fwhm_v_kms, lambda0]
            p_line = jnp.array([amp_line, vshift, fwhm_vkms, lambda0_i], dtype=x.dtype)
            total += base_func(x, p_line)
        return total

    return G
