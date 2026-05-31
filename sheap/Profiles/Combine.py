"""
Log-Lambda Profile Combination Utilities
========================================

This module defines utilities for **combining multiple emission-line components**
evaluated in **logarithmic wavelength space**, using velocity-based
parameterizations.

It provides:

- ``PROFILE_LINE_FUNC_MAP_loglambda`` :
  A registry mapping canonical profile names (e.g. ``"gaussian"``,
  ``"lorentzian"``, ``"skewed_gaussian"``) to their corresponding
  log-lambda profile functions.

- ``SPAF_loglambda`` :
  A SPAF (Sum Profiles Amplitude Free) constructor for log-lambda profiles,
  enabling physically motivated combinations of multiple emission lines
  with shared kinematic parameters.

The profiles referenced here operate internally in log(lambda) space via
the transformation:

.. math::

    v = c \\, \\ln(\\lambda / \\lambda_0)

ensuring exact symmetry in velocity space for Doppler-broadened features.

SPAF allows multiple lines to:
- Share kinematic parameters (velocity shift, FWHM, and shape parameters)
- Enforce fixed or semi-fixed amplitude ratios (e.g. doublets, multiplets)
- Be modeled with a reduced number of free parameters

Notes
-----
- Only profiles registered in ``PROFILE_LINE_FUNC_MAP_loglambda`` can be
  combined using ``SPAF_loglambda``.
- Base profiles must be decorated with ``@with_param_names`` and include
  at least ``"amplitude"`` and ``"lambda0"`` in their parameter list.
- Physical bounds and initial values for the combined parameters are
  handled by the constraint-building utilities elsewhere in *sheap*.

Examples
--------
.. code-block:: python

    from sheap.Profiles.combine import SPAF_loglambda

    # Hα + [NII] doublet with fixed 3:1 ratio
    centers = [6548.05, 6583.45]
    rules = [(0, 1.0, 0), (1, 3.0, 0)]

    G = SPAF_loglambda(
        centers=centers,
        amplitude_rules=rules,
        profile_name="gaussian",
    )

    # params = [amplitude0, vshift_kms, fwhm_v_kms]
    y = G(x_lambda, params)
"""

__author__ = "felavila"


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sheap.Core import ProfileFunc
import jax.numpy as jnp
from typing import List, Tuple, Callable
from sheap.Profiles.Utils import with_param_names,trapz_jax

from sheap.Profiles.profiles_lines_loglambda import (gaussian_fwhm_loglambda, lorentzian_fwhm_loglambda, skewed_gaussian_loglambda, top_hat_loglambda, voigt_pseudo_loglambda, emg_fwhm_loglambda,)

PROFILE_LINE_FUNC_MAP_loglambda: Dict[str, ProfileFunc] = {
    "gaussian": gaussian_fwhm_loglambda,
    "lorentzian": lorentzian_fwhm_loglambda,
    "skewed_gaussian": skewed_gaussian_loglambda,
    "top_hat": top_hat_loglambda,
    "voigt_pseudo": voigt_pseudo_loglambda,
    "emg": emg_fwhm_loglambda,
}

def SPAF_loglambda(
    centers: list[float],
    amplitude_rules: list[tuple[int, float, int]],
    profile_name: str,
):
    """
    SPAF (Sum Profiles Amplitude Free) wrapper for log-lambda line profiles.
    """
    centers = jnp.asarray(centers, dtype=jnp.float32)

    base_func = PROFILE_LINE_FUNC_MAP_loglambda.get(profile_name)
    if base_func is None:
        raise ValueError(
            f"Profile '{profile_name}' not found in PROFILE_LINE_FUNC_MAP_loglambda."
        )

    base_param_names = getattr(base_func, "param_names", None)
    if not base_param_names:
        raise ValueError(
            f"Base profile '{profile_name}' must expose 'param_names'."
        )

    if "amplitude" not in base_param_names or "lambda0" not in base_param_names:
        raise ValueError(
            f"Base profile '{profile_name}' must include 'amplitude' and 'lambda0'. "
            f"Got: {base_param_names}"
        )

    shared_names = [n for n in base_param_names if n not in ("amplitude", "lambda0")]

    raw_free = [r[2] for r in amplitude_rules]
    uniq = sorted({int(i) for i in raw_free})
    idx_map = {orig: new for new, orig in enumerate(uniq)}
    rules = [(int(li), float(coef), idx_map[int(fi)]) for li, coef, fi in amplitude_rules]
    n_free = len(uniq)

    param_names = [f"amplitude{k}" for k in range(n_free)] + shared_names
    linear_names = [f"amplitude{k}" for k in range(n_free)]

    @with_param_names(param_names, linear_param_names=linear_names)
    def G(x_lambda, params):
        x_dtype = x_lambda.dtype
        params = jnp.asarray(params, dtype=x_dtype)

        amps_linear = params[:n_free]
        shared_vals = params[n_free:]

        total = jnp.array(0.0, dtype=x_dtype)

        for line_idx, coef, free_idx in rules:
            amp_line = coef * amps_linear[free_idx]
            lambda0_i = centers[line_idx].astype(x_dtype)

            pdict = {"amplitude": amp_line, "lambda0": lambda0_i}
            for name, val in zip(shared_names, shared_vals):
                pdict[name] = val

            p_line = jnp.array([pdict[name] for name in base_param_names], dtype=x_dtype)
            total = total + base_func(x_lambda, p_line)

        return total

    return G