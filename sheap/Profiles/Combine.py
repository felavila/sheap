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
    centers: List[float],
    amplitude_rules: List[Tuple[int, float, int]],
    profile_name: str,
):
    """
    SPAF (Sum Profiles Amplitude Free) wrapper for *log-lambda* line profiles.

    This builds a composite profile made of multiple lines that share the same
    *shape parameters* (e.g., ``vshift_kms``, ``fwhm_v_kms``, and any extra shape
    params like ``alpha``, ``eta``, or ``tau_kms``), while allowing a flexible
    set of **free amplitudes** combined through ``amplitude_rules``.

    Parameters
    ----------
    centers : list[float]
        Per-line rest wavelengths :math:`\\lambda_0` (Å). These are required and
        injected as the last parameter of the base profile for each line.
    amplitude_rules : list[(line_idx, coefficient, free_amp_idx)]
        For each line: ``amp_line = coefficient * free_amplitudes[free_amp_idx]``.

        Example for a doublet with fixed 2:1 ratio sharing the same free amp 0::

            [(0, 1.0, 0), (1, 0.5, 0)]
    profile_name : str
        Name of the base profile to use. It must exist in
        ``PROFILE_LINE_FUNC_MAP_loglambda`` and be decorated with ``@with_param_names``.

        The base profile must include at least these parameter names:
        ``"amplitude"`` and ``"lambda0"``.
        Any additional parameters are treated as *shared* across all lines.

    Returns
    -------
    ProfileFunc
        A callable ``G(x_lambda, params)`` decorated with ``@with_param_names``.

        The parameter layout is:

        - ``[amplitude0, ..., amplitude{Nfree-1}, <shared_params...>]``

        where ``<shared_params...>`` are all base parameters except ``amplitude``
        and ``lambda0`` (in the same order as the base profile's ``param_names``).

    Notes
    -----
    - This works for any log-lambda base profile with signature
      ``base_func(x_lambda, params)`` and ``param_names`` containing
      ``"amplitude"`` and ``"lambda0"``.
    - Shape parameters are shared across all lines; only amplitudes are
      combined via ``amplitude_rules``.
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
            f"Base profile '{profile_name}' must be decorated with @with_param_names "
            f"and expose 'param_names'."
        )

    if "amplitude" not in base_param_names or "lambda0" not in base_param_names:
        raise ValueError(
            f"Base profile '{profile_name}' must include parameter names "
            f"'amplitude' and 'lambda0'. Got: {base_param_names}"
        )

    # Shared params are everything except amplitude + lambda0, in base order
    shared_names = [n for n in base_param_names if n not in ("amplitude", "lambda0")]

    # Normalize/compact free amplitude indices
    raw_free = [r[2] for r in amplitude_rules]
    uniq = sorted({int(i) for i in raw_free})
    idx_map = {orig: new for new, orig in enumerate(uniq)}
    rules = [(int(li), float(coef), idx_map[int(fi)]) for li, coef, fi in amplitude_rules]
    n_free = len(uniq)

    # Public param names for the composite
    param_names = [f"amplitude{k}" for k in range(n_free)] + shared_names

    @with_param_names(param_names)
    def G(x_lambda, params):
        x_dtype = x_lambda.dtype

        amps_linear = params[:n_free]                 # linear free amplitudes
        shared_vals = params[n_free:]                 # shared shape params in shared_names order

        total = jnp.array(0.0, dtype=x_dtype)

        # Build each line with correct base param ordering
        for line_idx, coef, free_idx in rules:
            amp_line = coef * amps_linear[free_idx]
            lambda0_i = centers[line_idx].astype(x_dtype)

            # map name->value for the base params
            pdict = {"amplitude": amp_line, "lambda0": lambda0_i}
            for name, val in zip(shared_names, shared_vals):
                pdict[name] = val

            p_line = jnp.array([pdict[name] for name in base_param_names], dtype=x_dtype)
            total = total + base_func(x_lambda, p_line)

        return total

    return G



def SPAF_loglambda_old(
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
