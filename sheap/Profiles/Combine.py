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

# def SPAF_loglambda(
#     centers: list[float],
#     amplitude_rules: list[tuple[int, float, int]],
#     profile_name: str,
# ):
#     """
#     SPAF (Sum Profiles Amplitude Free) wrapper for log-lambda line profiles.
#     """
#     centers = jnp.asarray(centers, dtype=jnp.float32)

#     base_func = PROFILE_LINE_FUNC_MAP_loglambda.get(profile_name)
#     if base_func is None:
#         raise ValueError(
#             f"Profile '{profile_name}' not found in PROFILE_LINE_FUNC_MAP_loglambda."
#         )

#     base_param_names = getattr(base_func, "param_names", None)
#     if not base_param_names:
#         raise ValueError(
#             f"Base profile '{profile_name}' must expose 'param_names'."
#         )

#     if "amplitude" not in base_param_names or "lambda0" not in base_param_names:
#         raise ValueError(
#             f"Base profile '{profile_name}' must include 'amplitude' and 'lambda0'. "
#             f"Got: {base_param_names}"
#         )

#     shared_names = [n for n in base_param_names if n not in ("amplitude", "lambda0")]

#     raw_free = [r[2] for r in amplitude_rules]
#     uniq = sorted({int(i) for i in raw_free})
#     idx_map = {orig: new for new, orig in enumerate(uniq)}
#     rules = [(int(li), float(coef), idx_map[int(fi)]) for li, coef, fi in amplitude_rules]
#     n_free = len(uniq)

#     param_names = [f"amplitude{k}" for k in range(n_free)] + shared_names
#     linear_names = [f"amplitude{k}" for k in range(n_free)]

#     @with_param_names(param_names, linear_param_names=linear_names)
#     def G(x_lambda, params):
#         x_dtype = x_lambda.dtype
#         params = jnp.asarray(params, dtype=x_dtype)

#         amps_linear = params[:n_free]
#         shared_vals = params[n_free:]

#         total = jnp.array(0.0, dtype=x_dtype)

#         for line_idx, coef, free_idx in rules:
#             amp_line = coef * amps_linear[free_idx]
#             lambda0_i = centers[line_idx].astype(x_dtype)

#             pdict = {"amplitude": amp_line, "lambda0": lambda0_i}
#             for name, val in zip(shared_names, shared_vals):
#                 pdict[name] = val

#             p_line = jnp.array([pdict[name] for name in base_param_names], dtype=x_dtype)
#             total = total + base_func(x_lambda, p_line)

#         return total

#     return G

def SPAF_loglambda(
    centers: list[float],
    amplitude_rules: list[tuple[int, float, int]],
    profile_name: str,
):
    """
    SPAF (Sum Profiles Amplitude Free) wrapper for log-lambda profiles.

    Each base profile is assumed to have a linear amplitude parameter
    and a .design_builder method.

    Parameters
    ----------
    centers
        Rest-frame wavelength for each line.

    amplitude_rules
        List of:

            (line_idx, coefficient, free_amplitude_idx)

        Example:

            [
                (0, 1.0, 0),
                (1, 1/3, 0),
            ]

        means both lines share the same free amplitude, with the
        second line fixed to 1/3 of the first.

    profile_name
        Base log-lambda profile.
    """

    # ============================================================
    # Fixed line centers
    # ============================================================

    centers = jnp.asarray(
        centers,
        dtype=jnp.float32,
    )

    # ============================================================
    # Base profile
    # ============================================================

    base_func = PROFILE_LINE_FUNC_MAP_loglambda.get(
        profile_name
    )

    if base_func is None:
        raise ValueError(
            f"Profile '{profile_name}' not found in "
            "PROFILE_LINE_FUNC_MAP_loglambda."
        )

    base_param_names = getattr(
        base_func,
        "param_names",
        None,
    )

    if not base_param_names:
        raise ValueError(
            f"Base profile '{profile_name}' must expose "
            "'param_names'."
        )

    base_param_names = list(
        base_param_names
    )

    # ============================================================
    # Required parameters
    # ============================================================

    if (
        "amplitude" not in base_param_names
        or "lambda0" not in base_param_names
    ):
        raise ValueError(
            f"Base profile '{profile_name}' must include "
            "'amplitude' and 'lambda0'. "
            f"Got: {base_param_names}"
        )

    amplitude_pos = base_param_names.index(
        "amplitude"
    )

    lambda0_pos = base_param_names.index(
        "lambda0"
    )

    # ============================================================
    # Shared nonlinear parameters
    # ============================================================

    shared_names = [
        name
        for name in base_param_names
        if name not in (
            "amplitude",
            "lambda0",
        )
    ]

    shared_positions = [
        base_param_names.index(name)
        for name in shared_names
    ]

    # ============================================================
    # Normalize free-amplitude indices
    # ============================================================

    raw_free = [
        rule[2]
        for rule in amplitude_rules
    ]

    uniq = sorted(
        {
            int(i)
            for i in raw_free
        }
    )

    idx_map = {
        original: new
        for new, original in enumerate(uniq)
    }

    rules = [
        (
            int(line_idx),
            float(coef),
            idx_map[int(free_idx)],
        )
        for (
            line_idx,
            coef,
            free_idx,
        ) in amplitude_rules
    ]

    n_free = len(
        uniq
    )

    # ============================================================
    # SPAF parameter names
    # ============================================================

    param_names = (
        [
            f"amplitude{k}"
            for k in range(n_free)
        ]
        + shared_names
    )

    linear_names = [
        f"amplitude{k}"
        for k in range(n_free)
    ]

    # ============================================================
    # Helper: construct one base-profile parameter vector
    # ============================================================

    def make_line_params(
        dtype,
        amplitude,
        lambda0,
        shared_vals,
    ):

        p_line = jnp.zeros(
            len(base_param_names),
            dtype=dtype,
        )

        p_line = p_line.at[
            amplitude_pos
        ].set(
            amplitude
        )

        p_line = p_line.at[
            lambda0_pos
        ].set(
            lambda0
        )

        # Number of these is static, so Python loop is fine.
        for local_idx, base_idx in enumerate(
            shared_positions
        ):

            p_line = p_line.at[
                base_idx
            ].set(
                shared_vals[
                    local_idx
                ]
            )

        return p_line

    # ============================================================
    # Normal SPAF model
    # ============================================================

    @with_param_names(
        param_names,
        linear_param_names=linear_names,
    )
    def G(
        x_lambda,
        params,
    ):

        dtype = x_lambda.dtype

        params = jnp.asarray(
            params,
            dtype=dtype,
        )

        amps_linear = params[
            :n_free
        ]

        shared_vals = params[
            n_free:
        ]

        total = jnp.zeros_like(
            x_lambda,
            dtype=dtype,
        )

        # --------------------------------------------------------
        # Sum all physical lines
        # --------------------------------------------------------

        for (
            line_idx,
            coef,
            free_idx,
        ) in rules:

            amp_line = (
                jnp.asarray(
                    coef,
                    dtype=dtype,
                )
                * amps_linear[
                    free_idx
                ]
            )

            lambda0_i = centers[
                line_idx
            ].astype(
                dtype
            )

            p_line = make_line_params(
                dtype=dtype,
                amplitude=amp_line,
                lambda0=lambda0_i,
                shared_vals=shared_vals,
            )

            total = (
                total
                + base_func(
                    x_lambda,
                    p_line,
                )
            )

        return total

    # ============================================================
    # DIRECT DESIGN BUILDER
    # ============================================================

    def design_builder(
        x_lambda,
        params,
    ):
        """
        Returns
        -------
        offset : (n_pixels,)

        columns : (n_pixels, n_free)

        Each column corresponds to one free SPAF amplitude.

        If several physical lines share the same free amplitude,
        their unit-amplitude profiles are summed into the same
        design-matrix column with the appropriate coefficients.
        """

        dtype = x_lambda.dtype

        params = jnp.asarray(
            params,
            dtype=dtype,
        )

        # --------------------------------------------------------
        # SPAF amplitudes intentionally ignored.
        #
        # FISTA will solve them.
        # --------------------------------------------------------

        shared_vals = params[
            n_free:
        ]

        offset_total = jnp.zeros_like(
            x_lambda,
            dtype=dtype,
        )

        columns = jnp.zeros(
            (
                x_lambda.shape[0],
                n_free,
            ),
            dtype=dtype,
        )

        # --------------------------------------------------------
        # Build each physical line once
        # --------------------------------------------------------

        for (
            line_idx,
            coef,
            free_idx,
        ) in rules:

            lambda0_i = centers[
                line_idx
            ].astype(
                dtype
            )

            # ----------------------------------------------------
            # amplitude value itself is irrelevant to the
            # base design_builder, because amplitude is linear.
            #
            # Use 1 for clarity.
            # ----------------------------------------------------

            p_line = make_line_params(
                dtype=dtype,
                amplitude=jnp.asarray(
                    1.0,
                    dtype=dtype,
                ),
                lambda0=lambda0_i,
                shared_vals=shared_vals,
            )

            # ----------------------------------------------------
            # Base profile gives:
            #
            # offset_i
            # columns_i[:, 0]
            #
            # for its own amplitude.
            # ----------------------------------------------------

            offset_i, columns_i = (
                base_func.design_builder(
                    x_lambda,
                    p_line,
                )
            )

            # Usually zero for your emission-line profiles.
            offset_total = (
                offset_total
                + offset_i
            )

            # ----------------------------------------------------
            # Add this physical line to the correct FREE
            # amplitude column.
            #
            # Important: use ADD, not SET, because multiple
            # physical lines can share the same free amplitude.
            # ----------------------------------------------------

            columns = columns.at[
                :,
                free_idx,
            ].add(
                jnp.asarray(
                    coef,
                    dtype=dtype,
                )
                * columns_i[
                    :,
                    0,
                ]
            )

        return (
            offset_total,
            columns,
        )

    # ============================================================
    # Attach builder
    # ============================================================

    G.design_builder = (
        design_builder
    )

    # Useful metadata
    G.base_func = base_func
    G.centers = centers
    G.amplitude_rules = tuple(
        rules
    )
    G.n_free_amplitudes = n_free

    return G