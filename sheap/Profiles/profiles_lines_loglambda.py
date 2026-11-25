"""
Log-Lambda Emission Line Profiles
================================

Velocity-symmetric line profiles defined on a linear wavelength grid,
using v = c * ln(lambda / lambda0) internally.

Each profile provides four entry points:

    profile(x, local_params)                      # JAX forward model
    profile.design_builder(x, local_params)       # existing JAX linear basis
    profile.numpy_model(x, local_params)           # NumPy float64 forward model
    profile.numpy_design_builder(x, local_params)  # NumPy float64 linear basis

The NumPy functions are attached directly to the original decorated profile.
They do not create dictionaries or use an optional-backend flag.

Parameter order is exactly the same for JAX and NumPy. The first parameter
(amplitude) is linear, including in skewed Gaussian and EMG profiles.
"""

__author__ = "felavila"

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erfc, erf
from scipy.special import ndtr, erfcx, erfc as scipy_erfc

from sheap.Profiles.Utils import (
    with_param_names,
    add_linear_amplitude_design_builder,
)
from sheap.Utils.Constants import DEFAULT_C_KMS, FWHM_TO_SIGMA

__all__ = [
    "gaussian_fwhm_loglambda",
    "lorentzian_fwhm_loglambda",
    "skewed_gaussian_loglambda",
    "top_hat_loglambda",
    "voigt_pseudo_loglambda",
    "emg_fwhm_loglambda",
]


# ============================================================================
# Existing JAX profiles
# ============================================================================

@add_linear_amplitude_design_builder()
@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "lambda0"], linear_param_names=["amplitude"], profile_name="gaussian_fwhm_loglambda",)
def gaussian_fwhm_loglambda(x_lambda, params):
    """Gaussian: [amplitude, vshift_kms, log10(FWHM), lambda0]."""
    amp, vshift_kms, fwhm_v_kms, lambda0 = params
    ratio = jnp.clip(
        x_lambda / lambda0,
        min=jnp.finfo(x_lambda.dtype).tiny,
        max=jnp.inf,
    )
    y = DEFAULT_C_KMS * jnp.log(ratio)
    fwhm_linear = 10.0 ** fwhm_v_kms
    sigma_v = fwhm_linear * FWHM_TO_SIGMA
    z = (y - vshift_kms) / sigma_v
    return amp * jnp.exp(-0.5 * z * z)


@add_linear_amplitude_design_builder()
@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "lambda0"],linear_param_names=["amplitude"], profile_name="lorentzian_fwhm_loglambda",)
def lorentzian_fwhm_loglambda(x_lambda, params):
    """Lorentzian: [amplitude, vshift_kms, log10(FWHM), lambda0]."""
    amp, vshift_kms, fwhm_v_kms, lambda0 = params
    ratio = jnp.clip(
        x_lambda / lambda0,
        min=jnp.finfo(x_lambda.dtype).tiny,
        max=jnp.inf,
    )
    y = DEFAULT_C_KMS * jnp.log(ratio)
    fwhm = 10.0 ** fwhm_v_kms
    gamma = 0.5 * fwhm
    z = (y - vshift_kms) / gamma
    return amp / (1.0 + z * z)


@add_linear_amplitude_design_builder()
@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "alpha", "lambda0"],linear_param_names=["amplitude"],profile_name="skewed_gaussian_loglambda",)
def skewed_gaussian_loglambda(x_lambda, params):
    """Skew normal: [amplitude, vshift, log10(FWHM), alpha, lambda0]."""
    amp, vshift_kms, fwhm_v_kms, alpha, lambda0 = params
    ratio = jnp.clip(
        x_lambda / lambda0,
        min=jnp.finfo(x_lambda.dtype).tiny,
        max=jnp.inf,
    )
    y = DEFAULT_C_KMS * jnp.log(ratio)
    fwhm = 10.0 ** fwhm_v_kms
    sigma = fwhm * FWHM_TO_SIGMA
    t = (y - vshift_kms) / sigma
    phi = jnp.exp(-0.5 * t * t) / jnp.sqrt(2.0 * jnp.pi)
    # jnp.erf is not present in some JAX versions. jax.scipy.special.erf is.
    Phi = 0.5 * (1.0 + erf(alpha * t / jnp.sqrt(2.0)))
    return amp * 2.0 * phi * Phi


@add_linear_amplitude_design_builder()
@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "lambda0"], linear_param_names=["amplitude"], profile_name="top_hat_loglambda",)
def top_hat_loglambda(x_lambda, params):
    """Top hat: [amplitude, vshift_kms, log10(FWHM), lambda0]."""
    amp, vshift_kms, fwhm_v_kms, lambda0 = params
    ratio = jnp.clip(
        x_lambda / lambda0,
        min=jnp.finfo(x_lambda.dtype).tiny,
        max=jnp.inf,
    )
    y = DEFAULT_C_KMS * jnp.log(ratio)
    fwhm = 10.0 ** fwhm_v_kms
    half = 0.5 * fwhm
    return amp * (jnp.abs(y - vshift_kms) <= half)


@add_linear_amplitude_design_builder()
@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "eta", "lambda0"], linear_param_names=["amplitude"], profile_name="voigt_pseudo_loglambda",)
def voigt_pseudo_loglambda(x_lambda, params):
    """Pseudo Voigt: [amplitude, vshift, log10(FWHM), eta, lambda0]."""
    amp, vshift_kms, fwhm_v_kms, eta, lambda0 = params
    ratio = jnp.clip(
        x_lambda / lambda0,
        min=jnp.finfo(x_lambda.dtype).tiny,
        max=jnp.inf,
    )
    y = DEFAULT_C_KMS * jnp.log(ratio)
    fwhm = 10.0 ** fwhm_v_kms
    sigma = fwhm * FWHM_TO_SIGMA
    gamma = 0.5 * fwhm
    tg = (y - vshift_kms) / sigma
    tl = (y - vshift_kms) / gamma
    G = jnp.exp(-0.5 * tg * tg)
    L = 1.0 / (1.0 + tl * tl)
    return amp * (eta * L + (1.0 - eta) * G)


@add_linear_amplitude_design_builder()
@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "tau_kms", "lambda0"], linear_param_names=["amplitude"], profile_name="emg_fwhm_loglambda",)
def emg_fwhm_loglambda(x_lambda, params):
    """EMG: [amplitude, vshift, log10(FWHM), tau_kms, lambda0]."""
    amp, vshift_kms, fwhm_v_kms, tau_kms, lambda0 = params
    ratio = jnp.clip(
        x_lambda / lambda0,
        min=jnp.finfo(x_lambda.dtype).tiny,
        max=jnp.inf,)
    y = DEFAULT_C_KMS * jnp.log(ratio)
    fwhm = 10.0 ** fwhm_v_kms
    sigma = fwhm * FWHM_TO_SIGMA
    tau = jnp.clip(tau_kms, min=jnp.finfo(x_lambda.dtype).tiny,max=jnp.inf,)
    arg = (sigma / tau - (y - vshift_kms) / sigma) / jnp.sqrt(2.0)
    core = jnp.exp((sigma * sigma) / (2.0 * tau * tau)
                   - (y - vshift_kms) / tau)
    return amp * 0.5 / tau * core * erfc(arg)


# ============================================================================
# NumPy float64 profiles - use exactly the same parameters and normalization
# ============================================================================

def _numpy_line_shape(x_lambda, params, kind):
    """Return unit-amplitude float64 profile matching the corresponding JAX model.

    This deliberately preserves SHEAP's current profile normalizations:
    e.g. skew-normal peak is not normalized to unity and EMG has 1/tau units.
    """
    x = np.asarray(x_lambda, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError(f"x_lambda must be a 1D array, got {x.shape}")
    n_parameters = 5 if kind in {"skewed_gaussian", "voigt_pseudo", "emg"} else 4
    if p.shape != (n_parameters,):
        raise ValueError(
            f"{kind}: expected {n_parameters} local parameters, got {p.shape}"
        )

    _, vshift_kms, logfwhm = p[:3]
    lambda0 = p[-1]

    ratio = np.clip(x / lambda0, np.finfo(np.float64).tiny, np.inf)
    y = float(DEFAULT_C_KMS) * np.log(ratio)
    distance = y - vshift_kms
    fwhm = 10.0 ** logfwhm

    if kind == "gaussian":
        sigma = fwhm * float(FWHM_TO_SIGMA)
        z = distance / sigma
        return np.exp(-0.5 * z * z)

    if kind == "lorentzian":
        gamma = 0.5 * fwhm
        z = distance / gamma
        return 1.0 / (1.0 + z * z)

    if kind == "skewed_gaussian":
        alpha = p[3]
        sigma = fwhm * float(FWHM_TO_SIGMA)
        t = distance / sigma
        phi = np.exp(-0.5 * t * t) / np.sqrt(2.0 * np.pi)
        return 2.0 * phi * ndtr(alpha * t)

    if kind == "top_hat":
        return (np.abs(distance) <= 0.5 * fwhm).astype(np.float64)

    if kind == "voigt_pseudo":
        eta = p[3]
        sigma = fwhm * float(FWHM_TO_SIGMA)
        gamma = 0.5 * fwhm
        tg = distance / sigma
        tl = distance / gamma
        G = np.exp(-0.5 * tg * tg)
        L = 1.0 / (1.0 + tl * tl)
        return eta * L + (1.0 - eta) * G

    if kind == "emg":
        sigma = fwhm * float(FWHM_TO_SIGMA)
        tau = np.clip(p[3], np.finfo(np.float64).tiny, np.inf)
        t = distance / sigma
        arg = (sigma / tau - t) / np.sqrt(2.0)

        # Original: 0.5 / tau * exp(sigma**2/(2*tau**2) - distance/tau) * erfc(arg)
        # For arg >= 0: exp(E)*erfc(arg) = exp(-0.5*t**2)*erfcx(arg)
        # For arg < 0:  E is negative, so the original expression is stable.
        # Splitting avoids 0*inf in the exponentially suppressed wings.
        shape = np.empty_like(distance, dtype=np.float64)
        high = arg >= 0.0
        if np.any(high):
            shape[high] = (
                (0.5 / tau)
                * np.exp(-0.5 * t[high] ** 2)
                * erfcx(arg[high])
            )
        if np.any(~high):
            exponent = sigma**2 / (2.0 * tau**2) - distance[~high] / tau
            shape[~high] = (
                (0.5 / tau)
                * np.exp(exponent)
                * scipy_erfc(arg[~high])
            )
        return shape

    raise ValueError(f"Unknown log-lambda profile kind: {kind}")


def _register_numpy_profile(profile, kind):
    """Attach the two methods expected by SPAF directly to the JAX profile."""

    def numpy_model(x_lambda, params):
        p = np.asarray(params, dtype=np.float64)
        unit = _numpy_line_shape(x_lambda, p, kind)
        return p[0] * unit

    def numpy_design_builder(x_lambda, params):
        unit = _numpy_line_shape(x_lambda, params, kind)
        return np.zeros_like(unit), unit[:, None]

    profile.numpy_model = numpy_model
    profile.numpy_design_builder = numpy_design_builder


_register_numpy_profile(gaussian_fwhm_loglambda, "gaussian")
_register_numpy_profile(lorentzian_fwhm_loglambda, "lorentzian")
_register_numpy_profile(skewed_gaussian_loglambda, "skewed_gaussian")
_register_numpy_profile(top_hat_loglambda, "top_hat")
_register_numpy_profile(voigt_pseudo_loglambda, "voigt_pseudo")
_register_numpy_profile(emg_fwhm_loglambda, "emg")
