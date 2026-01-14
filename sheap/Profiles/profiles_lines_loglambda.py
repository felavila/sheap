"""
Log-Lambda Emission Line Profiles
================================

This module defines a collection of **emission-line profile functions**
evaluated in **logarithmic wavelength space**, ensuring **velocity symmetry**
for Doppler-broadened spectral features.

All profiles accept a **linear wavelength grid** as input but internally
operate in log(lambda) space via the transformation:

.. math::

    v = c \\, \\ln(\\lambda / \\lambda_0)

where :math:`\\lambda_0` is the rest-frame wavelength of the line and
:math:`c` is the speed of light.

The profiles defined here are intended for use in *sheap* spectral fitting
pipelines where velocity-based parameterizations (FWHM, centroid shifts)
are preferred over linear wavelength widths.

Available profiles
------------------
- ``gaussian_fwhm_loglambda`` :
  Velocity-symmetric Gaussian profile.
- ``lorentzian_fwhm_loglambda`` :
  Lorentzian (Cauchy) profile in velocity space.
- ``skewed_gaussian_loglambda`` :
  Skew-normal Gaussian profile allowing asymmetric line shapes.
- ``top_hat_loglambda`` :
  Boxcar (top-hat) profile in velocity space.
- ``voigt_pseudo_loglambda`` :
  Pseudo-Voigt profile (Gaussian–Lorentzian mixture).
- ``emg_fwhm_loglambda`` :
  Exponentially Modified Gaussian (EMG) profile.

All profiles share a common parameterization:
- ``amplitude`` : Linear amplitude of the line.
- ``vshift_kms`` : Velocity shift of the centroid [km/s].
- ``fwhm_v_kms`` : log10 of the velocity FWHM [km/s].
- ``lambda0`` : Rest-frame wavelength of the line (fixed).

Some profiles include additional shape parameters (e.g. skewness,
Lorentzian mixing fraction, or exponential decay scale).

Notes
-----
- Using log(lambda) ensures exact symmetry in velocity space and avoids
  wavelength-dependent distortions present in linear-lambda profiles.
- These profiles are fully JAX-compatible and differentiable, enabling
  GPU-accelerated optimization and sampling.
- Physical bounds and initial values for these profiles are handled by
  the corresponding constraint-building utilities in *sheap*.

Examples
--------
.. code-block:: python

    import jax.numpy as jnp
    from sheap.Profiles.loglambda_profiles import gaussian_fwhm_loglambda

    x_lambda = jnp.linspace(6500.0, 6600.0, 2048)
    params = [1.0, 0.0, 3.0, 6563.0]  # amp, vshift, log10(FWHM), lambda0

    y = gaussian_fwhm_loglambda(x_lambda, params)
"""

__author__ = "felavila"



from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import jit, vmap,lax 
from jax.scipy.special import erfc
from jax.scipy.stats import norm #maybe dosent exist xd

from sheap.Profiles.Utils import with_param_names,trapz_jax
from sheap.Utils.Constants import C_KMS,FWHM_TO_SIGMA



__all__ = ["gaussian_fwhm_loglambda",
    "lorentzian_fwhm_loglambda",
    "skewed_gaussian_loglambda",
    "top_hat_loglambda",
    "voigt_pseudo_loglambda",
    "emg_fwhm_loglambda",]

@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "lambda0"])
def gaussian_fwhm_loglambda(x_lambda, params):
    """
    Velocity-symmetric Gaussian in log(lambda) space.

    Parameters
    ----------
    x_lambda : jnp.ndarray
        Wavelength grid.
    params : [amp, vshift_kms, fwhm_v_kms, lambda0]
        amp : Linear amplitude
        vshift_kms : Centroid velocity shift [km/s]
        fwhm_v_kms : log10(FWHM [km/s])
        lambda0 : Rest wavelength of the line (required, fixed)
    """
    amp, vshift_kms, fwhm_v_kms, lambda0 = params

    ratio = jnp.clip(x_lambda / lambda0, a_min=jnp.finfo(x_lambda.dtype).tiny, a_max=jnp.inf)
    y = C_KMS * jnp.log(ratio)

    fwhm_linear = 10.0 ** fwhm_v_kms
    sigma_v = fwhm_linear * FWHM_TO_SIGMA

    z = (y - vshift_kms) / sigma_v
    return amp * jnp.exp(-0.5 * z * z)


@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "lambda0"])
def lorentzian_fwhm_loglambda(x_lambda, params):
    """
    Lorentzian (Cauchy) profile evaluated in log(lambda) velocity space.

    Parameters
    ----------
    x_lambda : jnp.ndarray
        Wavelength grid (linear wavelength).
    params : [amplitude, vshift_kms, fwhm_v_kms, lambda0]
        amplitude : Linear amplitude.
        vshift_kms : Velocity shift of the centroid [km/s].
        fwhm_v_kms : log10(FWHM velocity width [km/s]).
        lambda0 : Rest-frame wavelength of the line.
    """
    amp, vshift_kms, fwhm_v_kms, lambda0 = params

    ratio = jnp.clip(x_lambda / lambda0,
                      a_min=jnp.finfo(x_lambda.dtype).tiny,
                      a_max=jnp.inf)
    y = C_KMS * jnp.log(ratio)

    fwhm = 10.0 ** fwhm_v_kms
    gamma = 0.5 * fwhm
    z = (y - vshift_kms) / gamma

    return amp / (1.0 + z * z)

@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "alpha", "lambda0"])
def skewed_gaussian_loglambda(x_lambda, params):
    """
    Skewed Gaussian (skew-normal) profile evaluated in log(lambda) space.

    Parameters
    ----------
    x_lambda : jnp.ndarray
        Wavelength grid (linear wavelength).
    params : [amplitude, vshift_kms, fwhm_v_kms, alpha, lambda0]
        amplitude : Linear amplitude.
        vshift_kms : Velocity shift of the centroid [km/s].
        fwhm_v_kms : log10(FWHM velocity width [km/s]).
        alpha : Skewness parameter (positive = red skew).
        lambda0 : Rest-frame wavelength of the line.
    """
    amp, vshift_kms, fwhm_v_kms, alpha, lambda0 = params

    ratio = jnp.clip(x_lambda / lambda0,
                      a_min=jnp.finfo(x_lambda.dtype).tiny,
                      a_max=jnp.inf)
    y = C_KMS * jnp.log(ratio)

    fwhm = 10.0 ** fwhm_v_kms
    sigma = fwhm * FWHM_TO_SIGMA
    t = (y - vshift_kms) / sigma

    phi = jnp.exp(-0.5 * t * t) / jnp.sqrt(2.0 * jnp.pi)
    Phi = 0.5 * (1.0 + jnp.erf(alpha * t / jnp.sqrt(2.0)))

    return amp * 2.0 * phi * Phi

@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "lambda0"])
def top_hat_loglambda(x_lambda, params):
    """
    Top-hat (boxcar) profile defined in log(lambda) velocity space.

    Parameters
    ----------
    x_lambda : jnp.ndarray
        Wavelength grid (linear wavelength).
    params : [amplitude, vshift_kms, fwhm_v_kms, lambda0]
        amplitude : Linear amplitude.
        vshift_kms : Velocity shift of the centroid [km/s].
        fwhm_v_kms : log10(FWHM velocity width [km/s]).
        lambda0 : Rest-frame wavelength of the line.
    """
    amp, vshift_kms, fwhm_v_kms, lambda0 = params

    ratio = jnp.clip(x_lambda / lambda0,
                      a_min=jnp.finfo(x_lambda.dtype).tiny,
                      a_max=jnp.inf)
    y = C_KMS * jnp.log(ratio)

    fwhm = 10.0 ** fwhm_v_kms
    half = 0.5 * fwhm

    return amp * (jnp.abs(y - vshift_kms) <= half)


@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "eta", "lambda0"])
def voigt_pseudo_loglambda(x_lambda, params):
    """
    Pseudo-Voigt profile (Gaussian + Lorentzian mixture)
    evaluated in log(lambda) velocity space.

    Parameters
    ----------
    x_lambda : jnp.ndarray
        Wavelength grid (linear wavelength).
    params : [amplitude, vshift_kms, fwhm_v_kms, eta, lambda0]
        amplitude : Linear amplitude.
        vshift_kms : Velocity shift of the centroid [km/s].
        fwhm_v_kms : log10(FWHM velocity width [km/s]).
        eta : Mixing fraction (eta=1 Lorentzian, eta=0 Gaussian).
        lambda0 : Rest-frame wavelength of the line.
    """
    amp, vshift_kms, fwhm_v_kms, eta, lambda0 = params

    ratio = jnp.clip(x_lambda / lambda0,
                      a_min=jnp.finfo(x_lambda.dtype).tiny,
                      a_max=jnp.inf)
    y = C_KMS * jnp.log(ratio)

    fwhm = 10.0 ** fwhm_v_kms

    sigma = fwhm * FWHM_TO_SIGMA
    gamma = 0.5 * fwhm

    tg = (y - vshift_kms) / sigma
    tl = (y - vshift_kms) / gamma

    G = jnp.exp(-0.5 * tg * tg)
    L = 1.0 / (1.0 + tl * tl)

    return amp * (eta * L + (1.0 - eta) * G)


@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "tau_kms", "lambda0"])
def emg_fwhm_loglambda(x_lambda, params):
    """
    Exponentially Modified Gaussian (EMG) profile evaluated in
    log(lambda) velocity space.

    Parameters
    ----------
    x_lambda : jnp.ndarray
        Wavelength grid (linear wavelength).
    params : [amplitude, vshift_kms, fwhm_v_kms, tau_kms, lambda0]
        amplitude : Linear amplitude.
        vshift_kms : Velocity shift of the centroid [km/s].
        fwhm_v_kms : log10(FWHM velocity width [km/s]).
        tau_kms : Exponential decay scale [km/s].
        lambda0 : Rest-frame wavelength of the line.
    """
    amp, vshift_kms, fwhm_v_kms, tau_kms, lambda0 = params

    ratio = jnp.clip(x_lambda / lambda0,
                      a_min=jnp.finfo(x_lambda.dtype).tiny,
                      a_max=jnp.inf)
    y = C_KMS * jnp.log(ratio)

    fwhm = 10.0 ** fwhm_v_kms
    sigma = fwhm * FWHM_TO_SIGMA
    tau = jnp.clip(tau_kms,
                   a_min=jnp.finfo(x_lambda.dtype).tiny,
                   a_max=jnp.inf)

    arg = (sigma / tau - (y - vshift_kms) / sigma) / jnp.sqrt(2.0)
    core = jnp.exp((sigma * sigma) / (2.0 * tau * tau)
                   - (y - vshift_kms) / tau)

    return amp * 0.5 / tau * core * erfc(arg)