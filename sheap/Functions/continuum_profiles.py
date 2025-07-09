import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
#import jax.scipy as jsp
#import numpy as np
from jax import jit, vmap

from sheap.Functions.utils import param_count,with_param_names



# @param_count(2)
# def linear(x, params):
#     return params[0] * (x / 1000.0) + params[1]


# @param_count(4)
# def brokenpowerlaw(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
#     """
#     Broken power law function in JAX.

#     Parameters
#     ----------
#     x : jnp.ndarray
#         Input wavelengths (Angstroms).
#     params : jnp.ndarray
#         Parameters array: [index1, index2, amplitude, refer]
#         - index1: slope for x <= refer
#         - index2: additional slope for x > refer
#         - amplitude: normalization at x = refer
#         - refer: reference wavelength (Angstroms)

#     Returns
#     -------
#     jnp.ndarray
#         Evaluated broken power law.
#     """
#     index1, index2, amplitude, refer = params
#     x = jnp.nan_to_num(x)

#     ratio = x / refer
#     # Create mask: x > refer gets index1 + index2; else gets index1
#     exponent = jnp.where(ratio > 1.0, index1 + index2, index1)
#     return amplitude * jnp.power(ratio, exponent)

# @param_count(2)
# def powerlaw(x, params):
#     x = jnp.nan_to_num(x)
#     return params[1] * jax.lax.pow(x / 1000.0, params[0])  # + params[1]


# @jit
def linear_combination(eieigenvectors, params):
    return jnp.nansum(eieigenvectors.T * 100 * params, axis=1)


# This requiere one more variable i guess.
@with_param_names(['amplitude', "T", 'τ0'])
def balmercontinuum(x, pars):
    """
    Compute the Balmer continuum (Dietrich+02) in pure JAX.

    Parameters
    ----------
    x : array-like
        Wavelengths in Angstrom.
    pars : array-like, shape (3,)
        pars[0] = A (amplitude)
        pars[1] = T (temperature in K)
        pars[2] = τ0 (optical‐depth scale)


    Returns
    -------
    result : ndarray
        Balmer continuum flux in the same shape as x.
    """
    # Constants
    h = 6.62607015e-34  # Planck’s constant, J·s
    c = 2.99792458e8  # Speed of light, m/s
    k_B = 1.380649e-23  # Boltzmann constant, J/K

    # Edge
    lambda_BE = 3646.0  # Å

    # Convert Å → m
    lam_m = x * 1e-10

    # Planck function B_λ(lam_m, T) [SI units]
    T = pars[1]
    exponent = h * c / (lam_m * k_B * T)
    B_lambda = (2.0 * h * c**2) / (lam_m**5 * (jnp.exp(exponent) - 1.0))

    # Apply the same “scale=10000” factor as in astropy’s BlackBody
    B_lambda *= 1e4

    # Optical depth τ(λ)
    tau = pars[2] * (x / lambda_BE) ** 3

    # Balmer-continuum formula
    result = pars[0] * B_lambda * (1.0 - jnp.exp(-tau))

    # Zero above the Balmer edge
    result = jnp.where(x > lambda_BE, 0.0, result) / 1e18  # factor the normalisacion

    return result
#############################

# import jax.numpy as jnp
# from typing import Mapping, Callable
# from sheap.Functions.profiles import with_param_names

# Basic continuum models with normalized wavelength (x/1000) ---------------------------
@with_param_names(["amplitude_slope", "amplitude_intercept"])
def linear(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """f(x) = intercept + slope * (x/1000)"""
    x = xs / 1000.0
    slope,intercept = params
    return intercept + slope * x

@with_param_names(["alpha","amplitude"])
def powerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """f(x) = amplitude * (x/1000)**alpha"""
    x = xs / 1000.0
    alpha, amplitude = params
    return amplitude * x**alpha

@with_param_names(["amplitude", "alpha1", "alpha2", "x_break"])
def brokenpowerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    f(x) = amplitude * (x/1000)**alpha1                              if x < x_break
         = amplitude * x_break**(alpha1-alpha2) * (x/1000)**alpha2   otherwise
    """
    x = xs / 1000.0
    amplitude, alpha1, alpha2, x_break = params
    x_break = x_break/1000
    low  = amplitude * x**alpha1
    high = amplitude * (x_break**(alpha1 - alpha2)) * x**alpha2
    return jnp.where(x < x_break, low, high)


@with_param_names(["amplitude", "alpha", "beta"])
def logparabola(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    f(x) = amplitude * (x/x0)**(-alpha - beta * log(x/x0)), with x0 = mean(x/1000)
    """
    x = xs / 1000.0
    amplitude, alpha, beta = params
    x0 = jnp.mean(x)
    return amplitude * (x / x0) ** (-alpha - beta * jnp.log(x / x0))

@with_param_names(["amplitude", "alpha", "x_cut"])
def exp_cutoff(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """f(x) = amplitude * (x/1000)**(-alpha) * exp(-x / x_cut)"""
    x = xs / 1000.0
    amplitude, alpha, x_cut = params
    return amplitude * x**(-alpha) * jnp.exp(-x / x_cut)

@with_param_names(["amplitude", "c0", "c1", "c2", "c3"])
def polynomial(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """f(x) = amplitude * (c0 + c1 x + c2 x^2 + c3 x^3) with x normalized to 1000 Å"""
    x = xs / 1000.0
    amplitude, *coeffs = params
    # coeffs corresponds to [c0, c1, c2, c3]
    # polyval expects highest to lowest: [c3, c2, c1, c0]
    poly_vals = jnp.polyval(jnp.flip(jnp.array(coeffs)), x)
    return amplitude * poly_vals