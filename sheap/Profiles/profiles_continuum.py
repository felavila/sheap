import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jit, vmap

from sheap.Profiles.utils import with_param_names


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

λ0 = 5500.0  # Å myabe do this make it more problematic in some cases

@with_param_names(["amplitude_slope", "amplitude_intercept"])
def linear(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """f(λ) = intercept + slope * (λ/λ0)"""
    slope, intercept = params
    x = xs / λ0
    return intercept + slope * x


@with_param_names(["alpha","amplitude"])
def powerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """f(λ) = amplitude * (λ/λ0)**alpha"""
    α, A = params
    x = xs / λ0
    return A * x**α


@with_param_names(["amplitude", "alpha1", "alpha2", "x_break"])
def brokenpowerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    f(λ) = A⋅(λ/λ0)**α1                              if λ/λ0 < x_break
         = A⋅x_break**(α1-α2)⋅(λ/λ0)**α2            otherwise
    """
    A, α1, α2, xbr = params
    x = xs / λ0
    xbr = xbr / λ0
    low  = A * x**α1
    high = A * (xbr**(α1 - α2)) * x**α2
    return jnp.where(x < xbr, low, high)


@with_param_names(["amplitude", "alpha", "beta"])
def logparabola(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    f(λ) = A * ( (λ/λ0) )**(-α - β*log(λ/λ0) )
    """
    A, α, β = params
    x = xs / λ0
    return A * x**(-α - β * jnp.log(x))


@with_param_names(["amplitude", "alpha", "x_cut"])
def exp_cutoff(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """f(λ) = A * (λ/λ0)**(-α) * exp(-λ/(x_cut))"""
    A, α, xcut = params
    x = xs / λ0
    return A * x**(-α) * jnp.exp(-xs / xcut)


@with_param_names(["amplitude", "c1", "c2", "c3"])
def polynomial(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    f(λ) = A * (1 + c1·(λ/λ0) + c2·(λ/λ0)^2 + c3·(λ/λ0)^3)
    """
    A, c1, c2, c3 = params
    x = xs / λ0
    return A * (1 + c1*x + c2*x**2 + c3*x**3)
####
def linear_combination(eieigenvectors, params):
    return jnp.nansum(eieigenvectors.T * 100 * params, axis=1)