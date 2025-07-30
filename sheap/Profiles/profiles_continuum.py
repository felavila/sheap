"""This module contains all the continuum profiles available in sheap."""
__version__ = "0.1.0"
__author__ = "Felipe Avila-Vera"

__all__ = [
    "balmercontinuum",
    "brokenpowerlaw",
    "exp_cutoff",
    "linear",
    "linear_combination",
    "logparabola",
    "polynomial",
    "powerlaw",
    "delta0",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jit, vmap

from sheap.Profiles.utils import with_param_names


"""
Note
--------
delta0 : Reference wavelength (5500 Å) used for continuum scaling.
"""

delta0 = 5500.0  #: Normalization wavelength in Ångström used for continuum models (λ/λ₀)

# This requiere one more variable i guess.
@with_param_names(["amplitude", "T", "τ0"])
def balmercontinuum(x, pars):
    """
    Compute the Balmer continuum using the Dietrich+2002 prescription.

    The model follows:
    .. math::
        f(\\lambda) = A \\cdot B_{\\lambda}(T) \\cdot \\left(1 - e^{-\\tau(\\lambda)}\\right)

    where:
    - :math:`B_{\\lambda}(T)` is the Planck function in wavelength units.
    - :math:`\\tau(\\lambda) = \\tau_0 \\cdot (\\lambda / \\lambda_{BE})^3`
    - :math:`\\lambda_{BE} = 3646` Å is the Balmer edge.

    Parameters
    ----------
    x : array-like
        Wavelengths in Ångström.
    pars : array-like, shape (3,)
        - `pars[0]`: Amplitude :math:`A`
        - `pars[1]`: Temperature :math:`T` (in Kelvin)
        - `pars[2]`: Optical depth scale :math:`\\tau_0`

    Returns
    -------
    jnp.ndarray
        Flux array with same shape as `x`.
    """
    # Constants
    h = 6.62607015e-34  # Planck’s constant, J·s
    c = 2.99792458e8  # Speed of light, m/s
    k_B = 1.380649e-23  # Boltzmann constant, J/K

    # Edge
    lambda_BE = 3646.0  # Å

    lam_m = x * 1e-10

    T = pars[1]
    exponent = h * c / (lam_m * k_B * T)
    B_lambda = (2.0 * h * c ** 2) / (lam_m ** 5 * (jnp.exp(exponent) - 1.0))

    # Apply the same “scale=10000” factor as in astropy’s BlackBody
    B_lambda *= 1e4

    tau = pars[2] * (x / lambda_BE) ** 3

    result = pars[0] * B_lambda * (1.0 - jnp.exp(-tau))

    result = jnp.where(x > lambda_BE, 0.0, result) / 1e18  # factor the normalisacion

    return result


#############################


@with_param_names(["amplitude_slope", "amplitude_intercept"])
def linear(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Linear continuum profile.

    .. math::
        f(\\lambda) = \text{intercept} + \text{slope} \cdot \left(\\frac{\\lambda}{\\lambda_0}\right)

    Parameters
    ----------
    xs : jnp.ndarray
        Wavelengths in Ångström.
    params : array-like
        - `params[0]`: Slope
        - `params[1]`: Intercept

    Returns
    -------
    jnp.ndarray
        Evaluated flux.
    """
    slope, intercept = params
    x = xs / delta0
    return intercept + slope * x


@with_param_names(["alpha", "amplitude"])
def powerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Power-law continuum profile.

    .. math::
        f(\\lambda) = A \cdot \left(\\frac{\\lambda}{\\lambda_0}\right)^{\alpha}

    Parameters
    ----------
    xs : jnp.ndarray
        Wavelengths in Ångström.
    params : array-like
        - `params[0]`: Slope :math:`\alpha`
        - `params[1]`: Amplitude :math:`A`
    Returns
    -------
    jnp.ndarray
        Evaluated flux.
    """
    alpha, A = params
    x = xs / delta0
    return A * x ** alpha


@with_param_names(["amplitude", "alpha1", "alpha2", "x_break"])
def brokenpowerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Broken power-law continuum profile.

    .. math::
        f(\\lambda) =
        \begin{cases}
            A \cdot \left(\\frac{\\lambda}{\\lambda_0}\right)^{\alpha_1} & \\text{if } \\lambda < x_{\\text{break}} \\\\
            A \cdot x_{\\text{break}}^{\alpha_1 - \alpha_2} \cdot \left(\\frac{\\lambda}{\\lambda_0}\right)^{\alpha_2} & \\text{otherwise}
        \end{cases}

    Parameters
    ----------
    xs : jnp.ndarray
        Wavelengths in Ångström.
    params : array-like
        - `params[0]`: Amplitude :math:`A`
        - `params[1]`: Slope below break :math:`\alpha_1`
        - `params[2]`: Slope above break :math:`\alpha_2`
        - `params[3]`: Break wavelength :math:`x_{break}` in Ångström

    Returns
    -------
    jnp.ndarray
        Evaluated flux.
    """
    A, alpha1, alpha2, xbr = params
    x = xs / delta0
    xbr = xbr / delta0
    low = A * x ** alpha1
    high = A * (xbr ** (alpha1 - alpha2)) * x ** alpha2
    return jnp.where(x < xbr, low, high)


@with_param_names(["amplitude", "alpha", "beta"])
def logparabola(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Log-parabolic continuum profile.

    .. math::
        f(\\lambda) = A \cdot \left(\\frac{\\lambda}{\\lambda_0}\right)^{-\\alpha - \\beta \cdot \log(\\lambda / \\lambda_0)}

    Parameters
    ----------
    xs : jnp.ndarray
        Wavelengths in Ångström.
    params : array-like
        - `params[0]`: Amplitude :math:`A`
        - `params[1]`: Spectral index :math:`\alpha`
        - `params[2]`: Curvature parameter :math:`\beta`

    Returns
    -------
    jnp.ndarray
        Evaluated flux.
    """
    A, alpha, beta = params
    x = xs / delta0
    return A * x ** (-alpha - beta * jnp.log(x))


@with_param_names(["amplitude", "alpha", "x_cut"])
def exp_cutoff(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Power-law with exponential cutoff.

    .. math::
        f(\\lambda) = A \cdot \left(\\frac{\\lambda}{\\lambda_0}\right)^{-\\alpha} \cdot \exp\left(-\\frac{\\lambda}{x_{cut}}\right)

    Parameters
    ----------
    xs : jnp.ndarray
        Wavelengths in Ångström.
    params : array-like
        - `params[0]`: Amplitude :math:`A`
        - `params[1]`: Slope :math:`\alpha`
        - `params[2]`: Cutoff wavelength :math:`x_{cut}` in Ångström

    Returns
    -------
    jnp.ndarray
        Evaluated flux.
    """
    A, alpha, xcut = params
    x = xs / delta0
    return A * x ** (-alpha) * jnp.exp(-xs / xcut)


@with_param_names(["amplitude", "c1", "c2", "c3"])
def polynomial(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Cubic polynomial continuum profile.

    .. math::
        f(\\lambda) = A \cdot \left(1 + c_1 \cdot x + c_2 \cdot x^2 + c_3 \cdot x^3\right), \quad x = \\frac{\\lambda}{\\lambda_0}

    Parameters
    ----------
    xs : jnp.ndarray
        Wavelengths in Ångström.
    params : array-like
        - `params[0]`: Amplitude :math:`A`
        - `params[1]`: Coefficient :math:`c_1`
        - `params[2]`: Coefficient :math:`c_2`
        - `params[3]`: Coefficient :math:`c_3`

    Returns
    -------
    jnp.ndarray
        Evaluated flux.
    """
    A, c1, c2, c3 = params
    x = xs / delta0
    return A * (1 + c1 * x + c2 * x ** 2 + c3 * x ** 3)


####
def linear_combination(eieigenvectors, params):
    return jnp.nansum(eieigenvectors.T * 100 * params, axis=1)
