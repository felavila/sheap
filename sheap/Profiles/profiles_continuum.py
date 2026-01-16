"""
Continuum Profiles
==================

This module defines all continuum profile functions available in *sheap*.
Each function is JAX-compatible and decorated with ``@with_param_names``
to provide consistent parameter naming for fitting routines.

Profiles
--------
- ``linear``          : Linear continuum (slope + intercept).
- ``powerlaw``        : Standard power law anchored at λ₀=5500 Å.
- ``brokenpowerlaw``  : Two-slope power law with a break wavelength.
- ``logparabola``     : Log-parabolic shape with curvature term.
- ``exp_cutoff``      : Power law with exponential cutoff.
- ``polynomial``      : Cubic polynomial expansion.

Constants
---------
- ``delta0`` : Reference wavelength (5500 Å) used for continuum scaling.

Notes
-----
- All functions take wavelength arrays in Ångström and return dimensionless
    continuum templates scaled by their amplitude parameter.
- The reference wavelength ``delta0`` ensures consistent normalization
    across continuum forms.
"""

__author__ = 'felavila'


__all__ = [
    "brokenpowerlaw",
    "delta0",
    "exp_cutoff",
    "linear",
    "logparabola",
    "polynomial",
    "powerlaw",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jit, vmap

from sheap.Profiles.Utils import with_param_names


"""
Note
--------
delta0 : Reference wavelength (5500 Å) used for continuum scaling.
"""

delta0 = 5500.0  #: Normalization wavelength in Ångström used for continuum models (λ/λ₀)

#TODO Check in the profiles with only one amplitude -> move all to logamp

@with_param_names(["amplitude_slope", "amplitude_intercept"])
def linear(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Linear continuum profile.

    .. math::
        f(\lambda) = \text{intercept} + \text{slope} \cdot \left(\frac{\lambda}{\lambda_0}\right)

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



@with_param_names(["logamp","alpha"])
def powerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Power-law continuum profile.

    .. math::
        f(\lambda) = A \cdot \left(\frac{\lambda}{\lambda_0}\right)^{\alpha}

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
    A, alpha = params
    x = xs / delta0
    return 10**A * x ** alpha


@with_param_names(["logamp", "alpha1", "alpha2", "x_break"])
def brokenpowerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Broken power-law continuum profile.

    .. math::
        f(\lambda) =
        \begin{cases}
        A \left(\dfrac{\lambda}{\lambda_0}\right)^{\alpha_1}, & \text{if } \lambda < x_{\text{break}} \\
        A \, x_{\text{break}}^{\alpha_1 - \alpha_2} \left(\dfrac{\lambda}{\lambda_0}\right)^{\alpha_2}, & \text{otherwise}
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
    low = 10**A * x ** alpha1
    high = 10**A * (xbr ** (alpha1 - alpha2)) * x ** alpha2
    return jnp.where(x < xbr, low, high)


@with_param_names(["amplitude", "alpha", "beta"])
def logparabola(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    r"""
    Log-parabolic continuum profile.

    .. math::
        f(\lambda) = A \cdot \left(\frac{\lambda}{\lambda_0}\right)^{-\alpha - \beta \cdot \log(\lambda / \lambda_0)}

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
        f(\lambda) = A \cdot \left(\frac{\lambda}{\lambda_0}\right)^{-\alpha} \cdot \exp\left(-\frac{\lambda}{x_{cut}}\right)

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
        f(\lambda) = A \cdot \left(1 + c_1 \cdot x + c_2 \cdot x^2 + c_3 \cdot x^3\right), \quad x = \frac{\lambda}{\lambda_0}

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



