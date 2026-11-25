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
- ``polynomial``      : Polynomial expansion.

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

import numpy as np

import jax
import jax.numpy as jnp
from jax import jit, vmap

from sheap.Profiles.Utils import with_param_names,add_linear_design_builder


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

########################################################################################################################

def make_linear_function(delta0: float = 5500.0):
    """Linear continuum: both slope and intercept are linear parameters."""
    @with_param_names(
        ["amplitude_slope", "amplitude_intercept"],
        linear_param_names=["amplitude_slope", "amplitude_intercept"],
        profile_name="linear",
    )
    def linear(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        slope, intercept = params
        return intercept + slope * (xs / delta0)

    def design_builder(xs, params):
        xs = jnp.asarray(xs)
        columns = jnp.stack([xs / delta0, jnp.ones_like(xs)], axis=-1)
        return jnp.zeros_like(xs), columns

    def numpy_model(xs, params):
        xs = np.asarray(xs, dtype=np.float64)
        slope, intercept = np.asarray(params, dtype=np.float64)
        return intercept + slope * (xs / delta0)

    def numpy_design_builder(xs, params):
        xs = np.asarray(xs, dtype=np.float64).reshape(-1)
        columns = np.column_stack((xs / delta0, np.ones_like(xs)))
        return np.zeros_like(xs), columns

    linear.design_builder = design_builder
    linear.numpy_model = numpy_model
    linear.numpy_design_builder = numpy_design_builder
    return linear


def make_powerlaw_function(delta0: float = 5500.0):
    """Power law with logarithmic amplitude (all parameters nonlinear)."""
    @with_param_names(["logamp", "alpha"], profile_name="powerlaw")
    def powerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        A, alpha = params
        return 10.0**A * (xs / delta0)**alpha

    def design_builder(xs, params):
        offset = powerlaw(xs, params)
        return offset, jnp.empty((xs.shape[0], 0), dtype=offset.dtype)

    def numpy_model(xs, params):
        xs = np.asarray(xs, dtype=np.float64)
        A, alpha = np.asarray(params, dtype=np.float64)
        return 10.0**A * (xs / delta0)**alpha

    def numpy_design_builder(xs, params):
        offset = np.asarray(numpy_model(xs, params), dtype=np.float64).reshape(-1)
        return offset, np.empty((offset.size, 0), dtype=np.float64)

    powerlaw.design_builder = design_builder
    powerlaw.numpy_model = numpy_model
    powerlaw.numpy_design_builder = numpy_design_builder
    return powerlaw


def make_brokenpowerlaw_function(delta0: float = 5500.0):
    """Broken power law with logarithmic amplitude (all nonlinear)."""
    @with_param_names(
        ["logamp", "alpha1", "alpha2", "x_break"],
        profile_name="brokenpowerlaw",
    )
    def brokenpowerlaw(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        A, alpha1, alpha2, xbr = params
        x = xs / delta0
        xbr = xbr / delta0
        low = 10.0**A * x**alpha1
        high = 10.0**A * xbr**(alpha1 - alpha2) * x**alpha2
        return jnp.where(x < xbr, low, high)

    def design_builder(xs, params):
        offset = brokenpowerlaw(xs, params)
        return offset, jnp.empty((xs.shape[0], 0), dtype=offset.dtype)

    def numpy_model(xs, params):
        xs = np.asarray(xs, dtype=np.float64)
        A, alpha1, alpha2, xbr = np.asarray(params, dtype=np.float64)
        x = xs / delta0
        xbr = xbr / delta0
        low = 10.0**A * x**alpha1
        high = 10.0**A * xbr**(alpha1 - alpha2) * x**alpha2
        return np.where(x < xbr, low, high)

    def numpy_design_builder(xs, params):
        offset = np.asarray(numpy_model(xs, params), dtype=np.float64).reshape(-1)
        return offset, np.empty((offset.size, 0), dtype=np.float64)

    brokenpowerlaw.design_builder = design_builder
    brokenpowerlaw.numpy_model = numpy_model
    brokenpowerlaw.numpy_design_builder = numpy_design_builder
    return brokenpowerlaw


def make_exp_cutoff_function(delta0: float = 5500.0):
    """Cutoff power law: linear amplitude, nonlinear alpha and x_cut."""
    @with_param_names(
        ["amplitude", "alpha", "x_cut"],
        linear_param_names=["amplitude"],
        profile_name="exp_cutoff",
    )
    def exp_cutoff(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        A, alpha, xcut = params
        return A * (xs / delta0)**(-alpha) * jnp.exp(-xs / xcut)

    def design_builder(xs, params):
        xs = jnp.asarray(xs)
        _, alpha, xcut = params
        unit = (xs / delta0)**(-alpha) * jnp.exp(-xs / xcut)
        return jnp.zeros_like(unit), unit[:, None]

    def numpy_model(xs, params):
        xs = np.asarray(xs, dtype=np.float64)
        A, alpha, xcut = np.asarray(params, dtype=np.float64)
        return A * (xs / delta0)**(-alpha) * np.exp(-xs / xcut)

    def numpy_design_builder(xs, params):
        xs = np.asarray(xs, dtype=np.float64).reshape(-1)
        _, alpha, xcut = np.asarray(params, dtype=np.float64)
        unit = (xs / delta0)**(-alpha) * np.exp(-xs / xcut)
        return np.zeros_like(xs), unit[:, None]

    exp_cutoff.design_builder = design_builder
    exp_cutoff.numpy_model = numpy_model
    exp_cutoff.numpy_design_builder = numpy_design_builder
    return exp_cutoff


def make_polynomial_function(degree: int, delta0: float = 5500.0):
    """Exponentiated polynomial with logarithmic amplitude (all nonlinear)."""
    if degree < 0:
        raise ValueError("degree must be >= 0")

    param_names = ["logamp"] + [f"c{i}" for i in range(1, degree + 1)]

    @with_param_names(param_names, profile_name="polynomial")
    def polynomial(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        A = params[0]
        coeffs = params[1:]
        x = (xs - delta0) / delta0
        corr = jnp.zeros_like(x)
        # Preserve original SHEAP Horner evaluation (c1*x + c2*x**2 + ...).
        for c in coeffs[::-1]:
            corr = (corr + c) * x
        return jnp.exp(A * jnp.log(10.0) + corr)

    def design_builder(xs, params):
        offset = polynomial(xs, params)
        return offset, jnp.empty((xs.shape[0], 0), dtype=offset.dtype)

    def numpy_model(xs, params):
        xs = np.asarray(xs, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        A, coeffs = p[0], p[1:]
        x = (xs - delta0) / delta0
        corr = np.zeros_like(x)
        for c in coeffs[::-1]:
            corr = (corr + c) * x
        return np.exp(A * np.log(10.0) + corr)

    def numpy_design_builder(xs, params):
        offset = np.asarray(numpy_model(xs, params), dtype=np.float64).reshape(-1)
        return offset, np.empty((offset.size, 0), dtype=np.float64)

    polynomial.design_builder = design_builder
    polynomial.numpy_model = numpy_model
    polynomial.numpy_design_builder = numpy_design_builder
    return polynomial


def make_logparabola_function(delta0: float = 5500.0):
    """Log-parabolic continuum: linear amplitude, nonlinear alpha and beta."""
    @with_param_names(
        ["amplitude", "alpha", "beta"],
        linear_param_names=["amplitude"],
        profile_name="logparabola",
    )
    def logparabola(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        A, alpha, beta = params
        x = xs / delta0
        return A * x**(-alpha - beta * jnp.log(x))

    def design_builder(xs, params):
        xs = jnp.asarray(xs)
        _, alpha, beta = params
        x = xs / delta0
        unit = x**(-alpha - beta * jnp.log(x))
        return jnp.zeros_like(unit), unit[:, None]

    def numpy_model(xs, params):
        xs = np.asarray(xs, dtype=np.float64)
        A, alpha, beta = np.asarray(params, dtype=np.float64)
        x = xs / delta0
        return A * x**(-alpha - beta * np.log(x))

    def numpy_design_builder(xs, params):
        xs = np.asarray(xs, dtype=np.float64).reshape(-1)
        _, alpha, beta = np.asarray(params, dtype=np.float64)
        x = xs / delta0
        unit = x**(-alpha - beta * np.log(x))
        return np.zeros_like(xs), unit[:, None]

    logparabola.design_builder = design_builder
    logparabola.numpy_model = numpy_model
    logparabola.numpy_design_builder = numpy_design_builder
    return logparabola
