"""This module contains all the continuum profiles available in sheap."""
__author__ = 'felavila'


__all__ = [
    "balmercontinuum",
    "brokenpowerlaw",
    "delta0",
    "exp_cutoff",
    "linear",
    "linear_combination",
    "logparabola",
    "polynomial",
    "powerlaw",
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

# # This requiere one more variable i guess.
# @with_param_names(["amplitude", "T", "τ0"])
# def balmercontinuum(x, pars):
#     """
#     Compute the Balmer continuum using the Dietrich+2002 prescription.

#     The model follows:
#     .. math::
#         f(\\lambda) = A \\cdot B_{\\lambda}(T) \\cdot \\left(1 - e^{-\\tau(\\lambda)}\\right)

#     where:
#     - :math:`B_{\\lambda}(T)` is the Planck function in wavelength units.
#     - :math:`\\tau(\\lambda) = \\tau_0 \\cdot (\\lambda / \\lambda_{BE})^3`
#     - :math:`\\lambda_{BE} = 3646` Å is the Balmer edge.

#     Parameters
#     ----------
#     x : array-like
#         Wavelengths in Ångström.
#     pars : array-like, shape (3,)
#         - `pars[0]`: Amplitude :math:`A`
#         - `pars[1]`: Temperature :math:`T` (in Kelvin)
#         - `pars[2]`: Optical depth scale :math:`\\tau_0`

#     Returns
#     -------
#     jnp.ndarray
#         Flux array with same shape as `x`.
#     """
#     # Constants
#     h = 6.62607015e-34  # Planck’s constant, J·s
#     c = 2.99792458e8  # Speed of light, m/s
#     k_B = 1.380649e-23  # Boltzmann constant, J/K

#     # Edge
#     lambda_BE = 3646.0  # Å

#     lam_m = x * 1e-10

#     T = pars[1]
#     exponent = h * c / (lam_m * k_B * T)
#     B_lambda = (2.0 * h * c ** 2) / (lam_m ** 5 * (jnp.exp(exponent) - 1.0))

#     # Apply the same “scale=10000” factor as in astropy’s BlackBody
#     B_lambda *= 1e4

#     tau = pars[2] * (x / lambda_BE) ** 3

#     result = pars[0] * B_lambda * (1.0 - jnp.exp(-tau))

#     result = jnp.where(x > lambda_BE, 0.0, result) / 1e18  # factor the normalisacion

#     return result

@with_param_names(["amplitude", "T", "tau0"])
def balmercontinuum(x, pars):
    """
    Balmer continuum (Dietrich+2002–style) normalized at the Balmer edge.

    The unnormalized model is:
        f(λ) = A · B_λ(T) · [1 - exp(-τ(λ))],   for λ ≤ λ_BE
    with
        τ(λ) = τ0 · (λ / λ_BE)^3,     λ_BE = 3646 Å.

    Here we normalize the *shape* by its value at λ_BE:
        f_norm(λ) = [B_λ(T) · (1 - exp(-τ(λ)))] / [B_λ(T) · (1 - exp(-τ0))]_{λ=λ_BE}
    so that the returned spectrum is:
        F(λ) = amplitude · f_norm(λ),  for λ ≤ λ_BE, and 0 otherwise.

    Parameters
    ----------
    x : array-like
        Wavelengths in Å (vacuum).
    pars : array-like, shape (3,)
        pars[0] -> amplitude (dimensionless weight; scales with SHEAP's global scale)
        pars[1] -> T (K), electron temperature controlling the shape
        pars[2] -> τ0, optical-depth scale at λ_BE

    Returns
    -------
    jnp.ndarray
        Dimensionless template scaled by `amplitude`; zero for λ > λ_BE.

    Notes
    -----
    - Because of the edge normalization, physical units cancel out; only the
      *relative* shape matters. The overall flux scaling should come from sheap’s
      rescaling pipeline (your global `scale`), via this component’s `amplitude`.
    - This function intentionally avoids extra constants (e.g., 1e18) or Astropy
      scaling factors. If you need a different normalization (e.g., unit integral
      over 3000–3646 Å), we can provide that variant.

    Math
    ----
    .. math::

        F(\\lambda) = A\\;\\frac{B_{\\lambda}(T)\\,[1 - e^{-\\tau_0 (\\lambda/\\lambda_{\\rm BE})^3}]}
                               {B_{\\lambda_{\\rm BE}}(T)\\,[1 - e^{-\\tau_0}]}
        \\quad (\\lambda \\le \\lambda_{\\rm BE}),\\; 0\\;\\text{otherwise}.

    """


    # Physical constants (SI) – they cancel in the edge normalization, but we keep them for clarity.
    h = 6.62607015e-34      # J s
    c = 2.99792458e8        # m s^-1
    k_B = 1.380649e-23      # J K^-1

    lambda_BE = 3646.0  # Å (Balmer edge)
    lam_m = x * 1e-10   # Å -> m

    A   = pars[0]
    T   = pars[1]
    tau0 = pars[2]

    # Planck function B_lambda(T) in SI (W m^-3 sr^-1). Units cancel after normalization.
    exponent = (h * c) / (lam_m * k_B * jnp.clip(T, 1.0, jnp.inf))
    B_lambda = (2.0 * h * c**2) / (jnp.clip(lam_m, 1e-30, jnp.inf)**5 * (jnp.exp(exponent) - 1.0))

    # Optical depth law
    tau = tau0 * (x / lambda_BE) ** 3

    # Unnormalized shape (only defined blueward of the edge)
    raw = B_lambda * (1.0 - jnp.exp(-tau))

    # Edge value for normalization (evaluate analytically at λ_BE)
    lam_BE_m = lambda_BE * 1e-10
    exponent_BE = (h * c) / (lam_BE_m * k_B * jnp.clip(T, 1.0, jnp.inf))
    B_lambda_BE = (2.0 * h * c**2) / (lam_BE_m**5 * (jnp.exp(exponent_BE) - 1.0))
    norm_edge = B_lambda_BE * (1.0 - jnp.exp(-jnp.clip(tau0, 0.0, jnp.inf)))

    # Avoid division by zero if tau0 ~ 0 or extreme T
    norm_edge = jnp.clip(norm_edge, 1e-300, jnp.inf)

    f_norm = raw / norm_edge

    # Zero redward of the edge
    f_norm = jnp.where(x <= lambda_BE, f_norm, 0.0)

    return A * f_norm

############################
# @with_param_names(["amplitude", "T", "tau0"])
# def balmercontinuum(x, pars):
#     """
#     Dietrich+2002-style Balmer continuum, edge-normalized at λ_BE=3646 Å.
#     Returns A * f_norm(λ) for λ ≤ λ_BE, else 0.
#     """
#     # Physical constants (SI)
#     h  = 6.62607015e-34      # J s
#     c  = 2.99792458e8        # m s^-1
#     kB = 1.380649e-23        # J K^-1

#     lambda_BE = 3646.0  # Å
#     A   = pars[0]
#     T   = jnp.clip(pars[1], 1.0, jnp.inf)   # avoid T<=0
#     tau0 = jnp.clip(pars[2], 1e-3, jnp.inf)  # optical depth scale ≥0

#     # Work only blueward
#     x = jnp.asarray(x)
#     in_blue = x <= lambda_BE
#     x_safe  = jnp.clip(x, 1e-6, jnp.inf)        # Å, avoid 0 Å
#     lam_m   = x_safe * 1e-10
#     lamBE_m = lambda_BE * 1e-10

#     # a = hc/(kB T)
#     a = (h * c) / (kB * T)                      # meters

#     # Planck ratio B_lambda / B_lambda_BE using expm1:
#     # ratio = (λ_BE/λ)^5 * (expm1(a/λ_BE)) / (expm1(a/λ))
#     def _expm1_pos(z):
#         # guard against extremely large z (expm1(large) ~ exp(z))
#         return jnp.where(z > 50.0, jnp.exp(z), jnp.expm1(z))

#     z  = a / lam_m
#     zB = a / lamBE_m
#     # For tiny z (very large T), expm1(z) ~ z; use jnp.where to keep it stable.
#     denom = jnp.where(z < 1e-6, z, _expm1_pos(z))
#     numer = jnp.where(zB < 1e-6, zB, _expm1_pos(zB))
#     planck_ratio = (lamBE_m / lam_m) ** 5 * (numer / jnp.clip(denom, 1e-300, jnp.inf))

#     # τ(λ) and stable (1 - exp(-τ)) using expm1
#     tau = tau0 * (x_safe / lambda_BE) ** 3
#     one_minus_e_m_tau  = -jnp.expm1(-jnp.clip(tau, 0.0, jnp.inf))
#     one_minus_e_m_tau0 = -jnp.expm1(-jnp.clip(tau0, 0.0, jnp.inf))
#     # If tau0 == 0 => both numerator and denominator → 0; the ratio → (λ/λ_BE)^3
#     tau_ratio = jnp.where(
#         one_minus_e_m_tau0 > 0.0,
#         one_minus_e_m_tau / one_minus_e_m_tau0,
#         (x_safe / lambda_BE) ** 3,  # small-τ limit
#     )

#     f_norm = planck_ratio * tau_ratio
#     f_norm = jnp.where(in_blue, f_norm, 0.0)

#     # Clean any residual numerical junk
#     f_norm = jnp.nan_to_num(f_norm, nan=0.0, posinf=0.0, neginf=0.0)

#     return A * f_norm


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
