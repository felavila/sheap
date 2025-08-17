"""This module contains all the line profiles available in sheap."""
__author__ = 'felavila'


__all__ = [
    "emg_fwhm",
    "eval_hermite",
    "gauss_hermite_losvd_jax",
    "gaussian_fwhm",
    "lorentzian_fwhm",
    "skewed_gaussian",
    "top_hat",
    "voigt_pseudo",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import jit, vmap,lax 
from jax.scipy.special import erfc
from jax.scipy.stats import norm #maybe dosent exist xd

from sheap.Profiles.utils import with_param_names,trapz_jax

@with_param_names(["logamp", "center", "fwhm"])
def gaussian_fwhm(x, params):
    r"""
    Standard Gaussian line profile using FWHM.

    .. math::
        f(x) = A \cdot \exp\left( -\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^2 \right)

    where:
    - :math:`A = 10^{\mathrm{logamp}}`
    - :math:`\sigma = \mathrm{fwhm} / 2.355`

    Parameters
    ----------
    x : jnp.ndarray
        Input wavelength array.
    params : array-like
        - `logamp`: Log base-10 amplitude.
        - `center`: Line center.
        - `fwhm`: Full width at half maximum.

    Returns
    -------
    jnp.ndarray
        Profile evaluated at `x`.
    """
    log_amp, center, fwhm = params
    #center = 10**logcenter
    amplitude = 10**log_amp 
    #amplitude = jnp.sign(log_amp) *10 ** jnp.abs(log_amp)
    #amplitude = log_amp 
    sigma = fwhm / 2.355 #fwhm -> logfwhm
    return amplitude * jnp.exp(-0.5 * ((x - center) / sigma) ** 2)


@with_param_names(["logamp", "center", "fwhm"])
def lorentzian_fwhm(x, params):
    r"""
    Lorentzian line profile using FWHM.

    .. math::
        f(x) = \frac{A}{1 + \left( \frac{x - \mu}{\gamma} \right)^2 }

    where:
    - :math:`A = 10^{\mathrm{logamp}}`
    - :math:`\gamma = \mathrm{fwhm} / 2`

    Parameters
    ----------
    x : jnp.ndarray
        Input wavelength array.
    params : array-like
        - `logamp`: Log base-10 amplitude.
        - `center`: Line center.
        - `fwhm`: Full width at half maximum.

    Returns
    -------
    jnp.ndarray
        Profile evaluated at `x`.
    """
    log_amp, center, fwhm = params
    amplitude = 10**log_amp
    gamma = fwhm / 2.0
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)

#################### Exotic ##############
@with_param_names(["logamp", "center", "fwhm_g", "fwhm_l"])
def voigt_pseudo(x, params):
    r"""
    Pseudo-Voigt profile (weighted sum of Gaussian and Lorentzian).

    .. math::
        f(x) = A \cdot \left[ \eta \cdot L(x) + (1 - \eta) \cdot G(x) \right]

    where:
    - :math:`A = 10^{\mathrm{logamp}}`
    - :math:`\sigma = \mathrm{fwhm_g} / 2.355`
    - :math:`\gamma = \mathrm{fwhm_l} / 2`
    - :math:`\eta` is an empirical function of :math:`\gamma` and :math:`\sigma`

    Parameters
    ----------
    x : jnp.ndarray
        Input wavelength array.
    params : array-like
        - `logamp`: Log base-10 amplitude.
        - `center`: Line center.
        - `fwhm_g`: Gaussian FWHM.
        - `fwhm_l`: Lorentzian FWHM.

    Returns
    -------
    jnp.ndarray
        Profile evaluated at `x`.
    """
    log_amp, center, fwhm_g, fwhm_l = params
    amplitude = 10**log_amp
    sigma = fwhm_g / 2.355
    gamma = fwhm_l / 2.0

    # Ratio for weighting
    r = gamma / (gamma + sigma * jnp.sqrt(2 * jnp.log(2)))
    eta = 1.36603 * r - 0.47719 * r**2 + 0.11116 * r**3

    # Gaussian and Lorentzian parts
    gauss = jnp.exp(-0.5 * ((x - center) / sigma) ** 2)
    lorentz = 1.0 / (1.0 + ((x - center) / gamma) ** 2)

    return amplitude * (eta * lorentz + (1.0 - eta) * gauss)


@with_param_names(["logamp", "center", "fwhm", "alpha"])
def skewed_gaussian(x, params):
    r"""
    Skewed Gaussian profile using the Azzalini formulation.

    .. math::
        f(x) = 2A \cdot \phi(t) \cdot \Phi(\alpha t)

    where:
    - :math:`t = \frac{x - \mu}{\sigma}`
    - :math:`\phi(t)` is the standard normal PDF
    - :math:`\Phi(t)` is the standard normal CDF
    - :math:`\sigma = \mathrm{fwhm} / 2.355`

    Parameters
    ----------
    x : jnp.ndarray
        Input wavelength array.
    params : array-like
        - `logamp`: Log base-10 amplitude.
        - `center`: Mean of the Gaussian.
        - `fwhm`: Full width at half maximum.
        - `alpha`: Skewness parameter.

    Returns
    -------
    jnp.ndarray
        Profile evaluated at `x`.
    """
    log_amp, center, fwhm, alpha = params  # alpha = skewness
    amplitude = 10**log_amp
    sigma = fwhm / 2.355
    t = (x - center) / sigma
    return 2 * amplitude * norm.pdf(t) * norm.cdf(alpha * t)



@with_param_names(["logamp", "center", "fwhm", "lambda"])
def emg_fwhm(x, params):
    r"""
    Exponentially Modified Gaussian (EMG) profile.

    .. math::
        f(x) = \frac{A \cdot \lambda}{2} \cdot \exp\left( \frac{\lambda}{2}(2\mu + \lambda\sigma^2 - 2x) \right) \cdot \mathrm{erfc}\left( \frac{\mu + \lambda\sigma^2 - x}{\sqrt{2}\sigma} \right)

    where:
    - :math:`A = 10^{\mathrm{logamp}}`
    - :math:`\sigma = \mathrm{fwhm} / 2.355`

    Parameters
    ----------
    x : jnp.ndarray
        Input wavelength array.
    params : array-like
        - `logamp`: Log base-10 amplitude.
        - `center`: Gaussian mean (μ).
        - `fwhm`: Gaussian full width at half maximum.
        - `lambda`: Exponential decay rate (1/τ).

    Returns
    -------
    jnp.ndarray
        Profile evaluated at `x`.
    """
    log_amp, mu, fwhm, lambda_ = params
    amplitude = 10**log_amp
    sigma = fwhm / 2.355
    arg1 = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
    arg2 = (mu + lambda_ * sigma**2 - x) / (jnp.sqrt(2) * sigma)
    return amplitude * 0.5 * lambda_ * jnp.exp(arg1) * erfc(arg2)


@with_param_names(["logamp", "center", "width"])
def top_hat(x, params):
    r"""
    Rectangular (top-hat) function.

    .. math::
        f(x) = A \quad \text{if } |x - \mu| \leq \frac{w}{2}; \quad 0 \text{ otherwise}

    where:
    - :math:`A = 10^{\mathrm{logamp}}`
    - :math:`\mu = \text{center}`
    - :math:`w = \text{width}`

    Parameters
    ----------
    x : jnp.ndarray
        Input wavelength array.
    params : array-like
        - `logamp`: Log base-10 amplitude.
        - `center`: Center of the box.
        - `width`: Width of the top-hat.

    Returns
    -------
    jnp.ndarray
        Profile evaluated at `x`.
    """
    log_amp, center, width = params
    amplitude = 10**log_amp
    half_width = width / 2.0
    return amplitude * ((x >= (center - half_width)) & (x <= (center + half_width))).astype(jnp.float32)

""" Experimental """

def eval_hermite(n: int, x: jnp.ndarray) -> jnp.ndarray:
    r"""
    Evaluate the physicist’s Hermite polynomial :math:`H_n(x)` recursively using JAX.

    The recurrence relation is:
    .. math::
        H_0(x) = 1 \\
        H_1(x) = 2x \\
        H_n(x) = 2x \cdot H_{n-1}(x) - 2(n-1) \cdot H_{n-2}(x)

    Parameters
    ----------
    n : int
        Order of the Hermite polynomial.
    x : jnp.ndarray
        Input array where the polynomial is evaluated.

    Returns
    -------
    jnp.ndarray
        Values of :math:`H_n(x)` with same shape as `x`.
    """
    def body(i, state):
        H0, H1 = state
        Hn = 2 * x * H1 - 2 * (i - 1) * H0
        return (H1, Hn)
    H0 = jnp.ones_like(x)
    H1 = 2 * x
    _, Hn = lax.fori_loop(2, n + 1, body, (H0, H1))
    return lax.select(n == 0, H0, lax.select(n == 1, H1, Hn))

#@jit


# 3. Gauss-Hermite LOSVD
#@jit
def gauss_hermite_losvd_jax(v, v0, sigma, h3=0.0, h4=0.0):
    r"""
    Line-of-sight velocity distribution (LOSVD) using Gauss-Hermite expansion.

    Based on van der Marel & Franx (1993) formulation:

    .. math::
        \mathcal{L}(v) = \frac{1}{\sqrt{2\pi} \sigma} \exp\left(-\frac{(v - v_0)^2}{2\sigma^2}\right) 
        \cdot \left[ 1 + h_3 H_3(x) + h_4 H_4(x) \right]

    where:
    - :math:`x = \frac{v - v_0}{\sigma}`
    - :math:`H_3(x)` and :math:`H_4(x)` are normalized Hermite polynomials.
    - Output is normalized to integrate to 1.

    Parameters
    ----------
    v : jnp.ndarray
        Velocity grid in km/s or appropriate units.
    v0 : float
        Mean velocity (center).
    sigma : float
        Standard deviation of the Gaussian core.
    h3 : float, optional
        Third Gauss-Hermite coefficient (skewness).
    h4 : float, optional
        Fourth Gauss-Hermite coefficient (kurtosis).

    Returns
    -------
    jnp.ndarray
        Normalized LOSVD array with same shape as `v`.
    """
    x = (v - v0) / sigma
    norm_gauss = jnp.exp(-0.5 * x**2) / (sigma * jnp.sqrt(2 * jnp.pi))
    H3 = eval_hermite(3, x) / jnp.sqrt(6.0)
    H4 = eval_hermite(4, x) / jnp.sqrt(24.0)
    losvd = norm_gauss * (1 + h3 * H3 + h4 * H4)
    losvd /= trapz_jax(losvd, v)
    return losvd
