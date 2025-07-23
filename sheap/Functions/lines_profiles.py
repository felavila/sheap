import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import jit, vmap,lax 
from jax.scipy.special import erfc
from jax.scipy.stats import norm #maybe dosent exist xd

from sheap.Functions.utils import with_param_names

@with_param_names(["logamp", "center", "fwhm"])
def gaussian_fwhm(x, params):
    log_amp, center, fwhm = params
    amplitude = 10**log_amp
    sigma = fwhm / 2.355 #fwhm -> logfwhm
    return amplitude * jnp.exp(-0.5 * ((x - center) / sigma) ** 2)

@with_param_names(["logamp", "center", "fwhm"])
def lorentzian_fwhm(x, params):
    log_amp, center, fwhm = params
    amplitude = 10**log_amp
    gamma = fwhm / 2.0
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)

#################### Exotic ##############
@with_param_names(["logamp", "center", "fwhm_g", "fwhm_l"])
def voigt_pseudo(x, params):
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
    log_amp, center, fwhm, alpha = params  # alpha = skewness
    amplitude = 10**log_amp
    sigma = fwhm / 2.355
    t = (x - center) / sigma
    return 2 * amplitude * norm.pdf(t) * norm.cdf(alpha * t)



@with_param_names(["logamp", "center", "fwhm", "lambda"])
def emg_fwhm(x, params):
    """
    Exponentially Modified Gaussian profile.
    
    Parameters:
        amplitude: peak scaling
        center: Gaussian mean (mu)
        fwhm: Gaussian FWHM (converted to sigma)
        lambda_: exponential decay rate (1 / tau)
    """
    log_amp, mu, fwhm, lambda_ = params
    amplitude = 10**log_amp
    sigma = fwhm / 2.355
    arg1 = 0.5 * lambda_ * (2 * mu + lambda_ * sigma**2 - 2 * x)
    arg2 = (mu + lambda_ * sigma**2 - x) / (jnp.sqrt(2) * sigma)
    return amplitude * 0.5 * lambda_ * jnp.exp(arg1) * erfc(arg2)


@with_param_names(["logamp", "center", "width"])
def top_hat(x, params):
    """
    Top-hat function: constant value over a fixed width.
    
    Parameters:
        amplitude: height of the top
        center: midpoint of the top-hat
        width: full width of the top-hat
    """
    log_amp, center, width = params
    amplitude = 10**log_amp
    half_width = width / 2.0
    return amplitude * ((x >= (center - half_width)) & (x <= (center + half_width))).astype(jnp.float32)

# ####### actual combination##############################
# def sum_gaussian_amplitude_free(centers, amplitude_rules,n_params):
#     """
#     centers: list of Gaussian centers
#     amplitude_rules: list of (idx, coefficient, free_idx)
#         idx → which Gaussian
#         coefficient → scale relative to free param
#         free_idx → which parameter index it depends on
#     """
#     centers = jnp.array(centers)
#     #_param_count = max(r[2] for r in amplitude_rules) + 2  # free amps + delta + fwhm

#     @param_count(n_params+2)
#     def G(x, params):
#         free = params[:-2]
#         delta = params[-2]
#         fwhm = params[-1]
#         sigma = fwhm / 2.355

#         result = 0.0
#         for idx, coef, free_idx in amplitude_rules:
#             amp = coef * free[free_idx]
#             dx = x - (centers[idx] + delta)
#             result += amp * jnp.exp(-0.5 * (dx / sigma) ** 2)
#         return result

#     return G


# def Gsum_model(centers, amplitudes):
#     """
#     Returns a Gaussian sum model function with shared FWHM and shift.

#     Args:
#         centers (array): Array of Gaussian centers.
#         amplitudes (array): Array of Gaussian amplitudes (same shape as centers).

#     Returns:
#         callable: Function G(x, params) with params = [amplitude, delta, fwhm].
#     """
#     centers = jnp.array(centers)
#     amplitudes = jnp.array(amplitudes)
#     @param_count(3)
#     def G(x, params):
#         amplitude = params[0]
#         delta = params[1]
#         fwhm = params[2]
#         sigma = fwhm / 2.355

#         shifted_centers = centers + delta
#         dx = jnp.expand_dims(x, 0) - jnp.expand_dims(shifted_centers, 1)
#         gaussians = amplitude * amplitudes[:, None] * jnp.exp(-0.5 * (dx / sigma) ** 2)
#         return jnp.sum(gaussians, axis=0)

#     return G

#################### Comming projects ##############


#@jit
#util more than proffer function
def eval_hermite(n: int, x: jnp.ndarray) -> jnp.ndarray:
    def body(i, state):
        H0, H1 = state
        Hn = 2 * x * H1 - 2 * (i - 1) * H0
        return (H1, Hn)
    H0 = jnp.ones_like(x)
    H1 = 2 * x
    _, Hn = lax.fori_loop(2, n + 1, body, (H0, H1))
    return lax.select(n == 0, H0, lax.select(n == 1, H1, Hn))

#@jit
def trapz_jax(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    dx = x[1:] - x[:-1]
    return jnp.sum((y[1:] + y[:-1]) * dx / 2)

# 3. Gauss-Hermite LOSVD
#@jit
def gauss_hermite_losvd_jax(v, v0, sigma, h3=0.0, h4=0.0):
    x = (v - v0) / sigma
    norm_gauss = jnp.exp(-0.5 * x**2) / (sigma * jnp.sqrt(2 * jnp.pi))
    H3 = eval_hermite(3, x) / jnp.sqrt(6.0)
    H4 = eval_hermite(4, x) / jnp.sqrt(24.0)
    losvd = norm_gauss * (1 + h3 * H3 + h4 * H4)
    losvd /= trapz_jax(losvd, v)
    return losvd
