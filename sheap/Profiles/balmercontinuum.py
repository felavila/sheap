"""
Balmer continuum
==================

This module implements Balmer continuum available in *sheap*, with a focus
on physically motivated prescriptions for AGN spectra.

Contents
--------
- **balmercontinuum** : Balmer edge continuum with edge normalization and optional
  velocity shift.
- **_planck_ratio_lambda** : Stable ratio of Planck functions B_λ(T)/B_λref(T).
- **_softplus** : Numerically stable softplus transform, used for reparameterization.

Notes
-----
- The Balmer continuum follows the Dietrich+2002 prescription but is normalized
  at the Balmer edge (λ_BE = 3646 Å) for stability.
- Temperature (`T_raw`), optical depth (`tau_raw`), and velocity (`v_raw`) are
  parameterized in raw space and transformed into physical values inside the
  function.
- The velocity parameter allows a global Doppler shift of the edge up to
  ±3000 km/s.
- All functions are JAX-compatible and differentiable.

Example
-------
.. code-block:: python

   import jax.numpy as jnp
   from sheap.Profiles.continuum_profiles import balmercontinuum

   lam = jnp.linspace(3000, 4000, 500)  # Å
   pars = [1.0, 0.5, -0.1, 0.0]         # [amplitude, T_raw, tau_raw, v_raw]
   flux = balmercontinuum(lam, pars)
"""

__author__ = 'felavila'


__all__ = ["_planck_ratio_lambda",
    "_softplus",
    "balmercontinuum",
]


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jit, vmap

from sheap.Profiles.utils import with_param_names

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

# @with_param_names(["amplitude", "T", "tau0"])
# def balmercontinuum(x, pars):
#     """
#     Balmer continuum (Dietrich+2002–style) normalized at the Balmer edge.

#     The unnormalized model is:
#         f(λ) = A · B_λ(T) · [1 - exp(-τ(λ))],   for λ ≤ λ_BE
#     with
#         τ(λ) = τ0 · (λ / λ_BE)^3,     λ_BE = 3646 Å.

#     Here we normalize the *shape* by its value at λ_BE:
#         f_norm(λ) = [B_λ(T) · (1 - exp(-τ(λ)))] / [B_λ(T) · (1 - exp(-τ0))]_{λ=λ_BE}
#     so that the returned spectrum is:
#         F(λ) = amplitude · f_norm(λ),  for λ ≤ λ_BE, and 0 otherwise.

#     Parameters
#     ----------
#     x : array-like
#         Wavelengths in Å (vacuum).
#     pars : array-like, shape (3,)
#         pars[0] -> amplitude (dimensionless weight; scales with SHEAP's global scale)
#         pars[1] -> T (K), electron temperature controlling the shape
#         pars[2] -> τ0, optical-depth scale at λ_BE

#     Returns
#     -------
#     jnp.ndarray
#         Dimensionless template scaled by `amplitude`; zero for λ > λ_BE.

#     Notes
#     -----
#     - Because of the edge normalization, physical units cancel out; only the
#       *relative* shape matters. The overall flux scaling should come from sheap’s
#       rescaling pipeline (your global `scale`), via this component’s `amplitude`.
#     - This function intentionally avoids extra constants (e.g., 1e18) or Astropy
#       scaling factors. If you need a different normalization (e.g., unit integral
#       over 3000–3646 Å), we can provide that variant.

#     Math
#     ----
#     .. math::

#         F(\\lambda) = A\\;\\frac{B_{\\lambda}(T)\\,[1 - e^{-\\tau_0 (\\lambda/\\lambda_{\\rm BE})^3}]}
#                                {B_{\\lambda_{\\rm BE}}(T)\\,[1 - e^{-\\tau_0}]}
#         \\quad (\\lambda \\le \\lambda_{\\rm BE}),\\; 0\\;\\text{otherwise}.

#     """


#     # Physical constants (SI) – they cancel in the edge normalization, but we keep them for clarity.
#     h = 6.62607015e-34      # J s
#     c = 2.99792458e8        # m s^-1
#     k_B = 1.380649e-23      # J K^-1

#     lambda_BE = 3646.0  # Å (Balmer edge)
#     lam_m = x * 1e-10   # Å -> m

#     A   = pars[0]
#     T   = pars[1]
#     tau0 = pars[2]

#     # Planck function B_lambda(T) in SI (W m^-3 sr^-1). Units cancel after normalization.
#     exponent = (h * c) / (lam_m * k_B * jnp.clip(T, 1.0, jnp.inf))
#     B_lambda = (2.0 * h * c**2) / (jnp.clip(lam_m, 1e-30, jnp.inf)**5 * (jnp.exp(exponent) - 1.0))

#     # Optical depth law
#     tau = tau0 * (x / lambda_BE) ** 3

#     # Unnormalized shape (only defined blueward of the edge)
#     raw = B_lambda * (1.0 - jnp.exp(-tau))

#     # Edge value for normalization (evaluate analytically at λ_BE)
#     lam_BE_m = lambda_BE * 1e-10
#     exponent_BE = (h * c) / (lam_BE_m * k_B * jnp.clip(T, 1.0, jnp.inf))
#     B_lambda_BE = (2.0 * h * c**2) / (lam_BE_m**5 * (jnp.exp(exponent_BE) - 1.0))
#     norm_edge = B_lambda_BE * (1.0 - jnp.exp(-jnp.clip(tau0, 0.0, jnp.inf)))

#     # Avoid division by zero if tau0 ~ 0 or extreme T
#     norm_edge = jnp.clip(norm_edge, 1e-300, jnp.inf)

#     f_norm = raw / norm_edge

#     # Zero redward of the edge
#     f_norm = jnp.where(x <= lambda_BE, f_norm, 0.0)

#     return A * f_norm

def _softplus(x):
    # numerically stable softplus
    return jnp.log1p(jnp.exp(-jnp.abs(x))) + jnp.maximum(x, 0.)

def _planck_ratio_lambda(lam_m, lam_ref_m, T):
    """
    Return B_lambda(T)/B_lambda_ref(T) without huge/small intermediates.
    Uses expm1 for stability and recovers RJ limit correctly.
    """
    # Wien's displacement constant in SI units: hc/k_B (meters·Kelvin)
    wien = 1.438776877e-2  # m·K
    T = jnp.clip(T, 1.0, jnp.inf)

    z   = wien / (jnp.clip(lam_m,    1e-30, jnp.inf) * T)
    z_r = wien / (jnp.clip(lam_ref_m,1e-30, jnp.inf) * T)

    # Bλ ∝ λ^-5 / (exp(z)-1)  ⇒  ratio = (λ_ref/λ)^5 * (expm1(z_ref)/expm1(z))
    return (lam_ref_m / lam_m)**5 * (
        jnp.expm1(z_r) / jnp.clip(jnp.expm1(z), 1e-300, jnp.inf)
    )

@with_param_names(["amplitude", "T_raw", "tau_raw", "v_raw"])  # NEW: v_raw
def balmercontinuum(x, pars):
    """
    Balmer continuum with edge normalization and a global velocity shift.

    Raw params
    ----------
    amplitude : linear (kept linear so Sheap post-scale can adjust it)
    T_raw     : T = T_floor + T_scale * softplus(T_raw)
    tau_raw   : tau0 = softplus(tau_raw)
    v_raw     : global shift of the Balmer edge via v = vmax * tanh(v_raw) [km/s]
    """
    A, T_raw, tau_raw, v_raw = pars

    # raw -> physical
    T_floor, T_scale = 4000.0, 1000.0
    T    = jnp.clip(T_floor + T_scale * _softplus(T_raw), T_floor, 5.0e4)
    tau0 = _softplus(tau_raw)

    vmax = 3000.0                       # cap shift to ±3000 km/s (tweak if you like)
    v    = vmax * jnp.tanh(v_raw)       # bounded, smooth
    c_kms = 299792.458
    beta  = 1.0 + v / c_kms             # Doppler factor (non-relativistic ok here)

    # Shift the model along wavelength: evaluate the rest-frame shape at x_eff = x / beta
    lambda_BE = 3646.0  # Å
    x = jnp.asarray(x)
    x_eff = x / beta

    lam_m   = jnp.clip(x_eff, 1e-6, jnp.inf) * 1e-10  # Å→m
    lamBE_m = lambda_BE * 1e-10

    # Planck ratio and τ ratio (stable τ0→0 limit)
    planck_ratio = _planck_ratio_lambda(lam_m, lamBE_m, T)

    tau = tau0 * (x_eff / lambda_BE) ** 3
    one_minus_e_m_tau  = -jnp.expm1(-jnp.clip(tau,  0.0, jnp.inf))
    one_minus_e_m_tau0 = -jnp.expm1(-jnp.clip(tau0, 0.0, jnp.inf))
    tau_ratio = jnp.where(
        one_minus_e_m_tau0 > 0.0,
        one_minus_e_m_tau / one_minus_e_m_tau0,
        (x_eff / lambda_BE) ** 3
    )

    f_norm = planck_ratio * tau_ratio
    f_norm = jnp.where(x_eff <= lambda_BE, f_norm, 0.0)  # edge at λ_BE * beta in the plotted frame

    f_norm = jnp.nan_to_num(f_norm, 0.0, 0.0, 0.0)
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
