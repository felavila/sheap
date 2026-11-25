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
- The velocity parameter allows a global Doppler shift of the edge up to ±3000 km/s.
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


__all__ = ["_planck_ratio_lambda","_softplus","balmercontinuum",]


from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np

from sheap.Profiles.Utils import with_param_names,add_linear_amplitude_design_builder

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

@add_linear_amplitude_design_builder(amplitude_index=0)
@with_param_names(["amplitude", "T_raw", "tau_raw", "v_raw"], linear_param_names=["amplitude"], profile_name="balmer_cont",)
def balmercontinuum(x, pars):
    """
    Balmer continuum with edge normalization and a global velocity shift.
    Now includes a soft exponential rolloff instead of hard cutoff.

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

    vmax = 3000.0
    v    = vmax * jnp.tanh(v_raw)
    c_kms = 299792.458
    beta  = 1.0 + v / c_kms

    lambda_BE = 3646.0  # Å
    x = jnp.asarray(x)
    x_eff = x / beta

    lam_m   = jnp.clip(x_eff, 1e-6, jnp.inf) * 1e-10
    lamBE_m = lambda_BE * 1e-10

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
    

    rolloff_width = 120.0  # Angstroms - adjust this to control softness
    soft_edge = jnp.exp(-jnp.clip((x_eff - lambda_BE) / rolloff_width, 0.0, 20.0))
    
    f_norm = f_norm * soft_edge 

    f_norm = jnp.nan_to_num(f_norm, 0.0, 0.0, 0.0)
    return A * f_norm

def _softplus_numpy(x):
    """Stable NumPy softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _planck_ratio_lambda_numpy(lam_m, lam_ref_m, T):
    """
    Same Planck ratio as the JAX implementation,
    evaluated in float64.
    """
    wien = 1.438776877e-2

    T = np.clip(T, 1.0, np.inf)

    z = wien / (np.clip(lam_m, 1e-30, np.inf) * T)
    z_ref = wien / (np.clip(lam_ref_m, 1e-30, np.inf) * T)

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = (
            (lam_ref_m / lam_m)**5
            * np.expm1(z_ref)
            / np.clip(np.expm1(z), 1e-300, np.inf)
        )

    return ratio


def _numpy_balmer_shape(x, params):
    """
    Unit-amplitude Balmer continuum.

    Parameter order:
        [amplitude, T_raw, tau_raw, v_raw]

    Amplitude is excluded from the shape calculation.
    """

    x = np.asarray(x, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)

    _, T_raw, tau_raw, v_raw = params

    # --------------------------------------------------
    # Physical transformations
    # --------------------------------------------------
    T_floor = 4000.0
    T_scale = 1000.0

    T = np.clip(
        T_floor + T_scale * _softplus_numpy(T_raw),
        T_floor,
        5.0e4,
    )

    tau0 = _softplus_numpy(tau_raw)

    vmax = 3000.0
    v = vmax * np.tanh(v_raw)

    c_kms = 299792.458
    beta = 1.0 + v / c_kms

    # --------------------------------------------------
    # Shifted Balmer edge
    # --------------------------------------------------
    lambda_BE = 3646.0

    x_eff = x / beta

    lam_m = np.clip(x_eff, 1e-6, np.inf) * 1e-10
    lamBE_m = lambda_BE * 1e-10

    # --------------------------------------------------
    # Planck continuum
    # --------------------------------------------------
    planck_ratio = _planck_ratio_lambda_numpy(lam_m, lamBE_m,T,)

    # --------------------------------------------------
    # Optical-depth contribution
    # --------------------------------------------------
    tau = tau0 * (x_eff / lambda_BE)**3

    one_minus_e_m_tau = -np.expm1(
        -np.clip(tau, 0.0, np.inf)
    )

    one_minus_e_m_tau0 = -np.expm1(
        -np.clip(tau0, 0.0, np.inf)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        tau_ratio = np.where(
            one_minus_e_m_tau0 > 0.0,
            one_minus_e_m_tau / one_minus_e_m_tau0,
            (x_eff / lambda_BE)**3,
        )

    f_norm = planck_ratio * tau_ratio

    # --------------------------------------------------
    # Soft Balmer-edge rolloff
    # --------------------------------------------------
    rolloff_width = 120.0

    soft_edge = np.exp(
        -np.clip(
            (x_eff - lambda_BE) / rolloff_width,
            0.0,
            20.0,
        )
    )

    f_norm *= soft_edge

    return np.nan_to_num(
        f_norm,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def numpy_balmer_model(x, params):
    """
    Complete Balmer continuum in float64.
    """
    params = np.asarray(params, dtype=np.float64)

    amplitude = params[0]

    return amplitude * _numpy_balmer_shape(x, params)


def numpy_balmer_design_builder(x, params):
    """
    Linear representation:

        F_balmer = offset + A @ weights

    Here:
        offset = 0
        A[:, 0] = unit-amplitude Balmer continuum
        weights = [amplitude]
    """

    x = np.asarray(x, dtype=np.float64)

    shape = _numpy_balmer_shape(x, params)

    offset = np.zeros_like(x, dtype=np.float64)

    columns = shape[:, None]

    return offset, columns


# --------------------------------------------------
# Attach NumPy backend to existing JAX profile
# --------------------------------------------------

balmercontinuum.numpy_model = numpy_balmer_model

balmercontinuum.numpy_design_builder = (numpy_balmer_design_builder)
