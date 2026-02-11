r"""
Combine FWHM utilities for line profiles
================================

Helpers to compute the full width at half maximum (FWHM) for different
profile families, with optional uncertainty propagation and batched
(vmap) evaluation.

Notes
-----
- For some analytic profiles, the FWHM is read directly from named
  shape parameters. For example:

  - Gaussian / Lorentzian:
    .. math::
       \mathrm{FWHM} = \text{fwhm}

  - Top-hat:
    .. math::
       \mathrm{FWHM} = \text{width}

  - Pseudo-Voigt:
    .. math::
       \mathrm{FWHM} \approx 0.5346\,\text{FWHM}_L
       + \sqrt{0.2166\,\text{FWHM}_L^2 + \text{FWHM}_G^2}

- For other profiles (e.g., skewed shapes), a numeric half‑maximum
  search is performed around the peak.
#TODO change name from fwhm_conv to convination -> utils
"""

__author__ = 'felavila'

__all__ = [
    "compute_fwhm_split",
    "compute_fwhm_split_with_error",
    "make_batch_fwhm_split",
    "make_batch_fwhm_split_with_error",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import warnings
from functools import partial
from jax import vmap,jit
import jax.numpy as jnp
from jax import jacfwd
import numpy as np 
from uncertainties import unumpy


from sheap.Profiles.Profiles import PROFILE_LINE_FUNC_MAP
from sheap.Utils.Constants import DEFAULT_C_KMS



def compute_fwhm_split(profile: str,amp:   jnp.ndarray,center:jnp.ndarray,extras:jnp.ndarray) -> jnp.ndarray:
    r"""
    Compute the FWHM of a single line component for a given profile.

    The function uses analytic formulas when available (Gaussian,
    Lorentzian, Top-hat, Pseudo‑Voigt). Otherwise, it estimates the
    half‑maximum width numerically around the peak.

    Parameters
    ----------
    profile : str
        Profile key (must exist in ``PROFILE_LINE_FUNC_MAP``),
        e.g. ``"gaussian"``, ``"lorentzian"``, ``"top_hat"``,
        ``"voigt_pseudo"``, etc.
    amp : jnp.ndarray
        Peak amplitude (scalar).
    center : jnp.ndarray
        Line center (scalar).
    extras : jnp.ndarray
        Remaining shape parameters in the order required by the profile,
        i.e. they correspond to ``param_names[2:]``.

    Returns
    -------
    jnp.ndarray
        The full width at half maximum for the component (scalar).

    Notes
    -----
    - Pseudo‑Voigt approximation:
      .. math::
         \mathrm{FWHM} \approx 0.5346\,\text{FWHM}_L
         + \sqrt{0.2166\,\text{FWHM}_L^2 + \text{FWHM}_G^2}
    - Numeric fallback scans a symmetric grid around the center and finds
      the left/right half‑max crossings.
    """
    func = PROFILE_LINE_FUNC_MAP[profile]

    # build the named‐param dict on‐the‐fly:
    # we know extras corresponds to param_names[2:]
    names = func.param_names
    p = { names[0]: amp,
          names[1]: center }
    for i,name in enumerate(names[2:]):
        p[name] = extras[i]

    # analytic cases:
    if profile == "gaussian" or profile == "lorentzian":
        return p["fwhm"]
    if profile == "top_hat":
        return p["width"]
    if profile == "voigt_pseudo":
        fg = p["fwhm_g"]; fl = p["fwhm_l"]
        return 0.5346*fl + jnp.sqrt(0.2166*fl*fl + fg*fg)

    # numeric‐fallback (e.g. skewed, EMG)
    half = amp/2.0
    def shape_fn(x):
        return func(x, jnp.concatenate([jnp.array([amp,center]), extras]))
    guess = p.get("fwhm", p.get("width",
                jnp.maximum(p.get("fwhm_g",0), p.get("fwhm_l",0))))
    lo,hi = center-5*guess, center+5*guess
    xs = jnp.linspace(lo, hi, 2001)
    ys = shape_fn(xs)

    maskL = (xs<center)&(ys<=half)
    maskR = (xs> center)&(ys<=half)
    xL = jnp.max(jnp.where(maskL, xs, lo))
    xR = jnp.min(jnp.where(maskR, xs, hi))
    return xR - xL


def compute_fwhm_split_with_error(
    profile: str,
    amp: jnp.ndarray,
    center: jnp.ndarray,
    extras: jnp.ndarray,
    amp_err: jnp.ndarray,
    center_err: jnp.ndarray,
    extras_err: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Compute FWHM and its 1σ uncertainty for a single component.

    Uncertainty is propagated via the Jacobian of FWHM with respect to
    all input parameters (``amp``, ``center``, ``extras``):

    .. math::
       \sigma_{\mathrm{FWHM}}^2
       = \sum_i \left( \frac{\partial\,\mathrm{FWHM}}{\partial p_i} \,
                       \sigma_{p_i} \right)^2

    Parameters
    ----------
    profile : str
        Profile key for :func:`compute_fwhm_split`.
    amp, center, extras : jnp.ndarray
        Scalar amplitude, scalar center, and extras vector for the profile.
    amp_err, center_err, extras_err : jnp.ndarray
        Matching 1σ uncertainties for the corresponding parameters.

    Returns
    -------
    fwhm_val : jnp.ndarray
        Estimated FWHM (scalar).
    fwhm_uncertainty : jnp.ndarray
        Propagated 1σ uncertainty (scalar).

    Notes
    -----
    Uses :func:`jax.jacfwd` to compute the gradient of the FWHM function
    with respect to the concatenated parameter vector.
    """

    fwhm_fn = lambda amp, center, extras: compute_fwhm_split(profile, amp, center, extras)
    fwhm_val = fwhm_fn(amp, center, extras)

    # Build parameter vector
    all_params = jnp.concatenate([amp[None], center[None], extras])
    all_errors = jnp.concatenate([amp_err[None], center_err[None], extras_err])

    # Compute gradient
    grad_fwhm = jacfwd(lambda p: fwhm_fn(p[0], p[1], p[2:]))(all_params)

    # Propagate uncertainty
    fwhm_uncertainty = jnp.sqrt(jnp.sum((grad_fwhm * all_errors) ** 2))
    return fwhm_val, fwhm_uncertainty



def make_batch_fwhm_split_with_error(profile: str):
    """
    Vectorized (batched) FWHM + uncertainty evaluator for a profile.

    Returns a function that accepts batched inputs for values and their
    uncertainties and computes both FWHM and its propagated 1σ error
    using two levels of ``vmap`` (over lines, then over batch).

    Parameters
    ----------
    profile : str
        Profile key for :func:`compute_fwhm_split_with_error`.

    Returns
    -------
    Callable
        A function
        ``batcher(amp, center, extras, amp_err, center_err, extras_err)``
        that returns ``(fwhm_val, fwhm_uncertainty)`` with shapes matching
        the leading batch dimensions of the inputs.
    """
    single = partial(compute_fwhm_split_with_error, profile)
    over_lines = vmap(single, in_axes=(0, 0, 0, 0, 0, 0))
    batcher = vmap(over_lines, in_axes=(0, 0, 0, 0, 0, 0))
    return batcher

def make_batch_fwhm_split(profile: str):
    """
    Create a batched FWHM evaluator for a given profile.

    This returns a function that computes the full width at half maximum (FWHM)
    for multiple objects and multiple line components in parallel, using JAX’s
    :func:`vmap` for vectorization.

    Parameters
    ----------
    profile : str
        Profile name (must exist in ``PROFILE_LINE_FUNC_MAP``),
        e.g. ``"gaussian"``, ``"lorentzian"``, ``"voigt_pseudo"``, etc.

    Returns
    -------
    callable
        A function with signature::

            fwhm_batch(amp, center, extras) -> jnp.ndarray

        where
        - ``amp`` has shape (n_objects, n_lines),
        - ``center`` has shape (n_objects, n_lines),
        - ``extras`` has shape (n_objects, n_lines, n_extras),

        and the result is a ``(n_objects, n_lines)`` array of FWHM values.

    Notes
    -----
    - Analytic shortcuts are used for common profiles
      (Gaussian, Lorentzian, Top-hat, pseudo-Voigt).
    - For other profiles, a numeric search is performed around the line center
      to locate the half-maximum crossing.
    """
    single = partial(compute_fwhm_split, profile)
    over_lines = vmap(single, in_axes=(0, 0, 0))
    batcher    = vmap(over_lines, in_axes=(0, 0, 0))

    return batcher


@jit
def combine_broad_moments(
    params_broad: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Combine multiple broad components into a single effective Gaussian
    using amplitude-weighted moments, without any virial filtering.

    Parameters
    ----------
    params_broad : ndarray, shape (N, 3*n_broad)
        Broad component parameters grouped as [amp_i, mu_i, fwhm_i, ...].

    Returns
    -------
    fwhm_eff : ndarray, shape (N,)
        Effective FWHM (same units as input fwhm_i).
    amp_eff : ndarray, shape (N,)
        Total effective amplitude (sum of amplitudes).
    mu_eff : ndarray, shape (N,)
        Effective line center (amplitude-weighted mean).
    """
    N = params_broad.shape[0]
    n_broad = params_broad.shape[1] // 3

    broad = params_broad.reshape(N, n_broad, 3)
    amp_b, mu_b, fwhm_b = broad[..., 0], broad[..., 1], broad[..., 2]

    # Total amplitude and amplitude-weighted center
    total_amp = jnp.sum(amp_b, axis=1)              # (N,)
    mu_eff    = jnp.sum(amp_b * mu_b, axis=1) / total_amp

    # Effective variance from mixture of Gaussians
    invf = 1.0 / 2.35482
    var_i   = (fwhm_b * invf) ** 2                  # σ_i^2
    dif2    = (mu_b - mu_eff[:, None]) ** 2
    var_eff = jnp.sum(amp_b * (var_i + dif2), axis=1) / total_amp

    fwhm_eff = jnp.sqrt(var_eff) * 2.35482          # back to FWHM

    return fwhm_eff, total_amp, mu_eff


@jit
def combine_fast(
    params_broad: jnp.ndarray,
    params_narrow: jnp.ndarray,
    limit_velocity: float = 150.0,
    C_KMS: float = DEFAULT_C_KMS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Efficiently combine multiple broad components with one narrow
    component into an effective line measurement.

    Parameters
    ----------
    params_broad : ndarray, shape (N, 3*n_broad)
        Broad component parameters grouped as [amp_i, mu_i, fwhm_i, ...].
    params_narrow : ndarray, shape (N, 3)
        Narrow component parameters [amp_n, mu_n, fwhm_n].
        Only ``mu_n`` is used in velocity filtering.
    limit_velocity : float, optional
        Velocity threshold in km/s for virial filtering. Default 150.
    C_KMS : float, optional
        Speed of light in km/s. Default 299792.

    Returns
    -------
    fwhm_final : ndarray, shape (N,)
        Effective full width at half maximum (same units as input).
    amp_final : ndarray, shape (N,)
        Effective amplitude.
    mu_final : ndarray, shape (N,)
        Effective line center.

    Notes
    -----
    - Virial filtering selects the nearest broad component relative
      to the narrow component if offsets exceed ``limit_velocity``.
    - Otherwise, amplitude-weighted averages of broad components are used.
    """
    N = params_broad.shape[0]
    n_broad = params_broad.shape[1] // 3
    broad = params_broad.reshape(N, n_broad, 3)
    amp_b, mu_b, fwhm_b = broad[..., 0], broad[..., 1], broad[..., 2]

    
    total_amp = jnp.sum(amp_b, axis=1)                      # (N,)
    mu_eff    = jnp.sum(amp_b * mu_b, axis=1) / total_amp

    invf = 1.0 / 2.35482
    var_i   = (fwhm_b * invf) ** 2
    dif2    = (mu_b - mu_eff[:, None]) ** 2
    var_eff = jnp.sum(amp_b * (var_i + dif2), axis=1) / total_amp
    fwhm_eff= jnp.sqrt(var_eff) * 2.35482                   # (N,)

    mu_nar   = params_narrow[:, 1]
    rel_vel  = jnp.abs((mu_b - mu_nar[:, None]) / mu_nar[:, None]) * C_KMS
    idx_near = jnp.argmin(rel_vel, axis=1)

    sel = lambda arr: arr[jnp.arange(N), idx_near]
    fwhm_nb  = sel(fwhm_b)
    amp_nb   = sel(amp_b)
    mu_nb    = sel(mu_b)

    amp_ratio = jnp.min(amp_b, axis=1) / jnp.max(amp_b, axis=1)
    mask_amp  = amp_ratio > 0.1

    fwhm_choice = jnp.where(mask_amp, fwhm_eff, fwhm_nb)
    amp_choice  = jnp.where(mask_amp, total_amp, amp_nb)
    mu_choice   = jnp.where(mask_amp, mu_eff, mu_nb)

    mask_vir = jnp.min(rel_vel, axis=1) >= limit_velocity
    fwhm_final = jnp.where(mask_vir, fwhm_nb,    fwhm_choice)
    amp_final  = jnp.where(mask_vir, amp_nb,     amp_choice)
    mu_final   = jnp.where(mask_vir, mu_nb,      mu_choice)

    return fwhm_final, amp_final, mu_final

def combine_fast_with_jacobian(
    amp_b,
    mu_b,
    fwhm_b,
    amp_n,
    mu_n,
    fwhm_n,
    limit_velocity: float = 150.0,
    C_KMS: float = DEFAULT_C_KMS,
    use_jacobian: bool = True,
    rough_scale: float = 1.0
):
    """
    Combine broad + narrow components with uncertainty propagation.

    Parameters
    ----------
    amp_b, mu_b, fwhm_b : Uncertainty
        Amplitude, center, and FWHM arrays for broad components.
    amp_n, mu_n, fwhm_n : Uncertainty
        Amplitude, center, and FWHM for the narrow component.
    limit_velocity : float, optional
        Velocity threshold (km/s) for virial filtering. Default 150.
    C_KMS : float, optional
        Speed of light (km/s). Default 299792.458.
    use_jacobian : bool, optional
        If True (default), propagate uncertainties using
        Jacobians via :func:`jax.jacfwd`.
        If False, apply a rough scaling factor.
    rough_scale : float, optional
        Multiplier for fallback uncertainty estimates.

    Returns
    -------
    fwhm : Uncertainty
        Effective FWHM with propagated uncertainty.
    amp : Uncertainty
        Effective amplitude with propagated uncertainty.
    mu : Uncertainty
        Effective center with propagated uncertainty.

    Notes
    -----
    - Jacobian-based propagation may fail for degenerate inputs;
      in that case, a fallback approximation is used.
    - This routine provides *approximate* error propagation; for
      full posterior distributions, use sampling-based methods.
    """
    #unumpy.std_devs,unumpy.nominal_values
    N = unumpy.nominal_values(amp_b).shape[0]
    n_broad = unumpy.nominal_values(amp_b).shape[1]
    results = []

    for i in range(N):
        # Flatten input vector
        x0 = jnp.concatenate([
            unumpy.nominal_values(amp_b)[i], unumpy.nominal_values(mu_b)[i], unumpy.nominal_values(fwhm_b)[i],
            unumpy.nominal_values(amp_n)[i], unumpy.nominal_values(mu_n)[i], unumpy.nominal_values(fwhm_n)[i]
        ])
        errors = jnp.concatenate([
            unumpy.std_devs(amp_b)[i], unumpy.std_devs(mu_b)[i], unumpy.std_devs(fwhm_b)[i],
            unumpy.std_devs(amp_n)[i], unumpy.std_devs(mu_n)[i], unumpy.std_devs(fwhm_n)[i]
        ])

        def wrapped_func(x):
            a_b = x[:n_broad]
            m_b = x[n_broad:2*n_broad]
            f_b = x[2*n_broad:3*n_broad]
            a_n = x[3*n_broad:3*n_broad+1]
            m_n = x[3*n_broad+1:3*n_broad+2]
            f_n = x[3*n_broad+2:3*n_broad+3]
            pb = jnp.stack([a_b, m_b, f_b], axis=-1).reshape(1, -1)
            pn = jnp.stack([a_n, m_n, f_n], axis=-1).reshape(1, -1)
            return jnp.array(combine_fast(pb, pn, limit_velocity, C_KMS)).squeeze()

        f0 = wrapped_func(x0)

        if use_jacobian:
            try:
                J = jacfwd(wrapped_func)(x0)  # shape (3, len(x0))
                propagated_var = jnp.sum((J * errors)**2, axis=1)
                propagated_err = jnp.sqrt(propagated_var)
            except Exception as e:
                print(f"[Warning] Jacobian failed for index {i}: {e}. Falling back to rough.")
                propagated_err = jnp.abs(f0) * 0.1 * rough_scale
        else:
            propagated_err = jnp.abs(f0) * 0.1 * rough_scale

        # Ensure each result is [(fwhm, err), (amp, err), (mu, err)]
        results.append(list(zip(f0, propagated_err)))

    # Transpose list of tuples into result groups
    results = list(zip(*results))  # [(fwhm, err), (amp, err), (mu, err)]
    fwhm_vals, fwhm_errs = zip(*results[0])
    amp_vals, amp_errs   = zip(*results[1])
    mu_vals, mu_errs     = zip(*results[2])

    return (
        unumpy.uarray(np.array(fwhm_vals), np.array(fwhm_errs)),
        unumpy.uarray(np.array(amp_vals),  np.array(amp_errs)),
        unumpy.uarray(np.array(mu_vals),   np.array(mu_errs))
    )