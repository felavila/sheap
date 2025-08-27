"""
Line Component Combination Utilities
====================================

This module provides functions to merge multiple spectral line
components (e.g., broad and narrow Gaussians) into an effective
single representation, with optional uncertainty propagation.

Main Features
-------------
- :func:`combine_components`  
  High-level wrapper that inspects fitted line parameters and,
  when possible, combines broad and narrow components into a
  single effective profile (flux, FWHM, luminosity, etc.).

- :func:`combine_fast`  
  Efficient JAX-compatible routine that merges any number of
  broad components with one narrow component, returning effective
  width, amplitude, and center. Used in batched pipelines.

- :func:`combine_fast_with_jacobian`  
  Variant that propagates uncertainties through the combination
  using JAX’s automatic differentiation (Jacobian). Falls back
  to rough scaling if Jacobian evaluation fails.

Use Cases
---------
- Constructing combined Hα or Hβ line properties when both narrow
  and broad components are fitted.
- Estimating effective FWHM and flux for virial mass estimators
  that require a single line measure.
- Propagating uncertainties from individual components into the
  combined measurement.

Notes
-----
- :func:`combine_components` distinguishes between deterministic
  and :class:`auto_uncertainties.Uncertainty` inputs.
- Virial filtering is applied to discard broad components with
  velocity offsets smaller than ``limit_velocity``.
- For rigorous posterior distributions, prefer sampling-based
  methods rather than the analytic approximation in
  :func:`combine_fast_with_jacobian`.
"""

__author__ = 'felavila'

__all__ = [
    "combine_components",
    "combine_fast",
    "combine_fast_with_jacobian",
]

from typing import Any, Dict, List, Union
import numpy as np
import jax.numpy as jnp
from jax import vmap,jit,jacfwd
from auto_uncertainties import Uncertainty

from sheap.ComplexAfterFit.Samplers.utils.physicalfunctions import calc_flux,calc_luminosity

def combine_components(
    basic_params,
    cont_group,
    cont_params,
    distances,
    LINES_TO_COMBINE=("Halpha", "Hbeta"),
    limit_velocity=150.0,
    c=299792.458,
    ucont_params = None 
):
    """
    Combine narrow and broad components of selected lines into a single
    effective profile with flux, FWHM, luminosity, and EQW.

    Parameters
    ----------
    basic_params : dict
        Dictionary of per-region fitted parameters, typically the output
        of :class:`AfterFitParams`.
        Must contain at least ``basic_params["broad"]`` and
        ``basic_params["narrow"]`` entries with keys:
        ``["lines","component","amplitude","center","fwhm_kms",...]``.
    cont_group : ComplexRegion
        Continuum region object with method ``combined_profile``.
    cont_params : jnp.ndarray
        Continuum parameter array of shape (N, P).
    distances : array-like
        Luminosity distance(s) in cm, one per object.
    LINES_TO_COMBINE : tuple of str, optional
        Line names to attempt to combine (default ``("Halpha","Hbeta")``).
    limit_velocity : float, optional
        Minimum velocity offset (km/s) for virial filtering.
    c : float, optional
        Speed of light in km/s.
    ucont_params : jnp.ndarray, optional
        Uncertainty array for continuum parameters. Required if
        input amplitudes/centers are :class:`auto_uncertainties.Uncertainty`.

    Returns
    -------
    dict
        Dictionary with combined line measurements containing:
        - ``"lines"`` : list of str
        - ``"component"`` : list of components used
        - ``"flux"`` : ndarray
        - ``"fwhm"`` : ndarray
        - ``"fwhm_kms"`` : ndarray
        - ``"center"`` : ndarray
        - ``"amplitude"`` : ndarray
        - ``"eqw"`` : ndarray
        - ``"luminosity"`` : ndarray

    Notes
    -----
    - If no valid combination is found, an empty dict is returned.
    - If inputs are :class:`Uncertainty`, then uncertainties are
      propagated using :func:`combine_fast_with_jacobian`.
    """
    combined = {}
    line_names, components = [], []
    flux_parts, fwhm_parts, fwhm_kms_parts = [], [], []
    center_parts, amp_parts, eqw_parts, lum_parts = [], [], [], []
    for line in LINES_TO_COMBINE:
        broad_lines = basic_params["broad"]["lines"]
        narrow_lines = basic_params["narrow"]["lines"]
        idx_broad = [i for i, L in enumerate(broad_lines) if L.lower() == line.lower()]
        idx_narrow = [i for i, L in enumerate(narrow_lines) if L.lower() == line.lower()]
        
        if len(idx_broad) >= 2 and len(idx_narrow) == 1:
            _components =  np.array(basic_params["broad"]["component"])[idx_broad]
            amp_b = basic_params["broad"]["amplitude"][:, idx_broad]
            mu_b = basic_params["broad"]["center"][:, idx_broad]
            fwhm_kms_b = basic_params["broad"]["fwhm_kms"][:, idx_broad]

            amp_n = basic_params["narrow"]["amplitude"][:, idx_narrow]
            mu_n = basic_params["narrow"]["center"][:, idx_narrow]
            fwhm_kms_n = basic_params["narrow"]["fwhm_kms"][:, idx_narrow]

            is_uncertainty = isinstance(amp_b, Uncertainty)

            if is_uncertainty:
                from sheap.ComplexAfterFit.Samplers.utils.afterfitprofilehelpers import evaluate_with_error 
                #print("amp_b",amp_b.shape)
                fwhm_c, amp_c, mu_c = combine_fast_with_jacobian(amp_b, mu_b, fwhm_kms_b,amp_n, mu_n, fwhm_kms_n,limit_velocity=limit_velocity,c=c)
                if fwhm_c.ndim==1:
                  #  print("fwhm_c",fwhm_c.shape)
                    #two objects 1 line 
                    fwhm_c, amp_c, mu_c = fwhm_c.reshape(-1, 1), amp_c.reshape(-1, 1), mu_c.reshape(-1, 1)
                 #   print("fwhm_c",fwhm_c.shape)
                fwhm_A = (fwhm_c / c) * mu_c
                #print(fwhm_A.shape)
                flux_c = calc_flux(amp_c, fwhm_A)
                cont_c = Uncertainty(*np.array(evaluate_with_error(cont_group.combined_profile,mu_c.value, cont_params,mu_c.error, ucont_params)))
                #ndim1 * ndim2 requires always a [:,None] to work 
                L_line = calc_luminosity(np.array(distances)[:,None], flux_c)
                eqw_c = flux_c / cont_c
                #

            else:
                N = amp_b.shape[0]
                params_broad = jnp.stack([amp_b, mu_b, fwhm_kms_b], axis=-1).reshape(N, -1)
                params_narrow = jnp.concatenate([amp_n, mu_n, fwhm_kms_n], axis=1)

                fwhm_c, amp_c, mu_c = combine_fast(params_broad, params_narrow, limit_velocity=limit_velocity, c=c)
                fwhm_A = (fwhm_c / c) * mu_c
                flux_c = calc_flux(jnp.array(amp_c), jnp.array(fwhm_A))
                cont_c = vmap(cont_group.combined_profile)(mu_c, cont_params)
                L_line = calc_luminosity(jnp.array(distances), flux_c)
                eqw_c = flux_c / cont_c
            
            line_names.extend([line])
            components.extend([_components])
            #print(flux_c)
            
            
            flux_parts.extend([flux_c])
            fwhm_parts.extend([fwhm_A])
            fwhm_kms_parts.extend([fwhm_c])
            center_parts.extend([mu_c])
            amp_parts.extend([amp_c])
            eqw_parts.extend([eqw_c])
            lum_parts.extend([L_line])
            
    if len(line_names)>0:
        #print("combination",np.concatenate(flux_parts, axis=1).shape)
        
        combined = {
            "lines": line_names,
            "component": components,
            "flux": np.concatenate(flux_parts, axis=1),
            "fwhm":  np.concatenate(fwhm_parts, axis=1),
            "fwhm_kms": np.concatenate(fwhm_kms_parts, axis=1),
            "center": np.concatenate(center_parts, axis=1),
            "amplitude": np.concatenate(amp_parts, axis=1),
            "eqw": np.concatenate(eqw_parts, axis=1),
            "luminosity": np.concatenate(lum_parts, axis=1),
            }   
        # for key,values in combined.items():
        #     try:
        #         print(key,values.shape)  
        #     except:
        #         print("list",key,values)  
        return combined
    else:
        return combined




@jit
def combine_fast(
    params_broad: jnp.ndarray,
    params_narrow: jnp.ndarray,
    limit_velocity: float = 150.0,
    c: float = 299_792.0,
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
    c : float, optional
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
    rel_vel  = jnp.abs((mu_b - mu_nar[:, None]) / mu_nar[:, None]) * c
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
    amp_b: Uncertainty,
    mu_b: Uncertainty,
    fwhm_b: Uncertainty,
    amp_n: Uncertainty,
    mu_n: Uncertainty,
    fwhm_n: Uncertainty,
    limit_velocity: float = 150.0,
    c: float = 299792.458,
    use_jacobian: bool = True,
    rough_scale: float = 1.0
) -> tuple[Uncertainty, Uncertainty, Uncertainty]:
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
    c : float, optional
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
    N = amp_b.value.shape[0]
    n_broad = amp_b.value.shape[1]
    results = []

    for i in range(N):
        # Flatten input vector
        x0 = jnp.concatenate([
            amp_b.value[i], mu_b.value[i], fwhm_b.value[i],
            amp_n.value[i], mu_n.value[i], fwhm_n.value[i]
        ])
        errors = jnp.concatenate([
            amp_b.error[i], mu_b.error[i], fwhm_b.error[i],
            amp_n.error[i], mu_n.error[i], fwhm_n.error[i]
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
            return jnp.array(combine_fast(pb, pn, limit_velocity, c)).squeeze()

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
        Uncertainty(np.array(fwhm_vals), np.array(fwhm_errs)),
        Uncertainty(np.array(amp_vals),  np.array(amp_errs)),
        Uncertainty(np.array(mu_vals),   np.array(mu_errs))
    )



