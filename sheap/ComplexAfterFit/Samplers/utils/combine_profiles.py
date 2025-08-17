"""Tools to combine components."""
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
    Combine any number of broad Gaussians + a narrow Gaussian per object,
    returning only (fwhm_final, amp_final, mu_final).

    Inputs
    ------
    params_broad : (N, 3*n_broad) array: [amp_i, mu_i, fwhm_i,...].
    params_narrow: (N, 3) array: [amp_n, mu_n, fwhm_n] but only mu_n used.
    limit_velocity : velocity threshold for virial filtering.
    c              : speed of light (same units as velocities).

    Returns
    -------
    fwhm_final : (N,) — chosen FWHM (in same units as input).
    amp_final  : (N,) — chosen amplitude.
    mu_final   : (N,) — chosen center.
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
    Rough analytic uncertainty propagation for `combine_fast`.

    Parameters
    ----------
    amp_b : Uncertainty
        Amplitudes of broad components.
    mu_b : Uncertainty
        Centers of broad components.
    fwhm_b : Uncertainty
        FWHMs of broad components.
    amp_n : Uncertainty
        Amplitudes of narrow component.
    mu_n : Uncertainty
        Centers of narrow component.
    fwhm_n : Uncertainty
        FWHM of narrow component.
    limit_velocity : float, optional
        Velocity threshold in km/s used in profile merging. Default is 150.
    c : float, optional
        Speed of light in km/s. Default is 299792.458.
    use_jacobian : bool, optional
        Whether to use JAX’s jacfwd for uncertainty propagation. If False, uses a rough scale estimate.
    rough_scale : float, optional
        Scaling factor applied to combined outputs if `use_jacobian` is False. Default is 1.0.

    Returns
    -------
    fwhm : Uncertainty
        FWHM of the combined profile.
    amp : Uncertainty
        Amplitude of the combined profile.
    mu : Uncertainty
        Center of the combined profile.

    Notes
    -----
    This function uses analytic uncertainty propagation via Jacobian computation with JAX,
    or falls back to a rough scaling of the combined output value. It is not intended
    to provide rigorous uncertainty estimates. For proper posterior sampling, use
    dedicated inference tools such as ?.
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



