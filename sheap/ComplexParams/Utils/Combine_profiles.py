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
  and :class:`uncertainties` inputs.
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
from jax import vmap,jit,jacfwd,lax
from uncertainties import unumpy

from sheap.ComplexParams.Utils.Physical_functions import calc_flux,calc_luminosity

c_kms = 299792.458 # this will have to move as the constant in every where 

def combine_components(
    basic_params,
    cont_group,
    cont_params,
    distances,
    flux_fe=None,
    LINES_TO_COMBINE=("Halpha", "Hbeta","MgII","CIV"),
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
        input amplitudes/centers are :class:`?`.

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
        broad_lines = basic_params.get("broad", {}).get("lines",[])
        narrow_lines = basic_params.get("narrow", {}).get("lines",[])
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

            #is_uncertainty = isinstance(amp_b, Uncertainty)
            is_uncertainty = amp_b.dtype== 'O'
            if is_uncertainty:
                from sheap.ComplexParams.Utils.After_fit_profile_helpers import evaluate_with_error 
                #print("amp_b",amp_b.shape)
                fwhm_c, amp_c, mu_c = combine_fast_with_jacobian(amp_b, mu_b, fwhm_kms_b,amp_n, mu_n, fwhm_kms_n,limit_velocity=limit_velocity,c=c)
                
                if fwhm_c.ndim==1:
                  #  print("fwhm_c",fwhm_c.shape)
                    #two objects 1 line 
                    fwhm_c, amp_c, mu_c = fwhm_c.reshape(-1, 1), amp_c.reshape(-1, 1), mu_c.reshape(-1, 1)
                 #   print("fwhm_c",fwhm_c.shape)
                fwhm_A = (fwhm_c / c) * mu_c
                #print(fwhm_A.shape)
                #unumpy.nominal_values,unumpy.std_devs
                flux_c = calc_flux(amp_c, fwhm_A)
                cont_c = unumpy.uarray(*np.array(evaluate_with_error(cont_group.combined_profile,unumpy.nominal_values(mu_c), cont_params,unumpy.std_devs(mu_c), ucont_params)))
                #ndim1 * ndim2 requires always a [:,None] to work 
                L_line = calc_luminosity(np.array(distances)[:,None], flux_c)
                eqw_c = flux_c / cont_c
                #

            else:
                N = amp_b.shape[0]
                params_broad = jnp.stack([amp_b, mu_b, fwhm_kms_b], axis=-1).reshape(N, -1)
                params_narrow = jnp.concatenate([amp_n, mu_n, fwhm_kms_n], axis=1)

                fwhm_c, amp_c, mu_c = combine_fast(params_broad, params_narrow, limit_velocity=limit_velocity, c=c)
                if fwhm_c.ndim==1:
                    fwhm_c, amp_c, mu_c = fwhm_c.reshape(-1, 1), amp_c.reshape(-1, 1), mu_c.reshape(-1, 1)

                fwhm_A = (fwhm_c / c) * mu_c
                flux_c = calc_flux(jnp.array(amp_c), jnp.array(fwhm_A))
                #print(flux_c.shape)
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
        #if flux_fe:
        unique_lines = np.unique(combined["lines"])
        new_dict = {}
        for line in unique_lines:
            # Boolean mask for this line
            mask = np.where(np.array(combined["lines"]) == line)[0]
            #print(mask)
            # Build a sub-dictionary slicing each array along axis=1
            new_dict[line] = {
                "component": np.array(combined["component"]),
                "flux": combined["flux"][:, mask],
                "fwhm": combined["fwhm"][:, mask],
                "fwhm_kms": combined["fwhm_kms"][:, mask],
                "center": combined["center"][:, mask],
                "amplitude": combined["amplitude"][:, mask],
                "eqw": combined["eqw"][:, mask],
                "luminosity": combined["luminosity"][:, mask],
                "extras":{"R_Fe":flux_fe}
            }        
        # combined["extras"] = {}
        # combined["extras"]["R_Fe"] = flux_fe
        # for key,values in combined.items():
        #     try:
        #         print(key,values.shape)  
        #     except:
        #         print("list",key,values)  
        
        
        return new_dict
    else:
        return combined



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
    amp_b,
    mu_b,
    fwhm_b,
    amp_n,
    mu_n,
    fwhm_n,
    limit_velocity: float = 150.0,
    c: float = 299792.458,
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
        unumpy.uarray(np.array(fwhm_vals), np.array(fwhm_errs)),
        unumpy.uarray(np.array(amp_vals),  np.array(amp_errs)),
        unumpy.uarray(np.array(mu_vals),   np.array(mu_errs))
    )



def region_helper(wavelength,region_name,complex_class_group_by_region,params,on_axis_wavelength = None):
    "?"
    if region_name not in complex_class_group_by_region.keys():
        return np.array([0])
    _combined_profile  = complex_class_group_by_region[region_name].combined_profile
    index_interest_params = complex_class_group_by_region[region_name].flat_param_indices_global
    from_complex_params = complex_class_group_by_region[region_name].params
    params = from_complex_params if on_axis_wavelength == 0 else params[:,index_interest_params]  
    #params = complex_class_group_by_region[region_name].params
    return vmap(_combined_profile,(on_axis_wavelength,0))(wavelength,params)

class GaussianSum:
    def __init__(self, n, constraints=None, inequalities=None):
        """
        Initialize the GaussianSum with parameter constraints.

        Parameters:
        - n (int): Number of Gaussian functions.
        - constraints (dict): Optional equality constraints on parameters.
            Example:
                {
                    'amp': [('amp0', 'amp1')],  # amp0 == amp1
                    'mu': [('mu2', 'mu3')],
                    'sigma': [('sigma1', 'sigma2')]
                }
        - inequalities (dict): Optional inequality constraints on parameters.
            Example:
                {
                    'sigma': [('sigma1', 'sigma2')]  # sigma2 > sigma1
                }
        """
        self.n = n
        self.constraints = constraints or {}
        self.inequalities = inequalities or {}
        # Determine free parameters based on constraints
        self.param_mapping = self._build_param_mapping()
        # Calculate the number of free parameters
        self.num_free_params = self._count_free_params()
        # Build the JIT-compiled Gaussian sum function
        self.sum_gaussians_jit = self._build_gaussian_sum()

    def _build_param_mapping(self):
        """
        Build a mapping from free parameters to all parameters,
        applying constraints as specified.
        """
        # Initialize mappings: each parameter maps to itself initially
        mapping = {
            'amp': list(range(self.n)),
            'mu': list(range(self.n)),
            'sigma': list(range(self.n))
        }

        # Apply equality constraints
        for param_type, pairs in self.constraints.items():
            for (p1, p2) in pairs:
                idx1 = int(p1.replace(param_type, ''))
                idx2 = int(p2.replace(param_type, ''))
                mapping[param_type][idx2] = mapping[param_type][idx1]

        return mapping

    def _count_free_params(self):
        """
        Count the number of free parameters after applying constraints.
        """
        free_amp = len(set(self.param_mapping['amp']))
        free_mu = len(set(self.param_mapping['mu']))
        free_sigma = len(set(self.param_mapping['sigma']))
        return free_amp + free_mu + free_sigma + self._count_inequality_free_params()

    def _count_inequality_free_params(self):
        """
        Count additional free parameters required for inequality constraints.
        For each inequality, an extra free parameter is needed to define the offset.
        """
        count = 0
        for param_type, pairs in self.inequalities.items():
            count += len(pairs)
        return count

    def _apply_constraints(self, params):
        """
        Apply equality constraints to the parameter vector to obtain full parameter sets.

        Parameters:
        - params (jnp.ndarray): Free parameters vector.

        Returns:
        - amps, mus, sigmas (tuple of jnp.ndarray): Full parameter sets.
        """
        free_amp = self.param_mapping['amp']
        free_mu = self.param_mapping['mu']
        free_sigma = self.param_mapping['sigma']

        num_free_amp = len(set(free_amp))
        num_free_mu = len(set(free_mu))
        num_free_sigma = len(set(free_sigma))

        # Extract free parameters
        idx = 0
        amps_free = params[idx:idx + num_free_amp]
        idx += num_free_amp
        mus_free = params[idx:idx + num_free_mu]
        idx += num_free_mu
        sigmas_free = params[idx:idx + num_free_sigma]
        idx += num_free_sigma

        # Map free parameters to all parameters using the mapping
        amps = jnp.array([amps_free[i] for i in self.param_mapping['amp']])
        mus = jnp.array([mus_free[i] for i in self.param_mapping['mu']])
        sigmas = jnp.array([sigmas_free[i] for i in self.param_mapping['sigma']])

        return amps, mus, sigmas

    def _apply_inequality_constraints(self, sigmas, params):
        """
        Apply inequality constraints to sigmas.

        For example, enforce sigma2 > sigma1 by setting sigma2 = sigma1 + softplus(delta)

        Parameters:
        - sigmas (jnp.ndarray): Current sigma parameters.
        - params (jnp.ndarray): Remaining parameters for inequality transformations.

        Returns:
        - jnp.ndarray: Transformed sigma parameters satisfying inequalities.
        """
        if not self.inequalities:
            return sigmas

        # Assuming all inequality constraints are on 'sigma'
        for (s1, s2) in self.inequalities.get('sigma', []):
            idx1 = int(s1.replace('sigma', ''))
            idx2 = int(s2.replace('sigma', ''))
            delta = params[0]
            params = params[1:]
            transformed_sigma2 = sigmas[idx1] + jax.nn.softplus(delta)
            sigmas = sigmas.at[idx2].set(transformed_sigma2)
        return sigmas

    def _build_gaussian_sum(self):
        """
        Build the JIT-compiled Gaussian sum function.

        Returns:
        - sum_gaussians_jit (function): JIT-compiled function.
        """
        def gaussian(x, amp, mu, sigma):
            return amp * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)


        def sum_gaussians(x, params):
            # Validate parameter length
            if params.shape[0] != self.num_free_params:
                raise ValueError(f"Expected {self.num_free_params} parameters, got {params.shape[0]}.")

            # Apply equality constraints
            amps, mus, sigmas = self._apply_constraints(params)
            
            # Apply inequality constraints if any
            if self.inequalities:
                # Extract deltas for inequalities
                delta_params = params[-len(self.inequalities.get('sigma', [])):]
                sigmas = self._apply_inequality_constraints(sigmas, delta_params)

            # Use a lambda to fix 'x' while vectorizing over amp, mu, sigma
            gaussians = vmap(lambda amp, mu, sigma: gaussian(x, amp, mu, sigma))(amps, mus, sigmas)
            
            return jnp.sum(gaussians, axis=0)
        self.n_params = self.num_free_params
        return jit(sum_gaussians)

    def __call__(self, x, params):
        """
        Compute the sum of Gaussians at points x with given parameters.

        Parameters:
        - x (jnp.ndarray): Points at which to evaluate the sum.
        - params (jnp.ndarray): Free parameters vector.

        Returns:
        - jnp.ndarray: Sum of Gaussians evaluated at x.
        """
        
        return self.sum_gaussians_jit(x, params)


def combine_pyqsofit(basic_params,complex_class_group_by_region,line,params,distances,flux_fe):
    #if isinstance(LINES_TO_COMBINE,str):
     #   LINES_TO_COMBINE = [LINES_TO_COMBINE]
    b_lines = np.array(basic_params["lines"])
    
    #for line in LINES_TO_COMBINE:
    # re-interpretation of https://github.com/legolason/PyQSOFit/issues/4?utm_source=chatgpt.com
    idx_b = np.where(np.char.lower(b_lines) == line.lower())[0]
    params = params.astype(jnp.float32)
    gg = GaussianSum(len(idx_b))
    b_mu = jnp.asarray(basic_params["center"])[:,idx_b].astype(jnp.float32)
    b_sigma = jnp.asarray(basic_params["fwhm"])[:,idx_b].astype(jnp.float32) /  (2.0 * np.sqrt(2.0 * np.log(2.0)))
    b_amp   = jnp.asarray(basic_params["amplitude"])[:,idx_b].astype(jnp.float32)    # (Nobj, NB)
    #print(b_amp)
    _ = np.stack([b_amp, b_mu,b_sigma], axis=2)
    line_params = jnp.array(_.transpose(0, 2, 1).reshape(_.shape[0], -1)).astype(jnp.float32)
    left = np.min(b_mu - 3*b_sigma,axis=1)
    right = np.max(b_mu + 3*b_sigma,axis=1)

    disp = 1.e-4 #hyperparam 
    npix = 50_000 #int(max((right-left)/disp))  #(maybe it is 2 much)
    #npix = int(max((right-left)/disp))  #(maybe it is 2 much)
    
    wave = jnp.linspace(np.min(left), np.max(right), npix, dtype=jnp.float32)
    model_sum = vmap(gg,in_axes=(None,0))(wave,line_params).astype(jnp.float32)
    cont_map = region_helper(wave,"continuum" ,complex_class_group_by_region, params,on_axis_wavelength = None).squeeze().astype(jnp.float32)
    lambda_ref = {"Halpha": 6564.61,  "Hbeta": 4862.68,"MgII": 2798.75,"CIV": 1549.48}[line]
    
    i_peak     = jnp.argmax(model_sum, axis=1)            
    #peak_A     = wave[i_peak]                         
    half       = 0.5 * jnp.max(model_sum, axis=1)     
    f          = model_sum - half[:, None]                 
    #s          = jnp.sign(f)                               


    cont_safe  = jnp.maximum(cont_map, 1e-30)
    eqw       = jnp.trapezoid(model_sum / cont_safe, wave, axis=1) 


    Nlam   = wave.shape[0]
    idxs   = jnp.arange(Nlam - 1)                    
    eps = 1e-30

    def interp_at(k, f_row):
        # linear interpolation of zero crossing between k and k+1
        x0, x1 = wave[k],   wave[k + 1]
        y0, y1 = f_row[k],  f_row[k + 1]
        t = -y0 / (y1 - y0 + eps)
        return x0 + t * (x1 - x0)

    def row_fwhm(f_row, i_peak_i):
        s_row      = jnp.sign(f_row)                  # (Nlam,)
        cross_mask = (s_row[:-1] * s_row[1:] ) < 0    # (Nlam-1,)

        left_cand  = jnp.where((idxs < i_peak_i) & cross_mask, idxs, -1)
        left_idx   = jnp.max(left_cand)               # -1 if none

        right_cand = jnp.where((idxs >= i_peak_i) & cross_mask, idxs, Nlam)
        right_idx  = jnp.min(right_cand)              # Nlam if none

        has_left   = left_idx  >= 0
        has_right  = right_idx <= (Nlam - 2)

        lam_L = jnp.where(has_left,  interp_at(left_idx,  f_row), jnp.nan)
        lam_R = jnp.where(has_right, interp_at(right_idx, f_row), jnp.nan)

        return lam_L, lam_R

    lam_L, lam_R = vmap(row_fwhm, in_axes=(0, 0))(f.astype(jnp.float32), i_peak.astype(jnp.float32))   # (Nobj,), (Nobj,)

    fwhm_kms = ((lam_R - lam_L) / lambda_ref) * c_kms   
    sigma_kms = fwhm_kms / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    #print(fwhm_kms)
    flux  = np.trapezoid(model_sum, wave, axis=1)
    luminosity = 4.0 * np.pi * distances**2 * flux
    #calc_luminosity(jnp.array(distances), flux_c)
    method_2 = {"fwhm_kms":fwhm_kms,"eqw":eqw,"lines":line,"sigma_kms": sigma_kms,"luminosity":luminosity,"flux":flux}
    method_2["extras"] = {}
    #print(flux_fe/flux)
    method_2["extras"]["R_Fe"] = flux_fe/flux
    return method_2


def combine_pyqsofit_single(basic_params,complex_class_group_by_region,line,distances,flux_fe):
    b_lines = np.array(basic_params["lines"])
    idx_b = np.where(np.char.lower(b_lines) == line.lower())[0]
    gg = GaussianSum(len(idx_b))
    b_mu = unumpy.nominal_values(np.asarray(basic_params["center"])[:,idx_b]) #.nominal_values #.astype(jnp.float32)
    b_sigma = unumpy.nominal_values(np.asarray(basic_params["fwhm"])[:,idx_b] /  (2.0 * np.sqrt(2.0 * np.log(2.0))))
    b_amp   =  unumpy.nominal_values(np.asarray(basic_params["amplitude"])[:,idx_b])
    _ = np.stack([b_amp, b_mu,b_sigma], axis=2)
    line_params = jnp.array(_.transpose(0, 2, 1).reshape(_.shape[0], -1)).astype(jnp.float32)

    ##########control########
    finite_mask = (
        np.all(np.isfinite(b_mu), axis=1)
        & np.all(np.isfinite(b_sigma), axis=1)
        & np.all(np.isfinite(b_amp), axis=1)
    )
    # optionally also require positive widths
    finite_mask &= np.all(b_sigma > 0, axis=1)
    ##########control########
    left = np.nanmin(b_mu[finite_mask] - 4*b_sigma[finite_mask],axis=1)
    right = np.nanmax(b_mu[finite_mask] + 4*b_sigma[finite_mask],axis=1)

    
    
    npix = 100_000 #int(max((right-left)/disp))  #(maybe it is 2 much)
    wave = jnp.linspace(np.min(left), np.max(right), npix, dtype=jnp.float32)
    #disp = wave[1] - wave[0]
    #print(disp)
    model_sum = vmap(gg,in_axes=(None,0))(wave,line_params).astype(jnp.float32)
    ###################################################
    _combined_profile  = complex_class_group_by_region["continuum"].combined_profile
    from_complex_params = complex_class_group_by_region["continuum"].params.astype(jnp.float32)
    continuum_vmap =  vmap(_combined_profile,(None,0))(wave,from_complex_params)
    lambda_ref = {"Halpha": 6564.61,  "Hbeta": 4862.68,"MgII": 2798.75,"CIV": 1549.48}[line]
    i_peak     = jnp.argmax(model_sum, axis=1)            
                       
    half       = 0.5 * jnp.max(model_sum, axis=1)     
    f          = model_sum - half[:, None]                                              
    cont_safe  = jnp.maximum(continuum_vmap, 1e-30)
    eqw       = jnp.trapezoid(model_sum / cont_safe, wave, axis=1) 


    Nlam   = wave.shape[0]
    idxs   = jnp.arange(Nlam - 1)                    
    eps = 1e-30

    def interp_at(k, f_row):
        # linear interpolation of zero crossing between k and k+1
        x0, x1 = wave[k],   wave[k + 1]
        y0, y1 = f_row[k],  f_row[k + 1]
        t = -y0 / (y1 - y0 + eps)
        return x0 + t * (x1 - x0)

    def row_fwhm(f_row, i_peak_i):
        s_row      = jnp.sign(f_row)                  # (Nlam,)
        cross_mask = (s_row[:-1] * s_row[1:] ) < 0    # (Nlam-1,)

        left_cand  = jnp.where((idxs < i_peak_i) & cross_mask, idxs, -1)
        left_idx   = jnp.max(left_cand)               # -1 if none

        right_cand = jnp.where((idxs >= i_peak_i) & cross_mask, idxs, Nlam)
        right_idx  = jnp.min(right_cand)              # Nlam if none

        has_left   = left_idx  >= 0
        has_right  = right_idx <= (Nlam - 2)

        lam_L = jnp.where(has_left,  interp_at(left_idx,  f_row), jnp.nan)
        lam_R = jnp.where(has_right, interp_at(right_idx, f_row), jnp.nan)

        return lam_L, lam_R

    lam_L, lam_R = vmap(row_fwhm, in_axes=(0, 0))(f.astype(jnp.float32), i_peak.astype(jnp.float32))   # (Nobj,), (Nobj,)

    fwhm_kms = ((lam_R - lam_L) / lambda_ref) * c_kms   
    
    sigma_kms = fwhm_kms / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    flux  = np.trapezoid(model_sum, wave, axis=1)
    #np.sqrt(2.0 * np.pi) * np.max(model_sum, axis=1) * sigma_kms # mmmm
    luminosity = 4.0 * np.pi * distances**2 * flux
    #calc_luminosity(jnp.array(distances), flux_c)
    
    method_2 = {"fwhm_kms":unumpy.uarray(fwhm_kms, np.zeros_like(fwhm_kms)) ,"eqw":unumpy.uarray(eqw, np.zeros_like(fwhm_kms)),"lines":line,"sigma_kms": unumpy.uarray(eqw, np.zeros_like(sigma_kms)),"luminosity": unumpy.uarray(luminosity, np.zeros_like(sigma_kms)),"flux":unumpy.uarray(flux, np.zeros_like(flux))}
    method_2["extras"] = {}
    try:
        method_2["extras"]["R_Fe"] = flux_fe.squeeze()/flux
    except:
        #check on this with more detail
        method_2["extras"]["R_Fe"] = 0
    return method_2



def combine_fastspecfit(wavelength_spectra,flux_spectra,targets,basic_params_broad,complex_class_group_by_region,on_axis_wavelength=None):
    c_kms = 299792.458 # this will have to move as the constant in every where 
    if isinstance(targets,str):
        targets = {"MgII"}
    #targets = {"MgII"}
#what happend when you have multiple names on it?
    index_map = defaultdict(list)
    for idx, name in enumerate(basic_params_broad["lines"]):
        index_map[name].append(idx)

    # convert back to normal dict if you don’t want defaultdict
    index_map = dict(index_map)
    positions = {t: index_map[t] for t in targets if t in index_map}
    wavelength_spectra = sheapspectral.spectra[[0],0,:]
    flux_spectra = sheapspectral.spectra[[0],1,:]
    #on_axis_wavelength = None 
    #complex_class_group_by_region = sheapspectral.result.complex_class.group_by("region")


    fe_map = region_helper(wavelength_spectra,"fe",complex_class_group_by_region, params,on_axis_wavelength = on_axis_wavelength).squeeze()
    cont_map = region_helper(wavelength_spectra,"continuum" ,complex_class_group_by_region, params,on_axis_wavelength = on_axis_wavelength).squeeze()
    host_map = region_helper(wavelength_spectra,"host",complex_class_group_by_region, params,on_axis_wavelength = on_axis_wavelength).squeeze()

    emission_spectra = np.clip(flux_spectra - (fe_map + cont_map + host_map),0.0, None).squeeze()


    _flux = basic_params_broad["flux"][:,list(*positions.values())]
    _center = basic_params_broad["center"][:,list(*positions.values())]
    _fwhm = basic_params_broad["fwhm"][:,list(*positions.values())]
                
    weighted_center = np.sum((_center*_flux),axis=1)/np.sum((_flux),axis=1)
    if isinstance(_fwhm, np.ndarray) and _fwhm.dtype == object and _fwhm.size:
        _sigma = unumpy.nominal_values(np.mean(_fwhm,axis=1))/(2*np.sqrt(2*np.log(2)))
        _low = unumpy.nominal_values(weighted_center - 5*_sigma)
        _up = unumpy.nominal_values(weighted_center + 5*_sigma)
    else:    
        _sigma = np.mean(_fwhm,axis=1)/(2*np.sqrt(2*np.log(2)))
        _low = weighted_center - 5*_sigma
        _up = weighted_center + 5*_sigma

    _mask = (wavelength_spectra >= _low[:, None]) & (wavelength_spectra  <= _up[:, None])
    W = np.sum(emission_spectra *  _mask,axis=1)
    M1 = np.sum(emission_spectra *  _mask * wavelength_spectra,axis=1)/W
    M2 = np.sum(emission_spectra *  _mask * (wavelength_spectra - M1[:,None])**2,axis=1)/W
    sigma_f = c_kms * np.sqrt(M2)/M1
    fwhm_f = 2 * np.sqrt(2 * np.log(2)) * sigma_f
    
    method_1 = {"weighted_center": weighted_center,"sigma_kms":sigma_f,"fwhm_kms":fwhm_f,"W":W,"M1":M1,"M2":M2,"targets":targets}
    return method_1




# def combine_pyqsofit(basic_params, complex_class_group_by_region, line, params, distances, flux_fe):
#     b_lines = np.array(basic_params["lines"])
    
#     idx_b = np.where(np.char.lower(b_lines) == line.lower())[0]
#     params = params.astype(jnp.float32)
#     gg = GaussianSum(len(idx_b))
    
#     # Extract and convert to float32 immediately, avoid intermediate copies
#     b_mu = jnp.asarray(basic_params["center"][:, idx_b], dtype=jnp.float32)
#     b_sigma = jnp.asarray(basic_params["fwhm"][:, idx_b], dtype=jnp.float32) / (2*np.sqrt(2)*np.log(2))
#     b_amp = jnp.asarray(basic_params["amplitude"][:, idx_b], dtype=jnp.float32)
    
#     # Compute line_params more efficiently
#     line_params = jnp.stack([b_amp, b_mu, b_sigma], axis=2).reshape(b_amp.shape[0], -1)
    
#     # Compute bounds
#     left = jnp.min(b_mu - 3*b_sigma, axis=1)
#     right = jnp.max(b_mu + 3*b_sigma, axis=1)
    
#     disp = 1.e-4
#     npix = int(max((right - left) / disp))
    
#     # Create wave grid
#     wave = jnp.linspace(float(jnp.min(left)), float(jnp.max(right)), npix, dtype=jnp.float32)
    
#     # Compute model_sum
#     model_sum = vmap(gg, in_axes=(None, 0))(wave, line_params)
    
#     # Compute continuum - use squeeze to reduce dimensionality
#     cont_map = region_helper(wave, "continuum", complex_class_group_by_region, 
#                             params, on_axis_wavelength=None).squeeze()
    
#     lambda_ref = {"Halpha": 6564.61, "Hbeta": 4862.68, "MgII": 2798.75, "CIV": 1549.48}[line]
    
#     # Find peaks and compute EQW
#     i_peak = jnp.argmax(model_sum, axis=1)
#     model_max = jnp.max(model_sum, axis=1)
#     half = 0.5 * model_max
#     f = model_sum - half[:, None]
    
#     # Compute EQW with safe continuum
#     cont_safe = jnp.maximum(cont_map, 1e-30)
#     eqw = jnp.trapezoid(model_sum / cont_safe, wave, axis=1)
    
#     # FWHM calculation setup
#     Nlam = wave.shape[0]
#     idxs = jnp.arange(Nlam - 1)
#     eps = 1e-30
    
#     def interp_at(k, f_row):
#         x0, x1 = wave[k], wave[k + 1]
#         y0, y1 = f_row[k], f_row[k + 1]
#         t = -y0 / (y1 - y0 + eps)
#         return x0 + t * (x1 - x0)
    
#     def row_fwhm(f_row, i_peak_i):
#         s_row = jnp.sign(f_row)
#         cross_mask = (s_row[:-1] * s_row[1:]) < 0
        
#         left_cand = jnp.where((idxs < i_peak_i) & cross_mask, idxs, -1)
#         left_idx = jnp.max(left_cand)
        
#         right_cand = jnp.where((idxs >= i_peak_i) & cross_mask, idxs, Nlam)
#         right_idx = jnp.min(right_cand)
        
#         has_left = left_idx >= 0
#         has_right = right_idx <= (Nlam - 2)
        
#         lam_L = jnp.where(has_left, interp_at(left_idx, f_row), jnp.nan)
#         lam_R = jnp.where(has_right, interp_at(right_idx, f_row), jnp.nan)
        
#         return lam_L, lam_R
    
#     lam_L, lam_R = vmap(row_fwhm, in_axes=(0, 0))(f, i_peak)
    
#     # Final calculations
#     fwhm_kms = ((lam_R - lam_L) / lambda_ref) * c_kms
#     sigma_kms = fwhm_kms / (2.0 * np.sqrt(2.0 * np.log(2.0)))
#     flux = np.sqrt(2.0 * np.pi) * model_max * sigma_kms
#     luminosity = 4.0 * np.pi * distances**2 * flux
    
#     return {
#         "fwhm_kms": fwhm_kms,
#         "eqw": eqw,
#         "lines": line,
#         "sigma_kms": sigma_kms,
#         "luminosity": luminosity,
#         "flux": flux,
#         "extras": {"R_Fe": flux_fe}
#     }

#batch_size=32, max_npix=50000
