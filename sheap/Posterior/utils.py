import jax.numpy as jnp
from jax import vmap,grad,jit


#from sheap.Mappers.LineMapper import LineSelectionResult

#This is more than utils 

def trapz_jax(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    dx = x[1:] - x[:-1]
    return jnp.sum((y[1:] + y[:-1]) * dx / 2)

def integrate_function_error(function, x: jnp.ndarray, p: jnp.ndarray, sigma_p: jnp.ndarray = None):
    """
    Computes the integral of a function and propagates the error on the parameters.

    Parameters:
    -----------
    function : Callable
        Function to evaluate: function(x, p)
    x : jnp.ndarray
        Grid over which to integrate.
    p : jnp.ndarray
        Parameters for the function.
    sigma_p : jnp.ndarray, optional
        Standard deviation (uncertainty) for each parameter. Defaults to zero.

    Returns:
    --------
    y_int : float
        The integral of the function over `x`.
    sigma_f : float
        Propagated uncertainty on the integral due to `sigma_p`.
    """
    p = jnp.atleast_1d(p)
    sigma_p = jnp.zeros_like(p) if sigma_p is None else jnp.atleast_1d(sigma_p)

    def int_function(p_):
        return trapz_jax(function(x, p_), x)

    y_int = int_function(p)
    grad_f = grad(int_function)(p)

    
    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2))
    return y_int, sigma_f

def integrate_function_error_single(function, x, p, sigma_p):
    def int_function(p_):
        return trapz_jax(function(x, p_), x)

    y_int = int_function(p)
    grad_f = grad(int_function)(p)
    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2))
    return y_int, sigma_f


# def compute_integrated_profiles(LineSelectionResult:LineSelectionResult, delta=1000, n_points=1000):
#     """
#     Compute integrated flux and propagated error for each broad component
#     in a LineSelectionResult, using preallocated JAX arrays.

#     Parameters
#     ----------
#     LineSelectionResult : LineSelectionResult
#         An instance of LineSelectionResult filtered with kind="broad".
#     delta : float
#         Half-width of integration window around the center.
#     n_points : int
#         Number of integration points.

#     Returns
#     -------
#     values_ : jnp.ndarray, shape (n_spectra, n_lines)
#         Integrated fluxes.
#     errors_ : jnp.ndarray, shape (n_spectra, n_lines)
#         Propagated uncertainties.
#     """
#     original_centers = LineSelectionResult.original_centers
#     n_lines = len(original_centers)
#     n_spectra = LineSelectionResult.params.shape[0]

#     values_ = jnp.zeros((n_spectra,n_lines))
#     errors_ = jnp.zeros((n_spectra,n_lines))
#     pos_idx = 0
#     for i in range(len(original_centers)):
#         L1=  LineSelectionResult.profile_params_index_list[i]
#         f1 = LineSelectionResult.profile_functions[i]
#         f1_params = LineSelectionResult.params[:,pos_idx:pos_idx+len(L1)]
#         f1_uncertainty_params = LineSelectionResult.uncertainty_params[:,pos_idx:pos_idx+len(L1)]
#         #print(pos_idx,LineSelectionResult.params_names[pos_idx:pos_idx+len(L1)])
#         idx = next((i for i, name in enumerate(LineSelectionResult.params_names[pos_idx:pos_idx+len(L1)]) if "center" in name), None)
#         if idx is not None:
#             centers = f1_params[:, idx]
#             x = jnp.stack([jnp.linspace(c - delta, c + delta, n_points)for c in centers])
#         else:
#             idx = next((i for i, name in enumerate(LineSelectionResult.params_names[pos_idx:pos_idx+len(L1)]) if "shift" in name), None)
#             #if it is shift a it is ok because the thing what we wan it is stimate a "realistic" x 
#             #f1_params = jnp.array(f1_params).at[:, idx].set(original_centers[i] - f1_params[:, idx])
#             x = jnp.stack([jnp.linspace(c - delta, c + delta, n_points)for c in centers])
#         vmapped_func = vmap(integrate_function_error_single, in_axes=(None, 0, 0, 0))
#         values,errors = vmapped_func(f1,x,f1_params,f1_uncertainty_params)
#         values_ = values_.at[:,i].set(values)
#         errors_ = errors_.at[:,i].set(errors)
#         pos_idx += len(L1)
#     return values_,errors_


# def effective_fwhm(params1, params2):
#     """
#     Estimate the FWHM of a combined two-Gaussian profile using moment analysis.
#     Each param is a tuple/list/array of (amp, mu, sigma).
#     """
#     amp1, mu1, sigma1 = params1
#     amp2, mu2, sigma2 = params2

#     total_amp = amp1 + amp2
#     mu_eff = (amp1 * mu1 + amp2 * mu2) / total_amp
#     var_eff = (
#         amp1 * (sigma1**2 + (mu1 - mu_eff) ** 2) + amp2 * (sigma2**2 + (mu2 - mu_eff) ** 2)
#     ) / total_amp

#     sigma_eff = jnp.sqrt(var_eff)
#     fwhm = 2.35482 * sigma_eff
#     return total_amp, mu_eff, sigma_eff, fwhm / mu_eff  # dimensionless


# batched_fwhm = vmap(
#     vmap(effective_fwhm, in_axes=(0, 0)), in_axes=(0, 0)  # over regions  # over samples
# )


# import jax.numpy as jnp
# from jax import jit

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

    # 1) Weighted mean center & moment‐based FWHM
    total_amp = jnp.sum(amp_b, axis=1)                      # (N,)
    mu_eff    = jnp.sum(amp_b * mu_b, axis=1) / total_amp

    invf = 1.0 / 2.35482
    var_i   = (fwhm_b * invf) ** 2                          # variance per component
    dif2    = (mu_b - mu_eff[:, None]) ** 2
    var_eff = jnp.sum(amp_b * (var_i + dif2), axis=1) / total_amp
    fwhm_eff= jnp.sqrt(var_eff) * 2.35482                   # (N,)

    # 2) Closest‐to‐narrow component
    mu_nar   = params_narrow[:, 1]
    rel_vel  = jnp.abs((mu_b - mu_nar[:, None]) / mu_nar[:, None]) * c
    idx_near = jnp.argmin(rel_vel, axis=1)

    sel = lambda arr: arr[jnp.arange(N), idx_near]
    fwhm_nb  = sel(fwhm_b)
    amp_nb   = sel(amp_b)
    mu_nb    = sel(mu_b)

    # 3) Amplitude‐ratio mask
    amp_ratio = jnp.min(amp_b, axis=1) / jnp.max(amp_b, axis=1)
    mask_amp  = amp_ratio > 0.1

    fwhm_choice = jnp.where(mask_amp, fwhm_eff, fwhm_nb)
    amp_choice  = jnp.where(mask_amp, total_amp, amp_nb)
    mu_choice   = jnp.where(mask_amp, mu_eff, mu_nb)

    # 4) Virial filter
    mask_vir = jnp.min(rel_vel, axis=1) >= limit_velocity
    fwhm_final = jnp.where(mask_vir, fwhm_nb,    fwhm_choice)
    amp_final  = jnp.where(mask_vir, amp_nb,     amp_choice)
    mu_final   = jnp.where(mask_vir, mu_nb,      mu_choice)

    return fwhm_final, amp_final, mu_final

