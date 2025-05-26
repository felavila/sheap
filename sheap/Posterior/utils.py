import jax.numpy as jnp
from jax import vmap,grad


from sheap.Mappers.LineMapper import LineSelectionResult

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


def compute_integrated_profiles(LineSelectionResult:LineSelectionResult, delta=1000, n_points=1000):
    """
    Compute integrated flux and propagated error for each broad component
    in a LineSelectionResult, using preallocated JAX arrays.

    Parameters
    ----------
    LineSelectionResult : LineSelectionResult
        An instance of LineSelectionResult filtered with kind="broad".
    delta : float
        Half-width of integration window around the center.
    n_points : int
        Number of integration points.

    Returns
    -------
    values_ : jnp.ndarray, shape (n_spectra, n_lines)
        Integrated fluxes.
    errors_ : jnp.ndarray, shape (n_spectra, n_lines)
        Propagated uncertainties.
    """
    original_centers = LineSelectionResult.original_centers
    n_lines = len(original_centers)
    n_spectra = LineSelectionResult.params.shape[0]

    values_ = jnp.zeros((n_spectra,n_lines))
    errors_ = jnp.zeros((n_spectra,n_lines))
    pos_idx = 0
    for i in range(len(original_centers)):
        L1=  LineSelectionResult.profile_params_index_list[i]
        f1 = LineSelectionResult.profile_functions[i]
        f1_params = LineSelectionResult.params[:,pos_idx:pos_idx+len(L1)]
        f1_uncertainty_params = LineSelectionResult.uncertainty_params[:,pos_idx:pos_idx+len(L1)]
        #print(pos_idx,LineSelectionResult.params_names[pos_idx:pos_idx+len(L1)])
        idx = next((i for i, name in enumerate(LineSelectionResult.params_names[pos_idx:pos_idx+len(L1)]) if "center" in name), None)
        if idx is not None:
            centers = f1_params[:, idx]
            x = jnp.stack([jnp.linspace(c - delta, c + delta, n_points)for c in centers])
        else:
            idx = next((i for i, name in enumerate(LineSelectionResult.params_names[pos_idx:pos_idx+len(L1)]) if "shift" in name), None)
            #if it is shift a it is ok because the thing what we wan it is stimate a "realistic" x 
            #f1_params = jnp.array(f1_params).at[:, idx].set(original_centers[i] - f1_params[:, idx])
            x = jnp.stack([jnp.linspace(c - delta, c + delta, n_points)for c in centers])
        vmapped_func = vmap(integrate_function_error_single, in_axes=(None, 0, 0, 0))
        values,errors = vmapped_func(f1,x,f1_params,f1_uncertainty_params)
        values_ = values_.at[:,i].set(values)
        errors_ = errors_.at[:,i].set(errors)
        pos_idx += len(L1)
    return values_,errors_


def effective_fwhm(params1, params2):
    """
    Estimate the FWHM of a combined two-Gaussian profile using moment analysis.
    Each param is a tuple/list/array of (amp, mu, sigma).
    """
    amp1, mu1, sigma1 = params1
    amp2, mu2, sigma2 = params2

    total_amp = amp1 + amp2
    mu_eff = (amp1 * mu1 + amp2 * mu2) / total_amp
    var_eff = (
        amp1 * (sigma1**2 + (mu1 - mu_eff) ** 2) + amp2 * (sigma2**2 + (mu2 - mu_eff) ** 2)
    ) / total_amp

    sigma_eff = jnp.sqrt(var_eff)
    fwhm = 2.35482 * sigma_eff
    return total_amp, mu_eff, sigma_eff, fwhm / mu_eff  # dimensionless


batched_fwhm = vmap(
    vmap(effective_fwhm, in_axes=(0, 0)), in_axes=(0, 0)  # over regions  # over samples
)


def combine(params_broad, params_narrow, limit_velocity=150.0, c=299792.0):
    """
    Combines multiple broad Gaussians with a single narrow Gaussian per object.

    params_broad: (N, 3 * n_broad) -> multiple Gaussians: amp1, mu1, sigma1, ...
    params_narrow: (N, 3) -> single narrow Gaussian: amp, mu, sigma
    """
    N = params_broad.shape[0]
    n_broad = params_broad.shape[1] // 3

    # Reshape to (N, n_broad, 3)
    broad_reshaped = params_broad.reshape(N, n_broad, 3)
    amp_broad = broad_reshaped[:, :, 0]
    mu_broad = broad_reshaped[:, :, 1]
    sigma_broad = broad_reshaped[:, :, 2]

    # Compute all effective FWHMs from every pair of broad components
    idx1, idx2 = jnp.triu_indices(n_broad, k=1)
    pairs1 = broad_reshaped[:, idx1, :]  # (N, n_pairs, 3)
    pairs2 = broad_reshaped[:, idx2, :]  # (N, n_pairs, 3)

    amp_eff, mu_eff, sigma_eff, fwhm_eff_ratio = batched_fwhm(pairs1, pairs2)

    # Reference: center of narrow component
    mu_narrow = params_narrow[:, 1:2]  # shape (N, 1)

    # Compute relative velocity for each broad component to the narrow
    delta_mu = (mu_broad - mu_narrow) / mu_narrow  # (N, n_broad)
    relative_velocity = jnp.abs(jnp.diff(delta_mu, axis=1).squeeze()) * c  # (N,)
    mask_virial = relative_velocity >= limit_velocity

    # Choose broad component closest in velocity to narrow
    idx_closest = jnp.argmin(jnp.abs(delta_mu), axis=1)
    fwhm_all = sigma_broad * 2.35482 / mu_broad  # shape (N, n_broad)

    fwhm_nonvirial = jnp.take_along_axis(fwhm_all, idx_closest[:, None], axis=1).squeeze()
    amp_nonvirial = jnp.take_along_axis(amp_broad, idx_closest[:, None], axis=1).squeeze()
    mu_nonvirial = jnp.take_along_axis(mu_broad, idx_closest[:, None], axis=1).squeeze()
    sigma_nonvirial = jnp.take_along_axis(sigma_broad, idx_closest[:, None], axis=1).squeeze()

    # Amplitude ratio test
    amp_ratio = jnp.min(amp_broad, axis=1) / jnp.max(amp_broad, axis=1)
    mask_amp = amp_ratio > 0.1
    idx_max_amp = jnp.argmax(amp_broad, axis=1)

    fwhm_amp = jnp.take_along_axis(fwhm_all, idx_max_amp[:, None], axis=1).squeeze()
    amp_amp = jnp.take_along_axis(amp_broad, idx_max_amp[:, None], axis=1).squeeze()
    mu_amp = jnp.take_along_axis(mu_broad, idx_max_amp[:, None], axis=1).squeeze()
    sigma_amp = jnp.take_along_axis(sigma_broad, idx_max_amp[:, None], axis=1).squeeze()

    # Choose amplitude-based or moment-based profile
    fwhm_masked = jnp.where(mask_amp, fwhm_eff_ratio[:, 0], fwhm_amp)
    amp_masked = jnp.where(mask_amp, amp_eff[:, 0], amp_amp)
    mu_masked = jnp.where(mask_amp, mu_eff[:, 0], mu_amp)
    sigma_masked = jnp.where(mask_amp, sigma_eff[:, 0], sigma_amp)

    # Final values depending on virial mask
    fwhm_final = jnp.where(mask_virial, fwhm_nonvirial, fwhm_masked) * c
    amp_final = jnp.where(mask_virial, amp_nonvirial, amp_masked)
    mu_final = jnp.where(mask_virial, mu_nonvirial, mu_masked)
    sigma_final = jnp.where(mask_virial, sigma_nonvirial, sigma_masked)

    return fwhm_final, amp_final, mu_final, sigma_final
