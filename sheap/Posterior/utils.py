import jax.numpy as jnp
from jax import vmap


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
