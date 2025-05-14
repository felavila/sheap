import jax.numpy as jnp
from jax import vmap,jit




def interpolate_flux_array(
    x_interp: jnp.ndarray,        # shape (N, M)
    template_wave: jnp.ndarray,         # shape (L,)
    template_flux: jnp.ndarray,         # shape (K, L)
    reshape_output: bool = True
) -> jnp.ndarray:
    """
    Interpolates multiple flux arrays over a given grid of wavelengths.

    Args:
        x_interp: 2D array of wavelengths to interpolate at, shape (N, M)
        gl_wave: 1D wavelength grid for the flux, shape (L,)
        gl_flux: 2D array of flux values, shape (K, L)
        reshape_output: Whether to reshape output to (K, N, M)

    Returns:
        Interpolated flux values:
            - shape (K, N, M) if reshape_output=True
            - shape (K, N*M) if reshape_output=False
    """
    x_flat = x_interp.reshape(-1)  # Flatten for vectorized interpolation

    def interp_one_flux(flux_row: jnp.ndarray) -> jnp.ndarray:
        return jnp.interp(x_flat, template_wave, flux_row)

    interp_vals = vmap(interp_one_flux)(template_flux)  # shape (K, N*M)

    if reshape_output:
        interp_vals = interp_vals.reshape(template_flux.shape[0], *x_interp.shape)

    return interp_vals

def normalize(vecs):
    norm = jnp.sqrt(jnp.sum(vecs**2, axis=-1, keepdims=True))
    return vecs / jnp.where(norm == 0, 1, norm)

@jit
def linear_combination(eieigenvectors,params):
    return jnp.nansum(eieigenvectors.T*params,axis=1)


def make_penalty_func(func, n_galaxies):
    """
    Returns a penalty function that penalizes negative flux in the model
    reconstructed from the first `n_galaxies` components.
    
    Args:
        func: Function taking (eigenvectors, params) → model
        n_galaxies: Number of galaxy components in params
    
    Returns:
        A function (eigenvectors, params) → penalty value
    """
    @jit
    def penalty_func(eigenvectors, params):
        qso_model = func(eigenvectors[n_galaxies:], params[n_galaxies:])
        galaxy_model = func(eigenvectors[:n_galaxies], params[:n_galaxies])
        penalty =  jnp.sum(galaxy_model < 0)
        penalty += jnp.sum(qso_model < 0)
        return penalty

    return penalty_func