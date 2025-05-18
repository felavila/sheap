import os

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jax import jit

from sheap.FunctionsMinimize.utils import param_count

# from sheap.tools.interp_tools import _interp_jax
from sheap.Tools.others import kms_to_wl

# import partial
# from util


templates_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "suport_data", "templates"
)

fe_template_OP_file = os.path.join(templates_path, 'fe2_Op.dat')
fe_template_OP = jnp.array(
    np.loadtxt(fe_template_OP_file, comments='#').transpose()
)  # y units?

fe_template_UV_file = os.path.join(templates_path, 'fe2_UV02.dat')
fe_template_UV = jnp.array(
    np.loadtxt(fe_template_UV_file, comments='#').transpose()
)  # y units?


@param_count(3)
def fitFeOP(x, params):
    "Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"
    log_FWHM_broad, shift_, scale = params
    central_wl = 4650.0  # Reference wavelength

    # Compute FWHM and convert to sigma (Gaussian dispersion)
    FWHM_broad = 10**log_FWHM_broad
    sigma_model = FWHM_broad / 2.355

    # Apply scaling to the shift parameter.
    shift = 10 * shift_

    # Replace NaNs in the wavelength array to ensure stability
    x = jnp.nan_to_num(x)

    # Read the normalized Fe emission template (assumes fe_template_read is loaded globally)
    x_fe_template_norm, y_fe_template_norm = fe_template_OP

    # Define intrinsic sigma of the template (from its FWHM of 900, for example)
    sigmatemplate = 900.0 / 2.355

    # Ensure the model sigma is not smaller than the template sigma to avoid sqrt of negative.
    safe_sigma_model = jnp.maximum(sigma_model, sigmatemplate + 1e-6)
    delta_sigma = jnp.sqrt(safe_sigma_model**2 - sigmatemplate**2)

    # Instead of a complex index, assume uniform wavelength array.
    # Use the first two elements to compute the step size.
    dl = x[1] - x[0]
    # Safety: if dl somehow is zero, assign a minimum value.
    dl = jnp.where(dl == 0, 1e-6, dl)

    # Convert the additional broadening from km/s to wavelength units.
    sigma_wl = kms_to_wl(delta_sigma, central_wl) / dl

    # Define the convolution kernel radius; ensure that it is at least 1.
    radius = jnp.maximum(jnp.round(4 * sigma_wl).astype(jnp.int32), 1)

    # Create a local grid for the convolution kernel.
    max_radius = 1000  # Over-dimension the grid for JIT compilation compatibility.
    x_local = jnp.arange(-max_radius, max_radius + 1)

    # Create a mask with a robust approach: only values within [-radius, radius] are nonzero.
    mask = jnp.where(jnp.abs(x_local) <= radius, 1.0, 0.0)
    # Compute the Gaussian kernel on the entire grid.
    kernel = jsp.stats.norm.pdf(x_local, scale=sigma_wl) * mask
    kernel = kernel / jnp.sum(kernel)

    # Convolve the normalized Fe template with the Gaussian kernel using FFT.
    broad_template = jsp.signal.convolve(y_fe_template_norm, kernel, mode='same', method='fft')

    # Shift the template's wavelength grid.
    shifted_wl = x_fe_template_norm + shift

    # Interpolate the convolved (broadened) template back onto the input wavelength grid.
    # Make sure _interp_jax is differentiable for gradient-based optimization.
    interpolated_broad_scaled_template = scale * jnp.interp(
        x, shifted_wl, broad_template, left=None, right=None
    )

    return interpolated_broad_scaled_template


# 2795
@param_count(3)
# 2795
def fitFeUV(x, params):
    "Fit the UV FeII component on the continuum from 1200 to 3500 A based on Boroson & Green 1992."
    log_FWHM_broad, shift_, scale = params
    central_wl = 2795  # Reference wavelength

    # Compute FWHM and convert to sigma (Gaussian dispersion)
    FWHM_broad = 10**log_FWHM_broad
    sigma_model = FWHM_broad / 2.355

    # Apply scaling to the shift parameter.
    shift = 10 * shift_

    # Replace NaNs in the wavelength array to ensure stability
    x = jnp.nan_to_num(x)

    # Read the normalized Fe emission template (assumes fe_template_read is loaded globally)
    x_fe_template_norm, y_fe_template_norm = fe_template_UV

    # Define intrinsic sigma of the template (from its FWHM of 900, for example)
    sigmatemplate = 900.0 / 2.355

    # Ensure the model sigma is not smaller than the template sigma to avoid sqrt of negative.
    safe_sigma_model = jnp.maximum(sigma_model, sigmatemplate + 1e-6)
    delta_sigma = jnp.sqrt(safe_sigma_model**2 - sigmatemplate**2)

    # Instead of a complex index, assume uniform wavelength array.
    # Use the first two elements to compute the step size.
    dl = x[1] - x[0]
    # Safety: if dl somehow is zero, assign a minimum value.
    dl = jnp.where(dl == 0, 1e-6, dl)

    # Convert the additional broadening from km/s to wavelength units.
    sigma_wl = kms_to_wl(delta_sigma, central_wl) / dl

    # Define the convolution kernel radius; ensure that it is at least 1.
    radius = jnp.maximum(jnp.round(4 * sigma_wl).astype(jnp.int32), 1)

    # Create a local grid for the convolution kernel.
    max_radius = 1000  # Over-dimension the grid for JIT compilation compatibility.
    x_local = jnp.arange(-max_radius, max_radius + 1)

    # Create a mask with a robust approach: only values within [-radius, radius] are nonzero.
    mask = jnp.where(jnp.abs(x_local) <= radius, 1.0, 0.0)
    # Compute the Gaussian kernel on the entire grid.
    kernel = jsp.stats.norm.pdf(x_local, scale=sigma_wl) * mask
    kernel = kernel / jnp.sum(kernel)

    # Convolve the normalized Fe template with the Gaussian kernel using FFT.
    broad_template = jsp.signal.convolve(y_fe_template_norm, kernel, mode='same', method='fft')

    # Shift the template's wavelength grid.
    shifted_wl = x_fe_template_norm + shift

    # Interpolate the convolved (broadened) template back onto the input wavelength grid.
    interpolated_broad_scaled_template = scale * jnp.interp(
        x, shifted_wl, broad_template, left=None, right=None
    )

    return interpolated_broad_scaled_template
