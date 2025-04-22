from jax import jit
import jax.numpy as jnp
#import partial
#from util

from sheap.tools.others import kms_to_wl
from sheap.tools.interp_tools import _interp_jax
import jax.scipy as jsp
import os 
import numpy as np 
from sheap.fitting.utils import param_count


templates_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"suport_data","templates")

fe_template_OP_file = os.path.join(templates_path,'fe2_Op.dat')
fe_template_read = jnp.array(np.loadtxt(fe_template_OP_file,comments='#').transpose()) # y units?
fe_template_norm = fe_template_read #.at[1].divide(jnp.max(fe_template_read[1]))

fe_template_UV_file = os.path.join(templates_path,'fe2_UV02.dat')
fe_template_UV_read = jnp.array(np.loadtxt(fe_template_UV_file,comments='#').transpose()) # y units?
fe_template_UV_read_norm = fe_template_UV_read.at[1].divide(jnp.max(fe_template_UV_read[1]))

@param_count(3)
def fitFeOP(x, params):
    # Unpack parameters: log_FWHM_broad, shift_, and scale.
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
    x_fe_template_norm, y_fe_template_norm = fe_template_read

    # Define intrinsic sigma of the template (from its FWHM of 900, for example)
    sigmatemplate = 900.0 / 2.355

    # Ensure the model sigma is not smaller than the template sigma to avoid sqrt of negative.
    safe_sigma_model = jnp.maximum(sigma_model, sigmatemplate + 1e-6)
    delta_sigma = jnp.sqrt(safe_sigma_model ** 2 - sigmatemplate ** 2)

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

@param_count(3)
def fitFeUV(x,params):
    #TODO look for the sigma of template
    #log_FWHM_broad,shift_,scale_ = params
    log_FWHM_broad,shift_,scale = params
    #xmin_fe=2550
    #xmax_fe=3040
    central_wl = 2795
    FWHM_broad = 10**log_FWHM_broad 
    #scale = scale_
    shift = 10 * shift_
    #could be good option add to the masking part nan to x for x axis
    x = jnp.nan_to_num(x) ##
    x_fe_template_norm,y_fe_template_norm = fe_template_read
    sigmatemplate = 900.0 / 2.355 #sigma del spectro (?) This from where cames ?
    sigma_model = FWHM_broad / 2.355
    delta_sigma = jnp.sqrt(sigma_model ** 2 - sigmatemplate ** 2)
    x2800 = jnp.nanargmin(jnp.abs(x - central_wl)).astype(jnp.int32)
    dl = (x[1:]-x[-1])[x2800] # what happens when this is 0 xd
    sigma = kms_to_wl(delta_sigma, central_wl) / dl
    radius = jnp.round(4 * sigma).astype(jnp.int32)
    max_radius = 1000  # Given the jit compilation in the end this is always the radius then we mask to get what we want
    x_local = jnp.arange(-max_radius, max_radius + 1)
    mask = jnp.logical_and(x_local >= -radius, x_local <= radius)
    x_masked = x_local * mask
    kernel = jsp.stats.norm.pdf(x_masked, scale=sigma) * mask
    kernel /= jnp.sum(kernel)
    broad_template = jsp.signal.convolve(y_fe_template_norm,kernel,mode='same',method='fft')
    interpolated_broad_scaled_template = scale * _interp_jax(x,x_fe_template_norm+shift,broad_template, left= None, right=None)
    return interpolated_broad_scaled_template

#partial(jit, static_argnums=(1,))

# def small_fe_fit(X,params,central_wl=4650):
#     #https://github.com/scipy/scipy/blob/v1.14.1/scipy/ndimage/_filters.py#L286-L390
    
#     ########
#     FWHM_broad = 10**log_FWHM_broad 
#     shift = 10 * shift_
#     scale = 10 * scale_
#     ################
#     mag_corrected_agn,fe_template_norm = X
#     sigmatemplate = 900.0 / 2.355 #sigma del spectro (?) This from where cames ?
#     sigma_model = FWHM_broad / 2.355
#     delta_sigma = jnp.sqrt(sigma_model ** 2 - sigmatemplate ** 2)
#     x2800 = jnp.nanargmin(jnp.abs(mag_corrected_agn[0, :] - central_wl)).astype(jnp.int32)
#     dl = (mag_corrected_agn[0,1:]-mag_corrected_agn[0,:-1])[x2800] # what happens when this is 0 xd
#     sigma = kms_to_wl(delta_sigma, central_wl) / dl
#     #truncate = 4
#     radius = jnp.round(4 * sigma).astype(jnp.int32)
#     max_radius = 1000  # Given the jit compilation in the end this is always the radius then we mask to get what we want  
#     #x_range = 2 * max_radius + 1
#     x = jnp.arange(-max_radius, max_radius + 1)
#     #x = np.arange(-radius, radius+1)
#     # Adjust x to match the computed radius
#     mask = jnp.logical_and(x >= -radius, x <= radius)
#     x_masked = x * mask

#     # Compute the convolution kernel
#     kernel = jsp.stats.norm.pdf(x_masked, scale=sigma) * mask

#     # Normalize the kernel to avoid changing the signal's amplitude
#     kernel /= jnp.sum(kernel)
    
#     broad_template = jsp.signal.convolve(fe_template_norm[1, :],kernel,mode='same',method='fft')
#     interpolated_broad_scaled_template = scale * _interp_jax(mag_corrected_agn[0,:],fe_template_norm[0,:]+shift,broad_template, left= None, right=None)
#     return interpolated_broad_scaled_template