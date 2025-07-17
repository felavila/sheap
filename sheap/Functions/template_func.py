from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path


import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from sheap.Functions.utils import param_count
from sheap.Tools.spectral_basic import kms_to_wl

#TEMPLATE FAMILIES SHOULD BE ADDED 

templates_path = Path(__file__).resolve().parent.parent / "SuportData" / "templates"
#print(templates_path)

fe_template_OP_file =   templates_path / 'fe2_Op.dat'

fe_template_OP = jnp.array(np.loadtxt(fe_template_OP_file, comments='#').transpose())  # y units?
fe_template_OP = fe_template_OP.at[1].divide(jnp.max(fe_template_OP[1]))
#fe_template_OP = fe_template_OP.at[]

fe_template_UV_file = templates_path / 'fe2_UV02.dat'

fe_template_UV = jnp.array(np.loadtxt(fe_template_UV_file, comments='#').transpose())
 
fe_template_UV = fe_template_UV.at[1].divide(jnp.max(fe_template_UV[1]))

class FeIITemplateModel:
    def __init__(self, template: Tuple[jnp.ndarray, jnp.ndarray], central_wl: float,sigmatemplate: float):
        self.wl, self.flux = template
        self.central_wl = central_wl #Ang
        self.sigmatemplate =sigmatemplate #km/s
    
    def __call__(self,
                 x: jnp.ndarray,
                 log_FWHM: float,
                 shift: float,
                 scale: float):
        # 1) Compute extra σ in km/s
        FWHM = 10 ** log_FWHM
        sigma_model = FWHM / 2.355
        delta_sigma = jnp.sqrt(jnp.maximum(sigma_model**2 - self.sigmatemplate**2, 1e-12))

        # 2) Convert that σ to Angstroms at central_wl
        sigma_lambda = kms_to_wl(delta_sigma, self.central_wl)  # Å

        # 3) Build the Gaussian transfer function in Fourier space
        dl = jnp.maximum(x[1] - x[0], 1e-6)        # Å per pixel
        n_pix = self.flux.shape[0]
        freq = jnp.fft.fftfreq(n_pix, d=dl)       # cycles per Å
        gauss_tf = jnp.exp(-2 * (jnp.pi * freq * sigma_lambda)**2)

        # 4) FFT‐convolve template
        flux_fft = jnp.fft.fft(self.flux)
        broadened = jnp.real(jnp.fft.ifft(flux_fft * gauss_tf))

        # 5) Apply additive shift (in Å), interpolate, and scale
        shifted_wl = self.wl + shift
        model = scale * jnp.interp(
            x,
            shifted_wl,
            broadened,
            left=0.0,
            right=0.0
        )
        return model
    # def __call__(self, x: jnp.ndarray, log_FWHM: float, shift_: float, scale: float):
    #     FWHM = 10 ** log_FWHM
    #     sigma_model = FWHM/ 2.355
    #     shift = shift_

    #     delta_sigma = jnp.sqrt(jnp.maximum(sigma_model**2 - self.sigmatemplate**2, 1e-12))
    #     dl = jnp.maximum(x[1] - x[0], 1e-6)
    #     sigma_wl = kms_to_wl(delta_sigma, self.central_wl) / dl
        
    #     max_radius = 1000
    #     x_local = jnp.arange(-max_radius, max_radius + 1)
    #     mask = jnp.where(jnp.abs(x_local) <= jnp.round(4 * sigma_wl), 1.0, 0.0)
    #     kernel = jsp.stats.norm.pdf(x_local, scale=sigma_wl) * mask
    #     kernel /= jnp.sum(kernel)

    #     broadened = jsp.signal.convolve(self.flux, kernel, mode='same', method='fft')
    #     shifted = self.wl + shift
    #     return scale * jnp.interp(x, shifted, broadened, left=None, right=None)

fitFeOP_model = FeIITemplateModel(template=fe_template_OP, central_wl=4650.0,sigmatemplate=900.0 / 2.355)
fitFeUV_model = FeIITemplateModel(template=fe_template_UV, central_wl=2795.0,sigmatemplate=900.0 / 2.355)

@param_count(3)
def fitFeOP(x, params): return fitFeOP_model(x, *params)

@param_count(3)
def fitFeUV(x, params): return fitFeUV_model(x, *params)




class FeIITemplateModelFixedDispersion:
    def __init__(self,
                 template: Tuple[jnp.ndarray, jnp.ndarray],
                 fixed_dispersion: float = 106.3,        # km/s per pixel
                 sigmatemplate: float = 900.0 / 2.355):  # template σ in km/s
        """
        template: (wl_array, flux_array) in Angstroms & arb. units
        fixed_dispersion: km/s per pixel (e.g. 106.3 for BG92 FeII)
        sigmatemplate: intrinsic σ of the template in km/s
        """
        self.wl, self.flux = template
        self.fixed_dispersion = fixed_dispersion
        self.sigmatemplate = sigmatemplate

    def __call__(self,
                 x: jnp.ndarray,
                 log_FWHM: float,
                 shift_frac: float,
                 scale: float):
        # --- 1) compute Δx (Angstrom per pixel) ---
        dx = x[1] - x[0]

        # --- 2) compute requested σ in pixel units ---
        FWHM = 10 ** log_FWHM
        sigma_model = FWHM / 2.355                    # km/s
        delta_sigma = jnp.sqrt(jnp.maximum(
            sigma_model**2 - self.sigmatemplate**2, 1e-12
        ))                                            # km/s
        sigma_pix = delta_sigma / self.fixed_dispersion

        # --- 3) convert that to Angstrom σ for the transfer function ---
        sigma_lambda = sigma_pix * dx                # Angstrom

        # --- 4) build Gaussian transfer function in Fourier space ---
        n_pix = self.flux.shape[0]
        freq = jnp.fft.fftfreq(n_pix, d=dx)           # cycles per Angstrom
        gauss_tf = jnp.exp(-2 * (jnp.pi * freq * sigma_lambda)**2)

        # --- 5) FFT‐convolution ---
        flux_fft = jnp.fft.fft(self.flux)
        broadened = jnp.real(jnp.fft.ifft(flux_fft * gauss_tf))

        # --- 6) apply fractional shift + linear interp + scaling ---
        shifted = self.wl + shift_frac
        model = scale * jnp.interp(x,
                                  shifted,
                                  broadened,
                                  left=0.0,
                                  right=0.0)
        return model


# Example usage:
# fitFeOP_model = FeIITemplateModelFixedDispersion(
#     template=fe_template_OP,
#     fixed_dispersion=106.3,
#     sigmatemplate=900.0/2.355
# )
#
# @param_count(3)
# def fitFeOP(x, params):
#     return fitFeOP_model(x, *params)


fe_template_OP_file =   templates_path / 'fe_optical.txt'
fe_op_qsofit = np.loadtxt(fe_template_OP_file, comments='#').transpose()

fe_op_qsofit[0,:] = 10**(fe_op_qsofit[0,:])
fe_op_qsofit[1,:] = fe_op_qsofit[1,:]/max(fe_op_qsofit[1,:])
fe_op_qsofit = jnp.array(fe_op_qsofit)

#fitFeOP_model = FeIITemplateModelFixedDispersion(template=fe_template_OP,fixed_dispersion=106.3,sigmatemplate=900.0/2.355)

# @param_count(3)
# def fitFeOP(x, params): return fitFeOP_model(x, *params)
# Example instantiation:

# And then wrap as before:
# @param_count(3)
# def fitFeOP(x, params): return fitFeOP_model(x, *params)