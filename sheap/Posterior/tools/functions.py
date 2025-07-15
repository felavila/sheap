import jax.numpy as np
from jax import vmap, random
import numpy as np 

from .constants import BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS



#I need a better name for this 
def calc_flux(norm_amplitude, fwhm):
    # norm_amplitude and fwhm are 1D arrays (shape: [N])
    return np.sqrt(2.0 * np.pi) * norm_amplitude * fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def calc_luminosity(distance, flux, center):
    # distance [scalar], flux [N], center [N]
    return 4.0 * np.pi * distance**2 * flux * center

def calc_fwhm_kms(fwhm, c, center):
    # c is scalar (speed of light), others 1D
    return (fwhm * c) / center

def calc_monochromatic_luminosity(distance, flux_at_wavelength, wavelength):
    # All 1D or scalar, returns [N]
    return wavelength * 4.0 * np.pi * distance**2 * flux_at_wavelength

def calc_bolometric_luminosity(monochromatic_lum, correction):
    return monochromatic_lum * correction

def calc_black_hole_mass(L_w, fwhm_kms, estimator):
    #print("a")
    # All are 1D, estimator is a dict with a, b, f
    a, b, f = estimator["a"], estimator["b"], estimator["f"]
    log_L = np.log10(L_w)
    #np.atleast_2d(np.log10(L_w)).T
    #print(log_L.shape)
    log_FWHM = np.log10(fwhm_kms) - 3  # FWHM to 10^3 km/s
    #print(log_FWHM.shape)
    log_M_BH = a + b * (log_L - 44.0) + 2 * log_FWHM
    return (10 ** log_M_BH) / f