import jax.numpy as np
from jax import vmap, random
import numpy as np 

from .constants import BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS



import jax.numpy as np
from jax import vmap, random
import numpy as npx  # standard NumPy if needed elsewhere

from .constants import BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS

def calc_flux(norm_amplitude, fwhm):
    """
    Compute flux from normalized amplitude and FWHM assuming a Gaussian profile.
    F = A × FWHM × sqrt(π / (4 * ln(2)))
    """
    return np.sqrt(2.0 * np.pi) * norm_amplitude * fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def calc_luminosity(distance, flux):
    """
    Line luminosity: L = 4π D^2 × F × λ_center
    """
    return 4.0 * np.pi * distance**2 * flux #* center

def calc_fwhm_kms(fwhm, c, center):
    """
    Convert FWHM [Å] to velocity [km/s].
    """
    return (fwhm * c) / center

def calc_monochromatic_luminosity(distance, flux_at_wavelength, wavelength):
    """
    Monochromatic luminosity: Lλ × λ = νLν (erg/s)
    """
    return wavelength * 4.0 * np.pi * distance**2 * flux_at_wavelength

def calc_bolometric_luminosity(monochromatic_lum, correction):
    """
    Apply bolometric correction to monochromatic luminosity.
    """
    return monochromatic_lum * correction

def calc_black_hole_mass(L_w, fwhm_kms, estimator):
    """
    Single-epoch black hole mass:
    log M_BH = a + b (log L - 44) + 2 log(FWHM/1000)
    M_BH = 10**log M_BH / f
    """
    a, b, f = estimator["a"], estimator["b"], estimator["f"]
    log_L = np.log10(L_w)
    log_FWHM = np.log10(fwhm_kms) - 3  # FWHM in 1000 km/s
    log_M_BH = a + b * (log_L - 44.0) + 2.0 * log_FWHM
    return (10 ** log_M_BH) / f

def calc_black_hole_mass_gh2015(L_halpha, fwhm_kms):
    """
    Greene & Ho 2015 (Eq. 6):
    log(M_BH/M_sun) = 6.57 + 0.47 * (log L_Hα - 42) + 2.06 * log(FWHM / 1000)
    """
    log_L = np.log10(L_halpha)
    log_FWHM = np.log10(fwhm_kms) - 3
    log_M_BH = 6.57 + 0.47 * (log_L - 42.0) + 2.06 * log_FWHM
    return 10 ** log_M_BH
