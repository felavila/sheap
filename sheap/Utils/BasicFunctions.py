"""
Basic Functions
===============

This module provides core physical conversions frequently used in 
spectral analysis, including wavelength ↔ velocity conversions and 
vacuum ↔ air wavelength corrections.

Functions
---------
kms_to_wl(kms, line_center, C_KMS=DEFAULT_DEFAULT_C_KMS)
    Convert velocity in km/s to a wavelength shift at a given line center.

wl_to_kms(wl, line_center, C_KMS=DEFAULT_DEFAULT_C_KMS)
    Convert wavelength shift to velocity in km/s at a given line center.

vac_to_air(lam_vac)
    Convert vacuum wavelengths to air wavelengths using the IAU standard.
"""

__author__ = 'felavila'

__all__ = [
    "kms_to_wl",
    "vac_to_air",
    "wl_to_kms",
    "air_to_vacuum"]


from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from uncertainties import unumpy as unp

from sheap.Utils.Constants  import DEFAULT_C_KMS

def log10(x):
    """
    Compute base-10 logarithm for both numpy arrays and
    uncertainties.unumpy arrays, replacing non-positive values with NaN.

    Parameters
    ----------
    x : array-like or unumpy.uarray
        Input values.

    Returns
    -------
    result : array-like
        log10(x), with non-positive values replaced by NaN.
        Uses np.log10 for pure numpy objects, or unp.log10 if x has uncertainties.
    """
   
    if isinstance(x, np.ndarray) and x.dtype == object and x.size:
        # convert to unumpy array if not already
        vals = unp.nominal_values(x)
        safe = unp.uarray(np.where(vals > 0, vals, np.nan),
                             unp.std_devs(x))
        return unp.log10(safe)

    
    x = np.asarray(x, dtype=float)
    safe = np.where(x > 0, x, np.nan)
    return np.log10(safe)

def kms_to_wl(kms, line_center, C_KMS=DEFAULT_C_KMS):
    """
    Convert a velocity in km/s to a wavelength shift based on the line center.

    Parameters:
    -----------
    kms : float or array-like
        The velocity value(s) in kilometers per second.
    line_center : float
        The central (reference) wavelength of the spectral line.
    C_KMS : float, optional
        The speed of light in km/s. The default value is 2.99792458e5 km/s.

    Returns:
    --------
    wl : float or array-like
        The calculated wavelength shift corresponding to the input velocity.
    """
    wl = kms * line_center / C_KMS
    return wl


def wl_to_kms(wl, line_center, C_KMS=DEFAULT_C_KMS):
    """
    Convert a velocity in km/s to a wavelength shift based on the line center.

    Parameters:
    -----------
    wl : float or array-like
        The calculated wavelength shift corresponding to the input velocity.

    line_center : float
        The central (reference) wavelength of the spectral line.
    C_KMS : float, optional
        The speed of light in km/s. The default value is 2.99792458e5 km/s.

    Returns:
    --------
    kms : float or array-like
        The velocity value(s) in kilometers per second.
    """
    kms = (wl * C_KMS) / line_center
    return kms


def vac_to_air(lam_vac):
    """
    Convert wavelength in vacuum to air.
        
        Parameters
        ----------
        lam_vac : float or array-like
            Wavelength in vacuum, in Angstroms.
        
        Returns
        -------
        lam_air : float or array-like
            Wavelength in air, in Angstroms.
        
        Valid roughly for 2000-20000 Å.
    """
    lam = np.asarray(lam_vac)
    sigma2 = (1e4 / lam) ** 2
    fact = 1 + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)

    return lam_vac / fact

def air_to_vacuum(lambda_air):
    """
    Convert wavelength in air to wavelength in vacuum.
    
    Parameters
    ----------
    lambda_air : float or array-like
        Wavelength in air, in Angstroms.
    
    Returns
    -------
    lambda_vac : float or array-like
        Wavelength in vacuum, in Angstroms.
    
    Valid roughly for 2000-20000 Å.
    """
    l2 = lambda_air ** 2
    n = (1
         + 2.735182e-4
         + 131.4182 / l2
         + 2.76249e8 / l2**2)
    return lambda_air * n



def calc_flux(norm_amplitude, fwhm):
    r"""
    Compute the integrated flux of a Gaussian line profile.

    .. math::
        F = A \cdot \mathrm{FWHM} \cdot \sqrt{ \frac{\pi}{4 \ln 2} }

    Parameters
    ----------
    norm_amplitude : array-like
        Normalized amplitude of the Gaussian peak.
    fwhm : array-like
        Full width at half maximum in wavelength units.

    Returns
    -------
    flux : jnp.ndarray
        Integrated line flux.
    """
    return np.sqrt(2.0 * np.pi) * norm_amplitude * fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def calc_fwhm_kms(fwhm, C_KMS, center):
    r"""
    Convert FWHM in Å to velocity width in km/s.

    .. math::
        v = \frac{\mathrm{FWHM}}{\lambda_0} \, C_KMS

    Parameters
    ----------
    fwhm : float or array
        Full width at half maximum in Å.
    C_KMS : float
        Speed of light in km/s.
    center : float or array
        Line center wavelength in Å.

    Returns
    -------
    v_kms : jnp.ndarray
        Velocity width in km/s.
    """
    return (fwhm * C_KMS) / center


def calc_luminosity(distance, flux):
    r"""
    Compute line luminosity from flux and luminosity distance.

    .. math::
        L = 4 \pi D^2 \, F

    Parameters
    ----------
    distance : float or array
        Luminosity distance in cm.
    flux : float or array
        Integrated line flux.

    Returns
    -------
    luminosity : jnp.ndarray
        Line luminosity in erg/s.
    """
    return 4.0 * np.pi * distance**2 * flux #* center

def calc_monochromatic_luminosity(distance, flux_at_wavelength, wavelength):
    r"""
    Compute monochromatic luminosity at a given wavelength.

    .. math::
        L_\lambda \cdot \lambda = \nu L_\nu = \lambda \, 4 \pi D^2 \, F_\lambda

    Parameters
    ----------
    distance : float or array
        Luminosity distance in cm.
    flux_at_wavelength : float or array
        Flux density at the wavelength (erg/s/cm^2/Å).
    wavelength : float
        Wavelength in Å.

    Returns
    -------
    L_lambda : jnp.ndarray
        Monochromatic luminosity in erg/s.
    """
    return wavelength * 4.0 * np.pi * distance**2 * flux_at_wavelength

def calc_bolometric_luminosity(monochromatic_lum, correction):
    r"""
    Apply a bolometric correction to a monochromatic luminosity.

    .. math::
        L_{\mathrm{bol}} = L_\lambda \cdot C

    Parameters
    ----------
    monochromatic_lum : float or array
        Monochromatic luminosity in erg/s.
    correction : float
        Bolometric correction factor.

    Returns
    -------
    L_bol : jnp.ndarray
        Bolometric luminosity in erg/s.
    """
    return monochromatic_lum * correction

def calc_black_hole_mass(L_w, fwhm_kms, estimator):
    r"""
    Single-epoch BH mass estimator (continuum-based).

    .. math::
        \log M_{\rm BH} =
        a + b \, (\log L - 44) + 2 \, \log \left(\frac{\mathrm{FWHM}}{1000}\right)

    .. math::
        M_{\rm BH} = \frac{10^{\log M_{\rm BH}}}{f}

    Parameters
    ----------
    L_w : float or array
        Monochromatic luminosity (erg/s).
    fwhm_kms : float or array
        Line width in km/s.
    estimator : dict
        Coefficients with keys ``a``, ``b``, ``f``.

    Returns
    -------
    MBH : jnp.ndarray
        Black hole mass in solar masses.
    """
    a, b, f = estimator["a"], estimator["b"], estimator["f"]
    log_L = log10(L_w)
    log_FWHM = log10(fwhm_kms) - 3  # FWHM in 1000 km/s
    log_M_BH = a + b * (log_L - 44.0) + 2.0 * log_FWHM
    return (10 ** log_M_BH) / f

def calc_black_hole_mass_gh2015(L_halpha, fwhm_kms):
    r"""
    Greene & Ho (2015) Hα mass estimator (Eq. 6).

    .. math::
        \log \left(\frac{M_{\rm BH}}{M_\odot}\right) =
        6.57 + 0.47 \, (\log L_{H\alpha} - 42)
        + 2.06 \, \log \left(\frac{\mathrm{FWHM}}{1000}\right)

    Parameters
    ----------
    L_halpha : float or array
        Hα line luminosity in erg/s.
    fwhm_kms : float or array
        FWHM in km/s.

    Returns
    -------
    MBH : jnp.ndarray
        Black hole mass in solar masses.
    """
    log_L = log10(L_halpha)
    log_FWHM = log10(fwhm_kms) - 3
    log_M_BH = 6.57 + 0.47 * (log_L - 42.0) + 2.06 * log_FWHM
    return 10 ** log_M_BH