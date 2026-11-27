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
    n = (1+ 2.735182e-4 + 131.4182 / l2 + 2.76249e8 / l2**2)
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


def calc_black_hole_mass(L_in, vwidth_kms, estimator, extras=None):
    r"""
    Unified single-epoch (SE) black-hole mass estimator.

    This function keeps the classical behavior of the SE mass formula while providing
    clear documentation for continuum-based and line-based calibrations. It also supports
    optional shape terms and Fe II strength corrections.

    Parameters
    ----------
    L_in : array-like or float
        Luminosity used by the calibration:
        - For ``kind="continuum"``: monochromatic luminosity :math:`L_\lambda \cdot \lambda`
        (erg s\ :sup:`-1`).
        - For ``kind="line"``: line luminosity :math:`L_\text{line}` (erg s\ :sup:`-1`).
    vwidth_kms : array-like or float
        Velocity width in km/s. Defaults to FWHM, but you can set
        ``width_def="sigma"`` in ``estimator`` to use :math:`\sigma`.
    estimator : dict
        Calibration dictionary. Required keys:

        - ``kind``: "continuum" or "line"
        - ``a``: intercept term (dimensionless)
        - ``b``: luminosity slope
        - ``f``: virial factor (applied multiplicatively to the mass)
        - ``fwhm_factor`` (alias ``vel_exp``): velocity-width exponent (default 2.0)
        - ``pivots``: dict with reference values (e.g., ``{"L": 1e44, "FWHM": 1e3}`` for continuum
        or ``{"L": 1e42, "FWHM": 1e3}`` for line)

        Optional:
        - ``width_def``: "fwhm" (default) or "sigma"
        - ``extras``: nested dict with optional switches:
            * ``le20_shape``: If True and ``width_def="fwhm"``, adds a shape term using
            :math:`\sigma`.
            * ``pan25_gamma``: Slope for Fe II strength correction (default :math:`-0.34`).

    extras : dict, optional
        Runtime extras for optional terms:
        - ``sigma_kms``: second velocity measure (km/s) for the Le20-like shape term.
        - ``R_Fe``: Fe II strength (e.g., :math:`R_\mathrm{FeII}`).

    Returns
    -------
    numpy-like
        :math:`M_\mathrm{BH}` in solar masses (:math:`M_\odot`), with the virial factor ``f``
        already applied.

    Notes
    -----
    Base (log) mass relation, valid for both continuum- and line-based inputs:

    .. math::
    \log_{10} M_\mathrm{BH} = 
    \log_{10} f + a + b \left[ \log_{10} L - \log_{10} L_0 \right]
    + \beta \left[ \log_{10} V - \log_{10} V_0 \right] \;,

    where:

    - :math:`L` is :math:`L_\lambda \cdot \lambda` (continuum) or :math:`L_\text{line}` (line),
    - :math:`V` is the velocity width (FWHM or :math:`\sigma`) in km/s,
    - :math:`L_0` and :math:`V_0` are the pivot luminosity and velocity from ``pivots``,
    - :math:`\beta` is ``fwhm_factor`` (or ``vel_exp``), by default 2.0,
    - :math:`f` is the virial factor.

    If ``width_def="fwhm"`` and ``extras["le20_shape"]`` is True (Leighly+20-like term),
    and a second velocity measure :math:`\sigma` is provided via ``extras["sigma_kms"]``:

    .. math::
    \Delta \log_{10} M_\mathrm{BH} =
    -1.14 \left[ \log_{10}(\mathrm{FWHM}) - \log_{10}(\sigma) \right] + 0.33 \;.

    If ``extras["R_Fe"]`` is provided, a Panessa+25-like correction is added:

    .. math::
    \Delta \log_{10} M_\mathrm{BH} = \gamma \, R_\mathrm{Fe} \;,

    with :math:`\gamma =` ``estimator["extras"]["pan25_gamma"]`` (default :math:`-0.34`).

    Examples
    --------
    Classical continuum-based recipe with FWHM:

    >>> est = {
    ...     "kind": "continuum",
    ...     "a": 0.0, "b": 0.5, "f": 1.0,
    ...     "fwhm_factor": 2.0,
    ...     "pivots": {"L": 1e44, "FWHM": 1e3},
    ...     "width_def": "fwhm",
    ... }
    >>> MBH = calc_black_hole_mass(L_5100, FWHM_kms, est)

    Same but with the Le20 shape term:

    >>> extras = {"sigma_kms": sigma_kms}
    >>> est["extras"] = {"le20_shape": True}
    >>> MBH = calc_black_hole_mass(L_5100, FWHM_kms, est, extras=extras)

    Line-based calibration:

    >>> est_line = {
    ...     "kind": "line",
    ...     "a": 6.57, "b": 0.47, "f": 1.0,
    ...     "fwhm_factor": 2.06,
    ...     "pivots": {"L": 1e42, "FWHM": 1e3},
    ...     "width_def": "fwhm",
    ... }
    >>> MBH = calc_black_hole_mass(L_Halpha, FWHM_kms, est_line)
    """

    if extras is None:
        extras = {}

    kind = str(estimator.get("kind", "continuum")).lower()
    width_def = str(estimator.get("width_def", "fwhm")).lower()

    piv = estimator.get("pivots", {})
    L0 = float(piv.get("L", 1e42 if kind == "line" else 1e44))
    V0 = float(piv.get("FWHM", 1e3))
    #print(V0,type(V0))
    a = estimator["a"]
    b = estimator["b"]
    beta = estimator.get("fwhm_factor", estimator.get("vel_exp", 2.0))
    f = estimator.get("f", 1.0)
    
    L = L_in
    V = vwidth_kms
    #print(type(f),type(L),type(L0),type(beta),type(V),type(V0))
    #logM = log10(f) + a + b * (log10(L) - log10(L0)) + beta * (log10(V) - log10(V0))
    logM = log10(f) + a  + b    * (log10(L) - log10(L0)) + beta * (log10(V) - log10(V0))
    # Le20 shape (only if baseline uses FWHM)
    if width_def == "fwhm" and estimator.get("extras", {}).get("le20_shape", False):
        sigma = extras.get("sigma_kms", None)
        if sigma is not None:
            sigma = sigma
            logM += (-1.14 * (log10(V) - log10(sigma)) + 0.33)

    # Pan25 iron term
    if "R_Fe" in extras:
        gamma = estimator.get("extras", {}).get("pan25_gamma", -0.21)#-0.34)
        RFe = extras["R_Fe"]
        
        #logM += gamma * RFe  # broadcasts across components
    return (10.0 ** logM)