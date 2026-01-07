"""
Physical Functions for AGN Spectral Analysis
============================================

This module provides helper functions to compute derived physical
quantities from fitted line profiles, such as flux, luminosity,
velocity widths, bolometric luminosities, and single-epoch black hole
masses.

Main Features
-------------
- Flux and luminosity calculations from profile amplitudes and widths.
- Conversions between FWHM (Å) and velocity (km/s).
- Monochromatic and bolometric luminosities.
- Multiple single-epoch BH mass estimators (continuum- and line-based).
- Helpers to compute derived parameters (Lbol, Ledd, mdot) from fitted spectra.

Notes
-----
- Unless stated otherwise, luminosities are in erg/s,
  distances are in cm, and velocities in km/s.
"""

__author__ = 'felavila'

__all__ = [
    "calc_black_hole_mass",
    "calc_black_hole_mass_gh2015",
    "calc_bolometric_luminosity",
    "calc_flux",
    "calc_fwhm_kms",
    "calc_luminosity",
    "calc_monochromatic_luminosity",
    "extra_params_functions",
]

import jax.numpy as np
import numpy as np 
from uncertainties import unumpy as unp


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

def calc_fwhm_kms(fwhm, c, center):
    r"""
    Convert FWHM in Å to velocity width in km/s.

    .. math::
        v = \frac{\mathrm{FWHM}}{\lambda_0} \, c

    Parameters
    ----------
    fwhm : float or array
        Full width at half maximum in Å.
    c : float
        Speed of light in km/s.
    center : float or array
        Line center wavelength in Å.

    Returns
    -------
    v_kms : jnp.ndarray
        Velocity width in km/s.
    """
    return (fwhm * c) / center


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

def _col(x):
    """
    Ensure input is a 2D column vector repetead.
    TODO move it .
    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    array-like
        If input is 1D, reshaped to (N, 1).
    """
#
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x
    

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
    
    L = _col(L_in)
    V = _col(vwidth_kms)
    #print(type(f),type(L),type(L0),type(beta),type(V),type(V0))
    #logM = log10(f) + a + b * (log10(L) - log10(L0)) + beta * (log10(V) - log10(V0))
    logM = log10(f) + a  + b    * (log10(L) - log10(L0)) + beta * (log10(V) - log10(V0))
    # Le20 shape (only if baseline uses FWHM)
    if width_def == "fwhm" and estimator.get("extras", {}).get("le20_shape", False):
        sigma = extras.get("sigma_kms", None)
        if sigma is not None:
            sigma = _col(sigma)
            logM += (-1.14 * (log10(V) - log10(sigma)) + 0.33)

    # Pan25 iron term
    if "R_Fe" in extras:
        gamma = estimator.get("extras", {}).get("pan25_gamma", -0.21)#-0.34)
        RFe = _col(extras["R_Fe"])
        
        #logM += gamma * RFe  # broadcasts across components
    return (10.0 ** logM)


def extra_params_functions(broad_params, L_w, L_bol, estimators, c):
    r"""
    Compute derived parameters (BH masses, Eddington ratios, accretion rates).

    This routine applies single-epoch (SE) virial estimators to broad-line
    measurements, combining continuum or line luminosities with velocity widths
    to derive black hole masses and accretion-related quantities.

    Parameters
    ----------
    broad_params : dict
        Dictionary of broad-line properties (e.g., ``fwhm_kms``, ``luminosity``).
    L_w : dict
        Monochromatic luminosities keyed by wavelength.
    L_bol : dict
        Bolometric luminosities keyed by wavelength.
    estimators : dict
        Single-epoch estimators for both continuum and line calibrations.
    c : float
        Speed of light in km/s.
    extras : dict, optional
        Extra quantities for corrections (e.g., ``sigma_kms``, ``R_Fe``).

    Returns
    -------
    dict
        Nested dictionary of derived parameters per line and calibration.

    Notes
    -----
    The general single-epoch black hole mass relation is:

    .. math::
    \log M_\mathrm{BH} =
    a
    + b \cdot (\log L - \log L_0)
    + \beta \cdot (\log V - \log V_0)
    + \log f \;,

    where:

    - :math:`L` is either a monochromatic continuum luminosity or a line luminosity
    - :math:`V` is the velocity width (FWHM or :math:`\sigma`)
    - :math:`(a, b, \beta, f)` are the calibration parameters
    - :math:`L_0, V_0` are the pivot values from the calibration

    Special cases
    -------------

    **Continuum-based estimators**  
    Use monochromatic luminosities :math:`L_\lambda` at a given wavelength
    with a bolometric correction:

    .. math::
    L_\mathrm{bol} = BC_\lambda \cdot (\lambda L_\lambda)

    From this, the Eddington ratio and accretion rate are derived:

    .. math::
    L_\mathrm{Edd} = 1.26 \times 10^{38} \; 
    \left( \frac{M_\mathrm{BH}}{M_\odot} \right)
    \; [\mathrm{erg\,s^{-1}}]

    .. math::
    \dot{M} = \frac{L_\mathrm{bol}}{\eta \, c^2}

    with :math:`\eta = 0.1` by default.

    **Line-based estimators**  
    Use the integrated line luminosity:

    .. math::
    \log M_\mathrm{BH} =
    a + b \cdot (\log L_\mathrm{line} - \log L_0)
    + \beta \cdot (\log V - \log V_0)

    Corrections supported
    ---------------------

    - **Le20 shape term**: additional dependence on the FWHM-to-σ ratio.  
    - **Pan25 iron term**: additional correction proportional to :math:`R_\mathrm{Fe}`.
    """

    #if extras is None:
    extras = broad_params.get("extras",{})

    out = {}

    fwhm_all = _col(broad_params.get("fwhm_kms"))
    lum_all  = _col(broad_params.get("luminosity"))
    sigma_all = broad_params.get("sigma_kms", None)
    flux_all = broad_params.get("flux", None)
    if sigma_all is not None:
        sigma_all = _col(sigma_all)

    lines = np.asarray(broad_params.get("lines", []))
    comps = np.asarray(broad_params.get("component", []))

    # if fwhm_all is None or lines.size == 0:
    #     return out

    # constants for mdot (continuum only)
    eta = 0.1
    c_cm = c * 1e5
    M_sun_g = 1.98847e33
    sec_yr = 3.15576e7

    for calib_key, est in estimators.items():
        line_name = est.get("line")
        kind = est.get("kind", "continuum")
        width_def = str(est.get("width_def", "fwhm")).lower()
        
        if broad_params.get(line_name):
            if width_def == "sigma":
                 Vwidth =  _col(broad_params.get(line_name).get("sigma_kms"))
            else:
                Vwidth = _col( broad_params.get(line_name).get("fwhm_kms"))
            L_line  = _col( broad_params.get(line_name).get("luminosity"))
            extras = broad_params.get(line_name).get("extras",{})
            local_extras = {}
            if est.get("extras", {}).get("le20_shape", False):
                local_extras["sigma_kms"] = Vwidth    
            #print(extras)
            comp_here = []
            #print(Vwidth.shape,est.get("kind"),est.get("width_def"))
            #continue 
        
        
         
        else:
            if not line_name or (line_name not in lines):
                continue
            #print(line_name,calib_key)
            idxs = np.where(lines == line_name)[0]
            comp_here = comps[idxs]

            # choose velocity width
            if width_def == "sigma":
                if sigma_all is not None:
                    Vwidth = sigma_all[:, idxs]
                elif "sigma_kms" in extras:
                    sig = _col(extras["sigma_kms"])
                    Vwidth = sig[:, idxs] if sig.ndim == 2 else sig
                else:
                    continue  # no sigma available
            else:
                Vwidth = fwhm_all[:, idxs]
            L_line = lum_all[:, idxs]
            
            # local extras (Le20 / Pan25)
            local_extras = {}
            if est.get("extras", {}).get("le20_shape", False):
                if sigma_all is not None:
                    local_extras["sigma_kms"] = sigma_all[:, idxs]
                elif "sigma_kms" in extras:
                    sig = _col(extras["sigma_kms"])
                    local_extras["sigma_kms"] = sig[:, idxs] if sig.ndim == 2 else sig
            local_extras["R_Fe"] = extras.get("flux_Fe",0)/flux_all[:, idxs]
            
        if "R_Fe" in extras:
            #print("R_fe")
            local_extras["R_Fe"] = extras["R_Fe"]
            
        if kind == "continuum":
            lam = est.get("wavelength", None)
            if lam is None:
                continue
            wkey = str(int(lam))
            if wkey not in L_w:
                continue

            Lmono = _col(L_w[wkey])
            MBH = calc_black_hole_mass(Lmono, Vwidth, est, extras=local_extras)

            # Ledd + mdot (only for continuum, and only if L_bol available)
            Ledd = 1.26e38 * MBH
            mdot_yr = None
            Lbol = None
            if wkey in L_bol:
                Lbol = _col(L_bol[wkey])
                mdot_gs = Lbol / (eta * c_cm**2)
                mdot_yr = mdot_gs / M_sun_g * sec_yr

            out.setdefault(line_name, {})[calib_key] = {
                "method": "continuum",
                "wavelength": lam,
                "vwidth_def": width_def,
                "vwidth_kms": Vwidth,
                "log10_smbh": log10(MBH),
                "Lwave": Lmono,
                "Lbol": Lbol,
                "Ledd": Ledd,
                "mdot_msun_per_year": mdot_yr,
                "component": comp_here,
            }

        elif kind == "line":

            #L_line = lum_all[:, idxs]
            MBH = calc_black_hole_mass(L_line, Vwidth, est, extras=local_extras)
            Ledd = 1.26e38 * MBH

            out.setdefault(line_name, {})[calib_key] = {
                "method": "line",
                "vwidth_def": width_def,
                "vwidth_kms": Vwidth,
                "Lline": L_line,
                "log10_smbh": log10(MBH),
                "Ledd": Ledd,
                "component": comp_here,
            }

    return out


