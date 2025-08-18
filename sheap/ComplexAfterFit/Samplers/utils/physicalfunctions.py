"""This module handles basic operations."""
__author__ = 'felavila'

__all__ = [
    "calc_black_hole_mass",
    "calc_black_hole_mass_gh2015",
    "calc_bolometric_luminosity",
    "calc_flux",
    "calc_fwhm_kms",
    "calc_luminosity",
    "calc_monochromatic_luminosity",
    "ensure_column_matrix",
    "extra_params_functions",
    "extra_params_functionsv0",
]

import jax.numpy as np
import numpy as np 
from auto_uncertainties import Uncertainty

def calc_flux(norm_amplitude, fwhm):
    """
    Compute flux from normalized amplitude and FWHM assuming a Gaussian profile.
    F = A × FWHM × sqrt(π / (4 * ln(2)))
    """
    return np.sqrt(2.0 * np.pi) * norm_amplitude * fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

def calc_luminosity(distance, flux):
    """
    Line luminosity: L = 4π D^2 × F 
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



def ensure_column_matrix(x):
    #to utils.
    if isinstance(x,Uncertainty):
        if x.ndim == 1:
            #print("1D")
            #print(x_.shape)
            return x.reshape(-1, 1)  # Convert to (N, 1)
        return x
    x = np.asarray(x)
    if x.ndim == 1:
        #print(x.reshape(1, -1).shape)
        return x.reshape(-1, 1)  # Convert to (N, 1)
    
    return x


def extra_params_functionsv0(broad_params, L_w, L_bol, estimators, c):
    """
    Compute derived AGN parameters from broad line measurements.

    Parameters
    ----------
    broad_params : dict
        Dictionary with extracted parameters including FWHM, luminosity, and line labels.
    L_w : dict
        Dictionary of monochromatic luminosities keyed by wavelength (as str).
    L_bol : dict
        Dictionary of bolometric luminosities keyed by wavelength (as str).
    estimators : dict
        Dictionary of estimator configurations, keys formatted as "{line}_{method}".
        Each value is a dict with keys: 'wavelength', 'a', 'b', 'f' or 'fwhm_factor'.
    c : float
        Speed of light in km/s.

    Returns
    -------
    dict
        Dictionary of derived parameters per line.
    """
    dict_extra_params = {}
    #n_obj,nlines_component
    fwhm_kms_all = broad_params.get("fwhm_kms")
    lum_all = broad_params.get("luminosity")
    line_list = np.array(broad_params.get("lines", []))
    component_list = np.array(broad_params.get("component", []))
    #print(line_list)
    if fwhm_kms_all is None or line_list.size == 0:
        return dict_extra_params
    for line_method, est in estimators.items():
        try:
            line_name, method = line_method.split("_", 1)
        except ValueError:
            continue  # invalid key format

        if line_name not in line_list:
            continue
        idxs = np.where(line_list == line_name)[0]

        compt = component_list[idxs]
        #print("compt",idxs,"fwhm_kms_all",fwhm_kms_all.shape,ensure_column_matrix(fwhm_kms_all).shape)
        fkm = ensure_column_matrix(fwhm_kms_all)[:, idxs]
        lum_custom = ensure_column_matrix(lum_all)[:, idxs]
        #print("lum_custom",lum_custom.shape,"fkm",fkm.shape)
        if line_name not in dict_extra_params:
            dict_extra_params[line_name] = {}

        
        log_FWHM = np.log10(fkm) - 3

        if method == "w":
            lam = est.get("wavelength", 0)
            wstr = str(int(lam))

            if wstr not in L_w:
                continue

            Lmono = L_w[wstr]
            Lbolval = L_bol[wstr]
            a, b, f = est["a"], est["b"], est["f"]
            log_L = np.log10(Lmono)
            log_M_BH = a + b * (log_L - 44.0) + 2 * log_FWHM
            M_BH = (10 ** log_M_BH) / f
            L_edd = 1.26e38 * M_BH

            # mass accretion rate (M_sun / yr)
            eta = 0.1
            c_cm = c * 1e5
            M_sun_g = 1.98847e33
            sec_yr = 3.15576e7
            mdot_gs = Lbolval / (eta * c_cm**2)
            mdot_yr = mdot_gs / M_sun_g * sec_yr

            dict_extra_params[line_name].update({
                "Lwave": Lmono,
                "Lbol": Lbolval,
                "fwhm_kms": fkm,
                "log10_smbh": np.log10(M_BH),
                "Ledd": L_edd,
                "mdot_msun_per_year": mdot_yr,
                "component": compt,
            })

        elif method == "l":
            a, b, fwhm_factor = est["a"], est["b"], est["fwhm_factor"]
            log_M_special = a + b * (np.log10(lum_custom) - 42) + fwhm_factor * log_FWHM
            dict_extra_params[line_name].update({
                f"log10_smbh_{line_name.lower()}": log_M_special
            })

    return dict_extra_params




def _col(x):
    if isinstance(x,Uncertainty): 
        return x.reshape(-1, 1) if x.ndim == 1 else x
    x = np.asarray(x)
    return x.reshape(-1, 1) if x.ndim == 1 else x
    
    
def calc_black_hole_mass(L_in, vwidth_kms, estimator, extras=None):
    """
    Unified SE BH-mass estimator (continuum or line), supports width_def and extras.
    Returns M_BH in Msun with the estimator's virial factor applied.
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
    logM = np.log10(f) + a + b * (np.log10(L) - np.log10(L0)) + beta * (np.log10(V) - np.log10(V0))

    # Le20 shape (only if baseline uses FWHM)
    if width_def == "fwhm" and estimator.get("extras", {}).get("le20_shape", False):
        sigma = extras.get("sigma_kms", None)
        if sigma is not None:
            sigma = _col(sigma)
            logM += (-1.14 * (np.log10(V) - np.log10(sigma)) + 0.33)

    # Pan25 iron term
    if "R_Fe" in extras:
        gamma = estimator.get("extras", {}).get("pan25_gamma", -0.34)
        RFe = _col(extras["R_Fe"])
        logM += gamma * RFe  # broadcasts across components

    return (10.0 ** logM)


def extra_params_functions(broad_params, L_w, L_bol, estimators, c, extras=None):
    """
    Compute derived parameters and SE BH masses.
    - kind='continuum': uses L_w[lambda]; computes Ledd and mdot (if L_bol available).
    - kind='line'    : uses per-component line luminosity; computes Ledd; no mdot.
    """
    if extras is None:
        extras = {}

    out = {}

    fwhm_all = _col(broad_params.get("fwhm_kms"))
    lum_all  = _col(broad_params.get("luminosity"))
    sigma_all = broad_params.get("sigma_kms", None)
    if sigma_all is not None:
        sigma_all = _col(sigma_all)

    lines = np.asarray(broad_params.get("lines", []))
    comps = np.asarray(broad_params.get("component", []))

    if fwhm_all is None or lines.size == 0:
        return out

    # constants for mdot (continuum only)
    eta = 0.1
    c_cm = c * 1e5
    M_sun_g = 1.98847e33
    sec_yr = 3.15576e7

    for calib_key, est in estimators.items():
        line_name = est.get("line")
        if not line_name or (line_name not in lines):
            continue

        kind = est.get("kind", "continuum")
        width_def = str(est.get("width_def", "fwhm")).lower()

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

        # local extras (Le20 / Pan25)
        local_extras = {}
        if est.get("extras", {}).get("le20_shape", False):
            if sigma_all is not None:
                local_extras["sigma_kms"] = sigma_all[:, idxs]
            elif "sigma_kms" in extras:
                sig = _col(extras["sigma_kms"])
                local_extras["sigma_kms"] = sig[:, idxs] if sig.ndim == 2 else sig
        if "R_Fe" in extras:
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
                "log10_smbh": np.log10(MBH),
                "Lwave": Lmono,
                "Lbol": Lbol,
                "Ledd": Ledd,
                "mdot_msun_per_year": mdot_yr,
                "component": comp_here,
            }

        elif kind == "line":

            L_line = lum_all[:, idxs]
            MBH = calc_black_hole_mass(L_line, Vwidth, est, extras=local_extras)
            Ledd = 1.26e38 * MBH

            out.setdefault(line_name, {})[calib_key] = {
                "method": "line",
                "vwidth_def": width_def,
                "vwidth_kms": Vwidth,
                "Lline": L_line,
                "log10_smbh": np.log10(MBH),
                "Ledd": Ledd,
                "component": comp_here,
            }

    return out
