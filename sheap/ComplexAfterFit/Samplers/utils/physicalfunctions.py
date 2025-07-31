"""This module handles basic operations."""
__version__ = '0.1.0'
__author__ = 'Unknown'
# Auto-generated __all__
__all__ = [
    "calc_black_hole_mass",
    "calc_black_hole_mass_gh2015",
    "calc_bolometric_luminosity",
    "calc_flux",
    "calc_fwhm_kms",
    "calc_luminosity",
    "calc_monochromatic_luminosity",
]

import jax.numpy as np
from jax import vmap, random
import numpy as np 


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


def extra_params(broad_params,L_w,L_bol,estimators,c,combine_mode):
    dict_extra_params = {}
    fwhm_kms_all = broad_params.get("fwhm_kms")
    lum_all = broad_params.get("luminosity")
    line_list    = np.array(broad_params.get("lines", []))
    component_list = np.array(broad_params.get("component", []))
    if combine_mode:
        line_list = np.array(list(broad_params.keys()))
        fwhm_kms_all =  np.stack([broad_params[l]["fwhm_kms"] for l in line_list],axis=1)
        lum_all =  np.stack([broad_params[l]["luminosity"] for l in line_list],axis=1)
    if fwhm_kms_all is None or line_list.size == 0:
        return dict_extra_params

    for line_name, est in estimators.items():
        lam  = est["wavelength"]
        wstr = str(int(lam))
        if line_name not in line_list or wstr not in L_w:
            continue
        idxs    = np.where(line_list == line_name)[0]
        if combine_mode:
            compt = broad_params.get("component")
        else:
            compt = component_list[idxs]   # (N,) or (N,1)
        fkm     = fwhm_kms_all[:, idxs].squeeze()   # (N,) or (N,1)
       
        Lmono   = L_w[wstr]                         # (N,)
        Lbolval = L_bol[wstr]   
        if fkm.ndim == 2:
            Lmono   = Lmono[..., None]
            Lbolval = Lbolval[..., None]

        a, b, f = est["a"], est["b"], est["f"]
        
        log_L = np.log10(Lmono)
        log_FWHM = np.log10(fkm) - 3  # FWHM to 10^3 km/s
        log_M_BH = a + b * (log_L - 44.0) + 2 * log_FWHM
        M_BH = (10 ** log_M_BH) / f
        L_edd    = 1.26e38 * M_BH  # [erg/s]

        # mass‐accretion rate (M⊙/yr)
        eta      = 0.1
        c_cm     = c * 1e5             # km/s → cm/s
        M_sun_g  = 1.98847e33          # g
        sec_yr   = 3.15576e7
        ########
        mdot_gs  = Lbolval / (eta * c_cm**2)  
        mdot_yr  = mdot_gs / M_sun_g * sec_yr

        dict_extra_params[line_name] = {
            "Lwave":              Lmono,
            "Lbol":               Lbolval,
            "fwhm_kms":           fkm,
            "log10_smbh":         np.log10(M_BH),
            "Ledd":               L_edd,
            "mdot_msun_per_year": mdot_yr,
            "component": compt,}
        if line_name=="Halpha":
            #https://iopscience.iop.org/article/10.1088/0004-637X/813/2/82/pdf
            Lhalpha     = lum_all[:, idxs].squeeze()   # (N,) or (N,1)
           # print("Lhalpha:",Lhalpha[0],log_FWHM[0])
            logMBH = np.log10(1.075) + 6.57 + 0.47 * (np.log10(Lhalpha) - 42) +  2.06 * (log_FWHM)
            print(logMBH[0])
            dict_extra_params[line_name].update({"log10_smbh_halpha":logMBH})
    return dict_extra_params
