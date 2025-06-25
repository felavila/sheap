from typing import Dict

import numpy as np 
import jax.numpy as jnp
from jax import vmap
import warnings

from sheap.Mappers.helpers import mapping_params
from .functions import calc_flux,calc_fwhm_kms,calc_luminosity,calc_monochromatic_luminosity,calc_bolometric_luminosity,calc_black_hole_mass
from .constants import BOL_CORRECTIONS,SINGLE_EPOCH_ESTIMATORS,c

#Anoher class is coming x.x
# Compute EW: flux / continuum_flux_at_center
# for line_center, flux_line in zip(center.T, flux.T):
#     # Evaluate continuum at the line center
#     cont_at_center = vmap(cont_fun, in_axes=(0, 0))(line_center, cont_params)  # shape (n_samples,)
#     eqw = flux_line / cont_at_center
#     eqws.append(eqw)
    
# Skew/Kurtosis


def summarize_samples(samples) -> Dict[str, np.ndarray]:
    """Compute 16/50/84 percentiles and return a summary dict using NumPy."""
    if isinstance(samples, jnp.ndarray):
        samples = np.asarray(samples)
    samples = np.atleast_2d(samples).T
    if np.isnan(samples).sum() / samples.size > 0.2:
        warnings.warn("High fraction of NaNs; uncertainty estimates may be biased.")
    if samples.shape[1]<=1:
        q = np.nanpercentile(samples, [16, 50, 84], axis=0)
    else:
        q = np.nanpercentile(samples, [16, 50, 84], axis=1)
    #else:
    
    return {
        "median": q[1],
        "err_minus": q[1] - q[0],
        "err_plus": q[2] - q[1]
    }


def summarize_nested_samples(d: dict) -> dict:
    """
    Recursively walk through a dictionary and apply summarize_samples_numpy
    to any array-like values.
    """
    summarized = {}
    for k, v in d.items():
        if isinstance(v, dict):
            summarized[k] = summarize_nested_samples(v)
        elif isinstance(v, (np.ndarray, jnp.ndarray)) and np.ndim(v) >= 1 and k!='component':
            summarized[k] = summarize_samples(v)
        else:
            summarized[k] = v
    return summarized


def full_params_sampled_to_posterior_params(wl_i, flux_i, yerr_i,mask_i,
                                            full_samples,
                                            kinds_map,d,
                                            c=c,BOL_CORRECTIONS=BOL_CORRECTIONS,
                                            SINGLE_EPOCH_ESTIMATORS=SINGLE_EPOCH_ESTIMATORS,summarize=False):
    

    full_dictionary = {}
    dict_basic_params = {}
    cont_map = kinds_map['continuum']
    cont_fun= cont_map.profile_functions_combine #
    cont_idx = jnp.array(list(cont_map.filtered_dict.values())) #a.
    cont_params = full_samples[:, cont_idx]
    


def full_params_sampled_to_posterior_params_old_stable(wl_i, flux_i, yerr_i,mask_i,
                                            full_samples,
                                            kinds_map,d,
                                            c=c,BOL_CORRECTIONS=BOL_CORRECTIONS,
                                            SINGLE_EPOCH_ESTIMATORS=SINGLE_EPOCH_ESTIMATORS,summarize=False):
    
    
    
    full_dictionary = {}
    dict_basic_params = {}
    cont_map = kinds_map['continuum']
    cont_fun= cont_map.profile_functions_combine #
    cont_idx = jnp.array(list(cont_map.filtered_dict.values())) #a.
    cont_params = full_samples[:, cont_idx]
    for k, k_map in kinds_map.items():
        #This code assume that all the lines in a certain region share "profile" that cant be always the case. 
        #the so called emission lines ? 
        #here the idea will be move from "combine models to more simples ones "
         if k not in ['fe', 'continuum']:
            
            idx_amplitude = mapping_params(k_map.filtered_dict, "amplitude")
            idx_fwhm = mapping_params(k_map.filtered_dict, "fwhm") 
            idx_center = mapping_params(k_map.filtered_dict, "center")
            norm_amplitude = full_samples[:, idx_amplitude]
            
            fwhm = np.abs(full_samples[:, idx_fwhm])
            center = full_samples[:, idx_center]
            flux = calc_flux(norm_amplitude, fwhm)
            fwhm_kms = np.abs(calc_fwhm_kms(fwhm, c, center))
            L_line = calc_luminosity(d, flux, center)
            cont_at_center = vmap(cont_fun, in_axes=(0, 0))(center, cont_params) 
            eqw = flux / cont_at_center
            
            dict_basic_params[k] = {
                         'lines': k_map.line_name,
                         "component": np.array(k_map.component),
                         'flux': flux, "fwhm": fwhm, "fwhm_kms": fwhm_kms, "L": L_line,
                         'center': center, 'amplitude': norm_amplitude,"eqw":eqw
                     }
    
    
    wavelengths = np.array(list(BOL_CORRECTIONS.keys())).astype(float)
    #idx_cont = mapping_params(params_dict, "scale")  # Adapt if needed
    L_w, L_bol = {}, {}
    for w in wavelengths:
        wave = str(int(w))
        hits = jnp.isclose(wl_i, jnp.array([w]), atol=1)
        valid = (hits & (~mask_i)).any()
        corr = BOL_CORRECTIONS.get(wave, 0.0)
        if valid:
            flux_at_w = vmap(cont_fun, in_axes=(None, 0))(jnp.array([w]), cont_params).squeeze()
            Lw = calc_monochromatic_luminosity(d, flux_at_w, w)
            L_w[wave] = Lw
            L_bol[wave] = calc_bolometric_luminosity(Lw, corr)
        else:
            continue
            #L_w[wave] = jnp.zeros(full_samples.shape[0])
            #L_bol[wave] = jnp.zeros(full_samples.shape[0])
    dict_broad = dict_basic_params.get("broad")
    masses = {}
    if dict_broad is not None:
        fwhm_kms = dict_broad.get('fwhm_kms')
        line_name_list = np.array(dict_broad["lines"])
        for line_name, estimator in SINGLE_EPOCH_ESTIMATORS.items():
            wave = estimator["wavelength"]
            if line_name not in line_name_list or wave not in L_w:
                continue
            idx_broad = list(jnp.where(line_name == line_name_list)[0])
            _fwhm_kms = fwhm_kms[:, idx_broad].squeeze()
            Lwave = L_w[wave][..., np.newaxis] if _fwhm_kms.ndim == 2 else L_w[wave].squeeze()
            Lbol = L_bol[wave][..., np.newaxis] if _fwhm_kms.ndim == 2 else L_bol[wave].squeeze()
            smbh_sample = calc_black_hole_mass(Lwave, _fwhm_kms, estimator)
            Ledd = 1.26e38 * smbh_sample  # erg/s this is assuming that the SMBH is in solar mass
            eta = 0.1
            c_cm = c * 1e5 # the code uses c in km/s we move it to cm
            Msun = 1.98847e33  # Solar mass in grams
            sec_per_year = 3.15576e7 #
            mdot = Lbol / (eta * c_cm**2)  # g/s
            mdot_msun_per_year = mdot / Msun * sec_per_year
            masses[line_name] = {'Lwave':Lwave,'Lbol':Lbol,'fwhm_kms':_fwhm_kms,"log10_smbh":np.log10(smbh_sample),"mdot_msun_per_year":mdot_msun_per_year,'Ledd':Ledd}
    #         #np.atleast_2d(L_w[wave]).T
    #         fwhm_kms_ = fwhm_kms[:, idx_broad].squeeze()
    #         smbh_sample = calc_black_hole_mass(Lwave, fwhm_kms_, estimator)
    #         Lbol = np.atleast_2d(L_bol[wave]).T # shape (n_samples,)
    #         #
    #         ########################
    #         Ledd = 1.26e38 * smbh_sample  # erg/s this is assuming that the SMBH is in solar mass
    #         edd_ratio = np.where(Ledd > 0, Lbol[:,None]/ Ledd, 0)
    #         ###
    #         eta = 0.1
    #         c_cm = c * 1e5 # the code uses c in km/s we move it to cm
    #         Msun = 1.98847e33  # Solar mass in grams
    #         sec_per_year = 3.15576e7 #
    #         mdot = Lbol / (eta * c_cm**2)  # g/s
    #         mdot_msun_per_year = mdot / Msun * sec_per_year
    #         masses[line_name] = {"smbh_log10":np.log10(smbh_sample),"Ledd": Ledd,"mdot_msun_per_year":mdot_msun_per_year,"edd_ratio":edd_ratio.squeeze(),"fwhm_kms":fwhm_kms_,
    #                              "Lw":Lwave}#,"median":median,"err_minus":err_minus,"err_plus":err_plus
    # #integrate flux fe_
    
    full_dictionary["dict_basic_params"] = dict_basic_params
    full_dictionary["L_w"] = L_w
    full_dictionary["L_bol"] = L_bol
    full_dictionary["masses"] = masses
    if summarize:
        full_dictionary = summarize_nested_samples(full_dictionary)
    return full_dictionary
            
    #Line Ratio Posteriors	E.g., [OIII]/Hβ, CIV/MgII	Use flux[:, i] / flux[:, j]
    #Line Peak SNR	Amplitude / local continuum noise	Estimate continuum std around center
    #Line Profile Skew/Kurtosis	Optional with samples	Approximate from parameters or synthetic line shape
    

def full_params_sampled_to_posterior_paramsv2(wl_i, flux_i, yerr_i,mask_i,
                                            full_samples,
                                            kinds_map,d,
                                            c=c,BOL_CORRECTIONS=BOL_CORRECTIONS,
                                            SINGLE_EPOCH_ESTIMATORS=SINGLE_EPOCH_ESTIMATORS,summarize=False):
    
    
    
    full_dictionary = {}
    dict_basic_params = {}
    cont_map = kinds_map['continuum']
    cont_fun= cont_map.profile_functions_combine #
    cont_idx = jnp.array(list(cont_map.filtered_dict.values())) #a.
    cont_params = full_samples[:, cont_idx]
    for k, k_map in kinds_map.items():
        #This code assume that all the lines in a certain region share "profile" that cant be always the case. 
        #the so called emission lines ? 
        
         if k not in ['fe', 'continuum']:
            idx_amplitude = mapping_params(k_map.filtered_dict, "amplitude")
            idx_fwhm = mapping_params(k_map.filtered_dict, "fwhm") 
            idx_center = mapping_params(k_map.filtered_dict, "center")
            norm_amplitude = full_samples[:, idx_amplitude]
            fwhm = full_samples[:, idx_fwhm]
            center = full_samples[:, idx_center]
            flux = calc_flux(norm_amplitude, fwhm)
            fwhm_kms = calc_fwhm_kms(fwhm, c, center)
            L_line = calc_luminosity(d, flux, center)
            cont_at_center = vmap(cont_fun, in_axes=(0, 0))(center, cont_params) 
            eqw = flux / cont_at_center
            
            dict_basic_params[k] = {
                         'lines': k_map.line_name,
                         "component": np.array(k_map.component),
                         'flux': flux, "fwhm": fwhm, "fwhm_kms": fwhm_kms, "L": L_line,
                         'center': center, 'amplitude': norm_amplitude,"eqw":eqw
                     }
    wavelengths = np.array(list(BOL_CORRECTIONS.keys())).astype(float)
    #idx_cont = mapping_params(params_dict, "scale")  # Adapt if needed
    L_w, L_bol = {}, {}
    for w in wavelengths:
        wave = str(int(w))
        hits = jnp.isclose(wl_i, jnp.array([w]), atol=1)
        valid = (hits & (~mask_i)).any()
        corr = BOL_CORRECTIONS.get(wave, 0.0)
        if valid:
            flux_at_w = vmap(cont_fun, in_axes=(None, 0))(jnp.array([w]), cont_params).squeeze()
            Lw = calc_monochromatic_luminosity(d, flux_at_w, w)
            L_w[wave] = Lw
            L_bol[wave] = calc_bolometric_luminosity(Lw, corr)
        else:
            continue
            #L_w[wave] = jnp.zeros(full_samples.shape[0])
            #L_bol[wave] = jnp.zeros(full_samples.shape[0])
    dict_broad = dict_basic_params.get("broad")
    masses = {}
    if dict_broad is not None:
        fwhm_kms = dict_broad.get('fwhm_kms')
        line_name_list = np.array(dict_broad["lines"])
        for line_name, estimator in SINGLE_EPOCH_ESTIMATORS.items():
            wave = estimator["wavelength"]
            if line_name not in line_name_list or wave not in L_w:
                continue
            idx_broad = list(jnp.where(line_name == line_name_list)[0])
            Lwave = L_w[wave]
            fwhm_kms_ = fwhm_kms[:, idx_broad].squeeze()
            smbh_sample = calc_black_hole_mass(Lwave, fwhm_kms_, estimator)
            Lbol = L_bol[wave]  # shape (n_samples,)
            ########################
            Ledd = 1.26e38 * smbh_sample  # erg/s this is assuming that the SMBH is in solar mass
            edd_ratio = jnp.where(Ledd > 0, Lbol / Ledd, 0)
            ###
            eta = 0.1
            c_cm = c * 1e5 # the code uses c in km/s we move it to cm
            Msun = 1.98847e33  # Solar mass in grams
            sec_per_year = 3.15576e7 #
            mdot = Lbol / (eta * c_cm**2)  # g/s
            mdot_msun_per_year = mdot / Msun * sec_per_year
            masses[line_name] = {"smbh_log10":np.log10(smbh_sample),"Ledd": Ledd,"mdot_msun_per_year":mdot_msun_per_year,"edd_ratio":edd_ratio}#,"median":median,"err_minus":err_minus,"err_plus":err_plus
    #integrate flux fe_
    
    full_dictionary["dict_basic_params"] = dict_basic_params
    full_dictionary["L_w"] = L_w
    full_dictionary["L_bol"] = L_bol
    full_dictionary["masses"] = masses
    if summarize:
        full_dictionary = summarize_nested_samples(full_dictionary)
    return full_dictionary




