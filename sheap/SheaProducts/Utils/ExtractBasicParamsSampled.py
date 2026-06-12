"""
Extract Basic Params Sampled
============================

?
"""

__author__ = 'felavila'

__all__ = []

from typing import Any, Callable, Dict, List, Optional, Tuple, Union,Iterable

import numpy as np
import jax.numpy as jnp
from jax import vmap,jit

from sheap.Profiles.Profiles import PROFILE_LINE_FUNC_MAP
from sheap.Profiles.Utils import make_integrator,make_fused_profiles

from sheap.SheaProducts.Utils.CombineUtils import make_batch_fwhm_split
from sheap.SheaProducts.Utils.Sample_handlers import concat_dicts

from sheap.Utils.Constants import DEFAULT_BOL_CORRECTIONS,DEFAULT_C_KMS
from sheap.Utils.BasicFunctions import calc_fwhm_kms,calc_luminosity,calc_monochromatic_luminosity,calc_bolometric_luminosity

#I dont like d has name of variable maybe D?
#wl_i = spectra[idx_obj, 0, :]
#mask_i = mask[idx_obj, :]

def extract_basic_params_sampled(sheapmodel,wavelength,mask,samples,continuum_idx_all,cont_profile_all,cont_profile,luminosity_distance=0,BOL_CORRECTIONS =DEFAULT_BOL_CORRECTIONS,C_KMS= DEFAULT_C_KMS,wavelength_grid=jnp.linspace(0, 20_000, 20_000)):
        """
        Extract line quantities (flux, FWHM, center, etc.) from posterior samples.
        Designed for use with MCMC or MC draws.
        ld: luminosity_distance
        """
        
        basic_params: Dict[str, Dict[str, np.ndarray]] = {}
        sheapmodel_group_by_region = sheapmodel.group_by("region")
        cont_group = sheapmodel_group_by_region["continuum"]
        idx_cont = cont_group.flat_param_indices_global
        cont_params = samples[:, idx_cont]
        cont_params_all = samples[:, continuum_idx_all]
        distances = np.full((samples.shape[0],),luminosity_distance, dtype=np.float64)
        for region, region_group in sheapmodel_group_by_region.items():
            if region in ("fe", "continuum", "host","balmer"):
                continue

            line_names, components = [], []
            flux_parts, fwhm_parts = [], []
            fwhm_kms_parts, center_parts = [], []
            amp_parts, eqw_parts, lum_parts = [], [], []
            shape_params_list = []

            region_group_by_profile = region_group.group_by("profile_name")

            for profile_name, prof_group in region_group_by_profile.items():
                if "_" in profile_name:
                    _, subprof = profile_name.split("_", 1)
                    profile_fn = PROFILE_LINE_FUNC_MAP[subprof]
                    batch_fwhm = make_batch_fwhm_split(subprof)
                    integrator = make_integrator(profile_fn, method="vmap")
                    (_line_names, _components, _flux, _fwhm, _fwhm_kms,_centers, _amps, _eqw, _lum, _shapes) = _accumulate_spaf_sampled(prof_group, 
                                                                                                                                        profile_fn, batch_fwhm, 
                                                                                                                                        integrator, cont_params_all, cont_profile_all,
                                                                                                                                        samples,distances=distances,C_KMS=C_KMS)

                else:
                    profile_fn = PROFILE_LINE_FUNC_MAP[profile_name]
                    batch_fwhm = make_batch_fwhm_split(profile_name)
                    integrator = make_integrator(profile_fn, method="vmap")

                    idxs = prof_group.flat_param_indices_global
                    params = samples[:, idxs]

                    _line_names = [l.line_name for l in prof_group.lines]
                    _components = [l.component for l in prof_group.lines]
                    params_by_line = params.reshape(params.shape[0], -1, profile_fn.n_params)

                    amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals = _extract_sampled_profile_quantities(
                        integrator, batch_fwhm, params_by_line, cont_params_all, distances)

                    _flux, _fwhm, _fwhm_kms = [flux], [fwhm], [fwhm_kms]
                    _centers, _amps, _eqw, _lum = [centers], [amps], [eqw], [lum_vals]
                    _shapes = [{k: v for k, v in zip(profile_fn.param_names[2:], shape_params.T)}]

                line_names.extend(_line_names)
                components.extend(_components)
                flux_parts.extend(_flux)
                fwhm_parts.extend(_fwhm)
                fwhm_kms_parts.extend(_fwhm_kms)
                center_parts.extend(_centers)
                amp_parts.extend(_amps)
                eqw_parts.extend(_eqw)
                lum_parts.extend(_lum)
                shape_params_list.extend(_shapes)

            line_names = np.array(line_names)
            basic_params[region] = {"lines": line_names,
                "component": components,
                "flux": np.concatenate(flux_parts, axis=1),
                "fwhm": np.concatenate(fwhm_parts, axis=1),
                "fwhm_kms": np.concatenate(fwhm_kms_parts, axis=1),
                "center": np.concatenate(center_parts, axis=1),
                "amplitude": np.concatenate(amp_parts, axis=1),
                "eqw": np.concatenate(eqw_parts, axis=1),
                "luminosity": np.concatenate(lum_parts, axis=1),
                "shape_params": concat_dicts(shape_params_list),
                "idx_by_name": {name: np.where(line_names == name)[0]  for name in np.unique(line_names)},
                "combined": [False] * len(components)
            }
        L_w, L_bol,F_cont = {}, {},{}
        for wave in map(float, BOL_CORRECTIONS.keys()):
            wstr = str(int(wave))
            if (jnp.isclose(wavelength, wave, atol=2) & ~mask).any():
                Fcont = cont_profile(jnp.array([wave]), cont_params).squeeze()
                Lmono = calc_monochromatic_luminosity(distances, Fcont, wave)
                Lbolval = calc_bolometric_luminosity(Lmono, BOL_CORRECTIONS[wstr])
                L_w[wstr], L_bol[wstr],F_cont[wstr] = np.array(Lmono), np.array(Lbolval), np.array(Fcont)     
        
        
        #list_to_get_extra_params = ["basic_params"]
        result = {"basic_params": basic_params, "L_w": L_w, "L_bol": L_bol,"F_cont":F_cont,"distances":distances[0]} #<-
        return result

    
def _accumulate_spaf_sampled(prof_group, profile_fn, batch_fwhm, integrator_fn, cont_params_all,cont_profile_all, samples,distances,C_KMS=DEFAULT_C_KMS):
    all_flux, all_fwhm, all_fwhm_kms = [], [], []
    all_centers, all_amps, all_eqws, all_lums = [], [], [], []
    all_line_names, all_components, all_shape_dicts = [], [], []
    params_names = prof_group._master_param_names
    
    for sp,idx_param in zip(prof_group.lines,prof_group.global_profile_params_index_list):
        params_by_line = _build_spaf_sampled_params(sp,idx_param,params_names,samples,C_KMS=C_KMS)
        amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals = _extract_sampled_profile_quantities(integrator_fn, batch_fwhm, 
                                                                                                               params_by_line, cont_params_all,cont_profile_all,
                                                                                                               np.full((samples.shape[0],), distances))
        
        all_flux.append(flux)
        all_fwhm.append(fwhm)
        all_fwhm_kms.append(fwhm_kms)
        all_centers.append(centers)
        all_amps.append(amps)
        all_eqws.append(eqw)
        all_lums.append(lum_vals)
        all_line_names.extend(sp.region_lines)
        all_components.extend([sp.component] * params_by_line.shape[1])
        all_shape_dicts.append({k: v for k, v in zip(profile_fn.param_names[2:], shape_params.T)})

    return (
        all_line_names, all_components, all_flux, all_fwhm, all_fwhm_kms,
        all_centers, all_amps, all_eqws, all_lums, all_shape_dicts)


 
def _build_spaf_sampled_params(sp,idx_param,params_names, samples,C_KMS=DEFAULT_C_KMS):
    "moving from velocity to ANGSTROMS"
    params = samples[:, idx_param]
    names = np.array(params_names)[idx_param]
    
    amplitude_relations = sp.amplitude_relations
    amplitude_index = [i for i, name in enumerate(names) if "amplitude" in name]
    ind_amplitude_index = {i[2] for i in amplitude_relations}
    dic_amp = {i: ii for i, ii in zip(ind_amplitude_index, amplitude_index)}
    idx_shift = max(amplitude_index) + 1
    full_params_by_line = []
    for i,(_,factor,idx) in enumerate(amplitude_relations):
        amp = params[:, [dic_amp[idx]]] *factor #+ np.log10(factor)
        center = sp.center[i] * (1+params[:, [idx_shift]]/C_KMS)
        extras = (10**params[:, idx_shift+1:]) * center/C_KMS
        full_params_by_line.append(np.column_stack([amp, center, extras]))

    return np.moveaxis(np.array(full_params_by_line), 0, 1)

def _extract_sampled_profile_quantities(integrator_fn, batch_fwhm, params_by_line, cont_params_all,cont_profile_all,
                                        distances,C_KMS=DEFAULT_C_KMS,wavelength_grid=jnp.linspace(0, 20_000, 20_000)):
    """
    distances -> 1D object
    """
    amps = params_by_line[:, :, 0]
    #print(amps)
    centers = params_by_line[:, :, 1]
    shape_params = jnp.abs(params_by_line[:, :, 2:])

    flux = integrator_fn(wavelength_grid, params_by_line)
    fwhm = batch_fwhm(amps, centers, shape_params)
    fwhm_kms = jnp.abs(calc_fwhm_kms(fwhm, C_KMS, centers))
    #cont_params = 
    cont_vals = cont_profile_all(centers, cont_params_all) #<-
    
    
    eqw = flux / cont_vals
    lum_vals = calc_luminosity(distances[:, None], flux)

    return amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals