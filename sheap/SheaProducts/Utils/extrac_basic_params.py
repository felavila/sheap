
from typing import Any, Callable, Dict, List, Optional, Tuple, Union,Iterable

import numpy as np
import jax.numpy as jnp
from jax import vmap
from uncertainties import unumpy


from sheap.Profiles.Profiles import PROFILE_LINE_FUNC_MAP#,PROFILE_FUNC_MAP,PROFILE_LINE_FUNC_MAP_classical
#from sheap.Profiles.Utils import make_integrator,make_fused_profiles

from sheap.SheaProducts.Utils.fwhm_conv import make_batch_fwhm_split_with_error
from sheap.SheaProducts.Utils.Physical_functions import calc_fwhm_kms,calc_luminosity,calc_monochromatic_luminosity,calc_bolometric_luminosity,extra_params_functions
from sheap.SheaProducts.Utils.After_fit_profile_helpers import integrate_batch_with_error,evaluate_with_error 
#from sheap.SheaProducts.Utils.Combine_profiles import combine_components,combine_fastspecfit,combine_pyqsofit,combine_pyqsofit_single

from sheap.SheaProducts.Utils.Sample_handlers import concat_dicts

from sheap.Utils.Constants import DEFAULT_BOL_CORRECTIONS,DEFAULT_C_KMS



#TODO update the continuum for the EW estimation
###########################SINGLE########################################
def extract_basic_params_single(spectra,mask,params,uncertainty_params,continuum_idx_all,
                                luminosity_distance,sheapmodel,cont_profile_all,
                                BOL_CORRECTIONS =DEFAULT_BOL_CORRECTIONS,C_KMS= DEFAULT_C_KMS,wavelength_grid=jnp.linspace(0, 20_000, 20_000)):
    #TODO the contnuum
    basic_params: Dict[str, Dict[str, np.ndarray]] = {}
    distances = luminosity_distance
    sheapmodel_group_by_region = sheapmodel.group_by("region")
    cont_group = sheapmodel_group_by_region["continuum"]
    idx_cont = cont_group.flat_param_indices_global
    cont_params = params[:, idx_cont]
    ucont_params = uncertainty_params[:, idx_cont]
    cont_params_full = params[:, continuum_idx_all]
    cont_uparams_full = uncertainty_params[:, continuum_idx_all]
    for region, region_group in sheapmodel_group_by_region.items():
        if region in ("fe", "continuum", "host","balmer"):
            continue

        line_names, components = [], []
        flux_parts, fwhm_parts, fwhm_kms_parts = [], [], []
        center_parts, amp_parts, eqw_parts, lum_parts = [], [], [], []
        shape_params_list = []

        region_group_by_profile = region_group.group_by("profile_name")

        for profile_name, prof_group in region_group_by_profile.items():
            if "_" in profile_name:  # SPAF or template Fe
                _, subprof = profile_name.split("_", 1)
                profile_fn = PROFILE_LINE_FUNC_MAP[subprof]
                batch_fwhm = make_batch_fwhm_split_with_error(subprof)

                (_line_names, _components, _flux, _fwhm, _fwhm_kms,_centers, _amps, _eqw, _lum, _shapes) = _accumulate_spaf_components(prof_group, profile_fn, batch_fwhm,params,uncertainty_params,cont_params_full, 
                                                                                                                                       cont_uparams_full,cont_profile_full=cont_profile_all
                                                                                                                                       ,distances=distances,wavelength_grid= wavelength_grid,C_KMS=C_KMS)

            else:
                profile_fn = PROFILE_LINE_FUNC_MAP[profile_name]
                batch_fwhm = make_batch_fwhm_split_with_error(profile_name)

                idxs = prof_group.flat_param_indices_global
                _params = params[:, idxs]
                _uparams = uncertainty_params[:, idxs]

                _line_names = [l.line_name for l in prof_group.lines]
                _components = [l.component for l in prof_group.lines]

                params_by_line = _params.reshape(_params.shape[0], -1, profile_fn.n_params)
                uparams_by_line = _uparams.reshape(_uparams.shape[0], -1, profile_fn.n_params)

                amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals = _extract_profile_quantities(profile_fn, batch_fwhm, params_by_line, uparams_by_line, 
                                                                                    cont_params_full, cont_uparams_full,cont_profile_full=cont_profile_all,
                                                                                    distances=distances,wavelength_grid=wavelength_grid,C_KMS=C_KMS)
                                                                                    #profile_fn, batch_fwhm, params_by_line, uparams_by_line, cont_params, ucont_params,distances,wavelength_grid= wavelength_grid,C_KMS=C_KMS)
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

        basic_params[region] = {
            "lines": line_names,
            "component": components,
            "flux": np.concatenate(flux_parts, axis=1),
            "fwhm": np.concatenate(fwhm_parts, axis=1),
            "fwhm_kms": np.concatenate(fwhm_kms_parts, axis=1),
            "center": np.concatenate(center_parts, axis=1),
            "amplitude": np.concatenate(amp_parts, axis=1),
            "eqw": np.concatenate(eqw_parts, axis=1),
            "luminosity": np.concatenate(lum_parts, axis=1),
            "shape_params": concat_dicts(shape_params_list) 
        }
    L_w, L_bol,F_cont = {}, {},{}
    for wave in map(float, BOL_CORRECTIONS.keys()):
        wstr = str(int(wave))
        hits = jnp.isclose(spectra[:, 0, :], wave, atol=2)
        valid = np.array((hits & (~mask)).any(axis=1, keepdims=True))

        if any(valid):
            x = jnp.full((cont_params.shape[0], 1), wave)
            Fcont = unumpy.uarray(*np.array(
                evaluate_with_error(cont_group.combined_profile, x, cont_params, jnp.zeros_like(x), ucont_params))) * valid.astype(float)
            #print(valid)
            Lmono = calc_monochromatic_luminosity(np.array(distances[:, None]), Fcont, wave)
            Lbolval = calc_bolometric_luminosity(Lmono, BOL_CORRECTIONS[wstr])
            L_w[wstr], L_bol[wstr],F_cont[wstr] = Lmono, Lbolval,Fcont
    
    result = {"basic_params": basic_params, "L_w": L_w, "L_bol": L_bol,"F_cont":F_cont,"distances":distances}
    
    return result
    
def _accumulate_spaf_components(prof_group, profile_fn, batch_fwhm,params,uncertainty_params,cont_params_full, cont_uparams_full,cont_profile_full=None,distances=None,wavelength_grid= jnp.linspace(0, 20_000, 20_000),C_KMS=DEFAULT_C_KMS):
    
    all_flux, all_fwhm, all_fwhm_kms = [], [], []
    all_centers, all_amps, all_eqws, all_lums = [], [], [], []
    all_line_names, all_components, all_shape_dicts = [], [], []
    #for sub_prof_gropu in 
    params_names = prof_group._master_param_names
    for sp,idx_params in zip(prof_group.lines,prof_group.global_profile_params_index_list,):
        params_by_line, uparams_by_line = _build_spaf_param_matrices(sp,idx_params,params,uncertainty_params,params_names,C_KMS=C_KMS)
        
        amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals = _extract_profile_quantities(profile_fn, batch_fwhm, params_by_line, uparams_by_line, 
                                                                            cont_params_full, cont_uparams_full,cont_profile_full=cont_profile_full,
                                                                            distances=distances,wavelength_grid= wavelength_grid,C_KMS=C_KMS)
        
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
        all_centers, all_amps, all_eqws, all_lums, all_shape_dicts
    )
                 
def _build_spaf_param_matrices(sp,idx_params,params,uncertainty_params,params_names,C_KMS=DEFAULT_C_KMS):
    #given that the center now is a variable here we have to change other stuff to
    full_params_by_line = []
    ufull_params_by_line = []
    _params = params[:, idx_params]
    _uncertainty_params = uncertainty_params[:, idx_params]
    names = np.array(params_names)[idx_params]
    
    amplitude_relations = sp.amplitude_relations
    #amplitude_index = [i for i, name in enumerate(names) if "logamp" in name] #keep log in case we endend using it
    amplitude_index = [i for i, name in enumerate(names) if "amplitude" in name]
    ind_amplitude_index = {i[2] for i in amplitude_relations}
    dic_amp = {i: ii for i, ii in zip(ind_amplitude_index, amplitude_index)}
    idx_shift = max(amplitude_index) + 1
    for i,(_, factor, idx) in enumerate(amplitude_relations):
        amp = _params[:, [dic_amp[idx]]] *factor #+ np.log10(factor)
        uamp = _uncertainty_params[:, [dic_amp[idx]]]
        #center = sp.center[i] + _params[:, [idx_shift]]
        center = sp.center[i] * (1+_params[:, [idx_shift]]/C_KMS)
        ucenter = _uncertainty_params[:, [idx_shift]]
        extras = (10**_params[:, idx_shift+1:]) * center/C_KMS
        uextras = _uncertainty_params[:, idx_shift+1:] * center/C_KMS
        full_params_by_line.append(np.column_stack([amp, center, extras]))
        ufull_params_by_line.append(np.column_stack([uamp, ucenter, uextras]))
    return np.moveaxis(np.array(full_params_by_line), 0, 1), np.moveaxis(np.array(ufull_params_by_line), 0, 1)

def _extract_profile_quantities(profile_fn, batch_fwhm, params_by_line, uparams_by_line, 
                                cont_params_full, cont_uparams_full,cont_profile_full=None,
                                distances=None,wavelength_grid= jnp.linspace(0, 20_000, 20_000),C_KMS=DEFAULT_C_KMS):
    
    #"amplitude", "vshift_kms", "fwhm_v_kms", "lambda0"
    
    #amps = 10**unumpy.uarray(params_by_line[:,:,0], uparams_by_line[:,:,0])
    amps = unumpy.uarray(params_by_line[:,:,0], uparams_by_line[:,:,0])
    #print(amps[0][0])
    centers = unumpy.uarray(params_by_line[:,:,1], uparams_by_line[:,:,1]) # centers => lambda0 * (1 + vshift_kms/c)
    #print(centers[0][0])
    shape_params = unumpy.uarray(params_by_line[:,:,2:], uparams_by_line[:,:,2:]) 
    shape_params = unumpy.uarray(params_by_line[:,:,2:], uparams_by_line[:,:,2:]) #* params_by_line[:,:,[1]])/C_KMS
    #print(shape_params[0][0])
    flux =  unumpy.uarray(*np.array(integrate_batch_with_error(profile_fn,wavelength_grid,params_by_line,uparams_by_line))) 
    #print("flujo",flux[0])
    fwhm = unumpy.uarray(*np.array(batch_fwhm(unumpy.nominal_values(amps), unumpy.nominal_values(centers), unumpy.nominal_values(shape_params),
                                                unumpy.std_devs(amps), unumpy.std_devs(centers), unumpy.std_devs(shape_params))))
    
    fwhm_kms = calc_fwhm_kms(fwhm, np.array(C_KMS), centers)
    cont_vals = unumpy.uarray(*np.array(evaluate_with_error(cont_profile_full, unumpy.nominal_values(centers), cont_params_full, unumpy.std_devs(centers), cont_uparams_full)))
    
    eqw = flux / cont_vals
    lum_vals = calc_luminosity(np.array(distances[:, None]), flux)

    return amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals
    
    
    