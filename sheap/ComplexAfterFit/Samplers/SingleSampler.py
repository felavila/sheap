"""This module handles basic operations."""
__author__ = 'felavila'

__all__ = [
    "SingleSampler",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax.numpy as jnp
from jax import vmap
import numpy as np
import pandas as pd
import yaml
from auto_uncertainties import Uncertainty


from sheap.Profiles.profiles import PROFILE_LINE_FUNC_MAP,PROFILE_FUNC_MAP
from sheap.ComplexAfterFit.Samplers.utils.afterfitprofilehelpers import integrate_batch,evaluate_with_error 
from sheap.ComplexAfterFit.Samplers.utils.samplehandlers import pivot_and_split
from sheap.ComplexAfterFit.Samplers.utils.physicalfunctions import extra_params,calc_flux,calc_luminosity#mmmm
from sheap.ComplexAfterFit.Samplers.utils.combine_profiles import combine_fast
#all of this have to go to profiles.
#from .tools.functions import calc_flux,calc_fwhm_kms,calc_luminosity,calc_monochromatic_luminosity,calc_bolometric_luminosity,calc_black_hole_mass

class SingleSampler:
    # TODO big how to combine distributions
    def __init__(self, estimator: "ParameterEstimation"):
        
        self.estimator = estimator  # ParameterEstimation instance
        self.model = estimator.model
        self.c = estimator.c
        self.dependencies = estimator.dependencies
        self.scale = estimator.scale
        self.fluxnorm = estimator.fluxnorm
        self.spec = estimator.spec
        self.mask = estimator.mask
        self.d = estimator.d
        
        
        self.BOL_CORRECTIONS = estimator.BOL_CORRECTIONS
        self.SINGLE_EPOCH_ESTIMATORS = estimator.SINGLE_EPOCH_ESTIMATORS
        self.names = estimator.names 
        self.complex_class = estimator.complex_class
        self.constraints = estimator.constraints
         
        self.params_dict = estimator.params_dict
        self.params = estimator.params
        self.uncertainty_params = estimator.uncertainty_params
    
    def extract_basic_line_parameters(self):
        wavelength_grid: jnp.ndarray = jnp.linspace(0, 20_000, 20_000)
        region_group = self.complex_class.group_by("region")
        cont_group = region_group["continuum"]
        cont_idx   = cont_group.flat_param_indices_global
        cont_params = self.params[:, cont_idx] #(n,n_params)
        ucont_params= self.uncertainty_params[:, cont_idx] #(n,n_params)
        basic_params: Dict[str, Dict[str, np.ndarray]] = {}
        for kind, kind_group in region_group.items():
            if kind in ("fe", "continuum","host"):
                continue
            line_names, components = [], []
            flux_parts, fwhm_parts = [], []
            fwhm_kms_parts, center_parts = [], []
            amp_parts, eqw_parts, lum_parts = [], [], []
            for profile_name, prof_group in kind_group.group_by("profile_name").items():
                if "_" in profile_name:
                    _, subprof = profile_name.split("_", 1)
                    profile_fn = PROFILE_FUNC_MAP[subprof]
                    for sp, param_idxs in zip(prof_group.lines, prof_group.global_profile_params_index_list):
                        #print(param_idxs)
                        #fwhm = Uncertainty(fwhm, np.abs(fwhm_u))
                        _params      = self.params[:, param_idxs]
                        _uncertainty_params      = self.uncertainty_params[:, param_idxs]
                        names       = np.array(prof_group._master_param_names)[param_idxs]
                        amplitude_relations = sp.amplitude_relations
                        #print(_params.shape,_uncertainty_params.shape)
                        amp_pos     = np.where(["logamp" in n for n in names])[0]
                        amplitude_index = [nx for nx,_ in  enumerate(names) if "logamp" in _ ]
                        ind_amplitude_index = {i[2] for i in amplitude_relations}
                        dic_amp = {i:ii for i,ii in (zip(ind_amplitude_index,amplitude_index))}
                        shift_idx   = amp_pos.max() + 1
                        full_params_by_line = []
                        ufull_params_by_line = []
                        for i,(_,factor,ids) in enumerate(amplitude_relations):
                            amplitude_line = _params[:, dic_amp[ids]] + np.log10(factor)
                            u_amplitude_line =_uncertainty_params[:, dic_amp[ids]]
                            #print(u_amplitude_line)
                            center_line = (sp.center[i]+ _params[:,[shift_idx]])
                            u_center_line = _uncertainty_params[:,[shift_idx]]
                            extra_params_profile = (_params[:,shift_idx+1:])
                            u_extra_params_profile = (_uncertainty_params[:,shift_idx+1:])
                            full_params_by_line.append(np.column_stack([amplitude_line, center_line, extra_params_profile]))
                            ufull_params_by_line.append(np.column_stack([u_amplitude_line, u_center_line, u_extra_params_profile]))
                    
                        params_by_line = np.array(full_params_by_line)
                        _uncertainty_params_by_line =np.array(ufull_params_by_line)
                        params_by_line = np.moveaxis(params_by_line,0,1)
                        _uncertainty_params_by_line = np.moveaxis(_uncertainty_params_by_line,0,1)
                        
                        
                        amps        = 10**Uncertainty(params_by_line[:,:,0],_uncertainty_params_by_line[:,:,0])
                        centers     = Uncertainty(params_by_line[:,:,1],_uncertainty_params_by_line[:,:,1])
                        fwhm = Uncertainty(params_by_line[:,:,2:],_uncertainty_params_by_line[:,:,2:])
                        flux = Uncertainty(*np.array(integrate_batch(profile_fn,wavelength_grid,params_by_line,_uncertainty_params_by_line)))
                        #TODO ADD the batch to handle other profiles in the calculus and the posibility of propagate the error.
                        fwhm_kms = (fwhm.squeeze() * self.c) / centers
                        #print("x:",centers.value.shape,"params:",cont_params.shape)
                        cont_vals   = Uncertainty(*np.array(evaluate_with_error(cont_group.combined_profile,centers.value,cont_params,centers.error,ucont_params)))
                        lum_vals    =  4.0 * np.pi * np.array(self.d)[:,None]**2 * flux #* centers
                        eqw        = flux / cont_vals
                        print(sp.component)
                        nsub = flux.shape[1]
                        line_names.extend(sp.region_lines)
                        components.extend([sp.component]*nsub)
                        flux_parts.append(flux)
                        fwhm_parts.append(fwhm)
                        fwhm_kms_parts.append(fwhm_kms)
                        center_parts.append(centers)
                        amp_parts.append(amps)
                        eqw_parts.append(eqw)
                        lum_parts.append(lum_vals)
                
                else:
                    profile_fn = PROFILE_FUNC_MAP[profile_name]
                    param_idxs       = prof_group.flat_param_indices_global
                    _params      = self.params[:, param_idxs]
                    _uncertainty_params      = self.uncertainty_params[:, param_idxs]
                    names      = list(prof_group.params_dict.keys())
                    params_by_line = _params.reshape(_params.shape[0], -1, profile_fn.n_params)
                    uncertainty_params_by_line = _uncertainty_params.reshape(_uncertainty_params.shape[0], -1, profile_fn.n_params)
                    flux     = Uncertainty(*np.array(integrate_batch(profile_fn,wavelength_grid,params_by_line,uncertainty_params_by_line)))
                    fwhm = Uncertainty(params_by_line[:,:,2:],uncertainty_params_by_line[:,:,2:])
                    centers     = Uncertainty(params_by_line[:,:,1],uncertainty_params_by_line[:,:,1])
                    amps        = 10**Uncertainty(params_by_line[:,:,0],uncertainty_params_by_line[:,:,0])
                    #TODO ADD the batch to handle other profiles in the calculus and the posibility of propagate the error.
                    fwhm_kms = (fwhm.squeeze() * self.c) / centers
                    #print("x:",centers.value.shape,"params:",cont_params.shape)
                    cont_vals   = Uncertainty(*np.array(evaluate_with_error(cont_group.combined_profile,centers.value,cont_params,centers.error,ucont_params)))
                    lum_vals    =  4.0 * np.pi * np.array(self.d)[:,None]**2 * flux #* centers
                    eqw        = flux / cont_vals
                   

                    line_names.extend([l.line_name for l in prof_group.lines])
                    components.extend([l.component for l in prof_group.lines])
                    flux_parts.append(flux)
                    fwhm_parts.append(fwhm)
                    fwhm_kms_parts.append(fwhm_kms)
                    center_parts.append(centers)
                    amp_parts.append(amps)
                    eqw_parts.append(eqw)
                    lum_parts.append(lum_vals)

            basic_params[kind] = {
                "lines":      line_names,
                "component":  components,
                "flux":       np.concatenate(flux_parts,     axis=1),
                "fwhm":       np.concatenate(fwhm_parts,     axis=1),
                "fwhm_kms":   np.concatenate(fwhm_kms_parts, axis=1),
                "center":     np.concatenate(center_parts,    axis=1),
                "amplitude":  np.concatenate(amp_parts,       axis=1),
                "eqw":        np.concatenate(eqw_parts,       axis=1),
                "luminosity": np.concatenate(lum_parts,       axis=1),
            }
        return basic_params
    
    def posterior_physical_parameters(self,extra_products=True,combine_components=True,LINES_TO_COMBINE = ["Halpha", "Hbeta"],limit_velocity=150.0):
        #this have to be the same? 
        basic_params =  self.extract_basic_line_parameters()
        result = {"basic_params":basic_params}
        if not extra_products:
            return pivot_and_split(self.names,result)
        region_group = self.complex_class.group_by("region")
        cont_group = region_group["continuum"]
        cont_idx   = cont_group.flat_param_indices_global
                        #
        cont_params= self.params[:, cont_idx] #(n,n_params)
        ucont_params= self.uncertainty_params[:, cont_idx] #(n,n_params)
        if combine_components and "broad" in basic_params and "narrow" in basic_params:
            combined = {}
            lines_combined = []
            for line in LINES_TO_COMBINE:
                broad_lines = basic_params["broad"]["lines"]
                narrow_lines = basic_params["narrow"]["lines"]

                idx_broad = [i for i, L in enumerate(broad_lines) if L.lower() == line.lower()]
                idx_narrow = [i for i, L in enumerate(narrow_lines) if L.lower() == line.lower()]

                if len(idx_broad) >= 2 and len(idx_narrow) == 1:
                    N = self.params.shape[0]

                    # Broad
                    amp_b = basic_params["broad"]["amplitude"][:, idx_broad].value
                    mu_b = basic_params["broad"]["center"][:, idx_broad].value
                    fwhm_kms_b = basic_params["broad"]["fwhm_kms"][:, idx_broad].value

                    params_broad = jnp.stack([amp_b, mu_b, fwhm_kms_b], axis=-1).reshape(N, -1)

                    # Narrow
                    amp_n = basic_params["narrow"]["amplitude"][:, idx_narrow].value
                    mu_n = basic_params["narrow"]["center"][:, idx_narrow].value
                    fwhm_kms_n = basic_params["narrow"]["fwhm_kms"][:, idx_narrow].value
                    params_narrow = jnp.concatenate([amp_n, mu_n, fwhm_kms_n], axis=1)

                    fwhm_c, amp_c, mu_c = combine_fast(
                        params_broad, params_narrow,
                        limit_velocity=limit_velocity, c=self.c
                    )

                    fwhm_A = (fwhm_c / self.c) * mu_c
                    flux_c = calc_flux(np.array(amp_c), np.array(fwhm_A))
                    cont_c = vmap(cont_group.combined_profile)(mu_c, cont_params)

                    L_line = calc_luminosity(np.array(self.d), flux_c) # x.x
                    eqw_c = flux_c / cont_c
                    print(line,amp_c)
                    #print(np.array(basic_params["broad"]["component"])[idx_broad])
                    combined[line] = {
                        "amplitude": np.array(amp_c),
                        "center": np.array(mu_c),
                        "fwhm_kms": np.array(fwhm_c),
                        "fwhm": np.array(fwhm_A),
                        "flux": np.array(flux_c),
                        "luminosity": np.array(L_line),
                        "eqw": np.array(eqw_c),
                        "component":  np.array(basic_params["broad"]["component"])[idx_broad]
                    }

                    lines_combined.append(line)
            if combined:
                result.update({'combined': combined})
        L_w, L_bol = {}, {}
        for wave in map(float, self.BOL_CORRECTIONS.keys()):
            wstr = str(int(wave))
            hits = jnp.isclose(self.spec[:, 0, :], wave, atol=1)
            valid = np.array((hits & (~self.mask)).any(axis=1, keepdims=True))
            if any(valid):
                #print(cont_params.shape[0])
                x = jnp.full((cont_params.shape[0],1), wave)
                #print(x.shape,cont_params.shape)
                flux_cont   = Uncertainty(*np.array(evaluate_with_error(cont_group.combined_profile,x,cont_params,jnp.zeros_like(x),ucont_params))) * valid.astype(float)
                Lmono   =  (np.array(x).squeeze() * 4.0 * np.pi * np.array(self.d**2).squeeze() * flux_cont.squeeze())
                #print(Lmono.shape,"aja")
                Lbolval = Lmono*self.BOL_CORRECTIONS[wstr]
                L_w[wstr], L_bol[wstr] = Lmono, Lbolval
        
        broad_params = basic_params.get("broad")
        if broad_params:
            extras = extra_params(broad_params,L_w,L_bol,self.SINGLE_EPOCH_ESTIMATORS,self.c,False)
            result.update({"extras_params":extras})
            if combined:
                extras_comb = extra_params(combined,L_w,L_bol,self.SINGLE_EPOCH_ESTIMATORS,self.c,True)
                combined.update({"extras_params":extras_comb})
        result.update({"L_w":L_w,"L_bol":L_bol})
        return  pivot_and_split(self.names,result)
