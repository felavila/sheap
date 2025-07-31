"""This class should concentrate all the routines for handle the params after the fit and after the sampling"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np 
import jax.numpy as jnp 
from jax import vmap
from auto_uncertainties import Uncertainty

from sheap.Profiles.profiles import PROFILE_LINE_FUNC_MAP,PROFILE_FUNC_MAP
from sheap.ComplexAfterFit.Samplers.utils.fwhm_conv import make_batch_fwhm_split,make_batch_fwhm_split_with_error
from sheap.Profiles.utils import make_integrator
from sheap.ComplexAfterFit.Samplers.utils.physicalfunctions import calc_fwhm_kms,calc_flux,calc_luminosity,calc_monochromatic_luminosity,calc_bolometric_luminosity
from sheap.ComplexAfterFit.Samplers.utils.afterfitprofilehelpers import integrate_batch,evaluate_with_error 
from sheap.ComplexAfterFit.Samplers.utils.combine_profiles import combine_fast


def combine_components(basic_params,cont_group,cont_params,d,LINES_TO_COMBINE= ["Halpha", "Hbeta"],limit_velocity=150.0,c= 299792.458):
    combined = {}
    for line in LINES_TO_COMBINE:
        broad_lines = basic_params["broad"]["lines"]
        narrow_lines = basic_params["narrow"]["lines"]
        idx_broad = [i for i, L in enumerate(broad_lines) if L.lower() == line.lower()]
        idx_narrow = [i for i, L in enumerate(narrow_lines) if L.lower() == line.lower()]
        if len(idx_broad) >= 2 and len(idx_narrow) == 1:
            #N = self.params.shape[0] #the code already has "basic_params"
            #broad components
            amp_b = basic_params["broad"]["amplitude"][:, idx_broad].value
            mu_b = basic_params["broad"]["center"][:, idx_broad].value
            fwhm_kms_b = basic_params["broad"]["fwhm_kms"][:, idx_broad].value
            params_broad = jnp.stack([amp_b, mu_b, fwhm_kms_b], axis=-1).reshape(N, -1)
            #Narrow line 
            amp_n = basic_params["narrow"]["amplitude"][:, idx_narrow].value
            mu_n = basic_params["narrow"]["center"][:, idx_narrow].value
            fwhm_kms_n = basic_params["narrow"]["fwhm_kms"][:, idx_narrow].value
            params_narrow = jnp.concatenate([amp_n, mu_n, fwhm_kms_n], axis=1)
            fwhm_c, amp_c, mu_c = combine_fast(params_broad, params_narrow,limit_velocity=limit_velocity, c=c)
            fwhm_A = (fwhm_c / c) * mu_c
            flux_c = calc_flux(np.array(amp_c), np.array(fwhm_A))
            cont_c = vmap(cont_group.combined_profile)(mu_c, cont_params)
            L_line = calc_luminosity(np.array(d), flux_c) # x.x
            eqw_c = flux_c / cont_c
            print(line,amp_c)
            combined[line] = {
                "amplitude": np.array(amp_c),
                "center": np.array(mu_c),
                "fwhm_kms": np.array(fwhm_c),
                "fwhm": np.array(fwhm_A),
                "flux": np.array(flux_c),
                "luminosity": np.array(L_line),
                "eqw": np.array(eqw_c),
                "component":  np.array(basic_params["broad"]["component"])[idx_broad]}
    return  combined
                
class AfterFitParams:
    def __init__(self, estimator: "ComplexAfterFit"):
        self.estimator = estimator
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
        self.method = estimator.method
        if not self.method:
            print("Not found method sampler")
            self.method = "single"
        self.wavelength_grid = jnp.linspace(0, 20_000, 20_000)
        self.LINES_TO_COMBINE = ["Halpha", "Hbeta"]
        self.limit_velocity = 150.
    #not quite independent of the method.
    def _extract_basic_params_sampled(self,full_samples,idx_obj):
        """
        Extract continuum‐corrected flux, FWHM, FWHM (km/s), center, amplitude,
        equivalent width and luminosity for each emission line, grouped by 'region'.
        Returns a dict mapping each region → dict with keys:
        'lines', 'component',
        'flux', 'fwhm', 'fwhm_kms',
        'center', 'amplitude',
        'eqw', 'luminosity'
        outside this class all have to be already in flux units.
        This function will work only for samples
        """
        basic_params: Dict[str, Dict[str, np.ndarray]] = {}
        complexclass_group_by_region = self.complex_class.group_by("region")
        cont_group = complexclass_group_by_region["continuum"]
        idx_cont   = cont_group.flat_param_indices_global
        cont_params= full_samples[:, idx_cont]
        distances = np.full((full_samples.shape[0],), self.d[idx_obj], dtype=np.float64)
        for region, region_group in complexclass_group_by_region.items():
            if region in ("fe", "continuum","host"):
                continue
            line_names, components = [], []
            flux_parts, fwhm_parts = [], []
            fwhm_kms_parts, center_parts = [], []
            amp_parts, eqw_parts, lum_parts = [], [], []
            shape_params_list = [] #just to save the params 
            region_group_group_by_profile = region_group.group_by("profile_name")
            print(region_group_group_by_profile.keys())
            for profile_name, prof_group in region_group_group_by_profile.items():
                print(profile_name)
                if "_" in profile_name:
                    _, subprof = profile_name.split("_", 1)
                    profile_fn = PROFILE_LINE_FUNC_MAP[subprof]
                    batch_fwhm = make_batch_fwhm_split(subprof)
                    integrator = make_integrator(profile_fn, method="vmap")
                    for sp, idx_param in zip(prof_group.lines, prof_group.global_profile_params_index_list):
                        params      = full_samples[:, idx_param]
                        names       = np.array(prof_group._master_param_names)[idx_param]
                        amplitude_relations = sp.amplitude_relations
                        idx_pos     = np.where(["logamp" in n for n in names])[0]
                        amplitude_index = [nx for nx,_ in  enumerate(names) if "logamp" in _ ]
                        ind_amplitude_index = {i[2] for i in amplitude_relations}
                        dic_amp = {i:ii for i,ii in (zip(ind_amplitude_index,amplitude_index))}
                        idx_shift   = idx_pos.max() + 1
                        full_params_by_line = []
                        for i,(_,factor,idx) in enumerate(amplitude_relations):
                            amplitude_line = params[:,[dic_amp[idx]]] + np.log10(factor)
                            center_line = (sp.center[i]+params[:,[idx_shift]])
                            extra_params_profile = (params[:,idx_shift+1:])
                            full_params_by_line.append(np.column_stack([amplitude_line, center_line, extra_params_profile]))
                        _lines_names = sp.region_lines
                        params_by_line = np.moveaxis(np.array(full_params_by_line),0,1)
                        _components = [sp.component]*params_by_line.shape[1]
                        amps     = 10**params_by_line[:,:,0]
                        centers  = params_by_line[:,:,1]
                        shape_params     = jnp.atleast_3d(jnp.abs(params_by_line[:,:,2:]))#check if this is correct)
                        flux     = integrator(self.wavelength_grid, params_by_line)
                        fwhm = batch_fwhm(amps, centers, shape_params)
                        fwhm_kms = jnp.abs(calc_fwhm_kms(fwhm, self.c, centers))
                        cont_vals= vmap(cont_group.combined_profile, in_axes=(0,0))(centers, cont_params)#this is the same for both ?
                        lum_vals = calc_luminosity(distances[:, None], flux)
                        eqw      = flux / cont_vals
                
                        line_names.extend(_lines_names)
                        components.extend(_components)
                        flux_parts.append(np.array(flux))
                        fwhm_parts.append(np.array(fwhm))
                        fwhm_kms_parts.append(np.array(fwhm_kms))
                        center_parts.append(np.array(centers))
                        amp_parts.append(np.array(amps))
                        eqw_parts.append(np.array(eqw))
                        lum_parts.append(np.array(lum_vals))
                        shape_params_list.append({k:v for k,v in zip(profile_fn.param_names[2:],shape_params)})#check 
                
                
                else:
                    profile_fn = PROFILE_LINE_FUNC_MAP[profile_name]
                    batch_fwhm = make_batch_fwhm_split(profile_name)
                    integrator = make_integrator(profile_fn, method="vmap")
                    idxs       = prof_group.flat_param_indices_global
                    params     = full_samples[:, idxs]
                    _lines_names = [l.line_name for l in prof_group.lines] # this is true?
                    _components = [l.component for l in prof_group.lines]
                    params_by_line = params.reshape(params.shape[0], -1, profile_fn.n_params)
                    #####################################
                    amps     = 10**params_by_line[:,:,0]
                    centers  = params_by_line[:,:,1]
                    shape_params     = jnp.atleast_3d(jnp.abs(params_by_line[:,:,2:]))#check if this is correct)
                    flux     = integrator(self.wavelength_grid, params_by_line)
                    fwhm = batch_fwhm(amps, centers, shape_params)
                    fwhm_kms = jnp.abs(calc_fwhm_kms(fwhm, self.c, centers))
                    cont_vals= vmap(cont_group.combined_profile, in_axes=(0,0))(centers, cont_params)#this is the same for both ?
                    lum_vals = calc_luminosity(distances[:, None], flux)
                    eqw      = flux / cont_vals
                    
                    line_names.extend(_lines_names)
                    components.extend(_components)
                    flux_parts.append(np.array(flux))
                    fwhm_parts.append(np.array(fwhm))
                    fwhm_kms_parts.append(np.array(fwhm_kms))
                    center_parts.append(np.array(centers))
                    amp_parts.append(np.array(amps))
                    eqw_parts.append(np.array(eqw))
                    lum_parts.append(np.array(lum_vals))
                    shape_params_list.append({k:v for k,v in zip(profile_fn.param_names[2:],shape_params)})#check 
            #maybe could be a good idea add profile.
            basic_params[region] = {
            "lines":      line_names,
            "component":  components,
            "flux":       np.concatenate(flux_parts,     axis=1),
            "fwhm":       np.concatenate(fwhm_parts,     axis=1),
            "fwhm_kms":   np.concatenate(fwhm_kms_parts, axis=1),
            "center":     np.concatenate(center_parts,    axis=1),
            "amplitude":  np.concatenate(amp_parts,       axis=1),
            "eqw":        np.concatenate(eqw_parts,       axis=1),
            "luminosity": np.concatenate(lum_parts,       axis=1),
            "shape_params":shape_params_list}
        wl_i= self.spec[idx_obj,0,:]
        mask_i= self.mask[idx_obj,:]
        L_w, L_bol = {}, {}
        for wave in map(float, self.BOL_CORRECTIONS.keys()):
            wstr = str(int(wave))
            if (jnp.isclose(wl_i, wave, atol=1) & ~mask_i).any():
                Fcont   = vmap(cont_group.combined_profile, in_axes=(None, 0))(jnp.array([wave]), cont_params).squeeze()
                Lmono   = calc_monochromatic_luminosity(distances, Fcont, wave)
                Lbolval = calc_bolometric_luminosity(Lmono, self.BOL_CORRECTIONS[wstr])
                L_w[wstr], L_bol[wstr] = np.array(Lmono), np.array(Lbolval)    
        combined = combine_components(basic_params,cont_group,cont_params,distances,LINES_TO_COMBINE=self.LINES_TO_COMBINE,limit_velocity=self.limit_velocity,c=self.c)
        return  {"basic_params":basic_params,"L_w":L_w,"L_bol":L_bol,"combined":combined}  
    
    def _extract_basic_params_single(self):
        basic_params: Dict[str, Dict[str, np.ndarray]] = {}
        distances = self.d
        complexclass_group_by_region = self.complex_class.group_by("region")
        cont_group = complexclass_group_by_region["continuum"]
        idx_cont   = cont_group.flat_param_indices_global
        cont_params = self.params[:, idx_cont]
        ucont_params = self.uncertainty_params[:, idx_cont]
        for region, region_group in complexclass_group_by_region.items():
            if region in ("fe", "continuum","host"):
                continue
            line_names, components = [], []
            flux_parts, fwhm_parts = [], []
            fwhm_kms_parts, center_parts = [], []
            amp_parts, eqw_parts, lum_parts = [], [], []
            shape_params_list = [] #just to save the params 
            region_group_group_by_profile = region_group.group_by("profile_name")
            for profile_name, prof_group in region_group_group_by_profile.items():
                if "_" in profile_name:
                    _, subprof = profile_name.split("_", 1)
                    profile_fn = PROFILE_LINE_FUNC_MAP[subprof]
                    batch_fwhm = make_batch_fwhm_split_with_error(subprof)
                    for sp, idx_params in zip(prof_group.lines, prof_group.global_profile_params_index_list):
                        _params      = self.params[:, idx_params]
                        _uncertainty_params = self.uncertainty_params[:, idx_params]
                        names       = np.array(prof_group._master_param_names)[idx_params]
                        amplitude_relations = sp.amplitude_relations
                        idx_pos     = np.where(["logamp" in n for n in names])[0]
                        amplitude_index = [nx for nx,_ in  enumerate(names) if "logamp" in _ ]
                        ind_amplitude_index = {i[2] for i in amplitude_relations}
                        dic_amp = {i:ii for i,ii in (zip(ind_amplitude_index,amplitude_index))}
                        idx_shift   = idx_pos.max() + 1
                        full_params_by_line = []
                        ufull_params_by_line = []
                        for i,(_,factor,idx) in enumerate(amplitude_relations):
                            amplitude_line = _params[:,[dic_amp[idx]]] + np.log10(factor)
                            uamplitude_line =_uncertainty_params[:, dic_amp[idx]]
                            center_line = (sp.center[i]+_params[:,[idx_shift]])
                            ucenter_line = _uncertainty_params[:,[idx_shift]]
                            extra_params_profile = (_params[:,idx_shift+1:])
                            uextra_params_profile = (_uncertainty_params[:,idx_shift+1:])
                            full_params_by_line.append(np.column_stack([amplitude_line, center_line, extra_params_profile]))
                            ufull_params_by_line.append(np.column_stack([uamplitude_line, ucenter_line, uextra_params_profile]))
                    params_by_line = np.moveaxis(np.array(full_params_by_line),0,1)
                    uparams_by_line = np.moveaxis(np.array(ufull_params_by_line),0,1)  
                    _lines_names = sp.region_lines
                    _components = [sp.component]*params_by_line.shape[1]
                
                else:
                    profile_fn = PROFILE_LINE_FUNC_MAP[profile_name]
                    batch_fwhm = make_batch_fwhm_split(profile_name)
                    idxs       = prof_group.flat_param_indices_global
                    _params     = self.params[:, idxs]
                    _uparams     = self.uncertainty_params[:, idxs]
                    _lines_names = [l.line_name for l in prof_group.lines] # this is true?
                    _components = [l.component for l in prof_group.lines]
                    params_by_line = _params.reshape(_params.shape[0], -1, profile_fn.n_params)
                    uparams_by_line = _uparams.reshape(_params.shape[0], -1, profile_fn.n_params)
                
                amps = 10**Uncertainty(params_by_line[:,:,0],uparams_by_line[:,:,0])
                centers = Uncertainty(params_by_line[:,:,1],uparams_by_line[:,:,1])
                shape_params = Uncertainty(params_by_line[:,:,2:],uparams_by_line[:,:,2:])
                flux     = Uncertainty(*np.array(integrate_batch(profile_fn,self.wavelength_grid,params_by_line,uparams_by_line))) #this can be combine with make_integrator i presume
                fwhm =  Uncertainty(*np.array(batch_fwhm(amps.value, centers.value, shape_params.value,amps.error, centers.error, shape_params.error))) #here we assume an error = 0?
                fwhm_kms = np.abs(calc_fwhm_kms(fwhm, np.array(self.c), centers))
                cont_vals   = Uncertainty(*np.array(evaluate_with_error(cont_group.combined_profile,centers.value,cont_params,centers.error,ucont_params)))
                lum_vals = calc_luminosity(np.array(distances[:, None]), flux) #mmm
                eqw      = flux / cont_vals
                line_names.extend(_lines_names)
                components.extend(_components)
                flux_parts.append(flux)
                fwhm_parts.append(fwhm)
                fwhm_kms_parts.append(fwhm_kms)
                center_parts.append(centers)
                amp_parts.append(amps)
                eqw_parts.append(eqw)
                lum_parts.append(lum_vals)
                shape_params_list.append({k:v for k,v in zip(profile_fn.param_names[2:],shape_params.T)})#check 
            basic_params[region] = {
                "lines":      line_names,
                "component":  components,
                "flux":       np.concatenate(flux_parts,     axis=1),
                "fwhm":       np.concatenate(fwhm_parts,     axis=1),
                "fwhm_kms":   np.concatenate(fwhm_kms_parts, axis=1),
                "center":     np.concatenate(center_parts,    axis=1),
                "amplitude":  np.concatenate(amp_parts,       axis=1),
                "eqw":        np.concatenate(eqw_parts,       axis=1),
                "luminosity": np.concatenate(lum_parts,       axis=1),
                "shape_params":shape_params_list}

        L_w, L_bol = {}, {}
        for wave in map(float, self.BOL_CORRECTIONS.keys()):
            wstr = str(int(wave))
            hits = jnp.isclose(self.spec[:, 0, :], wave, atol=1) #takes all the possible arrays
            valid = np.array((hits & (~self.mask)).any(axis=1, keepdims=True))
            if any(valid):
                x = jnp.full((cont_params.shape[0],1), wave)
                Fcont   = Uncertainty(*np.array(evaluate_with_error(cont_group.combined_profile,x,cont_params,jnp.zeros_like(x),ucont_params))) * valid.astype(float)
                Lmono   = calc_monochromatic_luminosity(np.array(distances[:, None]), Fcont, wave)
                Lbolval = calc_bolometric_luminosity(Lmono, self.BOL_CORRECTIONS[wstr])
                L_w[wstr], L_bol[wstr] = Lmono, Lbolval
        combined = combine_components(basic_params,cont_group,cont_params,distances,LINES_TO_COMBINE=self.LINES_TO_COMBINE,limit_velocity=self.limit_velocity,c=self.c)
        return {"basic_params":basic_params,"L_w":L_w,"L_bol":L_bol,"combined":combined}  
        
    def extract_basic_params(self,full_samples=None,idx_obj=None):
        print("method",self.method)
        if self.method == "single":
            return self._extract_basic_params_single()
        else:
            print(full_samples.shape[0])
            return self._extract_basic_params_sampled(full_samples=full_samples,idx_obj=idx_obj)
