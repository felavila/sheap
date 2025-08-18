"""This class should concentrate all the routines for handle the params after the fit and after the sampling"""
__author__ = 'felavila'

__all__ = [
    "AfterFitParams",
]
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np 
import jax.numpy as jnp 
from jax import vmap
from auto_uncertainties import Uncertainty
from collections import defaultdict

from sheap.Profiles.profiles import PROFILE_LINE_FUNC_MAP,PROFILE_FUNC_MAP
from sheap.ComplexAfterFit.Samplers.utils.fwhm_conv import make_batch_fwhm_split,make_batch_fwhm_split_with_error
from sheap.Profiles.utils import make_integrator
from sheap.ComplexAfterFit.Samplers.utils.physicalfunctions import calc_fwhm_kms,calc_luminosity,calc_monochromatic_luminosity,calc_bolometric_luminosity,extra_params_functions
from sheap.ComplexAfterFit.Samplers.utils.afterfitprofilehelpers import integrate_batch_with_error,evaluate_with_error 
from sheap.ComplexAfterFit.Samplers.utils.combine_profiles import combine_components
from sheap.ComplexAfterFit.Samplers.utils.samplehandlers import pivot_and_split

#TODO add hyper parameter "raw" that gives exactly the params like dict params. 


def concat_dicts(list_of_dicts):
    out = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            out[k].append(v)

    # flatten or stack if numeric/array-like
    for k, v in out.items():
        #try:
        out[k] = np.concatenate([x for x in v]).T
        #except Exception:
         #   out[k] = v  # fallback if not array-like
    return dict(out)

class AfterFitParams:
    def __init__(self, afterclass: "ComplexAfterFit"):
        self.afterclass = afterclass
        self.model = afterclass.model
        self.c = afterclass.c
        self.dependencies = afterclass.dependencies
        self.scale = afterclass.scale
        self.fluxnorm = afterclass.fluxnorm
        self.spec = afterclass.spec
        self.mask = afterclass.mask
        self.d = afterclass.d
        
        
        self.names = afterclass.names 
        self.complex_class = afterclass.complex_class
        self.constraints = afterclass.constraints
        
        self.params_dict = afterclass.params_dict
        self.params = afterclass.params
        self.uncertainty_params = afterclass.uncertainty_params
        self.method = afterclass.method
        if not self.method:
            print("Not found method sampler")
            self.method = "single"
        
        self.BOL_CORRECTIONS = afterclass.BOL_CORRECTIONS
        self.SINGLE_EPOCH_ESTIMATORS = afterclass.SINGLE_EPOCH_ESTIMATORS
        self.wavelength_grid = jnp.linspace(0, 20_000, 20_000)
        self.LINES_TO_COMBINE = ["Halpha", "Hbeta","MgII","CIV"]
        self.limit_velocity = 150.
    
    def extract_basic_params(self,full_samples=None,idx_obj=None):
        #Add the filtering an separation of the params for params_single and the sample reduction for params_sampled
        if self.method == "single":
            return pivot_and_split(self.names,self._extract_basic_params_single())
        else:
            #print(full_samples.shape[0])
            return self._extract_basic_params_sampled(full_samples=full_samples,idx_obj=idx_obj)

    def _extract_basic_params_sampled(self, full_samples, idx_obj):
        """
        Extract line quantities (flux, FWHM, center, etc.) from posterior samples.
        Designed for use with MCMC or MC draws.
        """
        basic_params: Dict[str, Dict[str, np.ndarray]] = {}
        complexclass_group_by_region = self.complex_class.group_by("region")
        cont_group = complexclass_group_by_region["continuum"]
        idx_cont = cont_group.flat_param_indices_global
        cont_params = full_samples[:, idx_cont]
        distances = np.full((full_samples.shape[0],), self.d[idx_obj], dtype=np.float64)

        for region, region_group in complexclass_group_by_region.items():
            if region in ("fe", "continuum", "host"):
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

                    (
                        _line_names, _components, _flux, _fwhm, _fwhm_kms,
                        _centers, _amps, _eqw, _lum, _shapes
                    ) = self._accumulate_spaf_sampled(
                        prof_group, profile_fn, batch_fwhm, integrator, cont_params, full_samples
                    )

                else:
                    profile_fn = PROFILE_LINE_FUNC_MAP[profile_name]
                    batch_fwhm = make_batch_fwhm_split(profile_name)
                    integrator = make_integrator(profile_fn, method="vmap")

                    idxs = prof_group.flat_param_indices_global
                    params = full_samples[:, idxs]

                    _line_names = [l.line_name for l in prof_group.lines]
                    _components = [l.component for l in prof_group.lines]
                    params_by_line = params.reshape(params.shape[0], -1, profile_fn.n_params)

                    amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals = self._extract_sampled_profile_quantities(
                        profile_fn, integrator, batch_fwhm, params_by_line, cont_params, distances
                    )

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

            basic_params[region] = {"lines": line_names,
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

        # Monochromatic luminosities
        wl_i = self.spec[idx_obj, 0, :]
        mask_i = self.mask[idx_obj, :]
        L_w, L_bol,F_cont = {}, {},{}
        for wave in map(float, self.BOL_CORRECTIONS.keys()):
            wstr = str(int(wave))
            if (jnp.isclose(wl_i, wave, atol=2) & ~mask_i).any():
                Fcont = vmap(cont_group.combined_profile, in_axes=(None, 0))(jnp.array([wave]), cont_params).squeeze()
                Lmono = calc_monochromatic_luminosity(distances, Fcont, wave)
                Lbolval = calc_bolometric_luminosity(Lmono, self.BOL_CORRECTIONS[wstr])
                L_w[wstr], L_bol[wstr],F_cont[wstr] = np.array(Lmono), np.array(Lbolval), np.array(Fcont)
        if complexclass_group_by_region["fe"]:
            #i guess meanwhile MgII is not here it is not necesary run this ?
            group_fe = complexclass_group_by_region["fe"]
            combine_profile_fe = group_fe.combined_profile
            integrator_fe = make_integrator(combine_profile_fe, method="vmap")
            wavelength_grid_fe = jnp.linspace(2200,3090, 1_000) #?
            params_fe = self.params[:, idx_fe]
            flux_fe = integrator_fe(wavelength_grid_fe, params_fe)
            idx_fe = cont_group.flat_param_indices_global
            print(flux_fe)
        
        combined = combine_components(basic_params, cont_group, cont_params, distances,LINES_TO_COMBINE=self.LINES_TO_COMBINE,limit_velocity=self.limit_velocity,c=self.c,ucont_params=None)
        result = {"basic_params": basic_params, "L_w": L_w, "L_bol": L_bol,"F_cont":F_cont, "combine_params": combined}
        for k in ["basic_params","combine_params"]:
            if k == "basic_params":
                result_local = result[k]["broad"]
            else:
                result_local = result[k]
            #print(extra_params_functions(result_local,L_w,L_bol,self.SINGLE_EPOCH_ESTIMATORS,self.c))
            result.update({f"extra_{k}": extra_params_functions(result_local,L_w,L_bol,self.SINGLE_EPOCH_ESTIMATORS,self.c)})
        return result
    
    
    def _extract_basic_params_single(self):
        basic_params: Dict[str, Dict[str, np.ndarray]] = {}
        distances = self.d.copy()
        complexclass_group_by_region = self.complex_class.group_by("region")
        cont_group = complexclass_group_by_region["continuum"]
        idx_cont = cont_group.flat_param_indices_global
        cont_params = self.params[:, idx_cont]
        ucont_params = self.uncertainty_params[:, idx_cont]

        for region, region_group in complexclass_group_by_region.items():
            if region in ("fe", "continuum", "host"):
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

                    (_line_names, _components, _flux, _fwhm, _fwhm_kms,_centers, _amps, _eqw, _lum, _shapes) = self._accumulate_spaf_components(prof_group, profile_fn, batch_fwhm, cont_params, ucont_params)

                else:
                    profile_fn = PROFILE_LINE_FUNC_MAP[profile_name]
                    batch_fwhm = make_batch_fwhm_split_with_error(profile_name)

                    idxs = prof_group.flat_param_indices_global
                    _params = self.params[:, idxs]
                    _uparams = self.uncertainty_params[:, idxs]

                    _line_names = [l.line_name for l in prof_group.lines]
                    _components = [l.component for l in prof_group.lines]

                    params_by_line = _params.reshape(_params.shape[0], -1, profile_fn.n_params)
                    uparams_by_line = _uparams.reshape(_uparams.shape[0], -1, profile_fn.n_params)

                    amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals = self._extract_profile_quantities(
                        profile_fn, batch_fwhm, params_by_line, uparams_by_line, cont_params, ucont_params)

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
        for wave in map(float, self.BOL_CORRECTIONS.keys()):
            wstr = str(int(wave))
            hits = jnp.isclose(self.spec[:, 0, :], wave, atol=2)
            valid = np.array((hits & (~self.mask)).any(axis=1, keepdims=True))

            if any(valid):
                x = jnp.full((cont_params.shape[0], 1), wave)
                Fcont = Uncertainty(*np.array(
                    evaluate_with_error(cont_group.combined_profile, x, cont_params, jnp.zeros_like(x), ucont_params)
                )) * valid.astype(float)
                #print(valid)
                Lmono = calc_monochromatic_luminosity(np.array(distances[:, None]), Fcont, wave)
                Lbolval = calc_bolometric_luminosity(Lmono, self.BOL_CORRECTIONS[wstr])
                L_w[wstr], L_bol[wstr],F_cont[wstr] = Lmono, Lbolval,Fcont
        # if complexclass_group_by_region["fe"]:
        #     #i guess meanwhile MgII is not here it is not necesary run this ?
        #     group_fe = complexclass_group_by_region["fe"]
        #     idx_fe = cont_group.flat_param_indices_global
        #     combine_profile_fe = group_fe.combined_profile
        #     params_fe = self.params[:, idx_fe]
        #     uparams_fe = self.uncertainty_params[:, idx_fe]
        #     wavelength_grid_fe = jnp.linspace(2200,3090, 1_000) #?
        #     flux_fe =  Uncertainty(*np.array(integrate_batch_with_error(combine_profile_fe,wavelength_grid_fe,params_fe,uparams_fe))) 
        #     print(flux_fe)
        #from here can be the same function only take care on the uncertainty params of the continuum
        combined = combine_components(basic_params, cont_group, cont_params, distances,LINES_TO_COMBINE=self.LINES_TO_COMBINE,limit_velocity=self.limit_velocity,c=self.c,ucont_params=ucont_params)
        result = {"basic_params": basic_params, "L_w": L_w, "L_bol": L_bol,"F_cont":F_cont, "combine_params": combined}
        for k in ["basic_params","combine_params"]:
         #   print(k)
            if k == "basic_params":
                result_local = result[k]["broad"]
            else:
                result_local = result[k]
            result.update({f"extra_{k}": extra_params_functions(result_local,L_w,L_bol,self.SINGLE_EPOCH_ESTIMATORS,self.c)})
        return result
    
    def _accumulate_spaf_components(self, prof_group, profile_fn, batch_fwhm, cont_params, ucont_params):
        
        all_flux, all_fwhm, all_fwhm_kms = [], [], []
        all_centers, all_amps, all_eqws, all_lums = [], [], [], []
        all_line_names, all_components, all_shape_dicts = [], [], []
        #for sub_prof_gropu in 
        params_names = prof_group._master_param_names
        for sp,idx_params in zip(prof_group.lines,prof_group.global_profile_params_index_list,):
            params_by_line, uparams_by_line = self._build_spaf_param_matrices(sp,idx_params,params_names)
            
            amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals = self._extract_profile_quantities(profile_fn, batch_fwhm, params_by_line, uparams_by_line, cont_params, ucont_params)
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
                    
    def _build_spaf_param_matrices(self,sp,idx_params,params_names):
        
        full_params_by_line = []
        ufull_params_by_line = []
        _params = self.params[:, idx_params]
        _uncertainty_params = self.uncertainty_params[:, idx_params]
        names = np.array(params_names)[idx_params]
        
        amplitude_relations = sp.amplitude_relations
        amplitude_index = [i for i, name in enumerate(names) if "logamp" in name]
        ind_amplitude_index = {i[2] for i in amplitude_relations}
        dic_amp = {i: ii for i, ii in zip(ind_amplitude_index, amplitude_index)}
        idx_shift = max(amplitude_index) + 1
        for i,(_, factor, idx) in enumerate(amplitude_relations):
            amp = _params[:, [dic_amp[idx]]] + np.log10(factor)
            uamp = _uncertainty_params[:, [dic_amp[idx]]]
            center = sp.center[i] + _params[:, [idx_shift]]
            ucenter = _uncertainty_params[:, [idx_shift]]
            extras = _params[:, idx_shift+1:]
            uextras = _uncertainty_params[:, idx_shift+1:]
            full_params_by_line.append(np.column_stack([amp, center, extras]))
            ufull_params_by_line.append(np.column_stack([uamp, ucenter, uextras]))
        return np.moveaxis(np.array(full_params_by_line), 0, 1), np.moveaxis(np.array(ufull_params_by_line), 0, 1)

    def _extract_profile_quantities(self, profile_fn, batch_fwhm, params_by_line, uparams_by_line, cont_params, ucont_params):
        amps = 10**Uncertainty(params_by_line[:,:,0], uparams_by_line[:,:,0])
        centers = Uncertainty(params_by_line[:,:,1], uparams_by_line[:,:,1])
        shape_params = Uncertainty(params_by_line[:,:,2:], uparams_by_line[:,:,2:])
        flux =  Uncertainty(*np.array(integrate_batch_with_error(profile_fn,self.wavelength_grid,params_by_line,uparams_by_line))) 
        
        fwhm = Uncertainty(*np.array(batch_fwhm(amps.value, centers.value, shape_params.value,amps.error, centers.error, shape_params.error)))
        
        fwhm_kms = np.abs(calc_fwhm_kms(fwhm, np.array(self.c), centers))
        cont_vals = Uncertainty(*np.array(
            evaluate_with_error(self.complex_class.group_by("region")["continuum"].combined_profile,
                                centers.value, cont_params, centers.error, ucont_params)))
        
        eqw = flux / cont_vals
        lum_vals = calc_luminosity(np.array(self.d[:, None]), flux)

        return amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals
    
    
    
    ############SAMPLED###############################################
    def _accumulate_spaf_sampled(self, prof_group, profile_fn, batch_fwhm, integrator_fn, cont_params, full_samples):
        all_flux, all_fwhm, all_fwhm_kms = [], [], []
        all_centers, all_amps, all_eqws, all_lums = [], [], [], []
        all_line_names, all_components, all_shape_dicts = [], [], []
        params_names = prof_group._master_param_names
        
        for sp,idx_param in zip(prof_group.lines,prof_group.global_profile_params_index_list,):
            params_by_line = self._build_spaf_sampled_params(sp,idx_param,params_names,full_samples)
            amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals = self._extract_sampled_profile_quantities(
                profile_fn, integrator_fn, batch_fwhm, params_by_line, cont_params, np.full((full_samples.shape[0],), self.d[0])
            )

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
    def _build_spaf_sampled_params(self,sp,idx_param,params_names, full_samples):
    
        params = full_samples[:, idx_param]
        names = np.array(params_names)[idx_param]
        
        amplitude_relations = sp.amplitude_relations
        idx_pos = np.where(["logamp" in n for n in names])[0]
        amplitude_index = [i for i, name in enumerate(names) if "logamp" in name]
        ind_amplitude_index = {i[2] for i in amplitude_relations}
        dic_amp = {i: ii for i, ii in zip(ind_amplitude_index, amplitude_index)}
        idx_shift = idx_pos.max() + 1

        full_params_by_line = []
        for i,(_,factor,idx) in enumerate(amplitude_relations):
            amp = params[:, [dic_amp[idx]]] + np.log10(factor)
            center = (sp.center[i]+params[:,[idx_shift]])
            extras = (params[:,idx_shift+1:])
            full_params_by_line.append(np.column_stack([amp, center, extras]))

        return np.moveaxis(np.array(full_params_by_line), 0, 1)
    
    def _extract_sampled_profile_quantities(self, profile_fn, integrator_fn, batch_fwhm, params_by_line, cont_params, distances):
        amps = 10**params_by_line[:, :, 0]
        centers = params_by_line[:, :, 1]
        shape_params = jnp.abs(params_by_line[:, :, 2:])

        flux = integrator_fn(self.wavelength_grid, params_by_line)
        fwhm = batch_fwhm(amps, centers, shape_params)
        fwhm_kms = jnp.abs(calc_fwhm_kms(fwhm, self.c, centers))

        cont_vals = vmap(self.complex_class.group_by("region")["continuum"].combined_profile, in_axes=(0, 0))(centers, cont_params)
        eqw = flux / cont_vals
        lum_vals = calc_luminosity(distances[:, None], flux)

        return amps, centers, shape_params, flux, fwhm, fwhm_kms, eqw, lum_vals

