from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax.numpy as jnp
from jax import vmap
import numpy as np
import pandas as pd
import yaml
from auto_uncertainties import Uncertainty


from sheap.Profiles.profiles import PROFILE_LINE_FUNC_MAP,PROFILE_FUNC_MAP
from sheap.Posterior.utils import integrate_function_error,integrate_batch,batched_evaluate,evaluate_with_error,pivot_and_split 

#all of this have to go to profiles.
#from .tools.functions import calc_flux,calc_fwhm_kms,calc_luminosity,calc_monochromatic_luminosity,calc_bolometric_luminosity,calc_black_hole_mass





class ParametersSingle:
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
        cont_params= self.params[:, cont_idx] #(n,n_params)
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
    
    def posterior_physical_parameters(self,extra_products=True):
        
        basic_params =  self.extract_basic_line_parameters()
        if not extra_products:
            result = {"basic_params":basic_params}
        else:
            #TODO: add combination rutine 
            region_group = self.complex_class.group_by("region")
            cont_group = region_group["continuum"]
            cont_idx   = cont_group.flat_param_indices_global
            #
            cont_params= self.params[:, cont_idx] #(n,n_params)
            ucont_params= self.uncertainty_params[:, cont_idx] #(n,n_params)
            #
            #cont_fun   = cont_group.combined_profile
            
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
            extras = {}
            if broad_params:
                fwhm_kms_all = broad_params.get("fwhm_kms")
                lum_all = broad_params.get("luminosity")
                line_list    = np.array(broad_params.get("lines", []))
                for line_name, est in self.SINGLE_EPOCH_ESTIMATORS.items():
                    lam  = est["wavelength"]
                    wstr = str(int(lam))
                    if line_name not in line_list or wstr not in L_w:
                        continue
                    idxs    = np.where(line_list == line_name)[0]
                    fkm     = fwhm_kms_all[:, idxs].squeeze()   # (N,) or (N,1)
                    Lmono   = L_w[wstr].squeeze()                        # (N,)
                    Lbolval = L_bol[wstr].squeeze()                       # (N,)
                    #if fkm.ndim == 2:
                     #   Lmono   = Lmono[..., None]
                      #  Lbolval = Lbolval[..., None]
                    a, b, f = est["a"], est["b"], est["f"]
                    
                    log_L = np.log10(Lmono)
                    #print(a, b, f,log_L[0],fkm[0])
                    #np.atleast_2d(np.log10(L_w)).T
                    
                    log_FWHM = np.log10(fkm) - 3  # FWHM to 10^3 km/s
                    #print(log_L.shape,log_FWHM.shape)
                    log_M_BH = a + b * (log_L - 44.0) + 2 * log_FWHM
                    M_BH = (10 ** log_M_BH) / f
                    #print(M_BH.shape,"xd")
                    L_edd    = 1.26e38 * M_BH  # [erg/s]

                    # mass‐accretion rate (M⊙/yr)
                    eta      = 0.1
                    c_cm     = self.c * 1e5             # km/s → cm/s
                    M_sun_g  = 1.98847e33          # g
                    sec_yr   = 3.15576e7
                    ########
                    mdot_gs  = Lbolval / (eta * c_cm**2)  
                    mdot_yr  = mdot_gs / M_sun_g * sec_yr

                    extras[line_name] = {
                        "Lwave":              Lmono,
                        "Lbol":               Lbolval,
                        "fwhm_kms":           fkm,
                        "log10_smbh":         np.log10(M_BH),
                        "Ledd":               L_edd,
                        "mdot_msun_per_year": mdot_yr,
                    }
                    if line_name=="Halpha":
                        #https://iopscience.iop.org/article/10.1088/0004-637X/813/2/82/pdf
                        Lhalpha     = lum_all[:, idxs].squeeze()   # (N,) or (N,1)
                        print("Lhalpha:",Lhalpha[0],log_FWHM[0])
                        logMBH = np.log10(1.075) + 6.57 + 0.47 * (np.log10(Lhalpha) - 42) +  2.06 * (log_FWHM)
                        extras[line_name].update({"log10_smbh_halpha":logMBH})
                    
            result = {"basic_params":basic_params,"extras_params":extras,"L_w":L_w,"L_bol":L_bol}
        return  pivot_and_split(self.names,result)
    
#    def posterior_physical_parameters(
#     wl_i: np.ndarray,
#     flux_i: np.ndarray,
#     yerr_i: np.ndarray,
#     mask_i: np.ndarray,
#     full_samples: np.ndarray,
#     region_group: Any,
#     distances: np.ndarray,
#     BOL_CORRECTIONS: Dict[str, float] = BOL_CORRECTIONS,
#     SINGLE_EPOCH_ESTIMATORS: Dict[str, Dict[str, Any]] =SINGLE_EPOCH_ESTIMATORS ,
#     c: float = c,
#     summarize: bool = False,
#     LINES_TO_COMBINE = ["Halpha", "Hbeta"],
#     combine_components = True,
#     limit_velocity = 150.0,
#     extra_products = True
# ) -> Dict[str, Any]:
#     """
#     Master routine: from samples → basic line params, monochromatic & bolometric
#     luminosities, single-epoch BH masses, Eddington L, and accretion rates.
#     """
#     #->region_group.dict_params - vmap_samples:concentrate
#     params_dict_values = {k:full_samples.T[i] for k,i in region_group.params_dict.items()}
    
#     if not extra_products:
#         result = {"params_dict_values":params_dict_values}

#     else:
#         basic_params = extract_basic_line_parameters(
#             full_samples=full_samples,
#             region_group=region_group,
#             distances=distances,
#             c=c,
#         )
#         cont_group = region_group.group_by("region")["continuum"]
#         cont_idx   = cont_group.flat_param_indices_global
#         cont_params= full_samples[:, cont_idx]
#         cont_fun   = cont_group.combined_profile
        
#         if combine_components and 'broad' in basic_params and 'narrow' in basic_params:
#             combined = {}
#             Line = []
#             for line in LINES_TO_COMBINE:
#                 # find all the broad‐component indices for this line
#                 broad_lines = basic_params["broad"]["lines"]
#                 idx_broad   = [i for i, L in enumerate(broad_lines) if L.lower() == line.lower()]
#                 # find the single narrow index (if any)
#                 narrow_lines = basic_params["narrow"]["lines"]
#                 idx_narrow   = [i for i, L in enumerate(narrow_lines) if L.lower() == line.lower()]
#                 # only combine if we actually have ≥2 broad and exactly one narrow
#                 if len(idx_broad) >= 2 and len(idx_narrow) == 1:
#                     N = full_samples.shape[0]

#                     # pull out amps & centers
#                     amps = basic_params["broad"]["amplitude"][:, idx_broad]   # (N, n_broad)
#                     mus  = basic_params["broad"]["center"][:, idx_broad]      # (N, n_broad)
#                     fwhms_kms = basic_params["broad"]["fwhm_kms"][:, idx_broad]  # (N, n_broad)

#                     # stack into (N, 3*n_broad)
#                     params_broad = jnp.stack([amps, mus, fwhms_kms], axis=-1).reshape(N, -1)

#                     # narrow triplet (N,3)
#                     amp_n     = basic_params["narrow"]["amplitude"][:, idx_narrow]
#                     mu_n      = basic_params["narrow"]["center"][:, idx_narrow]
#                     fwhm_nkms = basic_params["narrow"]["fwhm_kms"][:, idx_narrow]
#                     params_narrow = jnp.concatenate([amp_n, mu_n, fwhm_nkms], axis=1)

#                     fwhm_c, amp_c, mu_c = combine_fast(
#                         params_broad, params_narrow,
#                         limit_velocity=limit_velocity, c=c
#                     )

#                     fwhm_A = (fwhm_c / c) * mu_c 

#                     flux_c = calc_flux(np.array(amp_c), np.array(fwhm_A))

#                     fwhm_A = (fwhm_c / c) * mu_c       # shape (N,)

#                     flux_c = calc_flux(np.array(amp_c), np.array(fwhm_A))  # (N,)

#                     cont_c = vmap(cont_group.combined_profile)(mu_c, cont_params)  # (N,)

#                     L_line = calc_luminosity(distances, flux_c, mu_c)  # (N,)

#                     eqw_c = flux_c / cont_c

#                     combined[line] = {
#                         "amplitude":  np.array(amp_c),    
#                         "center":     np.array(mu_c),     
#                         "fwhm_kms":   np.array(fwhm_c),   
#                         "fwhm":     np.array(fwhm_A),   
#                         "flux":       np.array(flux_c),   
#                         "luminosity": np.array(L_line),   
#                         "eqw":        np.array(eqw_c),    
#                     }
#                     Line.append(line)
#         L_w, L_bol = {}, {}
        

#         for wave in map(float, BOL_CORRECTIONS.keys()):
#             wstr = str(int(wave))
#             if (jnp.isclose(wl_i, wave, atol=1) & ~mask_i).any():
#                 Fcont   = vmap(cont_fun, in_axes=(None, 0))(jnp.array([wave]), cont_params).squeeze()
#                 Lmono   = calc_monochromatic_luminosity(distances, Fcont, wave)
#                 Lbolval = calc_bolometric_luminosity(Lmono, BOL_CORRECTIONS[wstr])
#                 L_w[wstr], L_bol[wstr] = np.array(Lmono), np.array(Lbolval)

    
#         broad = basic_params.get("broad")
#         if broad:
#             extra_params = calculate_single_epoch_masses(broad,L_w,L_bol,SINGLE_EPOCH_ESTIMATORS,c) #for broad
#         #names have to be improve 
#         result = {
#             "basic_params": basic_params,
#             "L_w":           L_w,
#             "L_bol":         L_bol,
#             "extras_params":        extra_params,
#             "params_dict_values":params_dict_values
#         }
#         if len(combined.keys())>0:
#             #combined["lines"] = Line
#             combined["extras"] = calculate_single_epoch_masses(combined,L_w,L_bol,SINGLE_EPOCH_ESTIMATORS,c,combine_mode=True) #for broad
#             result["combined"] = combined

#     if summarize:
#         result = summarize_nested_samples(result)  
#     return result



# def extract_basic_line_parameters(
#     full_samples: np.ndarray,
#     region_group: Any, #we already have a class for this 
#     distances: np.ndarray,
#     c: float,
#     wavelength_grid: jnp.ndarray = jnp.linspace(0, 20_000, 20_000),
# ) -> Dict[str, Dict[str, np.ndarray]]:
#     """
#     Extract continuum‐corrected flux, FWHM, FWHM (km/s), center, amplitude,
#     equivalent width and luminosity for each emission line, grouped by 'region'.
#     # We will keep kind for a while. 
#     Returns a dict mapping each region → dict with keys:
#       'lines', 'component',
#       'flux', 'fwhm', 'fwhm_kms',
#       'center', 'amplitude',
#       'eqw', 'luminosity'
#     outside this class all have to be already in flux units.
#     """
#     # Precompute continuum params
#     cont_group = region_group.group_by("region")["continuum"]
#     cont_idx   = cont_group.flat_param_indices_global
#     cont_params= full_samples[:, cont_idx]

#     basic_params: Dict[str, Dict[str, np.ndarray]] = {}

#     for kind, kind_group in region_group.group_by("region").items():
#         if kind in ("fe", "continuum"):
#             continue

#         line_names, components = [], []
#         flux_parts, fwhm_parts = [], []
#         fwhm_kms_parts, center_parts = [], []
#         amp_parts, eqw_parts, lum_parts = [], [], []

#         for profile_name, prof_group in kind_group.group_by("profile_name").items():

#             # Determine integrator and handle sub-profiles
#             if "_" in profile_name:
#                 _, subprof = profile_name.split("_", 1)
#                 profile_fn = PROFILE_FUNC_MAP[subprof]
#                 batch_fwhm = make_batch_fwhm_split(subprof)  # jitted on first call
#                 integrator = make_integrator(profile_fn, method="vmap")
#                 for sp, param_idxs in zip(
#                     prof_group.lines, prof_group.global_profile_params_index_list
#                 ):
#                     params      = full_samples[:, param_idxs]
#                     names       = np.array(prof_group._master_param_names)[param_idxs]
#                     amplitude_relations = sp.amplitude_relations
#                     amp_pos     = np.where(["logamp" in n for n in names])[0]
#                     amplitude_index = [nx for nx,_ in  enumerate(names) if "logamp" in _ ]
#                     ind_amplitude_index = {i[2] for i in amplitude_relations}
#                     dic_amp = {i:ii for i,ii in (zip(ind_amplitude_index,amplitude_index))}
#                     shift_idx   = amp_pos.max() + 1
#                     full_params_by_line = []
#                     for i,(_,factor,ids) in enumerate(amplitude_relations):
#                         amplitude_line = (params[:,[dic_amp[ids]]]*factor)#from log to actual amplitude 
#                         center_line = (sp.center[i]+params[:,[shift_idx]])
#                         extra_params_profile = (params[:,shift_idx+1:])
#                         full_params_by_line.append(np.column_stack([amplitude_line, center_line, extra_params_profile]))
#                     params_by_line = np.array(full_params_by_line) 
#                     params_by_line = np.moveaxis(params_by_line,0,1)
                    
#                     flux        = integrator(wavelength_grid, params_by_line)
#                     fwhm        =  jnp.atleast_3d(params_by_line[:,:,-1])
                    
#                     centers     = params_by_line[:, :, 1]
#                     amps        = 10**params_by_line[:, :, 0]
                    
#                     fwhm = batch_fwhm(amps, centers, fwhm)  
#                     fwhm_kms    = jnp.abs(calc_fwhm_kms(fwhm, c, centers))
#                     cont_vals   = vmap(cont_group.combined_profile, in_axes=(0,0))(
#                                       centers, cont_params
#                                   )
#                     lum_vals    = calc_luminosity(distances[:, None], flux, centers)
#                     eqw        = flux / cont_vals

#                     nsub = flux.shape[1]
#                     line_names.extend(sp.region_lines)
#                     components.extend([sp.component]*nsub)
#                     flux_parts.append(np.array(flux))
#                     fwhm_parts.append(np.array(fwhm))
#                     fwhm_kms_parts.append(np.array(fwhm_kms))
#                     center_parts.append(np.array(centers))
#                     amp_parts.append(np.array(amps))
#                     eqw_parts.append(np.array(eqw))
#                     lum_parts.append(np.array(lum_vals))

#             else:
#                 profile_fn = PROFILE_FUNC_MAP[profile_name]
#                 batch_fwhm = make_batch_fwhm_split(profile_name)  # jitted on first call
#                 integrator = make_integrator(profile_fn, method="vmap")
#                 idxs       = prof_group.flat_param_indices_global
#                 params     = full_samples[:, idxs]
#                 names      = list(prof_group.params_dict.keys())

#                 amp_idx = [i for i,n in enumerate(names) if "logamp" in n]
#                 cen_idx = [i for i,n in enumerate(names) if "center" in n]
#                 other   = [i for i in range(params.shape[1]) 
#                            if i not in amp_idx + cen_idx]

#                 reshaped = params.reshape(params.shape[0], -1, profile_fn.n_params)
#                 flux     = integrator(wavelength_grid, reshaped)
#                 fwhm     = jnp.atleast_3d(jnp.abs(params[:, other]))
#                 centers  = params[:, cen_idx]
#                 amps     = 10**params[:, amp_idx]
#                 #print(amps.shape,centers.shape,fwhm.shape)
#                 fwhm = batch_fwhm(amps, centers, fwhm)         # → (1000,20)
#                 fwhm_kms = jnp.abs(calc_fwhm_kms(fwhm, c, centers))
#                 cont_vals= vmap(cont_group.combined_profile, in_axes=(0,0))(
#                               centers, cont_params
#                           )
#                 #print(distances.shape,flux.shape,centers.shape)
#                 lum_vals = calc_luminosity(distances[:, None], flux, centers)
#                 eqw      = flux / cont_vals

#                 line_names.extend([l.line_name for l in prof_group.lines])
#                 components.extend([l.component for l in prof_group.lines])
#                 flux_parts.append(np.array(flux))
#                 fwhm_parts.append(np.array(fwhm))
#                 fwhm_kms_parts.append(np.array(fwhm_kms))
#                 center_parts.append(np.array(centers))
#                 amp_parts.append(np.array(amps))
#                 eqw_parts.append(np.array(eqw))
#                 lum_parts.append(np.array(lum_vals))

#         basic_params[kind] = {
#             "lines":      line_names,
#             "component":  components,
#             "flux":       np.concatenate(flux_parts,     axis=1),
#             "fwhm":       np.concatenate(fwhm_parts,     axis=1),
#             "fwhm_kms":   np.concatenate(fwhm_kms_parts, axis=1),
#             "center":     np.concatenate(center_parts,    axis=1),
#             "amplitude":  np.concatenate(amp_parts,       axis=1),
#             "eqw":        np.concatenate(eqw_parts,       axis=1),
#             "luminosity": np.concatenate(lum_parts,       axis=1),
#         }

#     return basic_params