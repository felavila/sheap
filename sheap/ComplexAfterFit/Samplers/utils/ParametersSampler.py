"""I have to change the name of this function is not clear neither for me what is inside."""
__author__ = 'felavila'

__all__ = [
    "extract_basic_line_parameters",
    "posterior_parameters",
]

from typing import Any, Dict, List, Union
import numpy as np
import jax.numpy as jnp
from jax import vmap,jit
import warnings
from functools import partial

from sheap.Profiles.utils import make_integrator
from sheap.Profiles.profiles import PROFILE_LINE_FUNC_MAP,PROFILE_FUNC_MAP

from sheap.ComplexAfterFit.Samplers.utils.samplehandlers import summarize_nested_samples

from sheap.ComplexAfterFit.Samplers.utils.physicalfunctions import calc_flux,calc_fwhm_kms,calc_luminosity,calc_monochromatic_luminosity,calc_bolometric_luminosity,extra_params

from sheap.Utils.Constants  import BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS,c




def extract_basic_line_parameters(
    full_samples: np.ndarray,
    region_group: Any, #we already have a class for this 
    distances: np.ndarray,
    c: float,
    wavelength_grid: jnp.ndarray = jnp.linspace(0, 20_000, 20_000),
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract continuum‐corrected flux, FWHM, FWHM (km/s), center, amplitude,
    equivalent width and luminosity for each emission line, grouped by 'region'.
    # We will keep kind for a while. 
    Returns a dict mapping each region → dict with keys:
      'lines', 'component',
      'flux', 'fwhm', 'fwhm_kms',
      'center', 'amplitude',
      'eqw', 'luminosity'
    outside this class all have to be already in flux units.
    """
    # Precompute continuum params
    cont_group = region_group.group_by("region")["continuum"]
    cont_idx   = cont_group.flat_param_indices_global
    cont_params= full_samples[:, cont_idx]

    basic_params: Dict[str, Dict[str, np.ndarray]] = {}

    for kind, kind_group in region_group.group_by("region").items():
        if kind in ("fe", "continuum"):
            continue

        line_names, components = [], []
        flux_parts, fwhm_parts = [], []
        fwhm_kms_parts, center_parts = [], []
        amp_parts, eqw_parts, lum_parts = [], [], []

        for profile_name, prof_group in kind_group.group_by("profile_name").items():

            # Determine integrator and handle sub-profiles
            if "_" in profile_name:
                _, subprof = profile_name.split("_", 1)
                profile_fn = PROFILE_FUNC_MAP[subprof]
                batch_fwhm = make_batch_fwhm_split(subprof)  # jitted on first call
                integrator = make_integrator(profile_fn, method="vmap")
                for sp, param_idxs in zip(
                    prof_group.lines, prof_group.global_profile_params_index_list
                ):
                    params      = full_samples[:, param_idxs]
                    names       = np.array(prof_group._master_param_names)[param_idxs]
                    amplitude_relations = sp.amplitude_relations
                    amp_pos     = np.where(["logamp" in n for n in names])[0]
                    amplitude_index = [nx for nx,_ in  enumerate(names) if "logamp" in _ ]
                    ind_amplitude_index = {i[2] for i in amplitude_relations}
                    dic_amp = {i:ii for i,ii in (zip(ind_amplitude_index,amplitude_index))}
                    shift_idx   = amp_pos.max() + 1
                    full_params_by_line = []
                    for i,(_,factor,ids) in enumerate(amplitude_relations):
                        amplitude_line = params[:,[dic_amp[ids]]] + np.log10(factor)
                        center_line = (sp.center[i]+params[:,[shift_idx]])
                        extra_params_profile = (params[:,shift_idx+1:])
                        full_params_by_line.append(np.column_stack([amplitude_line, center_line, extra_params_profile]))
                    
                    params_by_line = np.array(full_params_by_line) 
                    params_by_line = np.moveaxis(params_by_line,0,1)
                    
                    flux        = integrator(wavelength_grid, params_by_line)
                    amps        = 10**params_by_line[:, :, 0]
                    centers     = params_by_line[:, :, 1]
                    
                    fwhm        =  jnp.atleast_3d(params_by_line[:,:,2:])
                    
                    
                   
                    
                    fwhm = batch_fwhm(amps, centers, fwhm)  
                    fwhm_kms    = jnp.abs(calc_fwhm_kms(fwhm, c, centers))
                    cont_vals   = vmap(cont_group.combined_profile, in_axes=(0,0))(
                                      centers, cont_params
                                  )
                    lum_vals    = calc_luminosity(distances[:, None], flux)
                    eqw        = flux / cont_vals

                    nsub = flux.shape[1]
                    line_names.extend(sp.region_lines)
                    components.extend([sp.component]*nsub)
                    flux_parts.append(np.array(flux))
                    fwhm_parts.append(np.array(fwhm))
                    fwhm_kms_parts.append(np.array(fwhm_kms))
                    center_parts.append(np.array(centers))
                    amp_parts.append(np.array(amps))
                    eqw_parts.append(np.array(eqw))
                    lum_parts.append(np.array(lum_vals))

            else:
                profile_fn = PROFILE_FUNC_MAP[profile_name]
                batch_fwhm = make_batch_fwhm_split(profile_name)  # jitted on first call
                integrator = make_integrator(profile_fn, method="vmap")
                idxs       = prof_group.flat_param_indices_global
                params     = full_samples[:, idxs]
                names      = list(prof_group.params_dict.keys())
                ###########################
                params_by_line = params.reshape(params.shape[0], -1, profile_fn.n_params)
                flux     = integrator(wavelength_grid, params_by_line)
                fwhm     = jnp.abs(params_by_line[:,:,2:])#check if this is correct)
                centers  = params_by_line[:,:,1]
                amps     = 10**params_by_line[:,:,0]
                #print(amps.shape,centers.shape,fwhm.shape)
                fwhm = batch_fwhm(amps, centers, fwhm)         # → (1000,20)
                fwhm_kms = jnp.abs(calc_fwhm_kms(fwhm, c, centers))
                cont_vals= vmap(cont_group.combined_profile, in_axes=(0,0))(
                              centers, cont_params
                          )
                #print(distances.shape,flux.shape,centers.shape)
                lum_vals = calc_luminosity(distances[:, None], flux)
                eqw      = flux / cont_vals

                line_names.extend([l.line_name for l in prof_group.lines])
                components.extend([l.component for l in prof_group.lines])
                flux_parts.append(np.array(flux))
                fwhm_parts.append(np.array(fwhm))
                fwhm_kms_parts.append(np.array(fwhm_kms))
                center_parts.append(np.array(centers))
                amp_parts.append(np.array(amps))
                eqw_parts.append(np.array(eqw))
                lum_parts.append(np.array(lum_vals))

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

def posterior_parameters(
    wl_i: np.ndarray,
    flux_i: np.ndarray,
    yerr_i: np.ndarray,
    mask_i: np.ndarray,
    full_samples: np.ndarray,
    region_group: Any,
    distances: np.ndarray,
    BOL_CORRECTIONS: Dict[str, float] = BOL_CORRECTIONS,
    SINGLE_EPOCH_ESTIMATORS: Dict[str, Dict[str, Any]] =SINGLE_EPOCH_ESTIMATORS ,
    c: float = c,
    summarize: bool = False,
    LINES_TO_COMBINE = ["Halpha", "Hbeta"],
    combine_components = True,
    limit_velocity = 150.0,
    extra_products = True
) -> Dict[str, Any]:
    """
    Master routine: from samples → basic line params, monochromatic & bolometric
    luminosities, single-epoch BH masses, Eddington L, and accretion rates.
    """
    #->region_group.dict_params - vmap_samples:concentrate
    params_dict_values = {k:full_samples.T[i] for k,i in region_group.params_dict.items()}
    
    if not extra_products:
        result = {"params_dict_values":params_dict_values}

    else:
        basic_params = extract_basic_line_parameters(
            full_samples=full_samples,
            region_group=region_group,
            distances=distances,
            c=c,
        )
        cont_group = region_group.group_by("region")["continuum"]
        cont_idx   = cont_group.flat_param_indices_global
        cont_params= full_samples[:, cont_idx]
        cont_fun   = cont_group.combined_profile
        
        if combine_components and 'broad' in basic_params and 'narrow' in basic_params:
            combined = {}
            Line = []
            for line in LINES_TO_COMBINE:
                # find all the broad‐component indices for this line
                broad_lines = basic_params["broad"]["lines"]
                idx_broad   = [i for i, L in enumerate(broad_lines) if L.lower() == line.lower()]
                # find the single narrow index (if any)
                narrow_lines = basic_params["narrow"]["lines"]
                idx_narrow   = [i for i, L in enumerate(narrow_lines) if L.lower() == line.lower()]
                # only combine if we actually have ≥2 broad and exactly one narrow
                if len(idx_broad) >= 2 and len(idx_narrow) == 1:
                    N = full_samples.shape[0]


                    amps = basic_params["broad"]["amplitude"][:, idx_broad]   # (N, n_broad)
                    mus  = basic_params["broad"]["center"][:, idx_broad]      # (N, n_broad)
                    fwhms_kms = basic_params["broad"]["fwhm_kms"][:, idx_broad]  # (N, n_broad)

                    # stack into (N, 3*n_broad)
                    params_broad = jnp.stack([amps, mus, fwhms_kms], axis=-1).reshape(N, -1)

                    # narrow triplet (N,3)
                    amp_n     = basic_params["narrow"]["amplitude"][:, idx_narrow]
                    mu_n      = basic_params["narrow"]["center"][:, idx_narrow]
                    fwhm_nkms = basic_params["narrow"]["fwhm_kms"][:, idx_narrow]
                    params_narrow = jnp.concatenate([amp_n, mu_n, fwhm_nkms], axis=1)

                    fwhm_c, amp_c, mu_c = combine_fast(
                        params_broad, params_narrow,
                        limit_velocity=limit_velocity, c=c
                    )

                    fwhm_A = (fwhm_c / c) * mu_c 

                    flux_c = calc_flux(np.array(amp_c), np.array(fwhm_A))

                    fwhm_A = (fwhm_c / c) * mu_c       # shape (N,)

                    flux_c = calc_flux(np.array(amp_c), np.array(fwhm_A))  # (N,)

                    cont_c = vmap(cont_group.combined_profile)(mu_c, cont_params)  # (N,)

                    L_line = calc_luminosity(distances, flux_c)  # (N,)

                    eqw_c = flux_c / cont_c

                    combined[line] = {
                        "amplitude":  np.array(amp_c),    
                        "center":     np.array(mu_c),     
                        "fwhm_kms":   np.array(fwhm_c),   
                        "fwhm":     np.array(fwhm_A),   
                        "flux":       np.array(flux_c),   
                        "luminosity": np.array(L_line),   
                        "eqw":        np.array(eqw_c),    
                    }
                    Line.append(line)
        L_w, L_bol = {}, {}
        for wave in map(float, BOL_CORRECTIONS.keys()):
            wstr = str(int(wave))
            if (jnp.isclose(wl_i, wave, atol=1) & ~mask_i).any():
                Fcont   = vmap(cont_fun, in_axes=(None, 0))(jnp.array([wave]), cont_params).squeeze()
                Lmono   = calc_monochromatic_luminosity(distances, Fcont, wave)
                Lbolval = calc_bolometric_luminosity(Lmono, BOL_CORRECTIONS[wstr])
                L_w[wstr], L_bol[wstr] = np.array(Lmono), np.array(Lbolval)

    
        broad = basic_params.get("broad")
        if broad:
            extra_params = extra_params(broad,L_w,L_bol,SINGLE_EPOCH_ESTIMATORS,c) #for broad
        #names have to be improve 
        result = {
            "basic_params": basic_params,
            "L_w":           L_w,
            "L_bol":         L_bol,
            "extras_params":        extra_params,
            "params_dict_values":params_dict_values
        }
        if len(combined.keys())>0:
            #combined["lines"] = Line
            combined["extras"] = extra_params(combined,L_w,L_bol,SINGLE_EPOCH_ESTIMATORS,c,combine_mode=True) #for broad
            result["combined"] = combined

    if summarize:
        result = summarize_nested_samples(result)  
    return result
