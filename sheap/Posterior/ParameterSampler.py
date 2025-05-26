from typing import Tuple, Dict, List

import jax.numpy as jnp
from jax import vmap, random
import numpy as np 
from tqdm import tqdm

from .functions import calc_flux,calc_luminosity,calc_fwhm_kms,calc_monochromatic_luminosity,calc_bolometric_luminosity,calc_black_hole_mass
from .constants import BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS
from sheap.Mappers.LineMapper import LineMapper 
from sheap.Mappers.helpers import mapping_params



# this have to be move outside 


#we have to add this. vmap and sum.

# from scipy.stats import skew, kurtosis

# sk = skew(emission_profiles, axis=1)
# print("sk",sk)
# kt = kurtosis(emission_profiles, axis=1, fisher=False)  
# print("kt",kt)

class ParameterSampler:
    """
    Monte Carlo sampler for spectral fit results and parameter uncertainties.
    BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS should came from ParameterEstimation
    """
    def __init__(self, estimator: "ParameterEstimation"):
        self.estimator = estimator  # ParameterEstimation instance
        self.model = estimator.model
        self.c = estimator.c
        self.dependencies = estimator.dependencies
        self.kinds_map = estimator.kinds_map
        self.max_flux = estimator.max_flux
        self.fluxnorm = estimator.fluxnorm
        self.spec = estimator.spec
        self.mask = estimator.mask
        self.d = estimator.d
        self.params = estimator.params
        self.params_dict = estimator.params_dict

    def sample_params(self, N: int = 2000, key_seed: int = 0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        from sheap.RegionFitting.uncertainty_functions import (
            apply_tied_and_fixed_params, make_residuals_free_fn, error_covariance_matrix
        )
        scaled = self.max_flux
        norm_spec = self.spec.at[:, [1, 2], :].divide(
            jnp.moveaxis(jnp.tile(scaled, (2, 1)), 0, 1)[:, :, None]
        )
        norm_spec = norm_spec.at[:, 2, :].set(jnp.where(self.mask, 1e31, norm_spec[:, 2, :]))
        idxs = mapping_params(self.params_dict, [["amplitude"], ["scale"]])
        params = self.params.at[:, idxs].divide(scaled[:, None])
        wl, flux, yerr = jnp.moveaxis(norm_spec, 0, 1)
        model = self.model
        dependencies = self.dependencies
        idx_target = [i[1] for i in dependencies]
        idx_free_params = list(set(range(len(params[0]))) - set(idx_target))
        key = random.PRNGKey(key_seed)
        mega_full_sample = []
        results_L_w, results_L_bol, results_masses = [], [], []
        #add tqm 
        for n, (params_i, wl_i, flux_i, yerr_i) in enumerate(tqdm(zip(params, wl, flux, yerr), total=len(params), desc="Sampling obj")):
            free_params = params_i[jnp.array(idx_free_params)]
            res_fn = make_residuals_free_fn(
                model_func=model, xs=wl_i, y=flux_i, yerr=yerr_i,
                template_params=params_i, dependencies=dependencies
            )
            _, cov_matrix = error_covariance_matrix(
                residual_fn=res_fn,
                params_i=free_params,
                xs_i=wl_i,
                y_i=flux_i,
                yerr_i=yerr_i,
                free_params=len(free_params),
                return_full=True
            )
            L = jnp.linalg.cholesky(cov_matrix + 1e-6 * jnp.eye(cov_matrix.shape[0]))
            z = random.normal(key, shape=(N, len(free_params)))
            samples_free = free_params + z @ L.T  # (N, n_free)

            def apply_one_sample(free_sample):
                return apply_tied_and_fixed_params(free_sample, params_i, dependencies)

            full_samples = vmap(apply_one_sample)(samples_free)
            full_samples = full_samples.at[:, idxs].multiply(scaled[n])
            mega_full_sample.append(full_samples)
        return mega_full_sample
    
    
    
        #     #this have to be a only one runite that can be share between the samplers.
        #     dict_ = {}
        #     for k, k_map in self.kinds_map.items():
        #         if k not in ['fe', 'continuum']:
        #             idx_amplitude = mapping_params(k_map.filtered_dict, "amplitude")
        #             idx_fwhm = mapping_params(k_map.filtered_dict, "fwhm")
        #             idx_center = mapping_params(k_map.filtered_dict, "center")
                    
        #             norm_amplitude = full_samples[:, idx_amplitude]
        #             fwhm = full_samples[:, idx_fwhm]
        #             center = full_samples[:, idx_center]
        #             flux = calc_flux(norm_amplitude, fwhm)
        #             fwhm_kms = calc_fwhm_kms(fwhm, self.c, center)
        #             L_line = calc_luminosity(self.d[n], flux, center)
        #             dict_[k] = {
        #                 'lines': k_map.line_name,
        #                 "component": jnp.array(k_map.component),
        #                 'flux': flux, "fwhm": fwhm, "fwhm_kms": fwhm_kms, "L": L_line,
        #                 'center': center, 'amplitude': norm_amplitude
        #             }

        #     L_w, L_bol = {}, {}
        #     wavelenghts = [1350.0, 1450.0, 3000.0, 5100.0, 6200.0]
        #     # Assume 'continuum' is present
        #     #idx_cont = mapping_params(self.params_dict, "scale")  # Adapt if needed
        #     #profile_func = self.estimator.RegionMap.profile_functions_combine
        #     map_cont = self.kinds_map['continuum']
        #     profile_func = map_cont.profile_functions_combine
        #     idx_cont = jnp.array(list(map_cont.filtered_dict.values()))
            
        #     cont_params = full_samples[:, idx_cont]
        #     for w in wavelenghts:
        #         wave = str(int(w))
        #         hits = jnp.isclose(norm_spec[n, 0, :], jnp.array([w]), atol=1)
        #         valid = (hits & (~self.mask[n])).any()
        #         corr = BOL_CORRECTIONS.get(wave, 0.0)
        #         if valid:
        #             flux_at_w = vmap(profile_func, in_axes=(None, 0))(jnp.array([w]), cont_params).squeeze()
        #             Lw = calc_monochromatic_luminosity(self.d[n], flux_at_w, w)
        #             L_w[wave] = Lw
        #             L_bol[wave] = calc_bolometric_luminosity(Lw, corr)
        #         else:
        #             L_w[wave] = jnp.zeros(full_samples.shape[0])
        #             L_bol[wave] = jnp.zeros(full_samples.shape[0])

        #     # --- Compute black hole masses ---
        #     dict_broad = dict_.get("broad")
        #     masses = {}
        #     if dict_broad is not None:
        #         fwhm_kms = dict_broad.get('fwhm_kms')
        #         line_name_list = np.array(dict_broad["lines"])
        #         for line_name, estimator in SINGLE_EPOCH_ESTIMATORS.items():
        #             wave = estimator["wavelength"]
        #             if line_name not in line_name_list or wave not in L_w:
        #                 continue
        #             idx_broad = list(jnp.where(line_name == line_name_list)[0])
        #             Lwave = L_w[wave]
        #             fwhm_kms_ = fwhm_kms[:, idx_broad].squeeze()
        #             masses[line_name] = calc_black_hole_mass(Lwave, fwhm_kms_, estimator)
            
        #     results_L_w.append(L_w)
        #     results_L_bol.append(L_bol)
        #     results_masses.append(masses)
            
        # return results_L_w, results_L_bol, results_masses
