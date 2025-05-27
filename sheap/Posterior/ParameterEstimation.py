import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from astropy.cosmology import FlatLambdaCDM
from auto_uncertainties import Uncertainty
from jax import grad, jit,vmap

from sheap.FunctionsMinimize.utils import combine_auto
#from sheap.Posterior.utils import combine
from sheap.LineMapper.LineMapper import LineMapper, mapping_params
#from sheap.Tools.others import vmap_get_EQW_mask
from .MonteCarloSampler import MonteCarloSampler
from .McMcSampler import McMcSampler

from .constants import BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS

cm_per_mpc = 3.08568e24

class ParameterEstimation:
    """
    Computes best-fit physical parameters and uncertainties for spectral regions.
    Monte Carlo sampling of distributions is provided via .sample_params() using ParameterSampler.
    """
    def __init__(
        self,
        sheap: Optional["Sheapectral"] = None,
        fit_result: Optional["FitResult"] = None,
        spectra: Optional[jnp.ndarray] = None,
        z: Optional[jnp.ndarray] = None,
        fluxnorm=None,
        cosmo=None,
        c=299792.458,
    ):
        if sheap is not None:
            self._from_sheap(sheap)
        elif fit_result is not None and spectra is not None:
            self._from_fit_result(fit_result, spectra, z)
        else:
            raise ValueError("Provide either `sheap` or (`fit_result` + `spectra`).")

        self.c = c
        self.RegionMap = LineMapper(
            complex_region=self.complex_region,
            profile_functions=self.profile_functions,
            params=self.params,
            uncertainty_params=self.uncertainty_params,
            profile_params_index_list=self.profile_params_index_list,
            params_dict=self.params_dict,
            profile_names=self.profile_names
        )

        if self.z is None:
            print("None informed redshift, assuming zero.")
            self.z = np.zeros(self.spec.shape[0])
        if cosmo is None:
            self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        else:
            self.cosmo = cosmo
        if fluxnorm is None:
            self.fluxnorm = np.ones(self.spec.shape[0])
        else:
            self.fluxnorm = fluxnorm

        self.d = self.cosmo.luminosity_distance(self.z) * cm_per_mpc

        self.kinds_map = {}
        for k in self.kind_list:
            self.kinds_map[k] = self.RegionMap._get(where="kind", what=k)

    def compute_params_wu(self):
        """
        Compute line flux, FWHM, luminosity, and FWHM in km/s with uncertainties for each emission line kind.
        Returns
        -------
        dict
            Mapping of line kind to computed quantities (with uncertainties).
        """
        dict_ = {}
        for k, k_map in self.kinds_map.items():
            if k in ('fe', 'continuum'):
                continue
            idx_amplitude = mapping_params(k_map.params_names, "amplitude")
            idx_fwhm = mapping_params(k_map.params_names, "fwhm")
            idx_center = mapping_params(k_map.params_names, "center")
            
            profile_vmap =  vmap(k_map.profile_functions_combine, in_axes=(0, 0))(self.spec[:,0,:], k_map.params)
            
            params = k_map.params
            uncertainty_params = np.array(k_map.uncertainty_params)

            norm_amplitude = params[:, idx_amplitude]
            norm_amplitude_u = uncertainty_params[:, idx_amplitude]

            fwhm = params[:, idx_fwhm]
            fwhm_u = uncertainty_params[:, idx_fwhm]

            center = params[:, idx_center]
            center_u = uncertainty_params[:, idx_center]

            norm_amplitude = (
                Uncertainty(norm_amplitude, norm_amplitude_u) * self.fluxnorm[:, None]
            )
            fwhm = Uncertainty(fwhm, np.abs(fwhm_u))
            center = Uncertainty(center, np.abs(center_u))

            flux = np.sqrt(2.0 * np.pi) * norm_amplitude * fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            L = 4.0 * np.pi * np.array(self.d[:, None] ** 2) * flux * center
            fwhm_kms = (fwhm * self.c) / center

            dict_[k] = {
                'lines': k_map.line_name,
                "component": np.array(k_map.component),
                'L': {'value': L.value, 'error': L.error},
                'flux': {'value': flux.value, 'error': flux.error},
                'fwhm': {'value': fwhm.value, 'error': fwhm.error},
                'fwhm_kms': {'value': fwhm_kms.value, 'error': fwhm_kms.error},
                "profile" : profile_vmap
            }
        self.dict_params = dict_
        return dict_

    def compute_Luminosity_w(self, wavelenghts=[1350.0, 1450.0, 3000.0, 5100.0, 6200.0]):
        """
        Compute monochromatic luminosities (and uncertainties) at key wavelengths.
        Returns a dict with wavelength (as str) to value/error arrays.
        """
        map_cont = self.kinds_map['continuum']
        profile_func = map_cont.profile_functions_combine
        params = map_cont.params.T
        uncertainty_params = map_cont.uncertainty_params.T
        L_w = {}
        for w in wavelenghts:
            hits = jnp.isclose(self.spec[:, 0, :], w, atol=1)
            valid = (hits & (~self.mask)).any(axis=1, keepdims=True)
            grad_f = grad(lambda p: jnp.sum(profile_func(jnp.array([w]), p)))(params)
            flux = jnp.where(valid.squeeze(), profile_func(jnp.array([w]), params), 0)
            sigma_f = jnp.where(
                valid.squeeze(),
                jnp.sqrt(jnp.sum((grad_f * uncertainty_params) ** 2, axis=0)),
                0,
            )
            flux = Uncertainty(np.array(flux), np.array(sigma_f))
            l = (w * 4.0 * np.pi * np.array(self.d**2) * flux).ravel()
            L_w[str(w)] = {'value': l.value, "error": l.error}
        self.L_w = L_w
        return L_w

    def compute_bolometric_luminosity(self, monochromatic_lums=None):
        """
        Estimate bolometric luminosity using empirical bolometric correction factors.
        Returns a dict keyed by wavelength.
        """
        if monochromatic_lums is None:
            monochromatic_lums = self.compute_Luminosity_w()
        L_bol = {}
        for wave, L in monochromatic_lums.items():
            corr = BOL_CORRECTIONS.get(str(int(float(wave))), 0.0)
            L_bol[wave] = {'value': L['value'] * corr, 'error': L['error'] * corr}
        return L_bol

    def compute_black_hole_mass(self, f=1.0, non_combine=False):
        """
        Compute black hole mass estimates using single-epoch virial estimators.
        Returns dict: line_name -> {'value': M_BH, 'error': sigma, 'component': comp_idx}
        """
        dict_broad = self.compute_params_wu().get("broad")
        L_w = self.compute_Luminosity_w()
        fwhm_kms = dict_broad.get('fwhm_kms')
        line_name_list = np.array(dict_broad["lines"])
        masses = {}

        for line_name, params in SINGLE_EPOCH_ESTIMATORS.items():
            wave = params["wavelength"]
            if line_name not in line_name_list or wave not in L_w.keys():
                continue
            else:
                idx = np.where(line_name == line_name_list)[0]
            a, b, f_vir = params["a"], params["b"], f
            l_ = Uncertainty(L_w[wave].get("value"), L_w[wave].get("error"))
            log_L = np.log10(l_).reshape(-1, 1)
            fwhm_kms_ = Uncertainty(fwhm_kms.get("value")[:, idx], fwhm_kms.get("error")[:, idx])
            log_FWHM = np.log10(fwhm_kms_) - 3
            log_M_BH = a + b * (log_L - 44.0) + 2 * log_FWHM
            M_BH = (10 ** log_M_BH) / f_vir
            masses[line_name] = {
                "value": M_BH.value.squeeze(),
                "error": M_BH.error.squeeze(),
                "component": dict_broad.get("component")[idx]
            }
        return masses
    #def 
    
    
    def sample_montecarlo(self, N: int = 2000, key_seed: int = 0):
        """
        Run Monte Carlo parameter sampling (see MonteCarloSampler for details).
        Returns megafullsample
        """
        sampler = MonteCarloSampler(self)
        return sampler.sample_params(N=N, key_seed=key_seed)
    
    
    def sample_mcmc(self,n_random = 0,num_warmup=500,num_samples=1000):
        """
        Run mcmc using numpyro parameter sampling.
        Returns megafullsample
        """
        sampler = McMcSampler(self)
        return sampler.sample_params(n_random,num_warmup,num_samples)

    
    def _from_sheap(self, sheap):
        self.spec = sheap.spectra
        self.z = sheap.z
        self.result = sheap.result

        result = sheap.result  # for convenience
        self.constraints = result.constraints
        self.params = result.params
        self.max_flux = result.max_flux
        self.uncertainty_params = result.uncertainty_params
        self.profile_params_index_list = result.profile_params_index_list
        self.profile_functions = result.profile_functions
        self.profile_names = result.profile_names
        self.complex_region = result.complex_region
        self.xlim = result.outer_limits
        self.mask = result.mask
        self.names = sheap.names
        self.model_keywords = result.model_keywords or {}
        self.fe_mode = self.model_keywords.get("fe_mode")
        self.model = jit(combine_auto(self.profile_functions))
        self.kind_list = result.kind_list
        self.params_dict = result.params_dict
        self.dependencies = result.dependencies
        

    def _from_fit_result(self, result, spectra, z):
        self.spec = spectra
        self.z = z
        self.params = result.params
        self.uncertainty_params = result.uncertainty_params
        self.profile_params_index_list = result.profile_params_index_list
        self.profile_functions = result.profile_functions
        self.profile_names = result.profile_names
        self.complex_region = result.complex_region
        self.xlim = result.outer_limits
        self.mask = result.mask
        self.names = [str(i) for i in range(self.params.shape[0])]
        self.model_keywords = result.model_keywords or {}
        self.fe_mode = self.model_keywords.get("fe_mode")
        self.model = jit(combine_auto(self.profile_functions))
        self.kind_list = result.kind_list
        self.params_dict = result.params_dict
        self.constraints = result.constraints

    
    # def sample_params_original(self,N = 2_000):
    #     from sheap.RegionFitting.uncertainty_functions import apply_tied_and_fixed_params,make_residuals_free_fn,error_covariance_matrix
    #     from jax import random
    #     import jax 
    #     print("the sampling will be N = ",N)
    #     #we move all to the "sheap scale"
    #     scaled = self.max_flux  # / (10**exp_factor)
    #     #we re scale all to avoid numerical issues
    #     norm_spec = self.spec.at[:, [1, 2], :].divide(jnp.moveaxis(jnp.tile(scaled, (2, 1)), 0, 1)[:, :, None])
    #     norm_spec = norm_spec.at[:,2,:].set(jnp.where(self.mask, 1e31,norm_spec[:,2,:]))
    #     dependencies = self.dependencies 
    #     idxs = mapping_params(
    #         self.params_dict, [["amplitude"], ["scale"]])  # check later on cont how it works
    #     print(idxs)
    #     params = self.params.at[:, idxs].divide(scaled[:, None])
    #     wl, flux, yerr = jnp.moveaxis(norm_spec, 0, 1)
    #     model = self.model
    #     idx_target = [i[1] for i in dependencies]
    #     idx_free_params = list(set(range(len(params[0])))-set(idx_target))
    #     key = random.PRNGKey(0) # i should look for this?
    #     def apply_one_sample(free_sample):
    #         return apply_tied_and_fixed_params(free_sample,params[0],dependencies)
        
    #     full_samples_n = jnp.zeros((params.shape[0],N,params.shape[1]))
    #     #print(full_samples_n.shape)
    #     map_cont = self.kinds_map['continuum']
    #     idx_cont = jnp.array(list(map_cont.filtered_dict.values()))
    #     #print(idx_cont)
    #     profile_func = map_cont.profile_functions_combine
    #     #maybe this also should be safe in some place? well 
    #     bolometric_corrections = {
    #         "1350": 3.81,
    #         "1450": 3.81,
    #         "3000": 5.15,
    #         "5100": 9.26,
    #         "6200": 9.26,
    #     }
    #     #1 at the moment  can be updated
    #     estimators = {
    #                 "Hbeta": {
    #                     "wavelength": "5100",
    #                     "a": 6.91,
    #                     "b": 0.5,
    #                     "f": 1,
    #                 },  # Vestergaard & Peterson 2006
    #                 "MgII": {"wavelength": "3000", "a": 6.86, "b": 0.5, "f": 1},  # Shen et al. 2011
    #                 "CIV": {
    #                     "wavelength": "1350",
    #                     "a": 6.66,
    #                     "b": 0.53,
    #                     "f": 1,
    #                 },  # Vestergaard & Peterson 2006
    #                 "Halpha": {
    #                     "wavelength": "6200",
    #                     "a": 6.98,
    #                     "b": 0.5,
    #                     "f": 1,
    #                 },  # Greene & Ho 2005
    #                     }
    #     for n, (params_i, wl_i, flux_i, yerr_i) in enumerate(zip(params, wl, flux, yerr)):
    #         free_params = params_i[jnp.array(idx_free_params)]#we took only the free params
    #         res_fn = make_residuals_free_fn(model_func=model,
    #                                     xs=wl_i,y=flux_i,
    #                                     yerr=yerr_i,
    #                                     template_params=params_i,
    #                                     dependencies=dependencies)
    #         _, cov_matrix = error_covariance_matrix(residual_fn=res_fn,
    #                                                         params_i=free_params,
    #                                                         xs_i=wl_i,
    #                                                         y_i=flux_i,
    #                                                         yerr_i=yerr_i,
    #                                                         free_params=len(free_params),
    #                                                         return_full=True)
    #         L = jnp.linalg.cholesky(cov_matrix + 1e-6 * jnp.eye(cov_matrix.shape[0]))
    #         z = random.normal(key, shape=(N, len(free_params)))
    #         samples_free = free_params + z @ L.T  # (N, n_free)
    #         full_samples = jax.vmap(apply_one_sample)(samples_free)
    #         #we take the params to the original units.
    #         full_samples = full_samples.at[:,idxs].multiply(scaled[n])
    #         dict_ = {}
    #         for k, k_map in self.kinds_map.items():
    #             if k not in ['fe','continuum']:
    #                 idx_amplitude = mapping_params(k_map.filtered_dict, "amplitude")
    #                 idx_fwhm = mapping_params(k_map.filtered_dict, "fwhm")
    #                 idx_center = mapping_params(k_map.filtered_dict, "center")
                    
    #                 norm_amplitude = full_samples[:, idx_amplitude]
    #                 fwhm  = full_samples[:, idx_fwhm]
    #                 center  = full_samples[:, idx_center]
    #                 flux = jnp.sqrt(2.0 * jnp.pi) * norm_amplitude * fwhm/(2.0 * np.sqrt(2.0 * jnp.log(2.0))) #approx we will move to integration soon
    #                 fwhm_kms = (fwhm * self.c) / center
    #                 L = 4.0 * np.pi * np.array(self.d[n] ** 2) * flux * center
    #                 dict_[k] = {'lines': k_map.line_name, "component":np.array(k_map.component),'flux':flux,"fwhm":fwhm,"fwhm_kms":fwhm_kms,"L":L}
    #         L_w = {}
    #         L_bol = {}
    #         wavelenghts = [1350.0, 1450.0, 3000.0, 5100.0, 6200.0]
    #         cont_params = full_samples[:, idx_cont]
    #         for w in wavelenghts:
    #             #maybe just pass if not here but mmm
    #             wave = str(int(w))
    #             hits = jnp.isclose(norm_spec[n, 0, :], jnp.array([w]), atol=1)
    #             valid = (hits & (~self.mask[n])).any()
    #             corr = bolometric_corrections.get(wave, 0.0)
    #             if valid:
    #                 L_w[wave] = w * 4.0 * np.pi * np.array(self.d[n]**2) * vmap(profile_func, in_axes=(None, 0))(jnp.array([w]), cont_params).squeeze()
    #                 L_bol[wave] = L_w[wave] * corr
    #             else:
    #                 L_w[wave] = jnp.zeros(full_samples.shape[0])
    #                 L_bol[wave] = L_w[wave] * corr
    #         dict_broad = dict_.get("broad")
    #         fwhm_kms = dict_broad.get('fwhm_kms')
    #         line_name_list = np.array(dict_broad["lines"])
    #         masses = {}
    #         for line_name, estimators_i in estimators.items():
    #             wave = estimators_i["wavelength"]
    #             if line_name not in line_name_list or wave not in L_w.keys():
    #                 continue
    #             else:
    #                 idx_broad = np.where(line_name==line_name_list)[0]
    #             a, b, f = estimators_i["a"], estimators_i["b"], estimators_i["f"]
    #             log_L = jnp.log10(L_w[wave])  # continuum luminosity in erg/s
    #             fwhm_kms_ = fwhm_kms[:,idx_broad].squeeze() # a matrix with all the values 
    #             log_FWHM = jnp.log10(fwhm_kms_) - 3  # convert FWHM to 10^3 km/s
    #             log_M_BH = a + b * (log_L - 44.0) + 2 * log_FWHM
    #             M_BH = (10 ** log_M_BH) / f  # in Msun
    #             masses[line_name] = M_BH.squeeze()
                
    #     return  L_w, L_bol,masses
