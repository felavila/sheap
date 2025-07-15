from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from astropy.cosmology import FlatLambdaCDM
from auto_uncertainties import Uncertainty
from jax import grad, jit,vmap


from sheap.MainSheap import Sheapectral
from sheap.DataClass.DataClass import FitResult

from sheap.Functions.utils import combine_auto
from sheap.Mappers.helpers import mapping_params
from .MonteCarloSampler import MonteCarloSampler
from .McMcSampler import McMcSampler

from .tools.constants import BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS,c

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
        BOL_CORRECTIONS = BOL_CORRECTIONS,
        SINGLE_EPOCH_ESTIMATORS = SINGLE_EPOCH_ESTIMATORS,
        c=c,
    ):
        if sheap is not None:
            self._from_sheap(sheap)
        elif fit_result is not None and spectra is not None:
            self._from_fit_result(fit_result, spectra, z)
        else:
            raise ValueError("Provide either `sheap` or (`fit_result` + `spectra`).")
        
        self.BOL_CORRECTIONS = BOL_CORRECTIONS
        self.SINGLE_EPOCH_ESTIMATORS = SINGLE_EPOCH_ESTIMATORS
        self.c = c
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
       
    def sample_montecarlo(self, num_samples: int = 2000, key_seed: int = 0,summarize=True, extra_products=True):
        """
        Run Monte Carlo parameter sampling (see MonteCarloSampler for details).
        Returns megafullsample, dic_posterior_params
        """
        sampler = MonteCarloSampler(self)
        if summarize:
            print("The samples will be summarize is you want to keep the samples summarize=False")
        return sampler.sample_params(num_samples=num_samples, key_seed=key_seed,summarize=summarize,extra_products=extra_products)
    
    
    def sample_mcmc(self,n_random = 0,num_warmup=500,num_samples=1000,summarize=True, extra_products=True):
        """
        Run mcmc using numpyro parameter sampling.
        Returns megafullsample, dic_posterior_params
        """
        sampler = McMcSampler(self)
        return sampler.sample_params(n_random=n_random,num_warmup=num_warmup,num_samples=num_samples,summarize=summarize,extra_products=extra_products)

    
    
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
            #print(k)
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

    def compute_Luminosity_w(self, wavelengths=[1350.0, 1450.0, 3000.0, 5100.0, 6200.0]):
        """
        Compute monochromatic luminosities (and uncertainties) at key wavelengths.
        Returns a dict with wavelength (as str) to value/error arrays.
        """
        map_cont = self.kinds_map['continuum']
        profile_func = map_cont.profile_functions_combine
        params = map_cont.params.T
        uncertainty_params = map_cont.uncertainty_params.T
        L_w = {}
        for w in wavelengths:
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
            wave = str(float(params["wavelength"]))
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
    
    
    def _from_sheap(self, sheap):
        self.spec = sheap.spectra
        self.z = sheap.z
        self.result = sheap.result

        result = sheap.result  # for convenience
        self.constraints = result.constraints
        self.params = result.params
        self.scale = result.scale
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
        self.model = jit(combine_auto(self.profile_functions)) #
        self.params_dict = result.params_dict
        self.dependencies = result.dependencies
        self.complex_class = result.complex_class
        
        

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
        self.model = jit(combine_auto(self.profile_functions)) #mmm
        self.params_dict = result.params_dict
        self.constraints = result.constraints