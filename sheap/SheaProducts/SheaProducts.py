"""
SheaProducts Handling
============================

Routines to post-process fitted or sampled parameter sets and compute
derived physical quantities.

This module provides the :class:`SheaProducts` class, which acts as a
bridge between raw fitting/sampling outputs (parameter vectors) and
scientifically useful quantities such as line fluxes, widths, equivalent
widths, luminosities, and single-epoch black hole mass estimators.

Main Features
-------------
- Unified interface to handle both:
  * **single best-fit parameters** (deterministic optimization), and
  * **sampled parameters** (Monte Carlo / MCMC posterior draws).
- Automatic grouping of parameters by spectral region and profile.
- Computation of:
  * line flux, FWHM, velocity width (km/s),
  * line centers, amplitudes, shape parameters,
  * equivalent width (EQW),
  * monochromatic and bolometric luminosities,
  * combined quantities (e.g. Hα+Hβ, Mg II+Fe, CIV blends).
- uncertainties propagation via :mod:`uncertainties`.

Public API
----------
- :class:`SheaProducts`:
    High-level handler that connects a :class:`ComplexSampler` result
    to physical parameter extraction.

Typical Workflow
----------------
1. Fit or sample spectra with :class:`RegionFitting` or a sampler.
2. Wrap the result in a :class:`ComplexSampler` instance.
3. Construct :class:`SheaProducts(samplerclass)` from it.
4. Call :meth:`SheaProducts.extract_params` to obtain dictionaries
    of physical line quantities, optionally summarized across samples.

Notes
-----
- The attribute ``method`` determines whether results are handled as
    ``"single"`` (best fit) or ``"sampled"`` (posterior draws).
- Many helpers internally rely on
    :func:`make_batch_fwhm_split[_with_error]`,
    :func:`make_integrator`, and profile-specific shape functions.
"""

__author__ = 'felavila'

__all__ = ["SheaProducts",]
from typing import Any, Callable, Dict, List, Optional, Tuple, Union,Iterable
from collections import defaultdict

import numpy as np 
import jax.numpy as jnp 
from jax import vmap,jit

from sheap.Profiles.Utils import make_fused_profiles
from sheap.SheaProducts.Utils.MasterCombineProfile import MasterCombineProfile
from sheap.SheaProducts.Utils.ExtractBasicParamsSampled import extract_basic_params_sampled
from sheap.SheaProducts.Utils.ExtractBasicParams import extract_basic_params_single
from sheap.Utils.Constants import DEFAULT_BOL_CORRECTIONS, DEFAULT_SINGLE_EPOCH_ESTIMATORS,DEFAULT_C_KMS

from sheap.SheaProducts.Utils.ExtractExtraParams import extra_params_functions


#TODO method still usefful ?
#TODO Fe ratio and Star cont ratio. with model-spectra-reconstruction
# fe_out = self.MSR.fe_integrated_flux(all_samples=samples,include_bestfit=False)
# stars = self.MSR.stars_Cont_5100(all_samples=samples)  
class SheaProducts:
    _BASE_REQUIRED = ("model", "dependencies", "spectra", "mask", "sheapmodel", "method", "d")

    def __init__(self,*,samplerclass: Optional[object] = None,model=None,dependencies=None,spectra=None,mask=None,sheapmodel=None,method=None,d=None,
        BOL_CORRECTIONS=None,SINGLE_EPOCH_ESTIMATORS=None,C_KMS=None, **extra,):

        self.BOL_CORRECTIONS = DEFAULT_BOL_CORRECTIONS if BOL_CORRECTIONS is None else BOL_CORRECTIONS
        self.SINGLE_EPOCH_ESTIMATORS = (DEFAULT_SINGLE_EPOCH_ESTIMATORS if SINGLE_EPOCH_ESTIMATORS is None else SINGLE_EPOCH_ESTIMATORS)
        self.C_KMS = DEFAULT_C_KMS if C_KMS is None else C_KMS
        
        if samplerclass is not None:
            self._from_any(samplerclass)

        manual = dict(model=model,dependencies=dependencies,spectra=spectra,mask=mask,sheapmodel=sheapmodel,method=method,d=d,)
        manual.update(extra)

        for name, value in manual.items():
            if value is None:
                continue
            if getattr(self, name, None) is None:
                setattr(self, name, value)

        self._require(self._BASE_REQUIRED)
       
        self.wavelength_grid = jnp.linspace(0, 20_000, 20_000)
        self.LINES_TO_COMBINE = ["Halpha", "Hbeta","MgII","CIV"]
        self.limit_velocity = 150.
        self.by_region = self.sheapmodel.group_by("region")
        self.full_cont_profile = jit(vmap(make_fused_profiles(np.concatenate([self.by_region[key].profile_functions for key in self.by_region.keys() if key in ["fe", "continuum", "host","balmer"]])), in_axes=(0, 0))) 
        self.full_cont_profile_NONE = jit(vmap(make_fused_profiles(np.concatenate([self.by_region[key].profile_functions for key in self.by_region.keys() if key in ["fe", "continuum", "host","balmer"]])), in_axes=(None,0))) 
        self.full_cont_idx = np.concatenate([self.by_region[key].flat_param_indices_global for key in self.by_region.keys() if key in ["fe", "continuum", "host","balmer"]])
        self.cont_profile = jit(vmap(self.by_region["continuum"].combined_profile, in_axes=(None, 0)))
        ##summarize_spectral_lines(sheapspectral.result.sheapmodel.lines) this have to be the way to handle this.
        n_broad = len(getattr(self.by_region.get("broad"), "lines", []))
        n_narrow = len(getattr(self.by_region.get("narrow"), "lines", []))
        self.MC = MasterCombineProfile(LINES_TO_COMBINE= self.LINES_TO_COMBINE,limit_velocity=self.limit_velocity
                                       ,C_KMS=self.C_KMS,full_cont_profile=self.full_cont_profile,ucont_params = None,
                                       full_cont_profile_NONE = self.full_cont_profile_NONE,n_broad=n_broad,n_narrow=n_narrow)
        
       # self.MSR = MoldelSpectraReconstruction(self, jit_compile=True)# <-- this should go with the sampler to be able of run it in one run
        
    def calculate_sheap_products_sampled(self,idx,samples,combine=True,extra_products=True,**kwargs):
        #full_samples -> samples
        #d->luminosity_distance?
        #wi,mi,samples,luminosity_distance = self.spectra[idx,0,:],self.mask[idx,:],samples,self.d[idx]
        wi = jnp.take(self.spectra[:, 0, :], idx, axis=0)  # shape: (Nidx, L)
        mi = jnp.take(self.mask, idx, axis=0)              # shape: (Nidx, L)
        luminosity_distance = jnp.take(self.d, idx, axis=0)
        #basic_params should be basic extraction.
        
        products = extract_basic_params_sampled(sheapmodel=self.sheapmodel,wavelength=wi,mask=mi,samples=samples,
                                                continuum_idx_all=self.full_cont_idx,cont_profile_all = self.full_cont_profile,
                                                    cont_profile=self.cont_profile,luminosity_distance=luminosity_distance,
                                                    BOL_CORRECTIONS =self.BOL_CORRECTIONS,C_KMS= self.C_KMS,wavelength_grid=self.wavelength_grid)


        if combine:
            full_cont_params = samples[:,self.full_cont_idx]
            all_params = self.MC.combine_both(products["basic_params"],luminosity_distance,full_cont_params) #<-
            products= {**all_params,**products}
        if extra_products:
            products = self._get_extraparams(products)
        return products
    

          
    def _get_extraparams(self,products):
        L_w,L_bol = products["L_w"],products["L_bol"]
        new_products = {}
        for key, local_result in products.items():
            if "basic_params" in key:
                key = key.replace("basic","extra")
                new_products[key] = extra_params_functions(local_result,L_w,L_bol,self.SINGLE_EPOCH_ESTIMATORS,self.C_KMS)
            else:
                pass
        products.update(new_products)
        return products
       
    def _from_any(self, src: object) -> None:
        for name in self._BASE_REQUIRED:
            setattr(self, name, getattr(src, name, None))

        if hasattr(src, "BOL_CORRECTIONS"):
            self.BOL_CORRECTIONS = src.BOL_CORRECTIONS
        if hasattr(src, "SINGLE_EPOCH_ESTIMATORS"):
            self.SINGLE_EPOCH_ESTIMATORS = src.SINGLE_EPOCH_ESTIMATORS
        if hasattr(src, "C_KMS"):
            self.C_KMS = src.C_KMS

    def _require(self, names: Iterable[str]) -> None:
        missing = [n for n in names if getattr(self, n, None) is None]
        if missing:
            raise ValueError(f"SheaProducts is missing required fields: {missing}")
    

    def calculate_sheap_products(self,combine=True,extra_products=True,**kwargs):
        "going to deprecated"
        params = jnp.asarray(self.sheapmodel.params, dtype=jnp.float32)
        uncertainty_params = jnp.asarray(self.sheapmodel.uncertainty_params, dtype=jnp.float32)
        spectra = jnp.asarray(self.spectra, dtype=jnp.float32)
        
        
        full_cont_profile = make_fused_profiles(np.concatenate([self.by_region[key].profile_functions for key in self.by_region.keys() if key in ["fe", "continuum", "host","balmer"]])) # <- ?
        products = extract_basic_params_single(spectra,self.mask,params,uncertainty_params,continuum_idx_all=self.full_cont_idx,
                                               luminosity_distance=self.d,sheapmodel=self.sheapmodel,cont_profile_all= full_cont_profile,
                                                BOL_CORRECTIONS =self.BOL_CORRECTIONS,C_KMS= self.C_KMS,wavelength_grid=self.wavelength_grid)
        if combine:
            full_cont_params = params[:,self.full_cont_idx] # <-jeje
            self.MC.ucont_params = uncertainty_params[:,self.full_cont_idx] # <-jeje
            self.MC.full_cont_profile = full_cont_profile
            all_params = self.MC.combine_both(products["basic_params"],self.d,full_cont_params) #<-
            products= {**all_params,**products}
        if extra_products:
            products = self._get_extraparams(products)
        return products