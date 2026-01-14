"""
ComplexSampler
===============

Post–fitting interface for extracting physical parameters and running
posterior sampling on spectral models.

This module defines :class:`ComplexSampler`, which acts as a high-level
wrapper around results from :class:`Sheapectral <sheap.Sheapectral.Sheapectral>` or :class:`ComplexResult <sheap.Core.ComplexResult>`.
It provides multiple strategies to handle parameters after fitting:

- **Single best-fit mode**  
  Extract physical quantities (flux, FWHM, EQW, luminosity, etc.)
  directly from the optimized parameters with propagated uncertainties.

- **Monte Carlo mode**  
  Generate pseudo-random realizations of parameters around the covariance
  matrix for uncertainty propagation.

- **Pseudo Monte Carlo mode**  
  Fast approximate sampler for posterior parameter exploration.

- **MCMC mode (NumPyro)**  
  Full Bayesian sampling of the posterior distribution using Hamiltonian
  Monte Carlo / NUTS.

Key Features
------------
- Organizes parameter arrays, constraints, and dependencies after a fit.
- Provides consistent access to spectra, masks, scaling factors, and model functions.
- Computes luminosity distances given redshifts and a cosmology.
- Interfaces with downstream tools (:class:`AfterFitParams`) to compute
  line and continuum physical properties.
- Exposes convenience methods:

  * :meth:`ComplexSampler.sample_single`
  * :meth:`ComplexSampler.montecarlosampler`
  * :meth:`ComplexSampler.sample_pseudomontecarlosampler`
  * :meth:`ComplexSampler.sample_mcmc`

Notes
-----
- By default, cosmology is set to ``FlatLambdaCDM(H0=70, Om0=0.3)``.
- Bolometric corrections and single-epoch estimators are loaded from
  :mod:`sheap.Utils.Constants`.
- This module will eventually centralize routines now duplicated in
  samplers and parameter estimation helpers.
"""

__author__ = 'felavila'


__all__ = [
    "ComplexSampler",]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

from jax import grad, jit,vmap


from sheap.Sheapectral.Sheapectral import Sheapectral
from sheap.Core import ComplexResult

from sheap.Profiles.Utils import make_fused_profiles


from sheap.Utils.Constants import DEFAULT_BOL_CORRECTIONS, DEFAULT_SINGLE_EPOCH_ESTIMATORS,c,cm_per_mpc


#TODO flat_param_indices_global is super difficult to know what it means.
#TODO here we have to move the entire subrutines for montecarlosampler/mcmcsampler and ParametersSingle to his respective places bc in general they require different subfunctions. 

class ComplexSampler:
    """
    Computes best-fit physical parameters and uncertainties for spectral regions.
    Provides Monte Carlo and MCMC posterior sampling.

    Parameters
    ----------
    sheap : Sheapectral, optional
        Configured Sheapectral instance with fit results.
    complexresult : ComplexResult, optional
        ComplexResult object containing parameters, uncertainties, and metadata.
    spectra : jnp.ndarray, optional
        Normalized spectra array, shape (n_objects, 3, n_pixels).
    z : jnp.ndarray, optional
        Redshift array for each object.
    cosmo : astropy.cosmology instance, optional
        Cosmology for distance calculations; defaults to FlatLambdaCDM(H0=70, Om0=0.3).
    BOL_CORRECTIONS : dict, optional
        Bolometric correction factors; defaults from module constant.
    SINGLE_EPOCH_ESTIMATORS : dict, optional
        Single-epoch estimators; defaults from module constant.
    c : float, optional
        Speed of light constant; defaults from module constant.

    Attributes
    ----------
    spec : jnp.ndarray
        Spectra array used for estimation.
    z : jnp.ndarray
        Redshifts for each spectrum.
    params : jnp.ndarray
        Best-fit parameter values.
    uncertainty_params : jnp.ndarray
        Uncertainty estimates for parameters.
    cosmo : ?
        Cosmology object for computing distances.
    d : ?
        Luminosity distances corresponding to `z` (in cm).
    BOL_CORRECTIONS : dict
        Bolometric correction lookup.
    SINGLE_EPOCH_ESTIMATORS : dict
        Single-epoch parameter estimators.
    c : float
        Speed of light.

    Methods
    -------
    sample_montecarlo(num_samples=2000, key_seed=0, summarize=True, extra_products=True)
        Run Monte Carlo sampling of physical parameters.

    sample_mcmc(n_random=0, num_warmup=500, num_samples=1000, summarize=True, extra_products=True)
        Run MCMC sampling via NumPyro.

    sample_single(extra_products=True)
        Compute parameter estimates without posterior sampling.

    _from_sheap(sheap)
        Initialize internal state from a Sheapectral object.

    _from_complexresult(result, spectra, z)
        Initialize internal state from ComplexResult and spectra.
    """
    def __init__(
        self,
        sheap: Optional["Sheapectral"] = None,
        complexresult: Optional["ComplexResult"] = None,
        spectra: Optional[jnp.ndarray] = None,
        z: Optional[jnp.ndarray] = None,
        cosmo=None,
        BOL_CORRECTIONS = None,
        SINGLE_EPOCH_ESTIMATORS = None,
        c=c,):
        """
        Initialize ParameterEstimation context.

        Parameters
        ----------
        sheap : Sheapectral, optional
            Use attributes from this Spectral fitting interface.
        complexresult : ComplexResult, optional
            Use direct fit results if `sheap` not provided.
        spectra : jnp.ndarray, optional
            Spectra corresponding to `complexresult`.
        z : jnp.ndarray, optional
            Redshifts for each spectrum.
        cosmo : ?
            Cosmology for computing luminosity distances.
        BOL_CORRECTIONS : dict, optional
            Bolometric corrections mapping.
        SINGLE_EPOCH_ESTIMATORS : dict, optional
            Single-epoch estimators mapping.
        c : float, optional
            Speed of light constant.
        """
        if sheap is not None:
            self._from_sheap(sheap)
        elif complexresult is not None and spectra is not None:
            self._from_complexresult(complexresult, spectra, z)
        else:
            raise ValueError("Provide either `sheap` or (`complexresult` + `spectra`).")
        if not BOL_CORRECTIONS:
            self.BOL_CORRECTIONS = DEFAULT_BOL_CORRECTIONS
        if not SINGLE_EPOCH_ESTIMATORS:
            self.SINGLE_EPOCH_ESTIMATORS = DEFAULT_SINGLE_EPOCH_ESTIMATORS
        self.c = c
        if self.z is None:
            print("None informed redshift, assuming zero.")
            self.z = np.zeros(self.spectra.shape[0])
        if cosmo is None:
            self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        else:
            self.cosmo = cosmo
        #depending on the version this could change after 7.0.0 this change    
        #self.d = self.cosmo.luminosity_distance(self.z).value * cm_per_mpc
        self.d = self.cosmo.luminosity_distance(self.z) * cm_per_mpc

    def sample_pseudomontecarlosampler(self, num_samples: int = 2000, key_seed: int = 0,summarize=True):
        """
        Run pseudomontecarlosamplerparameter sampling.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw.
        key_seed : int, optional
            Seed for random number generator.
        summarize : bool, optional
            If True, summarize posterior distributions.
        extra_products : bool, optional
            Whether to return additional derived products.

        Returns
        -------
        full_samples, summary_dict
            Array of samples and dictionary of summarized statistics.
        """
        from sheap.ComplexSampler.Samplers.PseudoMonteCarloSampler import PseudoMonteCarloSampler
        self.method = "pseudomontecarlos"
        sampler = PseudoMonteCarloSampler(self)
        if summarize:
            print("The samples will be summarize is you want to keep the samples summarize=False")
        return sampler.sample_params(num_samples=num_samples, key_seed=key_seed,summarize=summarize)
    
    def montecarlosampler(self, num_samples: int = 2000, key_seed: int = 0,summarize=True,return_only_draws=False,frac_box_sigma=0.02,k_sigma=0.3):
        """
        Run montecarlosampler sampling.

        Parameters
        ----------
        num_samples : int, optional
            Number of samples to draw.
        key_seed : int, optional
            Seed for random number generator.
        summarize : bool, optional
            If True, summarize posterior distributions.
        extra_products : bool, optional
            Whether to return additional derived products.

        Returns
        -------
        full_samples, summary_dict
            Array of samples and dictionary of summarized statistics.
        """
        from sheap.ComplexSampler.Samplers.MonteCarloSampler import MonteCarloSampler
        self.method = "montecarlo"
        sampler = MonteCarloSampler(self)
        if summarize:
            print("The samples will be summarize is you want to keep the samples summarize=False")
        return sampler.sample_params(num_samples=num_samples, key_seed=key_seed,summarize=summarize,return_only_draws=return_only_draws,frac_box_sigma=frac_box_sigma, k_sigma= k_sigma)
    
    def sample_mcmc(self,n_random = 0,num_warmup=500,num_samples=1000,summarize=True):
        """
        Run MCMC sampling using NumPyro.

        Parameters
        ----------
        n_random : int, optional
            Number of random initial chains.
        num_warmup : int, optional
            Number of warmup steps.
        num_samples : int, optional
            Number of MCMC samples.
        summarize : bool, optional
            If True, summarize the chains.
        extra_products : bool, optional
            Include extra derived products.

        Returns
        -------
        full_chain, summary_dict
            Array of MCMC samples and dictionary of statistics.
        """
        from sheap.ComplexSampler.Samplers.McMcSampler import McMcSampler
        self.method = "mcmc"
        sampler = McMcSampler(self)
        return sampler.sample_params(n_random=n_random,num_warmup=num_warmup,num_samples=num_samples,summarize=summarize)

    def sample_single(self,summarize=True):
        """
        Compute parameter estimates and uncertainties without sampling.

        Parameters
        ----------
        extra_products : bool, optional
            Include additional derived products.

        Returns
        -------
        summary_dict
            Dictionary of parameter estimates and uncertainties.
        """
        from sheap.ComplexParams.ComplexParams import ComplexParams
        self.method = "single"
        complexparams = ComplexParams(self)
        
        return complexparams.extract_params(summarize=summarize)
        
    def _from_sheap(self, sheap):
        """
        Initialize internal state from a Sheapectral instance.

        Parameters
        ----------
        sheap : Sheapectral
            Source of fit results and spectra.
        """
        self.spectra = sheap.spectra
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
        #self.fe_mode = self.model_keywords.get("fe_mode")
        self.model = jit(make_fused_profiles(self.profile_functions)) #
        self.params_dict = result.params_dict
        self.dependencies = result.dependencies
        self.complex_class = result.complex_class
        self.fitkwargs = result.fitkwargs
        self.initial_params = result.initial_params
        

    def _from_complexresult(self, result, spectra, z):
        """
        Initialize internal state from ComplexResult and spectra.

        Parameters
        ----------
        result : ComplexResult
            ComplexResult containing parameters and metadata.
        spectra : jnp.ndarray
            Spectra array corresponding to `result`.
        z : jnp.ndarray
            Redshifts for each spectrum.
        """
        self.spectra = spectra
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
        #self.fe_mode = self.model_keywords.get("fe_mode")
        self.model = jit(make_fused_profiles(self.profile_functions)) #mmm
        self.params_dict = result.params_dict
        self.constraints = result.constraints
        self.fitkwargs = result.fitkwargs
        self.initial_params = result.initial_params
