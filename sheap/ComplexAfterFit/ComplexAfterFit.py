"""This module ?."""
__author__ = 'felavila'

# Auto-generated __all__
__all__ = [
    "ComplexAfterFit",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from auto_uncertainties import Uncertainty
from jax import grad, jit,vmap


from sheap.MainSheap.Sheapectral import Sheapectral
from sheap.Core import ComplexResult

from sheap.Profiles.utils import make_fused_profiles


from sheap.Utils.Constants import BOL_CORRECTIONS, SINGLE_EPOCH_ESTIMATORS,c,cm_per_mpc



#TODO here we have to move the entire subrutines for montecarlosampler/mcmcsampler and ParametersSingle to his respective places bc in general they require different subfunctions. 

class ComplexAfterFit:
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
    fluxnorm : array-like, optional
        Flux normalization factors; defaults to ones.
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
    fluxnorm : array-like
        Flux normalization factors.
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
        fluxnorm=None,
        cosmo=None,
        BOL_CORRECTIONS = BOL_CORRECTIONS,
        SINGLE_EPOCH_ESTIMATORS = SINGLE_EPOCH_ESTIMATORS,
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
        fluxnorm : array-like, optional
            Flux normalization per object.
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
       

    def sample_pseudomontecarlosampler(self, num_samples: int = 2000, key_seed: int = 0,summarize=True, extra_products=True):
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
        from sheap.ComplexAfterFit.Samplers.PseudoMonteCarloSampler import PseudoMonteCarloSampler
        self.method = "pseudomontecarlos"
        sampler = PseudoMonteCarloSampler(self)
        if summarize:
            print("The samples will be summarize is you want to keep the samples summarize=False")
        return sampler.sample_params(num_samples=num_samples, key_seed=key_seed,summarize=summarize,extra_products=extra_products)
    
    def montecarlosampler(self, num_samples: int = 2000, key_seed: int = 0,summarize=True, extra_products=True):
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
        from sheap.ComplexAfterFit.Samplers.MonteCarloSampler import MonteCarloSampler
        self.method = "montecarlo"
        sampler = MonteCarloSampler(self)
        if summarize:
            print("The samples will be summarize is you want to keep the samples summarize=False")
        return sampler.sample_params(num_samples=num_samples, key_seed=key_seed,summarize=summarize,extra_products=extra_products)
    
    def sample_mcmc(self,n_random = 0,num_warmup=500,num_samples=1000,summarize=True, extra_products=True):
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
        from sheap.ComplexAfterFit.Samplers.McMcSampler import McMcSampler
        self.method = "mcmc"
        sampler = McMcSampler(self)
        return sampler.sample_params(n_random=n_random,num_warmup=num_warmup,num_samples=num_samples,summarize=summarize,extra_products=extra_products)

    def sample_single(self,extra_products=True):
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
        from sheap.ComplexAfterFit.AfterFitParams import AfterFitParams
        self.method = "single"
        afterfitparams = AfterFitParams(self)
        
        return afterfitparams.extract_basic_params()
        
    def _from_sheap(self, sheap):
        """
        Initialize internal state from a Sheapectral instance.

        Parameters
        ----------
        sheap : Sheapectral
            Source of fit results and spectra.
        """
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
        #self.fe_mode = self.model_keywords.get("fe_mode")
        self.model = jit(make_fused_profiles(self.profile_functions)) #mmm
        self.params_dict = result.params_dict
        self.constraints = result.constraints
        self.fitkwargs = result.fitkwargs
        self.initial_params = result.initial_params
