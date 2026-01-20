"""
Main sheap Interface
====================

Provides the `Sheapectral` class, the high-level entry point for
loading, correcting, fitting, and analyzing AGN spectra with *sheap*.

Contents
--------
- **Spectral I/O**: load spectra from arrays or files.
- **Corrections**: apply Galactic extinction and redshift corrections.
- **Modeling**: build complex spectral regions via `ComplexBuilder`.
- **Fitting**: run JAX-based optimization with `ComplexFitting`.
- **Posterior Sampling**: estimate parameters using single, pseudo-MC, MC, or MCMC.
- **Persistence**: save/load full state with pickle.
- **Visualization**: quicklook plotting and model visualization with `SheapPlot`.

Notes
-----
- Input spectra are expected in shape `(n_objects, 3[,4], n_pixels)`,
  with channels = (wavelength, flux, error[, wdisp]).
- Velocity resolution (FWHM) is computed from dispersion when available.
- Main workflow:

  .. code-block:: python

     sheap = Sheapectral("spectrum.fits", z=0.5, coords=(l, b))
     sheap.makecomplex(6500, 6600, n_narrow=1, n_broad=2)
     sheap.fitcomplex()
     sheap.posteriors(sampling_method="mcmc")

- Results are stored in `self.result` (`ComplexResult`).
"""

from __future__ import annotations

__author__ = 'felavila'


__all__ = [
    "Sheapectral",
    "logger",
]

import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np

from sheap.Core import SpectralLine,ComplexResult,ArrayLike

from sheap.Sheapectral.Utils.SpectralSetup import pad_error_channel,ensure_sfd_data
from sheap.ComplexFitting.ComplexFitting import ComplexFitting
from sheap.ComplexBuilder.ComplexBuilder import ComplexBuilder
from sheap.Plotting.SheapPlot import SheapPlot
from sheap.Utils.Constants  import c



logger = logging.getLogger(__name__)


#WE CAN mode fast, we will stay in this 32 to go faster. 

class Sheapectral:
    """
    Main interface class for loading, correcting, fitting, and analyzing AGN spectra.

    This class handles:
    - Spectral I/O and validation
    - Extinction and redshift correction
    - Spectral region definition and model building
    - JAX-based optimization and uncertainty estimation
    - Posterior sampling using Monte Carlo or MCMC
    - Saving/loading results from pickle
    - Quick visualization and result summaries

    Parameters
    ----------
    spectra : str or jnp.ndarray
        Input spectra. If a string, it should be a path to a file readable by `np.loadtxt`.
        Expected shape after parsing: (n_objects, 3[, or 4], n_pixels).
    z : float or jnp.ndarray, optional
        Redshift(s) for each spectrum. Scalar or 1D array of shape (n_objects,).
    coords : jnp.ndarray, optional
        Galactic coordinates (l, b) for extinction correction, shape (n_objects, 2).
    ebv : jnp.ndarray, optional
        E(B-V) values. If not provided, estimated from `coords` using the SFD map.
    names : list of str, optional
        Object names. Defaults to stringified index if not given.
    extinction_correction : {'pending', 'done'}, optional
        Whether to apply extinction correction during initialization.
    redshift_correction : {'pending', 'done'}, optional
        Whether to apply redshift correction during initialization.
    **kwargs
        Additional arguments passed to underlying utilities.

    Attributes
    ----------
    spectra : jnp.ndarray
        3D array with shape (n_objects, 3, n_pixels) [wavelength, flux, error].
    wdisp : jnp.ndarray or None
        Wavelength dispersion (if available) per pixel.
    fwhm_lambda : jnp.ndarray
        Instrumental resolution in Angstroms, if `wdisp` provided.
    fwhm_kms : jnp.ndarray
        Instrumental resolution in km/s.
    result : ComplexResult
        Output of the fitting routine, including parameters and metadata.
    complexbuild : ComplexBuilder
        Configuration used to build the model region.
    plotter : SheapPlot
        Plotting backend object.

    Methods
    -------
    complexmaker(xmin, xmax, n_narrow=1, n_broad=1, **kwargs)
        Create a model complex from line and continuum definitions.
    
    fitcomplex(...)
        Perform spectral model fitting using the configured complex.
    
    estimate_posteriors(...)
        Estimate posterior distributions using MC or MCMC or just give and estimation of the params.

    save_to_pickle(filepath)
        Save object state to a pickle file.

    from_pickle(filepath)
        Load a `Sheapectral` instance from a saved pickle.

    result_panda(n)
        Return a Pandas DataFrame of the fit parameters for object `n`.

    quicklook(idx, ax=None, xlim=None, ylim=None)
        Plot flux + error for spectrum `idx`.

    modelplot
        Return or initialize plotting interface (SheapPlot).
    """
    def __init__(
        self,
        spectra: Union[str, jnp.ndarray],
        z: Optional[Union[float, jnp.ndarray]] = None,
        coords: Optional[jnp.ndarray] = None,
        ebv: Optional[jnp.ndarray] = None,
        names: Optional[list[str]] = None,
        extinction_correction: str = "pending",  # this only can be pending or done
        redshift_correction: str = "pending",  # this only can be pending or done
        **kwargs,):
        
        """
        Initialize Sheapectral object, load and optionally correct spectra.

        Parameters
        ----------
        spectra : str or jnp.ndarray
            Path to data file or array of raw spectra.
        z : float or jnp.ndarray, optional
            Redshift(s) to apply; repeated if scalar.
        coords : ?
            Coordinates for extinction map lookup.
        ebv : ?
            E(B-V) values, overrides coords-based estimation.
        names : list of str, optional
            Names for each spectrum.
        extinction_correction : {'pending', 'done'}, optional
            Control flag for extinction step.
        redshift_correction : {'pending', 'done'}, optional
            Control flag for redshift step.
        **kwargs : ?
            Additional parameters passed internally.
        """
        
        self.log = logging.getLogger(self.__class__.__name__)
        self.extinction_correction = extinction_correction
        self.redshift_correction = redshift_correction
        self.wdisp = None
        spec_arr = self._load_spectra(spectra)
        if spec_arr.shape[1] == 4:
            self.wdisp = spec_arr[:,3,:]
            spec_arr = spec_arr[:,[0,1,2],:]
        spec_arr = pad_error_channel(spec_arr)
        self.spectra = spec_arr#.astype(jnp.float32)#
        if self.wdisp is not None:
            #Velocity scale in km/s per pixel (eq.8 of Cappellari 2017)
            #This aprouch only is usseful for log sample spectra
            # Resolution fwhm_lambda of every pixel, in Angstroms
            self.velscale = np.log(np.atleast_2d(self.spectra[:,0,-1]/self.spectra[:,0,0]).T)/(self.spectra.shape[2]- 1 ) * c
            self.dlam = np.gradient(self.spectra[:,0,:],axis=1)   
            self.fwhm_lambda = 2.355 * self.wdisp * self.dlam #A
            self.fwhm_kms = self.fwhm_lambda / self.spectra[:,0,:] * c
            # in cases without wdisp 
            #     
        self.coords = coords  # may be None – handle carefully downstream
        self.ebv = ebv
        self.z = self._prepare_z(z, self.spectra.shape[0])

        self.names = (np.atleast_1d(names) if names is not None else np.arange(self.spectra.shape[0]).astype(str))
        if self.names.shape[0] !=self.spectra.shape[0]:
            print(f"The number of names ({len(self.names.shape[0])}) is different from the number of spectra ({self.spectra.shape[0]}) the code will use the inner names")
            self.names = np.arange(self.spectra.shape[0]).astype(str)
        #print(self.names.shape,self.spectra.shape)
        
        if self.extinction_correction == "pending" and (self.coords is not None or self.ebv is not None):
            
            print("extinction correction will be do it, change 'extinction_correction' to done if you want to avoid this step")
            self._apply_extinction()
            self.extinction_correction = "done"

        if self.redshift_correction == "pending" and self.z is not None:
            print("redshift correction will be do it, change 'redshift_correction' to done if you want to avoid this step")
            self._apply_redshift()
            self.redshift_correction = "done"

        self.sheap_set_up()
        self.default_limits = (float(np.min(self.spectra[:,0,:])),float(np.max(self.spectra[:,0,:])))

    def _load_spectra(self, spectra: Union[str, ArrayLike]) -> jnp.ndarray:
        """
        Load spectra from file or array.

        Parameters
        ----------
        spectra : str, Path, np.ndarray, list, or jnp.ndarray
            Input data source.

        Returns
        -------
        jnp.ndarray
            Array of shape (n_objects, channels, n_pixels).

        Raises
        ------
        TypeError
            If input type is unsupported.
        """        
        if isinstance(spectra, (str, Path)):
            arr = np.loadtxt(spectra)
            return jnp.array(arr).T  # ensure (c, λ) then transpose later
        elif isinstance(spectra, np.ndarray):
            return jnp.array(spectra)
        elif isinstance(spectra,list):
            return jnp.array(spectra)
        elif isinstance(spectra, jnp.ndarray):
            return spectra
        raise TypeError("spectra must be a path or ndarray")

    def _prepare_z(
        self, z: Optional[Union[float, ArrayLike]], nobj: int
    ) -> Optional[jnp.ndarray]:
        """
        Normalize redshift input to array form.

        Parameters
        ----------
        z : float, array-like, or None
            Input redshift(s).
        nobj : int
            Number of spectra objects.

        Returns
        -------
        jnp.ndarray or None
            Array of length nobj or None if z was None.
        """
        if z is None:
            return None
        if isinstance(z, (int, float)):
            return jnp.repeat(z, nobj)
        return jnp.array(z)

    def _apply_extinction(self) -> None:
        """
        Apply Galactic extinction correction to the flux and error channels.

        Uses Cardelli et al. (1989) law; if coords provided, uses SFD map.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        from sfdmap2 import sfdmap
        from sheap.Sheapectral.Utils.BasicCorrections import unred
        ebv = self.ebv
        if self.coords is not None:
            self.coords = jnp.array(self.coords)
            l, b = self.coords.T  # type: ignore[union-attr]
            sfd_path = Path(__file__).resolve().parent.parent / "SuportData" / "sfddata/"
            ensure_sfd_data(sfd_path)
            ebv_func = sfdmap.SFDMap(sfd_path).ebv
            ebv = ebv_func(l, b)
        corrected = unred(*np.swapaxes(self.spectra[:, [0, 1], :], 0, 1), ebv)
        # propagate to error channel proportionally as pyqso
        ratio = corrected / self.spectra[:, 1, :]
        self.spectra = self.spectra.at[:, 1, :].set(corrected)
        self.spectra = self.spectra.at[:, 2, :].multiply(ratio)

    def _apply_redshift(self) -> None:
        """
        Apply redshift correction (deredshift) to wavelength axis.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        from sheap.Sheapectral.Utils.BasicCorrections import deredshift
        self.spectra = deredshift(self.spectra, self.z)

    def sheap_set_up(self):
        """
        Ensure spectra have leading object axis, record shape and NaN mask.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if len(self.spectra.shape) <= 2:
            self.spectra = self.spectra[jnp.newaxis, :]
        self.spectra_shape = self.spectra.shape  # ?
        self.spectra_nans = jnp.isnan(self.spectra)
    
    def makecomplex(self,limits: tuple = None ,n_narrow: int = 1,n_broad: int = 1,group_method=True,
                    add_balmer_continuum = True ,add_balmerhighorder_continuum = True ,**kwargs):
        """
        Initialize a ComplexBuilder for later fitting.

        Parameters
        ----------
        limits : tuple  xmin,xmax
        
        n_narrow : int, optional
            Number of narrow components per line.
        n_broad : int, optional
            Number of broad components per line.
        **kwargs : ?
            Additional ComplexBuilder options.

        Returns
        -------
        None
        """
        if not limits:
            print(f"We will use the defualt limits {self.default_limits}")
            xmin,xmax = self.default_limits
        else:
            xmin,xmax = min(limits),max(limits)
        if xmin < 3600 and add_balmer_continuum:
            add_balmer_continuum = add_balmer_continuum
        
        if (3700 > xmin and 4000 < xmax) and add_balmerhighorder_continuum:    
            add_balmerhighorder_continuum = add_balmerhighorder_continuum
        self.complexbuild = ComplexBuilder(xmin=xmin,xmax=xmax,n_narrow=n_narrow,n_broad=n_broad,group_method=group_method,
                                        add_balmerhighorder_continuum=add_balmerhighorder_continuum, add_balmer_continuum= add_balmer_continuum, **kwargs)
    

    def fitcomplex(self,run_fit=True, list_num_steps=None,list_learning_rate = None ,covariance_error = False,profile: str ='gaussian'
                ,add_penalty_function=False,method="adam",penalty_weight: float = 0.00
                ,curvature_weight: float = 0.0,smoothness_weight: float = 0.0,max_weight: float = 0.0):
        """
        Execute fitting of the prepared region on the spectra.

        Parameters
        ----------
        list_num_steps : list of int, optional
            Maximum optimization steps per routine stage.
        run_uncertainty_params : bool, optional
            Whether to compute parameter uncertainties.
        profile : str, optional
            Line profile type for fitting.
        list_learning_rate : list of float, optional
            Learning rates for each stage.
        run_fit : bool, optional
            If False, construct the ComplexFitting object without fitting.
        add_penalty_function : bool, optional
            If True, include host-model penalty.

        Raises
        ------
        RuntimeError
            If make_region() was not called first.

        Returns
        -------
        None
        """
        if not hasattr(self, "complexbuild"):
            raise RuntimeError("makecomplex() must be called before fitcomplex()")

        self.fitting_class = ComplexFitting.from_builder(self.complexbuild,limits_overrides=None,profile=profile) #until here only uses the things that it knows from complexbuild

        spectra = self.spectra.astype(jnp.float32)
        if run_fit:
            Warning(f"You selected run_fit = {run_fit}, if you want run the fit change to True")
            self.fitting_class(spectra,list_num_steps = list_num_steps,list_learning_rate =list_learning_rate,
                            covariance_error= covariance_error,add_penalty_function=add_penalty_function,method=method,
                                penalty_weight= penalty_weight, curvature_weight= curvature_weight,
                                        smoothness_weight= smoothness_weight,max_weight= max_weight)

            self.spectral_model = self.fitting_class.model #the actual model is
            self.params_obj = self.fitting_class.params_obj
            #build_Parameters(tied_map,self.params_dict,self.initial_params,self.constraints)
            
            fit_output = self.fitting_class.complexresult
            fit_output.source = "computed"
            self.result = ComplexResult(
                params=fit_output.params.astype(jnp.float64),
                uncertainty_params=fit_output.uncertainty_params,
                mask=fit_output.mask,
                profile_functions=fit_output.profile_functions,
                profile_names=fit_output.profile_names,
                loss=fit_output.loss,
                profile_params_index_list=fit_output.profile_params_index_list,
                initial_params=fit_output.initial_params.astype(jnp.float32),
                scale=fit_output.scale,
                params_dict=fit_output.params_dict,
                complex_region=fit_output.complex_region,
                outer_limits=fit_output.outer_limits,
                inner_limits=fit_output.inner_limits,
                model_keywords= fit_output.model_keywords,
                fitting_routine = fit_output.fitting_routine,
                constraints = fit_output.constraints.astype(jnp.float64),
                source=fit_output.source,
                dependencies=fit_output.dependencies,
                residuals = fit_output.residuals,
                free_params = fit_output.free_params,
                chi2_red = fit_output.chi2_red,
                fitkwargs = fit_output.fitkwargs)

            self.plotter = SheapPlot(sheap=self)
    
    def estimate_posteriors(self,sampling_method="single", num_samples: int = 2000, key_seed: int = 0,summarize=True,overwrite=False, num_warmup=500,n_random=1_000,frac_box_sigma=0.02,k_sigma=0.3):
        """
        Estimate or sample posterior distributions of fit parameters.

        Parameters
        ----------
        sampling_method : {'single', 'pseudomontecarlo', 'mcmc',"montecarlo"}
            Sampling algorithm to use.
        num_samples : int, optional
            Number of samples to draw.
        key_seed : int, optional
            Random seed for reproducibility.
        summarize : bool, optional
            If True, compute summary statistics.
        overwrite : bool, optional
            If True, rerun even if posterior exists.
        num_warmup : int, optional
            Warm-up steps for MCMC.
        n_random : int, optional
            Number of initial random positions for MCMC.
        extra_products : bool, optional
            Return additional diagnostics.

        Returns
        -------
        ParameterEstimation or dict
            Posterior object or results dictionary.

        Raises
        ------
        RuntimeError
            If fit has not been run (`self.result` missing).
        """
        from sheap.ComplexSampler.ComplexSampler import ComplexSampler
        if not hasattr(self, "result"):
            raise RuntimeError("self.result should exist to run this.")
        #TODO ADD break in case sampling method is not recognize 
        PM = ComplexSampler(sheap = self)
        if sampling_method == "none":
            print("Nothing will run if you dont choose between sampling_method [montecarlo or sampling_method=mcmc or sampling_method=single")
            return PM 
        
        else:
            if self.result.posterior is None:
                self.result.posterior = {}

            method = sampling_method.lower()

            # --- Protection against overwriting ---
            if method in self.result.posterior and not overwrite:
                raise RuntimeError(f"Posterior for method '{method}' already exists. " "Use overwrite=True to recompute it.")
            # ------------------------------------------------------------------
            # SINGLE (no sampling)
            # ------------------------------------------------------------------
            if method == "single":
                print("You chose no_sampling: parameter estimation using " "only fitting uncertainties.")

                dic_posterior_params = PM.sample_single(summarize=summarize)

                self.result.posterior[method] = {"posterior_result": dic_posterior_params,"summarize": summarize,}
            # ------------------------------------------------------------------
            # PSEUDO MONTE CARLO
            # ------------------------------------------------------------------
            elif method == "pseudomontecarlo":
                dic_posterior_params = PM.sample_pseudomontecarlosampler(num_samples=num_samples,key_seed=key_seed,summarize=summarize,)
                self.result.posterior[method] = {"posterior_result": dic_posterior_params,"num_samples": num_samples,"key_seed": key_seed,"summarize": summarize,}
            # ------------------------------------------------------------------
            # MONTE CARLO
            # ------------------------------------------------------------------
            elif method == "montecarlo":
                dic_posterior_params = PM.montecarlosampler(num_samples=num_samples,key_seed=key_seed,summarize=summarize,frac_box_sigma=frac_box_sigma,k_sigma=k_sigma)
                self.result.posterior[method] = {"posterior_result": dic_posterior_params,"num_samples": num_samples,"key_seed": key_seed,"summarize": summarize}
            # ------------------------------------------------------------------
            # MCMC
            # ------------------------------------------------------------------
            elif method == "mcmc":
                dic_posterior_params = PM.sample_mcmc(num_samples=num_samples,n_random=n_random,num_warmup=num_warmup,summarize=summarize,)
                self.result.posterior[method] = {"posterior_result": dic_posterior_params,"num_samples": num_samples,"num_warmup": num_warmup,"n_random": n_random,"summarize": summarize,}

            # ------------------------------------------------------------------
            # UNKNOWN METHOD
            # ------------------------------------------------------------------
            else:
                raise ValueError(
                    f"Unknown sampling method '{sampling_method}'. "
                    "Available methods: single, pseudomontecarlo, montecarlo, mcmc."
                )


                                        
    
    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> Sheapectral:
        """
        Load a saved Sheapectral instance from a pickle file.

        Parameters
        ----------
        filepath : str or Path
            Path to the pickle file created by save_to_pickle().

        Returns
        -------
        Sheapectral
            Restored object with loaded spectra and results.
        """
        from sheap.Profiles.Utils import make_fused_profiles
        
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            spectra=data["spectra"],
            z=data["z"],
            names=data["names"],
            coords=data["coords"],
            extinction_correction=data["extinction_correction"],
            redshift_correction=data["redshift_correction"],
        )

        complex_region = data.get("complex_region", [])
        obj.complex_region = [SpectralLine(**i) for i in complex_region]

        profile_names = data.get("profile_names", [])
        obj.result = ComplexResult(
            params=jnp.array(data.get("params")),
            uncertainty_params=jnp.array(data.get("uncertainty_params", jnp.zeros_like(data.get("params")))), 
            initial_params=jnp.array(data.get("initial_params")),
            mask=jnp.array(data.get("mask")),
            profile_functions= obj.profile_functions_from_complex_region(),
            #obj.profile_functions,
            profile_names=profile_names,
            loss=None,  # Not saved currently, could be added if needed
            profile_params_index_list=data.get("profile_params_index_list"),
            scale=data.get("scale"),  # Not saved currently, could be added if needed
            params_dict=data.get("params_dict"),
            complex_region=obj.complex_region,
            outer_limits=data.get("outer_limits"),
            inner_limits=data.get("inner_limits"),
            model_keywords=data.get("model_keywords"),
            dependencies = data.get("dependencies"),
            source=data.get("source", "pickle"),
            constraints = data.get('constraints'),
            fitting_routine = data.get("fitting_routine"),
            posterior = data.get("posterior"),
            chi2_red = data.get("chi2_red"),
            fitkwargs = data.get("fitkwargs")
        )
        obj.plotter = SheapPlot(sheap=obj)
        obj.spectral_model = make_fused_profiles(obj.result.profile_functions)
        return obj
    
    def _save(self):
        """
        Internal: assemble a dict of object state for pickling.

        Returns
        -------
        dict
            Keys/values for spectra, results, and metadata.
        """
        _complex_region = [i.to_dict() for i in self.result.complex_region]

        dic_ = {
            "names": self.names,
            "spectra": np.array(self.spectra),
            "coords": np.array(self.coords),
            "z": np.array(self.z),
            "extinction_correction": self.extinction_correction,
            "redshift_correction": self.redshift_correction,
            "params": np.array(self.result.params),
            "uncertainty_params": np.array(self.result.uncertainty_params),
            "initial_params": np.array(self.result.initial_params),  # explicitly saved
            "params_dict": self.result.params_dict,
            "mask": np.array(self.result.mask),
            "complex_region": _complex_region,
            "profile_params_index_list": self.result.profile_params_index_list,
            "profile_names": self.result.profile_names,
            "fitting_routine": self.result.fitting_routine,
            "outer_limits": self.result.outer_limits,
            "inner_limits": self.result.inner_limits,
            "model_keywords": self.result.model_keywords,
            "source": self.result.source,
            "scale":np.array(self.result.scale),
            'constraints':np.array(self.result.constraints),
            'dependencies': self.result.dependencies,
            'residuals' : np.array(self.result.residuals),
            'free_params' : self.result.free_params,
            'chi2_red' : np.array(self.result.chi2_red),
            "posterior" : self.result.posterior,
            "fitkwargs":self.result.fitkwargs
        }

        estimated_size = sys.getsizeof(pickle.dumps(dic_))
        print(f"Estimated pickle size: {estimated_size / 1024:.2f} KB")

        return dic_

    def save_to_pickle(self, filepath: Union[str, Path]):
        """
        Save the current object state to a pickle file (.pkl).

        Parameters
        ----------
        filepath : str or Path
            Destination path for the pickle.

        Returns
        -------
        None
        """
        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self._save(), f)

    
    def profile_functions_from_complex_region(self):
        """
        Recreate profile functions for each region component.

        Returns
        -------
        list of callables
            Profile model functions.
        """
        from sheap.Profiles.Profiles import PROFILE_FUNC_MAP
        profile_functions = []
        for _,sp in enumerate(self.complex_region):
            holder_profile = getattr(sp, "profile") # cant be none 
            if "SPAF" in holder_profile:
                if len(sp.profile.split("_")) == 2:
                    _, subprofile = sp.profile.split("_")
                else:
                    print("Warning this if u have an SPAF, you should have and subprofile otherwise it can be readed correctly")
                sm = PROFILE_FUNC_MAP["SPAF"](sp.center,sp.amplitude_relations,subprofile)
            elif sp.profile == "hostmiles":
                sm = PROFILE_FUNC_MAP[sp.profile](**sp.template_info)["model"]
            elif sp.profile == "template":
                sm =PROFILE_FUNC_MAP[sp.profile](**sp.template_info)["model"]
            else:
                sm = PROFILE_FUNC_MAP.get(holder_profile)
            profile_functions.append(sm)
        return profile_functions
    
    @property
    def modelplot(self):
        """
        Get or initialize the SheapPlot plotting interface.
        TODO modelplot or plotter?

        Returns
        -------
        SheapPlot
            Plotting backend for spectra and fit results.

        Raises
        ------
        RuntimeError
            If no fit result exists.
        """
        if not hasattr(self, "plotter"):
            if hasattr(self, "result"):
                self.plotter = SheapPlot(sheap=self)
            else:
                raise RuntimeError("No fit result found. Run `fitcomplex()` first.")
        return self.plotter
    
   
    def result_panda(self, n: int, param_filter: str | None = None,
                 regex: bool = False, case: bool = True) -> pd.DataFrame:
        """
        #TODO update this part to be able to show the actual name of the parameter lets say line with 0,1,2,3,4 is easier for code reason but for visualitation could be messy 
        #TODO say if the paramters are at scale or not.
        Return a pandas DataFrame of fit parameters for a given spectrum.

        Parameters
        ----------
        n : int
            Index of the spectrum object.
        param_filter : str, optional
            If provided, return only rows whose parameter name matches this
            pattern. Uses pandas .str.contains, so it can be a substring
            or a regex (see `regex` and `case`).
        regex : bool, default False
            If True, `param_filter` is interpreted as a regular expression.
        case : bool, default True
            If False, ignore case when matching `param_filter`.

        Returns
        -------
        pandas.DataFrame
            Index: parameter name.
            Columns: ['value', 'error', 'max_constraint',
                    'init_value', 'min_constraint'].
        """
        import pandas as pd 
        data = []
        scale = self.result.scale[n]

        for param_index,(param_name, i) in enumerate(self.result.params_dict.items()):
            param = float(self.result.params[n][i])
            init_value = float(self.result.initial_params[i])
            uncertainty = float(self.result.uncertainty_params[n][i])

            if "amplitude" in param_name:
                param /= scale
                uncertainty /= scale
            elif "logamp" in param_name:
                param -= np.log10(scale)
                # uncertainty is usually left as-is in log-space

            constraints = self.result.constraints[i]

            data.append({"param_index":param_index,"param_name": param_name, "value": param, "error": uncertainty, "max_constraint": constraints[1], "init_value": init_value,"min_constraint": constraints[0],})

        df = pd.DataFrame(data)

        # Optional filtering by parameter name (now using the column)
        if param_filter is not None:
            mask = df["param_name"].str.contains(param_filter, case=case, regex=regex, na=False)
            df = df[mask]

        return df

    def quicklook(self, idx: int, ax=None, xlim=None, ylim=None):
        """
        Produce a quick errorbar plot of flux vs. wavelength.

        Parameters
        ----------
        idx : int
            Spectrum index to plot.
        ax : matplotlib.axes.Axes, optional
            Axes to plot into (creates new if None).
        xlim : tuple, optional
            X-axis limits as (xmin, xmax).
        ylim : tuple, optional
            Y-axis limits as (ymin, ymax).

        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator

        lam, flux, err = self.spectra[idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 5))

        ax.errorbar(lam, flux, yerr=err, ecolor='dimgray', color="black", zorder=1)

        # Default xlim and ylim if not provided
        if xlim is None:
            xlim = (jnp.nanmin(lam), jnp.nanmax(lam))
        if ylim is None:
            ylim = (0, jnp.nanmax(flux) * 1.02)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Flux [arb]")

        # Plot ID label outside main plot area, above-left
        ax.text(
            0.0,
            1.05,
            f"ID {self.names[idx]} ({idx}), z = {self.z[idx]} ",
            fontsize=10,
            transform=ax.transAxes,
            ha='left',
            va='bottom',
        )

        ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))

        return ax

    #extra plots
    @property
    def plot_redshift_signal2noise_distribution(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        axes[0].hist(self.z, bins=20, color="#214994", edgecolor='black', alpha=0.8)
        axes[0].set_xlabel("Redshift (z)", fontsize=16)
        axes[0].set_ylabel("Number of objects", fontsize=16)
        #axes[0].set_title("Redshift Distribution", fontsize=18)
        axes[0].tick_params(axis='both', which='major', labelsize=14)

        snr = np.nanmean(self.spectra[:, 1, :] / self.spectra[:, 2, :], axis=1)
        axes[1].hist(snr, bins=20, color="#f5b041", edgecolor='black', alpha=0.8)
        axes[1].set_xlabel("Mean Signal-to-Noise Ratio", fontsize=16)
        #axes[1].set_title("S/N Distribution", fontsize=18)
        axes[1].tick_params(axis='both', which='major', labelsize=14)
        plt.tight_layout()
        return axes
    @property
    def plot_chi2(self):
        import matplotlib.pyplot as plt
        if not hasattr(self, "result"):
            raise RuntimeError("self.result should exist to run this.")
        chi2_model = self.result.chi2_red  # or whatever your model object is
        chi2_model = np.asarray(chi2_model)

        nan_mask = ~np.isfinite(chi2_model)
        nan_idx = np.where(nan_mask)[0]
        chi2_model = chi2_model[~nan_mask]
        print(f"NaN / non-finite entries at indices: {nan_idx.tolist()}")
        print(f"Number of NaNs / non-finite values: {nan_idx.size}")
        # --- Compute fraction in (0,5) ---
        mask_range_model = (chi2_model > 0.) & (chi2_model < 5.)
        frac_model_0_5 = np.nanmean(mask_range_model) * 100.0

        # --- Define bins from model only ---
        chi2_min = chi2_model.min()
        chi2_max = chi2_model.max()
        bins = np.linspace(chi2_min, chi2_max, 40)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            chi2_model,
            bins=bins,
            alpha=0.7,
            color="#d62728",
            label=fr"Model chi2 (median = {np.nanmedian(chi2_model):.2f})"
        )

        ax.set_xlabel(r"Reduced $\chi^2$", fontsize=18)
        ax.set_ylabel("Number of spectra", fontsize=18)
        ax.legend(fontsize=14, frameon=False)
        ax.tick_params(axis="both", labelsize=14)

        # Optional: vertical line at chi2=5
        ax.axvline(1.85, linestyle="--", linewidth=1)

        # --- Annotate fraction ---
        textstr = (
            fr"$0.<\chi^2_{{\rm red}}<5$ fraction: " 
            fr"Model: {frac_model_0_5:.1f}%"
        )

        ax.text(
            0.3, 0.5,
            textstr,
            transform=ax.transAxes,
            fontsize=20,
            va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
        )

        fig.tight_layout()
        plt.show()
        #plt.tight_layout()
        #plt.show()


# def _region_helper(region_name):
#             if region_name not in complex_class_group_by_region.keys():
#                 return 0
#             _combined_profile  = complex_class_group_by_region[region_name].combined_profile
#             params = complex_class_group_by_region[region_name].params
#             return vmap(_combined_profile,(0,0))(self.spectra[:,0,:],params)