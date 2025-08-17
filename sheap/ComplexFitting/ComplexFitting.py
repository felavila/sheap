"""This module ."""
from __future__ import annotations
__author__ = 'felavila'


__all__ = [
    "ComplexFitting",
    "logger",
]

import logging
#from dataclasses import dataclass
#from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import jax.numpy as jnp
import numpy as np
from jax import jit,vmap

from sheap.Core import FittingLimits, SpectralLine,ComplexResult

from sheap.Assistants.Parameters import build_Parameters
from sheap.Assistants.parser_mapper import mapping_params,parse_dependencies,make_get_param_coord_value,build_tied,flatten_tied_map,parse_dependencies


from sheap.Minimizer.Minimizer import Minimizer

from sheap.Profiles.profiles import PROFILE_FUNC_MAP,PROFILE_CONTINUUM_FUNC_MAP
from sheap.Profiles.profile_handler import ProfileConstraintMaker 
from sheap.Profiles.utils import make_fused_profiles,build_grid_penalty


from sheap.Utils.SpectralSetup import mask_builder, prepare_spectra
from sheap.Utils.Constants import DEFAULT_LIMITS

from sheap.ComplexAfterFit.UncertaintyFunction import Errorfromloop

# Configure module-level logger
logger = logging.getLogger(__name__)


class ComplexFitting:
    """
    Fits a spectral region containing multiple emission lines.

    This class wraps the workflow of:
      - Building parameterized line + continuum models
      - JAX‑based fitting (with optional penalty functions)
      - Uncertainty estimation via covariance or sampling
      - Post‑processing (renormalization, χ² calculation)
      - Packaging results into a ComplexResult object

    Parameters
    ----------
    region_dict : dict
        Dictionary of attributes produced by a ComplexBuilder, including:
        - `complex_class` (with `.lines` list)
        - `fitting_routine` (dict of fit steps and ties)
        - any other metadata needed for fitting
    profile : str, optional
        Default line profile to use for unlabeled components
        (e.g. 'gaussian', 'lorentzian', 'SPAF'), by default "gaussian"
    limits_overrides : dict[str, FittingLimits], optional
        Overrides for the default parameter‑limit lookup, by species or region.

    Attributes
    ----------
    profile : str
        Profile name used for unconstrained components.
    limits_map : dict[str, FittingLimits]
        Per‑species limits (lower/upper) for fit parameters.
    params_dict : dict[str, int]
        Mapping from parameter names to their index in the packed parameter vector.
    initial_params : jnp.ndarray, shape (n_params,)
        Initial guesses for all fit parameters.
    constraints : jnp.ndarray, shape (n_params, 2)
        Lower and upper bounds for each parameter.
    profile_functions : list[callable]
        JAX‑compiled model functions for each line/continuum component.
    model : callable
        Fused, jit‑compiled model combining all profile_functions.
    host_info : dict
        Extra metadata (e.g. stellar‑population grid) for penalty construction.
    complexresult : ComplexResult
        Final results (parameters, uncertainties, residuals, χ², etc.), set after fitting.

    Methods
    -------
    __call__(spectra, force_cut=False, run_uncertainty_params=True,
              inner_limits=None, outer_limits=None,
              learning_rate=None, add_penalty_function=False)
        Execute the full fit on one or more spectra. Raises if limits or
        routine are mis‑specified.

    _fit(iteration_number, norm_spec, model, initial_params,
         tied, learning_rate=1e-1, weighted=True,
         num_steps=1000, non_optimize_in_axis=3,
         penalty_function=None)
        Perform the JAX‑based minimization using Minimizer. Returns
        optimized parameters and final loss history.

    _prep_data(spectra, inner_limits, outer_limits, force_cut)
        Preprocess spectra: mask, cut region, normalize flux by max per pixel.

    _postprocess(norm_spec, params, uncertainty_params, scale)
        Scale fitted parameters back to original flux units and compute
        residuals, χ², and package intermediate arrays.

    _build_fit_components(profile="gaussian", **kwargs)
        Build parameter initialization lists, constraints, and profile functions
        from `complex_class.lines`.

    _build_tied(tied_params)
        Convert user‑specified tie lists into dependency strings for the minimizer.

    _stack_constraints(low, high)
        Stack lower & upper bound lists into an (n_params, 2) JAX array.

    _add_linear(idx)
        Add a linear continuum component if none was found in the region.

    to_result()
        Assemble a `ComplexResult` object from the final attributes.

    from_builder(builder, *, profile='gaussian', limits_overrides=None, **builder_kwargs)
        Alternate constructor: build region_dict via ComplexBuilder and return
        a new instance.

    init_linear(norm_spec, params)
        Compute and insert weighted least‑squares continuum slopes/intercepts.

    Notes
    -----
    - Uses JAX and a custom Minimizer for gradient‑based optimization.
    - Tied parameters are handled via the `_build_tied` helper.
    - Continuum can be injected via a linear fit or special continuum profiles.

    Examples
    --------
    >>> builder = ComplexBuilder(xmin=6500, xmax=6600, lines=['Halpha', 'NII'])
    >>> rf = ComplexFitting.from_builder(builder, profile='SPAF')
    >>> rf(spectra_array, inner_limits=(6520, 6580), outer_limits=(6500, 6600))
    >>> df = rf.pandas_params()
    >>> print(df.head())
    """

    def __init__(self, region_dict: dict, *, profile: str = "gaussian",limits_overrides: Optional[Dict[str, FittingLimits]] = None):
        
        """
        Initialize ComplexFitting with builder output and optional limits.

        Parameters
        ----------
        region_dict : dict
            Attributes from ComplexBuilder._make_fitting_routine(...)
        profile : str, optional
            Default profile type for unconstrained lines.
        limits_overrides : dict[str, FittingLimits], optional
            Overrides to DEFAULT_LIMITS per species or region.
        """
        
        self.profile = profile


        for key, val in region_dict.items():
            setattr(self, key, val)
        self.limits_map: Dict[str, FittingLimits] = {}
        for kind, cfg in DEFAULT_LIMITS.items():
            default_lim = FittingLimits.from_dict(cfg)
            # Use override if provided, else default
            self.limits_map[kind] = (
                limits_overrides[kind]
                if limits_overrides and kind in limits_overrides
                else default_lim
            )
        self.params_dict: Dict[str, int] = {}
        self.initial_params: jnp.ndarray = jnp.array([])
        self.profile_functions: List[Any] = []
        # self.profile_params_index: List[List[int]] = []
        self.profile_names: List[str] = []
        self.profile_params_index_list: List[List[int]] = []
        # self.list_dependencies: List[str] = []
        self.constraints: Optional[jnp.ndarray] = None
        self.params: Optional[jnp.ndarray] = None
        self.loss: Optional[float] = None
        self._build_fit_components(profile = profile)
        self.model = jit(make_fused_profiles(self.profile_functions))
        self.host_info = {}
    
    def __call__(
        self,
        spectra: Union[List[Any], jnp.ndarray],
        force_cut: bool = False,
        covariance_error = True,
        list_num_steps = None,
        list_learning_rate =None,
        inner_limits: Optional[Tuple[float, float]] = None, 
        outer_limits: Optional[Tuple[float, float]] = None,
        add_penalty_function = False,
        method = "adam",
        penalty_weight: float = 0.01,
        curvature_weight: float = 1e5,
        smoothness_weight: float = 0.0,
        max_weight: float = 0.1,
        ) -> None:
        """
        Execute the full fitting routine on provided spectra.

        Parameters
        ----------
        spectra : list or jnp.ndarray
            Input array of shape (n_spectra, 3, n_pixels).
        force_cut : bool, optional
            Force region cutting after mask is built.
        run_uncertainty_params : bool, optional
            Compute uncertainties via covariance matrix.
        inner_limits : tuple(float, float), optional
            Wavelength bounds for fitting.
        outer_limits : tuple(float, float), optional
            Wavelength bounds for masking.
        learning_rate : float or list, optional
            Learning rate(s) for each fitting step.
        add_penalty_function : bool, optional
            Add host‑grid penalty if host_info is available.

        Raises
        ------
        ValueError
            If inner/outer limits are not defined.
        TypeError
            If `fitting_routine` is not a dict.
        """
        # the idea is that is exp_factor dosent have the same shape of scale could be fully renormalice the spectra.
        print(f"Fitting {spectra.shape[0]} spectra with {spectra.shape[2]} wavelength pixels")
        
        _, mask, scale, norm_spec = self._prep_data(
            spectra, inner_limits, outer_limits, force_cut)

        inner_limits = self.inner_limits or inner_limits
        outer_limits = self.outer_limits or outer_limits
        params = jnp.tile(self.initial_params, (spectra.shape[0], 1))
        penalty_function = None 
        if add_penalty_function and self.host_info:
            print("Penalty function will be added.")
            weights_idx = mapping_params(self.params_dict,"weight")
            n_Z,n_age = (self.host_info[i] for i in ["n_Z","n_age"])
            penalty_function = build_grid_penalty(weights_idx,n_Z,n_age)
        
        if "linear" in self.profile_names:
            params = self.init_linear(norm_spec,params)            
        if not (self.inner_limits and self.outer_limits):
            raise ValueError("inner_limits and outer_limits must be specified")
        if not isinstance(self.fitting_routine, dict):
            raise TypeError("fitting_routine must be a dictionary.")
        #list_num_steps =
        #list_learning_rate = 
        if list_num_steps and list_learning_rate:
            assert len(list_num_steps) == len(list_learning_rate), "The  list_num_steps and list_learning_rate should be equal"
            n_steps = len(list_learning_rate)
        else:
            n_steps = len(list(self.fitting_routine.keys()))
        total_time = 0
        self._fitkwargs = []
        for _step in range(n_steps):
            key = f"step{_step+1}"
            step = self.fitting_routine[key]
            if isinstance(list_learning_rate,list):
                step["learning_rate"] = list_learning_rate[_step]
            if isinstance(list_num_steps,list):
                step["num_steps"] = list_num_steps[_step]
            print(f"\n{'='*40}\n{key.upper()} ({key}) free params {self.initial_params.shape[0]-len(step['tied'])}")
            step["non_optimize_in_axis"] = 4 #experimental 
            start_time = time.time()  # 
            params, loss = self._fit(_step,norm_spec, self.model, params, **step,penalty_function=penalty_function,method=method,
                                     penalty_weight = penalty_weight,
                                        curvature_weight = curvature_weight,
                                        smoothness_weight = smoothness_weight,
                                        max_weight = max_weight)
            uncertainty_params = jnp.zeros_like(params)
            end_time = time.time() 
            elapsed = end_time - start_time
            print(f"Time for step '{key}': {elapsed:.2f} seconds")
            total_time += elapsed
            self._fitkwargs.append({**step,"method":method,"penalty_weight" : penalty_weight,
                                            "curvature_weight" : curvature_weight,
                                            "smoothness_weight" : smoothness_weight,
                                            "max_weight" : max_weight})    
        dependencies = parse_dependencies(self._build_tied(step["tied"]))
        if covariance_error:
            print("\n==Running error_covariance_matrix==")
            start_time = time.time()  # 
            uncertainty_params = Errorfromloop(self.model,norm_spec,params,dependencies)
            end_time = time.time()  # 
            
            print(f"Time for error_covariance_matrix: {elapsed:.2f} seconds")
            total_time += elapsed            
        print(f'The entire process took {total_time:.2f} ({total_time/spectra.shape[0]:.2f}s by spectra)')
        self.dependencies = dependencies
        self.mask = mask
        self._postprocess(norm_spec, params, uncertainty_params, scale)
        self.loss = loss
        self.scale = scale
        self.outer_limits = outer_limits
        self.inner_limits = inner_limits
        self.to_result()
    
    
    def _fit(
        self,
        iteration_number: int,
        norm_spec: jnp.ndarray,
        model,
        initial_params,
        tied: List[List[str]],
        learning_rate=1e-1,
        weighted: bool = True,
        num_steps: int = 1000,
        non_optimize_in_axis=3,
        penalty_function = None,
        method = None,
        penalty_weight: float = 0.01,
        curvature_weight: float = 1e5,
        smoothness_weight: float = 0.0,
        max_weight: float = 0.1,
        verbose = True
    ) -> Tuple[jnp.ndarray, list]:
        """
        Perform the JAX‑based minimization using Minimizer.

        Parameters
        ----------
        iteration_number : int
            Index of the current fitting step.
        norm_spec : jnp.ndarray
            Normalized spectra array.
        model : callable
        initial_params : jnp.ndarray
        tied : list of list[str]
            Groups of parameter names to tie together.
        learning_rate : float, optional
        weighted : bool, optional
        num_steps : int, optional
        non_optimize_in_axis : int, optional
        penalty_function : callable, optional

        Returns
        -------
        params : jnp.ndarray
            Optimized parameter values.
        loss : list
            Loss history over iterations.

        Raises
        ------
        RuntimeError
            If minimizer encounters an error.
        """
        if verbose:
            print("learning_rate:",learning_rate,"num_steps:",num_steps,"non_optimize_in_axis:",non_optimize_in_axis,)
        list_dependencies = parse_dependencies(self._build_tied(tied))
        tied_map = {T[1]: T[2:] for  T in list_dependencies}
        tied_map = flatten_tied_map(tied_map)
        
        params_obj = build_Parameters(tied_map,self.params_dict,initial_params,self.constraints)
        raw_init = params_obj.phys_to_raw(initial_params)
        
        self.params_obj = params_obj
        minimizer = Minimizer(
            model,
            non_optimize_in_axis=non_optimize_in_axis,
            num_steps=num_steps,
            list_dependencies=list_dependencies,
            weighted=weighted,
            learning_rate=learning_rate,
            param_converter=params_obj,
            penalty_function = penalty_function,
        method=method,
        penalty_weight= penalty_weight,curvature_weight= curvature_weight,smoothness_weight= smoothness_weight,max_weight= max_weight)
        try:
            raw_params, loss = minimizer(raw_init, *norm_spec.transpose(1, 0, 2), self.constraints)
            params = params_obj.raw_to_phys(raw_params)
            
        except Exception as e:
            logger.exception("Fitting failed")
            raise RuntimeError(f"Fitting error: {e}")
        return params, loss


    def _prep_data(
        self,
        spectra: Union[List[Any], jnp.ndarray],
        inner_limits: Optional[Tuple[float, float]],
        outer_limits: Optional[Tuple[float, float]],
        force_cut: bool,
        #exp_factor,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Preprocess spectra for fitting.

        Parameters
        ----------
        spectra : list or jnp.ndarray
        inner_limits : tuple(float, float)
        outer_limits : tuple(float, float)
        force_cut : bool

        Returns
        -------
        spec : jnp.ndarray
        mask : jnp.ndarray
        scale : jnp.ndarray
        norm_spec : jnp.ndarray

        Raises
        ------
        ValueError
            On preprocessing or normalization errors.
        """
        
        self.inner_limits = inner_limits or self.inner_limits
        self.outer_limits = outer_limits or self.outer_limits
        
        if not (self.inner_limits and self.outer_limits):
            raise ValueError("inner_limits and outer_limits must be specified")
        
        try:
            if isinstance(spectra, list):
                spec, mask = prepare_spectra(spectra, outer_limits=self.outer_limits)
            else:
                spec, _, _, mask = mask_builder(spectra, outer_limits=self.outer_limits)
                if force_cut:
                    spec, mask = prepare_spectra(spec, outer_limits=self.outer_limits)
        except Exception as e:
            logger.exception("Failed to preprocess spectra")
            raise ValueError(f"Preprocessing error: {e}")

        try:
            scale = jnp.nanmax(jnp.where(mask, 0, spec[:, 1, :]), axis=1) 
            norm_spec = spec.at[:, [1, 2], :].divide(jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None])
            
        except Exception as e:
            logger.exception("Normalization error")
            raise ValueError(f"Normalization error: {e}")

        return spec, mask, scale, norm_spec
    
    def _postprocess(
        self,
        norm_spec: jnp.ndarray,
        params: jnp.ndarray,
        uncertainty_params: jnp.ndarray,
        scale: jnp.ndarray,
        ) -> None:
        """
        Scale parameters back to original flux units and compute diagnostics.

        Parameters
        ----------
        norm_spec : jnp.ndarray
        params : jnp.ndarray
        uncertainty_params : jnp.ndarray
        scale : jnp.ndarray

        Raises
        ------
        ValueError
            If renormalization fails.
        """
        
        try:
            idxs = mapping_params(self.params_dict, [["amplitude"]]) #, ["scale"]
            idxs_log = mapping_params(self.params_dict, [["logamp"]])
            self.params = (params.at[:, idxs].multiply(scale[:, None]).at[:, idxs_log].add(jnp.log10(scale[:, None])))
            self.uncertainty_params = uncertainty_params.at[:, idxs].multiply(scale[:, None])          
            self.spec = norm_spec.at[:, [1, 2], :].multiply(jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None])
            y_model  = vmap(self.model, in_axes=(0,0))(self.spec[:,0,:],self.params)
            y_data  = self.spec[:,1,:]
            mask = self.mask
            y_error = self.spec[:,2,:]#.at[mask].set(1e41) #already in 1e41 error
            self.residuals = (y_model-y_data)/y_error
            self.free_params = np.sum(~mask,axis=1) - self.params.shape[1]- len(self.dependencies)
            
            self.chi2_red = np.sum(self.residuals**2,axis=1)/self.free_params
            
        except Exception as e:
            logger.exception("Renormalization failed")
            raise ValueError(f"Renormalization error: {e}")
    
    def _build_fit_components(self, profile="gaussian", **kwargs):
        """
        Build parameter initializations, constraints, and profile functions.

        Parameters
        ----------
        profile : str, optional
            Default line profile name.
        **kwargs
            Additional options (not currently used).

        Side Effects
        ------------
        - Populates `initial_params`, `constraints`,
          `profile_functions`, `params_dict`, etc.
        """
        init_list: List[float] = []
        low_list: List[float] = []
        high_list: List[float] = []
        self.profile_functions.clear()
        self.params_dict.clear()
        self.profile_names.clear()
        self.profile_params_index_list.clear()
        self.list = []
        add_linear = True
        idx = 0  # parameter_position
        complex_region = []
        for _,sp in enumerate(self.complex_class.lines):
            holder_profile = getattr(sp, "profile", None) or profile
            sp.profile = holder_profile
            if "SPAF" in holder_profile:
                if len(sp.profile.split("_")) == 2:
                    sp.profile,sp.subprofile = sp.profile.split("_")
                elif not sp.subprofile:
                    sp.subprofile = profile
                #print(sp.center,sp.amplitude_relations,sp.subprofile)
                profile_fn = PROFILE_FUNC_MAP["SPAF"](sp.center,sp.amplitude_relations,sp.subprofile)
            elif sp.profile == "hostmiles":
                host_dict = PROFILE_FUNC_MAP[sp.profile](**sp.template_info)
                profile_fn = host_dict["model"]
                self.host_info = host_dict["host_info"] #host info different from fe_template info 
                #print(local_model.param_names)
            elif sp.profile == "fetemplate":
                fe_dict = PROFILE_FUNC_MAP[sp.profile](**sp.template_info)
                profile_fn = fe_dict["model"]
            else:
                profile_fn =  PROFILE_FUNC_MAP.get(holder_profile, PROFILE_FUNC_MAP["gaussian"])#?
            constraints = ProfileConstraintMaker(sp, self.limits_map.get(sp.region), subprofile= sp.subprofile,local_profile=profile_fn) #this should give the sp.updated?
            sp.profile = constraints.profile
            complex_region.append(sp)
            init_list.extend(constraints.init)
            high_list.extend(constraints.upper)
            low_list.extend(constraints.lower)
            self.profile_functions.append(constraints.profile_fn)
            self.profile_names.append(constraints.profile)
            if sp.profile in list(PROFILE_CONTINUUM_FUNC_MAP.keys()):
                add_linear = False
                self.continuum_params_names = []
                for i, name in enumerate(constraints.param_names):
                    key = f"{name}_{sp.line_name}_{sp.component}_{sp.region}"
                    self.params_dict[key] = idx + i
                    self.continuum_params_names.append(key)     
            else:
                for i, name in enumerate(constraints.param_names):
                    key = f"{name}_{sp.line_name}_{sp.component}_{sp.region}"
                    self.params_dict[key] = idx + i
            # self.profile_params_index.append([idx,idx + len(constraints.param_names)])
            self.profile_params_index_list.append(
                np.arange(idx, idx + len(constraints.param_names))
            )
            idx += len(constraints.param_names)
            #profile="gaussian"

        if add_linear:
            print("Continuum profile not found a linear profile will be added")
            init_,upper_,lower_,spl=self._add_linear(idx)
            init_list.extend(init_)
            high_list.extend(upper_)
            low_list.extend(lower_)
            
            complex_region.append(spl)
            
        self.initial_params = jnp.array(init_list).astype(jnp.float32)
        self.constraints = self._stack_constraints(low_list, high_list)  # constrains or limits
        self.get_param_coord_value = make_get_param_coord_value(self.params_dict, self.initial_params)  # important
        self.complex_region = complex_region #complex_region_list?
    

    def _build_tied(self, tied_params):
        """
        Convert tied‑parameter specifications into dependency strings.

        Parameters
        ----------
        tied_params : list of list
            Each inner list is `[param_target, param_source, ..., optional_value]`.

        Returns
        -------
        list[str]
            Dependency expressions for the minimizer.
        """
        return build_tied(tied_params,self.get_param_coord_value)
        
        # list_tied_params = []
        # if len(tied_params) > 0:
        #     for tied in tied_params:
        #         param1, param2 = tied[:2]
        #         pos_param1, val_param1, param_1 = self.get_param_coord_value(
        #             *param1.split("_"))
                
        #         pos_param2, val_param2, param_2 = self.get_param_coord_value(
        #             *param2.split("_"))
        #         if len(tied) == 2:
        #             if param_1 == param_2 == "center" and len(tied):
        #                 delta = val_param1 - val_param2
        #                 tied_val = "+" + str(delta) if delta > 0 else "-" + str(abs(delta))
        #             elif param_1 == param_2:
        #                 tied_val = "*1"
        #             #elif param_1 == param_2 == "logamp":
                       
        #             else:
        #                 print(f"Define constraints properly. {tied_params}")
        #         else:
        #             tied_val = tied[-1]
        #         if param_1 == param_2 == "logamp":
        #             tied_val = f"{np.log10(extract_float(tied_val))}"
        #             #print(tied_val)
        #         if isinstance(tied_val, str):
        #             list_tied_params.append(f"{pos_param1} {pos_param2} {tied_val}")
        #         else:
        #             print("Define constraints properly.")
        # else:
        #     list_tied_params = []
        # print("Remember move this functions to Assistants and also change it in Montecarlo.")
        # return list_tied_params
    
    @staticmethod
    def _stack_constraints(low: List[float], high: List[float]) -> jnp.ndarray:
        """
        Stack lower and upper bound lists into a JAX array.

        Parameters
        ----------
        low : list of float
        high : list of float

        Returns
        -------
        jnp.ndarray, shape (n_params, 2)
        """
        return jnp.stack([jnp.array(low), jnp.array(high)], axis=1).astype(jnp.float32)
    
    def _add_linear(self,idx):
        """
        Append a linear continuum component when none is present.

        Parameters
        ----------
        idx : int
            Starting index for new continuum parameters.

        Returns
        -------
        init : list[float]
            Initial slope & intercept.
        upper : list[float]
            Upper bounds.
        lower : list[float]
            Lower bounds.
        spl : SpectralLine
            Continuum SpectralLine placeholder.
        """
        self.profile_names.append("linear")
        self.profile_functions.append(PROFILE_FUNC_MAP["linear"])
        for i, name in enumerate(["amplitude_slope", "amplitude_intercept"]):
            key = f"{name}_{'continuum'}_{0}_{'linear'}"
            self.params_dict[key] = idx + i
        self.profile_params_index_list.append(np.arange(idx, idx + 2))
        return [0.1e-4, 0.5],[10.0, 10.0],[-10.0, -10.0],SpectralLine(line_name='linear',region='continuum',component=0,profile='linear')
    
    def to_result(self) -> ComplexResult:
        """
        Assemble and store the ComplexResult object.

        Returns
        -------
        None
        """
        self.complexresult= ComplexResult(
            params=self.params,
            uncertainty_params=self.uncertainty_params,
            constraints=self.constraints,
            mask=self.mask,
            profile_functions=self.profile_functions,
            profile_names=self.profile_names,
            scale=self.scale,
            params_dict=self.params_dict,
            complex_region=self.complex_region,
            loss = self.loss,
            initial_params = self.initial_params,
            profile_params_index_list = self.profile_params_index_list,
            outer_limits = self.outer_limits,
            inner_limits = self.inner_limits,
            fitting_routine = self.fitting_routine,
            dependencies = self.dependencies,
            model_keywords= self.fitting_routine.get("model_keywords"),
            residuals = self.residuals,
            free_params = self.free_params,
            chi2_red = self.chi2_red,
            fitkwargs = self._fitkwargs)
        
    @classmethod
    def from_builder(
        cls,
        builder: "ComplexBuilder",
        *,
        profile: str = "gaussian",limits_overrides = None,
        **builder_kwargs,
    ) -> "ComplexFitting":
        
        """
        Construct ComplexFitting directly from a ComplexBuilder.

        Parameters
        ----------
        builder : ComplexBuilder
        profile : str, optional
            Default profile name for unconstrained lines.
        limits_overrides : dict, optional
            Per‑species parameter limits overrides.
        **builder_kwargs
            Passed to builder._make_fitting_routine(...)

        Returns
        -------
        ComplexFitting
        """
    
        region_dict = builder._make_fitting_routine(**builder_kwargs)

        
        return cls(region_dict, profile=profile,limits_overrides= limits_overrides)
    
    def init_linear(self,norm_spec,params):
        """
        Fit and insert a linear continuum via weighted least squares.

        Parameters
        ----------
        norm_spec : jnp.ndarray
            Array of normalized spectra, shape (n, 3, m).
        params : jnp.ndarray
            Uninitialized parameter array to be updated.

        Returns
        -------
        jnp.ndarray
            Updated parameters with continuum slopes/intercepts.
        """
        
        def wls_one(xi, fi, ei):
            """
            Weighted least‐squares fit of y = m x + b to points (xi, fi),
            using weights w_i = 1/ei^2 over all pixels.
            Returns (slope, intercept).
            """
            # inverse‐variance weights for every pixel
            w      = 1.0 / (ei**2)

            # compute weighted sums
            sum_wxx = jnp.sum(w * xi * xi)    # Σ w x²
            sum_wx  = jnp.sum(w * xi)         # Σ w x
            sum_wxy = jnp.sum(w * xi * fi)    # Σ w x y
            sum_wy  = jnp.sum(w * fi)         # Σ w y
            sum_w   = jnp.sum(w)              # Σ w

            # normal equations: [[Σw x², Σw x], [Σw x, Σw]] · [m, b] = [Σw x y, Σw y]
            M   = jnp.array([[sum_wxx, sum_wx],
                            [sum_wx , sum_w ]])
            rhs = jnp.array([sum_wxy, sum_wy])

            # solve for [m, b]
            slope, intercept = jnp.linalg.solve(M, rhs)
            return slope, intercept

        # prepare inputs:
        x_batch   = (norm_spec[:, 0, :] / 5500.0)
        f_batch   =  norm_spec[:, 1, :]
        e_batch   =  norm_spec[:, 2, :]
        ols_vmapped = vmap(wls_one, in_axes=(0, 0, 0))
        _arr = ols_vmapped(x_batch, f_batch,e_batch)
        for dx,param_name in enumerate(self.continuum_params_names):
            idx_l     = self.params_dict[param_name]
            params = (params.at[:, idx_l].set(_arr[dx])) #.at[:, idx_intercept].set(intercept_arr))
        return params
