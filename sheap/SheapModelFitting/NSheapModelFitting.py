"""
Complex Fitting
===============

This module defines :class:`SheapModelFitting`, the main driver for fitting
multi-component spectral regions with JAX-based minimization.

Main Features
-------------
- Builds parameter initialization, constraints, and profile functions.
- Performs iterative optimization using custom JAX minimizers.
- Supports tied parameters, penalties, and continuum fitting.
- Computes uncertainties via covariance matrices or samplers.
- Packages results into :class:`SheapResult`.

Notes
-----
- All fitting is GPU-accelerated via JAX.
- Residual loss is log-cosh with optional smoothness/curvature penalties.
- Continuum slopes/intercepts can be initialized via weighted least squares.
"""
from __future__ import annotations
__author__ = 'felavila'


__all__ = [
    "SheapModelFitting",
    "logger",
]

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import numpy as np 
# import jax.numpy as np
# import numpy as np
# from jax import jit,vmap
# from jax.scipy.stats import chi2

from sheap.Core import FittingLimits, SpectralLine,SheapResult

from sheap.Assistants.Parameters import build_Parameters
from sheap.Assistants.parser_mapper import build_param_index_cache,mapping_params,parse_dependencies,make_get_param_coord_value,build_tied,flatten_tied_map,parse_dependencies


from sheap.Minimizer.Minimizer import Minimizer

from sheap.Profiles.Profiles import PROFILE_FUNC_MAP,PROFILE_CONTINUUM_FUNC_MAP
from sheap.Profiles.ProfileConstraintMaker import ProfileConstraintMaker 
from sheap.Profiles.Utils import make_fused_profiles,build_grid_penalty


from sheap.Sheapectral.Utils.SpectralSetup import mask_builder, prepare_spectra #
from sheap.Utils.Constants import DEFAULT_LIMITS

from sheap.Utils.UncertaintyFunction import Errorfromloop

from sheap.Minimizer.NMinimizer import fit_sheap_varpro_gradient

# Configure module-level logger
logger = logging.getLogger(__name__)

#SheapModelFitting
class NSheapModelFitting:
    """
    Fits a spectral region containing multiple emission lines.

    This class wraps the workflow of:
      - Building parameterized line + continuum models
      - JAX‑based fitting (with optional penalty functions)
      - Uncertainty estimation via covariance or sampling
      - Post‑processing (renormalization, χ² calculation)
      - Packaging results into a SheapResult object

    Parameters
    ----------
    region_dict : dict
        Dictionary of attributes produced by a ComplexBuilder, including:
        - `sheapmodel` (with `.lines` list)
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
    initial_params : np.ndarray, shape (n_params,)
        Initial guesses for all fit parameters.
    constraints : np.ndarray, shape (n_params, 2)
        Lower and upper bounds for each parameter.
    profile_functions : list[callable]
        JAX‑compiled model functions for each line/continuum component.
    model : callable
        Fused, jit‑compiled model combining all profile_functions.
    host_info : dict
        Extra metadata (e.g. stellar‑population grid) for penalty construction.
    sheapesult : SheapResult
        Final results (parameters, uncertainties, residuals, χ², etc.), set after fitting.

    Methods
    -------
    __call__(spectra, force_cut=False, run_uncertainty_params=True, outer_limits=None,
              learning_rate=None, add_penalty_function=False)
        Execute the full fit on one or more spectra. Raises if limits or
        routine are mis‑specified.

    _fit(iteration_number, norm_spec, model, initial_params,
         tied, learning_rate=1e-1, weighted=True,
         num_steps=1000, non_optimize_in_axis=3,
         penalty_function=None)
        Perform the JAX‑based minimization using Minimizer. Returns
        optimized parameters and final loss history.

    _prep_data(spectra, outer_limits, force_cut)
        Preprocess spectra: mask, cut region, normalize flux by max per pixel.

    _postprocess(norm_spec, params, uncertainty_params, scale)
        Scale fitted parameters back to original flux units and compute
        residuals, χ², and package intermediate arrays.

    _build_fit_components(profile="gaussian", **kwargs)
        Build parameter initialization lists, constraints, and profile functions
        from `sheapmodel.lines`.

    _build_tied(tied_params)
        Convert user‑specified tie lists into dependency strings for the minimizer.

    _stack_constraints(low, high)
        Stack lower & upper bound lists into an (n_params, 2) JAX array.

    _add_linear(idx)
        Add a linear continuum component if none was found in the region.

    to_result()
        Assemble a `SheapResult` object from the final attributes.

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
    >>> rf = SheapModelFitting.from_builder(builder, profile='SPAF')
    >>> rf(spectra_array, outer_limits=(6500, 6600))
    >>> df = rf.pandas_params()
    >>> print(df.head())
    """

    def __init__(self, region_dict: dict, *, profile: str = "gaussian",limits_overrides: Optional[Dict[str, FittingLimits]] = {}):
        
        """
        Initialize SheapModelFitting with builder output and optional limits.

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
        for region, cfg in DEFAULT_LIMITS.items():
            if region in limits_overrides:
                if isinstance(limits_overrides,dict):
                    for k,val in limits_overrides[region].items():
                        if k in cfg.keys():
                            cfg[k] = val
                        else:
                            print(f"{k} is not an avalaible variable try {list(cfg.keys())}")
            default_lim = FittingLimits.from_dict(cfg)
            self.limits_map[region] = default_lim
        non_regions = set(list(limits_overrides.keys())) - set(list(DEFAULT_LIMITS.keys()))
        if len(non_regions) > 0:
            print(f"region(s) {non_regions} are not avalaible try with {list(DEFAULT_LIMITS.keys())}" )
       
              
        self.params_dict: Dict[str, int] = {}
        self.initial_params: np.ndarray = np.array([])
        self.profile_functions: List[Any] = []
        self.profile_names: List[str] = []
        self.profile_params_index_list: List[List[int]] = []
        self.constraints: Optional[np.ndarray] = None
        self.params: Optional[np.ndarray] = None
        self.loss: Optional[float] = None
        self._build_fit_components(profile = profile)
        self.model = make_fused_profiles(self.profile_functions) #TODO we should change all the "model" to spectral_model or similars to make it more explicit
        #self.model_vmap = vmap(self.model, in_axes=(0,0))
        self.host_info = {}
    
    def __call__(
        self,
        spectra: Union[List[Any], np.ndarray],
        force_cut: bool = False,
        outer_limits: Optional[Tuple[float, float]] = None,
        mask_list= [],
        ) -> None:
        """
        Execute the full fitting routine on provided spectra.

        Parameters
        ----------
        spectra : list or np.ndarray
            Input array of shape (n_spectra, 3, n_pixels).
        force_cut : bool, optional
            Force region cutting after mask is built.
        run_uncertainty_params : bool, optional
            Compute uncertainties via covariance matrix.
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
        #print(spectra.shape)
        _, mask, scale, norm_spec = self._prep_data(spectra, outer_limits, force_cut,mask_list=mask_list)
        initial_params = self.initial_params
        profile_functions = list(self.model.sub_funcs)
        profile_params_index_list = self.profile_params_index_list
        self.params_class = self._build_params_class(initial_params,shared_params=None)
        x, y, yerr = norm_spec.transpose(1, 0, 2)

        result_logcosh = fit_sheap_varpro_gradient(self.params_class,profile_functions,profile_params_index_list,initial_params,x=x[0], y=y[0], yerr = yerr[0],fit_mask=~mask[0],)
        self.mask = mask
        self._params = result_logcosh["p_best"][None,:]
        uncertainty_params = np.zeros_like(self._params)
        self.param_index_cache = build_param_index_cache(self.params_dict)
        self.idxs_amplitude = self.param_index_cache["amplitude"]
        self.idxs_logamp = self.param_index_cache["logamp"]
        self.scale = scale

        #nsheapspectral.spectral_model.numpy_model

        self._postprocess(norm_spec, self._params,uncertainty_params,scale)
        self._to_result()
    
    def _fit(self, norm_spec: np.ndarray, model, initial_params, tied: List[List[str]], learning_rate=1e-1, weighted: bool = True, num_steps: int = 1000, non_optimize_in_axis=3, penalty_function = None,
            method = None, penalty_weight: float = 0.01, curvature_weight: float = 1e5, smoothness_weight: float = 0.0, max_weight: float = 0.1,shared_params=[],verbose = True,
             batch_mode: str = "global_independent",convergence_options: Optional[Dict] = None,global_reduction: str = "sum",
            ) -> Tuple[np.ndarray, list]:
        """
        Perform the JAX‑based minimization using Minimizer.

        Parameters
        ----------
        norm_spec : np.ndarray
            Normalized spectra array.
        model : callable
        initial_params : np.ndarray
        learning_rate : float, optional
        weighted : bool, optional
        num_steps : int, optional
        non_optimize_in_axis : int, optional
        penalty_function : callable, optional

        Returns
        -------
        params : np.ndarray
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
        self.params_class = self._build_params_class(initial_params,shared_params=shared_params)
        minimizer = Minimizer(model,non_optimize_in_axis=non_optimize_in_axis,num_steps=num_steps,weighted=weighted,learning_rate=learning_rate,param_converter=self.params_class,
            penalty_function = penalty_function,method=method, penalty_weight= penalty_weight,curvature_weight= curvature_weight,smoothness_weight= smoothness_weight,max_weight= max_weight,
            batch_mode = batch_mode,convergence_options=convergence_options,global_reduction=global_reduction
            )

        try:
            #faster why?
            params, loss = minimizer(initial_params, *norm_spec.transpose(1, 0, 2), self.constraints)
            self.minimizer = minimizer
            self.norm_spec = norm_spec
            
        except Exception as e:
            logger.exception("Fitting failed")
            raise RuntimeError(f"Fitting error: {e}")
        return params, loss


    def _prep_data(self, spectra: Union[List[Any], np.ndarray], outer_limits: Optional[Tuple[float, float]], force_cut: bool,
        mask_list=[]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess spectra for fitting.

        Parameters
        ----------
        spectra : list or np.ndarray
        outer_limits : tuple(float, float)
        force_cut : bool

        Returns
        -------
        spec : np.ndarray
        mask : np.ndarray
        scale : np.ndarray
        norm_spec : np.ndarray

        Raises
        ------
        ValueError
            On preprocessing or normalization errors.
        """
        self.outer_limits = outer_limits or self.outer_limits
        
        if not  self.outer_limits:
            raise ValueError("outer_limits must be specified")
        
        try:
            if isinstance(spectra, list):
                spec, mask = prepare_spectra(spectra, outer_limits=self.outer_limits)
            else:
                spec, _, _, mask = mask_builder(spectra, outer_limits=self.outer_limits,mask_list=mask_list)
                if force_cut:
                    spec, mask = prepare_spectra(spec, outer_limits=self.outer_limits)
        except Exception as e:
            logger.exception("Failed to preprocess spectra")
            raise ValueError(f"Preprocessing error: {e}")

        try:
            scale = np.nanmax(np.where(mask, 0, spec[:, 1, :]), axis=1) 
            norm_spec = np.asarray(spec, dtype=np.float64)
            mask = np.asarray(mask)
            norm_spec[:, [1, 2], :] = spec[:, [1, 2], :]/(np.moveaxis(np.tile(scale, (2, 1)), 0, 1)[:, :, None])
            
        except Exception as e:
            logger.exception("Normalization error")
            raise ValueError(f"Normalization error: {e}")

        return spec, mask, scale, norm_spec
    
    def _postprocess(self, norm_spec: np.ndarray, params: np.ndarray,uncertainty_params: np.ndarray,scale: np.ndarray,) -> None:
        """
        Scale parameters back to original flux units and compute diagnostics.

        Parameters
        ----------
        norm_spec : np.ndarray
        params : np.ndarray
        uncertainty_params : np.ndarray
        scale : np.ndarray

        Raises
        ------
        ValueError
            If renormalization fails.
        """
        try:
            idxs_log = self.idxs_logamp
            idxs = self.idxs_amplitude 
            self.params_desc = params
            if len(idxs_log) == 0:
                self.params[:, idxs] = params[:, idxs] * (scale[:, None])
                self.uncertainty_params[:, idxs] = uncertainty_params[:, idxs] * (scale[:, None])
            elif len(idxs) == 0:
                self.params[:, idxs_log] = params[:, idxs_log] + (np.log10(scale[:, None]))
                self.uncertainty_params[:, idxs_log] = uncertainty_params[:, idxs_log] + (np.log10(scale[:, None])) 
            else:
                self.params = params.copy()
                self.params[:, idxs] *= scale[:, None]
                self.params[:, idxs_log] += np.log10(scale[:, None])

                self.uncertainty_params = uncertainty_params.copy()
                self.uncertainty_params[:, idxs] *= scale[:, None]
                self.uncertainty_params[:, idxs_log] += np.log10(scale[:, None]) 
            self.spec = norm_spec.copy()
            self.spec[:, [1, 2], :] = norm_spec[:, [1, 2], :] * (np.moveaxis(np.tile(scale, (2, 1)), 0, 1)[:, :, None])
            #print(self.params)
            y_model  = self.model.numpy_model(self.spec[0,0,:],self.params[0])
            self.spec[:, 2, :] = np.where(self.mask, 1e41, self.spec[:, 2, :])
            y_model = y_model[None,:]
            self.residuals = (y_model-self.spec[:,1,:])/self.spec[:,2,:]
            self.free_params = np.sum(~self.mask,axis=1) - self.params.shape[1] + len(self.dependencies)
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
        #self.list = []
        add_linear = True
        idx = 0  # parameter_position
        region_list = []
        for _,sp in enumerate(self.sheapmodel.lines):
            region_name = sp.region
            holder_profile = getattr(sp, "profile", None) or profile
            sp.profile = holder_profile
            if "SPAF" in holder_profile:
                if len(sp.profile.split("_")) == 2:
                    sp.profile,sp.subprofile = sp.profile.split("_")
                elif not sp.subprofile:
                    sp.subprofile = profile
                profile_fn = PROFILE_FUNC_MAP["SPAF"](sp.center,sp.amplitude_relations,sp.subprofile)
            elif sp.profile == "hostmiles":
                host_dict = PROFILE_FUNC_MAP[sp.profile](**sp.template_info)
                profile_fn = host_dict["model"]
                self.host_info = host_dict["host_info"] 
            #here maybe could be good option
            elif sp.profile == "template":
                if sp.line_name == "balmerhighorder":
                    region_name = sp.line_name
                template_dict = PROFILE_FUNC_MAP[sp.profile](**sp.template_info)
                profile_fn = template_dict["model"]
            elif sp.region =="continuum":
                #if sp.profile == "polynomial":
                profile_fn =  PROFILE_FUNC_MAP.get(sp.profile)(**sp.template_info["keywords"]) 
            else:
                profile_fn =  PROFILE_FUNC_MAP.get(holder_profile, PROFILE_FUNC_MAP["gaussian"])#?
            
            constraints = ProfileConstraintMaker(sp, self.limits_map.get(region_name), subprofile= sp.subprofile,local_profile=profile_fn) #this should give the sp.updated?
            sp.profile = constraints.profile
            region_list.append(sp)
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
            self.profile_params_index_list.append(np.arange(idx, idx + len(constraints.param_names)))
            idx += len(constraints.param_names)

            
        self.initial_params = np.array(init_list).astype(np.float32)
        self.constraints = self._stack_constraints(low_list, high_list)  # constrains or limits
        self.get_param_coord_value = make_get_param_coord_value(self.params_dict, self.initial_params)  # important
        #cold_init_
        self.dependencies = parse_dependencies([])
        self.params_class = self._build_params_class(self.initial_params)
        self.region_list = region_list
    

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
        
    def _build_params_class(self,initial_params,shared_params=[]):
        tied_map = {T[1]: T[2:] for  T in self.dependencies}
        self.tied_map  = flatten_tied_map(tied_map)
        return build_Parameters(tied_map,self.params_dict,initial_params,self.constraints,shared_params=shared_params)



    @staticmethod
    def _stack_constraints(low: List[float], high: List[float]) -> np.ndarray:
        """
        Stack lower and upper bound lists into a JAX array.

        Parameters
        ----------
        low : list of float
        high : list of float

        Returns
        -------
        np.ndarray, shape (n_params, 2)
        """
        return np.stack([np.array(low), np.array(high)], axis=1).astype(np.float32)
    
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
    
    def _to_result(self) -> SheapResult:
        """
        Assemble and store the SheapResult object.

        Returns
        -------
        None
        """
        self.sheapresult= SheapResult(
            params=self.params,
            uncertainty_params=self.uncertainty_params,
            constraints=self.constraints,
            mask=self.mask,
            profile_functions=self.profile_functions,
            profile_names=self.profile_names,
            scale=self.scale,
            params_dict=self.params_dict,
            region_list=self.region_list,
            initial_params = self.initial_params,
            profile_params_index_list = self.profile_params_index_list,
            #loss = self.loss,
            outer_limits = self.outer_limits,
            #fitting_routine = self.fitting_routine,
            #dependencies = self.dependencies,
            #model_keywords= self.fitting_routine.get("model_keywords"),
            residuals = self.residuals,
            free_params = self.free_params,
            chi2_red = self.chi2_red,
            #fitkwargs = self._fitkwargs,
            #elapsed_time = self.total_time
            )
        
    @classmethod
    def from_builder(cls,builder: "ComplexBuilder",*,profile: str = "gaussian",limits_overrides = None,**builder_kwargs,) -> "NSheapModelFitting":
        
        """
        Construct SheapModelFitting directly from a ComplexBuilder.

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
        SheapModelFitting
        """
    
        region_dict = builder._make_fitting_routine(**builder_kwargs)
        return cls(region_dict, profile=profile,limits_overrides= limits_overrides)
    
    def init_linear(self,norm_spec,params):
        """
        Fit and insert a linear continuum via weighted least squares.

        Parameters
        ----------
        norm_spec : np.ndarray
            Array of normalized spectra, shape (n, 3, m).
        params : np.ndarray
            Uninitialized parameter array to be updated.

        Returns
        -------
        np.ndarray
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
            sum_wxx = np.sum(w * xi * xi)   
            sum_wx  = np.sum(w * xi)         
            sum_wxy = np.sum(w * xi * fi)    
            sum_wy  = np.sum(w * fi)         
            sum_w   = np.sum(w)              

           
            M   = np.array([[sum_wxx, sum_wx],
                            [sum_wx , sum_w ]])
            rhs = np.array([sum_wxy, sum_wy])

            # solve for [m, b]
            slope, intercept = np.linalg.solve(M, rhs)
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
