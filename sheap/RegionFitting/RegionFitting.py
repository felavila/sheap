from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from jax import jit

from sheap.DataClass.DataClass import FittingLimits, SpectralLine,FitResult
from sheap.Functions.profiles import PROFILE_FUNC_MAP
from sheap.Minimizer.utils import parse_dependencies
from sheap.Minimizer.MasterMinimizer import MasterMinimizer

# from sheap.Fitting.template_fe_func import
from sheap.Functions.utils import combine_auto,make_fused_profiles,make_super_fused
from sheap.RegionFitting.utils import make_constraints, make_get_param_coord_value,DEFAULT_LIMITS
from sheap.Mappers.helpers import mapping_params
from sheap.Tools.setup_utils import mask_builder, prepare_spectra
from sheap.DataClass.utils import is_list_of_SpectralLine

from sheap.RegionFitting.uncertainty_functions import error_for_loop

# Configure module-level logger
logger = logging.getLogger(__name__)

# Constant identifiers for special components
OUTFLOW_COMPONENT = 10  # ID used for outflow line components
FE_COMPONENT = 20  # ID used for Fe emission components
CONT_COMPONENT = 0  # ID used for continuum component


class RegionFitting:
    """
    Fits a spectral region containing multiple emission lines.

    This class:
      - Loads line definitions from YAML or provided dict/list.
      - Normalizes and masks spectra.
      - Builds parameter arrays with bounds.
      - Runs JAX-based minimization (MasterMinimizer).
      - Supports renormalization and parameter mapping.
    """

    def __init__(
        self,
        region_template: Union[str, dict, List[dict]],
        yaml_dir: Optional[Union[str, Path]] = None,
        tied_params: Optional[List[List[Any]]] = None,
        log_mode: bool = False,
        limits_overrides: Optional[Dict[str, FittingLimits]] = None,
        profile = "gaussian"
    ) -> None:
        """
        Initialize RegionFitting.

        Args:
            region_template: Path or name of YAML file, or dict/list defining lines.
            yaml_dir: Directory to search for YAML templates.
            tied_params: Initial parameter tie definitions.
            log_mode: If True, log additional debug info.
            limits_overrides: User-specified FittingLimits per kind.
        """
        if profile not in ['gaussian','lorentzian']:
            print(profile," not added to the code yet")
            profile = "gaussian"
        
        self.complex_region_ = self._load_region(region_template, yaml_dir)  # fitting rutine
        self.complex_region = self.complex_region_.get("complex_region") or []
        self.fitting_routine = self.complex_region_.get("fitting_routine") or {}
        limits = self.complex_region_.get("inner_limits")
        self.inner_limits = (
            tuple(limits) if isinstance(limits, list) and len(limits) == 2 else None
        )
        limits = self.complex_region_.get("outer_limits")
        self.outer_limits = (
            tuple(limits) if isinstance(limits, list) and len(limits) == 2 else None
        )
        self.log_mode = log_mode
        self.limits_map: Dict[str, FittingLimits] = {}
        for kind, cfg in DEFAULT_LIMITS.items():
            default_lim = FittingLimits.from_dict(cfg)
            # Use override if provided, else default
            self.limits_map[kind] = (
                limits_overrides[kind]
                if limits_overrides and kind in limits_overrides
                else default_lim
            )

        # Attributes to be populated during fitting

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
        self._build_fit_components(profile="gaussian")
        self.model = jit(make_fused_profiles(self.profile_functions))
    
    def __call__(
        self,
        spectra: Union[List[Any], jnp.ndarray],
        inner_limits: Optional[Tuple[float, float]] = None,
        outer_limits: Optional[Tuple[float, float]] = None,
        force_cut: bool = False,
        #exp_factor: jnp.ndarray = jnp.array([0.0]),# this is part of the posterior that cames from the class Mainsheap
        #N: int = 2_000,
        do_return=False,  # meanwhile variable
        sigma_params=True,
        learning_rate=None) -> None:
        # the idea is that is exp_factor dosent have the same shape of scale could be fully renormalice the spectra.
        print(f"Fitting {spectra.shape[0]} spectra with {spectra.shape[2]} wavelength pixels")
        #make_fused_profiles(funcs)
        #self.model = jit(
         #   make_fused_profiles(self.profile_functions))  # maybe this could be taked before
        
        
        _, mask, scale, norm_spec = self._prep_data(
            spectra, inner_limits, outer_limits, force_cut)

        inner_limits = self.inner_limits or inner_limits
        outer_limits = self.outer_limits or outer_limits

        if not (self.inner_limits and self.outer_limits):
            raise ValueError("inner_limits and outer_limits must be specified")
        if not isinstance(self.fitting_routine, dict):
            raise TypeError("fitting_routine must be a dictionary.")
        params = self.initial_params
        total_time = 0
        for i, (key, step) in enumerate(self.fitting_routine.items()):
            print(f"\n{'='*40}\n{key.upper()} (step {i+1}) free params {self.initial_params.shape[0]-len(step['tied'])}")
            step["non_optimize_in_axis"] = 4 #experimental
            if len(params.shape)==1:
                params = jnp.tile(params, (spectra.shape[0], 1))
            if isinstance(learning_rate,list):
                step["learning_rate"] = learning_rate[i]
                #print(params.dtype,params.shape)
                #break 
            start_time = time.time()  # 
            params, loss = self._fit(norm_spec, self.model, params, **step)
            uncertainty_params = jnp.zeros_like(params)
            end_time = time.time()  # 
            elapsed = end_time - start_time
            print(f"Time for step '{key}': {elapsed:.2f} seconds")
            total_time += elapsed
        dependencies = parse_dependencies(self._build_tied(step["tied"]))
        if sigma_params:
            print("\n==Running error_covariance_matrix==")
            start_time = time.time()  # 
            uncertainty_params = error_for_loop(self.model,norm_spec,params,dependencies)
            end_time = time.time()  # 
            elapsed = end_time - start_time
            print(f"Time for error_covariance_matrix: {elapsed:.2f} seconds")
            total_time += elapsed
        print(f'The entire process took {total_time:.2f} ({total_time/spectra.shape[0]:.2f}s by spectra)')
        self.dependencies = dependencies
        self._postprocess(norm_spec, params, uncertainty_params, scale)
        self.mask = mask
        self.loss = loss
        self.scale = scale
        self.outer_limits = outer_limits
        self.inner_limits = inner_limits
        if do_return:
            return self.to_result()
        #else:
           
        # spec.at[:,[1,2],:].multiply(jnp.moveaxis(jnp.tile(scale,(2,1)),0,1)[:,:,None]) #This could be forget after   
    
    def _fit(
        self,
        norm_spec: jnp.ndarray,
        model,
        initial_params,
        tied: List[List[str]],
        learning_rate=1e-1,
        weighted: bool = True,
        num_steps: int = 1000,
        non_optimize_in_axis=3,
        # optimizer?
    ) -> Tuple[jnp.ndarray, list]:
        """
        Perform the JAX-based minimization using MasterMinimizer.
        Returns optimized parameters and final loss.
        """
        print(
            "learning_rate:",
            learning_rate,
            "num_steps:",
            num_steps,
            "non_optimize_in_axis:",
            non_optimize_in_axis,
        )
        list_dependencies = self._build_tied(tied)
        #print(list_dependencies)
        minimizer = MasterMinimizer(
            model,
            non_optimize_in_axis=non_optimize_in_axis,
            num_steps=num_steps,
            list_dependencies=list_dependencies,
            weighted=weighted,
            learning_rate=learning_rate,
        )
        try:
            params, loss = minimizer(
                initial_params, *norm_spec.transpose(1, 0, 2), self.constraints
            )
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
        Preprocess spectra:
          - Apply masks
          - Optionally cut region
          - Normalize flux by max per pixel

        Returns:
            spec, mask, scale, normalized spec
        """
        # Set or verify limits
        self.inner_limits = inner_limits or self.inner_limits
        self.outer_limits = outer_limits or self.outer_limits
        if not (self.inner_limits and self.outer_limits):
            raise ValueError("inner_limits and outer_limits must be specified")
        # Build spectrum and mask
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

        # Normalize flux dimension
        try:
            scale = jnp.nanmax(jnp.where(mask, 0, spec[:, 1, :]), axis=1) #maybe is best sum to move all the spectra to 1?
            #print(scale) 
            norm_spec = spec.at[:, [1, 2], :].divide(
                jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None]
            )
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
        # exp_factor: Union[float, jnp.ndarray],
        # srenormalize: bool
    ) -> None:
        """
        Scale parameters back to original flux units if requested.
        Store final params and loss.
        """
        # self.loss = loss
        # if renormalize:
        try:
            #scaled = scaleux  # / (10**exp_factor)
            idxs = mapping_params(
                self.params_dict, [["amplitude"], ["scale"]]
            )  # check later on cont how it works
            self.params = params.at[:, idxs].multiply(scale[:, None])
            self.uncertainty_params = uncertainty_params.at[:, idxs].multiply(scale[:, None])
            self.spec = norm_spec.at[:, [1, 2], :].multiply(
                jnp.moveaxis(jnp.tile(scale, (2, 1)), 0, 1)[:, :, None]
            )
        except Exception as e:
            logger.exception("Renormalization failed")
            raise ValueError(f"Renormalization error: {e}")
        # else:
        #     self.params = params
        #     self.spec = norm_spec

    def _load_region(
        self, template: Union[str, dict, List[dict]], yaml_dir: Optional[Union[str, Path]]
    ) -> Dict[str, Any]:
        """
        Load line definitions from YAML, dict, or list of SpectralLine-compatible entries.

        Returns:
            Dict containing complex_region, fitting_routine, inner_limits, outer_limits
        """
        if isinstance(template, str):
            path = Path(template)
            if not path.exists() and yaml_dir:
                path = Path(yaml_dir) / f"{template}.yaml"
            if not path.exists():
                logger.error("Region template not found: %s", template)
                raise FileNotFoundError(f"Region template not found: {template}")
            data = yaml.safe_load(path.read_text())
        elif isinstance(template, dict):
            data = template
        elif isinstance(template, list):
            # Assume this is a list of dicts defining SpectralLine entries
            data = {
                "complex_region": [SpectralLine(**entry) for entry in template],
                "fitting_routine": {},
                "inner_limits": None,
                "outer_limits": None,
            }
            return data
        else:
            raise TypeError("Unsupported type for region_template")

        raw = data.get("complex_region")
        if raw is None or not isinstance(raw, list):
            logger.error("complex_region definition missing or not a list")
            raise ValueError("complex_region definition must contain a 'complex_region' list")

        # Convert raw to SpectralLine objects if needed
        region_lines = (
            raw if is_list_of_SpectralLine(raw) else [SpectralLine(**entry) for entry in raw]
        )

        return {
            "complex_region": region_lines,
            "fitting_routine": data.get("fitting_routine", {}),
            "inner_limits": data.get("inner_limits"),
            "outer_limits": data.get("outer_limits"),
        }

    def _build_fit_components(self, profile="gaussian", **kwargs):
        """
        Build the region constraints and ties for the spectral fitting.
        Parameters:
            as_array (bool): Whether to store the region constraints as a JAX array.
            tied_params (list, optional): Overrides the instance tied_params if provided.
            tie param_target to param_source
            [param_target, param_source,operand,value]
            limits_list (list, optional): Overrides the instance limits_list if provided.
        """
        init_list: List[float] = []
        low_list: List[float] = []
        high_list: List[float] = []
        self.profile_functions.clear()
        self.params_dict.clear()
        self.profile_names.clear()
        # self.profile_params_index.clear()
        self.profile_params_index_list.clear()
        add_linear = True
        self.list = []
        # Loop over each line configuration
        idx = 0  # parameter_position
        complex_region = []
        #I have to decide between sp or cfg for the lines 
        for cfg in self.complex_region:
            holder_profile = getattr(cfg, "profile", None) or profile
            cfg.profile = holder_profile
            if "SPAF" in holder_profile:
                if len(cfg.profile.split("_")) == 2:
                    cfg.profile,cfg.subprofile = cfg.profile.split("_")
                elif not cfg.subprofile:
                    cfg.subprofile = profile
            constraints = make_constraints(cfg, self.limits_map.get(cfg.kind), profile=  cfg.profile, subprofile= cfg.subprofile)
            cfg.profile = constraints.profile  #re writte the complex line 
            #print(cfg.profile,cfg.subprofile)
            complex_region.append(cfg)
            init_list.extend(constraints.init)
            high_list.extend(constraints.upper)
            low_list.extend(constraints.lower)
            if 'SPAF' in cfg.profile:
                sm = PROFILE_FUNC_MAP["SPAF"](cfg.center,cfg.amplitude_relations,cfg.subprofile)
                self.profile_names.append(cfg.profile)
                self.profile_functions.append(sm)
            else:
                self.profile_functions.append(
                    PROFILE_FUNC_MAP.get(constraints.profile, PROFILE_FUNC_MAP["gaussian"]))
                self.profile_names.append(constraints.profile)
            if cfg.profile in ["powerlaw","brokenpowerlaw",'linear']:
                add_linear = False
            #print(constraints.param_names)
            for i, name in enumerate(constraints.param_names):
                key = f"{name}_{cfg.line_name}_{cfg.component}_{cfg.kind}"
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
        self.get_param_coord_value = make_get_param_coord_value(
            self.params_dict, self.initial_params
        )  # important
        self.complex_region = complex_region

    def _build_tied(self, tied_params):
        list_tied_params = []
        if len(tied_params) > 0:
            for tied in tied_params:
                param1, param2 = tied[:2]
                pos_param1, val_param1, param_1 = self.get_param_coord_value(
                    *param1.split("_")
                )
                pos_param2, val_param2, param_2 = self.get_param_coord_value(
                    *param2.split("_")
                )
                if len(tied) == 2:
                    if param_1 == param_2 == "center" and len(tied):
                        #print(param_1,param_2)
                        delta = val_param1 - val_param2
                        tied_val = "+" + str(delta) if delta > 0 else "-" + str(abs(delta))
                        # if log_mode:
                    elif param_1 == param_2:
                        tied_val = "*1"
                    else:
                        print(f"Define constraints properly. {tied_params}")
                else:
                    tied_val = tied[-1]
                if isinstance(tied_val, str):
                    list_tied_params.append(f"{pos_param1} {pos_param2} {tied_val}")
                else:
                    print("Define constraints properly.")
        else:
            list_tied_params = []
        return list_tied_params
    
    
    def to_result(self) -> FitResult:
        return FitResult(
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
            #fitting_routine = fitting_routine.get("fitting_routine"),
            #model_keywords=self.fitting_routine.get("model_keywords", {})
        )
    
    def _add_linear(self,idx):
        self.profile_names.append("linear")
        self.profile_functions.append(PROFILE_FUNC_MAP["linear"])
        for i, name in enumerate(["scale_b", "scale_m"]):
            key = f"{name}_{'continuum'}_{0}_{'linear'}"
            self.params_dict[key] = idx + i
        self.profile_params_index_list.append(np.arange(idx, idx + 2))
        return [0.1e-4, 0.5],[10.0, 10.0],[-10.0, -10.0],SpectralLine(center=None,line_name='linear',kind='continuum',component=0,profile='linear',region='continuum')
    @property
    def pandas_params(self) -> pd.DataFrame:
        """Return fit parameters as a pandas DataFrame."""
        return pd.DataFrame(self.params, columns=list(self.params_dict.keys()))

    @property
    def pandas_region(self) -> pd.DataFrame:
        """Return region definitions as a pandas DataFrame."""
        return pd.DataFrame([vars(cfg) for cfg in self.complex_region])

    @staticmethod
    def _stack_constraints(low: List[float], high: List[float]) -> jnp.ndarray:
        """
        Utility to stack lower and upper bounds into a (N,2) array.
        """
        return jnp.stack([jnp.array(low), jnp.array(high)], axis=1).astype(jnp.float32)
