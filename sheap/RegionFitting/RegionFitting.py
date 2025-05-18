from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from jax import jit

from sheap.DataClass.DataClass import FittingLimits, SpectralLine
from sheap.FunctionsMinimize.functions import (
    Gsum_model,
    balmerconti,
    fitFeOP,
    fitFeUV,
    gaussian_func,
    linear,
    lorentzian_func,
    powerlaw,
)
from sheap.FunctionsMinimize.MasterMinimizer import MasterMinimizer

# from sheap.Fitting.template_fe_func import
from sheap.FunctionsMinimize.utils import combine_auto
from sheap.RegionFitting.utils import make_constraints, make_get_param_coord_value
from sheap.SuportFunctions.functions import mapping_params
from sheap.utils import mask_builder, prepare_spectra

from .uncertainty_functions import batch_error_covariance_in_chunks, error_for_loop

# Configure module-level logger
logger = logging.getLogger(__name__)

# Constant identifiers for special components
OUTFLOW_COMPONENT = 10  # ID used for outflow line components
FE_COMPONENT = 20  # ID used for Fe emission components
CONT_COMPONENT = 0  # ID used for continuum component

# Default kinematic limits (in km/s) per component kind
DEFAULT_LIMITS = {
    'broad': dict(
        upper_width=5000.0, lower_width=425.0, center_shift=5000.0, max_amplitude=10.0
    ),
    'narrow': dict(
        upper_width=200.0, lower_width=50.0, center_shift=2500.0, max_amplitude=10.0
    ),
    'outflow': dict(
        upper_width=5000.0, lower_width=425.0, center_shift=2500.0, max_amplitude=10.0
    ),
    'fe': dict(upper_width=3000.0, lower_width=210.0, center_shift=2500.0, max_amplitude=0.07),
}

PROFILE_FUNC_MAP: Dict[str, Any] = {
    'gaussian': gaussian_func,
    'lorentzian': lorentzian_func,
    'powerlaw': powerlaw,
    'fitFeOP': fitFeOP,
    'fitFeUV': fitFeUV,
    'linear': linear,
    "balmerconti": balmerconti,
}

# @dataclass
# class FittingLimits:
#     """
#     Stores width and shift limits for a line component kind.
#     Attributes:
#         upper_width (float): Maximum velocity width (km/s).
#         lower_width (float): Minimum velocity width (km/s).
#         center_shift (float): Maximum center shift (km/s).
#     """
#     upper_width: float
#     lower_width: float
#     center_shift: float
#     max_amplitude: float

#     @classmethod
#     def from_dict(cls, d: Dict[str, float]) -> FittingLimits:
#         """Create FittingLimits from a dict with keys matching attributes."""
#         return cls(
#             upper_width=d['upper_width'],
#             lower_width=d['lower_width'],
#             center_shift=d['center_shift'],
#             max_amplitude=d['max_amplitude']
#         )


def is_list_of_SpectralLine(data: object) -> bool:
    return isinstance(data, list) and all(isinstance(item, SpectralLine) for item in data)


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
        self.region_defs_ = self._load_region(region_template, yaml_dir)  # fitting rutine
        self.region_defs = self.region_defs_.get("complex_region") or []
        self.fitting_rutine = self.region_defs_.get("fitting_rutine") or {}
        limits = self.region_defs_.get("inner_limits")
        self.inner_limits = (
            tuple(limits) if isinstance(limits, list) and len(limits) == 2 else None
        )
        limits = self.region_defs_.get("outer_limits")
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
        self._build_fit_components()

    def __call__(
        self,
        spectra: Union[List[Any], jnp.ndarray],
        inner_limits: Optional[Tuple[float, float]] = None,
        outer_limits: Optional[Tuple[float, float]] = None,
        force_cut: bool = False,
        exp_factor: jnp.ndarray = jnp.array(
            [0.0]
        ),  # this is part of the posterior that cames from the class Mainsheap
        # renormalize: bool = True, #This should be allways trues
        do_return=False,  # meanwhile variable
        sigma_params=True,
    ) -> None:
        # the idea is that is exp_factor dosent have the same shape of max_flux could be fully renormalice the spectra.
        print(f"Fitting {spectra.shape[0]} spectra")
        self.model = jit(
            combine_auto(self.profile_functions)
        )  # maybe this could be taked before
        _, mask, max_flux, norm_spec, exp_factor = self._prep_data(
            spectra, inner_limits, outer_limits, force_cut, exp_factor
        )

        inner_limits = self.inner_limits or inner_limits
        outer_limits = self.outer_limits or outer_limits

        if not (self.inner_limits and self.outer_limits):
            raise ValueError("inner_limits and outer_limits must be specified")
        if not isinstance(self.fitting_rutine, dict):
            raise TypeError("fitting_rutine must be a dictionary.")
        params = self.initial_params
        for i, (key, step) in enumerate(self.fitting_rutine.items()):
            print(key.upper())
            # step #step is a dictionary so it can take all the parameters directly to fit
            params, loss = self._fit(norm_spec, self.model, params, **step)
            uncertainty_params = jnp.zeros_like(params)
        if sigma_params:
            print("running uncertainty")
            # error_for_loop(model,params,spectra,free_params=0)
            uncertainty_params = error_for_loop(
                self.model,
                params,
                norm_spec,
                free_params=params.shape[1] - len(step.get("tied", [])),
            )

        self._postprocess(norm_spec, params, uncertainty_params, max_flux)
        if do_return:
            return (
                self.params,
                self.uncertainty_params,
                outer_limits,
                inner_limits,
                loss,
                mask,
                step,
                self.params_dict,
                self.initial_params,
                self.profile_params_index_list,
                self.profile_functions,
                max_flux,
                self.profile_names,
                self.region_defs,
            )
        else:
            self.mask = mask
            self.loss = loss
            self.max_flux = max_flux
            self.outer_limits = outer_limits
            self.inner_limits = inner_limits
        # spec.at[:,[1,2],:].multiply(jnp.moveaxis(jnp.tile(max_flux,(2,1)),0,1)[:,:,None]) #This could be forget after

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
        # print(list_dependencies)
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
        exp_factor,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Preprocess spectra:
          - Apply masks
          - Optionally cut region
          - Normalize flux by max per pixel

        Returns:
            spec, mask, max_flux, normalized spec
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
            max_flux = jnp.nanmax(jnp.where(mask, 0, spec[:, 1, :]), axis=1)
            norm_spec = spec.at[:, [1, 2], :].divide(
                jnp.moveaxis(jnp.tile(max_flux, (2, 1)), 0, 1)[:, :, None]
            )
            if exp_factor.shape != max_flux.shape:
                exp_factor = 0
        except Exception as e:
            logger.exception("Normalization error")
            raise ValueError(f"Normalization error: {e}")

        return spec, mask, max_flux, norm_spec, exp_factor

    def _postprocess(
        self,
        norm_spec: jnp.ndarray,
        params: jnp.ndarray,
        uncertainty_params: jnp.ndarray,
        max_flux: jnp.ndarray,
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
            scaled = max_flux  # / (10**exp_factor)
            idxs = mapping_params(
                self.params_dict, [["amplitude"], ["scale"]]
            )  # check later on cont how it works
            self.params = params.at[:, idxs].multiply(scaled[:, None])
            self.uncertainty_params = uncertainty_params.at[:, idxs].multiply(scaled[:, None])
            self.spec = norm_spec.at[:, [1, 2], :].multiply(
                jnp.moveaxis(jnp.tile(scaled, (2, 1)), 0, 1)[:, :, None]
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
            Dict containing complex_region, fitting_rutine, inner_limits, outer_limits
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
                "fitting_rutine": {},
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
            "fitting_rutine": data.get("fitting_rutine", {}),
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
        region_defs = []
        for cfg in self.region_defs:

            constraints = make_constraints(cfg, self.limits_map.get(cfg.kind), profile=profile)
            cfg.profile = constraints.profile
            region_defs.append(cfg)
            init_list.extend(constraints.init)
            high_list.extend(constraints.upper)
            low_list.extend(constraints.lower)
            if cfg.how == "combine":
                ngaussian = Gsum_model(cfg.center, cfg.amplitude)
                self.profile_names.append(f"combine_{cfg.profile}")
                self.profile_functions.append(ngaussian)
            else:
                self.profile_functions.append(
                    PROFILE_FUNC_MAP.get(constraints.profile, gaussian_func)
                )
                self.profile_names.append(constraints.profile)
            if cfg.profile == "powerlaw":
                add_linear = False
            for i, name in enumerate(constraints.param_names):
                key = f"{name}_{cfg.line_name}_{cfg.component}_{cfg.kind}"
                self.params_dict[key] = idx + i
            # self.profile_params_index.append([idx,idx + len(constraints.param_names)])
            self.profile_params_index_list.append(
                np.arange(idx, idx + len(constraints.param_names))
            )
            idx += len(constraints.param_names)

        # if len(self.list)>0:
        #   amplitudes,centers = jnp.array([[i.amplitude,i.center] for i in self.list]).T
        #  ngaussian = Gsum_model(centers,amplitudes)
        # self.profile_names.append("combinedG")
        # self.profile_functions.append(ngaussian)
        # init_list.extend([0.1,0,10])
        # high_list.extend([10.,+30.,100.])
        # low_list.extend([0.0,-30.,0.01])
        # for i, name in enumerate(["amplitude","shift","width"]):
        #   key = f"{name}_{'fe'}_{20}_{'combinedG'}"
        #  self.params_dict[key] = idx + i
        # self.profile_params_index.append([idx,idx+2])
        # self.profile_params_index_list.append(np.arange(idx,idx+3))
        # region_defs.append(self.list)
        if add_linear:
            print("adding linear")
            # maybe a class method?
            self.profile_names.append("linear")
            self.profile_functions.append(linear)
            init_list.extend([0.1e-4, 0.5])
            high_list.extend([10.0, 10.0])
            low_list.extend([-10.0, -10.0])
            for i, name in enumerate(["m", "scale"]):
                key = f"{name}_{'continiumm'}_{0}_{'linear'}"
                self.params_dict[key] = idx + i
            # self.profile_params_index.append([idx,idx+2])
            self.profile_params_index_list.append(np.arange(idx, idx + 2))
            region_defs.append(
                SpectralLine(
                    center=0.0,
                    line_name='linear',
                    kind='continuum',
                    component=0,
                    profile='linear',
                    region='continuum',
                )
            )
        # Always add a linear continuum fallback

        self.initial_params = jnp.array(init_list)
        self.constraints = self._stack_constraints(low_list, high_list)  # constrains or limits
        self.get_param_coord_value = make_get_param_coord_value(
            self.params_dict, self.initial_params
        )  # important
        self.region_defs = region_defs

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

    # #This two will be move to other place soon.
    # def mapping_params(self,params,verbose=False):
    #     """
    #     params is a str or list
    #     [["width","broad"],"cont"]
    #     if verbose you can check if the mapping of parameters was correctly done
    #     """
    #     if isinstance(params,str):
    #         params = [params]
    #     match_list = []
    #     for param in params:
    #         if isinstance(param,str):
    #             param = [param]
    #         #print(self.params_dict.keys())
    #         #print([[self.params_dict[key],key] for key in self.params_dict.keys() if all([p in key for p in param])])

    #         match_list += ([self.params_dict[key] for key in self.params_dict.keys() if all([p in key for p in param])])

    #     match_list = jnp.array(match_list)
    #     unique_arr = jnp.unique(match_list)
    #     if verbose:
    #         print(np.array(list(self.params_dict.keys()))[unique_arr])#[])
    #     return unique_arr

    # This two will be move to other place soon.
    # def mapping_lines(self, where, what):
    #     entries = self.region_defs  # list of Lines
    #     idx, regions, centers, kinds, component, line_names = np.array([
    #         [i, e.region, e.center, e.kind, e.component, e.line_name] for i, e in enumerate(entries)
    #     ]).T

    #     dic_ = {"region": regions, "center": centers, "kind": kinds, "component": component,"line_names":line_names}

    #     # Normalize where and what to lists
    #     if isinstance(where, str):
    #         where = [where]
    #     if isinstance(what, str):
    #         what = [what]

    #     assert len(where) == len(what), "where and what must have the same length."

    #     # Build the mask
    #     mask = np.ones(len(entries), dtype=bool)  # Start with everything True
    #     for w, v in zip(where, what):
    #         mask &= np.char.find(dic_[w], v) >= 0

    #     # Apply mask
    #     mask_idx = np.where(mask)[0]

    #     idx = idx[mask_idx].astype(int)
    #     regions = regions[mask_idx]
    #     centers = centers[mask_idx].astype(float)
    #     kinds = kinds[mask_idx]
    #     components = component[mask_idx]
    #     entries = np.array(entries)[mask_idx]
    #     profile_functions = np.array(self.profile_functions)[mask_idx]
    #     initial_params = np.array(self.initial_params)[mask_idx]
    #     profile_params_index = np.array(self.profile_params_index)[mask_idx].astype(int)
    #     profile_params_index_list = [item for id in idx for item in self.profile_params_index_list[id]]
    #     line_names = line_names[mask_idx]

    #     # Return as a dictionary
    #     result = {
    #         "idx": idx,
    #         "region": regions,
    #         "center": centers,
    #         "kind": kinds,
    #         "component": components,
    #         "entries": entries,
    #         "profile_functions": profile_functions,
    #         "initial_params": initial_params,
    #         "profile_params_index": profile_params_index,
    #         "line_name": line_names,
    #         "profile_params_index_list":profile_params_index_list
    #     }

    #     return result
    ##This two will be move to other place soon.
    @property
    def pandas_params(self) -> pd.DataFrame:
        """Return fit parameters as a pandas DataFrame."""
        return pd.DataFrame(self.params, columns=list(self.params_dict.keys()))

    @property
    def pandas_region(self) -> pd.DataFrame:
        """Return region definitions as a pandas DataFrame."""
        return pd.DataFrame([vars(cfg) for cfg in self.region_defs])

    @staticmethod
    def _stack_constraints(low: List[float], high: List[float]) -> jnp.ndarray:
        """
        Utility to stack lower and upper bounds into a (N,2) array.
        """
        return jnp.stack([jnp.array(low), jnp.array(high)], axis=1)
