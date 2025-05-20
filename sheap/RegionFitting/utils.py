from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp

from sheap.DataClass.DataClass import ConstraintSet, FittingLimits, SpectralLine
from sheap.Tools.others import kms_to_wl
from sheap.FunctionsMinimize.profiles import PROFILE_FUNC_MAP 



def make_constraints(
    cfg: SpectralLine, limits: FittingLimits, profile="gaussian"
) -> ConstraintSet:
    """
    Compute initial values and bounds for the profile parameters of a spectral line.

    Args:
        cfg: SpectralLine configuration.
        limits: Kinematic constraints (velocity width and center shift in km/s).

    Returns:
        A ConstraintSet containing init values, upper/lower bounds, profile type, and parameter names.
    """
    selected_profile = cfg.profile or profile

    if selected_profile not in PROFILE_FUNC_MAP:
        raise ValueError(
            f"Profile '{selected_profile}' is not defined in PROFILE_FUNC_MAP. "
            f"Available profiles: {list(PROFILE_FUNC_MAP.keys())}"
        )

    if cfg.kind.lower() == 'fe':
        if cfg.how == 'template':
            if not cfg.which:
                raise ValueError("Fe template must define 'which' (e.g., 'OP', 'UV')")

            return ConstraintSet(
                init=[3.045, 0.0, 1.0],
                upper=[3.5, 100.0, 100.0],
                lower=[2.7, -100.0, 0.0],
                profile='fitFe' + cfg.which,
                param_names=['logFWHM', 'shift', 'scale'],
            )

        elif cfg.how == "combine":
            center = cfg.center
            shift = -5 if cfg.kind == "outflow" else 0

            shift_upper = 10.0
            shift_lower = -10.0
            width_upper = 85
            width_lower = 8.5

            return ConstraintSet(
                init=[1.0, 0, float(width_lower)],
                upper=[5.0, shift_upper, width_upper],
                lower=[0.0, shift_lower, width_lower],
                profile=selected_profile,
                param_names=['amplitude', 'shift', 'width'],
            )

    if selected_profile == 'powerlaw':
        return ConstraintSet(
            init=[-1.1, 0.0],
            upper=[-1.0, 10.0],
            lower=[-3.0, 0.0],
            profile='powerlaw',
            param_names=['index', 'scale'],
        )

    if selected_profile == "brokenpowerlaw":
        return ConstraintSet(
            init=[-1.7, 0.0, 0.1, 5500.0],
            upper=[0.0, 1.0, 10.0, 7000.0],
            lower=[-3.0, -1.0, 0.0, 4000],
            profile='brokenpowerlaw',
            param_names=['index1', 'index2', 'scale', 'refer'],
        )

    if selected_profile == "balmerconti":
        return ConstraintSet(
            init=[1.0, 10000.0, 1.0],
            upper=[10.0, 50000, 2.0],
            lower=[0.0, 5000.0, 0.01],
            profile='balmerconti',
            param_names=['scale', "T", 'τ0'],
        )

    if selected_profile == "gaussian":
        center = cfg.center
        shift = -5 if cfg.kind == "outflow" else 0

        center_upper = center + kms_to_wl(limits.center_shift, center)
        center_lower = center - kms_to_wl(limits.center_shift, center)
        width_upper = kms_to_wl(limits.upper_width, center)
        width_lower = kms_to_wl(limits.lower_width, center)

        return ConstraintSet(
            init=[float(cfg.amplitude), float(center + shift), float(width_lower)],
            upper=[limits.max_amplitude, center_upper, width_upper],
            lower=[0.0, center_lower, width_lower],
            profile='gaussian',
            param_names=['amplitude', 'center', 'width'],
        )

    raise NotImplementedError(
        f"No constraints defined for profile '{selected_profile}'. "
        f"Define its ConstraintSet explicitly in make_constraints."
    )



def make_get_param_coord_value(
    params_dict: Dict[str, int], initial_params: jnp.ndarray
) -> Callable[[str, str, Union[str, int], str, bool], Tuple[int, float, str]]:
    """
    Returns a function to retrieve the index, value, and name of a parameter based on its key parts.

    Args:
        params_dict: Mapping of parameter keys to indices.
        initial_params: Initial parameter array.

    Returns:
        A function to get parameter info by name, line name, component, and kind.
    """

    def get_param_coord_value(
        param: str,
        line_name: str,
        component: Union[str, int],
        kind: str,
        verbose: bool = False,
    ) -> Tuple[int, float, str]:
        key = f"{param}_{line_name}_{component}_{kind}"
        pos = params_dict.get(key)
        if pos is None:
            raise KeyError(f"Key '{key}' not found in params_dict.")
        if verbose:
            print(f"{key}: value = {initial_params[pos]}")
        return pos, float(initial_params[pos]), param

    return get_param_coord_value
