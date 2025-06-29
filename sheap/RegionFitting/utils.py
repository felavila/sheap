from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp

from sheap.DataClass.DataClass import ConstraintSet, FittingLimits, SpectralLine
from sheap.Tools.spectral_basic import kms_to_wl
from sheap.Functions.profiles import PROFILE_FUNC_MAP,PROFILE_LINE_FUNC_MAP,PROFILE_CONTINUUM_FUNC_MAP




CANONICAL_WAVELENGTHS = {
    'broad': 4861.0,    # Hbeta
    'narrow': 5007.0,   # [OIII]
    'outflow': 5007.0,  # [OIII]
    'fe': 4570.0,       # Mean FeII blend
    'nlr': 6583.0,       # [NII]
    "winds": 5007.0} #why?


DEFAULT_LIMITS = {
    'broad': dict(
        upper_fwhm=11775.0,  # FWHM ~ 1000–10000 km/s for broad lines
        lower_fwhm=1000.875,
        center_shift=5000.0,
        max_amplitude=10.0,
        # Ref: Sulentic+2000, Shen+2011
    ),
    'narrow': dict(
        upper_fwhm=471.0,   # FWHM ~ 200–1000 km/s typical for NLR
        lower_fwhm=117.75,
        center_shift=2500.0,
        max_amplitude=10.0,
        # Ref: Osterbrock & Ferland 2006, Véron-Cetty+2001
    ),
    #not sure about this 
    'outflow': dict(
        upper_fwhm=11775.0,   # FWHM for blueshifted or broad outflowing components
        lower_fwhm=5000.875,
        center_shift=3000.0,
        max_amplitude=10.0,
        # Ref: Bischetti+2017, Perrotta+2019
    ),
    'fe': dict(
        upper_fwhm=7065.0,   # Typical Fe II FWHM from 800 to 2500 km/s
        lower_fwhm=494.55,
        center_shift=2500.0,
        max_amplitude=0.07,
        # Ref: Kovačević+2010, Ilic+2022
    ),
    'nlr': dict(
        upper_fwhm=2355.0,   # NLR lines are narrow; similar to 'narrow' but possibly less broadened
        lower_fwhm=117.75,
        center_shift=1500.0,
        max_amplitude=10.0,),
        # Ref: Bennert+2006, Hainline+2013
    "winds": dict(upper_fwhm   = 15000.0,   # up to 15 000 km/s for very fast winds
    lower_fwhm   = 5000.0,    # minimum ~ 5 000 km/s
    center_shift = 8000.0,    # allow blueshifts up to ~8 000 km/s
    max_amplitude= 10.0,      # same cap as your other lines
        )
}



def make_constraints(
    cfg: SpectralLine,
    limits: FittingLimits,
    profile: str = "gaussian",
    subprofile: Optional[str] = None
) -> ConstraintSet:
    """
    Compute initial values and bounds for the profile parameters of a spectral line.

    Args:
        cfg: SpectralLine configuration.
        limits: Kinematic constraints (FWHM and center shift in km/s).
        profile: Default profile if cfg.profile is None.
        subprofile: Sub-profile function to use within compound models like SPAF.

    Returns:
        ConstraintSet: Contains initial values, bounds, profile type, and parameter names.
    """
    selected_profile = cfg.profile or profile
    if selected_profile not in PROFILE_FUNC_MAP and selected_profile not in {"SPAF", "balmerconti", "brokenpowerlaw"}:
        raise ValueError(
            f"Profile '{selected_profile}' is not defined. "
            f"Available: {list(PROFILE_FUNC_MAP.keys()) + ['SPAF', 'balmerconti', 'brokenpowerlaw']}"
        )

    # ---- Template Fe profiles (logFWHM, shift, scale) ----
    if cfg.region.lower() == 'fe' and cfg.how == 'template':
        if not cfg.which:
            raise ValueError("Fe template must define 'which' (e.g., 'OP', 'UV')")
        return ConstraintSet(
            init=[3.045, 0.0, 1.0],
            upper=[3.8, 100.0, 100.0],
            lower=[2.7, -100.0, 0.0],
            profile='fitFe' + cfg.which,
            param_names=['logFWHM', 'shift', 'scale'],
        )

    # ---- Balmer continuum ----
    if selected_profile == "balmerconti":
        return ConstraintSet(
            init=[1.0, 10000.0, 1.0],
            upper=[10.0, 50000.0, 2.0],
            lower=[0.0, 5000.0, 0.01],
            profile='balmerconti',
            param_names=['scale', "T", 'τ0'],
        )

    # ---- Power-law continuum ----
    if selected_profile == 'powerlaw':
        return ConstraintSet(
            init=[-1.1, 0.0],
            upper=[-1.0, 10.0],
            lower=[-3.0, 0.0],
            profile='powerlaw',
            param_names=['index', 'scale'],
        )

    # ---- Linear continuum ----
    if selected_profile == 'linear':
        return ConstraintSet(
            init=[0.1e-4, 0.5],
            upper=[10.0, 10.0],
            lower=[-3.0, 0.0],
            profile='linear',
            param_names=["scale_b", "scale_m"],
        )

    # ---- Broken Power-law ----
    if selected_profile == "brokenpowerlaw":
        return ConstraintSet(
            init=[-1.7, 0.0, 0.1, 5500.0],
            upper=[0.0, 1.0, 10.0, 7000.0],
            lower=[-3.0, -1.0, 0.0, 4000.0],
            profile='brokenpowerlaw',
            param_names=['index1', 'index2', 'scale', 'refer'],
        )

    # ---- Standard Gaussian ----
    if selected_profile == "gaussian":
        center = cfg.center
        shift = -5 if cfg.region == "outflow" else 0

        center_upper = center + kms_to_wl(limits.center_shift, center)
        center_lower = center - kms_to_wl(limits.center_shift, center)
        fwhm_upper = kms_to_wl(limits.upper_fwhm, center)
        fwhm_lower = kms_to_wl(limits.lower_fwhm, center)
        fwhm_init = fwhm_lower * (2.0 if cfg.region == "outflow" else 1.0)

        return ConstraintSet(
            init=[float(cfg.amplitude) / 10, float(center + shift), float(fwhm_init)],
            upper=[limits.max_amplitude, center_upper, fwhm_upper],
            lower=[0.0, center_lower, fwhm_lower],
            profile='gaussian',
            param_names=['amplitude', 'center', 'fwhm'],
        )

    # ---- SPAF: Sum of Profiles with Free Amplitudes ----
    if selected_profile == "SPAF":
        #print(type(cfg.amplitude))
        if not subprofile:
            raise ValueError("SPAF profile requires a defined subprofile (e.g., 'gaussian').")
        if not isinstance(cfg.amplitude, list):
            raise ValueError("SPAF profile requires cfg.amplitude to be a list of amplitudes.")
        if cfg.region not in CANONICAL_WAVELENGTHS:
            raise KeyError(f"Missing canonical wavelength for region='{cfg.region}' in CANONICAL_WAVELENGTHS.")

        lambda0 = CANONICAL_WAVELENGTHS[cfg.region]
        shift_upper = kms_to_wl(limits.center_shift, lambda0)
        fwhm_upper = kms_to_wl(limits.upper_fwhm, lambda0)
        fwhm_lower = kms_to_wl(limits.lower_fwhm, lambda0)

        amp_list = list(cfg.amplitude)
        amp_upper = [1.0] * len(amp_list)
        if cfg.region == "fe":
            #print("xd")
            amp_upper = [0.2] * len(amp_list)
        amp_lower = [0.0] * len(amp_list)
        shift_init = 0.0 if cfg.component == 1 else (-2.0) ** cfg.component

        return ConstraintSet(
            init=amp_list + [shift_init, (fwhm_upper - fwhm_lower) / 2.0],
            upper=amp_upper + [shift_upper * 5.0, fwhm_upper],
            lower=amp_lower + [-shift_upper * 5.0, fwhm_lower],
            profile= f"{selected_profile}_{subprofile}",
            param_names=[f"amplitude{n}" for n in range(len(amp_list))] + ['shift', 'fwhm'],
        )

    # ---- If no known configuration matched ----
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
        A function to get parameter info by name, line name, component, and region.
    """

    def get_param_coord_value(
        param: str,
        line_name: str,
        component: Union[str, int],
        region: str,
        verbose: bool = False,
    ) -> Tuple[int, float, str]:
        key = f"{param}_{line_name}_{component}_{region}"
        pos = params_dict.get(key)
        if pos is None:
            raise KeyError(f"Key '{key}' not found in params_dict.")
        if verbose:
            print(f"{key}: value = {initial_params[pos]}")
        return pos, float(initial_params[pos]), param

    return get_param_coord_value
