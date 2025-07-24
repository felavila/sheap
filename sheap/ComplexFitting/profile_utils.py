from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import jax.numpy as jnp
import jax
import numpy as np 

from sheap.Assistants import ConstraintSet, FittingLimits, SpectralLine
from sheap.Tools.spectral_basic import kms_to_wl
from sheap.Functions.profiles import PROFILE_FUNC_MAP,PROFILE_LINE_FUNC_MAP,PROFILE_CONTINUUM_FUNC_MAP


#TODO change make _get_param_coord_value to some other place 
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


CANONICAL_WAVELENGTHS = {
    'broad': 4861.0,    # Hbeta
    'narrow': 4861.0,   # [OIII]
    'outflow': 5007.0,  # [OIII]
    'fe': 4570.0,       # Mean FeII blend
    'nlr': 6583.0,       # [NII]
    "winds": 5007.0} #why?



DEFAULT_LIMITS = {
    'broad':   {'upper_fwhm': 10000.0,  'lower_fwhm': 1000.875, 'center_shift': 5000.0,  'v_shift': 5000.0,  'max_amplitude': 10.0},
    'narrow':  {'upper_fwhm': 1000.0,   'lower_fwhm': 100.0,     'center_shift': 2500.0,  'v_shift': 2500.0,  'max_amplitude': 10.0},
    'outflow': {'upper_fwhm': 20000.0,  'lower_fwhm': 5000.875,  'center_shift': 3000.0,  'v_shift': 3000.0,  'max_amplitude': 10.0},
    'fe':      {'upper_fwhm': 7065.0,   'lower_fwhm': 117.75,    'center_shift': 4570.0,  'v_shift': 4570.0,  'max_amplitude': 0.07},
    'nlr':     {'upper_fwhm': 2355.0,   'lower_fwhm': 117.75,    'center_shift': 1500.0,  'v_shift': 1500.0,  'max_amplitude': 10.0},
    'winds':   {'upper_fwhm': 15000.0,  'lower_fwhm': 5000.0,    'center_shift': 8000.0,  'v_shift': 8000.0,  'max_amplitude': 10.0},
    'host':    {'upper_fwhm': 0.0,      'lower_fwhm': 0.0,       'center_shift': 0.0,     'v_shift': 0.0,     'max_amplitude': 0.0},
}

        
def profile_handler(
    sp: SpectralLine,
    limits: FittingLimits,
    subprofile: Optional[str] = None,
    local_profile: Optional[callable] = None 
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
    selected_profile = sp.profile
    if selected_profile not in PROFILE_FUNC_MAP:
        raise ValueError(
            f"Profile '{selected_profile}' is not defined. "
        f"Available for continuum are : {list(PROFILE_CONTINUUM_FUNC_MAP.keys())+["balmercontinuum"]} and for the profiles are {list(PROFILE_LINE_FUNC_MAP.keys())+ ["SPAF"]}")
    if selected_profile == "SPAF":
        # ---- SPAF: Sum of Profiles with Free Amplitudes ----
        if not subprofile:
            raise ValueError(f"SPAF profile requires a defined subprofile avalaible options are {list(PROFILE_LINE_FUNC_MAP.keys())}.")
        if not isinstance(sp.amplitude, list):
            raise ValueError("SPAF profile requires cfg.amplitude to be a list of amplitudes.")
        if sp.region not in CANONICAL_WAVELENGTHS:
            raise KeyError(f"Missing canonical wavelength for region='{sp.region}' in CANONICAL_WAVELENGTHS.")
        
    if selected_profile == "balmercontinuum":
        return ConstraintSet(
            init=[1.0, 10000.0, 1.0],
            upper=[10.0, 50000.0, 2.0],
            lower=[0.0, 5000.0, 0.01],
            profile = selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
            profile_fn = local_profile)

    if selected_profile == 'powerlaw':
        return ConstraintSet(
            init=[-1.7, 0.0],
            upper=[0.0, 10.0],
            lower=[-5.0, 0.0],
            profile=selected_profile,
            param_names=PROFILE_FUNC_MAP.get(selected_profile).param_names,
            profile_fn = local_profile)#['index', 'scale'],

    if selected_profile == 'linear':
        return ConstraintSet(
            init=[-0.01, 0.2],
            upper=[1.0, 1.0],
            lower=[-1.0, -1.0],
            profile=selected_profile,
            param_names=PROFILE_FUNC_MAP.get(selected_profile).param_names,
            profile_fn = local_profile)
    
    
    if selected_profile == "brokenpowerlaw":
        return ConstraintSet(
            init=[0.1,-1.5, -2.5, 5500.0],
            upper=[10.0,0.0, 0.0, 8000.0],
            lower=[0.0,-5.0, -5.0, 3000.0],
            profile=selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
            profile_fn = local_profile)
    #UNTIL HERE THE CONSTRAINS ARE TESTED AFTER THAT I dont know?
    if selected_profile == "logparabola":
        #should be testted
        return ConstraintSet(
            init=[ 1.0,1.5, 0.1],
            upper=[10,3.0, 1.0, 10.0],
            lower=[0.0,0.0, 0.0],
            profile=selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
            profile_fn = local_profile)
    if selected_profile == "exp_cutoff":
        #should be testted
        return ConstraintSet(
            init=[1.0,1.5,5000.0],
            upper=[10.0,3.0, 1.0, 1e5],
            lower=[0.0,0.0, 0.0],
            profile=selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
            profile_fn = local_profile)
    if selected_profile == "polynomial":
        #should be testted
        return ConstraintSet(
            init=[1.0,0.0,0.0,0.0],
            upper=[10.0,10.0,10.0,10.0],
            lower=[0.0,-10.0,-10.0,-10.0],
            profile=selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
            profile_fn = local_profile)
    if selected_profile in PROFILE_LINE_FUNC_MAP:
        func = PROFILE_LINE_FUNC_MAP[selected_profile]
        names = func.param_names 
        # base kinematics
        center0   = sp.center
        shift0    = -5.0 if sp.region in ["outflow", "winds"] else 0.0
        cen_up    = center0 + kms_to_wl(limits.center_shift, center0)
        cen_lo    = center0 - kms_to_wl(limits.center_shift, center0)
        fwhm_lo   = kms_to_wl(limits.lower_fwhm,    center0)
        fwhm_up   = kms_to_wl(limits.upper_fwhm,    center0)
        fwhm_init = fwhm_lo * (2.0 if sp.region in ["outflow", "winds"] else 1.0)
        amp_init  = np.log10(float(sp.amplitude) / 10.0)

        init, upper, lower = [], [], []

        for p in names:
            if p == "logamp":
                init.append(amp_init)
                upper.append(np.log10(limits.max_amplitude))
                lower.append(-10.0)

            elif p == "center":
                init.append(center0 + shift0)
                upper.append(cen_up)
                lower.append(cen_lo)

            elif p in ("fwhm", "width", "fwhm_g", "fwhm_l"):
                # both Gaussian & Lorentzian widths share same kinematic bounds
                init.append(fwhm_init)
                upper.append(fwhm_up)
                lower.append(fwhm_lo)

            elif p == "alpha":
                # skewness parameter: start symmetric, allow ±5
                init.append(0.0)
                upper.append(5.0)
                lower.append(-5.0)

            elif p in ("lambda", "lambda_"):
                # EMG decay: start at 1, allow up to 1/tau ~ 1e3
                init.append(1.0)
                upper.append(1e3)
                lower.append(0.0)

            else:
                raise ValueError(f"Unknown profile parameter '{p}' for '{selected_profile}'")
        return ConstraintSet(
            init=init,
            upper=upper,
            lower=lower,
            profile=selected_profile,
            param_names=names,
            profile_fn = local_profile
        )
        
    if selected_profile == "SPAF":
        #func = PROFILE_LINE_FUNC_MAP[subprofile]
        params_names = local_profile.param_names
        #print(params_names)
        #amp_list = sp.amplitude

        #names = [f"amplitude{n}" for n in range(len(amp_list))] + ["shift"] + func.param_names[2:]
        # base kinematics
        lambda0 = CANONICAL_WAVELENGTHS[sp.region]
        shift_init = 0.0 if sp.component == 1 else (-5.0 if sp.region=="outflow" else 2*(-1.0) ** (sp.component))
        shift_upper = kms_to_wl(limits.center_shift, lambda0)
        fwhm_up   = kms_to_wl(limits.upper_fwhm,    lambda0)
        fwhm_lo   = kms_to_wl(limits.lower_fwhm,    lambda0)
        logamp = -0.25 if sp.region=="narrow" else -2.0
        #print(shift_upper,fwhm_lo,fwhm_up)
        #if sp.region in ["narrow"]:
        #   fwhm_init = fwhm_up
        #else:
        fwhm_init = fwhm_lo * (1.0 if sp.region in ["outflow", "winds"] else 2.0)

        init, upper, lower = [], [], []

        for _,p in enumerate(params_names):
            #print(p)
            if "logamp" in p:
                init.append(logamp)
                upper.append(1.0)
                lower.append(-15.0)
            elif p == "shift":
                init.append(shift_init)
                upper.append(shift_upper)
                lower.append(-shift_upper)
                
            elif p == "v_shift":
                init.append(0)
                upper.append(limits.v_shift)
                lower.append(-limits.v_shift)

            elif p in ("fwhm", "width", "fwhm_g", "fwhm_l"):
                # both Gaussian & Lorentzian widths share same kinematic bounds
                init.append(fwhm_init)
                upper.append(fwhm_up)
                lower.append(fwhm_lo)
                   
            elif p in ("logfwhm", "logwidth", "logfwhm_g", "logfwhm_l"):
                # both Gaussian & Lorentzian widths share same kinematic bounds
                init.append(np.log10(fwhm_init))
                upper.append(np.log10(fwhm_up))
                lower.append(np.log10(fwhm_lo))

            elif p == "alpha":
                # skewness parameter: start symmetric, allow ±5
                init.append(0.0)
                upper.append(5.0)
                lower.append(-5.0)

            elif p in ("lambda", "lambda_"):
                # EMG decay: start at 1, allow up to 1/tau ~ 1e3
                init.append(1.0)
                upper.append(1e3)
                lower.append(0.0)
            
            elif p == "p_shift":
                init.append(0)
                upper.append(1.)
                lower.append(-1.)
            #  elif p == "logshift":
            #     init.append(0.0+(sp.component-1.0)*1e-3)
            #     upper.append(np.log10( (lambda0 + 2*shift_upper) / lambda0 ))
            #     lower.append(np.log10( (lambda0 - 2*shift_upper) / lambda0 ))
                
        #print("n total params",len(init))
        if not (len(init) == len(upper) == len(lower) == len(params_names)):
            raise RuntimeError(f"Builder mismatch for '{selected_profile}_{subprofile}': {params_names}")
        
        return ConstraintSet(
            init=init,
            upper=upper,
            lower=lower,
            profile=f"{selected_profile}_{subprofile}",
            param_names=params_names,
            profile_fn = local_profile
        )

    if selected_profile == "fetemplate":
        #maybe add a warning here
        params_names = local_profile.param_names
        init = [1.0,3.0, 0.0] 
        upper = [10.0,3.8, 50.0] 
        lower = [-2.0,2.0, -50.0]  
        #print(params_names)
        return ConstraintSet(
            init= init,
            upper=upper,
            lower=lower,
            profile=selected_profile,
            param_names= params_names,
            profile_fn = local_profile
        )
        
    if selected_profile == "hostmiles":
        params_names = local_profile.param_names
        #print(len(params_names[2:]))
        init = [5.0,3.0, 0.0] + [0.0] * len(params_names[3:])
        upper = [10.0,3.8, 50.0] + [1.0] * len(params_names[3:])
        lower = [-2.0,2.0, -50.0]  + [0.0] * len(params_names[3:])
        return ConstraintSet(
                init=init,
                upper=upper,
                lower=lower,
                profile=selected_profile,
                param_names=params_names,
                profile_fn = local_profile)



