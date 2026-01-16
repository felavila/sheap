"""
Profile Constraint Maker
========================

This module defines the `ProfileConstraintMaker`, the central routine in *sheap*
for generating **initial values** and **bounds** of profile parameters associated
with each `SpectralLine`.

The constraint sets are specific to the type of profile being modeled:
- **Continuum profiles** (e.g. powerlaw, linear, broken powerlaw, Balmer continuum)
- **Emission line profiles** (e.g. gaussian, lorentzian, skewed)
- **Composite profiles** such as SPAF (Sum of Profiles with Adjustable Fractions)
- **Template profiles** (e.g. Fe templates, Balmer high-order templates, host MILES)

Returned objects are `ProfileConstraintSet` instances, which encapsulate:
- Initial parameter values
- Upper and lower bounds
- Profile name
- Parameter names
- The callable profile function

Notes
-----
- Constraints are informed by physically motivated defaults such as
    velocity FWHM limits, Doppler shift limits, and expected amplitude scales.
- SPAF and template profiles require additional metadata (subprofiles,
    canonical wavelengths, or template info).
- The `balmercontinuum` case uses raw parameterization
    (`T_raw`, `tau_raw`, `v_raw`) with transformations applied in the profile.

Examples
--------
.. code-block:: python

    from sheap.Core import SpectralLine, FittingLimits
    from sheap.Profiles.profile_handler import ProfileConstraintMaker

    sp = SpectralLine(line_name="Halpha", center=6563.0,
                    region="narrow", component=1,
                    amplitude=1.0, profile="gaussian")
    limits = FittingLimits(upper_fwhm_kms=5000, lower_fwhm_kms=200,
                        vshift_kms=600, max_amplitude=100)
    constraints = ProfileConstraintMaker(sp, limits)

    print(constraints.init, constraints.upper, constraints.lower)
"""

__author__ = 'felavila'


__all__ = [
    "ProfileConstraintMaker",
]

#TODO rename this 
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import jax.numpy as jnp
import jax
import numpy as np 

from sheap.Core import ProfileConstraintSet, FittingLimits, SpectralLine
from sheap.Utils.BasicFunctions import kms_to_wl
from sheap.Profiles.Profiles import PROFILE_FUNC_MAP,PROFILE_LINE_FUNC_MAP,PROFILE_CONTINUUM_FUNC_MAP


#TODO vshift -> vshift_kms in all the place  fwhm -> fwhm_v_kms in where we are using it.

def ProfileConstraintMaker(
    sp: SpectralLine,
    limits: FittingLimits,
    subprofile: Optional[str] = None,
    local_profile: Optional[callable] = None 
    ) ->ProfileConstraintSet:
    """
    Compute initial values and bounds for the profile parameters of a spectral line.

    Args:
        cfg: SpectralLine configuration.
        limits: Kinematic constraints (FWHM and center shift in km/s).
        profile: Default profile if cfg.profile is None.
        subprofile: Sub-profile function to use within compound models like SPAF.
    Returns:
        ProfileConstraintSet: Contains initial values, bounds, profile type, and parameter param_names.
    """
    selected_profile = sp.profile
    if selected_profile not in PROFILE_FUNC_MAP:
        raise ValueError(
            f"Profile '{selected_profile}' is not defined. "
        f"Available for continuum are : {list(PROFILE_CONTINUUM_FUNC_MAP.keys())+['balmercontinuum']} and for the profiles are {list(PROFILE_LINE_FUNC_MAP.keys())+ ['SPAF']}")
    if selected_profile == "SPAF":
        if not subprofile:
            raise ValueError(f"SPAF profile requires a defined subprofile avalaible options are {list(PROFILE_LINE_FUNC_MAP.keys())}.")
        if not isinstance(sp.amplitude, list):
            raise ValueError("SPAF profile requires cfg.amplitude to be a list of amplitudes.")
    if selected_profile in PROFILE_CONTINUUM_FUNC_MAP:  
        if selected_profile == 'powerlaw':
            return ProfileConstraintSet(
                init=[ -1,-1.7],
                upper=[0.0, 0.0],
                lower=[-5.0, -5.0],
                profile=selected_profile,
                param_names=PROFILE_FUNC_MAP.get(selected_profile).param_names,
                profile_fn = local_profile)

        if selected_profile == 'linear':
            return ProfileConstraintSet(
                init=[-0.01, 0.2],
                upper=[1.0, 1.0],
                lower=[-1.0, -1.0],
                profile=selected_profile,
                param_names=PROFILE_FUNC_MAP.get(selected_profile).param_names,
                profile_fn = local_profile)
        
        
        if selected_profile == "brokenpowerlaw":
            return ProfileConstraintSet(
                init=[0.0,-1.5, -2.5, 5500.0],
                upper=[5.0,0.0, 0.0, 8000.0],
                lower=[-5.0,-5.0, -5.0, 3000.0],
                profile=selected_profile,
                param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
                profile_fn = local_profile)
        #UNTIL HERE THE CONSTRAINS ARE TESTED AFTER THAT I dont know?
        if selected_profile == "logparabola":
            #should be testted
            return ProfileConstraintSet(
                init=[ 1.0,1.5, 0.1],
                upper=[10,3.0, 1.0, 10.0],
                lower=[0.0,0.0, 0.0],
                profile=selected_profile,
                param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
                profile_fn = local_profile)
        if selected_profile == "exp_cutoff":
            #should be testted
            return ProfileConstraintSet(
                init=[1.0,1.5,5000.0],
                upper=[10.0,3.0, 1.0, 1e5],
                lower=[0.0,0.0, 0.0],
                profile=selected_profile,
                param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
                profile_fn = local_profile)
        if selected_profile == "polynomial":
            #should be testted
            return ProfileConstraintSet(
                init=[1.0,0.0,0.0,0.0],
                upper=[10.0,10.0,10.0,10.0],
                lower=[0.0,-10.0,-10.0,-10.0],
                profile=selected_profile,
                param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
                profile_fn = local_profile)
    
    if selected_profile in PROFILE_LINE_FUNC_MAP:
        func = PROFILE_LINE_FUNC_MAP[selected_profile]
        param_names = func.param_names 
        center0   = sp.center
        shift0    = -1.0 if sp.region in ["outflow"] else 0.0
        cen_up    = center0 + kms_to_wl(limits.vshift_kms, center0)
        cen_lo    = center0 - kms_to_wl(limits.vshift_kms, center0)
        fwhm_lo   = kms_to_wl(limits.lower_fwhm_kms,    center0)
        fwhm_up   = kms_to_wl(limits.upper_fwhm_kms,    center0)
        amp_init =  float(sp.amplitude) / 10.0 * (-1.0 if sp.region in ["bal"] else 1.0)
        amp_lo =  limits.max_amplitude * (1.0 if sp.region in ["bal"] else 0.0)
        amp_up = limits.max_amplitude * (0.0 if sp.region in ["bal"] else 1.0)
        #fwhm_init = (fwhm_lo+fwhm_up)/2 * (1.0 if sp.region in ["outflow", "winds"] else 2.0)
        ##fwhm_init = fwhm_lo * (2.0 if sp.region in ["outflow", "winds"] else 1.0)
        fwhm_init = fwhm_lo * (1.0 if sp.region in ["outflow", "winds"] else (4.0 if sp.region in ["narrow"] else 2.0))
        logamp = -0.25 if sp.region=="narrow" else -2.0
        init, upper, lower = [], [], []
        for p in param_names:
            if p == "amplitude":
                init.append(10**logamp)
                upper.append(limits.max_amplitude)
                lower.append(0.0)
            
            # elif p == "amp":
            #     init.append(amp_init)
            #     upper.append(amp_up)
            #     lower.append(amp_lo)
                
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
        return ProfileConstraintSet(
            init=init,
            upper=upper,
            lower=lower,
            profile=selected_profile,
            param_names=param_names,
            profile_fn = local_profile
        )
        
    if selected_profile == "SPAF":
        param_names = local_profile.param_names
        logamp = -0.25 if sp.region=="narrow" else -2.0
        #the change here change all the results care.
        #fwhm_init =  fwhm_up if sp.region in ["outflow", "winds","narrow"] else fwhm_lo
        
        #fwhm_init = fwhm_lo * (1.0 if sp.region in ["outflow", "winds"] else (4.0 if sp.region in ["narrow"] else 2.0))
        init, upper, lower = [], [], []
        for _,p in enumerate(param_names):
            if "logamp" in p:
                if sp.region == "bal":
                    print("In log scale can be use bals.")
                    break
                # #for sign
                # if sp.region == "bal":
                #     init.append(-0.01)
                #     upper.append(0.0)
                #     lower.append(-1.0)
                # else:
                #     init.append(0.01)
                #     upper.append(1.0)
                #     lower.append(0.0)
                init.append(logamp)
                upper.append(1.0)
                lower.append(-15.0)
            
            elif "amplitude" in p:
                if sp.region == "bal":
                    init.append(0.0)
                    upper.append(0.0)
                    lower.append(-10)
                else:
                    init.append(10**logamp)
                    upper.append(10**1.0)
                    lower.append(0.0)
            
            elif p == "vshift_kms":
                init.append(0.0 if sp.component == 1 else (-1.5) ** (sp.component))
                upper.append(float(limits.vshift_kms))
                lower.append(-float(limits.vshift_kms))
                
            elif p == "fwhm_v_kms":
                init.append(np.log10((limits.lower_fwhm_kms+limits.upper_fwhm_kms)/2))
                #init.append(np.log10(float(limits.lower_fwhm)*(1.0 if sp.region in ["outflow", "winds"] else (4.0 if sp.region in ["narrow"] else 2.0))))
                upper.append(np.log10(float(limits.upper_fwhm_kms)))
                lower.append(np.log10(float(limits.lower_fwhm_kms)))
            
            # elif p in ("fwhm", "width", "fwhm_g", "fwhm_l"):
            #     # both Gaussian & Lorentzian widths share same kinematic bounds
            #     init.append(fwhm_init)
            #     upper.append(fwhm_up)
            #     lower.append(fwhm_lo)
                   
            # elif p in ("logfwhm", "logwidth", "logfwhm_g", "logfwhm_l"):
            #     # both Gaussian & Lorentzian widths share same kinematic bounds
            #     init.append(np.log10(fwhm_init))
            #     upper.append(np.log10(fwhm_up))
            #     lower.append(np.log10(fwhm_lo))

            # elif p == "alpha":
            #     # skewness parameter: start symmetric, allow ±5
            #     init.append(0.0)
            #     upper.append(5.0)
            #     lower.append(-5.0)

            # elif p in ("lambda", "lambda_"):
            #     # EMG decay: start at 1, allow up to 1/tau ~ 1e3
            #     init.append(1.0)
            #     upper.append(1e3)
            #     lower.append(0.0)
            
            # elif p == "p_shift":
            #     init.append(0)
            #     upper.append(1.)
            #    lower.append(-1.)
            else:
                raise ValueError(f"Unknown profile parameter '{p}' for '{selected_profile}' check ProfileeConstraintMaker or the define profile param_names {param_names}")

        if not (len(init) == len(upper) == len(lower) == len(param_names)):
            raise RuntimeError(f"Builder mismatch for '{selected_profile}_{subprofile}': {param_names}")
        
        return ProfileConstraintSet(
            init=init,
            upper=upper,
            lower=lower,
            profile=f"{selected_profile}_{subprofile}",
            param_names=param_names,
            profile_fn = local_profile
        )

    if selected_profile == "template" and sp.region == "fe":
        params_names = local_profile.param_names
        #logamplitude
        init = [1.0,np.log10(4000.0), 0.0] 
        upper = [2.0,np.log10(limits.upper_fwhm_kms), limits.vshift_kms] 
        lower = [-2.0,np.log10(limits.lower_fwhm_kms), -limits.vshift_kms]  
        #print(init,upper,lower)
        return ProfileConstraintSet(
            init= init,
            upper=upper,
            lower=lower,
            profile=selected_profile,
            param_names= params_names,
            profile_fn = local_profile
        )
    if sp.line_name == "balmerhighorder" and sp.profile == "template":
        params_names = local_profile.param_names
        init= [1.0, np.log10( 2000.0),0.0]
        upper= [2.0, np.log10(limits.upper_fwhm_kms), limits.vshift_kms]
        lower= [-2.0,np.log10(limits.lower_fwhm_kms) , -limits.vshift_kms]
        #print(PROFILE_FUNC_MAP.get(selected_profile))
        return ProfileConstraintSet(
            init= init,
            upper=upper,
            lower=lower,
            profile=selected_profile,
            param_names= params_names,
            profile_fn = local_profile
        )
        
    if selected_profile == "hostmiles":
        params_names = local_profile.param_names
        #testing limits
        init = [0.0,1e-3, 0.0] + [0.0] * len(params_names[3:])
        upper = [4.0,3.5, limits.vshift_kms] + [1.0] * len(params_names[3:]) # ? 
        lower = [-4.0,np.log10(limits.lower_fwhm_kms), -limits.vshift_kms]  + [0.0] * len(params_names[3:])
        return ProfileConstraintSet(
                init=init,
                upper=upper,
                lower=lower,
                profile=selected_profile,
                param_names=params_names,
                profile_fn = local_profile)

    
    if selected_profile == "balmercontinuum":
        return ProfileConstraintSet(
            init = [1e-2,  9.0,   -1.0,0.0],   # amplitude ~ 0.01 (in normalized units), T ≈ 4000+softplus(9) ~ 13k, tau0 ~ 0.31
            lower = [0.0,  -10.0,  -10.0,-5.0],  # keep amplitude >= 0; T_raw, tau_raw unconstrained but reasonable
            upper = [10.0,  20.0,   20.0,5.0],
            profile = selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
            profile_fn = local_profile)
