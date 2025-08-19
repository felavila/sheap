"""This module ?."""
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
from sheap.Profiles.profiles import PROFILE_FUNC_MAP,PROFILE_LINE_FUNC_MAP,PROFILE_CONTINUUM_FUNC_MAP

#from sheap.Utils.Constants import CANONICAL_WAVELENGTHS

        

#TODO profile handler is a unclear name we have to change it.
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
    #print("########",selected_profile,"##################")
    if selected_profile not in PROFILE_FUNC_MAP:
        raise ValueError(
            f"Profile '{selected_profile}' is not defined. "
        f"Available for continuum are : {list(PROFILE_CONTINUUM_FUNC_MAP.keys())+['balmercontinuum']} and for the profiles are {list(PROFILE_LINE_FUNC_MAP.keys())+ ['SPAF']}")
    if selected_profile == "SPAF":
        # ---- SPAF: Sum of Profiles with Free Amplitudes ----
        if not subprofile:
            raise ValueError(f"SPAF profile requires a defined subprofile avalaible options are {list(PROFILE_LINE_FUNC_MAP.keys())}.")
        if not isinstance(sp.amplitude, list):
            raise ValueError("SPAF profile requires cfg.amplitude to be a list of amplitudes.")
        #if sp.region not in CANONICAL_WAVELENGTHS:
         #   raise KeyError(f"Missing canonical wavelength for region='{sp.region}' in CANONICAL_WAVELENGTHS.")
    if selected_profile in PROFILE_CONTINUUM_FUNC_MAP:  
        if selected_profile == 'powerlaw':
            return ProfileConstraintSet(
                init=[-1.7, 0.1],
                upper=[0.0, 10.0],
                lower=[-5.0, 0.0],
                profile=selected_profile,
                param_names=PROFILE_FUNC_MAP.get(selected_profile).param_names,
                profile_fn = local_profile)#['index', 'scale'],

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
                init=[0.1,-1.5, -2.5, 5500.0],
                upper=[10.0,0.0, 0.0, 8000.0],
                lower=[0.0,-5.0, -5.0, 3000.0],
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
        cen_up    = center0 + kms_to_wl(limits.v_shift, center0)
        cen_lo    = center0 - kms_to_wl(limits.v_shift, center0)
        fwhm_lo   = kms_to_wl(limits.lower_fwhm,    center0)
        fwhm_up   = kms_to_wl(limits.upper_fwhm,    center0)
        amp_init =  float(sp.amplitude) / 10.0 * (-1.0 if sp.region in ["bal"] else 1.0)
        amp_lo =  limits.max_amplitude * (1.0 if sp.region in ["bal"] else 0.0)
        amp_up = limits.max_amplitude * (0.0 if sp.region in ["bal"] else 1.0)
        #fwhm_init = fwhm_lo * (2.0 if sp.region in ["outflow", "winds"] else 1.0)
        fwhm_init = (fwhm_lo+fwhm_up)/2 * (1.0 if sp.region in ["outflow", "winds"] else 2.0)
        init, upper, lower = [], [], []
        for p in param_names:
            if p == "logamp":
                init.append(np.log10(amp_init))
                upper.append(np.log10(limits.max_amplitude))
                lower.append(-10.0)
            
            elif p == "amp":
                init.append(amp_init)
                upper.append(amp_up)
                lower.append(amp_lo)
                
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
        #func = PROFILE_LINE_FUNC_MAP[subprofile]
        param_names = local_profile.param_names
        #print(limits.canonical_wavelengths)
        lambda0 = limits.canonical_wavelengths
        #CANONICAL_WAVELENGTHS[sp.region]
        shift_init = 0.0 if sp.component == 1 else (-1.0 if sp.region=="outflow" else 2*(-1.0) ** (sp.component))
        shift_limit = kms_to_wl(limits.v_shift, lambda0)
        fwhm_up   = kms_to_wl(limits.upper_fwhm,    lambda0)
        fwhm_lo   = kms_to_wl(limits.lower_fwhm,    lambda0)
        logamp = -0.25 if sp.region=="narrow" else -2.0
        #amp_init =  (limits.max_amplitude / 10.0) * (-1.0 if sp.region in ["bal"] else 1.0)
        #amp_lo =  limits.max_amplitude * (-1.0 if sp.region in ["bal"] else 0.0)
        #amp_up = limits.max_amplitude * (0.0 if sp.region in ["bal"] else 1.0)
        #fwhm_init = fwhm_lo * (2.0 if sp.region in ["outflow", "winds"] else 1.0)
        #fwhm_init = (fwhm_lo+fwhm_up)/2 * (1.0 if sp.region in ["outflow", "winds"] else 2.0)
        fwhm_init = fwhm_lo * (1.0 if sp.region in ["outflow", "winds"] else 2.0)
        #fwhm_init = fwhm_lo * (1.0 if sp.region in ["outflow", "winds"] else 2.0)
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
                    init.append(-1.0)
                    upper.append(0.0)
                    lower.append(-100)
                else:
                    init.append(10**logamp)
                    upper.append(10**1.0)
                    lower.append(10**-15.0)
                   
            elif p == "shift":
                init.append(shift_init)
                upper.append(shift_limit)
                lower.append(-shift_limit)
                
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
            else:
                raise ValueError(f"Unknown profile parameter '{p}' for '{selected_profile}' check ProfileeConstraintMaker or the define profile param_names {param_names}")
            #  elif p == "logshift":
            #     init.append(0.0+(sp.component-1.0)*1e-3)
            #     upper.append(np.log10( (lambda0 + 2*shift_upper) / lambda0 ))
            #     lower.append(np.log10( (lambda0 - 2*shift_upper) / lambda0 ))
                
        #print("n total params",len(init))
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

    if selected_profile == "fetemplate" and sp.region == "fe":
        #maybe add a warning here
        lambda0 = limits.canonical_wavelengths
        shift = kms_to_wl(limits.v_shift, lambda0)
        params_names = local_profile.param_names
        #logamplitude
        init = [1.0,np.log10(4000.0), 0.0] 
        upper = [10.0,np.log10(limits.upper_fwhm), shift] 
        lower = [-2.0,np.log10(limits.lower_fwhm), -shift]  
        #print(init,upper,lower)
        return ProfileConstraintSet(
            init= init,
            upper=upper,
            lower=lower,
            profile=selected_profile,
            param_names= params_names,
            profile_fn = local_profile
        )
    if sp.line_name == "balmerhighorder" and sp.profile == "fetemplate":
        print("trying to add balmerhighorder")
        lambda0 = 3675.0 #limits.canonical_wavelengths
        v_shift = 1500.0 
        init_fwhm = 2000.0
        upper_fwhm =  8000.0
        lower_fwhm =  800.0
        shift = kms_to_wl(v_shift, lambda0)
        params_names = local_profile.param_names
        init= [1.0, np.log10(init_fwhm),0.0]
        upper= [10.0, np.log10(upper_fwhm), shift]
        lower= [-2.0,np.log10(lower_fwhm) , -shift]
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
        lambda0 = limits.canonical_wavelengths
        shift = kms_to_wl(limits.v_shift, lambda0)
        params_names = local_profile.param_names
        init = [5.0,np.log10(400.0), 0.0] + [0.0] * len(params_names[3:])
        upper = [10.0,np.log10(limits.upper_fwhm), shift] + [1.0] * len(params_names[3:])#
        lower = [-2.0,np.log10(limits.lower_fwhm), -shift]  + [0.0] * len(params_names[3:])
        #print(init,upper,lower)
        return ProfileConstraintSet(
                init=init,
                upper=upper,
                lower=lower,
                profile=selected_profile,
                param_names=params_names,
                profile_fn = local_profile)

    
    if selected_profile == "balmercontinuum":
        return ProfileConstraintSet(
            init=[0.01, 10000.0, 1.0],
            upper=[2.0, 25000.0, 2.0],
            lower=[0.0,8000.0 , 1e-3],
            profile = selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,
            profile_fn = local_profile)
    
    

# balmer_highorder:
#   upper_fwhm: 8000.0      # km/s
#   lower_fwhm: 800.0       # km/s
#   v_shift: 1500.0         # km/s (±)
#   max_amplitude: 10.0     # dimensionless scaling