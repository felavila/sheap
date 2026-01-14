"""
Profiles
================

This module defines the profile registry and composite profile logic used in *sheap*.

It collects all available spectral profile functions (line, continuum, templates)
into centralized dictionaries for lookup during region building and fitting. It
also provides helper routines to wrap existing profiles and to build compound
SPAF profiles (Sum of Profiles with Free Amplitudes).

Contents
--------
- ``PROFILE_LINE_FUNC_MAP`` : Registry of line profiles
    (Gaussian, Lorentzian, Voigt, skewed Gaussian, EMG, top-hat).
- ``PROFILE_CONTINUUM_FUNC_MAP`` : Registry of continuum profiles
    (linear, power law, broken power law, log-parabola, exponential cutoff, polynomial).
- ``PROFILE_FUNC_MAP`` : Unified registry combining line, continuum, template,
    and composite profiles (including ``SPAF`` and Balmer continuum).

Functions
---------
- ``wrap_profile_with_center_override`` :
    Wraps a profile to allow overriding its center parameter at call time.
    Ensures JAX compatibility and handles linear→log amplitude conversion.

- ``SPAF`` :
    Constructs a Sum-of-Profiles-with-Free-Amplitudes profile from a set of
    line centers and amplitude-tying rules, sharing global shape parameters.

Notes
-----
- All profiles are JAX-compatible and decorated with
    ``@with_param_names`` for consistent parameter naming.
- The registries are used by ``ComplexBuilder`` and downstream fitting classes
    to resolve profile names defined in YAML templates into callable functions.
"""
__author__ = 'felavila'


__all__ = [
    "SPAF",
    "wrap_profile_with_center_override",
]

from typing import Callable, Dict, List, Tuple



from sheap.Core import ProfileFunc

from sheap.Profiles.profiles_continuum import (linear, powerlaw, brokenpowerlaw,logparabola,exp_cutoff,polynomial)
from sheap.Profiles.profiles_lines import (gaussian_fwhm, lorentzian_fwhm, skewed_gaussian,emg_fwhm, top_hat, voigt_pseudo)

from sheap.Profiles.profiles_templates import make_template_function,make_host_function

from sheap.Profiles.Combine import SPAF_loglambda
from sheap.Profiles.balmercontinuum import balmercontinuum


# from sheap.sheap.Profiles.depreted.Combine import SPAF

# # Low-level line profiles (require center+amplitude inside param vector)
PROFILE_LINE_FUNC_MAP: Dict[str, ProfileFunc] = {'gaussian': gaussian_fwhm,'lorentzian': lorentzian_fwhm,'voigt_pseudo': voigt_pseudo,'skewed_gaussian': skewed_gaussian,'emg_fwhm': emg_fwhm,'top_hat': top_hat}

PROFILE_CONTINUUM_FUNC_MAP: Dict[str, ProfileFunc] = {'linear': linear,'powerlaw': powerlaw,'brokenpowerlaw': brokenpowerlaw,'logparabola': logparabola,'exp_cutoff': exp_cutoff,'polynomial': polynomial}


# Full profile registry (for spectral modeling)
PROFILE_FUNC_MAP: Dict[str, ProfileFunc] = {'balmercontinuum': balmercontinuum,'template': make_template_function,'SPAF': SPAF_loglambda,"hostmiles": make_host_function} #

PROFILE_FUNC_MAP.update(PROFILE_LINE_FUNC_MAP)
PROFILE_FUNC_MAP.update(PROFILE_CONTINUUM_FUNC_MAP)
