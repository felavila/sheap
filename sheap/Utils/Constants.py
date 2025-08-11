"""This module contains constant and stuff."""
__version__ = '0.1.0'
__author__ = 'Felipe Avila-Vera'


import numpy as np 

# Auto-generated __all__
__all__ = [
    "BOL_CORRECTIONS",
   # "CANONICAL_WAVELENGTHS",
    "DEFAULT_LIMITS",
    "SINGLE_EPOCH_ESTIMATORS",
    "c",
    "cm_per_mpc",
]




c = 299792.458 #speed of light in km/s
cm_per_mpc = 3.08568e24 #mpc to cm


# CANONICAL_WAVELENGTHS = {
#     'broad': 4861.0,    # Hbeta
#     'narrow': 4861.0,   # [OIII]
#     'outflow': 5007.0,  # [OIII]
#     'fe': 4570.0,       # Mean FeII blend
#     'nlr': 6583.0,       # [NII]
#     "winds": 5007.0,
#      'bal': 1549.0  } # CIV doublet average



DEFAULT_LIMITS = {
    'broad':   {'upper_fwhm': 10000.0,  'lower_fwhm': 1000.875, 'center_shift': 5000.0,  'v_shift': 5000.0,  'max_amplitude': 10.0,"canonical_wavelengths":4861.0 },
    'narrow':  {'upper_fwhm': 1000.0,   'lower_fwhm': 100.0,     'center_shift': 2500.0,  'v_shift': 2500.0,  'max_amplitude': 10.0,"canonical_wavelengths":4861.0 },
    'outflow': {'upper_fwhm': 10000.0,  'lower_fwhm': 1000.875,  'center_shift': 3000.0,  'v_shift': 3000.0,  'max_amplitude': 10.0,"canonical_wavelengths":5007.0},
    'fe':      {'upper_fwhm': 7065.0,   'lower_fwhm': 117.75,    'center_shift': 4570.0,  'v_shift': 4570.0,  'max_amplitude': 0.07,"canonical_wavelengths":4570.0},
    'nlr':     {'upper_fwhm': 2355.0,   'lower_fwhm': 117.75,    'center_shift': 1500.0,  'v_shift': 1500.0,  'max_amplitude': 10.0,"canonical_wavelengths":6583.0},
    'winds':   {'upper_fwhm': 15000.0,  'lower_fwhm': 5000.0,    'center_shift': 8000.0,  'v_shift': 8000.0,  'max_amplitude': 10.0,"canonical_wavelengths":5007.0},
    'host':    {'upper_fwhm': 0.0,      'lower_fwhm': 0.0,       'center_shift': 0.0,     'v_shift': 0.0,     'max_amplitude': 0.0,"canonical_wavelengths": 0.0},
    'bal': {'upper_fwhm': 20000.0,'lower_fwhm': 2000.0,'center_shift': 30000.0,'v_shift': 30000.0,'max_amplitude': 10.0,"canonical_wavelengths":1549.0  } # mmmm
}



#Common correction factors (e.g., Richards et al. 2006 or Netzer 2019) TODO:look for this references
#This can be change for the users the "only" condition is this should be a dictionary with keys with wavelenght and the values are floats.
BOL_CORRECTIONS = {
    "1350": 3.81,
    "1450": 3.81,
    "3000": 5.15,
    "5100": 9.26,
    "6200": 9.26,
}

# Standard single-epoch virial estimators for common broad lines
# Reference: see Vestergaard & Peterson 2006; Shen et al. 2011; Greene & Ho 2005 TODO:look for this references
#This can be change for the users the "only" condition is the user use the same line_name convenction as sheap
#and the requrided paramter a,b,f,wavelength and also the bol correction should be also inside the wavelength 
#https://arxiv.org/pdf/1603.03437
#maybe the single epoch estimators can be comined in a table with all the values and the corresponding referee, and the name of their corresponding eq.
SINGLE_EPOCH_ESTIMATORS = {
    "Hbeta_w": {
        "wavelength": "5100",
        "a": 6.864,
        "b": 0.568,
        "f": 1.0,
    },
    "MgII_w": {
        "wavelength": "3000",
        "a": 6.86,
        "b": 0.5,
        "f": 1.0,
    },
    "CIV_w": {
        "wavelength": "1350",
        "a": 6.66,
        "b": 0.53,
        "f": 1.0,
    },
    "Halpha_w": {
        "wavelength": "5100",
        "a": 6.958,
        "b":  0.569,
        "f": 1.0,}
    ,
    "Halpha_l": {"a": 6.57 + np.log10(1.075),  # Greene+Ho 2015
                "b": 0.47,
                "fwhm_factor": 2.06,}
}


# "Halpha": {
#         "a": 6.57 + np.log10(1.075),  # Greene+Ho 2015
#         "b": 0.47,
#         "fwhm_factor": 2.06,
#         "luminosity_key": "luminosity"
#     }
