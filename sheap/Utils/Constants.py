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

DEFAULT_LIMITS_paper = {
    'broad':   {'upper_fwhm': 10000.0,  'lower_fwhm': 1500.0, 'center_shift': 5000.0,  'v_shift': 3000.0,  'max_amplitude': 10.0,"canonical_wavelengths":4861.0 },
    'narrow':  {'upper_fwhm': 500.0,   'lower_fwhm': 100.0,     'center_shift': 2500.0,  'v_shift': 500.0,  'max_amplitude': 10.0,"canonical_wavelengths":4861.0 },
    'outflow': {'upper_fwhm': 1500,  'lower_fwhm': 500.0,  'center_shift': 3000.0,  'v_shift': 500.0,  'max_amplitude': 10.0,"canonical_wavelengths":5007.0},
    'winds':   {'upper_fwhm': 15000.0,  'lower_fwhm': 3000.0,    'center_shift': 8000.0,  'v_shift': 8000.0,  'max_amplitude': 10.0,"canonical_wavelengths":5007.0},
    'fe':      {'upper_fwhm': 7065.0,   'lower_fwhm': 100.0,    'center_shift': 3278.0,  'v_shift': 3278.0,  'max_amplitude': 0.07,"canonical_wavelengths":4570.0},
    'host':    {'upper_fwhm': 7065.0,      'lower_fwhm': 100.0,       'center_shift': 0.0,     'v_shift':2895.7,     'max_amplitude': 0.0,"canonical_wavelengths": 5175.0},
    'bal': {'upper_fwhm': 20000.0,'lower_fwhm': 2000.0,'center_shift': 30000.0,'v_shift': 30000.0,'max_amplitude': 10.0,"canonical_wavelengths":1549.0  } # mmmm
}


#host +-50.0 in \AA, FWHM 10**3.8 FWHM 10**2.0 broadening weights [0,1]
#Fe template +- 50.0 in \AA , FWHM 10**3.8496 FWHM 10**2.0 broadening
#shift - > lambda0 kms_to_wl(limits.center_shift, lambda0)
# kms_to_wl(limits.center_shift, center0)

# Common bolometric corrections (k_bol ≡ L_bol / λLλ)
# Baseline constants below are widely used “Richards+06” values, as adopted in large SDSS catalogs (e.g., Shen+11).
# Notes:
# - 1350/3000/5100 Å factors (3.81/5.15/9.26) come from the mean quasar SED in Richards et al. (2006, ApJS 166, 470),
#   and are used directly by Shen et al. (2011, ApJS 194, 45). Many later works keep these same constants.
# - 1450 Å is often taken to be the same as 1350 Å (k≈3.81) by assumption (very small slope difference).
#   If you prefer an explicitly fitted 1450 Å correction, Runnoe et al. (2012) recommend ≈4.2 instead of 3.81.
# - 6200 Å is not standard in Richards+06; we mirror 5100 Å (k≈9.26) as a pragmatic choice in the optical continuum.
# - Caveat: Netzer (2019, MNRAS 488, 5185) argues for luminosity-dependent k_bol; keep that in mind if you need precision.
BOL_CORRECTIONS = {
    "1350": 3.81,  # Richards+06; adopted in Shen+11 and many later catalogs.
    "1450": 3.81,  # Commonly set equal to 1350 Å. (Alt: Runnoe+12 suggest ~4.2 for 1450 Å.)
    "3000": 5.15,  # Richards+06; adopted in Shen+11.
    "5100": 9.26,  # Richards+06; adopted in Shen+11; still widely used.
    "6200": 9.26,  # Practical proxy: assume same as 5100 Å in the optical.
}


# # Standard single-epoch virial estimators for common broad lines
# # Reference: see Vestergaard & Peterson 2006; Shen et al. 2011; Greene & Ho 2005 TODO:look for this references
# #This can be change for the users the "only" condition is the user use the same line_name convenction as sheap
# #and the requrided paramter a,b,f,wavelength and also the bol correction should be also inside the wavelength 
# #https://arxiv.org/pdf/1603.03437
# #maybe the single epoch estimators can be comined in a table with all the values and the corresponding referee, and the name of their corresponding eq.
# SINGLE_EPOCH_ESTIMATORS = {
#     "Hbeta_w": {
#         "wavelength": "5100",
#         "a": 6.864,
#         "b": 0.568,
#         "f": 1.0,
#     },
#     "MgII_w": {
#         "wavelength": "3000",
#         "a": 6.86,
#         "b": 0.5,
#         "f": 1.0,
#     },
#     "CIV_w": {
#         "wavelength": "1350",
#         "a": 6.66,
#         "b": 0.53,
#         "f": 1.0,
#     },
#     "Halpha_w": {
#         "wavelength": "5100",
#         "a": 6.958,
#         "b":  0.569,
#         "f": 1.0,}
#     ,
#     "Halpha_l": {"a": 6.57 + np.log10(1.075),  # Greene+Ho 2015
#                 "b": 0.47,
#                 "fwhm_factor": 2.06,}
# }


# # "Halpha": {
# #         "a": 6.57 + np.log10(1.075),  # Greene+Ho 2015
# #         "b": 0.47,
# #         "fwhm_factor": 2.06,
# #         "luminosity_key": "luminosity"
# #     }

# ==============================
# SINGLE_EPOCH_ESTIMATORS (sorted by year of original paper)
# ==============================
# Required per entry:
#   - line:       target line (must match entries in broad_params["lines"])
#   - kind:       "continuum" (uses L_w[ wavelength ]) or "line" (uses line luminosity array)
#   - a, b:       SE coefficients
#   - vel_exp or fwhm_factor: velocity exponent β (defaults to 2.0 if omitted)
#   - f:          virial factor (keep 1.0 unless you want to inject a scale)
#   - pivots:     {"L": luminosity pivot (erg/s), "FWHM": velocity pivot (km/s)}
#   - wavelength: ONLY for kind="continuum" (Å; used to pick L_w and optional L_bol)
# Optional:
#   - width_def:  "fwhm" | "sigma"  (which velocity width you provide)
#   - extras:     flags/params for optional corrections (e.g., {"le20_shape": True}, {"pan25_gamma": -0.34})
#   - enabled:    bool (soft-disable an entry)
#   - note/variant: free text

SINGLE_EPOCH_ESTIMATORS = {
    # --------------------------
    # 2005 — Greene & Ho (Hα, line luminosity)
    # --------------------------
    "GH05_Halpha_Lha": {
        "line": "Halpha", "kind": "line",
        "a": 6.57, "b": 0.47, "fwhm_factor": 2.06, "f": 1.0,
        "pivots": {"L": 1e42, "FWHM": 1e3}, "extras": {},
        "ref": "2005ApJ...630..122G", "width_def": "fwhm",
    },

    # --------------------------
    # 2006 — Vestergaard & Peterson (continuum recipes)
    # --------------------------
    "VP06_Hbeta_5100": {
        "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
        "a": 6.91, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2006ApJ...641..689V", "width_def": "fwhm",
    },
    "VP06_CIV_1350": {
        "line": "CIV", "kind": "continuum", "wavelength": 1350,
        "a": 6.66, "b": 0.53, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2006ApJ...641..689V", "width_def": "fwhm",
    },

    # --------------------------
    # 2009 — Vestergaard & Osmer (Mg II, continuum)
    # --------------------------
    "VO09_MgII_1350": {
        "line": "MgII", "kind": "continuum", "wavelength": 1350,
        "a": 6.72, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2009ApJ...699..800V", "width_def": "fwhm",
    },
    "VO09_MgII_2100": {
        "line": "MgII", "kind": "continuum", "wavelength": 2100,
        "a": 6.79, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2009ApJ...699..800V", "width_def": "fwhm",
    },
    "VO09_MgII_3000": {
        "line": "MgII", "kind": "continuum", "wavelength": 3000,
        "a": 6.86, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2009ApJ...699..800V", "width_def": "fwhm",
    },
    "VO09_MgII_5100": {
        "line": "MgII", "kind": "continuum", "wavelength": 5100,
        "a": 6.96, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2009ApJ...699..800V", "width_def": "fwhm",
    },

    # --------------------------
    # 2011 — Shen et al. (continuum on VP06 scale)
    # --------------------------
    # "Shen11_Hbeta_5100": {
    #     "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
    #     "a": 6.91, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
    #     "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
    #     "ref": "2011ApJS..194...45S", "width_def": "fwhm",
    # },
    # "Shen11_CIV_1350": {
    #     "line": "CIV", "kind": "continuum", "wavelength": 1350,
    #     "a": 6.66, "b": 0.53, "vel_exp": 2.0, "f": 1.0,
    #     "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
    #     "ref": "2011ApJS..194...45S", "width_def": "fwhm",
    # },
    "Shen11_MgII_3000": {
        "line": "MgII", "kind": "continuum", "wavelength": 3000,
        "a": 6.74, "b": 0.62, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2011ApJS..194...45S", "width_def": "fwhm",
    }, #new calibrations 

    # --------------------------
    # 2015 — Reines & Volonteri (Hα, line luminosity; ε=f/4 with f=4.3)
    # --------------------------
    "RV15_Halpha_Lha": {
        "line": "Halpha", "kind": "line",
        "a": 6.6014,  # 6.57 + log10(1.075) with f=4.3 ⇒ ε=f/4=1.075
        "b": 0.47, "fwhm_factor": 2.06, "f": 1.0,
        "pivots": {"L": 1e42, "FWHM": 1e3}, "extras": {},
        "ref": "2015ApJ...813...82R", "width_def": "fwhm",
    },

    # --------------------------
    # 2016 — Mejía-Restrepo et al. (Table 7; β=2, f=1; L in 1e44, FWHM in 1e3)
    # Variants: "local", "global", "local_corr". All are FWHM-based.
    # --------------------------
    # Hα with L5100
    "MR16_local_Halpha_L5100_FWHM": {
        "line": "Halpha", "kind": "continuum", "wavelength": 5100,
        "a": 6.779, "b": 0.650, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_global_Halpha_L5100_FWHM": {
        "line": "Halpha", "kind": "continuum", "wavelength": 5100,
        "a": 6.958, "b": 0.569, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_localcorr_Halpha_L5100_FWHM": {
        "line": "Halpha", "kind": "continuum", "wavelength": 5100,
        "a": 6.845, "b": 0.650, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },

    # Hα with L6200
    "MR16_local_Halpha_L6200_FWHM": {
        "line": "Halpha", "kind": "continuum", "wavelength": 6200,
        "a": 6.842, "b": 0.634, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_global_Halpha_L6200_FWHM": {
        "line": "Halpha", "kind": "continuum", "wavelength": 6200,
        "a": 7.062, "b": 0.524, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_localcorr_Halpha_L6200_FWHM": {
        "line": "Halpha", "kind": "continuum", "wavelength": 6200,
        "a": 6.891, "b": 0.634, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },

    # Hα with L(Hα) — line-luminosity calibration (note pivot L=1e44 here per table)
    "MR16_local_Halpha_Lha_FWHM": {
        "line": "Halpha", "kind": "line",
        "a": 7.072, "b": 0.563, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_global_Halpha_Lha_FWHM": {
        "line": "Halpha", "kind": "line",
        "a": 7.373, "b": 0.514, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_localcorr_Halpha_Lha_FWHM": {
        "line": "Halpha", "kind": "line",
        "a": 7.389, "b": 0.563, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },

    # Hβ with L5100
    "MR16_local_Hbeta_L5100_FWHM": {
        "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
        "a": 6.721, "b": 0.650, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_global_Hbeta_L5100_FWHM": {
        "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
        "a": 6.864, "b": 0.568, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_localcorr_Hbeta_L5100_FWHM": {
        "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
        "a": 6.740, "b": 0.650, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },

    # Mg II with L3000
    "MR16_local_MgII_L3000_FWHM": {
        "line": "MgII", "kind": "continuum", "wavelength": 3000,
        "a": 6.906, "b": 0.609, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_global_MgII_L3000_FWHM": {
        "line": "MgII", "kind": "continuum", "wavelength": 3000,
        "a": 6.955, "b": 0.599, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_localcorr_MgII_L3000_FWHM": {
        "line": "MgII", "kind": "continuum", "wavelength": 3000,
        "a": 6.925, "b": 0.609, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },

    # C IV with L1450
    "MR16_local_CIV_L1450_FWHM": {
        "line": "CIV", "kind": "continuum", "wavelength": 1450,
        "a": 6.331, "b": 0.599, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_global_CIV_L1450_FWHM": {
        "line": "CIV", "kind": "continuum", "wavelength": 1450,
        "a": 6.349, "b": 0.588, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },
    "MR16_localcorr_CIV_L1450_FWHM": {
        "line": "CIV", "kind": "continuum", "wavelength": 1450,
        "a": 6.353, "b": 0.599, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
        "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
    },

    # --------------------------
    # 2018 — Mejía-Restrepo et al. (C IV caution)
    # --------------------------
    "MR18_CIV_1350_FWHM": {
        "line": "CIV", "kind": "continuum", "wavelength": 1350,
        "a": 6.66, "b": 0.53, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2018MNRAS.478.1929M", "width_def": "fwhm",
        "enabled": False,
    },

    # --------------------------
    # 2020 — Le et al. (Mg II with profile-shape term)
    # --------------------------
    "Le20_MgII_3000_FWHM": {
        "line": "MgII", "kind": "continuum", "wavelength": 3000,
        "a": 6.86, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3},
        "extras": {"le20_shape": True},  # requires extras['sigma_kms']
        "ref": "2020ApJ...901...35L", "width_def": "fwhm",
    },

    # --------------------------
    # 2023 — Yu et al. (continuum)
    # --------------------------
    "Yu23_Hbeta_5100": {
        "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
        "a": 6.91, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2023MNRAS.522.4132Y", "width_def": "fwhm",
    },
    "Yu23_MgII_3000": {
        "line": "MgII", "kind": "continuum", "wavelength": 3000,
        "a": 6.86, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2023MNRAS.522.4132Y", "width_def": "fwhm",
    },
    "Yu23_CIV_1350": {
        "line": "CIV", "kind": "continuum", "wavelength": 1350,
        "a": 6.66, "b": 0.53, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
        "ref": "2023MNRAS.522.4132Y", "width_def": "fwhm",
    },

    # --------------------------
    # 2025 — Pan et al. (Mg II with iron-strength term)
    # --------------------------
    "Pan25_MgII_3000_RFe": {
        "line": "MgII", "kind": "continuum", "wavelength": 3000,
        "a": 6.86, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
        "pivots": {"L": 1e44, "FWHM": 1e3},
        "extras": {"pan25_gamma": -0.34},  # requires extras['R_Fe']
        "ref": "2025ApJ...987...48P", "width_def": "fwhm",
    },
}
