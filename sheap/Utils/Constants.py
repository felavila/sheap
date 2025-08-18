"""This module contains constant and stuff."""
from __future__ import annotations
__author__ = 'felavila'


from pathlib import Path
import numpy as np 
from functools import lru_cache
import yaml  



__all__ = [
    "BOL_CORRECTIONS",
    "DEFAULT_LIMITS",
    "SINGLE_EPOCH_ESTIMATORS",
    "c",
    "cm_per_mpc",
]



#BolometricCorrections
c = 299792.458 #speed of light in km/s
cm_per_mpc = 3.08568e24 #mpc to cm
_DEFAULT_LIMITS =  Path(__file__).resolve().parent.parent / "SuportData"/ "DefaultLimits" / "DefaultLimits.yaml"
_SINGLE_EPOCH_ESTIMATORS = Path(__file__).resolve().parent.parent / "SuportData"/ "SingleEpochEstimators" / "SingleEpochEstimators.yaml"
_BOL_CORRECTIONS = Path(__file__).resolve().parent.parent / "SuportData"/ "BolometricCorrections" / "BolometricCorrections.yaml"


def _assert_exists(p: Path) -> None:
    if not p.is_file():
        raise FileNotFoundError(f"YAML not found: {p}")

@lru_cache(maxsize=None)
def read_yaml(p: Path) -> dict:
    """Load a YAML file into a Python dict (cached)."""
    _assert_exists(p)
    with p.open("r") as f:
        return yaml.safe_load(f)  # returns dict/list/str/etc.

# Usage
DEFAULT_LIMITS           = read_yaml(_DEFAULT_LIMITS)
SINGLE_EPOCH_ESTIMATORS    = read_yaml(_SINGLE_EPOCH_ESTIMATORS)
BOL_CORRECTIONS  = read_yaml(_BOL_CORRECTIONS)

# Common bolometric corrections (k_bol ≡ L_bol / λLλ)
# Baseline constants below are widely used “Richards+06” values, as adopted in large SDSS catalogs (e.g., Shen+11).
# Notes:
# - 1350/3000/5100 Å factors (3.81/5.15/9.26) come from the mean quasar SED in Richards et al. (2006, ApJS 166, 470),
#   and are used directly by Shen et al. (2011, ApJS 194, 45). Many later works keep these same constants.
# - 1450 Å is often taken to be the same as 1350 Å (k≈3.81) by assumption (very small slope difference).
#   If you prefer an explicitly fitted 1450 Å correction, Runnoe et al. (2012) recommend ≈4.2 instead of 3.81.
# - 6200 Å is not standard in Richards+06; we mirror 5100 Å (k≈9.26) as a pragmatic choice in the optical continuum.
# - Caveat: Netzer (2019, MNRAS 488, 5185) argues for luminosity-dependent k_bol; keep that in mind if you need precision.
# BOL_CORRECTIONS = {
#     "1350": 3.81,  # Richards+06; adopted in Shen+11 and many later catalogs.
#     "1450": 3.81,  # Commonly set equal to 1350 Å. (Alt: Runnoe+12 suggest ~4.2 for 1450 Å.)
#     "3000": 5.15,  # Richards+06; adopted in Shen+11.
#     "5100": 9.26,  # Richards+06; adopted in Shen+11; still widely used.
#     "6200": 9.26,  # Practical proxy: assume same as 5100 Å in the optical.
# }


#experimental
# DEFAULT_LIMITS = {
#     'broad': {
#         'upper_fwhm': 10000.0, 'lower_fwhm': 1500.0, 'v_shift': 3000.0,
#         'max_amplitude': 10.0, 'canonical_wavelengths': 4861.0,
#         'references': ['2011ApJS..194...45S', '2017ApJS..229...39R']  # Shen+11; Rakshit+17
#     },
#     'narrow': {
#         'upper_fwhm': 500.0, 'lower_fwhm': 100.0, 'v_shift': 500.0,
#         'max_amplitude': 10.0, 'canonical_wavelengths': 5007.0,          # anchor at [O III]
#         'references': ['2013MNRAS.433..622M', '2017MNRAS.472.4051C']     # Mullaney+13; Calderone+17 (QSFit)
#     },
#     'outflow': {
#         'upper_fwhm': 1500.0, 'lower_fwhm': 500.0, 'v_shift': 500.0,
#         'max_amplitude': 10.0, 'canonical_wavelengths': 5007.0,
#         'references': ['2013MNRAS.433..622M', '2018A&A...620A..82C', '2018NatAs...2..198H']
#     },
#     'winds': {
#         'upper_fwhm': 15000.0, 'lower_fwhm': 3000.0, 'v_shift': 8000.0,
#         'max_amplitude': 10.0, 'canonical_wavelengths': 1549.0,          # anchor at C IV
#         'references': ['2011AJ....141..167R', '2016MNRAS.461..647C', '2017MNRAS.465.2120C']
#     },

#     # --- Fe II split ---
#     'fe_broad': {
#         'upper_fwhm': 7000.0, 'lower_fwhm': 900.0, 'v_shift': 3000.0,
#         'max_amplitude': 0.07, 'canonical_wavelengths': 4570.0,
#         'references': ['2004A&A...417..515V', '2001ApJS..134....1V', '2010ApJS..189...15K']
#     },
#     'fe_narrow': {
#         'upper_fwhm': 600.0, 'lower_fwhm': 100.0, 'v_shift': 500.0,
#         'max_amplitude': 0.07, 'canonical_wavelengths': 4570.0,
#         'references': ['2004A&A...417..515V', '2010ApJ...721L.143D', '2008ApJ...674..668W']
#     },

#     'host': {
#         'upper_fwhm': 1000.0, 'lower_fwhm': 100.0, 'v_shift': 500,    # keep your current centroid cap
#         'max_amplitude': 0.0, 'canonical_wavelengths': 5175.0,
#         'references': ['2003MNRAS.344.1000B', '2006MNRAS.371..703S', '2004PASP..116..138C']
#     },
#     'bal': {
#         'upper_fwhm': 20000.0, 'lower_fwhm': 2000.0, 'v_shift': 30000.0,
#         'max_amplitude': 10.0, 'canonical_wavelengths': 1549.0,
#         'references': ['1991ApJ...373...23W', '2006ApJS..165....1T']
#     },

#     # Optional: backward-compat alias if other code expects 'fe'
#     'fe': {
#         'upper_fwhm': 7000.0, 'lower_fwhm': 100.0, 'v_shift': 3000.0,
#         'max_amplitude': 0.07, 'canonical_wavelengths': 4570.0,
#         'references': ['2004A&A...417..515V', '2001ApJS..134....1V', '2010ApJS..189...15K']
#     }
# }


#host +-50.0 in \AA, FWHM 10**3.8 FWHM 10**2.0 broadening weights [0,1]
#Fe template +- 50.0 in \AA , FWHM 10**3.8496 FWHM 10**2.0 broadening
#shift - > lambda0 kms_to_wl(limits.center_shift, lambda0)
# kms_to_wl(limits.center_shift, center0)



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

# SINGLE_EPOCH_ESTIMATORS = {
#     # --------------------------
#     # 2005 — Greene & Ho (Hα, line luminosity)
#     # --------------------------
#     "GH05_Halpha_Lha": {
#         "line": "Halpha", "kind": "line",
#         "a": 6.57, "b": 0.47, "fwhm_factor": 2.06, "f": 1.0,
#         "pivots": {"L": 1e42, "FWHM": 1e3}, "extras": {},
#         "ref": "2005ApJ...630..122G", "width_def": "fwhm",
#     },

#     # --------------------------
#     # 2006 — Vestergaard & Peterson (continuum recipes)
#     # --------------------------
#     "VP06_Hbeta_5100": {
#         "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
#         "a": 6.91, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2006ApJ...641..689V", "width_def": "fwhm",
#     },
#     "VP06_CIV_1350": {
#         "line": "CIV", "kind": "continuum", "wavelength": 1350,
#         "a": 6.66, "b": 0.53, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2006ApJ...641..689V", "width_def": "fwhm",
#     },

#     # --------------------------
#     # 2009 — Vestergaard & Osmer (Mg II, continuum)
#     # --------------------------
#     "VO09_MgII_1350": {
#         "line": "MgII", "kind": "continuum", "wavelength": 1350,
#         "a": 6.72, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2009ApJ...699..800V", "width_def": "fwhm",
#     },
#     "VO09_MgII_2100": {
#         "line": "MgII", "kind": "continuum", "wavelength": 2100,
#         "a": 6.79, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2009ApJ...699..800V", "width_def": "fwhm",
#     },
#     "VO09_MgII_3000": {
#         "line": "MgII", "kind": "continuum", "wavelength": 3000,
#         "a": 6.86, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2009ApJ...699..800V", "width_def": "fwhm",
#     },
#     "VO09_MgII_5100": {
#         "line": "MgII", "kind": "continuum", "wavelength": 5100,
#         "a": 6.96, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2009ApJ...699..800V", "width_def": "fwhm",
#     },

#     # --------------------------
#     # 2011 — Shen et al. (continuum on VP06 scale)
#     # --------------------------
#     # "Shen11_Hbeta_5100": {
#     #     "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
#     #     "a": 6.91, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#     #     "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#     #     "ref": "2011ApJS..194...45S", "width_def": "fwhm",
#     # },
#     # "Shen11_CIV_1350": {
#     #     "line": "CIV", "kind": "continuum", "wavelength": 1350,
#     #     "a": 6.66, "b": 0.53, "vel_exp": 2.0, "f": 1.0,
#     #     "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#     #     "ref": "2011ApJS..194...45S", "width_def": "fwhm",
#     # },
#     "Shen11_MgII_3000": {
#         "line": "MgII", "kind": "continuum", "wavelength": 3000,
#         "a": 6.74, "b": 0.62, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2011ApJS..194...45S", "width_def": "fwhm",
#     }, #new calibrations 

#     # --------------------------
#     # 2015 — Reines & Volonteri (Hα, line luminosity; ε=f/4 with f=4.3)
#     # --------------------------
#     "RV15_Halpha_Lha": {
#         "line": "Halpha", "kind": "line",
#         "a": 6.6014,  # 6.57 + log10(1.075) with f=4.3 ⇒ ε=f/4=1.075
#         "b": 0.47, "fwhm_factor": 2.06, "f": 1.0,
#         "pivots": {"L": 1e42, "FWHM": 1e3}, "extras": {},
#         "ref": "2015ApJ...813...82R", "width_def": "fwhm",
#     },

#     # --------------------------
#     # 2016 — Mejía-Restrepo et al. (Table 7; β=2, f=1; L in 1e44, FWHM in 1e3)
#     # Variants: "local", "global", "local_corr". All are FWHM-based.
#     # --------------------------
#     # Hα with L5100
#     "MR16_local_Halpha_L5100_FWHM": {
#         "line": "Halpha", "kind": "continuum", "wavelength": 5100,
#         "a": 6.779, "b": 0.650, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_global_Halpha_L5100_FWHM": {
#         "line": "Halpha", "kind": "continuum", "wavelength": 5100,
#         "a": 6.958, "b": 0.569, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_localcorr_Halpha_L5100_FWHM": {
#         "line": "Halpha", "kind": "continuum", "wavelength": 5100,
#         "a": 6.845, "b": 0.650, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },

#     # Hα with L6200
#     "MR16_local_Halpha_L6200_FWHM": {
#         "line": "Halpha", "kind": "continuum", "wavelength": 6200,
#         "a": 6.842, "b": 0.634, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_global_Halpha_L6200_FWHM": {
#         "line": "Halpha", "kind": "continuum", "wavelength": 6200,
#         "a": 7.062, "b": 0.524, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_localcorr_Halpha_L6200_FWHM": {
#         "line": "Halpha", "kind": "continuum", "wavelength": 6200,
#         "a": 6.891, "b": 0.634, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },

#     # Hα with L(Hα) — line-luminosity calibration (note pivot L=1e44 here per table)
#     "MR16_local_Halpha_Lha_FWHM": {
#         "line": "Halpha", "kind": "line",
#         "a": 7.072, "b": 0.563, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_global_Halpha_Lha_FWHM": {
#         "line": "Halpha", "kind": "line",
#         "a": 7.373, "b": 0.514, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_localcorr_Halpha_Lha_FWHM": {
#         "line": "Halpha", "kind": "line",
#         "a": 7.389, "b": 0.563, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },

#     # Hβ with L5100
#     "MR16_local_Hbeta_L5100_FWHM": {
#         "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
#         "a": 6.721, "b": 0.650, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_global_Hbeta_L5100_FWHM": {
#         "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
#         "a": 6.864, "b": 0.568, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_localcorr_Hbeta_L5100_FWHM": {
#         "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
#         "a": 6.740, "b": 0.650, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },

#     # Mg II with L3000
#     "MR16_local_MgII_L3000_FWHM": {
#         "line": "MgII", "kind": "continuum", "wavelength": 3000,
#         "a": 6.906, "b": 0.609, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_global_MgII_L3000_FWHM": {
#         "line": "MgII", "kind": "continuum", "wavelength": 3000,
#         "a": 6.955, "b": 0.599, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_localcorr_MgII_L3000_FWHM": {
#         "line": "MgII", "kind": "continuum", "wavelength": 3000,
#         "a": 6.925, "b": 0.609, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },

#     # C IV with L1450
#     "MR16_local_CIV_L1450_FWHM": {
#         "line": "CIV", "kind": "continuum", "wavelength": 1450,
#         "a": 6.331, "b": 0.599, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_global_CIV_L1450_FWHM": {
#         "line": "CIV", "kind": "continuum", "wavelength": 1450,
#         "a": 6.349, "b": 0.588, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "global"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },
#     "MR16_localcorr_CIV_L1450_FWHM": {
#         "line": "CIV", "kind": "continuum", "wavelength": 1450,
#         "a": 6.353, "b": 0.599, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {"variant": "local_corr"},
#         "ref": "2016MNRAS.460..187M", "width_def": "fwhm",
#     },

#     # --------------------------
#     # 2018 — Mejía-Restrepo et al. (C IV caution)
#     # --------------------------
#     "MR18_CIV_1350_FWHM": {
#         "line": "CIV", "kind": "continuum", "wavelength": 1350,
#         "a": 6.66, "b": 0.53, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2018MNRAS.478.1929M", "width_def": "fwhm",
#         "enabled": False,
#     },

#     # --------------------------
#     # 2020 — Le et al. (Mg II with profile-shape term)
#     # --------------------------
#     "Le20_MgII_3000_FWHM": {
#         "line": "MgII", "kind": "continuum", "wavelength": 3000,
#         "a": 6.86, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3},
#         "extras": {"le20_shape": True},  # requires extras['sigma_kms']
#         "ref": "2020ApJ...901...35L", "width_def": "fwhm",
#     },

#     # --------------------------
#     # 2023 — Yu et al. (continuum)
#     # --------------------------
#     "Yu23_Hbeta_5100": {
#         "line": "Hbeta", "kind": "continuum", "wavelength": 5100,
#         "a": 6.91, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2023MNRAS.522.4132Y", "width_def": "fwhm",
#     },
#     "Yu23_MgII_3000": {
#         "line": "MgII", "kind": "continuum", "wavelength": 3000,
#         "a": 6.86, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2023MNRAS.522.4132Y", "width_def": "fwhm",
#     },
#     "Yu23_CIV_1350": {
#         "line": "CIV", "kind": "continuum", "wavelength": 1350,
#         "a": 6.66, "b": 0.53, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3}, "extras": {},
#         "ref": "2023MNRAS.522.4132Y", "width_def": "fwhm",
#     },

#     # --------------------------
#     # 2025 — Pan et al. (Mg II with iron-strength term)
#     # --------------------------
#     "Pan25_MgII_3000_RFe": {
#         "line": "MgII", "kind": "continuum", "wavelength": 3000,
#         "a": 6.86, "b": 0.50, "vel_exp": 2.0, "f": 1.0,
#         "pivots": {"L": 1e44, "FWHM": 1e3},
#         "extras": {"pan25_gamma": -0.34},  # requires extras['R_Fe']
#         "ref": "2025ApJ...987...48P", "width_def": "fwhm",
#     },
# }

###
# Halpha 5600,7300
# Hbeta 4400, 5600
# MII 2500, 3000
# cIII 1600, 2000
# CIV 1100, 2000
# La 1000,1500
####
#Set of functions to handle different paths and cross match. 