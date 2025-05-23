

#Common correction factors (e.g., Richards et al. 2006 or Netzer 2019) TODO:look for this references
BOL_CORRECTIONS = {
    "1350": 3.81,
    "1450": 3.81,
    "3000": 5.15,
    "5100": 9.26,
    "6200": 9.26,
}

# Standard single-epoch virial estimators for common broad lines
# Reference: see Vestergaard & Peterson 2006; Shen et al. 2011; Greene & Ho 2005
SINGLE_EPOCH_ESTIMATORS = {
    "Hbeta": {
        "wavelength": "5100",
        "a": 6.91,
        "b": 0.5,
        "f": 1.0,
    },
    "MgII": {
        "wavelength": "3000",
        "a": 6.86,
        "b": 0.5,
        "f": 1.0,
    },
    "CIV": {
        "wavelength": "1350",
        "a": 6.66,
        "b": 0.53,
        "f": 1.0,
    },
    "Halpha": {
        "wavelength": "6200",
        "a": 6.98,
        "b": 0.5,
        "f": 1.0,
    },
}