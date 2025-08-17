"""This module handles basic functions."""
__author__ = 'felavila'

__all__ = [
    "kms_to_wl",
    "vac_to_air",
    "wl_to_kms",
]


from typing import Callable, Dict, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from sheap.Utils.Constants  import c

def kms_to_wl(kms, line_center, c=c):
    """
    Convert a velocity in km/s to a wavelength shift based on the line center.

    Parameters:
    -----------
    kms : float or array-like
        The velocity value(s) in kilometers per second.
    line_center : float
        The central (reference) wavelength of the spectral line.
    c : float, optional
        The speed of light in km/s. The default value is 2.99792458e5 km/s.

    Returns:
    --------
    wl : float or array-like
        The calculated wavelength shift corresponding to the input velocity.
    """
    wl = kms * line_center / c
    return wl


def wl_to_kms(wl, line_center, c=c):
    """
    Convert a velocity in km/s to a wavelength shift based on the line center.

    Parameters:
    -----------
    wl : float or array-like
        The calculated wavelength shift corresponding to the input velocity.

    line_center : float
        The central (reference) wavelength of the spectral line.
    c : float, optional
        The speed of light in km/s. The default value is 2.99792458e5 km/s.

    Returns:
    --------
    kms : float or array-like
        The velocity value(s) in kilometers per second.
    """
    kms = (wl * c) / line_center
    return kms


def vac_to_air(lam_vac):
    """
    Convert vacuum to air wavelengths

    :param lam_vac - Wavelength in Angstroms
    :return: lam_air - Wavelength in Angstroms

    """
    lam = np.asarray(lam_vac)
    sigma2 = (1e4 / lam) ** 2
    fact = 1 + 5.792105e-2 / (238.0185 - sigma2) + 1.67917e-3 / (57.362 - sigma2)

    return lam_vac / fact
