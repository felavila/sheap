import numpy as np
from typing import Tuple

c = 299792.458  # speed of light in km/s

def resample_to_log_lambda_npinterp(
    wave: np.ndarray,
    flux: np.ndarray,
    wdisp_kms: float,
    npix: int = None
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    Resample a linearly spaced spectrum to a log(λ) grid using numpy.interp.

    Parameters
    ----------
    wave : ndarray
        Original wavelength array in Ångstroms (linearly spaced).
    flux : ndarray
        Flux array (same length as wave).
    wdisp_kms : float
        Instrumental dispersion in km/s (σ).
    npix : int, optional
        Number of output pixels. Defaults to len(wave).

    Returns
    -------
    wave_log : ndarray
        Logarithmically spaced wavelength array.
    flux_log : ndarray
        Resampled flux on log(λ) grid.
    velscale : float
        Velocity scale in km/s per pixel.
    fwhm_lambda : ndarray
        FWHM in Ångstroms at each pixel.
    """
    npix = npix or len(wave)

    # Define log(λ) grid
    loglam = np.log(wave)
    loglam_new = np.linspace(loglam[0], loglam[-1], npix)
    wave_log = np.exp(loglam_new)

    # Use np.interp (no extrapolation: clip wave_log to original domain)
    wave_min, wave_max = wave[0], wave[-1]
    wave_log_clipped = np.clip(wave_log, wave_min, wave_max)
    flux_log = np.interp(wave_log_clipped, wave, flux)

    # Constant velscale in km/s
    velscale = np.log(wave_log[1] / wave_log[0]) * c

    # Compute Δλ per pixel
    dlam = np.gradient(wave_log)
    fwhm_lambda = 2.355 * (wdisp_kms / c) * wave_log

    return wave_log, flux_log, velscale, fwhm_lambda,dlam
