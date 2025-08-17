"""This module handles basic operations."""
__author__ = 'felavila'

__all__ = [
    "build_cube_from_fits_header_comments",
    "resample_to_log_lambda_npinterp",
]

from typing import Tuple

import numpy as np
import glob
from astropy.io import fits
from pathlib import Path
from scipy.fft import fft, ifft, fftfreq


from sheap.Utils.Constants import c

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





def build_cube_from_fits_header_comments(
    template_dir: str,
    output_file: str = "miles_cube_log.npz",
    log_age_bins: int = None,
    z_bins: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read MILES-like template FITS files, build a (n_Z, n_age, n_pix) cube
    on a log-lambda grid, **resample all templates to the max FWHM**,
    and save as constant-resolution .npz.
    """
    def parse_resolution(comment: str, wave0: float, wave1: float, dlam: float) -> np.ndarray:
        vals = comment.split(":",1)[1].split("(")[0].strip()
        if "-" in vals:
            lo, hi = map(float, vals.split("-"))
        else:
            lo = hi = float(vals)
        n_pix = int(round((wave1 - wave0)/dlam)) + 1
        return np.linspace(lo, hi, n_pix)

    files = sorted(glob.glob(f"{template_dir}/*.fits"))
    if not files:
        raise FileNotFoundError(f"No FITS in {template_dir}")

    fluxes, ages, zs = [], [], []
    wave0s, wave1s, dlam_list = [], [], []
    sampling_types, res_comments = [], []

    for fn in files:
        with fits.open(fn) as hdul:
            hdr = hdul[0].header
            data = hdul[0].data.astype(np.float32)
            cmt = hdr["COMMENT"]
            wave0 = float(cmt[48].split(":")[-1].split("(")[0])
            wave1 = float(cmt[49].split(":")[-1].split("(")[0])
            dlam = float(cmt[50].split(":")[-1].split("(")[0])
            res_c = next(c for c in cmt if "Spectral resolution" in c)
            sampling = cmt[51].replace("'", "").split()[-1].lower()
            age = float(cmt[40].split(":")[-1].replace("'", "").strip())
            z   = float(cmt[41].split(":")[-1].replace("'", "").strip())

            wave0s.append(wave0); wave1s.append(wave1)
            dlam_list.append(dlam); sampling_types.append(sampling)
            res_comments.append(res_c)
            fluxes.append(data); ages.append(age); zs.append(z)

    # consistency checks
    if len({round(v,6) for v in wave0s})>1: raise ValueError("wave0 mismatch")
    if len({round(v,6) for v in wave1s})>1: raise ValueError("wave1 mismatch")
    if len({round(v,6) for v in dlam_list})>1: raise ValueError("dlam mismatch")
    if len(set(sampling_types))>1: raise ValueError("sampling mismatch")

    wave0, wave1, dlam = wave0s[0], wave1s[0], dlam_list[0]
    wave = np.arange(wave0, wave1 + dlam, dlam)
    wave_log = np.exp(np.linspace(np.log(wave[0]), np.log(wave[-1]), len(wave)))

    # parse resolution and pick target
    fwhm_arr = parse_resolution(res_comments[0], wave0, wave1, dlam)
    FWHM_target = float(np.max(fwhm_arr))
    sigma_target = FWHM_target / 2.355

    # compute per-pixel template sigma and pixscale
    c = 3e5
    pixscale = np.log(wave[1]/wave[0])
    sigma_A_arr = fwhm_arr / 2.355
    sigma_pix_template = (sigma_A_arr / wave) * c / pixscale
    sigma_pix_target = (sigma_target / wave) * c / pixscale
    # squared convolution sigma per pix (mean)
    sigma2_conv = np.maximum(sigma_pix_target**2 - sigma_pix_template**2, 0.0)
    sigma2_mean = np.mean(sigma2_conv)

    # FFT prep
    freqs = fftfreq(len(wave_log), d=pixscale)
    # single, constant kernel for mean sigma
    gauss_tf = np.exp(-2 * (np.pi * freqs)**2 * sigma2_mean)

    # build cube, convolving each template to target
    fluxes = np.array(fluxes); ages = np.array(ages); zs = np.array(zs)
    uniq_ages, uniq_zs = np.unique(ages), np.unique(zs)
    ages_sub = (np.unique([uniq_ages[np.argmin(abs(uniq_ages-a))] 
                 for a in np.logspace(np.log10(uniq_ages.min()), np.log10(uniq_ages.max()), log_age_bins)])
                if log_age_bins else uniq_ages)
    zs_sub = (np.unique([uniq_zs[np.argmin(abs(uniq_zs-z))]
               for z in np.linspace(uniq_zs.min(), uniq_zs.max(), z_bins)])
              if z_bins else uniq_zs)

    cube_log = np.empty((len(zs_sub), len(ages_sub), len(wave_log)), dtype=np.float32)
    for i,Z in enumerate(zs_sub):
        for j,age in enumerate(ages_sub):
            idx = np.where((zs==Z)&(ages==age))[0]
            if idx.size!=1:
                raise ValueError(f"Z={Z},age={age} missing")
            spec = np.interp(wave_log, wave, fluxes[idx[0]])
            # convolve template to target resolution
            sp_fft = fft(spec)
            spec_conv = np.real(ifft(sp_fft * gauss_tf))
            cube_log[i,j] = spec_conv

    # save constant-resolution metadata
    sigma_A = FWHM_target/2.355
    sigmatemplate = (sigma_A/5500.0)*c
    fixed_dispersion = (dlam/5500.0)*c

    np.savez_compressed(
        output_file,
        cube_log=cube_log,
        wave_log=wave_log,
        ages_sub=ages_sub,
        zs_sub=zs_sub,
        fwhm_const=FWHM_target,
        sigmatemplate=sigmatemplate,
        fixed_dispersion=fixed_dispersion,
        dlam=dlam, wave0=wave0, wave1=wave1
    )

    return cube_log, zs_sub, ages_sub, wave_log


