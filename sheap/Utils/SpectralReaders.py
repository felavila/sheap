"""
FITS Spectrum Readers
=====================

This module provides utilities to read spectra from different survey
and simulation formats (SDSS, DESI, PyQSO, and custom simulations).
It also includes parallel and batched readers to handle multiple files
efficiently, with fallbacks for sequential reading.

Readers
-------
- fits_reader_sdss:       SDSS spectra (PLATE-MJD-FIBERID format)
- fits_reader_desi:       DESI spectra
- fits_reader_pyqso:      PyQSO pipeline spectra
- fits_reader_simulation: Simulated spectra

Batching / Parallel utilities
-----------------------------
- parallel_reader
- batched_reader
- sequential_reader
"""

__author__ = 'felavila'

__all__ = [
    "READER_FUNCTIONS",
    "batched_reader",
    "fits_reader_desi",
    "fits_reader_pyqso",
    "fits_reader_sdss",
    "fits_reader_simulation",
    "n_cpu",
    "parallel_reader",
    "sequential_reader",
]

import os
import numpy as np
from multiprocessing import Pool, set_start_method
from astropy.io import fits
from functools import partial
from concurrent.futures import ThreadPoolExecutor

#from sheap.Utils.SpectralSetup import resize_and_fill_with_nans

# Limit CPUs for safety
n_cpu = min(4, os.cpu_count())


def fits_reader_desi(file: str):
    """
    Read a DESI FITS spectrum.

    Parameters
    ----------
    file : str
        Path to DESI FITS file.

    Returns
    -------
    data_array : np.ndarray
        Array with shape (3, n_pix): [wavelength, flux, error].
    header_array : np.ndarray
        Array with RA and DEC from header.
    """
    hdul = fits.open(file)
    flux_scale = float(hdul[1].header["TUNIT2"].split(" ")[0])
    ivar_scale = float(hdul[1].header["TUNIT3"].split(" ")[0])
    data = hdul[1].data
    data_array = np.array([
        data["WAVELENGTH"],
        data["FLUX"] * flux_scale,
        1 / np.sqrt(data["IVAR"] * ivar_scale)
    ])
    data_array[np.isinf(data_array)] = 1e20
    header_array = np.array([hdul[0].header["RA"], hdul[0].header["DEC"]])
    return data_array, header_array

def fits_reader_4most(file: str):
    """
    Read a 4most FITS spectrum SPV.

    Parameters
    ----------
    file : str
        Path to 4most FITS file.

    Returns
    -------
    data_array : np.ndarray
        Array with shape (3, n_pix): [wavelength, flux, error].
    header_array : np.ndarray
        Array with RA and DEC from header.
    """
    hdul = fits.open(file)
    data = hdul[1].data
    header_array = np.array([hdul[0].header["RA"], hdul[0].header["DEC"]])
    data_array = np.array([
        data["WAVE"],
        data["FLUX"],
        data["ERR"]])
    return data_array.squeeze(), header_array

def fits_reader_simulation(file: str, chanel: int = 1, template: bool = False):
    """
    Read a simulated spectrum from a FITS file.

    Parameters
    ----------
    file : str
        Path to simulation FITS file.
    chanel : int, default=1
        HDU extension index to read.
    template : bool, default=False
        If True, reads template arrays.

    Returns
    -------
    data_array : np.ndarray
        Array with shape (n_channels, n_pix).
    header_array : list
        Empty or metadata, depending on template.
    """
    hdul = fits.open(file)
    header_array = []
    if template:
        data_array = np.array([
            hdul[chanel].data['LAMBDA'],
            hdul[chanel].data["FLUX_DENSITY"]
        ])
        return data_array.squeeze(), header_array

    if chanel == 1:
        data_array = np.array([
            hdul[chanel].data["WAVE"],
            hdul[chanel].data["FLUX"],
            hdul[chanel].data["ERR_FLUX"],
        ])
    else:
        data_array = np.array([
            hdul[chanel].data["WAVE"],
            hdul[chanel].data["FLUX"],
            hdul[chanel].data["ERR"],
        ])
    return data_array.squeeze(), header_array


def fits_reader_sdss(file: str):
    """
    Read an SDSS FITS spectrum.

    Parameters
    ----------
    file : str
        Path to SDSS FITS file.

    Returns
    -------
    data_array : np.ndarray
        Array with shape (4, n_pix): [wavelength, flux, error, wdisp].
    header_array : np.ndarray
        Array with RA and DEC from header.
    """
    hdul = fits.open(file)
    flux_scale = float(hdul[0].header["BUNIT"].split(" ")[0])
   
    data = hdul[1].data
    data_array = np.array([
        10 ** data["loglam"],
        data["flux"] * flux_scale,
        flux_scale / np.sqrt(data["ivar"]),
        data["wdisp"]
    ])
    
    data_array[np.isinf(data_array)] = 1e20
    header_array = np.array([hdul[0].header["PLUG_RA"], hdul[0].header["PLUG_DEC"]])
    return data_array, header_array

def json_reader(file):
    #this only wrok for desijson
    import json
    import pandas as pd

    json_path = (file)

    with open(json_path, "r", encoding="utf-8") as file:
        raw_json = json.load(file)

    # print(type(raw_json))

    # if isinstance(raw_json, dict):
    #     for key, value in raw_json.items():
    #         length = len(value) if hasattr(value, "__len__") else None
    #         print(key, type(value).__name__, length)


def fits_reader_pyqso(file: str):
    """
    Read a PyQSO-format spectrum.

    Parameters
    ----------
    file : str
        Path to PyQSO FITS file.

    Returns
    -------
    spectra : np.ndarray
        Array with shape (3, n_pix): [wavelength, flux, error].
    header_array : list
        Empty list (no coords stored).
    """
    hdul = fits.open(file)
    spectra = np.array([
        hdul[3].data["wave_prereduced"],
        hdul[3].data["flux_prereduced"],
        hdul[3].data["err_prereduced"],
    ])
    return spectra, []


READER_FUNCTIONS = {
    "fits_reader_sdss": fits_reader_sdss,
    "fits_reader_simulation": fits_reader_simulation,
    "fits_reader_pyqso": fits_reader_pyqso,
    "fits_reader_desi": fits_reader_desi,
    "fits_reader_4most": fits_reader_4most,
}


def parallel_reader(paths, n_cpu=n_cpu, function=fits_reader_sdss, **kwargs):
    """
    Parallel reader using multiprocessing.

    Parameters
    ----------
    paths : list of str
        Paths to FITS files.
    n_cpu : int, optional
        Number of processes to use (default=min(4, os.cpu_count())).
    function : callable or str, optional
        Reader function or key in `READER_FUNCTIONS`.

    Returns
    -------
    coords : np.ndarray
        Coordinates from headers (RA, DEC).
    spectra_reshaped : list
        Placeholder for reshaped spectra (currently empty).
    spectra : list of np.ndarray
        Raw spectra arrays.
    """
    if isinstance(function, str):
        function = READER_FUNCTIONS[function]

    func_with_args = partial(function, **kwargs)

    with Pool(processes=min(n_cpu, len(paths))) as pool:
        results = pool.map(func_with_args, paths, chunksize=1)

    spectra = [result[0] for result in results]
    coords = np.array([result[1] for result in results])
    shapes_min= [s.shape[1] for s in spectra]
    #spectra_reshaped = []  # TODO: enable resize_and_fill_with_nans
    return coords, shapes_min, spectra


def parallel_reader(paths, n_cpu=None, function=fits_reader_sdss, **kwargs):
    """
    Parallel FITS reader using threads.

    This avoids multiprocessing fork problems with JAX/CUDA.
    """
    paths = list(paths)

    if len(paths) == 0:
        return np.empty((0, 2)), [], []

    if isinstance(function, str):
        function = READER_FUNCTIONS[function]

    if n_cpu is None:
        n_cpu = min(4, os.cpu_count() or 1)

    n_workers = min(n_cpu, len(paths))
    func_with_args = partial(function, **kwargs)

    if n_workers <= 1:
        results = [func_with_args(path) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(func_with_args, paths))

    spectra = [result[0] for result in results]
    coords = np.array([result[1] for result in results])
    shapes_min = [s.shape[1] for s in spectra]

    return coords, shapes_min, spectra

def sequential_reader(paths, function=fits_reader_sdss):
    """
    Sequential FITS reader (fallback for debugging).

    Parameters
    ----------
    paths : list of str
        Paths to FITS files.
    function : callable or str, optional
        Reader function or key in `READER_FUNCTIONS`.

    Returns
    -------
    coords : np.ndarray
        Coordinates from headers (RA, DEC).
    spectra_reshaped : np.ndarray
        Reshaped spectra array.
    spectra : list of np.ndarray
        Raw spectra arrays.
    """
    results = []
    for i in paths:
        try:
            results.append(function(i))
        except Exception as e:
            print(f"Failed to read {i}: {e}")
    spectra = [result[0] for result in results]
    coords = np.array([result[1] for result in results])
    shapes_max = max(s.shape[1] for s in spectra)
    spectra_reshaped = []  # TODO: enable resize_and_fill_with_nans
    return coords, spectra_reshaped, spectra


def rebin_one_spectrum(
    spectrum,
    n_pix,
    fill=np.nan,
    clean_invalid=True,
):
    """
    Rebin one spectrum to a fixed number of pixels.

    Parameters
    ----------
    spectrum : tuple or list
        Spectrum in the form ``(wl, flux, error)``.
    n_pix : int
        Number of pixels in the output spectrum.
    fill : float, optional
        Fill value used by ``spectres`` outside the valid range.
    clean_invalid : bool, optional
        If True, non-finite flux/error values are replaced by 0.

    Returns
    -------
    original : ndarray
        Original spectrum with shape ``(3, n_original)``.
        Rows are wavelength, flux, error.
    new : ndarray
        Rebinned spectrum with shape ``(4, n_pix)``.
        Rows are wavelength, flux, error, mask.
    conservation_ratio : float
        Percentage ratio between rebinned and original integrated flux.
    """
    from spectres import spectres
    #print(spectrum)
    try:
        wl, flux, error,_ = spectrum
    except:
        wl, flux, error = spectrum
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)

    order = np.argsort(wl)
    wl = wl[order]
    flux = flux[order]
    error = error[order]

    regrid = np.linspace(wl[0], wl[-1], int(n_pix))

    new_flux, new_error = spectres(
        regrid,
        wl,
        flux,
        error,
        fill=fill,
        verbose=False,
    )

    valid = np.isfinite(new_flux) & np.isfinite(new_error)

    if clean_invalid:
        new_flux = np.where(valid, new_flux, 0.0)
        new_error = np.where(valid, new_error, 0.0)

    original_flux = np.trapezoid(flux, wl)
    rebinned_flux = np.trapezoid(new_flux, regrid)

    conservation_ratio = (
        100.0 * rebinned_flux / original_flux
        if original_flux != 0
        else np.nan
    )

    original = np.stack([wl, flux, error], axis=0)

    new = np.stack(
        [
            regrid,
            new_flux,
            new_error,
            valid.astype(float),
        ],
        axis=0,
    )

    return original, new, conservation_ratio

def rebin_spectra_list(
    spectra_list,redshifts,
    coords,
    n_pix=None,
    file_names=None,
    npix_original=None,
):
    """
    Rebin a list of spectra to the same number of pixels.

    Parameters
    ----------
    spectra_list : list
        List of spectra. Each item must be ``(wl, flux, error)``.
    n_pix : int or None, optional
        Number of pixels for the output grid. If None, uses the minimum
        input spectrum length.
    file_names : list or None, optional
        Names used as dictionary keys.
    redshifts : list or None, optional
        Redshifts associated with each spectrum.
    coords : list or None, optional
        Coordinates associated with each spectrum.
    npix_original : list or None, optional
        Original number of pixels for each spectrum.

    Returns
    -------
    new_dic : dict
        Dictionary containing original and rebinned spectra.
    """

    if n_pix is None:
        n_pix = min(len(s[0]) for s in spectra_list)

    n_pix = int(n_pix)
    n_spectra = len(spectra_list)


    if npix_original is None:
        npix_original = [len(s[0]) for s in spectra_list]

    new_dic = {}

    for i, spectrum in enumerate(spectra_list):
        original, new, conservation_ratio = rebin_one_spectrum(
            spectrum=spectrum,
            n_pix=n_pix,
        )

        file_name = file_names[i]

        new_dic[file_name] = {
            "original": original,
            "new": new,
            "conservation_ratio": conservation_ratio,
            "coords": coords[i],
            "z": redshifts[i],
            "dr_name": file_name,
            "npix": npix_original[i],
        }

    return new_dic
