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
    header_array = np.array([hdul[0].header["RA"], hdul[0].header["DEC"]])
    return data_array, header_array


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
    shapes_max = max(s.shape[1] for s in spectra)
    spectra_reshaped = []  # TODO: enable resize_and_fill_with_nans
    return coords, spectra_reshaped, spectra


def batched_reader(paths, batch_size=8, function=fits_reader_sdss):
    """
    Batch reader for safer memory usage.

    Parameters
    ----------
    paths : list of str
        Paths to FITS files.
    batch_size : int, optional
        Number of files to read per batch.
    function : callable or str, optional
        Reader function or key in `READER_FUNCTIONS`.

    Returns
    -------
    coords : np.ndarray
        Stacked coordinates from all batches.
    spectra_reshaped : str
        Placeholder (currently unused).
    spectra_raw : list of np.ndarray
        All raw spectra arrays.
    """
    all_coords, all_reshaped, all_raw = [], [], []

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        coords, reshaped, raw = parallel_reader(
            batch, n_cpu=min(n_cpu, len(batch)), function=function
        )
        all_coords.append(coords)
        all_reshaped.append(reshaped)
        all_raw.extend(raw)

    coords = np.vstack(all_coords)
    return coords, "unused", all_raw


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


# Ensure start method is set safely when calling as a script
if __name__ == '__main__':
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set
