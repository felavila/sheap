import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from astropy.io import fits

from sheap.utils import resize_and_fill_with_nans

n_cpu = os.cpu_count()  # Number of CPUs to use

def fits_reader_sdss(file):
    """
    Read an SDSS FITS file and extract wavelength, flux, and inverse variance,
    scaled appropriately by the BUNIT header.
    """
    hdul = fits.open(file)
    flux_scale = float(hdul[0].header["BUNIT"].split(" ")[0])
    data_array = np.array([
        10**hdul[1].data["loglam"],
        hdul[1].data["flux"] * flux_scale,
        flux_scale / np.sqrt(hdul[1].data["ivar"])
    ])
    header_array = np.array([
        hdul[0].header["PLUG_RA"],
        hdul[0].header["PLUG_DEC"]
    ])
    return data_array, header_array

def fits_reader_pyqso(file):
    """
    Read a PyQSO FITS file and extract prereduced wavelength, flux, and error.
    """
    hdul = fits.open(file)
    spectra = np.array([
        hdul[3].data["wave_prereduced"],
        hdul[3].data["flux_prereduced"],
        hdul[3].data["err_prereduced"]
    ])
    return spectra

def parallel_reader(paths, n_cpu=n_cpu, function=fits_reader_sdss, parallel=True):
    """
    Read multiple FITS files either in parallel or sequentially.

    Parameters:
    - paths: list of str, paths to FITS files.
    - n_cpu: int, number of CPUs to use.
    - function: function to use for reading each file.
    - parallel: bool, whether to run in parallel.

    Returns:
    - Coordinates and spectra arrays.
    """
    if not parallel:
        print("Doing the reading not parallel.")
        results = [function(i) for i in paths]
        spectra = [result[0] for result in results]
        shapes_max = max(s.shape[1] for s in spectra)
        coords = np.array([result[1] for result in results])
        spectra_reshaped = np.array([resize_and_fill_with_nans(s, shapes_max) for s in spectra])
        return coords, spectra_reshaped, spectra

    multiprocessing.set_start_method('spawn', force=True)
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        results = list(executor.map(function, paths))

        if function == fits_reader_sdss:
            spectra = [result[0] for result in results]
            coords = np.array([result[1] for result in results])
            shapes_max = max(s.shape[1] for s in spectra)
            spectra_reshaped = np.array([resize_and_fill_with_nans(s, shapes_max) for s in spectra])
            return coords, spectra_reshaped, spectra

        spectra = results
        shapes_max = 4644  # Fixed shapes_max for PyQSO
        spectra_reshaped = np.array([resize_and_fill_with_nans(s, shapes_max) for s in spectra])
        return spectra_reshaped, spectra
