"""This module handles basic operations."""
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

from sheap.Utils.SpectralSetup import resize_and_fill_with_nans
# Limit CPUs for safety
n_cpu = min(4, os.cpu_count())  # Adjustable

def fits_reader_desi(file):
    "wdisp in pixels i guess"
    
    
    hdul = fits.open(file)
    flux_scale = float(hdul[1].header["TUNIT2"].split(" ")[0])
    ivar_scale = float(hdul[1].header["TUNIT3"].split(" ")[0])
    data = hdul[1].data
    data_array = np.array([data["WAVELENGTH"],data["FLUX"] * flux_scale,1/np.sqrt(data["IVAR"]* ivar_scale)])
    data_array[np.isinf(data_array)] = 1e20
    header_array = np.array([hdul[0].header["RA"], hdul[0].header["DEC"]])#PLUG_RA/PLUG_DEC
    return data_array, header_array


#('WAVE', '>f4', (23198,)), ('FLUX', '>f4', (23198,)), ('ERR_FLUX', '>f4', (23198,)
def fits_reader_simulation(file,chanel=1,template=False):
    hdul = fits.open(file)
    header_array = []
    if template:
        data_array = np.array([hdul[chanel].data['LAMBDA'], hdul[chanel].data["FLUX_DENSITY"]])
        return data_array.squeeze(), header_array
    if chanel==1:
        data_array = np.array([hdul[chanel].data["WAVE"], hdul[chanel].data["FLUX"],hdul[chanel].data["ERR_FLUX"]])
    else:
        data_array = np.array([hdul[chanel].data["WAVE"], hdul[chanel].data["FLUX"],hdul[chanel].data["ERR"]])
    
    return data_array.squeeze(), header_array

def fits_reader_sdss(file):
    "wdisp in pixels i guess"
    
    
    hdul = fits.open(file)
    flux_scale = float(hdul[0].header["BUNIT"].split(" ")[0])
    data = hdul[1].data
    data_array = np.array([
        10 ** data["loglam"],
        data["flux"] * flux_scale,
        flux_scale / np.sqrt(data["ivar"]),data["wdisp"]])
    data_array[np.isinf(data_array)] = 1e20
    header_array = np.array([hdul[0].header["RA"], hdul[0].header["DEC"]])#PLUG_RA/PLUG_DEC
    return data_array, header_array

def fits_reader_pyqso(file):
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
    "fits_reader_desi" :fits_reader_desi
}




def parallel_reader(paths, n_cpu=n_cpu, function=fits_reader_sdss, **kwargs):
    """
    Safe parallel reading using multiprocessing.Pool.
    Accepts additional keyword arguments for the reader function.
    """
    if isinstance(function, str):
        function = READER_FUNCTIONS[function]

    func_with_args = partial(function, **kwargs)

    with Pool(processes=min(n_cpu, len(paths))) as pool:
        results = pool.map(func_with_args, paths, chunksize=1)

    spectra = [result[0] for result in results]
    coords = np.array([result[1] for result in results])
    shapes_max = max(s.shape[1] for s in spectra)
    spectra_reshaped = []
    #np.array([
        #resize_and_fill_with_nans(s, shapes_max) for s in spectra
    #])
    return coords, spectra_reshaped, spectra

# def parallel_reader_safe(paths, n_cpu=n_cpu, function=fits_reader_sdss):
#     """
#     Safe parallel reading using multiprocessing.Pool.
#     """
#     if isinstance(function, str):
#         function = READER_FUNCTIONS[function]

#     with Pool(processes=min(n_cpu, len(paths))) as pool:
#         results = pool.map(function, paths, chunksize=1)

#     spectra = [result[0] for result in results]
#     coords = np.array([result[1] for result in results])
#     shapes_max = max(s.shape[1] for s in spectra)
#     spectra_reshaped = np.array([
#         resize_and_fill_with_nans(s, shapes_max) for s in spectra
#     ])
#     return coords, spectra_reshaped, spectra

def batched_reader(paths, batch_size=8, function=fits_reader_sdss):
    """
    Batch files in groups for safer memory usage.
    """
    all_coords, all_reshaped, all_raw = [], [], []

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        coords, reshaped, raw = parallel_reader_safe(
            batch, n_cpu=min(n_cpu, len(batch)), function=function
        )
        all_coords.append(coords)
        all_reshaped.append(reshaped)
        all_raw.extend(raw)

    coords = np.vstack(all_coords)
    _ = "a"
    #spectra_reshaped = np.vstack(all_reshaped)
    return coords, _, all_raw

def sequential_reader(paths, function=fits_reader_sdss):
    """
    Fully sequential fallback reader.
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
    spectra_reshaped = np.array([
        resize_and_fill_with_nans(s, shapes_max) for s in spectra
    ])
    return coords, spectra_reshaped, spectra

# Ensure start method is set safely when calling as a script
if __name__ == '__main__':
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set
