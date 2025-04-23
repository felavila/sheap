import numpy as np 
from astropy.io import fits 
from .utils import resize_and_fill_with_nans
import os 
import multiprocessing
from concurrent.futures import ProcessPoolExecutor



n_cpu = os.cpu_count()  # Number of CPUs to use
def fits_reader_sdss(file):
    # we have to check for the all the important things that came in the fits file from SDSS
    hdul = fits.open(file)
    aD = np.array([10**hdul[1].data[key] if key == "loglam" else (float(hdul[0].header["BUNIT"].split(" ")[0]) / np.sqrt(hdul[1].data[key]) if key == "ivar" else hdul[1].data[key]*float(hdul[0].header["BUNIT"].split(" ")[0])) for key in ["loglam", "flux", "ivar"]])
    aH = np.array([hdul[0].header[key] for key in ["PLUG_RA", "PLUG_DEC"]])
    return aD, aH
def fits_reader_pyqso(file):
    hdul = fits.open(file)
    spectra = np.array([hdul[3].data[key] for key in ["wave_prereduced", "flux_prereduced", "err_prereduced"]])
    return spectra 

# def parallel_reader(paths,n_cpu=n_cpu,function=fits_reader_sdss):
#     with ProcessPoolExecutor(max_workers=n_cpu) as executor:
#         results = list(executor.map(function,paths ))
#         spectra = [result[0] for result in results]
#         shapes_max = max([s.shape[1] for s in spectra])
#         coords = np.array([result[1] for result in results])
#         spectra_reshaped = np.array([resize_and_fill_with_nans(s,shapes_max) for s in spectra])
#     return coords,spectra_reshaped

# import multiprocessing
# from concurrent.futures import ProcessPoolExecutor

# Set the start method to spawn at the beginning of your program


# Now in your function
def parallel_reader(paths, n_cpu=n_cpu, function=fits_reader_sdss,parallel=True):
    if not parallel:
        print("doing the reading not parallel ")
        results = []
        for i in paths:
            results.append(function(i))
        spectra = [result[0] for result in results]
        shapes_max = max(s.shape[1] for s in spectra)
        coords = np.array([result[1] for result in results])
        spectra_reshaped = np.array([resize_and_fill_with_nans(s, shapes_max) for s in spectra])
        return coords, spectra_reshaped,spectra
    # Use the default ProcessPoolExecutor, which now uses spawn under the hood
    multiprocessing.set_start_method('spawn', force=True)
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        if function==fits_reader_sdss:
            results = list(executor.map(function, paths))
            spectra = [result[0] for result in results]
            shapes_max = max(s.shape[1] for s in spectra)
            coords = np.array([result[1] for result in results])
            spectra_reshaped = np.array([resize_and_fill_with_nans(s, shapes_max) for s in spectra])
            return coords, spectra_reshaped,spectra
        else:
            results = list(executor.map(fits_reader_pyqso, paths))
            spectra = [result for result in results]
            shapes_max = 4644
            spectra_reshaped = np.array([resize_and_fill_with_nans(s, shapes_max) for s in spectra])
            return spectra_reshaped,spectra