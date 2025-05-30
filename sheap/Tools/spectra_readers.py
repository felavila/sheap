import os
import numpy as np
from multiprocessing import Pool, set_start_method
from astropy.io import fits

# Limit CPUs for safety
n_cpu = min(4, os.cpu_count())  # Adjustable

def resize_and_fill_with_nans(original_array, new_xaxis_length, number_columns=3):
    """
    Resize an array to the target shape, filling new entries with NaNs.
    """
    new_array = np.full((number_columns, new_xaxis_length), np.nan, dtype=float)
    slices = tuple(
        slice(0, min(o, t))
        for o, t in zip(original_array.shape, (number_columns, new_xaxis_length))
    )
    new_array[slices] = original_array[slices]
    return new_array

def fits_reader_simulation(file):
    hdul = fits.open(file)
    data_array = np.array([hdul[1].data["LAMBDA"], hdul[1].data["FLUX_DENSITY"]])
    header_array = []
    return data_array, header_array

def fits_reader_sdss(file):
    hdul = fits.open(file)
    flux_scale = float(hdul[0].header["BUNIT"].split(" ")[0])
    data_array = np.array([
        10 ** hdul[1].data["loglam"],
        hdul[1].data["flux"] * flux_scale,
        flux_scale / np.sqrt(hdul[1].data["ivar"]),
    ])
    data_array[np.isinf(data_array)] = 1e20
    header_array = np.array([hdul[0].header["PLUG_RA"], hdul[0].header["PLUG_DEC"]])
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
}

def parallel_reader_safe(paths, n_cpu=n_cpu, function=fits_reader_sdss):
    """
    Safe parallel reading using multiprocessing.Pool.
    """
    if isinstance(function, str):
        function = READER_FUNCTIONS[function]

    with Pool(processes=min(n_cpu, len(paths))) as pool:
        results = pool.map(function, paths, chunksize=1)

    spectra = [result[0] for result in results]
    coords = np.array([result[1] for result in results])
    shapes_max = max(s.shape[1] for s in spectra)
    spectra_reshaped = np.array([
        resize_and_fill_with_nans(s, shapes_max) for s in spectra
    ])
    return coords, spectra_reshaped, spectra

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
