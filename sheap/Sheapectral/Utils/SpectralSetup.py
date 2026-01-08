"""
Spectral preparation & masking utilities.

This module provides small, JAX‑friendly helpers to:
- hard‑cut spectra to wavelength windows,
- pad a missing error channel,
- normalize/reshape batches to a common pixel length,
- build robust masks and uncertainty arrays for fitting.

Notes
-----
- All functions are written to be light on copies and JAX‑compatible
    (using `jnp` where the arrays participate in later JAX code).
"""

__author__ = 'felavila'

__all__ = [
    "cut_spectra",
    "mask_builder",
    "pad_error_channel",
    "prepare_spectra",
    "prepare_uncertainties",
    "resize_and_fill_with_nans",
    "ensure_sfd_data"
]

from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from pathlib import Path
import requests
from sheap.Core import ArrayLike


def resize_and_fill_with_nans(
    original_array: np.ndarray,
    new_xaxis_length: int,
    number_columns: Optional[int] = None,
) -> np.ndarray:
    """
    Resize a (C, N) spectral array to (C_out, new_xaxis_length), padding with NaNs.

    Parameters
    ----------
    original_array
        Input array with shape (C, N).
    new_xaxis_length
        Target number of pixels along the wavelength axis.
    number_columns
        Target number of channels C_out. If None, uses original C.

    Returns
    -------
    np.ndarray
        New array of shape (C_out, new_xaxis_length) with the original
        data copied into the upper-left corner and NaNs elsewhere.
    """
    C_in, N_in = original_array.shape
    C_out = number_columns or C_in
    out = np.full((C_out, new_xaxis_length), np.nan, dtype=float)
    out[: min(C_in, C_out), : min(N_in, new_xaxis_length)] = original_array[
        : min(C_in, C_out), : min(N_in, new_xaxis_length)
    ]
    return out


def prepare_spectra(
    spectra_list: Sequence[np.ndarray],
    outer_limits: Tuple[float, float],
):
    """
    Cut each spectrum to `outer_limits`, pad to common length, and build masks.

    Parameters
    ----------
    spectra_list
        Iterable of per‑object arrays with shape (C, N).
    outer_limits
        (xmin, xmax) hard window to keep.

    Returns
    -------
    spectral_region : jnp.ndarray
        Batched spectra of shape (n_obj, C, N_max) after cutting/padding.
    mask_region : jnp.ndarray
        Boolean mask with True where samples are masked (ignored in fit).
    """
    xmin, xmax = outer_limits
    clipped = [cut_spectra(s, xmin, xmax) for s in spectra_list]
    n_max = max(s.shape[1] for s in clipped)
    stacked = jnp.array([resize_and_fill_with_nans(s, n_max) for s in clipped])

    spectral_region, _, _, mask_region = mask_builder(
        stacked, outer_limits=outer_limits
    )
    return spectral_region, mask_region


def cut_spectra(spectra: ArrayLike, xmin: float, xmax: float) -> ArrayLike:
    """
    Hard cut a spectrum to a wavelength interval.

    Parameters
    ----------
    spectra
        Array with shape (C, N) whose first row is wavelength [Å].
    xmin, xmax
        Interval bounds.

    Returns
    -------
    ArrayLike
        The sliced spectrum with pixels xmin ≤ λ ≤ xmax.
    """
    wl = spectra[0, :]
    sel = (wl >= xmin) & (wl <= xmax)
    return spectra[:, sel]


def mask_builder(
    sheap_array: jnp.ndarray,
    inner_limits: Tuple[float, float] = (0.0, 0.0),
    outer_limits: Optional[Tuple[float, float]] = None,
    instrumental_limit: float = 1e50,  # kept for API compatibility; not used internally
):
    """
    Build a robust mask and uncertainty channel for a batch of spectra.

    Rules:
    - Mask pixels inside `inner_limits` (to exclude e.g. strong tellurics).
    - If `outer_limits` is provided, mask pixels outside it.
    - Mask NaN wavelengths / infinite errors / non‑positive flux.
    - Convert any NaNs in the error channel to a very large uncertainty (1e31).

    Parameters
    ----------
    sheap_array
        Array with shape (n_obj, C, N), channels: [λ, flux, err, (opt ...)].
    inner_limits
        (xmin, xmax) wavelengths to *mask out* inside this window.
    outer_limits
        If given, wavelengths outside (xmin, xmax) are masked.
    instrumental_limit
        Unused placeholder (kept to avoid breaking callers).

    Returns
    -------
    array : jnp.ndarray
        Copy of input with the error channel replaced by prepared uncertainties.
    prepared_uncertainties : jnp.ndarray
        The prepared error channel (same shape as input error channel).
    original_array : jnp.ndarray
        Reference to the original `sheap_array`.
    mask : jnp.ndarray
        Boolean mask (True = masked / ignored).
    """
    copy_array = jnp.array(sheap_array)
    wl = sheap_array[:, 0, :]
    flux = sheap_array[:, 1, :]
    err = sheap_array[:, 2, :]

    # Mask inside inner limits
    mask = (wl >= inner_limits[0]) & (wl <= inner_limits[1])

    # And outside outer limits, if provided
    if outer_limits is not None:
        mask |= (wl < outer_limits[0]) | (wl > outer_limits[1])

    # Invalidate bad values
    mask |= jnp.isnan(wl) | jnp.isinf(err) | (flux <= 0)

    # Set error to NaN where masked; then prepare_uncertainties → 1e31 at those places
    err_masked = jnp.where(mask, jnp.nan, err)
    copy_array = copy_array.at[:, 2, :].set(err_masked)

    prepared = prepare_uncertainties(copy_array[:, 2, :], flux)
    copy_array = copy_array.at[:, 2, :].set(prepared)

    return copy_array, prepared, sheap_array, mask


def prepare_uncertainties(
    y_uncertainties: Optional[jnp.ndarray],
    y_data: jnp.ndarray,
) -> jnp.ndarray:
    """
    Prepare an uncertainty channel consistent with masking rules.

    - If `y_uncertainties` is None, returns an array of ones.
    - Any NaNs in the uncertainties or the data are set to a very large value (1e31),
        so inverse‑variance weighting effectively ignores those pixels.

    Parameters
    ----------
    y_uncertainties
        Error channel or None.
    y_data
        Flux channel (used only to propagate NaN locations).

    Returns
    -------
    jnp.ndarray
        Prepared uncertainties (same shape as `y_data`).
    """
    if y_uncertainties is None:
        y_uncertainties = jnp.ones_like(y_data)

    bad = jnp.isnan(y_data) | jnp.isnan(y_uncertainties)
    return jnp.where(bad, 1e31, y_uncertainties)


def pad_error_channel(spectra: ArrayLike, frac: float = 0.01) -> ArrayLike:
    """
    Ensure a third channel (error) by padding with a fraction of the signal.

    Parameters
    ----------
    spectra
        Array with shape (n_obj, C, N) or (C, N). If C==2 (λ, flux), the error
        channel is appended as `frac * flux`.
    frac
        Error fraction applied to the flux to fabricate uncertainties.

    Returns
    -------
    ArrayLike
        Spectra with shape (..., 3, N).
    """
    if spectra.shape[1] != 2:
        return spectra  # already has ≥3 channels
    signal = spectra[:, 1, :]
    error = jnp.expand_dims(signal * frac, axis=1)
    return jnp.concatenate((spectra, error), axis=1)


def ensure_sfd_data(sfd_path: Path = None):
    """
    Ensure the Schlegel, Finkbeiner & Davis (1998) dust maps are available locally.
    Downloads the 4 required FITS files into `sfd_path` if missing.

    Parameters
    ----------
    sfd_path : Path, optional
        Directory where the SFD data should be stored.
        Defaults to `SuportData/sfddata` relative to this file.

    Files
    -----
        - SFD_dust_4096_ngp.fits
        - SFD_dust_4096_sgp.fits
        - SFD_mask_4096_ngp.fits
        - SFD_mask_4096_sgp.fits
    """
    if sfd_path is None:
        sfd_path = Path(__file__).resolve().parent.parent / "SuportData" / "sfddata"

    sfd_path.mkdir(parents=True, exist_ok=True)

    files = [
        "SFD_dust_4096_ngp.fits",
        "SFD_dust_4096_sgp.fits",
        "SFD_mask_4096_ngp.fits",
        "SFD_mask_4096_sgp.fits",
    ]

    base_url = "https://raw.githubusercontent.com/kbarbary/sfddata/master"

    missing = [fname for fname in files if not (sfd_path / fname).exists()]
    if not missing:
        return
    print(f"For the SFD correction is necessary download a list of files ({missing}) this will be done just ones")
    for fname in missing:
        url = f"{base_url}/{fname}"
        outpath = sfd_path / fname
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(outpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)






# """
# Spectral preparation & masking utilities.

# This module provides small, JAX‑friendly helpers to:
# - hard‑cut spectra to wavelength windows,
# - pad a missing error channel,
# - normalize/reshape batches to a common pixel length,
# - build robust masks and uncertainty arrays for fitting.

# Notes
# -----
# - All functions are written to be light on copies and JAX‑compatible
#     (using `jnp` where the arrays participate in later JAX code).
# """

# __author__ = 'felavila'


# __all__ = [
#     "cut_spectra",
#     "mask_builder",
#     "pad_error_channel",
#     "prepare_spectra",
#     "prepare_uncertainties",
#     "resize_and_fill_with_nans",
# ]

# from typing import Callable, Dict, Optional, Tuple, Union

# import jax.numpy as jnp
# import numpy as np 

# from sheap.Core import ArrayLike





# def resize_and_fill_with_nans(original_array, new_xaxis_length, number_columns=4):
#     """
#     Resize an array to the target shape, filling new entries with NaNs.
#     """
#     new_array = np.full((number_columns, new_xaxis_length), np.nan, dtype=float)
#     slices = tuple(
#         slice(0, min(o, t))
#         for o, t in zip(original_array.shape, (number_columns, new_xaxis_length))
#     )
#     new_array[slices] = original_array[slices]
#     return new_array


# def prepare_spectra(spectra_list, outer_limits):
#     list_cut = [cut_spectra(s, *outer_limits) for s in spectra_list]
#     shapes_max = max(s.shape[1] for s in list_cut)
#     spectra_reshaped = jnp.array([resize_and_fill_with_nans(s, shapes_max) for s in list_cut])
#     spectral_region, _, _, mask_region = mask_builder(
#         spectra_reshaped, outer_limits=outer_limits
#     )
#     return spectral_region, mask_region


# def cut_spectra(spectra, xmin, xmax):
#     """hard cut of the spectra"""
#     mask = (spectra[0, :] >= xmin) & (spectra[0, :] <= xmax)
#     spectra = spectra[:, mask]
#     return spectra


# def mask_builder(
#     sheap_array, inner_limits=[0, 0], outer_limits=None, instrumental_limit=10e50
# ):
#     """
#     -full nan the error matrix

#     if outer_limits is not None:
#         mask_outside_outer = (sheap_array[:, 0, :] < outer_limits[0]) | (sheap_array[:, 0, :] > outer_limits[1])
#     Parameters:
#     - sheap_array: Input array with shape (N, 3, M).
#     - inner_limits: List of two values [min, max] for the inner limits.
#     - outer_limits: Optional list of two values [min, max] for the outer limits.
#     - instrumental_limit: in units of flux this defines the limit that can reach the instrument after understimate the error
#     Returns:
#     - array: Array with masked values based on the limits.
#     - mask: Prepared uncertainties array.
#     - original_array: The original sheap_array.
#     - masked_uncertainties: The mask applied to the array this means the error in these regions go to 1e11
#     comment:
#         # Combine masks to mask values inside inner_limits or outside outter_limits
#         # take the uncertainties and put it to nan in the region that we wan to not take in account
#         #place in where we want to not fit
#     """
#     copy_array = jnp.copy(sheap_array)
#     mask = (sheap_array[:, 0, :] >= inner_limits[0]) & (
#         sheap_array[:, 0, :] <= inner_limits[1]
#     )
#     if outer_limits is not None:
#         mask_outside_outter = (sheap_array[:, 0, :] < outer_limits[0]) | (
#             sheap_array[:, 0, :] > outer_limits[1]
#         )
#         mask = mask | mask_outside_outter
#     mask = (
#         mask
#         | (jnp.isnan(sheap_array[:, 0, :]) | jnp.isinf(sheap_array[:, 2, :]))
#         | (sheap_array[:, 1, :] <= 0)
#     )
#     copy_array = copy_array.at[:, 2, :].set(jnp.where(mask, jnp.nan, copy_array[:, 2, :]))
#     masked_uncertainties = prepare_uncertainties(copy_array[:, 2, :], copy_array[:, 1, :])
#     copy_array = copy_array.at[:, 2, :].set(masked_uncertainties)
#     # masked_uncertainties = masked_uncertainties == 1.e+31
#     return copy_array, masked_uncertainties, sheap_array, mask



# def prepare_uncertainties(
#     y_uncertainties: Optional[jnp.ndarray], y_data: jnp.ndarray
# ) -> jnp.ndarray:
#     """
#     Prepare the y_uncertainties array. If None, return an array of ones.
#     If there are NaN values in y_data, set the corresponding uncertainties to 1e11.

#     Parameters:
#     - y_uncertainties: Provided uncertainties or None.
#     - y_data: The target data array.

#     Returns:
#     - y_uncertainties: An array of uncertainties.
#     """
#     if y_uncertainties is None:
#         y_uncertainties = jnp.ones_like(y_data)

#     # Identify positions where y_data has NaN values
#     nan_positions = jnp.isnan(y_data) | jnp.isnan(y_uncertainties)

#     # Set uncertainties to 1e11 at positions where y_data is NaN/here i have some corncerns about is it is weight or not
#     y_uncertainties = jnp.where(nan_positions, 1e31, y_uncertainties)

#     return y_uncertainties



# # TODO Add multiple models to the reading.
# def pad_error_channel(spectra: ArrayLike, frac: float = 0.01) -> ArrayLike:
#     """Ensure *spectra* has a third channel (error) by padding with *frac* × signal."""
#     if spectra.shape[1] != 2:
#         return spectra  # already 3‑channel
#     signal = spectra[:, 1, :]
#     error = jnp.expand_dims(signal * frac, axis=1)
#     return jnp.concatenate((spectra, error), axis=1)
