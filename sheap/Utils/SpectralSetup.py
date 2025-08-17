"""This module handles basic operations."""
__author__ = 'felavila'


__all__ = [
    "cut_spectra",
    "mask_builder",
    "pad_error_channel",
    "prepare_spectra",
    "prepare_uncertainties",
    "resize_and_fill_with_nans",
]

from typing import Callable, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np 

from sheap.Core import ArrayLike





def resize_and_fill_with_nans(original_array, new_xaxis_length, number_columns=4):
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


def prepare_spectra(spectra_list, outer_limits):
    list_cut = [cut_spectra(s, *outer_limits) for s in spectra_list]
    shapes_max = max(s.shape[1] for s in list_cut)
    spectra_reshaped = jnp.array([resize_and_fill_with_nans(s, shapes_max) for s in list_cut])
    spectral_region, _, _, mask_region = mask_builder(
        spectra_reshaped, outer_limits=outer_limits
    )
    return spectral_region, mask_region


def cut_spectra(spectra, xmin, xmax):
    """hard cut of the spectra"""
    mask = (spectra[0, :] >= xmin) & (spectra[0, :] <= xmax)
    spectra = spectra[:, mask]
    return spectra


def mask_builder(
    sheap_array, inner_limits=[0, 0], outer_limits=None, instrumental_limit=10e50
):
    """
    -full nan the error matrix

    if outer_limits is not None:
        mask_outside_outer = (sheap_array[:, 0, :] < outer_limits[0]) | (sheap_array[:, 0, :] > outer_limits[1])
    Parameters:
    - sheap_array: Input array with shape (N, 3, M).
    - inner_limits: List of two values [min, max] for the inner limits.
    - outer_limits: Optional list of two values [min, max] for the outer limits.
    - instrumental_limit: in units of flux this defines the limit that can reach the instrument after understimate the error
    Returns:
    - array: Array with masked values based on the limits.
    - mask: Prepared uncertainties array.
    - original_array: The original sheap_array.
    - masked_uncertainties: The mask applied to the array this means the error in these regions go to 1e11
    comment:
        # Combine masks to mask values inside inner_limits or outside outter_limits
        # take the uncertainties and put it to nan in the region that we wan to not take in account
        #place in where we want to not fit
    """
    copy_array = jnp.copy(sheap_array)
    mask = (sheap_array[:, 0, :] >= inner_limits[0]) & (
        sheap_array[:, 0, :] <= inner_limits[1]
    )
    if outer_limits is not None:
        mask_outside_outter = (sheap_array[:, 0, :] < outer_limits[0]) | (
            sheap_array[:, 0, :] > outer_limits[1]
        )
        mask = mask | mask_outside_outter
    mask = (
        mask
        | (jnp.isnan(sheap_array[:, 0, :]) | jnp.isinf(sheap_array[:, 2, :]))
        | (sheap_array[:, 1, :] <= 0)
    )
    copy_array = copy_array.at[:, 2, :].set(jnp.where(mask, jnp.nan, copy_array[:, 2, :]))
    masked_uncertainties = prepare_uncertainties(copy_array[:, 2, :], copy_array[:, 1, :])
    copy_array = copy_array.at[:, 2, :].set(masked_uncertainties)
    # masked_uncertainties = masked_uncertainties == 1.e+31
    return copy_array, masked_uncertainties, sheap_array, mask



def prepare_uncertainties(
    y_uncertainties: Optional[jnp.ndarray], y_data: jnp.ndarray
) -> jnp.ndarray:
    """
    Prepare the y_uncertainties array. If None, return an array of ones.
    If there are NaN values in y_data, set the corresponding uncertainties to 1e11.

    Parameters:
    - y_uncertainties: Provided uncertainties or None.
    - y_data: The target data array.

    Returns:
    - y_uncertainties: An array of uncertainties.
    """
    if y_uncertainties is None:
        y_uncertainties = jnp.ones_like(y_data)

    # Identify positions where y_data has NaN values
    nan_positions = jnp.isnan(y_data) | jnp.isnan(y_uncertainties)

    # Set uncertainties to 1e11 at positions where y_data is NaN/here i have some corncerns about is it is weight or not
    y_uncertainties = jnp.where(nan_positions, 1e31, y_uncertainties)

    return y_uncertainties



# TODO Add multiple models to the reading.
def pad_error_channel(spectra: ArrayLike, frac: float = 0.01) -> ArrayLike:
    """Ensure *spectra* has a third channel (error) by padding with *frac* × signal."""
    if spectra.shape[1] != 2:
        return spectra  # already 3‑channel
    signal = spectra[:, 1, :]
    error = jnp.expand_dims(signal * frac, axis=1)
    return jnp.concatenate((spectra, error), axis=1)
