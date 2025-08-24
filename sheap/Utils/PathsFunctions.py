"""
Cross-matching spectra with metadata
====================================

This module provides utilities to cross-match lists of spectra
stored in a directory with metadata stored in a CSV or pandas DataFrame.
It supports both SDSS and DESI-style naming conventions.

Functions
---------
cross_pandas_spectra(path_dr16, path_data, name_csv)
    Cross-match SDSS-style spectra (PLATE-MJD-FIBERID) with metadata.

cross_pandas_spectra_desi(path_dr16, path_data, name_csv)
    Cross-match DESI-style spectra with metadata.
"""

__author__ = 'felavila'

__all__ = [
    "cross_pandas_spectra",
    "cross_pandas_spectra_desi",
]

import glob
import os
import pandas as pd


def cross_pandas_spectra_desi(path_dr16: str, path_data: str, name_csv: str):
    """
    Cross-match DESI spectra with entries in a metadata CSV.

    Parameters
    ----------
    path_dr16 : str
        Base path to the DR16/DR data directory.
    path_data : str
        Subdirectory under `path_dr16` containing `.fits` spectra.
    name_csv : str
        CSV file with metadata table. Must include ``PLATE``, ``MJD``, ``FIBERID``.

    Returns
    -------
    file_paths : list of str
        All spectrum file paths found in the directory.
    objs_panda_paths_filtered : pandas.DataFrame or None
        Filtered DataFrame of metadata entries with matches to actual spectra.
        Includes a new column ``fit_path`` with the resolved file paths.
        Returns ``None, None`` if no matches were found.
    """
    file_paths = glob.glob(f"{path_dr16}/{path_data}/*.fits")
    objs_panda = pd.read_csv(f"{path_dr16}/{name_csv}")
    objs_panda["dr_name"] = [
        f"{PLATE:04d}-{MJD:05d}-{FIBERID:04d}"
        for PLATE, MJD, FIBERID in objs_panda[["PLATE", "MJD", "FIBERID"]].values
    ]
    objs_panda_paths_list = [
        os.path.basename(path).replace(".fits", "") for path in file_paths
    ]
    objs_panda_paths_filtered = objs_panda[
        objs_panda["dr_name"].isin(objs_panda_paths_list)
    ].reset_index(drop=True)

    if len(objs_panda_paths_filtered) == 0:
        print("No matches found.")
        return None, None
    else:
        print(f"Cross-match found {len(objs_panda_paths_filtered)} entries.")

    objs_panda_paths_filtered["fit_path"] = (
        objs_panda_paths_filtered["dr_name"]
        .apply(lambda x: os.path.join(path_dr16, f"{path_data}/{x}.fits"))
        .values
    )
    return file_paths, objs_panda_paths_filtered


def cross_pandas_spectra(path_dr16: str, path_data: str, name_csv: str):
    """
    Cross-match SDSS spectra with entries in a metadata CSV.

    Parameters
    ----------
    path_dr16 : str
        Base path to the DR16/DR data directory.
    path_data : str
        Subdirectory under `path_dr16` containing `.fits` spectra.
    name_csv : str
        CSV file with metadata table. Must include ``PLATE``, ``MJD``, ``FIBERID``.

    Returns
    -------
    file_paths : list of str
        All spectrum file paths found in the directory.
    objs_panda_paths_filtered : pandas.DataFrame or None
        Filtered DataFrame of metadata entries with matches to actual spectra.
        Includes a new column ``fit_path`` with the resolved file paths.
        Returns ``None, None`` if no matches were found.
    """
    file_paths = glob.glob(f"{path_dr16}/{path_data}/*.fits")
    objs_panda = pd.read_csv(f"{path_dr16}/{name_csv}")
    objs_panda["dr_name"] = [
        f"{int(PLATE):04d}-{int(MJD):05d}-{int(FIBERID):04d}"
        for PLATE, MJD, FIBERID in objs_panda[["PLATE", "MJD", "FIBERID"]].values
    ]
    objs_panda_paths_list = [
        os.path.basename(path).replace(".fits", "") for path in file_paths
    ]
    objs_panda_paths_filtered = objs_panda[
        objs_panda["dr_name"].isin(objs_panda_paths_list)
    ].reset_index(drop=True)

    if len(objs_panda_paths_filtered) == 0:
        print("No matches found.")
        return None, None
    else:
        print(f"Cross-match found {len(objs_panda_paths_filtered)} entries.")

    objs_panda_paths_filtered["fit_path"] = (
        objs_panda_paths_filtered["dr_name"]
        .apply(lambda x: os.path.join(path_dr16, f"{path_data}/{x}.fits"))
        .values
    )
    return file_paths, objs_panda_paths_filtered
