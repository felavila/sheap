"""
Flattening and Summarizing Utilities
====================================

This module provides helpers to:
- Concatenate lists of dictionaries with array-like leaves.
- Flatten nested “masses” / parameter dictionaries to pandas DataFrames.
- Pivot nested results per object into per-object dictionaries.
- Summarize posterior samples via 16/50/84 percentiles.

Notes
-----
- All functions are NumPy/JAX friendly; arrays may be ``numpy.ndarray`` or ``jax.numpy.ndarray``.
- Percentile summaries follow a common convention:
    median = 50th percentile, err_minus = 50th - 16th, err_plus = 84th - 50th.
"""

__author__ = 'felavila'

__all__ = [
    "flatten_mass_dict",
    "flatten_mass_samples_to_df",
    "flatten_param_dict",
    "flatten_scalar_dict",
    "pivot_and_split",
    "summarize_nested_samples",
    "summarize_samples",
    "concat_dicts"
]

from typing import Dict, Any
import pandas as pd
import warnings
from collections import defaultdict
#from auto_uncertainties.uncertainty.uncertainty_containers import VectorUncertainty
import numpy as np 
import jax.numpy as jnp
from uncertainties import unumpy
#?

def concat_dicts(list_of_dicts):
    """
    Concatenate lists/arrays across a list of homogeneous dictionaries.

    Parameters
    ----------
    list_of_dicts : list of dict
        Each dict must share the same keys. Values are arrays (or array-like)
        that can be concatenated along their first axis.

    Returns
    -------
    dict
        Dictionary with the same keys; each value is the concatenation of
        the corresponding values across the input list, then transposed
        (so that samples shape usually becomes ``(N, ...)``).

    Notes
    -----
    This is used to merge per-profile/line dictionaries into a single
    per-region dict. It expects all leaves to be concatenable; non-numeric
    leaves should be filtered out before calling.
    """
    out = defaultdict(list)
    for d in list_of_dicts:
        for k, v in d.items():
            out[k].append(v)

    # flatten or stack if numeric/array-like
    for k, v in out.items():
        out[k] = np.concatenate([x for x in v]).T
    return dict(out)


def flatten_mass_samples_to_df(dict_samples: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten nested mass sample summaries into a tidy DataFrame.

    Parameters
    ----------
    dict_samples : dict
        Mapping ``object_name -> {"masses": {line: {quantity: stats_dict}}}``.
        Each ``stats_dict`` must have keys ``median``, ``err_minus``, ``err_plus``
        (scalars or 0-d arrays).

    Returns
    -------
    pandas.DataFrame
        Columns: ``object``, ``line``, ``quantity``, ``median``, ``err_minus``, ``err_plus``.
    """
    records = []
    
    for object_key, item in dict_samples.items():
        if not isinstance(item, Dict) or "masses" not in item:
            continue
        for line_name, stats in item["masses"].items():
            for stat_name, values in stats.items():
                records.append({
                    "object": object_key,
                    "line": line_name,
                    "quantity": stat_name,
                    "median": values["median"].item(),
                    "err_minus": values["err_minus"].item(),
                    "err_plus": values["err_plus"].item()
                })
    
    return pd.DataFrame(records)


def flatten_param_dict(dict_basic_params):
    """
    Convert a nested parameter dictionary into a tidy table.

    Parameters
    ----------
    dict_basic_params : dict
        Structure like:
        ``{kind: {"lines": [...], "component": [...], <param>: {"median": [...], "err_minus": [...], "err_plus": [...]}, ...}}``

    Returns
    -------
    pandas.DataFrame
        One row per (line, component, kind, parameter), with median and error bars.
    """
    rows = []
    for kind, values in dict_basic_params.items():
        lines = values["lines"]
        components = values["component"]
        for param_name, param_values in values.items():
            if param_name in ["lines", "component"]:
                continue
            medians = param_values["median"]
            err_minus = param_values.get("err_minus", [None]*len(medians))
            err_plus = param_values.get("err_plus", [None]*len(medians))

            for _, (line, comp, med, err_m, err_p) in enumerate(zip(lines, components, medians, err_minus, err_plus)):
                rows.append({
                    "line_name": line,
                    "component": comp,
                    "kind": kind,
                    "parameter": param_name,
                    "median": med,
                    "err_minus": err_m,
                    "err_plus": err_p
                })
    return pd.DataFrame(rows)

def flatten_scalar_dict(name, scalar_dict):
    """
    Flatten a scalar-valued dictionary (e.g., L_bol/L_w summaries) into a DataFrame.

    Parameters
    ----------
    name : str
        Label for the quantity (e.g., ``"L_bol"`` or ``"L_w"``).
    scalar_dict : dict
        Mapping ``key -> {"median": scalar, "err_minus": scalar, "err_plus": scalar}``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``quantity``, ``wavelength_or_line``, ``median``, ``err_minus``, ``err_plus``.
    """
    rows = []
    for key, stats in scalar_dict.items():
        rows.append({
            "quantity": name,
            "wavelength_or_line": key,
            "median": stats["median"].item(),
            "err_minus": stats["err_minus"].item(),
            "err_plus": stats["err_plus"].item()
        })
    return pd.DataFrame(rows)


def flatten_mass_dict(masses):
    """
    Flatten a masses dictionary into a DataFrame.

    Parameters
    ----------
    masses : dict
        Mapping ``line -> {stat_name: {"median": scalar, "err_minus": scalar, "err_plus": scalar}}``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``line_name``, ``quantity``, ``median``, ``err_minus``, ``err_plus``.
    """
    rows = []
    for line, metrics in masses.items():
        #print(line)
        for stat_name, stats in metrics.items():
            rows.append({
                "line_name": line,
                "quantity": stat_name,
                "median": stats["median"].item(),
                "err_minus": stats["err_minus"].item(),
                "err_plus": stats["err_plus"].item()
            })
    return pd.DataFrame(rows)



def pivot_and_split(obj_names, result):
    """
    Two-pass approach:
      1) Normalize the tree once: replace uarray leaves with {'value': vals, 'error': errs}
         and plain arrays with {'median': arr, 'error': 0} when leading dim == N.
      2) Create per-object slices without calling unumpy again.
    """
    N = len(obj_names)
    _memo = {}

    def _normalize(node):
        oid = id(node)
        if oid in _memo:
            return _memo[oid]

        # dict -> recurse
        if isinstance(node, dict):
            out = {k: _normalize(v) for k, v in node.items()}

        # simple scalars/strings/None -> as-is
        elif isinstance(node, (str, int, float, type(None))):
            out = node

        # arrays of uncertainties (dtype=object) -> split once
        elif isinstance(node, np.ndarray) and node.dtype == object and node.size:
            # Try extracting; if it's not uarray-like, fall back as-is.
            try:
                vals = unumpy.nominal_values(node)
                errs = unumpy.std_devs(node)
                out = {'median': vals, 'error': errs}
            except Exception:
                out = node  # not an uncertainties array after all

        # numeric arrays whose first axis is N -> batch leaf
        elif isinstance(node, np.ndarray) and node.ndim >= 1 and node.shape[0] == N:
            out = {'median': node, 'error': 0}  # keep a scalar 0 to avoid huge zero arrays

        # lists/tuples -> recurse elementwise (kept shape)
        elif isinstance(node, (list, tuple)):
            seq = [_normalize(x) for x in node]
            out = tuple(seq) if isinstance(node, tuple) else seq

        else:
            out = node

        _memo[oid] = out
        return out

    normalized = _normalize(result)

    def _slice_idx(node, idx):
        # If this is a normalized leaf with 'value'/'error', slice only if the first dim is N
        if isinstance(node, dict) and 'median' in node and 'error' in node:
            v, e = node['median'], node['error']
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == N:
                # e can be 0 (scalar) or an array aligned with v
                ei = (e[idx].squeeze() if isinstance(e, np.ndarray) and
                      e.ndim >= 1 and e.shape[0] == N else e)
                return {'median': v[idx].squeeze(), 'error': ei}
            # not a batch leaf → recurse normally below

        if isinstance(node, dict):
            return {k: _slice_idx(v, idx) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            seq = [_slice_idx(x, idx) for x in node]
            return tuple(seq) if isinstance(node, tuple) else seq
        return node

    return {name: _slice_idx(normalized, i) for i, name in enumerate(obj_names)}


def summarize_samples(samples) -> Dict[str, np.ndarray]:
    """
    Summarize a sample vector by 16th/50th/84th percentiles.

    Parameters
    ----------
    samples : array-like
        1D vector of draws, or 2D array where each column is a separate variable.
        JAX arrays are accepted and converted to NumPy for percentile computation.

    Returns
    -------
    dict
        ``{"median": ..., "err_minus": ..., "err_plus": ...}``.

    Notes
    -----
    - If more than 20% of entries are NaN, a warning is emitted and
      percentiles are computed with ``np.nanpercentile``.
    - For 1D input, returns scalars; for 2D (n, m), returns length-m arrays.
    """
    if isinstance(samples, jnp.ndarray):
        samples = np.asarray(samples)
    samples = np.atleast_2d(samples).T
    #print(type(samples))
    if np.isnan(samples).sum() / samples.size > 0.2:
        warnings.warn("High fraction of NaNs; uncertainty estimates may be biased.")
    if samples.shape[1]<=1:
        q = np.nanpercentile(samples, [16, 50, 84], axis=0)
    else:
        q = np.nanpercentile(samples, [16, 50, 84], axis=1)
    #else:
    
    return {
        "median": q[1],
        "err_minus": q[1] - q[0],
        "err_plus": q[2] - q[1]
    }
    
    
def summarize_nested_samples(d: dict,run_summarize:bool = True) -> dict:
    """
    Recursively apply :func:`summarize_samples` to array-like leaves.

    Parameters
    ----------
    d : dict
        Nested dictionary whose leaves may be arrays to summarize.
    run_summarize : bool, default True
        If False, returns ``d`` unchanged.

    Returns
    -------
    dict
        Same structure as input with arrays replaced by percentile summaries.

    Notes
    -----
    - Keys named ``"component"`` are passed through untouched (often categorical).
    - JAX arrays are accepted; they are converted to NumPy within
        :func:`summarize_samples`.
    """
    if not run_summarize:
        return d
    summarized = {}
    for k, v in d.items():
        if isinstance(v, dict):
            summarized[k] = summarize_nested_samples(v)
        elif isinstance(v, (np.ndarray, jnp.ndarray)) and np.ndim(v) >= 1 and k!='component':
            summarized[k] = summarize_samples(v)
        else:
            summarized[k] = v
    return summarized
