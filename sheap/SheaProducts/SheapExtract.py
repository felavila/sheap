from collections.abc import Mapping

import numpy as np
import pandas as pd

from sheap.Utils.Paper import median_with_errors
from sheap.SheaProducts.Utils.MoldelSpectraReconstruction import MoldelSpectraReconstruction


def _extract_extra_params(
    n_obj,
    obj_name,
    available_extra_params,
    values,
    low=0.16,
    high=0.84,
):
    rows = []

    def _pick_indexed_value(arr_like, idx, n_expected, default=None):
        """
        Pick one value from metadata like component/combined.

        Cases:
        - size == n_expected: use arr[idx]
        - size == 1: use scalar value
        - otherwise: use default
        """
        arr = np.asarray(arr_like).squeeze()

        if arr.size == 0:
            return default

        if arr.size == n_expected:
            out = arr.ravel()[idx]
            out_arr = np.asarray(out)
            return out_arr.item() if out_arr.size == 1 else out

        if arr.size == 1:
            return arr.item()

        return default

    for extra_key in available_extra_params:
        extra = values[extra_key]

        for line, line_dict in extra.items():
            for combo, combo_dict in line_dict.items():

                meta = {}
                quantities = {}

                # ---------------------------------
                # Separate metadata from quantities
                # ---------------------------------
                for key, val in combo_dict.items():

                    if key in ["component", "combined"]:
                        meta[key] = val

                    elif isinstance(val, Mapping) and "median" in val:
                        quantities[key] = ("stats_dict", val)

                    elif isinstance(val, (np.ndarray, list, tuple)):
                        arr = np.asarray(val)

                        if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                            quantities[key] = ("samples", arr)
                        else:
                            meta[key] = val

                    else:
                        meta[key] = val

                # ---------------------------------
                # Build rows
                # ---------------------------------
                for quantity_name, (qkind, payload) in quantities.items():

                    base_row = {
                        "n_obj": n_obj,
                        "name": obj_name,
                        "line": line,
                        "SMBHEstimator": combo,
                        "quantity_name": quantity_name,
                        "extra_key": extra_key,
                    }

                    # Add metadata except per-component metadata
                    for m_key, m_val in meta.items():

                        if m_key in ["component", "combined"]:
                            continue

                        elif isinstance(m_val, (np.ndarray, list, tuple)):
                            arr = np.asarray(m_val)
                            base_row[m_key] = arr.item() if arr.size == 1 else m_val

                        else:
                            base_row[m_key] = m_val

                    # ---------------------------------
                    # Case 1: already computed stats
                    # ---------------------------------
                    if qkind == "stats_dict":
                        row = base_row.copy()

                        for stat_name, stat_val in payload.items():
                            arr = np.asarray(stat_val).squeeze()
                            row[stat_name] = arr.item() if arr.size == 1 else arr

                        # Add component / combined if scalar metadata exists
                        if "component" in meta:
                            components = np.asarray(meta["component"]).squeeze()
                            if components.size == 1:
                                row["component"] = components.item()
                            else:
                                row["component"] = components

                        if "combined" in meta:
                            combined = np.asarray(meta["combined"]).squeeze()
                            if combined.size == 1:
                                row["combined"] = combined.item()
                            else:
                                row["combined"] = combined

                        rows.append(row)

                    # ---------------------------------
                    # Case 2: samples, compute stats
                    # ---------------------------------
                    elif qkind == "samples":
                        samples = np.asarray(payload, dtype=float)

                        # Expected shape: (nsamples, n_components)
                        if samples.ndim == 1:
                            samples = samples[:, None]

                        components = np.asarray(meta.get("component", []))
                        combined = np.asarray(meta.get("combined", []))

                        n_components = samples.shape[1]

                        for y, x in enumerate(samples.T):

                            # New independent row for each component
                            row = base_row.copy()

                            med, em, ep = median_with_errors(x,low=low,high=high,)
                            #print(components[y])
                            row["component"] = components[y]
                            
                            row["combined"] = _pick_indexed_value(
                                combined,
                                idx=y,
                                n_expected=n_components,
                                default=None,
                            )

                            row["median"] = med
                            row["err_minus"] = em
                            row["err_plus"] = ep
                            row["nsamp"] = int(x.size)

                            rows.append(row)

    return rows

def _extract_continuum_params(n_obj, obj_name, available_others, values, low=0.16, high=0.84,):
    rows = []
    for k in available_others:
        dict_w = values[k]
        for w, values_w in dict_w.items():
            row = {}
            samples = values_w
            med, em, ep = median_with_errors(samples, low=low, high=high)
            row["median"] = med
            row["err_minus"] = em
            row["err_plus"] = ep
            row["wavelenght"] = w
            row["quantity"] = k
            row["obj_name"] = obj_name
            row["n_obj"] = n_obj
            rows.append(row)
    return rows


def _extract_basic_params(n_obj, obj_name, available_basic_params, values, low=0.16, high=0.84,):
    rows = []

    for basic_param in available_basic_params:

        values_k = values[basic_param]

        for region_name, inner_line_region in values_k.items():
            meta = {}
            quantities = {}
            for key, val in inner_line_region.items():

                if isinstance(val, Mapping) and "median" in val:
                    quantities[key] = ("stats_dict", val)

                elif isinstance(val, (np.ndarray, list, tuple)):
                    arr = np.asarray(val)

                    if (arr.size > 0 and np.issubdtype(arr.dtype, np.number) and key not in ["component", "lines"]
                    ):
                        quantities[key] = ("samples", arr)
                    else:
                        meta[key] = val

                else:
                    meta[key] = val

            lines = np.asarray(meta["lines"])
            components = np.asarray(meta["component"])

            n_lines = len(lines)

            for quantity_name, (qkind, payload) in quantities.items():
                if quantity_name==["shape_params"]:
                    #we will drope this for now 
                    continue
                if qkind == "stats_dict":

                    stats = {
                        stat_name: np.asarray(stat_val).squeeze()
                        for stat_name, stat_val in payload.items()
                    }

                    for i in range(n_lines):

                        row = {
                            "n_obj": n_obj,
                            "name": obj_name,
                            "region": region_name,
                            "basic_param": basic_param,
                            "quantity_name": quantity_name,
                            "lines": lines[i],
                            "component": components[i],
                        }

                        # Add metadata
                        for m_key, m_val in meta.items():

                            if m_key in ["lines", "component"]:
                                continue

                            arr = np.asarray(m_val)

                            if arr.ndim == 0:
                                row[m_key] = arr.item()

                            elif len(arr) == n_lines:
                                row[m_key] = arr[i]

                            else:
                                row[m_key] = m_val

                        # Add stats
                        for stat_name, stat_val in stats.items():

                            arr = np.asarray(stat_val)

                            if arr.ndim == 0:
                                row[stat_name] = arr.item()

                            elif len(arr) == n_lines:
                                row[stat_name] = arr[i]

                            else:
                                row[stat_name] = arr

                        rows.append(row)

                else:

                    samples = np.asarray(payload)

                    med, em, ep = median_with_errors(samples, low=low,high=high,axis=0,)

                    med = np.asarray(med).squeeze()
                    em = np.asarray(em).squeeze()
                    ep = np.asarray(ep).squeeze()

                    for i in range(n_lines):

                        row = {
                            "n_obj": n_obj,
                            "name": obj_name,
                            "region": region_name,
                            "basic_param": basic_param,
                            "quantity_name": quantity_name,
                            "line": lines[i],
                            "component": components[i],
                            "median": med[i] if med.ndim > 0 else med.item(),
                            "err_minus": em[i] if em.ndim > 0 else em.item(),
                            "err_plus": ep[i] if ep.ndim > 0 else ep.item(),
                            "nsamp": samples.shape[0],
                        }

                        # Add metadata
                        for m_key, m_val in meta.items():

                            if m_key in ["lines", "component"]:
                                continue

                            arr = np.asarray(m_val)

                            if arr.ndim == 0:
                                row[m_key] = arr.item()

                            elif len(arr) == n_lines:
                                row[m_key] = arr[i]

                            else:
                                row[m_key] = m_val

                        rows.append(row)
    return rows

def posterior_param_extraction(sheapspectral, low=0.16, high=0.84, method="montecarlo",selected_index = [],calculate_host=True):
    #TODO next update should put this inside param extraction-combined with Fe ? 
    #TODO selected n_index go for name is to confuse.
    posterior = sheapspectral.result.posterior[method]["posterior_result"]
    rows_extra = []
    rows_cont = []
    rows_basic = []
    obj_list = []
    if len(selected_index) == 0:
        selected_index = np.arange(len(sheapspectral.names))
    for n_obj, (obj_name, values) in enumerate(posterior.items()):
        if n_obj not in selected_index:
            continue
        obj_list.append(obj_name)
        keys = list(set(values.keys()) - {"distances", "samples_phys"})

        available_extra_params = [k for k in keys if "extra" in k]

        available_basic_params = [k for k in keys if "basic" in k]

        available_others = list(set(keys) - set(available_extra_params) - set(available_basic_params))

        rows_extra.extend(_extract_extra_params(n_obj=n_obj, obj_name=obj_name, available_extra_params=available_extra_params, values=values, low=low, high=high,))
        rows_cont.extend(_extract_continuum_params(n_obj=n_obj, obj_name=obj_name, available_others=available_others, values=values, low=low, high=high,))
        rows_basic.extend(_extract_basic_params(n_obj=n_obj, obj_name=obj_name, available_basic_params=available_basic_params, values=values, low=low, high=high,))
    
    df_extra = pd.DataFrame(rows_extra)
    df_cont = pd.DataFrame(rows_cont)
    df_basic = pd.DataFrame(rows_basic)
    if np.any(["host" in  line.line_name for line in sheapspectral.result.region_list]) and calculate_host:
        print("----Running host reconstruction-----")
        ra = MoldelSpectraReconstruction(sheapspectral, jit_compile=True,posterior_group=method)
        stars = ra.stars_Cont_5100(all_samples = selected_index)
        if len(stars.shape) != 2:
            stars = stars[:,None]
        med, _low, _up= median_with_errors(stars,axis=1, low=low, high=high)
        row = pd.DataFrame({"median":med,"err_minus":_low,"err_plus":_up,"obj_name": obj_list,"wavelenght":[5100]*len(selected_index),
                            "quantity":["cont_ratio"]*len(selected_index),"n_obj":selected_index})
        df_cont=pd.concat([df_cont, row], ignore_index=True)
    
    df_chi = pd.DataFrame({"obj_name":obj_list,"n_obj":selected_index,
                           "chi_2_reduced":sheapspectral.result.chi2_red[*selected_index],"snr":sheapspectral.snr[*selected_index],"z":sheapspectral.z[*selected_index]})

    if df_extra.empty:
        return df_extra

    non_numeric = {"n_obj", "name","line", "SMBHEstimator", "quantity", "method", "vwidth_def", "component", "extra_key",}

    # for col in df_extra.columns:
    #     if col not in non_numeric:
    #         df_extra[col] = pd.to_numeric(df_extra[col], errors="ignore")
    # for col in df_cont.columns:
    #     if col not in non_numeric:
    #         df_cont[col] = pd.to_numeric(df_cont[col], errors="ignore")
    # for col in df_basic.columns:
    #     if col not in non_numeric:
    #         df_basic[col] = pd.to_numeric(df_basic[col], errors="ignore")

    return df_extra,df_cont,df_basic,df_chi