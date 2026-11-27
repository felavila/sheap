import numpy as np
from collections.abc import Mapping
import numbers
import pandas as pd 
from joblib import Parallel, delayed

from sheap.Utils.Paper import median_with_errors
from sheap.SheaProducts.Utils.MoldelSpectraReconstruction import MoldelSpectraReconstruction
from sheap.SheaProducts.Utils.ExtractExtraParams import process_luminosity,process_smbh

def _extract_extra_params(n_obj, obj_name, available_extra_params, values, low=0.16, high=0.84,):
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
                        "obj_name": obj_name,
                        "line": line,
                        "SMBHEstimator": combo,
                        "quantity": quantity_name,
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

            lines = np.asarray(meta.get("lines"))
            components = np.asarray(meta.get("component"))

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
                            "obj_name": obj_name,
                            "region": region_name,
                            "basic_param": basic_param,
                            "quantity": quantity_name,
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
                            "obj_name": obj_name,
                            "region": region_name,
                            "basic_param": basic_param,
                            "quantity": quantity_name,
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




def quantities_to_wide(dataframe, index_columns=("obj_name", "n_obj"), value_columns=("median", "err_minus", "err_plus"), quantity_column="quantity",
    wavelength_column="wavelenght", include_wavelength=False, label_columns=("line", "region", "component"), estimator_column="SMBHEstimator"):
    """
    Convert basic-parameter or extra-parameter results from long to
    wide format.

    Column names can include:

    - statistic: median, err_minus, err_plus
    - quantity
    - wavelength, when available
    - line
    - region
    - component
    - combined_classical or combined_kinematic
    - SMBHEstimator, when available

    Examples
    --------
    median_flux_Halpha_broad_components_1_2_combined_classical

    err_plus_FWHM_Hbeta_narrow_component_0

    median_M_BH_Hbeta_broad_combined_kinematic_VP06
    """
    dataframe = dataframe.copy()

    if dataframe.empty:
        return dataframe

    # Automatically include wavelength only when the column exists.
    if include_wavelength is None:
        include_wavelength = (wavelength_column in dataframe.columns)

    required_columns = (list(index_columns)+ list(value_columns)+ [quantity_column])

    if include_wavelength:
        required_columns.append(wavelength_column)

    missing_columns = [column for column in required_columns if column not in dataframe.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    available_label_columns = [column for column in label_columns if column in dataframe.columns]

    def is_missing(value):
        """Check missing values without failing on lists or arrays."""
        if value is None:
            return True

        if isinstance(value, (list, tuple, set, dict)):
            return False

        try:
            result = pd.isna(value)
            return bool(result)
        except (TypeError, ValueError):
            return False

    def clean_text(value):
        """Convert a value into a safe column-name string."""
        # value = str(value).strip()

        # value = re.sub(r"\s+", "_", value)
        # value = re.sub(
        #     r"[^A-Za-z0-9_.+-]",
        #     "_",
        #     value,
        # )
        # value = re.sub(r"_+", "_", value)

        return value
    
    def format_value(value):
        """Format scalar or array-like values."""
        if is_missing(value):
            return None

        # Convert NumPy arrays/scalars to Python objects.
        if (
            not isinstance(
                value,
                (str, list, tuple, set, dict),
            )
            and hasattr(value, "tolist")
        ):
            value = value.tolist()

        if isinstance(value, (list, tuple, set)):
            values = [
                format_value(item)
                for item in value
                if not is_missing(item)
            ]

            values = [
                item for item in values
                if item is not None
            ]

            return "_".join(values) if values else None

        if (
            isinstance(value, numbers.Real)
            and not isinstance(value, bool)
        ):
            numeric_value = float(value)

            if numeric_value.is_integer():
                return str(int(numeric_value))

            return f"{numeric_value:g}"

        return clean_text(value)

    def format_component(value):
        """Format scalar or combined component indices."""
        if is_missing(value):
            return None

        if (
            not isinstance(
                value,
                (str, list, tuple, set),
            )
            and hasattr(value, "tolist")
        ):
            value = value.tolist()

        if isinstance(value, (list, tuple, set)):
            components = [
                format_value(item)
                for item in value
                if not is_missing(item)
            ]

            components = [
                component
                for component in components
                if component is not None
            ]

            if not components:
                return None

            if len(components) == 1:
                return f"component_{components[0]}"

            return (
                "components_"
                + "_".join(components)
            )

        component = format_value(value)

        if component is None:
            return None

        return f"component_{component}"

    # ------------------------------------------------------------
    # Extract the combined classification.
    # ------------------------------------------------------------
    dataframe["_combined_label"] = pd.Series(
        pd.NA,
        index=dataframe.index,
        dtype="string",
    )

    for source_column in ("basic_param", "extra_key"):
        if source_column not in dataframe.columns:
            continue

        extracted = (
            dataframe[source_column]
            .astype("string")
            .str.extract(
                r"(combined_(?:classical|kinematic))",
                expand=False,
            )
        )

        dataframe["_combined_label"] = (
            dataframe["_combined_label"]
            .fillna(extracted)
        )

    # Only retain the classification for rows with combined=True.
    if "combined" in dataframe.columns:
        combined_mask = (
            dataframe["combined"]
            .astype("string")
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

        dataframe.loc[
            ~combined_mask,
            "_combined_label",
        ] = pd.NA

    # ------------------------------------------------------------
    # Construct the output label for every row.
    # ------------------------------------------------------------
    def build_quantity_label(row):
        parts = [format_value(row[quantity_column])]

        if include_wavelength:
            wavelength = format_value(
                row[wavelength_column]
            )

            if wavelength:
                parts.append(wavelength)

        for column in available_label_columns:
            if column == "component":
                value = format_component(row[column])
            else:
                value = format_value(row[column])

            if value:
                parts.append(value)

        combined_label = row["_combined_label"]

        if pd.notna(combined_label):
            parts.append(str(combined_label))

        # Add the SMBH estimator only to rows where it is defined.
        if estimator_column in dataframe.columns:
            estimator = format_value(
                row[estimator_column]
            )

            if (
                estimator
                and estimator.lower()
                not in {"none", "nan", "null"}
            ):
                parts.append(estimator)

        return "_".join(
            part for part in parts
            if part is not None
        )

    dataframe["_quantity_label"] = dataframe.apply(build_quantity_label, axis=1,)

    # ------------------------------------------------------------
    # Convert to wide format.
    # ------------------------------------------------------------
    wide = dataframe.pivot_table(index=list(index_columns), columns="_quantity_label",values=list(value_columns), aggfunc="first",)

    wide.columns = [f"{statistic}_{quantity_label}" for statistic, quantity_label in wide.columns]

    wide = wide.reset_index()
    wide.columns.name = None

    return wide


def TablExtractor(sheapspectral, low=0.16, high=0.84, method=None,selected_index = [],calculate_host=True,calculate_fe=True
                               ,continuum_wavelengths = [2500.0, 3000.0, 4200.0, 5100.0,6200.0,],fe_windows = {"FE_UV": (2250.0, 2650.0)
                               ,"FE_OPT": (4434.0, 4684.0),},calculate_lum=True,calculate_extraproducts=True,verbose=True):
    #TODO selected n_index go for name is to confuse.
    #TODO Parallel dosent work "great" with CUDA carefull
    if not method:
        method = list(sheapspectral.result.posterior.keys())[0]
        print(f"Default method is {method}")
    posterior = sheapspectral.result.posterior[method]["posterior_result"]
    rows_extra = []
    rows_cont = []
    rows_basic = []
    obj_list = []
    chi2_red = np.array(sheapspectral.result.chi2_red)
    snr = np.array(sheapspectral.snr)
    z = np.array(sheapspectral.z)
    if calculate_lum:
        if verbose:
            print("calculating luminosities")
        distances = sheapspectral._calcualte_d()
        lums = Parallel(n_jobs=-1)(delayed(process_luminosity)(d, xx) for d, xx in zip(distances, posterior.values()))
        posterior.update(zip(posterior.keys(), lums))
    if calculate_extraproducts:
        if verbose:
            print("calculating extra-params")
        mass = Parallel(n_jobs=-1)(delayed(process_smbh)(xx) for xx in posterior.values())
        posterior.update(zip(posterior.keys(), mass))

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
    df_cont["quantity"] = df_cont["quantity"]+"_"+df_cont["wavelenght"]
    df_cont.drop(columns=["wavelenght"], inplace=True)
    df_basic = pd.DataFrame(rows_basic)
    reconstruction = MoldelSpectraReconstruction(sheapspectral,jit_compile=True,posterior_group=method)
    if (np.any(["host" in line.line_name for line in sheapspectral.result.region_list]) and calculate_host):
        if verbose:
            print("---- Running host reconstruction -----")
        rows = []
        for wavelength in continuum_wavelengths:
            if verbose:
                print(f"Calculating host fraction at {wavelength:.0f} Å")

            stellar_fraction = reconstruction.stars_cont_ratio_parallel(wavelength=wavelength, object_batch_size=128, return_numpy=True,object_indices=selected_index)
            
            # Ensure shape: (n_objects, n_samples)
            stellar_fraction = np.asarray(stellar_fraction)
            stellar_fraction = stellar_fraction.reshape(len(selected_index), -1)
            #print(stellar_fraction.shape)
            med, _low, _up = median_with_errors(stellar_fraction,axis=1,low=low,high=high)

            row = pd.DataFrame({"median": med, "err_minus": _low, "err_plus": _up, "obj_name": obj_list,"quantity": f"cont_ratio_{int(wavelength)}", "n_obj": selected_index})

            rows.append(row)

        df_cont = pd.concat([df_cont, *rows],ignore_index=True)
    if (np.any(["fe" in line.line_name for line in sheapspectral.result.region_list])) and calculate_fe:
        if verbose:
            print("---- Running Fe EW reconstruction -----")

        rows = []

        for region, (x_min, x_max) in fe_windows.items():
            if verbose:
                print(f"Calculating {region}: {x_min:.0f}-{x_max:.0f} Å")

            # Equivalent width
            ew_result = reconstruction.fe_integrated_ew_parallel(x_min=x_min, x_max=x_max, n_grid=2000, region_name="fe", object_indices=selected_index, include_bestfit=False, object_batch_size=64,return_numpy=True,)

            # Integrated flux
            flux_result = reconstruction.fe_integrated_flux_parallel(x_min=x_min,x_max=x_max, n_grid=2000, region_name="fe", object_indices=selected_index, include_bestfit=False, object_batch_size=64, return_numpy=True,)

            # Store EW and flux in the same DataFrame
            for quantity, result in ((f"{region}_EW", ew_result), (f"{region}_FLUX", flux_result),):

                values = np.asarray(result["samples"])

                med, _low, _up = median_with_errors(values, axis=1,low=low,high=high,)

                rows.append(pd.DataFrame({"median": med, "err_minus": _low, "err_plus": _up,"obj_name": obj_list,"quantity": quantity,"n_obj": selected_index,}))

        df_cont = pd.concat([df_cont, *rows],ignore_index=True,)
    df_chi = pd.DataFrame({"obj_name":obj_list,"n_obj":selected_index,
                           "chi_2_reduced":chi2_red[selected_index],"snr":snr[selected_index],"z":z[selected_index]})

    return df_extra,df_cont,df_basic,df_chi

def combine_table_result(df_chi, df_cont, df_basic, df_extra, coords=None, merge_columns=("obj_name", "n_obj"),):
    """
    Combine chi-square results, coordinates, continuum quantities,
    basic quantities, and extra quantities into one DataFrame.

    Empty optional DataFrames are skipped with a warning.
    """
    merge_columns = list(merge_columns)
    df_cont_wide = quantities_to_wide(df_cont)
    df_extra_wide = quantities_to_wide(df_extra)
    df_basic_wide = quantities_to_wide(df_basic)

    # Preserve the object order appearing in df_chi.
    unique_objects = (df_chi[merge_columns].drop_duplicates().reset_index(drop=True))

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("`coords` must have shape (n_objects, 2).")

    if len(unique_objects) != len(coords):
        raise ValueError("The number of unique objects does not match the " f"number of coordinates: {len(unique_objects)} objects " f"and {len(coords)} coordinates.")

    coordinates_df = unique_objects.copy()
    coordinates_df[["ra", "dec"]] = coords

    # Start with chi-square results and coordinates.
    combined_df = (df_chi.copy().merge(coordinates_df,on=merge_columns,how="left", validate="many_to_one",))

    # Optional DataFrames to merge.
    dataframes_to_merge = [("df_cont", df_cont_wide, "_cont"), ("df_basic", df_basic_wide, "_basic"), ("df_extra", df_extra_wide, "_extra"),]

    for name, dataframe, suffix in dataframes_to_merge:
        if dataframe is None or len(dataframe) == 0:
            print(f"Warning: {name} is empty; skipping merge.")
            continue

        combined_df = combined_df.merge(dataframe,on=merge_columns, how="left", validate="many_to_one", suffixes=("", suffix),)
    return combined_df