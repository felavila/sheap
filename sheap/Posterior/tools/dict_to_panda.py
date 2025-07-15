from typing import Dict, Any


import pandas as pd

def flatten_mass_samples_to_df(dict_samples: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract 'masses' from nested sample dictionaries and return a flat pandas DataFrame.
    
    Parameters:
    - dict_samples: Dictionary of objects, each containing a 'masses' dictionary.

    Returns:
    - A pandas DataFrame with columns: object, line, quantity, median, err_minus, err_plus
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
