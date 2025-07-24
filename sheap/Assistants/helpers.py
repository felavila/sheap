from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np 
import jax.numpy as jnp


#_, target, source, op, operand = dep

def mapping_params(params_dict, params, verbose=False):
    """
    params_dict: full params dict
    params: obj to select
    """
    if isinstance(params_dict, np.ndarray):
        params_dict = {str(key): n for n, key in enumerate(params_dict)}
    if isinstance(params, str):
        params = [params]
    match_list = []
    for param in params:
        if isinstance(param, str):
            param = [param]
        # print(params_dict.keys())
        # print([[params_dict[key],key] for key in params_dict.keys() if all([p in key for p in param])])

        match_list += [
            params_dict[key] for key in params_dict.keys() if all([p in key for p in param])
        ]

    match_list = jnp.array(match_list)
    unique_arr = jnp.unique(match_list)
    if verbose:
        print(np.array(list(params_dict.keys()))[unique_arr])  # [])
    return unique_arr

#scale[:, None]
def scale_amp(params_dict,params,scale):
    idxs = mapping_params(params_dict, [["amplitude"]]) #, ["scale"]
    idxs_log = mapping_params(params_dict, [["logamp"]])
    params = (params.at[:, idxs].multiply(scale).at[:, idxs_log].add(jnp.log10(scale)))
    return params

def descale_amp(params_dict,params,scale):
    
    idxs = mapping_params(params_dict, [["amplitude"]]) #, ["scale"]
    idxs_log = mapping_params(params_dict, [["logamp"]])
    params = (params.at[:, idxs].divide(scale).at[:, idxs_log].subtract(jnp.log10(scale)))
    return params



# def _build_tied(get_param_coord_value,tied_params):
#         """this is a copy from RegionFtitting given both do the same maybe is moment to think about make this its own function
#         Arithmetic: "source target  *2"  (target = source * 2)
#         """""
#         list_tied_params = []
#         if len(tied_params) > 0:
#             for tied in tied_params:
#                 param1, param2 = tied[:2]
#                 pos_param1, val_param1, param_1 = get_param_coord_value(
#                     *param1.split("_")
#                 )
#                 pos_param2, val_param2, param_2 = get_param_coord_value(
#                     *param2.split("_")
#                 )
#                 if len(tied) == 2:
#                     if param_1 == param_2 == "center" and len(tied):
#                         delta = val_param1 - val_param2
#                         tied_val = "+" + str(delta) if delta > 0 else "-" + str(abs(delta))
#                         # if log_mode:
#                     elif param_1 == param_2:
#                         tied_val = "*1"
#                     else:
#                         print(f"Define constraints properly. {tied_params}")
#                 else:
#                     tied_val = tied[-1]
#                 if isinstance(tied_val, str):
#                     list_tied_params.append(f"{pos_param1} {pos_param2} {tied_val}")
#                 else:
#                     print("Define constraints properly.")
#         else:
#             list_tied_params = []
#         return list_tied_params


# def apply_arithmetic_ties(params: Dict[str, float], ties: List[Tuple]) -> Dict[str, float]:
#     for tag, src_idx, target_idx, op, val in ties:
#         src = params[f"theta_{src_idx}"]
#         if op == '+':
#             result = src + val
#         elif op == '-':
#             result = src - val
#         elif op == '*':
#             result = src * val
#         elif op == '/':
#             result = src / val
#         else:
#             raise ValueError(f"Unsupported operation: {op}")
#         params[f"theta_{target_idx}"] = result
#     return params

# def apply_arithmetic_ties_restore(samples, ties):
#     tag, src_idx, target_idx, op, val = ties
#     src = samples[f"theta_{src_idx}"]
#     if op == '-':
#         result = src + val
#     elif op == '+':
#         result = src - val
#     elif op == '/':
#         result = src * val
#     elif op == '*':
#         result = src / val
#     else:
#         raise ValueError(f"Unsupported operation: {op}")
#         #params[f"theta_{target_idx}"] = result
#     return result