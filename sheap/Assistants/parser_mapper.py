"""This module contains all the mappers and parsers."""

__author__ = 'felavila'

__all__ = [
    "apply_arithmetic_ties",
    "apply_tied_and_fixed_params",
    "build_tied",
    "descale_amp",
    "extract_float",
    "flatten_tied_map",
    "make_get_param_coord_value",
    "mapping_params",
    "parse_dependencies",
    "parse_dependency",
    "project_params",
    "project_params_clasic",
    "scale_amp",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial
import re 

import numpy as np 
import jax.numpy as jnp
from jax import jit


#TODO this is full of repeated or functions that can be simplified.
#_, target, source, op, operand = dep
# (target, source, op, operand)


def extract_float(s: str) -> float:
            # Extract the first number in the string (supports +, -, and decimal points)
            match = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            if match:
                return float(match.group())
            else:
                raise ValueError(f"No numeric value found in: {s}")

def mapping_params(params_dict, params, verbose=False):
    """
    Identify indices in the parameter dictionary that match the given name patterns.

    Parameters
    ----------
    params_dict : dict or np.ndarray
        Dictionary mapping parameter names to indices.
    params : str or list of str or list of list of str
        Parameter name patterns to match.
    verbose : bool, optional
        If True, print matching parameter names.

    Returns
    -------
    jnp.ndarray
        Array of unique matching indices.
    """
    if isinstance(params_dict, np.ndarray):
        params_dict = {str(key): n for n, key in enumerate(params_dict)}
    if isinstance(params, str):
        params = [params]
    match_list = []
    for param in params:
        if isinstance(param, str):
            param = [param]
        match_list += [
            params_dict[key] for key in params_dict.keys() if all([p in key for p in param])
        ]
    match_list = jnp.array(match_list)
    unique_arr = jnp.unique(match_list)
    if verbose:
        print(np.array(list(params_dict.keys()))[unique_arr])
    return unique_arr


def scale_amp(params_dict, params, scale):
    """
    Scale amplitude and log-amplitude parameters by a multiplicative factor.

    Parameters
    ----------
    params_dict : dict
        Dictionary mapping parameter names to indices.
    params : jnp.ndarray
        Parameter array of shape (N, D).
    scale : jnp.ndarray
        Scale values of shape (N,).

    Returns
    -------
    jnp.ndarray
        Scaled parameter array.
    """
    idxs = mapping_params(params_dict, [["amplitude"]])
    idxs_log = mapping_params(params_dict, [["logamp"]])
    params = (params.at[:, idxs].multiply(scale).at[:, idxs_log].add(jnp.log10(scale)))
    return params


def descale_amp(params_dict, params, scale):
    """
    Reverse amplitude scaling on both amplitude and log-amplitude parameters.

    Parameters
    ----------
    params_dict : dict
        Dictionary mapping parameter names to indices.
    params : jnp.ndarray
        Parameter array of shape (N, D).
    scale : jnp.ndarray
        Scale values of shape (N,).

    Returns
    -------
    jnp.ndarray
        Descaled parameter array.
    """
    idxs = mapping_params(params_dict, [["amplitude"]])
    idxs_log = mapping_params(params_dict, [["logamp"]])
    params = (params.at[:, idxs].divide(scale).at[:, idxs_log].subtract(jnp.log10(scale)))
    return params


@jit
def project_params_clasic(params: jnp.ndarray, constraints: jnp.ndarray) -> jnp.ndarray:
    """
    Project flat parameters to satisfy individual min/max constraints.

    Parameters
    ----------
    params : jnp.ndarray
        Parameter vector.
    constraints : jnp.ndarray
        Constraint array of shape (N, 2) with lower and upper bounds.

    Returns
    -------
    jnp.ndarray
        Projected parameters within bounds.
    """
    lower_bounds = constraints[:, 0]
    upper_bounds = constraints[:, 1]
    return jnp.clip(params, lower_bounds, upper_bounds)


def parse_dependency(dep_str: str):
    """
    Parse a single dependency string into structured format.

    Supported formats
    -----------------
    - Arithmetic: "target source *2"
    - Inequality: "target source <"
    - Range: "target in [lower,upper]"

    Parameters
    ----------
    dep_str : str
        A dependency string.

    Returns
    -------
    tuple
        Parsed representation of the dependency.
    """
    tokens = dep_str.split()
    if len(tokens) == 3:
        if tokens[1] == "in":
            target = int(tokens[0])
            range_str = tokens[2]
            if range_str.startswith("[") and range_str.endswith("]"):
                lower_str, upper_str = range_str[1:-1].split(",")
                return ("range_literal", target, float(lower_str), float(upper_str))
            else:
                raise ValueError(f"Invalid range specification: {dep_str}")
        else:
            try:
                _ = int(tokens[2])
                return ("range_between", int(tokens[0]), int(tokens[1]), int(tokens[2]))
            except ValueError:
                target, source = int(tokens[0]), int(tokens[1])
                op_token = tokens[2]
                if op_token in {"<", ">"}:
                    return ("inequality", target, source, op_token, None)
                op = op_token[0]
                operand = float(op_token[1:])
                return ("arithmetic", target, source, op, operand)
    elif len(tokens) == 4 and tokens[1] == "in":
        target = int(tokens[0])
        range_str = (tokens[2] + " " + tokens[3]).strip()
        if range_str.startswith("[") and range_str.endswith("]"):
            lower_str, upper_str = range_str[1:-1].split(",")
            return ("range_literal", target, float(lower_str), float(upper_str))
        else:
            raise ValueError(f"Invalid range specification: {dep_str}")
    raise ValueError(f"Invalid dependency format: {dep_str}")


def parse_dependencies(dependencies: list[str]):
    """
    Parse a list of dependency strings into structured constraints.

    Parameters
    ----------
    dependencies : list of str
        List of dependency definitions.

    Returns
    -------
    tuple
        Parsed dependencies.
    """
    return tuple(parse_dependency(dep) for dep in dependencies)


@partial(jit, static_argnums=(2,))
def project_params(
    params: jnp.ndarray,
    constraints: jnp.ndarray,
    parsed_dependencies: Optional[List[Tuple]] = None,
) -> jnp.ndarray:
    """
    Project parameters to satisfy individual bounds and inter-parameter constraints.

    Parameters
    ----------
    params : jnp.ndarray
        Flat parameter vector.
    constraints : jnp.ndarray
        Array of shape (N, 2) with lower and upper bounds.
    parsed_dependencies : list of tuple, optional
        Output of `parse_dependencies`.

    Returns
    -------
    jnp.ndarray
        Projected parameter vector.
    """
    params = jnp.clip(params, constraints[:, 0], constraints[:, 1])
    epsilon = 1e-6
    if parsed_dependencies is not None:
        for dep in parsed_dependencies:
            dep_type = dep[0]
            if dep_type == "arithmetic":
                _, tgt, src, op, val = dep
                if op == "*":
                    new_val = params[src] * val
                elif op == "/":
                    new_val = params[src] / val
                elif op == "+":
                    new_val = params[src] + val
                elif op == "-":
                    new_val = params[src] - val
                params = params.at[tgt].set(new_val)
            elif dep_type == "inequality":
                _, tgt, src, op, _ = dep
                if op == "<":
                    new_val = jnp.where(params[tgt] < params[src], params[tgt], params[src] - epsilon)
                else:
                    new_val = jnp.where(params[tgt] > params[src], params[tgt], params[src] + epsilon)
                params = params.at[tgt].set(new_val)
            elif dep_type == "range_literal":
                _, tgt, lo, hi = dep
                params = params.at[tgt].set(jnp.clip(params[tgt], lo, hi))
            elif dep_type == "range_between":
                _, tgt, lo_idx, hi_idx = dep
                params = params.at[tgt].set(jnp.clip(params[tgt], params[lo_idx], params[hi_idx]))
    return params


def make_get_param_coord_value(
    params_dict: Dict[str, int], initial_params: jnp.ndarray
) -> Callable[[str, str, Union[str, int], str, bool], Tuple[int, float, str]]:
    """
    Generate a function to retrieve the index and value of a parameter by key components.

    Parameters
    ----------
    params_dict : dict
        Mapping from parameter key to index.
    initial_params : jnp.ndarray
        Array of parameter values.

    Returns
    -------
    callable
        Function to extract (index, value, param_name).
    """
    def get_param_coord_value(
        param: str,
        line_name: str,
        component: Union[str, int],
        region: str,
        verbose: bool = False,
    ) -> Tuple[int, float, str]:
        if param == "amplitude":
            param = "logamp" #this is assuming all the profiles in sheap use logamp but what happen in the cases where this doesn't happen :c
        key = f"{param}_{line_name}_{component}_{region}"
        pos = params_dict.get(key)
        if pos is None:
            raise KeyError(f"Key '{key}' not found in params_dict.")
        if verbose:
            print(f"{key}: value = {initial_params[pos]}")
        return pos, float(initial_params[pos]), param

    return get_param_coord_value


def apply_arithmetic_ties(samples: jnp.ndarray, ties: Tuple) -> jnp.ndarray:
    """
    Apply arithmetic constraints to parameter vector.

    Parameters
    ----------
    samples : jnp.ndarray
        Parameter values.
    ties : tuple
        Arithmetic tie specification.

    Returns
    -------
    jnp.ndarray
        Updated value for the tied parameter.
    """
    _, target_idx, src_idx, op, val = ties
    src = samples[src_idx]
    if op == '+':
        return src + val
    elif op == '-':
        return src - val
    elif op == '*':
        return src * val
    elif op == '/':
        return src / val
    else:
        raise ValueError(f"Unsupported operation: {op}")


def apply_tied_and_fixed_params(
    free_params: jnp.ndarray,
    template_params: jnp.ndarray,
    dependencies: List[Tuple],
) -> jnp.ndarray:
    """
    Insert tied parameters into the full parameter vector using a template.

    Parameters
    ----------
    free_params : jnp.ndarray
        Vector of free (optimized) parameters.
    template_params : jnp.ndarray
        Template full-length parameter vector.
    dependencies : list of tuple
        Structured arithmetic ties.

    Returns
    -------
    jnp.ndarray
        Full parameter vector including tied values.
    """
    if not dependencies:
        return free_params
    idx_target = [i[1] for i in dependencies]
    idx_free_params = list(set(range(len(template_params))) - set(idx_target))
    template_params = template_params.at[jnp.array(idx_free_params)].set(free_params)
    template_params = template_params.at[jnp.array(idx_target)].set(
        [apply_arithmetic_ties(template_params, tie) for tie in dependencies]
    )
    return template_params


def build_tied(tied_params,get_param_coord_value):
    list_tied_params = []
    if len(tied_params) > 0:
        for tied in tied_params:
            param1, param2 = tied[:2]
            pos_param1, val_param1, param_1 = get_param_coord_value(*param1.split("_"))    
            pos_param2, val_param2, param_2 = get_param_coord_value(*param2.split("_"))
            if len(tied) == 2:
                if param_1 == param_2 == "center" and len(tied):
                    delta = val_param1 - val_param2
                    tied_val = "+" + str(delta) if delta > 0 else "-" + str(abs(delta))
                elif param_1 == param_2:
                    tied_val = "*1"
                else:
                    print(f"Define constraints properly. {tied_params}") #add how to writte the constrains properly 
            else:
                tied_val = tied[-1]
                if param_1 == param_2 == "logamp":
                    tied_val = f"{np.log10(extract_float(tied_val))}"
                    #print(tied_val)
                if isinstance(tied_val, str):
                    list_tied_params.append(f"{pos_param1} {pos_param2} {tied_val}")
                else:
                    print("Define constraints properly.")
        else:
            list_tied_params = []
    return list_tied_params
        #print("Remember move this functions to Assistants and also change it in Montecarlo.")
        
def flatten_tied_map(tied_map: dict[int, tuple[int, str, float]]) -> dict[int, tuple[int, str, float]]:
    """
    Resolve all ties in `tied_map` to point only to free (non-tied) sources.
    
    Parameters
    ----------
    tied_map : dict
        Maps target index -> (source index, operator, operand)

    Returns
    -------
    dict
        Flattened map: target index -> (free source index, operator, operand)
    """
    def resolve(idx, visited=None):
        if visited is None:
            visited = set()
        if idx in visited:
            raise ValueError(f"Circular dependency detected at index {idx}")
        visited.add(idx)

        src, op, operand = tied_map[idx]
        if src not in tied_map:
            return src, op, operand  # base case

        # resolve further back
        src2, op2, operand2 = resolve(src, visited)

        # Combine ops
        if op == '*' and op2 == '*':
            combined_op, combined_operand = '*', operand * operand2
        elif op == '*' and op2 == '+':
            combined_op, combined_operand = '*', operand  # can't combine "* followed by +"
        elif op == '+' and op2 == '+':
            combined_op, combined_operand = '+', operand + operand2
        elif op == '+' and op2 == '-':
            combined_op, combined_operand = '+', operand - operand2
        elif op == '-' and op2 == '+':
            combined_op, combined_operand = '-', operand + operand2
        elif op == '-' and op2 == '-':
            combined_op, combined_operand = '-', operand - operand2
        elif op == '*' and op2 == '-':
            combined_op, combined_operand = '*', operand * -1 * operand2
        else:
            raise NotImplementedError(f"Cannot combine {op2} followed by {op}")

        return src2, combined_op, combined_operand

    result = {}
    for tgt in tied_map:
        result[tgt] = resolve(tgt)
    return result
#####################################

#list_dependencies = parse_dependencies(self._build_tied(tied))

#                 #print("obj_name",list(self.params_dict.keys())[idx],"src_name",list(self.params_dict.keys())[src_idx])
            
#                 tie = (name, src_name, op, operand)
#                 print(tie)
#                 
            
#             else:
#                 val = self.initial_params[idx]
#                 min,max = self.constraints[idx]
#                 params_obj.add(name, val, min=min, max=max)
# from sheap.Assistants.parser_mapper import parse_dependencies
        
#         list_dependencies = parse_dependencies(self._build_tied(tied))
#         tied_map = {T[1]: T[2:] for  T in list_dependencies}
        
#         #print(tied_map)
#         tied_map = flatten_tied_map(tied_map)
#         #print()
#         params_obj = Parameters()
#         #(target, source, op, operand)
#         for name, idx in self.params_dict.items():
#             #print(name, idx)
#             val = initial_params[:,idx]
#             min,max = self.constraints[idx]
#             if name in ["amplitude_slope_linear_0_continuum","amplitude_intercept_linear_0_continuum"] and iteration_number==10:
#                 params_obj.add(name, val, fixed=True)
#             elif idx in tied_map.keys():
#                 src_idx, op, operand = tied_map[idx]
#                 #print("obj_name",list(self.params_dict.keys())[idx],"src_name",list(self.params_dict.keys())[src_idx])
#                 src_name = list(self.params_dict.keys())[src_idx]
#                 tie = (name, src_name, op, operand)
#                 print(tie)
#                 params_obj.add(name, val, min=min, max=max, tie=tie)
            
#             else:
#                 val = self.initial_params[idx]
#                 min,max = self.constraints[idx]
#                 params_obj.add(name, val, min=min, max=max)


