"""
Mappers & Parsers
=================

Utility functions to (a) map parameter names to indices, (b) scale/unscale
amplitudes, and (c) parse and enforce inter‑parameter constraints (“ties”)
both during optimization and when reconstructing full parameter vectors.

Public API
----------
- :func:`mapping_params`:
    Resolve indices in ``params_dict`` that match name patterns.
- :func:`scale_amp` / :func:`descale_amp`:
    Apply / undo multiplicative flux scaling on amplitude-like params.
- :func:`parse_dependency` / :func:`parse_dependencies`:
    Parse dependency strings into structured constraints.
- :func:`project_params_clasic` / :func:`project_params`:
    Clip to bounds and enforce parsed dependencies (JAX‑JIT friendly).
- :func:`make_get_param_coord_value`:
    Build a helper that retrieves (index, value, name) for a param key.
- :func:`apply_arithmetic_ties` / :func:`apply_tied_and_fixed_params`:
    Compose free→full parameter vectors using arithmetic ties.
- :func:`build_tied`:
    Convert human‑readable ties to dependency strings with indices.
- :func:`flatten_tied_map`:
    Resolve chained ties so all targets depend on free sources directly.

Constraint String Grammar
-------------------------
Each dependency is written as a single string and later parsed to a tuple.
Supported forms (indices refer to positions in the *flat* parameter vector):

Arithmetic:
    ``"target source *k"`` → ``param[target] = param[source] * k``  
    ``"target source /k"`` → divide;  
    ``"target source +k"`` / ``-k`` → additive ties.

Inequality (strict, enforced with small ε):
    ``"target source <"``  → ``param[target] < param[source]``  
    ``"target source >"``  → ``param[target] > param[source]``

Range (literals):
    ``"target in [lo,hi]"`` → clip target to the closed interval.

Range (between params):
    ``"target lower_idx upper_idx"`` → clip target between two *indices*.

Examples
--------
- Tie two line centers with an offset of +1.2 Å:
    ``"15 12 +1.2"``
- Force a width to be less than another width:
    ``"7 6 <"``
- Keep a continuum slope within [-5, 5]:
    ``"3 in [-5, 5]"``
- Constrain a parameter between two others:
    ``"9 2 5"``

Notes
-----
- All helpers are **JAX‑compatible** where marked with ``@jit``; inputs should
  be JAX arrays whenever you need tracing/compilation.
- Arithmetic ties combine naturally for reconstruction of full parameter
  vectors (see :func:`apply_tied_and_fixed_params`).
"""


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
    Identify indices in the parameter dictionary that match given name patterns.

    Parameters
    ----------
    params_dict : dict | np.ndarray
        Maps full parameter keys (e.g., ``'amplitude_Hbeta_1_broad'``) to indices.
        If an array is passed, indices are inferred from enumeration of its elements.
    params : str | list[str] | list[list[str]]
        One or more *substring* patterns. Each pattern can be a list of substrings
        that must all appear in the key (logical AND).
        Examples: ``"amplitude"``, ``["amplitude", "Hbeta"]``, or
        ``[["amplitude"], ["center","OIII"]]``.
    verbose : bool, optional
        If ``True``, print matching keys.

    Returns
    -------
    jnp.ndarray
        1‑D array of **unique** matching indices (sorted).
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
    Works with both NumPy and JAX arrays.

    Parameters
    ----------
    params_dict : dict
        Dictionary mapping parameter names to indices.
    params : jnp.ndarray or np.ndarray
        Parameter array of shape (N, D).
    scale : jnp.ndarray or np.ndarray
        Scale values of shape (N,).

    Returns
    -------
    jnp.ndarray or np.ndarray
        Scaled parameter array.
    """
    idxs = mapping_params(params_dict, [["amplitude"]])
    idxs_log = mapping_params(params_dict, [["logamp"]])

    if isinstance(params, jnp.ndarray):
        if len(idxs_log) == 0:
            params = params.at[:, idxs].multiply(scale[:, None])
        else:
            params = (params.at[:, idxs].multiply(scale[:, None]).at[:, idxs_log].add(jnp.log10(scale[:, None])))
    elif isinstance(params, np.ndarray):
        if len(idxs_log) == 0:
            params[:, idxs] *=  scale
        else:
            params[:, idxs] *=  scale
            params[:, idxs_log] += np.log10(scale)
    else:
        raise TypeError(f"Unsupported array type: {type(params)}")

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
    
    #print(params.shape)
    if isinstance(params, jnp.ndarray):
        if len(idxs_log) == 0:
            params = params.at[:, idxs].divide(scale[:, None])
        else:
            params = (params.at[:, idxs].divide(scale[:, None]).at[:, idxs_log].subtract(jnp.log10(scale[:, None])))
    elif isinstance(params, np.ndarray):
        if len(idxs_log) == 0:
            params[:, idxs] /= scale
        else:
            params[:, idxs] /= scale
            params[:, idxs_log] -= np.log10(scale)
    else:
        raise TypeError(f"Unsupported array type: {type(params)}")
    
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
    Parse multiple dependency strings into a tuple of structured constraints.

    See the module docstring “Constraint String Grammar” for supported forms.

    Parameters
    ----------
    dependencies : list[str]
        Dependency strings, e.g., ``["7 6 <", "3 in [0.1, 10.0]"]``.

    Returns
    -------
    tuple
        Parsed constraints (each an ``("arithmetic"|... , ...)`` tuple).
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
        # if param == "amplitude":
        #     param = "logamp" #this is assuming all the profiles in sheap use logamp but what happen in the cases where this doesn't happen :c
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
    """
    Convert human‑readable ties (using semantic keys) into dependency strings
    that reference **indices** in the flattened parameter vector.

    Parameters
    ----------
    tied_params : list[list[str]]
        Each item is either ``[target_key, source_key]`` (auto “*1” or center offset)
        or ``[target_key, source_key, op_str]`` where ``op_str`` is one of
        ``'*k'``, ``'/k'``, ``'+k'``, ``'-k'`` (``k`` numeric).
    get_param_coord_value : Callable
        Function returned by :func:`make_get_param_coord_value` to translate
        semantic keys into (index, value, name).

    Returns
    -------
    list[str]
        Dependency strings like ``"target_idx source_idx *k"`` ready for
        :func:`parse_dependencies`.
    """
    list_tied_params = []
    #print(len(tied_params))
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
                list_tied_params.append(f"{pos_param1} {pos_param2} {tied_val}")
            else:
                tied_val = tied[-1]
                if param_1 == param_2 == "logamp":
                    #print(tied)
                    tied_val = f"{np.log10(extract_float(tied_val))}"
                    list_tied_params.append(f"{pos_param1} {pos_param2} {tied_val}")
                    #print(f"{pos_param1} {pos_param2} {tied_val}")
                if isinstance(tied_val, str):
                    list_tied_params.append(f"{pos_param1} {pos_param2} {tied_val}")
                else:
                    print("Define constraints properly.")
    else:
        list_tied_params = []
    #print(list_tied_params)
    return list_tied_params
        #print("Remember move this functions to Assistants and also change it in Montecarlo.")
        
def flatten_tied_map(tied_map: dict[int, tuple[int, str, float]]) -> dict[int, tuple[int, str, float]]:
    """
    Resolve chained ties so that every target ultimately depends on a **free**
    (non‑tied) source.

    Parameters
    ----------
    tied_map : dict[int, tuple[int, str, float]]
        ``target_idx -> (source_idx, op, operand)``

    Returns
    -------
    dict[int, tuple[int, str, float]]
        ``target_idx -> (free_source_idx, op', operand')`` with combined ops.
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
