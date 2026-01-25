r"""
NumPyro Model Utilities
=======================
.. note::
    This require cleaning
    
Helper functions to construct NumPyro models from sheap parameters,
including dictionary conversion, handling of tied parameters, and
safe initialization near constraints.

Main Features
-------------
- Apply arithmetic ties between parameters.
- Convert flat parameter arrays to dictionaries for NumPyro.
- Initialize values safely away from hard bounds.
- Build a full NumPyro model for flux fitting with uncertainty.


Notes
-----
- Dependencies are expressed as tuples
  ``(tag, target_idx, src_idx, op, val)`` where ``op`` is one of
  ``{'+','-','*','/'}``.
- Parameters tied via dependencies are excluded from sampling and
  reconstructed after free parameters are sampled.
"""

__author__ = 'felavila'

__all__ = [
    "NumPyro_apply_arithmetic_ties",
    "make_numpyro_model",
    "NumPyro_params_to_dict",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist




def make_numpyro_model(name_list,wl,flux,sigma,constraints,params_i,theta_to_sheap,fixed_params,dependencies,model_func):
    r"""
    Build a NumPyro probabilistic model for spectrum fitting.

    Parameters
    ----------
    name_list : list of str
        Names of NumPyro parameters (e.g. ``["theta_0", "theta_1", ...]``).
    wl : jnp.ndarray
        Wavelength grid, shape (n_pix,).
    flux : jnp.ndarray
        Observed flux, shape (n_pix,).
    sigma : jnp.ndarray
        Flux uncertainties, shape (n_pix,).
    constraints : list of tuple
        Bounds for each parameter as (low, high).
    params_i : array-like
        Initial parameter values.
    theta_to_sheap : dict
        Mapping from NumPyro parameter name to SHEAP parameter name.
    fixed_params : dict
        Dictionary of fixed parameter values keyed by SHEAP names.
    dependencies : list of tuple
        List of ties (see :func:`apply_arithmetic_ties`).
    model_func : Callable
        Spectral model function with signature ``model_func(wl, theta)``.

    Returns
    -------
    numpyro_model : Callable
        A function defining the NumPyro model.
    init_value : dict
        Dictionary of initial parameter values for ``init_to_value``.

    Notes
    -----
    The model:
    .. math::

        \\theta_i \sim \mathrm{Uniform}(low_i, high_i)

        f(\\lambda) = \\mathrm{model\_func}(\\lambda, \\theta)

        y(\\lambda) \sim \mathcal{N}(f(\\lambda), \sigma)

    - Dependent (tied) parameters are excluded from sampling and
      reconstructed afterward.
    - Fixed parameters are held at their provided values.
    """
    init_values = {key: params_i[_] for _,key in enumerate(theta_to_sheap.values())}
    init_value = NumPyro_params_to_dict(params_i,dependencies,constraints)
    if not dependencies:
        dependencies = ()
    def numpyro_model():
        params = {}    
        idx_targets = [i[1] for i in dependencies]
        for i, (name, (low, high)) in enumerate(zip(name_list, constraints)):
            sheap_name = theta_to_sheap[name]
            if i in idx_targets:
                #print(i,idx_targets,dependencies)
                continue  # skip tied targets; they'll be calculated later
            elif sheap_name in fixed_params.keys():
                val = fixed_params[sheap_name]
                if val is None:
                    val = init_values.get(sheap_name)
                    if val is None:
                        raise ValueError(f"Fixed param '{sheap_name}' is None and not found in init_values.")
            else:
                val = numpyro.sample(name, dist.Uniform(low, high))
            params[name] = val
        params = NumPyro_apply_arithmetic_ties(params, dependencies)
        theta = jnp.array([params[name] for name in name_list])
        pred = model_func(wl, theta)
        numpyro.sample("obs", dist.Normal(pred, sigma), obs=flux)
    
    
    return numpyro_model,init_value


def NumPyro_apply_arithmetic_ties(params: Dict[str, float], dependencies: List[Tuple]) -> Dict[str, float]:
    r"""
    Apply arithmetic ties to update dependent parameters in place.
    #TODO this is already in other place?
    Parameters
    ----------
    params : dict
        Dictionary of parameter values keyed by ``"theta_{i}"``.
    dependencies : list of tuple
        List of constraints of the form:
        ``(tag, target_idx, src_idx, op, val)``.

    Returns
    -------
    dict
        Updated parameter dictionary with tied targets assigned.

    Notes
    -----
    - Supported operations are ``+``, ``-``, ``*``, ``/``.
    - Example: if ``theta_3 = theta_1 * 2``, then dependency is
      ``("arith", 3, 1, "*", 2.0)``.
    """
    for tag, target_idx,src_idx, op, val in dependencies:
        src = params[f"theta_{src_idx}"]
        if op == '+':
            result = src + val
        elif op == '-':
            result = src - val
        elif op == '*':
            result = src * val
        elif op == '/':
            result = src / val
        else:
            raise ValueError(f"Unsupported operation: {op}")
        params[f"theta_{target_idx}"] = result
    return params




def NumPyro_params_to_dict(params, dependencies, constraints=None, eps=1e-2):
    r"""
    Convert flat parameters into a dictionary for NumPyro initialization.

    Excludes dependent parameters and applies jitter to avoid initializing
    exactly at bounds.

    Parameters
    ----------
    params : array-like
        Flat parameter values (e.g. from a minimizer).
    dependencies : list of tuple
        Ties of the form ``(tag, target_idx, src_idx, op, val)``.
    constraints : dict or list, optional
        Parameter bounds. Either a dict mapping ``theta_i -> (low, high)``
        or a list aligned with params.
    eps : float, default=1e-2
        Minimum offset from bounds to avoid initialization at the edge.

    Returns
    -------
    dict
        Dictionary of free parameters, keyed as ``theta_i``.

    Notes
    -----
    - If a parameter value is invalid (NaN, inf, outside bounds),
      it is reset to the midpoint of its allowed range.
    - Exact zeros are nudged to ``eps`` to avoid degenerate starts.
    """
    if not dependencies:
        dependencies = ()
    target_idx_list = [d[1] for d in dependencies]
    init_dict = {}

    for idx, val in enumerate(params):
        if idx in target_idx_list:
            continue  # skip dependent (tied) parameters
        name = f"theta_{idx}"
        val = float(val)

        # Apply jitter if constraints are available
        if constraints is not None:
            if isinstance(constraints, dict):
                low, high = constraints.get(name, (-jnp.inf, jnp.inf))
            else:  # assume list aligned with params
                low, high = constraints[idx]

            if not jnp.isfinite(val) or val <= low or val >= high:
                val = 0.5 * (low + high) if jnp.isfinite(low) and jnp.isfinite(high) else val
            if val - low < eps:
                val = low + eps
            elif high - val < eps:
                val = high - eps

        # Optional: prevent exact zero
        if val == 0:
            val = eps

        init_dict[name] = val

    return init_dict






# def params_to_dict_old(params,dependencies):
#     r"""
#     Convert parameters to a dict (legacy version).

#     Parameters
#     ----------
#     params : array-like
#         Flat list of parameter values.
#     dependencies : list of tuple
#         Constraints (same format as in :func:`apply_arithmetic_ties`).

#     Returns
#     -------
#     dict
#         Dictionary mapping free parameter indices to values,
#         keyed as ``theta_i``.
#     """
#     #tag,target_idx,src_idx, op, val = dependencies 
#     target_idx_list = [d[1] for d in dependencies]
#     init_value = {}
#     for idx,val in enumerate(params):
#         if idx in target_idx_list:
#             continue
#         else:
#             init_value[f"theta_{idx}"] = val
#     return init_value
