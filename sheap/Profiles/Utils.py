"""
Profile Utilities
=================

This module provides utility functions to support the definition,
integration, and composition of spectral line and continuum profiles
in *sheap*.

Functions
---------
- ``make_fused_profiles(funcs)`` :
    Combine multiple profile functions into a single callable that
    evaluates the sum of all components.

- ``with_param_names(param_names)`` :
    Decorator to attach parameter names and count metadata to profile
    functions for consistent handling across the codebase.

- ``make_integrator(profile_fn, method="broadcast")`` :
    Factory for JAX-based integrators of profile functions. Supports
    either broadcasting across batches or nested `vmap` evaluation.

- ``build_grid_penalty(weights_idx, n_Z, n_age)`` :
    Construct a Laplacian smoothness penalty on 2D grids of host
    template weights (Z × age), useful for regularization.

- ``trapz_jax(y, x)`` :
    Lightweight trapezoidal integration implemented with JAX.

Notes
-----
- All utilities are JAX-compatible and designed for use in differentiable
  fitting pipelines.
- ``with_param_names`` ensures that each profile function exposes
  ``.param_names`` and ``.n_params`` attributes for downstream
  bookkeeping.
- ``make_integrator`` can be used to integrate profiles per spectrum,
  per component, or across batches without manual looping.
"""

__author__ = 'felavila'


__all__ = [
    "build_grid_penalty",
    "make_fused_profiles",
    "make_integrator",
    "trapz_jax",
    "with_param_names",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np 
import jax.numpy as jnp 
from jax.scipy.integrate import trapezoid
from jax import vmap, jit




def make_fused_profiles(funcs):
    """
    Fuse multiple profile functions into a single callable.

    Parameters
    ----------
    funcs : list of callables
        Each function must have a `.n_params` attribute
        and a signature `(x, params)`.

    Returns
    -------
    fused_profile : callable
        A function that evaluates the sum of all profiles given
        a single concatenated parameter vector.
    """
    n_params = [f.n_params for f in funcs]
    param_splits = np.cumsum([0] + n_params)  # [0, 3, 6, ...]
    def fused_profile(x, all_args):
        result = 0.0
        for i, f in enumerate(funcs):
            fargs = all_args[param_splits[i]:param_splits[i+1]]
            result = result + f(x, fargs)
        return result
    return fused_profile

def with_param_names(param_names: list[str]):
    """
    Decorator to attach parameter names and count to a profile function.

    Parameters
    ----------
    param_names : list of str
        Names of the parameters for the decorated profile function.

    Returns
    -------
    decorator : callable
        A decorator that attaches `.param_names` and `.n_params` attributes
        to the target function.
    """
    def decorator(func):
        func.param_names = param_names
        func.n_params = len(param_names)
        return func
    return decorator


# def make_g(list):
#     amplitudes, centers = list.amplitude, list.center
#     return PROFILE_FUNC_MAP["Gsum_model"](centers, amplitudes)
#here add the function to reconstruct sum_gaussian_amplitude_free 


def make_integrator(profile_fn, method="broadcast"):
    """
    Create an integrator for profile functions. This works for 1D wavelength  and 3D params 
    n_sample,n_lines,n_params.

    Parameters
    ----------
    profile_fn : callable
        Profile function with signature `(x, params) -> y`.
    method : {"broadcast", "vmap"}, optional
        Integration strategy:
        - "broadcast": expand x for broadcasting across batches
        - "vmap": use nested vectorization

    Returns
    -------
    integrate : callable
        Function `(x, params) -> integral` returning integrated flux.
    """
    if method == "broadcast":
        @jit
        def integrate(x, params):
            # ensure jnp arrays
            x      = jnp.asarray(x)                    # (n_pixels,)
            params = jnp.asarray(params)               # (n_spec, n_lines, n_params)

            # expand x to broadcast against params’ leading dims
            x_exp = x[:, None, None]                   # (n_pixels,1,1)
            y     = profile_fn(x_exp, params)          # -> (n_pixels, n_spec, n_lines)
            return trapezoid(y, x, axis=0)             # integrate over 0 → (n_spec, n_lines)

        return integrate

    elif method == "vmap":
        # first define a scalar integrator for a single (x,p) pair
        def single_int(x, p):
            y = profile_fn(x, p)        # p: (n_params,) → y: (n_pixels,)
            return trapezoid(y, x)

        # lift over lines, then over spectra
        int_lines = vmap(single_int, in_axes=(None, 0))  # maps over p-lines
        int_specs = vmap(int_lines,  in_axes=(None, 0))  # maps over spectra
        integrate  = jit(lambda x, params: int_specs(x, params))
        return integrate

    else:
        raise ValueError(f"unknown method {method!r}")
    
    
def build_grid_penalty(
    weights_idx,
    n_Z: int,
    n_age: int,
) -> Callable[[jnp.ndarray], float]:
    """
    Construct a Laplacian smoothness penalty over a 2D template grid.

    Parameters
    ----------
    weights_idx : list[int]
        Indices of weight parameters in the global parameter vector.
    n_Z : int
        Number of metallicity bins.
    n_age : int
        Number of age bins.

    Returns
    -------
    penalty : callable
        Function `(params) -> float` computing the smoothness penalty.
    """
    if len(weights_idx) != n_Z * n_age:
        raise ValueError(f"Expected {n_Z * n_age} weight indices, got {len(weights_idx)}")

    def penalty(params: jnp.ndarray) -> float:
        weights = params[jnp.array(weights_idx)]
        weights_grid = weights.reshape(n_Z, n_age)

        d2_age = weights_grid[:, :-2] - 2 * weights_grid[:, 1:-1] + weights_grid[:, 2:]
        d2_Z   = weights_grid[:-2, :] - 2 * weights_grid[1:-1, :] + weights_grid[2:, :]
        return jnp.sum(d2_age**2) + jnp.sum(d2_Z**2)

    return penalty

def trapz_jax(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    JAX-compatible trapezoidal integration.

    Parameters
    ----------
    y : jnp.ndarray
        Function values.
    x : jnp.ndarray
        Grid over which integration is performed.

    Returns
    -------
    jnp.ndarray
        Approximate integral of y over x.
    """
    dx = x[1:] - x[:-1]
    return jnp.sum((y[1:] + y[:-1]) * dx / 2)