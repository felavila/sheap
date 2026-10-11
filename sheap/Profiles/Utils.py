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
    "GaussianSum",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union,Iterable 

import numpy as np 
import jax.numpy as jnp 
from jax.scipy.integrate import trapezoid
from jax import vmap, jit,nn


def with_param_names(
    param_names: Iterable[str],
    linear_param_names: Optional[Iterable[str]] = None,
    linear_param_mask: Optional[Iterable[bool]] = None,
    profile_name = None):
    """
    Decorate a profile function with parameter metadata.

    Parameters
    ----------
    param_names : iterable of str
        Ordered parameter names for the function.
    linear_param_names : iterable of str, optional
        Names of parameters that enter linearly.
    linear_param_mask : iterable of bool, optional
        Boolean mask of the same length as ``param_names`` indicating which
        parameters are linear.

    Returns
    -------
    callable
        Decorated function with attributes:

        - ``param_names``
        - ``n_params``
        - ``linear_param_mask``
        - ``linear_param_indices``
        - ``nonlinear_param_indices``
        - ``linear_param_names``
        - ``nonlinear_param_names``

    Notes
    -----
    Provide either ``linear_param_names`` or ``linear_param_mask``.
    If neither is provided, all parameters are assumed nonlinear.
    """
    param_names = list(param_names)
    n_params = len(param_names)

    if linear_param_names is not None and linear_param_mask is not None:
        raise ValueError("Use only one of linear_param_names or linear_param_mask.")

    if linear_param_names is not None:
        linear_param_names = set(linear_param_names)
        mask = [name in linear_param_names for name in param_names]
    elif linear_param_mask is not None:
        mask = list(linear_param_mask)
        if len(mask) != n_params:
            raise ValueError("linear_param_mask must have the same length as param_names.")
    else:
        mask = [False] * n_params

    linear_idx = tuple(i for i, m in enumerate(mask) if m)
    nonlinear_idx = tuple(i for i, m in enumerate(mask) if not m)

    def decorator(func):
        func.param_names = tuple(param_names)
        func.n_params = n_params
        func.linear_param_mask = tuple(bool(m) for m in mask)
        func.linear_param_indices = linear_idx
        func.nonlinear_param_indices = nonlinear_idx
        func.linear_param_names = tuple(param_names[i] for i in linear_idx)
        func.nonlinear_param_names = tuple(param_names[i] for i in nonlinear_idx)
        func.profile_name = profile_name
        return func

    return decorator


# def make_fused_profiles(funcs):
#     """
#     Fuse multiple profile functions into a single callable.
#     #TODO when we make a fused profile it lose the info about the n_params and the names (this can be repetead so are not trustworthy) we can add at least the n param to the combine ones.
#     Parameters
#     ----------
#     funcs : list of callables
#         Each function must have a `.n_params` attribute
#         and a signature `(x, params)`.

#     Returns
#     -------
#     fused_profile : callable
#         A function that evaluates the sum of all profiles given
#         a single concatenated parameter vector.
#     """
#     n_params = [f.n_params for f in funcs]
#     param_splits = np.cumsum([0] + n_params)  # [0, 3, 6, ...]
#     def fused_profile(x, all_args):
#         result = 0.0
#         for i, f in enumerate(funcs):
#             fargs = all_args[param_splits[i]:param_splits[i+1]]
#             result = result + f(x, fargs)
#         return result
#     return fused_profile

# def make_g(list):
#     amplitudes, centers = list.amplitude, list.center
#     return PROFILE_FUNC_MAP["Gsum_model"](centers, amplitudes)
#here add the function to reconstruct sum_gaussian_amplitude_free 


def make_fused_profiles(funcs):
    """
    Fuse multiple profile functions into a single callable while preserving
    parameter metadata, including which parameters are linear.
    """
    n_params = [f.n_params for f in funcs]
    param_splits = np.cumsum([0] + n_params)

    fused_param_names = []
    fused_linear_mask = []

    for i, f in enumerate(funcs):
        names = list(getattr(f, "param_names", [f"p{i}_{k}" for k in range(f.n_params)]))
        linear_mask = list(getattr(f, "linear_param_mask", [False] * f.n_params))

        if len(names) != f.n_params:
            raise ValueError(f"Function {i} has inconsistent param_names length.")
        if len(linear_mask) != f.n_params:
            raise ValueError(f"Function {i} has inconsistent linear_param_mask length.")

        fused_param_names.extend(names)
        fused_linear_mask.extend(linear_mask)

    @with_param_names(
        fused_param_names,
        linear_param_mask=fused_linear_mask,)
    def fused_profile(x, all_args):
        all_args = jnp.asarray(all_args)
        result = jnp.array(0.0, dtype=x.dtype if hasattr(x, "dtype") else jnp.float32)

        for i, f in enumerate(funcs):
            fargs = all_args[param_splits[i]:param_splits[i + 1]]
            result = result + f(x, fargs)

        return result

    fused_profile.param_splits = tuple(param_splits.tolist())
    fused_profile.sub_funcs = tuple(funcs)

    return fused_profile

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

class GaussianSum:
    def __init__(self, n, constraints=None, inequalities=None):
        """
        Initialize the GaussianSum with parameter constraints.

        Parameters:
        - n (int): Number of Gaussian functions.
        - constraints (dict): Optional equality constraints on parameters.
            Example:
                {
                    'amp': [('amp0', 'amp1')],  # amp0 == amp1
                    'mu': [('mu2', 'mu3')],
                    'sigma': [('sigma1', 'sigma2')]
                }
        - inequalities (dict): Optional inequality constraints on parameters.
            Example:
                {
                    'sigma': [('sigma1', 'sigma2')]  # sigma2 > sigma1
                }
        """
        self.n = n
        self.constraints = constraints or {}
        self.inequalities = inequalities or {}
        self.param_mapping = self._build_param_mapping()
        self.num_free_params = self._count_free_params()
        self.sum_gaussians_jit = self._build_gaussian_sum()

    def _build_param_mapping(self):
        """
        Build a mapping from free parameters to all parameters,
        applying constraints as specified.
        """
        # Initialize mappings: each parameter maps to itself initially
        mapping = {
            'amp': list(range(self.n)),
            'mu': list(range(self.n)),
            'sigma': list(range(self.n))
        }

        # Apply equality constraints
        for param_type, pairs in self.constraints.items():
            for (p1, p2) in pairs:
                idx1 = int(p1.replace(param_type, ''))
                idx2 = int(p2.replace(param_type, ''))
                mapping[param_type][idx2] = mapping[param_type][idx1]

        return mapping

    def _count_free_params(self):
        """
        Count the number of free parameters after applying constraints.
        """
        free_amp = len(set(self.param_mapping['amp']))
        free_mu = len(set(self.param_mapping['mu']))
        free_sigma = len(set(self.param_mapping['sigma']))
        return free_amp + free_mu + free_sigma + self._count_inequality_free_params()

    def _count_inequality_free_params(self):
        """
        Count additional free parameters required for inequality constraints.
        For each inequality, an extra free parameter is needed to define the offset.
        """
        count = 0
        for param_type, pairs in self.inequalities.items():
            count += len(pairs)
        return count

    def _apply_constraints(self, params):
        """
        Apply equality constraints to the parameter vector to obtain full parameter sets.

        Parameters:
        - params (jnp.ndarray): Free parameters vector.

        Returns:
        - amps, mus, sigmas (tuple of jnp.ndarray): Full parameter sets.
        """
        free_amp = self.param_mapping['amp']
        free_mu = self.param_mapping['mu']
        free_sigma = self.param_mapping['sigma']

        num_free_amp = len(set(free_amp))
        num_free_mu = len(set(free_mu))
        num_free_sigma = len(set(free_sigma))

        # Extract free parameters
        idx = 0
        amps_free = params[idx:idx + num_free_amp]
        idx += num_free_amp
        mus_free = params[idx:idx + num_free_mu]
        idx += num_free_mu
        sigmas_free = params[idx:idx + num_free_sigma]
        idx += num_free_sigma

        # Map free parameters to all parameters using the mapping
        amps = jnp.array([amps_free[i] for i in self.param_mapping['amp']])
        mus = jnp.array([mus_free[i] for i in self.param_mapping['mu']])
        sigmas = jnp.array([sigmas_free[i] for i in self.param_mapping['sigma']])

        return amps, mus, sigmas

    def _apply_inequality_constraints(self, sigmas, params):
        """
        Apply inequality constraints to sigmas.

        For example, enforce sigma2 > sigma1 by setting sigma2 = sigma1 + softplus(delta)

        Parameters:
        - sigmas (jnp.ndarray): Current sigma parameters.
        - params (jnp.ndarray): Remaining parameters for inequality transformations.

        Returns:
        - jnp.ndarray: Transformed sigma parameters satisfying inequalities.
        """
        if not self.inequalities:
            return sigmas

        # Assuming all inequality constraints are on 'sigma'
        for (s1, s2) in self.inequalities.get('sigma', []):
            idx1 = int(s1.replace('sigma', ''))
            idx2 = int(s2.replace('sigma', ''))
            delta = params[0]
            params = params[1:]
            transformed_sigma2 = sigmas[idx1] + nn.softplus(delta)
            sigmas = sigmas.at[idx2].set(transformed_sigma2)
        return sigmas

    def _build_gaussian_sum(self):
        """
        Build the JIT-compiled Gaussian sum function.

        Returns:
        - sum_gaussians_jit (function): JIT-compiled function.
        """
        def gaussian(x, amp, mu, sigma):
            return amp * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)


        def sum_gaussians(x, params):
            # Validate parameter length
            if params.shape[0] != self.num_free_params:
                raise ValueError(f"Expected {self.num_free_params} parameters, got {params.shape[0]}.")

            # Apply equality constraints
            amps, mus, sigmas = self._apply_constraints(params)
            
            # Apply inequality constraints if any
            if self.inequalities:
                # Extract deltas for inequalities
                delta_params = params[-len(self.inequalities.get('sigma', [])):]
                sigmas = self._apply_inequality_constraints(sigmas, delta_params)

            # Use a lambda to fix 'x' while vectorizing over amp, mu, sigma
            gaussians = vmap(lambda amp, mu, sigma: gaussian(x, amp, mu, sigma))(amps, mus, sigmas)
            
            return jnp.sum(gaussians, axis=0)
        self.n_params = self.num_free_params
        return jit(sum_gaussians)

    def __call__(self, x, params):
        """
        Compute the sum of Gaussians at points x with given parameters.

        Parameters:
        - x (jnp.ndarray): Points at which to evaluate the sum.
        - params (jnp.ndarray): Free parameters vector.

        Returns:
        - jnp.ndarray: Sum of Gaussians evaluated at x.
        """
        
        return self.sum_gaussians_jit(x, params)
    
def add_linear_amplitude_design_builder(amplitude_index=0,):
    """
    Attach a design_builder to a profile whose amplitude enters linearly.

    Assumes:

        model(x, params) = amplitude * shape(x, nonlinear_params)

    The returned design builder gives:

        offset  = 0
        columns = unit-amplitude profile[:, None]

    Parameters
    ----------
    amplitude_index : int
        Local position of the linear amplitude parameter.
        For SHEAP log-lambda profiles this is normally 0.
    """

    def decorator(func):

        def design_builder(x,params,):
            params = jnp.asarray(params)

            # Set amplitude = 1.
            # Everything else stays exactly the same.
            unit_params = params.at[amplitude_index].set(jnp.asarray(1.0,dtype=params.dtype,))

            # Exact unit-amplitude basis
            shape = func(x,unit_params,)

            offset = jnp.zeros_like(shape)

            columns = shape[:,None,]

            return (offset,columns,)

        func.design_builder = design_builder

        return func

    return decorator


def add_linear_design_builder(linear_indices=None,zero_offset=False,):
    """
    Attach a generic design_builder to a function that is affine
    in one or more parameters.

    The function must satisfy:

        model(x, params)
        =
        offset(x, nonlinear_params)
        + sum_i linear_i * column_i(x, nonlinear_params)

    Parameters
    ----------
    linear_indices : sequence[int] or None
        Local indices of the linear parameters.

        If None, they are obtained automatically from:

            func.linear_param_mask

        which must already exist.

    zero_offset : bool
        If True, assume that setting all linear parameters to zero
        gives exactly zero.

        This avoids one model evaluation.

        Good for profiles like:

            amplitude * shape

        or:

            slope*x + intercept

        where no nonlinear additive component exists.

    Returns
    -------
    Decorated function with:

        func.design_builder(x, params)
            -> offset, columns
    """

    def decorator(func):
        if linear_indices is None:
            linear_mask = getattr(func,"linear_param_mask",None,)

            if linear_mask is None:
                raise ValueError(
                    f"{func.__name__} does not expose "
                    "'linear_param_mask'.")

            indices = tuple(i for i, is_linear in enumerate(linear_mask) if is_linear)

        else:

            indices = tuple(int(i) for i in linear_indices)

        if len(indices) == 0:
            raise ValueError(f"{func.__name__} has no linear parameters.")

        # JAX static array containing local linear positions.
        linear_idx = jnp.asarray(indices,dtype=jnp.int32,)


        def design_builder(x,params,):

            params = jnp.asarray(params)

            dtype = params.dtype

            # ====================================================
            # Set ALL linear parameters to zero.
            # ====================================================

            params_zero = params.at[linear_idx].set(jnp.zeros(len(indices),dtype=dtype,))

            # ====================================================
            # Offset
            # ====================================================

            if zero_offset:

                offset = jnp.zeros_like(x,dtype=x.dtype,)

            else:

                offset = func(x,params_zero,)

            # ====================================================
            # Build one column for each linear parameter.
            # ====================================================

            columns = []

            for idx in indices:

                params_one = params_zero.at[idx].set(jnp.asarray(1.0,dtype=dtype,))

                column = func(x,params_one,)

                # Required for a nonzero nonlinear offset.
                if not zero_offset:
                    column = (column - offset)

                columns.append(column)

            columns = jnp.stack(columns,axis=1,)

            return (offset,columns,)

        # --------------------------------------------------------
        # Attach metadata
        # --------------------------------------------------------

        func.design_builder = (design_builder)

        func.design_linear_indices = (indices)

        return func

    return decorator

def add_design_builder():
    """
    Attach a generic design_builder using the linear parameter
    information provided by @with_param_names.

    Assumes the model is affine in the parameters marked as linear:

        model = offset(nonlin) + columns(nonlin) @ linear_params

    If there are no linear parameters:
        offset = full model
        columns = empty
    """

    def decorator(func):

        linear_mask = getattr(func,"linear_param_mask",None,)

        if linear_mask is None:
            raise ValueError(
                f"{func.__name__} must first be decorated "
                "with @with_param_names."
            )

        linear_indices = tuple(i for i, is_linear in enumerate(linear_mask) if is_linear)

        n_linear = len(linear_indices)

        # ========================================================
        # No linear parameters
        # ========================================================
        if n_linear == 0:

            def design_builder(x, params):

                offset = func(x,params,)

                columns = jnp.empty((x.shape[0],0,),dtype=offset.dtype,)

                return offset, columns

        # ========================================================
        # One or more linear parameters
        # ========================================================

        else:

            linear_idx = jnp.asarray(linear_indices,dtype=jnp.int32,)

            def design_builder(x, params):

                params = jnp.asarray(params)


                params_zero = params.at[linear_idx].set(jnp.zeros(n_linear,dtype=params.dtype,))

                offset = func(x,params_zero,)


                columns = []

                for idx in linear_indices:

                    params_one = params_zero.at[idx].set(jnp.asarray(1.0,dtype=params.dtype,))

                    column = (func(x,params_one,)- offset)

                    columns.append(column)

                columns = jnp.stack(columns,axis=1,)
                return offset, columns

        func.design_builder = design_builder

        return func

    return decorator