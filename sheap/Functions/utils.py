from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np 
import jax.numpy as jnp 
from jax.scipy.integrate import trapezoid
from jax import vmap, jit



# def wrap_profile_with_center_override(profile_func: Callable) -> Callable:
#     """
#     JIT-compatible version that inserts `center` into correct param position using `jnp.insert`.
#     """
#     param_names = profile_func.param_names
#     if "center" not in param_names:
#         raise ValueError(f"Profile '{profile_func.__name__}' has no 'center' parameter.")

#     center_idx = param_names.index("center")

#     def wrapped(x, params, override_center):
#         full_params = jnp.insert(params, center_idx, override_center)
#         return profile_func(x, full_params)

#     return wrapped

# TODO add continium to gaussian sum as and option
def combine_auto(funcs):
    """
    Assumes each function 'f' has an attribute `f.n_params` that tells how many
    parameters it needs. Then automatically slices based on that.
    """

    def combined_func(x, all_args):
        start = 0
        total = 0
        for f in funcs:
            part_size = f.n_params  # e.g., if gauss.n_params = 3
            fargs = all_args[start : start + part_size]
            start += part_size
            total += f(x, fargs)
        return total

    return combined_func

def make_fused_profiles(funcs):
    n_params = [f.n_params for f in funcs]
    param_splits = np.cumsum([0] + n_params)  # [0, 3, 6, ...]
    def fused_profile(x, all_args):
        result = 0.0
        for i, f in enumerate(funcs):
            fargs = all_args[param_splits[i]:param_splits[i+1]]
            result = result + f(x, fargs)
        return result
    return fused_profile



def make_super_fused(funcs):
    # For clarity, give each function a label
    fn_labels = [f"fn{i}" for i in range(len(funcs))]
    param_counts = [f.n_params for f in funcs]
    param_splits = [0] + list(jnp.cumsum(jnp.array(param_counts)))
    
    # Build the code string for the fused function
    code_lines = ["def fused(x, all_args, " + ", ".join(fn_labels) + "):"]
    code_lines.append("    out = 0.0")
    for i, (fn, start, end) in enumerate(zip(fn_labels, param_splits[:-1], param_splits[1:])):
        code_lines.append(f"    out += {fn}(x, all_args[{start}:{end}])")
    code_lines.append("    return out")
    code_str = "\n".join(code_lines)
    
    # Local namespace to exec into
    local_ns = {}
    exec(code_str, {}, local_ns)
    fused = local_ns["fused"]
    # Partially apply the profile functions as fixed arguments
    def fused_fixed(x, all_args):
        return fused(x, all_args, *funcs)
    return fused_fixed

def param_count(n):
    """
    A decorator that attaches an attribute `.n_params` to the function,
    indicating how many parameters it expects.
    """

    def decorator(func):
        func.n_params = n
        return func

    return decorator

def with_param_names(param_names: list[str]):
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
    profile_fn : callable
        f(x, p) → y, where
          x has shape (n_pixels,) or (n_pixels,1,1,…),
          p has shape (..., n_params),
        and y broadcasts to shape (n_pixels, ...).
    method : {"broadcast", "vmap"}
    Returns
    -------
    integrate(x, params) → integral over x of profile_fn(x,p)
       x      shape (n_pixels,)
       params shape (n_spectra, n_lines, n_params)
       → returns array of shape (n_spectra, n_lines)
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