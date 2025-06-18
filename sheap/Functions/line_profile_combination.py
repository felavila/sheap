# import os
# from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# #import jax
# import jax.numpy as jnp
# #import jax.scipy as jsp
# #import numpy as np
# from jax import jit, vmap,lax 
# from jax.scipy.special import erfc
# #from jax.scipy.stats import norm #maybe dosent exist xd

# from sheap.Functions.utils import param_count,with_param_names
# from sheap.Functions.profiles import PROFILE_LINE_FUNC_MAP



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



# def SPAF(centers, amplitude_rules, profile_name: str):
#     """
#     SPAF = Sum Profiles Amplitude Free

#     Args:
#         centers (List[float]): Rest-frame centers of the lines.
#         amplitude_rules (List[Tuple[int, float, int]]): 
#             Each tuple is (line_idx, coef, free_param_idx).
#         profile_name (str): Key from PROFILE_LINE_FUNC_MAP.

#     Returns:
#         Callable G(x, params), where:
#             - First N are free amplitudes (inferred from rules)
#             - Followed by 1 global delta
#             - Followed by shared profile parameters (e.g., fwhm, alpha)
#     """
#     centers = jnp.array(centers)
    
#     base_func = PROFILE_LINE_FUNC_MAP.get(profile_name)
#     if base_func is None:
#         raise ValueError(f"Profile '{profile_name}' not found in PROFILE_LINE_FUNC_MAP.")
#     #print(base_func.param_names[2:])
#     # Wrap base_func to inject center dynamically
#     wrapped_profile = wrap_profile_with_center_override(base_func)
    
#     # Count how many independent amplitude parameters are needed
#     unique_amplitudes = sorted({rule[2] for rule in amplitude_rules})
#     n_params = len(unique_amplitudes)

#     # Number of profile params (excluding amplitude and center)
#     #full_param_names = base_func.param_names
#     #extra_param_count = len(full_param_names) - 2  # amplitude, center excluded

#     #@param_count(n_params + 1 + extra_param_count)
#     @with_param_names(["amplitude"+str(n) for n in range(n_params)]+["shift"]+base_func.param_names[2:])
#     def G(x, params):
#         free = params[:n_params]             # Free amplitude values
#         delta = params[n_params]             # Shared shift
#         extras = params[n_params + 1:]       # Shared profile-specific params (e.g. fwhm, alpha)

#         result = 0.0
#         for idx, coef, free_idx in amplitude_rules:
#             amp = coef * free[free_idx]
#             center = centers[idx] + delta
#             profile_params = jnp.concatenate([jnp.array([amp]), extras])
#             result += wrapped_profile(x, profile_params, center)
#         return result

#     return G
