from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import jit, vmap

from sheap.Profiles.Utils import with_param_names

class ContinuumProfiles:
    def __init__(self,profile_name,delta0 = 5500.0,**kwargs):
        self.profile_name
        self.delta0 = delta0
    
    def make_polynomial_function(degree: int, delta0: float = 5500.0):
        if degree < 0:
            raise ValueError("degree must be >= 0")

        param_names = ["logamp"] + [f"c{i}" for i in range(1, degree + 1)]

        @with_param_names(param_names, profile_name="polynomial")
        def polynomial(xs: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            A = params[0]
            coeffs = params[1:]

            x = (xs - delta0) / delta0

            corr = 0.0
            for i, c in enumerate(coeffs, start=1):
                corr = corr + c * x**i

            return 10**A * jnp.exp(corr)

        return polynomial
