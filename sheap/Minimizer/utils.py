from functools import partial
from typing import Callable, List, Optional, Tuple

import jax.numpy as jnp
from jax import jit


@jit
def project_params_clasic(
    params: jnp.ndarray,
    constraints: jnp.ndarray,
) -> jnp.ndarray:
    """
    Project flat parameters to satisfy individual bounds and apply multiplicative and additive constraints.

    Parameters:
    - params: Flat array of parameters.
    - constraints: Array of (lower, upper) bounds for each parameter.
    Returns:
    - Projected params: Flat array with constraints enforced.
    """
    # Apply individual bounds
    lower_bounds = constraints[:, 0]
    upper_bounds = constraints[:, 1]
    params = jnp.clip(params, lower_bounds, upper_bounds)
    return params


def parse_dependency(dep_str: str):
    """
    Parse a dependency string.

    Supported formats:
      - Arithmetic: "target source *2"  (target = source * 2)
      - Arithmetic: "target source +1"  (target = source + 1)
      - Inequality: "target source <"   (target should be less than source)
      - Inequality: "target source >"   (target should be greater than source)
      - Range literal: "target in [lower,upper]" forces param[target] to be between literal bounds.
      - Range between: "target lower_source upper_source" forces param[target] to be
                       between param[lower_source] and param[upper_source].
    """
    tokens = dep_str.split()

    if len(tokens) == 3:
        if tokens[1] == "in":
            # Format: "target in [lower,upper]"
            target = int(tokens[0])
            range_str = tokens[2]
            if range_str.startswith("[") and range_str.endswith("]"):
                range_contents = range_str[1:-1]
                try:
                    lower_str, upper_str = range_contents.split(",")
                    lower = float(lower_str.strip())
                    upper = float(upper_str.strip())
                except Exception as e:
                    raise ValueError(f"Could not parse range in dependency '{dep_str}': {e}")
                return ("range_literal", target, lower, upper)
            else:
                raise ValueError(f"Invalid range specification in dependency '{dep_str}'")
        else:
            # Check if tokens[2] is purely numeric (i.e. no operator): "target lower_idx upper_idx"
            try:
                _ = int(tokens[2])  # will fail if the 3rd token isn't integer
                target = int(tokens[0])
                lower_idx = int(tokens[1])
                upper_idx = int(tokens[2])
                return ("range_between", target, lower_idx, upper_idx)
            except ValueError:
                # Otherwise, assume arithmetic or inequality:
                target = int(tokens[0])
                source = int(tokens[1])
                op_token = tokens[2]
                if op_token in {"<", ">"}:
                    # Inequality
                    return ("inequality", target, source, op_token, None)
                else:
                    # Arithmetic (+, -, *, /)
                    op = op_token[0]  # e.g., '*' for "*2", '+' for "+1"
                    try:
                        operand = float(op_token[1:])  # e.g., '2'
                    except Exception as e:
                        raise ValueError(f"Could not parse operand in token '{op_token}': {e}")
                    return ("arithmetic", target, source, op, operand)

    elif len(tokens) == 4 and tokens[1] == "in":
        # Alternate format for range literal: "4 in [2, 6]"
        target = int(tokens[0])
        range_str = (tokens[2] + " " + tokens[3]).strip()  # "[2," + "6]"
        if range_str.startswith("[") and range_str.endswith("]"):
            range_contents = range_str[1:-1]
            try:
                lower_str, upper_str = range_contents.split(",")
                lower = float(lower_str.strip())
                upper = float(upper_str.strip())
            except Exception as e:
                raise ValueError(f"Could not parse range in dependency '{dep_str}': {e}")
            return ("range_literal", target, lower, upper)
        else:
            raise ValueError(f"Invalid range specification in dependency '{dep_str}'")
    else:
        raise ValueError(f"Invalid dependency format: {dep_str}")


def parse_dependencies(dependencies: list[str]):
    """Parse a list of dependency strings into structured constraints."""
    return tuple(parse_dependency(dep) for dep in dependencies)


@partial(jit, static_argnums=(2,))
def project_params(
    params: jnp.ndarray,
    constraints: jnp.ndarray,
    parsed_dependencies: Optional[List[Tuple]] = None,
) -> jnp.ndarray:
    """
    Project parameters by clipping to individual bounds and then applying dependency constraints.

    Parameters:
      params: 1D array of parameters.
      constraints: Array of shape (n, 2) with lower and upper bounds for each parameter.
      parsed_dependencies: List of parsed dependency tuples (from `parse_dependencies`).

    Returns:
      A new array with parameters projected according to all constraints.
    """
    # 1) Apply individual lower/upper bounds first
    lower_bounds = constraints[:, 0]
    upper_bounds = constraints[:, 1]
    params = jnp.clip(params, lower_bounds, upper_bounds)

    epsilon = 1e-6  # Small value used for strict inequality adjustments

    if parsed_dependencies is not None:
        for dep in parsed_dependencies:
            dep_type = dep[0]

            if dep_type == "arithmetic":
                # ("arithmetic", target, source, op, operand)
                _, target, source, op, operand = dep

                if op == "*":
                    new_val = params[source] * operand
                elif op == "/":
                    new_val = params[source] / operand
                elif op == "+":
                    new_val = params[source] + operand
                elif op == "-":
                    new_val = params[source] - operand
                else:
                    raise ValueError(f"Unsupported operator: {op}")

                # Update target parameter
                params = params.at[target].set(new_val)

            elif dep_type == "inequality":
                # ("inequality", target, source, op, None)
                _, target, source, op, _ = dep
                if op == "<":
                    # Force params[target] to be strictly less than params[source]
                    new_val = jnp.where(
                        params[target] < params[source],
                        params[target],
                        params[source] - epsilon,
                    )
                elif op == ">":
                    # Force params[target] to be strictly greater than params[source]
                    new_val = jnp.where(
                        params[target] > params[source],
                        params[target],
                        params[source] + epsilon,
                    )
                else:
                    raise ValueError(f"Unsupported inequality operator: {op}")

                params = params.at[target].set(new_val)

            elif dep_type == "range_literal":
                # ("range_literal", target, lower, upper)
                _, target, lower, upper = dep
                new_val = jnp.clip(params[target], lower, upper)
                params = params.at[target].set(new_val)

            elif dep_type == "range_between":
                # ("range_between", target, lower_idx, upper_idx)
                _, target, lower_idx, upper_idx = dep
                new_val = jnp.clip(params[target], params[lower_idx], params[upper_idx])
                params = params.at[target].set(new_val)

            else:
                raise ValueError(f"Unknown dependency type: {dep_type}")

    return params



import jax.numpy as jnp
from typing import Callable, Optional

def build_loss_function(
    func: Callable,
    weighted: bool = True,
    penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
    penalty_weight: float = 0.01,
    param_converter: Optional["Parameters"] = None,
    curvature_weight: float = 0.0,      # γ: second-derivative match
    smoothness_weight: float = 0.0,     # δ: first-derivative smoothness
    max_weight: float = .5,            # α: weight on worst‐pixel term
) -> Callable:
    """
    Build a loss function with:
      - log_cosh data term (mean + max)
      - optional parameter penalty
      - curvature matching (d²)
      - smoothness regularization (d¹)
    """
    print("xdd")
    def log_cosh(x):
        # numerically stable log(cosh(x))
        return jnp.logaddexp(x, -x) - jnp.log(2.0)

    def wrapped(xs, raw_params):
        phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
        return func(xs, phys)

    def curvature_term(y_pred, y):
        d2p = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
        d2o = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
        return jnp.nanmean((d2p - d2o)**2)

    def smoothness_term(y_pred, y):
        dr = y_pred - y
        dp = jnp.gradient(dr, axis=-1)
        return jnp.nanmean(dp**2)

    # -------------------------------------------------------------------
    # Weighted + penalty
    if weighted and penalty_function:
        def loss(params, xs, y, yerr):
            y_pred   = wrapped(xs, params)
            r        = (y_pred - y) / jnp.clip(yerr, 1e-8)

            # data term = mean + max
            Lmean    = jnp.nanmean(log_cosh(r))
            Lmax     = jnp.max   (log_cosh(r))
            data_term = Lmean + max_weight * Lmax

            # penalty on params
            reg_term = penalty_weight * penalty_function(xs, params) * 1e3

            # curvature & smoothness
            curv_term   = curvature_weight  * curvature_term(y_pred, y)
            smooth_term = smoothness_weight * smoothness_term(y_pred, y)

            return data_term + reg_term + curv_term + smooth_term

        return loss

    # -------------------------------------------------------------------
    # Weighted only
    elif weighted:
        def loss(params, xs, y, yerr):
            y_pred   = wrapped(xs, params)
            r        = (y_pred - y) / jnp.clip(yerr, 1e-8)

            Lmean    = jnp.nanmean(log_cosh(r))
            Lmax     = jnp.max   (log_cosh(r))
            data_term = Lmean + max_weight * Lmax

            curv_term   = curvature_weight  * curvature_term(y_pred, y)
            smooth_term = smoothness_weight * smoothness_term(y_pred, y)

            return data_term + curv_term + smooth_term

        return loss

    # -------------------------------------------------------------------
    # Unweighted + penalty
    elif penalty_function:
        def loss(params, xs, y, yerr):
            y_pred   = wrapped(xs, params)
            r        = (y_pred - y)

            Lmean    = jnp.nanmean(log_cosh(r))
            Lmax     = jnp.max   (log_cosh(r))
            data_term = Lmean + max_weight * Lmax

            reg_term    = penalty_weight * penalty_function(xs, params) * 1e3
            curv_term   = curvature_weight  * curvature_term(y_pred, y)
            smooth_term = smoothness_weight * smoothness_term(y_pred, y)

            return data_term + reg_term + curv_term + smooth_term

        return loss

    # -------------------------------------------------------------------
    # Unweighted only
    else:
        def loss(params, xs, y, yerr):
            y_pred   = wrapped(xs, params)
            r        = (y_pred - y)

            Lmean    = jnp.nanmean(log_cosh(r))
            Lmax     = jnp.max   (log_cosh(r))
            data_term = Lmean + max_weight * Lmax

            curv_term   = curvature_weight  * curvature_term(y_pred, y)
            smooth_term = smoothness_weight * smoothness_term(y_pred, y)

            return data_term + curv_term + smooth_term

        return loss

# import jax
# import jax.numpy as jnp
# from typing import Callable, Optional
# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     curvature_weight: float = 1e6,      # γ: second-derivative match
#     smoothness_weight: float = 1e6,     # δ: first-derivative smoothness
# ) -> Callable:
#     """
#     Build a loss function with:
#       - log_cosh data term
#       - optional parameter penalty
#       - curvature matching (d²)
#       - smoothness regularization (d¹)
#     """
#     print("loss smut curvature")
#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def curvature_term(y_pred, y):
#         d2p = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
#         d2o = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
#         return jnp.nanmean((d2p - d2o)**2)

#     def smoothness_term(y_pred, y):
#         dr  = y_pred - y
#         dp  = jnp.gradient(dr, axis=-1)
#         return jnp.nanmean(dp**2)

#     # Weighted + penalty
#     if weighted and penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred     = wrapped(xs, params)
#             r          = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term  = jnp.nanmean(log_cosh(r))
#             reg_term   = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term  = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term= smoothness_weight * smoothness_term(y_pred, y)
#             return data_term + reg_term + curv_term + smooth_term
#         return loss

#     # Weighted only
#     elif weighted:
#         def loss(params, xs, y, yerr):
#             y_pred     = wrapped(xs, params)
#             r          = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term  = jnp.nanmean(log_cosh(r))
#             curv_term  = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term= smoothness_weight * smoothness_term(y_pred, y)
#             return data_term + curv_term + smooth_term
#         return loss

#     # Unweighted + penalty
#     elif penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred     = wrapped(xs, params)
#             r          = (y_pred - y)
#             data_term  = jnp.nanmean(log_cosh(r))
#             reg_term   = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term  = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term= smoothness_weight * smoothness_term(y_pred, y)
#             return data_term + reg_term + curv_term + smooth_term
#         return loss

#     # Unweighted only
#     else:
#         def loss(params, xs, y, yerr):
#             y_pred     = wrapped(xs, params)
#             data_term  = jnp.nanmean(log_cosh(y_pred - y))
#             curv_term  = curvature_weight  * curvature_term(y_pred, y)
#             smooth_term= smoothness_weight * smoothness_term(y_pred, y)
#             return data_term + curv_term + smooth_term
#         return loss
    
# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     curvature_weight: float = 1e6,      # new curvature weight γ
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function with optional curvature penalty.

#     Args:
#         func: The model function, called as func(xs, params)
#         weighted: If True, normalize residuals by yerr
#         penalty_function: Optional regularization function; called as penalty_function(xs, params)
#         penalty_weight: Multiplier for that regularization term
#         param_converter: Optional Parameters() object to transform raw → phys
#         curvature_weight: Weight for the curvature-matching term

#     Returns:
#         A loss function with signature (params, xs, y, yerr) -> scalar loss
#     """
#     print("loss_with_curvature_penalty")
#     def log_cosh(x):
#         # log(cosh(x)) in a numerically stable way
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     def curvature_term(y_pred, y):
#         # assumes last axis is the spectral pixel dimension
#         d2_pred = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
#         d2_obs  = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
#         return jnp.nanmean((d2_pred - d2_obs) ** 2)

#     # Weighted + penalty
#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + reg_term + curv_term
#         return weighted_with_penalty

#     # Weighted only
#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(log_cosh(r))
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + curv_term
#         return weighted_loss

#     # Unweighted + penalty
#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = y_pred - y
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + reg_term + curv_term
#         return unweighted_with_penalty

#     # Unweighted only
#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             data_term = jnp.nanmean(log_cosh(y_pred - y))
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + curv_term
#         return unweighted_loss


# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     huber_k: float = 2.0,   # Δ = huber_k * scale_estimate(r)
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using Adaptive Huber.
#     The worst one 
#     Args:
#         func:             Model function, called as func(xs, phys_params).
#         weighted:         If True, divide residuals by yerr.
#         penalty_function: Optional extra penalty, called as penalty_function(xs, params).
#         penalty_weight:   Multiplier for that penalty.
#         param_converter:  Optional raw→physical parameter converter.
#         huber_k:          Multiplier for adaptive Δ (default 1.0).

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar
#     """

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def adaptive_huber(r: jnp.ndarray) -> jnp.ndarray:
#         # 1) robust scale estimate via MAD
#         med  = jnp.nanmedian(r)
#         mad  = jnp.nanmedian(jnp.abs(r - med)) + 1e-8
#         scale = 1.4826 * mad

#         # 2) threshold Δ = k * scale
#         Δ = huber_k * scale

#         # 3) Huber formula
#         abs_r = jnp.abs(r)
#         return jnp.where(
#             abs_r <= Δ,
#             0.5 * r**2,
#             Δ * (abs_r - 0.5 * Δ)
#         )

#     def make_reg(xs, params):
#         if penalty_function:
#             return penalty_weight * penalty_function(xs, params) * 1e3
#         return 0.0

#     # Four‐branch logic, identical shape to your original:
#     if weighted and penalty_function:
#         def fn(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data     = jnp.nanmean(adaptive_huber(r))
#             return data + make_reg(xs, params)
#         return fn

#     elif weighted:
#         def fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(adaptive_huber(r))
#         return fn

#     elif penalty_function:
#         def fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y)
#             data   = jnp.nanmean(adaptive_huber(r))
#             return data + make_reg(xs, params)
#         return fn

#     else:
#         def fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y)
#             return jnp.nanmean(adaptive_huber(r))
#         return fn

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     cauchy_scale: float = 5.0,
#     curvature_weight: float = 10,    # new γ parameter
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using a Cauchy robust loss plus
#     an optional curvature penalty (second-derivative match).

#     Args:
#         func: model function, called as func(xs, params)
#         weighted: normalize residuals by yerr if True
#         penalty_function: optional regularizer func(xs, params)
#         penalty_weight: multiplier for that regularizer
#         param_converter: optional raw→phys parameter transformer
#         cauchy_scale: scale for robust Cauchy loss
#         curvature_weight: weight for curvature term
#     Returns:
#         loss(params, xs, y, yerr) → scalar
#     """

#     def cauchy_loss(r):
#         return jnp.log1p((r / cauchy_scale) ** 2)

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def curvature_term(y_pred, y):
#         # assumes last axis is the spectral pixel dimension
#         d2_pred = jnp.gradient(jnp.gradient(y_pred, axis=-1), axis=-1)
#         d2_obs  = jnp.gradient(jnp.gradient(y,      axis=-1), axis=-1)
#         return jnp.nanmean((d2_pred - d2_obs) ** 2)

#     # Weighted + penalty
#     if weighted and penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + reg_term + curv_term

#     # Weighted only
#     elif weighted:
#         def loss(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + curv_term

#     # Unweighted + penalty
#     elif penalty_function:
#         def loss(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = (y_pred - y)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + reg_term + curv_term

#     # Unweighted only
#     else:
#         def loss(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = (y_pred - y)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             curv_term = curvature_weight * curvature_term(y_pred, y)
#             return data_term + curv_term

#     return loss

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     cauchy_scale: float = 100.0,  # heavier clipping for smaller scale
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using a Cauchy robust loss.

#     Args:
#         func: The model function, called as func(xs, params)
#         weighted: If True, normalize residuals by yerr
#         penalty_function: Optional regularization function; called as penalty_function(xs, params)
#         penalty_weight: Multiplier for that regularization term
#         param_converter: Optional Parameters() object to transform raw → phys
#         cauchy_scale: Scale parameter for the Cauchy loss:
#                       loss(r) = log(1 + (r / cauchy_scale)**2)
#     Returns:
#         A loss(params, xs, y, yerr) -> scalar loss
#     """
#     def cauchy_loss(r):
#         # elementwise: log(1 + (r/scale)^2)
#         return jnp.log1p((r / cauchy_scale) ** 2)

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(cauchy_loss(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(cauchy_loss(r))
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = y_pred - y
#             data_term = jnp.nanmean(cauchy_loss(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = y_pred - y
#             return jnp.nanmean(cauchy_loss(r))
#         return unweighted_loss

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using L2 (“p=2”) residuals.

#     Args:
#         func: The model function, called as func(xs, params)
#         weighted: If True, normalize residuals by yerr
#         penalty_function: Optional regularization function; called as penalty_function(xs, params)
#         penalty_weight: Multiplier for that regularization term
#         param_converter: Optional Parameters() object to transform raw → phys

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar loss
#     """
#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred   = wrapped(xs, params)
#             r        = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(jnp.abs(r)**2)                            # <-- L2 here
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(jnp.abs(r)**2)                                # <-- L2 here
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred    = wrapped(xs, params)
#             r         = y_pred - y
#             data_term = jnp.nanmean(jnp.abs(r)**2)                          # <-- L2 here
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r      = y_pred - y
#             return jnp.nanmean(jnp.abs(r)**2)                                # <-- L2 here
#         return unweighted_loss
    
# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     huber_delta: float = 30,         
# ) -> Callable:
#     """
#     Build a JIT‐compiled loss function using a Huber (“rubber”) loss.

#     Args:
#         func: The model function, called as func(xs, params)
#         weighted: whether to divide by yerr
#         penalty_function: optional regularization term
#         penalty_weight: its multiplier
#         param_converter: optional raw → phys transform
#         huber_delta: transition point between quadratic and linear

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar
#     """
#     def huber(r):
#         abs_r = jnp.abs(r)
#         # quadratic for |r|<=δ, linear beyond
#         return jnp.where(abs_r <= huber_delta,
#                          0.5 * r**2,
#                          huber_delta * (abs_r - 0.5 * huber_delta))

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(huber(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(huber(r))
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y
#             data_term = jnp.nanmean(huber(r))
#             reg_term  = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             return jnp.nanmean(huber(y_pred - y))
#         return unweighted_loss
# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     residual_power: float = 10.0,    # Emphasis on large residuals
#     uncertainty_power: float = 1.0, # Emphasis on small uncertainties
#     huber_delta: float = 30.0,      # Transition point for Huber loss
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using Huber loss with hybrid adaptive weighting.

#     Args:
#         func: Model function, called as func(xs, params).
#         weighted: Whether to normalize residuals by yerr.
#         penalty_function: Optional regularization function.
#         penalty_weight: Multiplier for the regularization term.
#         param_converter: Optional raw → physical parameter converter.
#         residual_power: Controls emphasis on large residuals.
#         uncertainty_power: Controls emphasis on small uncertainties.
#         huber_delta: Threshold δ between quadratic and linear Huber regions.

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar loss
#     """

#     def huber(r):
#         abs_r = jnp.abs(r)
#         return jnp.where(
#             abs_r <= huber_delta,
#             0.5 * r**2,
#             huber_delta * (abs_r - 0.5 * huber_delta)
#         )

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def make_reg_term(params, xs):
#         # PPXF or any custom penalty can go here via penalty_function
#         if penalty_function:
#             return penalty_weight * penalty_function(xs, params) * 1e3
#         else:
#             return 0.0

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             rw = jnp.abs(residuals) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             w = rw * uw

#             data_term = jnp.nansum(huber(residuals) * w) / (jnp.nansum(w) + 1e-8)
#             return data_term + make_reg_term(params, xs)

#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             rw = jnp.abs(residuals) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             w = rw * uw

#             return jnp.nansum(huber(residuals) * w) / (jnp.nansum(w) + 1e-8)

#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = y_pred - y

#             rw = jnp.abs(residuals) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             w = rw * uw

#             data_term = jnp.nansum(huber(residuals) * w) / (jnp.nansum(w) + 1e-8)
#             return data_term + make_reg_term(params, xs)

#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = y_pred - y

#             rw = jnp.abs(residuals) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             w = rw * uw

#             return jnp.nansum(huber(residuals) * w) / (jnp.nansum(w) + 1e-8)

#         return unweighted_loss

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     residual_power: float = 2.0,      # emphasis on large residuals
#     uncertainty_power: float = 1.0,   # emphasis on small yerr
#     derivative_power: float = 2.0,    # emphasis on steep spectral gradients
#     huber_delta: float = 1e3,        # Huber transition
# ) -> Callable:
#     """
#     Build a JIT-compiled loss that
#       1) uses Huber for robust residuals,
#       2) adapts weights by residual magnitude,
#       3) adapts weights by 1/yerr,
#       4) adapts weights by |d(obs)/dx|,
#       5) optionally adds a penalty_function(xs, params).

#     Args:
#         func: model → flux, called func(xs, phys_params).
#         weighted: if True, normalize residuals by yerr.
#         penalty_function: extra reg term, signature (xs, params)->scalar.
#         penalty_weight: multiplier for penalty_function.
#         param_converter: raw→physical converter.
#         residual_power: power on |residual|.
#         uncertainty_power: power on 1/yerr.
#         derivative_power: power on |d(obs)/dx|.
#         huber_delta: δ for Huber loss.

#     Returns:
#         loss(params, xs, y, yerr) → scalar
#     """

#     def huber(r):
#         a = jnp.abs(r)
#         return jnp.where(a <= huber_delta,
#                          0.5 * r**2,
#                          huber_delta * (a - 0.5 * huber_delta))

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def make_penalty(xs, params):
#         if penalty_function:
#             return penalty_weight * penalty_function(xs, params) * 1e3
#         return 0.0

#     def feature_weight(y, xs):
#         # |d(obs)/dx|^derivative_power
#         # xs: wavelengths, y: observed flux (same shape)
#         grad = jnp.abs(jnp.gradient(y, xs))
#         return grad ** derivative_power

#     # four branches:
#     if weighted:
#         def loss_fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             rw = jnp.abs(r) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             dw = feature_weight(y, xs)

#             w = rw * uw * (1.0 + dw)   # add 1 so flat regions still count

#             data = jnp.nansum(huber(r) * w) / (jnp.nansum(w) + 1e-8)
#             return data + make_penalty(xs, params)

#     else:
#         def loss_fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y

#             rw = jnp.abs(r) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             dw = feature_weight(y, xs)

#             w = rw * uw * (1.0 + dw)

#             data = jnp.nansum(huber(r) * w) / (jnp.nansum(w) + 1e-8)
#             return data + make_penalty(xs, params)

#     return loss_fn

# def ppxf_smoothness_penalty(params: jnp.ndarray, lambda_reg: float = 1.0) -> jnp.ndarray:
#     """
#     Compute a PPXF-like smoothness penalty on `params`:
#       penalty = lambda_reg * sum( (params[i+2] - 2*params[i+1] + params[i])^2 ) over i
#     """
#     second_diff = jnp.diff(params, n=2)              # shape (N-2,)
#     penalty    = jnp.sum(second_diff**2)             # scalar
#     return lambda_reg * penalty
# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,  # ignore for now
#     penalty_weight: float = 0.01,   # multiplier for *that* penalty
#     param_converter: Optional["Parameters"] = None,
#     ppxf_lambda: float = 1.0,       # << new!
# ) -> Callable:
#     """
#     Build a JIT-compiled loss with optional PPXF smoothness penalty.
#     """
#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     # choose where to inject the PPXF penalty:
#     def make_reg_term(params):
#         # original penalty_function is still available if you want to combine both:
#         extra = 0.0
#         if penalty_function:
#             extra = penalty_weight * penalty_function(xs, params) * 1e3
#         # now add the PPXF smoothness:
#         smooth = ppxf_smoothness_penalty(params, lambda_reg=ppxf_lambda)
#         return smooth + extra

#     if weighted:
#         def weighted_with_ppxf(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data = jnp.nanmean(log_cosh(r))
#             return data + make_reg_term(params)
#         return weighted_with_ppxf

#     else:
#         def unweighted_with_ppxf(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             data = jnp.nanmean(log_cosh(y_pred - y))
#             return data + make_reg_term(params)
#         return unweighted_with_ppxf

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function.

#     Args:
#         func: The model function, called as func(xs, params)
#         param_converter: Optional Parameters() object to transform raw → phys

#     Returns:
#         A loss function with signature (params, xs, y, yerr) -> scalar loss
#     """
#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(log_cosh(r))
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             return jnp.nanmean(log_cosh(y_pred - y))
#         return unweighted_loss

# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     residual_power: float = 2.0,       # Emphasis on large residuals
#     uncertainty_power: float = 1.0,    # Emphasis on small uncertainties
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function using hybrid adaptive weighting (residual + uncertainty).

#     Args:
#         func: Model function, called as func(xs, params).
#         weighted: Whether to normalize residuals by uncertainties (yerr).
#         penalty_function: Optional regularization function.
#         penalty_weight: Multiplier for penalty term.
#         param_converter: Optional object to transform raw → physical parameters.
#         residual_power: Controls emphasis on large residuals (adaptive weighting).
#         uncertainty_power: Controls emphasis on uncertainties.

#     Returns:
#         A loss(params, xs, y, yerr) -> scalar loss
#     """

#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             residual_weights = jnp.abs(residuals) ** residual_power
#             uncertainty_weights = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             combined_weights = residual_weights * uncertainty_weights

#             data_term = jnp.nansum(log_cosh(residuals) * combined_weights) / (jnp.nansum(combined_weights) + 1e-8)

#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term

#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             residual_weights = jnp.abs(residuals) ** residual_power
#             uncertainty_weights = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             combined_weights = residual_weights * uncertainty_weights

#             return jnp.nansum(log_cosh(residuals) * combined_weights) / (jnp.nansum(combined_weights) + 1e-8)

#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = y_pred - y

#             residual_weights = jnp.abs(residuals) ** residual_power
#             uncertainty_weights = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             combined_weights = residual_weights * uncertainty_weights

#             data_term = jnp.nansum(log_cosh(residuals) * combined_weights) / (jnp.nansum(combined_weights) + 1e-8)

#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term

#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             residuals = y_pred - y

#             residual_weights = jnp.abs(residuals) ** residual_power
#             uncertainty_weights = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             combined_weights = residual_weights * uncertainty_weights

#             return jnp.nansum(log_cosh(residuals) * combined_weights) / (jnp.nansum(combined_weights) + 1e-8)

#         return unweighted_loss


# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
#     residual_power: float = 2.0,      # emphasis on large residuals
#     uncertainty_power: float = 1.0,   # emphasis on small yerr
#     derivative_power: float = 1.0,    # emphasis on steep spectral gradients
#     huber_delta: float = 15.0,        # Huber transition
# ) -> Callable:
#     """
#     Build a JIT-compiled loss that
#       1) uses Huber for robust residuals,
#       2) adapts weights by residual magnitude,
#       3) adapts weights by 1/yerr,
#       4) adapts weights by |d(obs)/dx|,
#       5) optionally adds a penalty_function(xs, params).

#     Args:
#         func: model → flux, called func(xs, phys_params).
#         weighted: if True, normalize residuals by yerr.
#         penalty_function: extra reg term, signature (xs, params)->scalar.
#         penalty_weight: multiplier for penalty_function.
#         param_converter: raw→physical converter.
#         residual_power: power on |residual|.
#         uncertainty_power: power on 1/yerr.
#         derivative_power: power on |d(obs)/dx|.
#         huber_delta: δ for Huber loss.

#     Returns:
#         loss(params, xs, y, yerr) → scalar
#     """

#     def huber(r):
#         a = jnp.abs(r)
#         return jnp.where(a <= huber_delta,
#                          0.5 * r**2,
#                          huber_delta * (a - 0.5 * huber_delta))

#     def wrapped(xs, raw_params):
#         phys = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys)

#     def make_penalty(xs, params):
#         if penalty_function:
#             return penalty_weight * penalty_function(xs, params) * 1e3
#         return 0.0

#     def feature_weight(y, xs):
#         # |d(obs)/dx|^derivative_power
#         # xs: wavelengths, y: observed flux (same shape)
#         grad = jnp.abs(jnp.gradient(y, xs))
#         return grad ** derivative_power

#     # four branches:
#     if weighted:
#         def loss_fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)

#             rw = jnp.abs(r) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             dw = feature_weight(y, xs)

#             w = rw * uw * (1.0 + dw)   # add 1 so flat regions still count

#             data = jnp.nansum(huber(r) * w) / (jnp.nansum(w) + 1e-8)
#             return data + make_penalty(xs, params)

#     else:
#         def loss_fn(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y

#             rw = jnp.abs(r) ** residual_power
#             uw = (1.0 / jnp.clip(yerr, 1e-8)) ** uncertainty_power
#             dw = feature_weight(y, xs)

#             w = rw * uw * (1.0 + dw)

#             data = jnp.nansum(huber(r) * w) / (jnp.nansum(w) + 1e-8)
#             return data + make_penalty(xs, params)

#     return loss_fn


# def build_loss_function(
#     func: Callable,
#     weighted: bool = True,
#     penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#     penalty_weight: float = 0.01,
#     param_converter: Optional["Parameters"] = None,
# ) -> Callable:
#     """
#     Build a JIT-compiled loss function.

#     Args:
#         func: The model function, called as func(xs, params)
#         param_converter: Optional Parameters() object to transform raw → phys

#     Returns:
#         A loss function with signature (params, xs, y, yerr) -> scalar loss
#     """
#     def log_cosh(x):
#         return jnp.logaddexp(x, -x) - jnp.log(2.0)

#     def wrapped(xs, raw_params):
#         phys_params = param_converter.raw_to_phys(raw_params) if param_converter else raw_params
#         return func(xs, phys_params)

#     if weighted and penalty_function:
#         def weighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return weighted_with_penalty

#     elif weighted:
#         def weighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = (y_pred - y) / jnp.clip(yerr, 1e-8)
#             return jnp.nanmean(log_cosh(r))
#         return weighted_loss

#     elif penalty_function:
#         def unweighted_with_penalty(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             r = y_pred - y
#             data_term = jnp.nanmean(log_cosh(r))
#             reg_term = penalty_weight * penalty_function(xs, params) * 1e3
#             return data_term + reg_term
#         return unweighted_with_penalty

#     else:
#         def unweighted_loss(params, xs, y, yerr):
#             y_pred = wrapped(xs, params)
#             return jnp.nanmean(log_cosh(y_pred - y))
#         return unweighted_loss
