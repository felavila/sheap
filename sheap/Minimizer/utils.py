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
