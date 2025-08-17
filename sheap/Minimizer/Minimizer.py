"""This module Contains the main minimization routines in sheap."""
__author__ = 'felavila'

__all__ = [
    "Minimizer",
]

from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import optax
from jax import jit, vmap, lax, value_and_grad

from sheap.Assistants.parser_mapper import parse_dependencies, project_params
from .loss_builder import build_loss_function


class Minimizer:
    """
    Handles constrained optimization for a given model function using JAX and Optax.

    Attributes
    ----------
    func : Callable
        The model function to be optimized.
    non_optimize_in_axis : int
        Determines vmap axis behavior:
        - 3: same initial values and constraints across data
        - 4: same constraints, different initial values
        - 5: different initial values and constraints
    num_steps : int
        Number of optimization iterations.
    learning_rate : float
        Learning rate for the optimizer (ignored for LBFGS).
    list_dependencies : list of str
        Parameter dependency specifications for tied parameters.
    method : str
        Optimization method to use ('adam' or 'lbfgs').
    lbfgs_options : dict
        Options specific to LBFGS optimization (e.g., maxiter, tolerance_grad).
    optimizer : optax.GradientTransformation
        Optax optimizer instance.
    loss_function : Callable
        JIT-compiled loss function including penalties.
    optimize_model : Callable
        Function that performs the optimization loop.
    """

    def __init__(
        self,
        func: Callable,
        non_optimize_in_axis: int = 3,
        num_steps: int = 1_000,
        learning_rate: Optional[float] = None,
        list_dependencies: List[str] = [],
        weighted: bool = True,
        method: str = "adam",
        lbfgs_options: Optional[Dict] = None,
        penalty_function: Optional[Callable] = None,
        param_converter: Optional["Parameters"] = None,
        penalty_weight: float = 0.01,
        curvature_weight: float = 1e3,
        smoothness_weight: float = 1e5,
        max_weight: float = 0.1,
        **kwargs,
    ):
        self.func = func
        self.non_optimize_in_axis = non_optimize_in_axis
        self.num_steps = num_steps
        self.learning_rate = learning_rate or 1e-2
        self.list_dependencies = list_dependencies
        self.method = method.lower()
        self.lbfgs_options = lbfgs_options or {}
        #self.optimizer = kwargs.get("optimizer", optax.adam(self.learning_rate))
        #print(method,penalty_weight,curvature_weight,smoothness_weight,max_weight)
        #self.parsed_dependencies_tuple = parse_dependencies(self.list_dependencies)

        self.loss_function, self.optimize_model = Minimizer.minimization_function(
            self.func,
            weighted=weighted,
            penalty_function=penalty_function,
            penalty_weight=penalty_weight,
            param_converter=param_converter,
            curvature_weight=curvature_weight,
            learning_rate = learning_rate,
            smoothness_weight=smoothness_weight,
            max_weight=max_weight,
            method=self.method,
            lbfgs_options=self.lbfgs_options,
            num_steps = num_steps
        )

    def __call__(
        self,
        initial_params,
        y,
        x,
        yerror,
        constraints,
    ):
        """
        Execute the optimization process across batches.

        Parameters
        ----------
        initial_params : jnp.ndarray
            Initial parameters for optimization.
        y : jnp.ndarray
            Observed data values.
        x : jnp.ndarray
            Wavelength or independent variable.
        yerror : jnp.ndarray
            Uncertainty for each observation.
        constraints : jnp.ndarray
            Parameter constraints, shape (N_params, 2).

        Returns
        -------
        jnp.ndarray
            Optimized parameters.
        list
            Final loss history.
        """
        optimize_in_axis = (
            (None, 0, 0, 0, None)
            if self.non_optimize_in_axis == 3
            else (0, 0, 0, 0, None)
        )

        vmap_optimize_model = vmap(
            self.optimize_model, in_axes=optimize_in_axis, out_axes=0
        )

        return vmap_optimize_model(
            initial_params,
            y,
            x,
            yerror,
            constraints,
        )

    @staticmethod
    def minimization_function(
        func: Callable,
        weighted: bool,
        penalty_function: Optional[Callable],
        penalty_weight: float,
        param_converter: Optional["Parameters"],
        curvature_weight: float,
        learning_rate : float,
        smoothness_weight: float,
        max_weight: float,
        method: str,
        lbfgs_options: dict,
        num_steps
    ) -> Tuple[Callable, Callable]:
        """
        Builds the loss function and corresponding optimization routine.

        Parameters
        ----------
        func : Callable
            The model function.
        weighted : bool
            Whether to apply inverse variance weighting.
        penalty_function : Callable, optional
            Optional penalty function for parameters.
        penalty_weight : float
            Scalar penalty strength.
        param_converter : Parameters, optional
            Object to convert raw to physical parameters.
        curvature_weight : float
            Strength of curvature matching regularization.
        smoothness_weight : float
            Strength of smoothness regularization.
        max_weight : float
            Penalty on worst residual.
        method : str
            Optimizer method ('adam' or 'lbfgs').
        lbfgs_options : dict
            Dictionary of LBFGS-specific options.

        Returns
        -------
        Tuple[Callable, Callable]
            The compiled loss function and optimization routine.
        """

        loss_function = build_loss_function(
            func,
            weighted,
            penalty_function,
            penalty_weight,
            param_converter,
            curvature_weight,
            smoothness_weight,
            max_weight,
        )
        loss_function = jit(loss_function)

        def optimize_model(initial_params, xs, y, y_uncertainties, constraints):
            loss_history = []

            if method == "lbfgs":
                optimizer = optax.lbfgs(**lbfgs_options)
                state = optimizer.init(initial_params)

                def lbfgs_step(carry):
                    params, state = carry
                    loss, grads = value_and_grad(loss_function)(params, xs, y, y_uncertainties)
                    updates, state = optimizer.update(
                        grads, state, params,
                        value=loss,
                        grad=grads,
                        value_fn=lambda p: loss_function(p, xs, y, y_uncertainties)
                    )
                    params = optax.apply_updates(params, updates)
                    return (params, state), loss

                def cond_fn(carry):
                    (_, _), _, i = carry
                    return i < lbfgs_options.get("maxiter", 200)

                def body_fn(carry):
                    (params, state), loss_hist, i = carry
                    (params, state), loss = lbfgs_step((params, state))
                    loss_hist = loss_hist.at[i].set(loss)  # Store into preallocated array
                    return (params, state), loss_hist, i + 1

                # Preallocate the history buffer
                maxiter = lbfgs_options.get("maxiter", 200)
                loss_hist_init = jnp.zeros((maxiter,), dtype=jnp.float64)

                # Run loop
                ((final_params, _), loss_history, _i) = lax.while_loop(
                    cond_fn,
                    body_fn,
                    ((initial_params, state), loss_hist_init, 0)
)

            else:  # adam
                #here should go a way to choose as a dictionary the name of the optimizer.
                optimizer = optax.adam(learning_rate=learning_rate)
                opt_state = optimizer.init(initial_params)

                def step_fn(carry, _):
                    params, opt_state = carry
                    loss, grads = value_and_grad(loss_function)(params, xs, y, y_uncertainties)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, opt_state), loss

                (final_params, _), loss_history = lax.scan(
                    step_fn, (initial_params, opt_state), None, length=num_steps
                )

            return final_params, loss_history

        return loss_function, optimize_model

