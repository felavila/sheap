"""This module ?."""
__version__ = '0.1.0'
__author__ = 'Felipe Avila-Vera'
# Auto-generated __all__
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
        self.parsed_dependencies_tuple = parse_dependencies(self.list_dependencies)

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
                    return i < lbfgs_options.get("maxiter", 100)

                def body_fn(carry):
                    (params, state), loss_hist, i = carry
                    (params, state), loss = lbfgs_step((params, state))
                    loss_hist = loss_hist.at[i].set(loss)  # Store into preallocated array
                    return (params, state), loss_hist, i + 1

                # Preallocate the history buffer
                maxiter = lbfgs_options.get("maxiter", 100)
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

# from typing import Callable, Dict, List, Optional, Tuple

# import jax.numpy as jnp
# import optax
# from jax import jit, vmap,lax,value_and_grad
 
# from sheap.Assistants.parser_mapper import parse_dependencies, project_params
# from .loss_builder import build_loss_function




# class Minimizer:
#     """
#     Handles constrained optimization for a given model function using JAX and Optax.

#     Attributes
#     ----------
#     func : Callable
#         The model function to be optimized.
#     non_optimize_in_axis : int
#         Determines vmap axis behavior:
#         - 3: same initial values and constraints across data
#         - 4: same constraints, different initial values
#         - 5: different initial values and constraints
#     num_steps : int
#         Number of optimization iterations.
#     learning_rate : float
#         Learning rate for the optimizer.
#     list_dependencies : list of str
#         Parameter dependency specifications for tied parameters.
#     parsed_dependencies_tuple : tuple
#         Parsed dependency rules for parameter projection.
#     optimizer : optax.GradientTransformation
#         Optax optimizer instance.
#     loss_function : Callable
#         JIT-compiled loss function including penalties.
#     optimize_model : Callable
#         Function that performs the optimization loop.
#     """

#     def __init__(
#         self,
#         func: Callable,
#         non_optimize_in_axis: int = 3,
#         #constraints: Optional[Callable] = None,
#         num_steps: int = 1000,
#         #optimizer: optax.GradientTransformation = None,
#         learning_rate=None,
#         list_dependencies=[],
#         weighted=True,
#         **kwargs,):
        
        
#         """
#         Initialize the Minimizer with function and optimization settings.

#         Parameters
#         ----------
#         func : Callable
#             Model function that maps parameters and inputs to predictions.
#         non_optimize_in_axis : int, optional
#             Vmap axis mode (3, 4, or 5) controlling batching, by default 3.
#         num_steps : int, optional
#             Maximum number of optimization steps, by default 1000.
#         learning_rate : float, optional
#             Learning rate for gradient updates; defaults to 1e-3 if None.
#         list_dependencies : list of str, optional
#             Parameter tie definitions, by default empty list.
#         weighted : bool, optional
#             Whether to weight loss by inverse variance, by default True.
#         **kwargs : ?
#             Additional options: 'penalty_function', 'penalty_weight', 'param_converter', 'optimizer'.
#         """
        
#         self.func = func  # TODO desing the function class
#         self.non_optimize_in_axis = (
#             non_optimize_in_axis  # axis in where is require enter data of same dimension
#         )
#         self.num_steps = num_steps
#         self.learning_rate = learning_rate or 1e-3
#         self.list_dependencies = list_dependencies
#         self.parsed_dependencies_tuple = parse_dependencies(self.list_dependencies)
#         self.optimizer = kwargs.get("optimizer", optax.adam(self.learning_rate))
#         # print('optimizer:',self.optimizer)

#         self.loss_function, self.optimize_model = Minimizer.minimization_function(self.func,
#                                                     weighted=weighted,
#                                                     penalty_function=kwargs.get("penalty_function"),
#                                                     penalty_weight=kwargs.get("penalty_weight", 0.01),
#                                                     param_converter=kwargs.get("param_converter")
#                                                 )

#         #self.vmap_func = vmap(self.func, in_axes=(0, 0), out_axes=0)  # ?

#     def __call__(
#         self,
#         initial_params,
#         y,
#         x,
#         yerror,
#         constraints,
#         learning_rate=None,
#         num_steps=None,
#         optimizer=None,
#         non_optimize_in_axis=None,
#         list_dependencies=None,
#     ):
#         """
#         Run the optimization over batched data using vmap.

#         Parameters
#         ----------
#         initial_params : jnp.ndarray
#             Initial parameter array, shape (n_params,).
#         y : jnp.ndarray
#             Target data array, shape broadcastable to x.
#         x : jnp.ndarray
#             Input data array(s), same shape as y.
#         yerror : jnp.ndarray
#             Uncertainties in y, same shape as y.
#         constraints : jnp.ndarray
#             Array of shape (n_params, 2) specifying [min, max] bounds.
#         learning_rate : float, optional
#             Override learning rate for this call.
#         num_steps : int, optional
#             Override number of steps for this call.
#         optimizer : optax.GradientTransformation, optional
#             Override optimizer instance.
#         non_optimize_in_axis : int, optional
#             Override vmap axis mode.
#         list_dependencies : list of str, optional
#             Override parameter tie definitions.

#         Returns
#         -------
#         jnp.ndarray, list
#             Tuple of (optimized parameters for each batch, loss history list).
#         """
#         self.learning_rate = learning_rate or self.learning_rate
#         list_dependencies = list_dependencies or self.list_dependencies
#         self.parsed_dependencies_tuple = parse_dependencies(list_dependencies)
#         self.num_steps = num_steps or self.num_steps
#         self.optimizer = optimizer or self.optimizer
#         non_optimize_in_axis = non_optimize_in_axis or self.non_optimize_in_axis
#         #     schedule = optax.join_schedules(
#         # schedules=[
#         #     optax.linear_schedule(init_value=0.0, end_value=1e-3, transition_steps=500),
#         #     optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.95)
#         # ],
#         # boundaries=[500]
#         # )
#         # print("eje")
#         self.default_args = (
#             self.parsed_dependencies_tuple,
#             self.learning_rate,
#             self.num_steps,
#             self.optimizer,
#             False,
#         )

#         # print('learning_rate:',self.learning_rate)
#         # print('optimizer:',optax.adabelief.__name__)
#         # print('num_steps:',self.num_steps)

#         if non_optimize_in_axis == 3:
#             # print("vmap Optimize over y,x,yerror")
#             optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None)
#         elif non_optimize_in_axis == 4:
#             # print("vmap Optimize over init_val,y,x,yerror")
#             optimize_in_axis = (0, 0, 0, 0, None, None, None, None, None, None)
#         # elif non_optimize_in_axis==5:
#         #     #means the first values will be arrays
#         #     optimize_in_axis = (0, 0, 0, 0, None, None, None, None, None, None)
#         else:
#             print("This value of non_optimize_in_axis not cover it will replace for 3")
#             # print("vmap Optimize over y,x,yerror")
#             non_optimize_in_axis = 3
#             optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None)
#         self.optimize_in_axis = optimize_in_axis
#         vmap_optimize_model = vmap(self.optimize_model, in_axes=optimize_in_axis, out_axes=0)
#         jitted_vm = vmap_optimize_model#jit(vmap_optimize_model, static_argnums=(5, 6, 7, 8, 9))
#         return jitted_vm(
#             initial_params,
#             y,
#             x,
#             yerror,
#             constraints,
#             *self.default_args
#         )

#     @staticmethod
#     def minimization_function(
#         func: Callable[[List[jnp.ndarray], jnp.ndarray], jnp.ndarray],
#         penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#         penalty_weight: float = 0.01,
#         weighted: bool = True,
#         param_converter: Optional["Parameters"] = None,
#     ) -> Tuple[
#         Callable[..., jnp.ndarray],  # loss_function
#         Callable[..., Tuple[jnp.ndarray, list]],  # optimize_model
#     ]:
#         """
#         Factory to create JIT-compiled loss and optimizer functions.

#         Parameters
#         ----------
#         func : Callable
#             Model function mapping params and inputs to predictions.
#         penalty_function : Callable, optional
#             Function computing penalty given params.
#         penalty_weight : float, optional
#             Weight applied to penalty in loss, by default 0.01.
#         weighted : bool, optional
#             Whether to weight residuals by inverse variance.
#         param_converter : Parameters, optional
#             Converter for raw ↔ phys parameter transformations.

#         Returns
#         -------
#         loss_function : Callable
#             JIT-compiled function computing scalar loss.
#         optimize_model : Callable
#             Function that runs the optimization loop for given data.
#         """
#         loss_function = build_loss_function(func, weighted, penalty_function, penalty_weight, param_converter)
#         loss_function = jit(loss_function)

#         def optimize_model(
#             initial_params: jnp.ndarray,
#             xs: List[jnp.ndarray],  #
#             y: jnp.ndarray,
#             y_uncertainties: jnp.ndarray,
#             constraints: Optional[jnp.ndarray] = None,
#             parsed_dependencies = None,
#             learning_rate: float = 1e-2,
#             num_steps: int = 1000,
#             optimizer=None,
#             verbose: bool = False,
#         ) -> Tuple[jnp.ndarray, list]:
#             # Initialize parameters and optimizer state
#             params = initial_params
#             optimizer = optimizer or optax.adam(learning_rate)
#             opt_state = optimizer.init(params)
            
#             loss_history = []

#             if constraints is None:
#                 constraints = jnp.array([[-1e41, 1e41]] * params.shape[0])

#             # This works we can add it as a keyword
            
#             def step_fn(carry, _):
#                 #print("Tracing!")
#                 params, opt_state = carry
#                 loss, grads = value_and_grad(loss_function)(params, jnp.nan_to_num(xs), jnp.nan_to_num(y), y_uncertainties)
#                 updates, opt_state = optimizer.update(grads, opt_state, params)
#                 params = optax.apply_updates(params, updates)
#                 #params = project_params(params, constraints, parsed_dependencies)
#                 return (params, opt_state), loss

#             # Use lax.scan to run for num_steps
#             (final_params, _), loss_history = lax.scan(step_fn,(params, opt_state),None,length=num_steps,)
#             # loss_history: shape (num_steps,)

#             if verbose:
#                 print("Final loss:", loss_history[-1])

#             return final_params, loss_history
            
#             # @jit
#             # def step(params, opt_state, xs, y):
#             #     # Compute loss and gradients
#             #     loss, grads = jax.value_and_grad(loss_function)(
#             #         params, jnp.nan_to_num(xs), jnp.nan_to_num(y), y_uncertainties
#             #     )

#             #     updates, opt_state = optimizer.update(grads, opt_state, params)
#             #     params = optax.apply_updates(params, updates)

#             #     params = project_params(params, constraints, parsed_dependencies)
#             #     return params, opt_state, loss

#             # # Optimization loop
#             # for step_num in range(num_steps):
#             #     params, opt_state, loss = step(params, opt_state, xs, y)
#             #     loss_history.append(loss)
#             #     if step_num % 100 == 0 and verbose:
#             #         print(f"Step {step_num}, Loss: {loss:.4f}")

#             # return params, loss_history

#         return loss_function, optimize_model
