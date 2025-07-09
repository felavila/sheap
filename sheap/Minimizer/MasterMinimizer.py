from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from jax import jit, vmap,lax
#from jax import 
from .utils import parse_dependencies, project_params
from .loss_builder import build_loss_function

class MasterMinimizer:
    """
    MasterMinimizer handles constrained optimization for a given function using JAX and Optax.
    Attributes:
        func (Callable): The model function to optimize.
        optimize_in_axis (int) :
            -3  will be optimize assuming the same initial values for all and constraints
            -4  will be optimize assuming the same constraints values for all the function
            -5  will be optimize asuming different values of init and constraints
        penalty_weight (float): The weight for constraint penalties in the loss function.
        num_steps (int): The number of optimization steps.
        optimizer (optax.GradientTransformation): The optimizer to use for gradient-based optimization.
        constraints (Optional[Callable]): A function to compute constraints and their penalties.
        loss_function (Callable): The JIT-compiled loss function.
        optimize_model (Callable): The optimization routine.
        residuals (Callable): The residuals computation function.
        vmap_func (Callable): Vectorized version of the model function.
        vmap_optimize_model (Callable): Vectorized optimization model.
    """

    def __init__(
        self,
        func: Callable,
        non_optimize_in_axis: int = 3,
        #constraints: Optional[Callable] = None,
        num_steps: int = 1000,
        #optimizer: optax.GradientTransformation = None,
        learning_rate=None,
        list_dependencies=[],
        weighted=True,
        **kwargs,
    ):
        self.func = func  # TODO desing the function class
        self.non_optimize_in_axis = (
            non_optimize_in_axis  # axis in where is require enter data of same dimension
        )
        self.num_steps = num_steps
        self.learning_rate = learning_rate or 1e-3
        self.list_dependencies = list_dependencies
        self.parsed_dependencies_tuple = parse_dependencies(self.list_dependencies)
        self.optimizer = kwargs.get("optimizer", optax.adam(self.learning_rate))
        # print('optimizer:',self.optimizer)

        self.loss_function, self.optimize_model = MasterMinimizer.minimization_function(self.func,
                                                    weighted=weighted,
                                                    penalty_function=kwargs.get("penalty_function"),
                                                    penalty_weight=kwargs.get("penalty_weight", 0.01),
                                                    param_converter=kwargs.get("param_converter")
                                                )

        #self.vmap_func = vmap(self.func, in_axes=(0, 0), out_axes=0)  # ?

    def __call__(
        self,
        initial_params,
        y,
        x,
        yerror,
        constraints,
        learning_rate=None,
        num_steps=None,
        optimizer=None,
        non_optimize_in_axis=None,
        list_dependencies=None,
    ):
        """_summary_
        shapes initial_params (35,) y (413, 4633) x (413, 4633) yerror (413, 4633) constraints (35, 2)
        Args:
            initial_params (_type_): _description_
            y (_type_): _description_
            x (_type_): _description_
            yerror (_type_): _description_
            constraints (_type_): _description_
            parsed_dependencies_tuple (_type_, optional): _description_. Defaults to None.
            learning_rate (_type_, optional): _description_. Defaults to None.
            num_steps (_type_, optional): _description_. Defaults to None.
            optimizer (_type_, optional): _description_. Defaults to None.
            non_optimize_in_axis (_type_, optional): _description_. Defaults to None.
            list_dependencies (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.learning_rate = learning_rate or self.learning_rate
        list_dependencies = list_dependencies or self.list_dependencies
        self.parsed_dependencies_tuple = parse_dependencies(list_dependencies)
        self.num_steps = num_steps or self.num_steps
        self.optimizer = optimizer or self.optimizer
        non_optimize_in_axis = non_optimize_in_axis or self.non_optimize_in_axis
        #     schedule = optax.join_schedules(
        # schedules=[
        #     optax.linear_schedule(init_value=0.0, end_value=1e-3, transition_steps=500),
        #     optax.exponential_decay(init_value=1e-3, transition_steps=1000, decay_rate=0.95)
        # ],
        # boundaries=[500]
        # )
        # print("eje")
        self.default_args = (
            self.parsed_dependencies_tuple,
            self.learning_rate,
            self.num_steps,
            self.optimizer,
            False,
        )

        # print('learning_rate:',self.learning_rate)
        # print('optimizer:',optax.adabelief.__name__)
        # print('num_steps:',self.num_steps)

        if non_optimize_in_axis == 3:
            # print("vmap Optimize over y,x,yerror")
            optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None)
        elif non_optimize_in_axis == 4:
            # print("vmap Optimize over init_val,y,x,yerror")
            optimize_in_axis = (0, 0, 0, 0, None, None, None, None, None, None)
        # elif non_optimize_in_axis==5:
        #     #means the first values will be arrays
        #     optimize_in_axis = (0, 0, 0, 0, None, None, None, None, None, None)
        else:
            print("This value of non_optimize_in_axis not cover it will replace for 3")
            # print("vmap Optimize over y,x,yerror")
            non_optimize_in_axis = 3
            optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None)
        self.optimize_in_axis = optimize_in_axis
        vmap_optimize_model = vmap(self.optimize_model, in_axes=optimize_in_axis, out_axes=0)
        jitted_vm = vmap_optimize_model#jit(vmap_optimize_model, static_argnums=(5, 6, 7, 8, 9))
        return jitted_vm(
            initial_params,
            y,
            x,
            yerror,
            constraints,
            *self.default_args
        )
        #return vmap_optimize_model(
         #   initial_params, y, x, yerror, constraints, *self.default_args
        #)

    @staticmethod
    def minimization_function(
        func: Callable[[List[jnp.ndarray], jnp.ndarray], jnp.ndarray],
        penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        penalty_weight: float = 0.01,
        weighted: bool = True,
        param_converter: Optional["Parameters"] = None,
    ) -> Tuple[
        Callable[..., jnp.ndarray],  # loss_function
        Callable[..., Tuple[jnp.ndarray, list]],  # optimize_model
    ]:
        """
        Factory function to create a JIT-compiled constrained loss function with multiple input variables.

        Parameters:
        - func: The model function to compute predictions, accepts multiple x inputs as a list of arrays.
        - penalty_weight: Weight for the bound violation penalty.
        Returns:
        - A JIT-compiled loss function.
        TODO:
        - be carefull with uncertainty and weight
        """
        loss_function = build_loss_function(func, weighted, penalty_function, penalty_weight, param_converter)
        loss_function = jit(loss_function)

        def optimize_model(
            initial_params: jnp.ndarray,
            xs: List[jnp.ndarray],  #
            y: jnp.ndarray,
            y_uncertainties: jnp.ndarray,
            constraints: Optional[jnp.ndarray] = None,
            parsed_dependencies=None,
            learning_rate: float = 1e-2,
            num_steps: int = 1000,
            optimizer=None,
            verbose: bool = False,
        ) -> Tuple[jnp.ndarray, list]:
            # Initialize parameters and optimizer state
            params = initial_params
            optimizer = optimizer or optax.adam(learning_rate)
            opt_state = optimizer.init(params)
            
            loss_history = []

            if constraints is None:
                constraints = jnp.array([[-1e41, 1e41]] * params.shape[0])

            # This works we can add it as a keyword
            
            def step_fn(carry, _):
                #print("Tracing!")
                params, opt_state = carry
                loss, grads = jax.value_and_grad(loss_function)(params, jnp.nan_to_num(xs), jnp.nan_to_num(y), y_uncertainties)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                #params = project_params(params, constraints, parsed_dependencies)
                return (params, opt_state), loss

            # Use lax.scan to run for num_steps
            (final_params, _), loss_history = lax.scan(step_fn,(params, opt_state),None,length=num_steps,)
            # loss_history: shape (num_steps,)

            if verbose:
                print("Final loss:", loss_history[-1])

            return final_params, loss_history
            
            # @jit
            # def step(params, opt_state, xs, y):
            #     # Compute loss and gradients
            #     loss, grads = jax.value_and_grad(loss_function)(
            #         params, jnp.nan_to_num(xs), jnp.nan_to_num(y), y_uncertainties
            #     )

            #     updates, opt_state = optimizer.update(grads, opt_state, params)
            #     params = optax.apply_updates(params, updates)

            #     params = project_params(params, constraints, parsed_dependencies)
            #     return params, opt_state, loss

            # # Optimization loop
            # for step_num in range(num_steps):
            #     params, opt_state, loss = step(params, opt_state, xs, y)
            #     loss_history.append(loss)
            #     if step_num % 100 == 0 and verbose:
            #         print(f"Step {step_num}, Loss: {loss:.4f}")

            # return params, loss_history

        return loss_function, optimize_model


# class MasterMinimizer_old:
#     """
#     MasterMinimizer handles constrained optimization for a given function using JAX and Optax.
#     Attributes:
#         func (Callable): The model function to optimize.
#         optimize_in_axis (int) :
#             -3  will be optimize assuming the same initial values for all and constraints
#             -4  will be optimize assuming the same constraints values for all the function
#             -5  will be optimize asuming different values of init and constraints
#         penalty_weight (float): The weight for constraint penalties in the loss function.
#         num_steps (int): The number of optimization steps.
#         optimizer (optax.GradientTransformation): The optimizer to use for gradient-based optimization.
#         constraints (Optional[Callable]): A function to compute constraints and their penalties.
#         loss_function (Callable): The JIT-compiled loss function.
#         optimize_model (Callable): The optimization routine.
#         residuals (Callable): The residuals computation function.
#         vmap_func (Callable): Vectorized version of the model function.
#         vmap_optimize_model (Callable): Vectorized optimization model.
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
#         **kwargs,
#     ):
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

#         self.loss_function, self.optimize_model = (
#             MasterMinimizer.minimization_function(
#                 self.func,
#                 weighted=weighted,
#                 penalty_function=kwargs.get("penalty_function"),
#                 penalty_weight=kwargs.get("penalty_weight", 0.01),
#             )
#         )

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
#         """_summary_
#         shapes initial_params (35,) y (413, 4633) x (413, 4633) yerror (413, 4633) constraints (35, 2)
#         Args:
#             initial_params (_type_): _description_
#             y (_type_): _description_
#             x (_type_): _description_
#             yerror (_type_): _description_
#             constraints (_type_): _description_
#             parsed_dependencies_tuple (_type_, optional): _description_. Defaults to None.
#             learning_rate (_type_, optional): _description_. Defaults to None.
#             num_steps (_type_, optional): _description_. Defaults to None.
#             optimizer (_type_, optional): _description_. Defaults to None.
#             non_optimize_in_axis (_type_, optional): _description_. Defaults to None.
#             list_dependencies (_type_, optional): _description_. Defaults to None.

#         Returns:
#             _type_: _description_
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
#         #return vmap_optimize_model(
#          #   initial_params, y, x, yerror, constraints, *self.default_args
#         #)

#     @staticmethod
#     def minimization_function(
#         func: Callable[[List[jnp.ndarray], jnp.ndarray], jnp.ndarray],
#         penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#         penalty_weight: float = 0.01,
#         weighted: bool = True,
#     ) -> Tuple[
#         Callable[..., jnp.ndarray],  # loss_function
#         Callable[..., Tuple[jnp.ndarray, list]],  # optimize_model
#     ]:
#         """
#         Factory function to create a JIT-compiled constrained loss function with multiple input variables.

#         Parameters:
#         - func: The model function to compute predictions, accepts multiple x inputs as a list of arrays.
#         - penalty_weight: Weight for the bound violation penalty.
#         Returns:
#         - A JIT-compiled loss function.
#         TODO:
#         - be carefull with uncertainty and weight
#         """
#         loss_function = build_loss_function(func, weighted, penalty_function, penalty_weight)
#         loss_function = jit(loss_function)

#         def optimize_model(
#             initial_params: jnp.ndarray,
#             xs: List[jnp.ndarray],  #
#             y: jnp.ndarray,
#             y_uncertainties: jnp.ndarray,
#             constraints: Optional[jnp.ndarray] = None,
#             parsed_dependencies=None,
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
#                 loss, grads = jax.value_and_grad(loss_function)(
#                 params, jnp.nan_to_num(xs), jnp.nan_to_num(y), y_uncertainties)
#                 updates, opt_state = optimizer.update(grads, opt_state, params)
#                 params = optax.apply_updates(params, updates)
#                 params = project_params(params, constraints, parsed_dependencies)
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


# class MasterMinimizer:
#     """
#     MasterMinimizer handles constrained optimization for a given function using JAX and Optax.

#     Attributes:
#         func (Callable): The model function to optimize.
#         optimize_in_axis (int) :
#             -3  will optimize assuming the same initial values for all and constraints
#             -4  will optimize assuming the same constraints values for all the function
#             -5  will optimize assuming different values of init and constraints
#         penalty_weight (float): The weight for constraint penalties in the loss function.
#         num_steps (int): The number of optimization steps.
#         optimizer (optax.GradientTransformation): The optimizer to use for gradient-based optimization.
#         loss_function (Callable): The JIT-compiled loss function.
#         optimize_model (Callable): The optimization routine.
#     """

#     def __init__(
#         self,
#         func: Callable,
#         non_optimize_in_axis: int = 3,
#         num_steps: int = 1000,
#         learning_rate: Optional[float] = None,
#         list_dependencies: Optional[List[str]] = None,
#         weighted: bool = True,
#         **kwargs,
#     ):
#         self.func = func
#         self.non_optimize_in_axis = non_optimize_in_axis
#         self.num_steps = num_steps
#         self.learning_rate = learning_rate or 1e-3
#         self.list_dependencies = list_dependencies or []
#         self.parsed_dependencies_tuple = parse_dependencies(self.list_dependencies)
#         # Default to Adam unless another optimizer is provided
#         self.optimizer = kwargs.get("optimizer", optax.adam(self.learning_rate))

#         # Build loss and optimizer function
#         self.loss_function, self.optimize_model = MasterMinimizer.minimization_function(
#             self.func,
#             weighted=weighted,
#             penalty_function=kwargs.get("penalty_function"),
#             penalty_weight=kwargs.get("penalty_weight", 0.01),
#         )

#     def __call__(
#         self,
#         initial_params: jnp.ndarray,
#         y: jnp.ndarray,
#         x: jnp.ndarray,
#         yerror: jnp.ndarray,
#         constraints: jnp.ndarray,
#         learning_rate: Optional[float] = None,
#         num_steps: Optional[int] = None,
#         optimizer: Optional[optax.GradientTransformation] = None,
#         non_optimize_in_axis: Optional[int] = None,
#         list_dependencies: Optional[List[str]] = None,
#     ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#         # Update run-time settings
#         self.learning_rate = learning_rate or self.learning_rate
#         list_dependencies = list_dependencies or self.list_dependencies
#         self.parsed_dependencies_tuple = parse_dependencies(list_dependencies)
#         self.num_steps = num_steps or self.num_steps
#         self.optimizer = optimizer or self.optimizer
#         non_optimize_in_axis = non_optimize_in_axis or self.non_optimize_in_axis

#         # Determine vmap axes for batched optimization
#         if non_optimize_in_axis == 3:
#             optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None)
#         elif non_optimize_in_axis == 4:
#             optimize_in_axis = (0, 0, 0, 0, None, None, None, None, None, None)
#         else:
#             # fall back to axis=3
#             optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None)

#         # Vectorized optimization call
#         vmap_opt = vmap(self.optimize_model, in_axes=optimize_in_axis, out_axes=0)
#         return vmap_opt(
#             initial_params,
#             y,
#             x,
#             yerror,
#             constraints,
#             parse_dependencies(list_dependencies),
#             self.learning_rate,
#             self.num_steps,
#             self.optimizer,
#             False,
#         )

#     @staticmethod
#     def minimization_function(
#         func: Callable[[List[jnp.ndarray], jnp.ndarray], jnp.ndarray],
#         penalty_function: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
#         penalty_weight: float = 0.01,
#         weighted: bool = True,
#         param_converter: Optional[object] = None,
#     ) -> Tuple[
#         Callable[..., jnp.ndarray],
#         Callable[..., Tuple[jnp.ndarray, jnp.ndarray]],
#     ]:
#         # Build and JIT the loss function
#         loss_fn = build_loss_function(func, weighted, penalty_function, penalty_weight, param_converter)
#         loss_fn = jit(loss_fn)

#         def optimize_model(
#             initial_params: jnp.ndarray,
#             xs: List[jnp.ndarray],
#             y: jnp.ndarray,
#             y_unc: jnp.ndarray,
#             constraints: Optional[jnp.ndarray] = None,
#             parsed_deps=None,
#             learning_rate: float = 1e-2,
#             num_steps: int = 1000,
#             optimizer: optax.GradientTransformation = None,
#             verbose: bool = False,
#         ) -> Tuple[jnp.ndarray, jnp.ndarray]:
#             params = initial_params
#             optimizer = optimizer or optax.adabelief(learning_rate)
#             opt_state = optimizer.init(params)

#             if constraints is None:
#                 constraints = jnp.array([[-1e41, 1e41]] * params.shape[0])

#             # Check if this optimizer supports L-BFGS signature
#             import inspect
#             sig = inspect.signature(optimizer.update)

#             if "value_fn" in sig.parameters:
#                 # L-BFGS style loop
#                 val_and_grad = optax.value_and_grad_from_state(loss_fn)

#                 def lbfgs_step(carry, _):
#                     p, st = carry
#                     val, grads = val_and_grad(p, state=st)
#                     updates, st = optimizer.update(
#                         grads,
#                         st,
#                         p,
#                         value=val,
#                         grad=grads,
#                         value_fn=lambda q: loss_fn(q, xs, y, y_unc),
#                     )
#                     p = optax.apply_updates(p, updates)
#                     return (p, st), val

#                 (final_p, _), history = lax.scan(
#                     lbfgs_step, (params, opt_state), None, length=num_steps
#                 )
#             else:
#                 # First-order optimizer loop (Adam, AdaBelief, etc.)
#                 def step_fn(carry, _):
#                     p, st = carry
#                     loss, grads = jax.value_and_grad(loss_fn)(p, xs, y, y_unc)
#                     updates, st = optimizer.update(grads, st, p)
#                     p = optax.apply_updates(p, updates)
#                     p = project_params(p, constraints, parsed_deps)
#                     return (p, st), loss

#                 (final_p, _), history = lax.scan(
#                     step_fn, (params, opt_state), None, length=num_steps
#                 )

#             if verbose:
#                 print("Final loss:", history[-1])

#             return final_p, history

#         return loss_fn, optimize_model