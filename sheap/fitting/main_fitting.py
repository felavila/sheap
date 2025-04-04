# import jax
# import jax.numpy as jnp
# from jax import jit,vmap, lax
# import optax
# from typing import Callable, Dict, Tuple,List
# from jax import random
# from SHEAP.utils import *
# #jax.config.update("jax_enable_x64", True)
# #from SHEAP.fiter import project_params

# class MasterMinimizer:
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

#     def __init__(self,func: Callable,non_optimize_in_axis: int = 3,constraints: Optional[Callable] = None,\
#         penalty_weight: float = 1e6,num_steps: int = 100,optimizer: optax.GradientTransformation = None,learning_rate=1e-1,**kwargs,):
#         self.func = func
#         self.non_optimize_in_axis = non_optimize_in_axis
#         self.penalty_weight = penalty_weight
#         self.num_steps = num_steps
#         self.learning_rate = learning_rate
#         self.optimizer = optimizer or optax.adabelief(1e-1)
#         self.inequality_threshold =  1e-6
#         if non_optimize_in_axis==3:
#             self.optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
#         elif non_optimize_in_axis==4:
#             #means the first values will be arrays
#             self.optimize_in_axis = (0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
#         elif non_optimize_in_axis==5:
#             #means the first values will be arrays
#             self.optimize_in_axis = (0, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
#         else:
#             print("This value of non_optimize_in_axis not cover it will replace for 3")
#             self.non_optimize_in_axis = 3
#             self.optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None)
#         self.loss_function, self.optimize_model, self.residuals = MasterMinimizer.minimization_function(func)
#         self.vmap_func = vmap(self.func, in_axes=(0, 0), out_axes=0)
#         self.vmap_optimize_model = vmap(self.optimize_model,in_axes=self.optimize_in_axis,out_axes=0)
#         self.default_args = (jnp.array([]),
#             jnp.array([]),
#             jnp.array([]),
#             jnp.array([]),
#             jnp.array([]),
#             jnp.array([]),
#             jnp.array([]),
#             self.inequality_threshold,
#             self.learning_rate,
#             self.num_steps,
#             self.penalty_weight,
#             self.optimizer,
#             False)
        
#     #def kwargs? 
    
#     @staticmethod
#     def minimization_function(
#     func: Callable[[List[jnp.ndarray], jnp.ndarray], jnp.ndarray],
#     penalty_weight: float = 1e6
#     ) -> Callable[..., jnp.ndarray]:
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
#         @jit
#         def residuals(params: jnp.ndarray, xs: List[jnp.ndarray], y: jnp.ndarray, y_uncertainties: jnp.ndarray):
#             predictions = func(xs, params)
#             return jnp.abs(y - predictions) / y_uncertainties

#         @jit
#         def loss_function(
#             params: jnp.ndarray,
#             xs: List[jnp.ndarray],
#             y: jnp.ndarray,
#             y_uncertainties: jnp.ndarray,
#             constraints: jnp.ndarray,
#             inequality_constraints: jnp.ndarray = jnp.array([]),
#             penalty_weight: float = 1e6,
#             inequality_threshold: float = 1e-6
#         ) -> jnp.ndarray:
#             y_pred = func(xs, params)
#             #r = (y - y_pred)/y_uncertainties
#             wmse = jnp.nanmean(((y - y_pred)/y_uncertainties) ** 2)
#             #wmse = jnp.nanmean(jnp.abs(r)) #the bestone
#             #tau = 0.5
#             #c = 2.0
#             #delta=1.0
#             #swmse = jnp.nanmean(jnp.where(jnp.abs(r) <= delta, 0.5 * r**2, delta * (jnp.abs(r) - 0.5 * delta)))
#             return wmse 
#             lower_bounds = constraints[:, 0]
#             upper_bounds = constraints[:, 1]
#             penalties_lower = jnp.where((params < lower_bounds),(lower_bounds - params) ** 2,0.0)
#             penalties_upper = jnp.where((params > upper_bounds),(params - upper_bounds) ** 2,0.0)
            
#             if inequality_constraints.size > 0:
#                 # Extract indices
#                 i_indices = inequality_constraints[:, 0]
#                 j_indices = inequality_constraints[:, 1]

#                 # Get the corresponding parameter values
#                 theta_i = params[i_indices]
#                 theta_j = params[j_indices]

#                 # Compute the difference theta_j - theta_i
#                 diff = theta_j - theta_i

#                 # Compute penalties where theta_i >= theta_j - threshold
#                 # We allow a small threshold to account for numerical precision
#                 penalties_ineq = jnp.where(
#                     diff <= inequality_threshold,
#                     (inequality_threshold + theta_i - theta_j) ** 2,
#                     0.0
#                 )
#                 penalty_ineq = jnp.sum(penalties_ineq)
#             else:
#                 penalty_ineq = 0.0
#             #penalties_lower = jnp.where(params < lower_bounds, (lower_bounds - params) ** 2, 0.0)
#             #penalties_upper = jnp.where(params > upper_bounds, (params - upper_bounds) ** 2, 0.0)
#             penalty = jnp.sum(penalties_lower + penalties_upper)

#             return wmse + penalty_weight * penalty

#         def optimize_model(
#             initial_params: jnp.ndarray,
#             xs: List[jnp.ndarray],
#             y: jnp.ndarray,
#             y_uncertainties: jnp.ndarray,
#             constraints: jnp.ndarray= None,
#             multiplicative_dep_indices: jnp.ndarray= jnp.array([]),
#             multiplicative_indep_indices: jnp.ndarray= jnp.array([]),
#             multiplicative_multipliers: jnp.ndarray= jnp.array([]),
#             additive_dep_indices: jnp.ndarray= jnp.array([]),
#             additive_indep_indices: jnp.ndarray= jnp.array([]),
#             additive_deltas: jnp.ndarray= jnp.array([]),
#             inequality_constraints: jnp.ndarray = jnp.array([]),
#             inequality_threshold: float = 1e-6,
#             learning_rate: float = 1e-2,
#             num_steps: int = 1000,
#             penalty_weight: float = 1e6,
#             optimizer = optax.adam(1e-2),
#             verbose: bool = False
#         ) -> Tuple[jnp.ndarray, list]:
#             # Initialize parameters and optimizer state
#             params = initial_params
#             opt_state = optimizer.init(params)
#             loss_history = []
#             if constraints is None:
#                 constraints = jnp.array([[-1e41,1e41]] * params.shape[0])
#             # Define the step function with constraints captured via closure
#             @jit
#             def step(params, opt_state, xs, y):
#                 # Compute loss and gradients
#                 loss, grads = jax.value_and_grad(loss_function)(
#                     params, jnp.nan_to_num(xs), jnp.nan_to_num(y), y_uncertainties, constraints, penalty_weight=penalty_weight
#                 )

#                 # Compute parameter updates
#                 updates, opt_state = optimizer.update(grads, opt_state, params)
#                 params = optax.apply_updates(params, updates)

#                 # Project parameters to enforce constraints
#                 params = project_params(
#                     params,
#                     constraints,
#                     multiplicative_dep_indices,
#                     multiplicative_indep_indices,
#                     multiplicative_multipliers,
#                     additive_dep_indices,
#                     additive_indep_indices,
#                     additive_deltas
#                 )
#                 return params, opt_state, loss

#             # Optimization loop
#             for step_num in range(num_steps):
#                 params, opt_state, loss = step(params, opt_state, xs, y)
#                 loss_history.append(loss)
#                 if step_num % 100 == 0 and verbose:
#                     print(f"Step {step_num}, Loss: {loss:.4f}")

#             return params, loss_history

#         return loss_function, optimize_model, residuals

# @jit
# def project_params(
#     params: jnp.ndarray,
#     constraints: jnp.ndarray,
#     multiplicative_dep_indices: jnp.ndarray= jnp.array([]),
#     multiplicative_indep_indices: jnp.ndarray= jnp.array([]),
#     multiplicative_multipliers: jnp.ndarray= jnp.array([]),
#     additive_dep_indices: jnp.ndarray= jnp.array([]),
#     additive_indep_indices: jnp.ndarray= jnp.array([]),
#     additive_deltas: jnp.ndarray= jnp.array([])
# ) -> jnp.ndarray:
#     """
#     Project flat parameters to satisfy individual bounds and apply multiplicative and additive constraints.

#     Parameters:
#     - params: Flat array of parameters.
#     - constraints: Array of (lower, upper) bounds for each parameter.
#     - multiplicative_dep_indices: Array of dependent parameter indices for multiplicative constraints.
#     - multiplicative_indep_indices: Array of independent parameter indices for multiplicative constraints.
#     - multiplicative_multipliers: Array of multipliers for multiplicative constraints.
#     - additive_dep_indices: Array of dependent parameter indices for additive constraints.
#     - additive_indep_indices: Array of independent parameter indices for additive constraints.
#     - additive_deltas: Array of deltas for additive constraints.

#     Returns:
#     - Projected params: Flat array with constraints enforced.
#     """
#     # Apply individual bounds
#     lower_bounds = constraints[:, 0]
#     upper_bounds = constraints[:, 1]
#     params = jnp.clip(params, lower_bounds, upper_bounds)

#     # Enforce multiplicative constraints (vectorized)
#     if multiplicative_dep_indices.size > 0:
#         # Compute new values for dependent parameters
#         new_dep_params = multiplicative_multipliers * params[multiplicative_indep_indices]
#         # Clip to bounds
#         new_dep_params = jnp.clip(new_dep_params, lower_bounds[multiplicative_dep_indices], upper_bounds[multiplicative_dep_indices])
#         # Update dependent parameters
#         params = params.at[multiplicative_dep_indices].set(new_dep_params)

#     # Enforce additive constraints (vectorized)
#     if additive_dep_indices.size > 0:
#         # Compute new values for dependent parameters
#         new_dep_params = params[additive_indep_indices] + additive_deltas
#         # Clip to bounds
#         new_dep_params = jnp.clip(new_dep_params, lower_bounds[additive_dep_indices], upper_bounds[additive_dep_indices])
#         # Update dependent parameters
#         params = params.at[additive_dep_indices].set(new_dep_params)

#     # Re-apply individual bounds after constraints
#     params = jnp.clip(params, lower_bounds, upper_bounds)

#     return params


# def minimization_function(
#     func: Callable[[List[jnp.ndarray], jnp.ndarray], jnp.ndarray],
#     penalty_weight: float = 1e6
# ) -> Callable[..., jnp.ndarray]:
#     """
#     Factory function to create a JIT-compiled constrained loss function with multiple input variables.

#     Parameters:
#     - func: The model function to compute predictions, accepts multiple x inputs as a list of arrays.
#     - penalty_weight: Weight for the bound violation penalty.

#     Returns:
#     - A JIT-compiled loss function.
#     """
#     @jit
#     def residuals(params: jnp.ndarray, xs: List[jnp.ndarray], y: jnp.ndarray, y_uncertainties: jnp.ndarray):
#         predictions = func(xs, params)
#         return jnp.abs(y - predictions) / y_uncertainties

#     @jit
#     def loss_function(
#         params: jnp.ndarray,
#         xs: List[jnp.ndarray],
#         y: jnp.ndarray,
#         y_uncertainties: jnp.ndarray,
#         constraints: jnp.ndarray,
#         inequality_constraints: jnp.ndarray = jnp.array([]),
#         penalty_weight: float = 1e6,
#         inequality_threshold: float = 1e-6
#     ) -> jnp.ndarray:
#         # Compute model predictions with multiple inputs
#         y_pred = func(xs, params)

#         # Compute Weighted Mean Squared Error
#         wmse = jnp.nanmean(((y - y_pred) / y_uncertainties) ** 2)

#         # Compute penalties for each parameter
#         lower_bounds = constraints[:, 0]
#         upper_bounds = constraints[:, 1]
#         penalties_lower = jnp.where((params < lower_bounds),(lower_bounds - params) ** 2,0.0)
#         penalties_upper = jnp.where((params > upper_bounds),(params - upper_bounds) ** 2,0.0)
#         # Compute penalties for inequality constraints
#         if inequality_constraints.size > 0:
#             # Extract indices
#             i_indices = inequality_constraints[:, 0]
#             j_indices = inequality_constraints[:, 1]

#             # Get the corresponding parameter values
#             theta_i = params[i_indices]
#             theta_j = params[j_indices]

#             # Compute the difference theta_j - theta_i
#             diff = theta_j - theta_i

#             # Compute penalties where theta_i >= theta_j - threshold
#             # We allow a small threshold to account for numerical precision
#             penalties_ineq = jnp.where(
#                 diff <= inequality_threshold,
#                 (inequality_threshold + theta_i - theta_j) ** 2,
#                 0.0
#             )
#             penalty_ineq = jnp.sum(penalties_ineq)
#         else:
#             penalty_ineq = 0.0
#         #penalties_lower = jnp.where(params < lower_bounds, (lower_bounds - params) ** 2, 0.0)
#         #penalties_upper = jnp.where(params > upper_bounds, (params - upper_bounds) ** 2, 0.0)
#         penalty = jnp.sum(penalties_lower + penalties_upper)

#         return wmse + penalty_weight * penalty

#     def optimize_model(
#         initial_params: jnp.ndarray,
#         xs: List[jnp.ndarray],
#         y: jnp.ndarray,
#         y_uncertainties: jnp.ndarray,
#         constraints: jnp.ndarray= jnp.array([]),
#         multiplicative_dep_indices: jnp.ndarray= jnp.array([]),
#         multiplicative_indep_indices: jnp.ndarray= jnp.array([]),
#         multiplicative_multipliers: jnp.ndarray= jnp.array([]),
#         additive_dep_indices: jnp.ndarray= jnp.array([]),
#         additive_indep_indices: jnp.ndarray= jnp.array([]),
#         additive_deltas: jnp.ndarray= jnp.array([]),
#         inequality_constraints: jnp.ndarray = jnp.array([]),
#         inequality_threshold: float = 1e-6,
#         learning_rate: float = 1e-2,
#         num_steps: int = 1000,
#         penalty_weight: float = 1e6,
#         optimizer = optax.adam(1e-2),
#         verbose: bool = False
#     ) -> Tuple[jnp.ndarray, list]:
#         # Initialize parameters and optimizer state
#         params = initial_params
#         opt_state = optimizer.init(params)
#         loss_history = []

#         # Define the step function with constraints captured via closure
#         @jit
#         def step(params, opt_state, xs, y):
#             # Compute loss and gradients
#             loss, grads = jax.value_and_grad(loss_function)(
#                 params, jnp.nan_to_num(xs), jnp.nan_to_num(y), y_uncertainties, constraints, penalty_weight=penalty_weight
#             )

#             # Compute parameter updates
#             updates, opt_state = optimizer.update(grads, opt_state, params)
#             params = optax.apply_updates(params, updates)

#             # Project parameters to enforce constraints
#             params = project_params(
#                 params,
#                 constraints,
#                 multiplicative_dep_indices,
#                 multiplicative_indep_indices,
#                 multiplicative_multipliers,
#                 additive_dep_indices,
#                 additive_indep_indices,
#                 additive_deltas
#             )
#             return params, opt_state, loss

#         # Optimization loop
#         for step_num in range(num_steps):
#             params, opt_state, loss = step(params, opt_state, xs, y)
#             loss_history.append(loss)
#             if step_num % 100 == 0 and verbose:
#                 print(f"Step {step_num}, Loss: {loss:.4f}")

#         return params, loss_history

#     return loss_function, optimize_model, residuals