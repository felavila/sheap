"""
Minimization Routines
=====================

This module contains the main minimization routines in *sheap*.
It defines the `Minimizer` class, which wraps JAX and Optax
optimizers for constrained spectral model fitting.

Contents
--------
- **Minimizer**: high-level interface for fitting spectral models with
  Adam or LBFGS optimizers.
- **Loss Function**: constructed via `loss_builder.build_loss_function`,
  supporting weighted residuals, penalties, and regularization terms.
- **Vectorization**: optimization can be run across batches via `jax.vmap`.
- **Constraints & Dependencies**: supports tied parameters and physical
  constraints through `Parameters` converters.

Notes
-----
- Optimization supports two methods:
  - `"adam"` (gradient descent with adaptive moments, default)
  - `"lbfgs"` (quasi-Newton optimizer via Optax)
- Regularization options include:
  curvature matching, smoothness penalties, and maximum residual weighting.
- `non_optimize_in_axis` controls how constraints and initial conditions
  are shared across batched spectra:
  
  * 3 → same initial values and constraints  
  * 4 → same constraints, different initial values  
  * 5 → both constraints and initial values vary

Example
-------
.. code-block:: python

   from sheap.Minimizer.Minimizer import Minimizer

   minimizer = Minimizer(model_fn, num_steps=2000, learning_rate=1e-2)
   final_params, loss_history = minimizer(
       initial_params, flux, wavelength, errors, constraints
   )
"""

__author__ = 'felavila'

__all__ = [
    "Minimizer",
]

from typing import Callable, Dict, List, Optional, Tuple

import jax.numpy as jnp
import optax
from jax import jit, vmap, lax, value_and_grad

#from sheap.Assistants.parser_mapper import parse_dependencies, project_params
from .loss_builder import build_loss_function
from .LB_nonlinear import build_varpro_loss_from_profile_and_params_obj

class Minimizer:
    """
    Handles constrained optimization for a given model function using JAX and Optax.
    #TODO maybe for one object remove the JIT
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
        #list_dependencies: List[str] = [],
        weighted: bool = True,
        method: str = "adam",
        lbfgs_options: Optional[Dict] = None,
        penalty_function: Optional[Callable] = None,
        param_converter: Optional["Parameters"] = None,
        penalty_weight: float = 0.01,
        curvature_weight: float = 1e3,
        smoothness_weight: float = 1e5,
        max_weight: float = 0.1,
        batch_mode: str = "independent",
        convergence_options: Optional[Dict] = None,
        global_reduction: str = "sum",
        **kwargs,
    ):
        self.func = func
        self.non_optimize_in_axis = non_optimize_in_axis
        self.num_steps = num_steps
        self.learning_rate = learning_rate or 1e-2
        #TODO param_converter ->param_class 
        self.param_converter = param_converter
        self.method = method.lower()
        self.lbfgs_options = lbfgs_options or {}
        self.batch_mode = batch_mode.lower()
        self.global_reduction = global_reduction.lower()
        self.weighted = weighted
        self.penalty_weight = penalty_weight
        self.curvature_weight=curvature_weight

        valid_batch_modes = {"independent", "global_independent"}
        if self.batch_mode not in valid_batch_modes:
            raise ValueError(
                f"batch_mode must be one of {sorted(valid_batch_modes)}, "
                f"not {batch_mode!r}."
            )

        if self.global_reduction not in {"sum", "mean"}:
            raise ValueError("global_reduction must be 'sum' or 'mean'.")

        self.convergence_options = {
            "loss_tolerance": 1e-6,
            "gradient_tolerance": 1e-4,
            "patience": 10,
            "min_steps": 100,
            "required_fraction": 1.0,
        }
        if convergence_options is not None:
            self.convergence_options.update(convergence_options)

        required_fraction = self.convergence_options["required_fraction"]
        if not 0.0 < required_fraction <= 1.0:
            raise ValueError("required_fraction must be in the interval (0, 1].")

        
        self.kwargs = vars(self).copy()
        self.kwargs.pop("func")
        self.kwargs.pop("param_converter")

        self.loss_function, self.optimize_model = Minimizer.minimization_function(self.func, weighted=self.weighted, penalty_function=penalty_function, 
                                                                                  penalty_weight=self.penalty_weight,param_converter=self.param_converter,
            curvature_weight=self.curvature_weight, learning_rate=self.learning_rate, smoothness_weight=smoothness_weight, max_weight=max_weight,
            method=self.method, lbfgs_options=self.lbfgs_options, num_steps = num_steps)

        optimize_in_axis = (
            (None, 0, 0, 0, None)
            if self.non_optimize_in_axis == 3
            else (0, 0, 0, 0, None)
        )

        self.independent_batch_optimizer = vmap(
            self.optimize_model,
            in_axes=optimize_in_axis,
            out_axes=0,
        )

        self.optimize_global_model = None
        self.global_batch_optimizer = None
        self.batch_optimizers = {"independent": self.independent_batch_optimizer,}

        # The global-independent implementation currently uses Adam. Build it
        # for every Adam Minimizer so either batch callable remains available.
        if self.method == "adam":
            self.optimize_global_model = (
                Minimizer.global_independent_minimization_function(
                    loss_function=self.loss_function,
                    learning_rate=self.learning_rate,
                    max_steps=self.num_steps,
                    convergence_options=self.convergence_options,
                    reduction=self.global_reduction,
                )
            )
            self.global_batch_optimizer = self._global_batch_optimizer
            self.batch_optimizers[
                "global_independent"
            ] = self.global_batch_optimizer

        if self.batch_mode == "global_independent" and self.method != "adam":
            raise NotImplementedError(
                "global_independent currently supports method='adam' only."
            )
       

    def __call__(self, initial_params, x, y, yerror, constraints):
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
            Optimized physical parameters.
        optimizer_output
            Loss histories in independent mode or a convergence diagnostics
            dictionary in global-independent mode.
        """
        has_shared_parameters = bool(
            getattr(self.param_converter, "_any_shared", False)
        )

        if has_shared_parameters:
            if self.batch_mode == "global_independent":
                raise ValueError(
                    "global_independent requires a separate parameter vector "
                    "for every object. Use batch_mode='independent' for the "
                    "shared-parameter branch."
                )
            return self._optimize_shared_parameters(
                initial_params,
                x,
                y,
                yerror,
                constraints,
            )

        if self.batch_mode not in self.batch_optimizers:
            raise ValueError(
                f"No optimizer is available for batch_mode={self.batch_mode!r}."
            )

        raw_initial_params = initial_params
        if self.param_converter is not None:
            raw_initial_params = self.param_converter.phys_to_raw(
                initial_params
            )

        batch_optimizer = self.batch_optimizers[self.batch_mode]
        raw_params, optimizer_output = batch_optimizer(
            raw_initial_params,
            x,
            y,
            yerror,
            constraints,
        )

        if self.param_converter is not None:
            final_params = self.param_converter.raw_to_phys(raw_params)
        else:
            final_params = raw_params

        return final_params, optimizer_output

    def _global_batch_optimizer(
        self,
        initial_params,
        x,
        y,
        yerror,
        constraints,
    ):
        """Adapt the global optimizer to the common batch-call signature."""
        del constraints
        return self.optimize_global_model(
            initial_params,
            x,
            y,
            yerror,
        )

    def _optimize_shared_parameters(
        self,
        initial_params,
        x,
        y,
        yerror,
        constraints,
    ):
        """Optimize a packed vector containing physically shared parameters."""
        del constraints

        print(
            "Running the shared-parameter method; this mode is experimental."
        )

        parameter_converter = self.param_converter
        model_vmap = vmap(self.func)

        if initial_params is None:
            physical_initial_params = parameter_converter.phys_init()
        else:
            physical_initial_params = initial_params

        raw_initial_params = parameter_converter.phys_to_raw(
            physical_initial_params
        )

        def shared_loss(raw_vector):
            physical_params = parameter_converter.raw_to_phys(raw_vector)
            model_params = [
                physical_params[
                    :,
                    parameter_converter.names.index(parameter_name),
                ]
                for parameter_name in parameter_converter.params_dict
            ]

            prediction = model_vmap(x, model_params)
            residuals = (y - prediction) / yerror
            object_chi2 = jnp.sum(residuals * residuals, axis=1)
            return jnp.sum(object_chi2)

        shared_loss_and_grad = jit(value_and_grad(shared_loss))
        optimizer = optax.adam(learning_rate=self.learning_rate)

        def optimize_shared(raw_params):
            optimizer_state = optimizer.init(raw_params)

            def step_fn(carry, _):
                params, state = carry
                loss, gradients = shared_loss_and_grad(params)
                updates, state = optimizer.update(
                    gradients,
                    state,
                    params,
                )
                params = optax.apply_updates(params, updates)
                return (params, state), loss

            (final_raw_params, _), loss_history = lax.scan(
                step_fn,
                (raw_params, optimizer_state),
                None,
                length=self.num_steps,
            )
            final_loss = shared_loss(final_raw_params)
            return final_raw_params, final_loss, loss_history

        optimize_shared = jit(optimize_shared)
        raw_params, final_loss, loss_history = optimize_shared(
            raw_initial_params
        )

        physical_params = parameter_converter.raw_to_phys(raw_params)
        diagnostics = {
            "final_loss": final_loss,
            "loss_history": loss_history,
            "steps_completed": jnp.asarray(
                self.num_steps,
                dtype=jnp.int32,
            ),
        }
        return physical_params, diagnostics

    @staticmethod
    def global_independent_minimization_function(
        loss_function: Callable,
        learning_rate: float,
        max_steps: int,
        convergence_options: Dict,
        reduction: str = "sum",
    ) -> Callable:
        """Build a joint optimizer for independently parameterized objects.

        The objective is the sum (or mean) of the individual-object losses,
        but every object retains its own parameter vector. Consequently,
        gradients cannot mix physical parameters between different objects.

        Convergence is evaluated for every object using both the relative loss
        change and the gradient norm. The complete optimization stops once the
        requested fraction of objects has remained converged for ``patience``
        iterations, or after ``max_steps``. The best parameter vector visited
        for each object is returned, including for objects that do not satisfy
        the convergence criterion.

        Notes
        -----
        ``constraints`` is not an argument to ``loss_function`` in the current
        Minimizer API. Physical bounds must therefore be enforced by the
        parameter converter or by penalties included in ``loss_function``.
        """
        loss_tolerance = float(convergence_options["loss_tolerance"])
        gradient_tolerance = float(
            convergence_options["gradient_tolerance"]
        )
        patience = int(convergence_options["patience"])
        min_steps = int(convergence_options["min_steps"])
        required_fraction = float(
            convergence_options["required_fraction"]
        )

        if patience < 1:
            raise ValueError("patience must be at least 1.")
        if min_steps < 0:
            raise ValueError("min_steps cannot be negative.")

        per_object_loss = vmap(
            loss_function,
            in_axes=(0, 0, 0, 0),
            out_axes=0,
        )

        def objective_with_aux(params, xs, y, y_uncertainties):
            object_losses = per_object_loss(
                params,
                xs,
                y,
                y_uncertainties,
            )
            if reduction == "mean":
                total_loss = jnp.mean(object_losses)
            else:
                total_loss = jnp.sum(object_losses)
            return total_loss, object_losses

        objective_and_grad = jit(
            value_and_grad(objective_with_aux, has_aux=True)
        )
        evaluate_object_losses = jit(per_object_loss)
        optimizer = optax.adam(learning_rate=learning_rate)

        def optimize_batch(initial_params, xs, y, y_uncertainties):
            initial_params = jnp.asarray(initial_params)
            if initial_params.ndim == 1:
                initial_params = jnp.broadcast_to(
                    initial_params[None, :],
                    (xs.shape[0], initial_params.shape[0]),
                )
            elif initial_params.ndim != 2:
                raise ValueError(
                    "global_independent expects initial_params with shape "
                    "(n_parameters,) or (n_objects, n_parameters)."
                )

            if initial_params.shape[0] != xs.shape[0]:
                raise ValueError(
                    "The number of parameter rows must match the number "
                    "of objects in x, y, and y_uncertainties."
                )

            initial_losses = evaluate_object_losses(
                initial_params,
                xs,
                y,
                y_uncertainties,
            )
            n_objects = initial_params.shape[0]

            opt_state = optimizer.init(initial_params)
            previous_losses = jnp.full_like(initial_losses, jnp.inf)
            best_losses = jnp.full_like(initial_losses, jnp.inf)
            best_params = initial_params
            stable_counts = jnp.zeros((n_objects,), dtype=jnp.int32)

            history_dtype = initial_losses.dtype
            total_loss_history = jnp.full(
                (max_steps,),
                jnp.nan,
                dtype=history_dtype,
            )
            fraction_converged_history = jnp.full(
                (max_steps,),
                jnp.nan,
                dtype=history_dtype,
            )

            initial_carry = (
                initial_params,
                opt_state,
                previous_losses,
                best_params,
                best_losses,
                stable_counts,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0, dtype=history_dtype),
                total_loss_history,
                fraction_converged_history,
            )

            def condition(carry):
                step = carry[6]
                fraction_converged = carry[7]
                below_maximum = step < max_steps
                needs_more_steps = step < min_steps
                needs_more_objects = (
                    fraction_converged < required_fraction
                )
                return below_maximum & (
                    needs_more_steps | needs_more_objects
                )

            def body(carry):
                (
                    params,
                    opt_state,
                    previous_losses,
                    best_params,
                    best_losses,
                    stable_counts,
                    step,
                    _,
                    total_loss_history,
                    fraction_converged_history,
                ) = carry

                (total_loss, object_losses), gradients = objective_and_grad(
                    params,
                    xs,
                    y,
                    y_uncertainties,
                )

                finite_previous = jnp.isfinite(previous_losses)
                relative_loss_change = jnp.where(
                    finite_previous,
                    jnp.abs(object_losses - previous_losses)
                    / jnp.maximum(jnp.abs(previous_losses), 1e-12),
                    jnp.inf,
                )

                gradient_norm = jnp.linalg.norm(
                    gradients.reshape((n_objects, -1)),
                    axis=1,
                )
                stable_now = (
                    (relative_loss_change <= loss_tolerance)
                    & (gradient_norm <= gradient_tolerance)
                    & jnp.isfinite(object_losses)
                    & jnp.isfinite(gradient_norm)
                )
                stable_counts = jnp.where(
                    stable_now,
                    stable_counts + 1,
                    0,
                )
                object_converged = stable_counts >= patience
                fraction_converged = jnp.mean(
                    object_converged.astype(history_dtype)
                )

                improved = object_losses < best_losses
                best_losses = jnp.where(
                    improved,
                    object_losses,
                    best_losses,
                )
                best_params = jnp.where(
                    improved[:, None],
                    params,
                    best_params,
                )

                updates, opt_state = optimizer.update(
                    gradients,
                    opt_state,
                    params,
                )
                params = optax.apply_updates(params, updates)

                total_loss_history = total_loss_history.at[step].set(
                    total_loss
                )
                fraction_converged_history = (
                    fraction_converged_history.at[step].set(
                        fraction_converged
                    )
                )

                return (
                    params,
                    opt_state,
                    object_losses,
                    best_params,
                    best_losses,
                    stable_counts,
                    step + 1,
                    fraction_converged,
                    total_loss_history,
                    fraction_converged_history,
                )

            final_carry = lax.while_loop(
                condition,
                body,
                initial_carry,
            )
            (
                final_params,
                _,
                previous_losses,
                best_params,
                best_losses,
                stable_counts,
                steps_completed,
                _,
                total_loss_history,
                fraction_converged_history,
            ) = final_carry

            # Include the parameters produced by the last optimizer update in
            # the best-per-object comparison and final convergence test.
            (final_total_loss, final_losses), final_gradients = (
                objective_and_grad(
                    final_params,
                    xs,
                    y,
                    y_uncertainties,
                )
            )
            final_improved = final_losses < best_losses
            best_losses = jnp.where(
                final_improved,
                final_losses,
                best_losses,
            )
            best_params = jnp.where(
                final_improved[:, None],
                final_params,
                best_params,
            )

            final_relative_change = (
                jnp.abs(final_losses - previous_losses)
                / jnp.maximum(jnp.abs(previous_losses), 1e-12)
            )
            final_gradient_norm = jnp.linalg.norm(
                final_gradients.reshape((n_objects, -1)),
                axis=1,
            )
            final_stable = (
                (final_relative_change <= loss_tolerance)
                & (final_gradient_norm <= gradient_tolerance)
                & jnp.isfinite(final_losses)
                & jnp.isfinite(final_gradient_norm)
            )
            stable_counts = jnp.where(
                final_stable,
                stable_counts + 1,
                0,
            )
            object_converged = stable_counts >= patience
            fraction_converged = jnp.mean(
                object_converged.astype(history_dtype)
            )

            diagnostics = {
                "steps_completed": steps_completed,
                "converged": fraction_converged >= required_fraction,
                "fraction_converged": fraction_converged,
                "object_converged": object_converged,
                "best_object_losses": best_losses,
                "best_total_loss": jnp.sum(best_losses),
                "final_total_loss": final_total_loss,
                "final_gradient_norm": final_gradient_norm,
                "total_loss_history": total_loss_history,
                "fraction_converged_history": fraction_converged_history,
            }
            return best_params, diagnostics

        return jit(optimize_batch)

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
        #build_varpro_loss_function
        loss_function = jit(build_loss_function(func,weighted,penalty_function,penalty_weight,param_converter,curvature_weight,smoothness_weight,max_weight,))
        #loss_function = jit(build_varpro_loss_function(func,weighted,penalty_function,penalty_weight,param_converter,curvature_weight,smoothness_weight,max_weight,))
        #loss_function = jit(loss_function)
        loss_and_grad = jit(value_and_grad(loss_function))
        
        def optimize_model(initial_params, xs, y, y_uncertainties, constraints):
            #Why this works slow?
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
                    loss, grads = loss_and_grad(params, xs, y, y_uncertainties) #value_and_grad(loss_function)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, opt_state), loss

                (final_params, _), loss_history = lax.scan(
                    step_fn, (initial_params, opt_state), None, length=num_steps
                )

            return final_params, loss_history
        optimize_model = jit(optimize_model) #powerfull when we apply montecarlo-in in 1-2 objects sample not much impact +3 sec
        return loss_function, optimize_model

    @staticmethod
    def minimization_function2(
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

        loss_function = jit(build_varpro_loss_function(func,weighted,penalty_function,penalty_weight,param_converter,curvature_weight,smoothness_weight,max_weight,))
        #loss_function = jit(loss_function)
        loss_and_grad = jit(value_and_grad(loss_function))
        
        def optimize_model(initial_params, xs, y, y_uncertainties, constraints):
            #Why this works slow?
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
                    loss, grads = loss_and_grad(params, xs, y, y_uncertainties) #value_and_grad(loss_function)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, opt_state), loss

                (final_params, _), loss_history = lax.scan(
                    step_fn, (initial_params, opt_state), None, length=num_steps
                )

            return final_params, loss_history
        optimize_model = jit(optimize_model) #powerfull when we apply montecarlo-in in 1-2 objects sample not much impact +3 sec
        return loss_function, optimize_model


  
class Minimizer_c:
    """
    Handles constrained optimization for a given model function using JAX and Optax.
    #TODO maybe for one object remove the JIT
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
        #list_dependencies: List[str] = [],
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
        #self.list_dependencies = list_dependencies
        self.param_converter = param_converter
        self.method = method.lower()
        self.lbfgs_options = lbfgs_options or {}
        #self.optimizer = kwargs.get("optimizer", optax.adam(self.learning_rate))
        #print(method,penalty_weight,curvature_weight,smoothness_weight,max_weight)
        #self.parsed_dependencies_tuple = parse_dependencies(self.list_dependencies)

        self.loss_function, self.optimize_model = Minimizer.minimization_function(self.func, weighted=weighted, penalty_function=penalty_function, penalty_weight=penalty_weight,param_converter=self.param_converter,
            curvature_weight=curvature_weight, learning_rate = learning_rate, smoothness_weight=smoothness_weight, max_weight=max_weight,
            method=self.method, lbfgs_options=self.lbfgs_options, num_steps = num_steps)

    def __call__(self, initial_params, x, y, yerror, constraints):
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
        if self.param_converter._any_shared:
            print("Runing shared parameter method it is in a experimental face experimental.")
            sp_model_vmap = vmap(self.func)
            P = self.param_converter
            raw0 = P.raw_init()  # packed 1D raw vector (handled internally)
            raw0 = P.phys_to_raw(P.phys_init())  # packed 1D raw vector (handled internally)
            def loss_fn(raw_vec):
                phys = P.raw_to_phys(raw_vec)
                params = [phys[:, P.names.index(p)] for p in P.params_dict]
                yhat = sp_model_vmap(x,params)
                r = (y - yhat) /yerror
                chi2 = jnp.sum(r * r, axis=1)
                return jnp.mean(chi2)
            loss_and_grad = jit(value_and_grad(loss_fn))
            opt = optax.adam(learning_rate=0.05) #we keep this for now
            state = opt.init(raw0)
            raw = raw0
            for step in range(self.num_steps):
                val, g = loss_and_grad(raw)
                updates, state = opt.update(g, state, raw)
                raw = optax.apply_updates(raw, updates)
    
            params = P.raw_to_phys(raw)
            return params,0
        else:
            optimize_in_axis = (
                        (None, 0, 0, 0, None)
                        if self.non_optimize_in_axis == 3
                        else (0, 0, 0, 0, None)
                    )
            vmap_optimize_model = vmap(self.optimize_model, in_axes=optimize_in_axis, out_axes=0)
            if self.param_converter:
                initial_params = self.param_converter.phys_to_raw(initial_params)
                raw_params,loss = vmap_optimize_model(initial_params,x,y,yerror,constraints,)
                return self.param_converter.raw_to_phys(raw_params),loss
            else:
                #print warning sayng about no param class is defined
                return vmap_optimize_model(initial_params,x,y,yerror,constraints,)

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
        #build_varpro_loss_function
        loss_function = jit(build_loss_function(func,weighted,penalty_function,penalty_weight,param_converter,curvature_weight,smoothness_weight,max_weight,))
        #loss_function = jit(build_varpro_loss_function(func,weighted,penalty_function,penalty_weight,param_converter,curvature_weight,smoothness_weight,max_weight,))
        #loss_function = jit(loss_function)
        loss_and_grad = jit(value_and_grad(loss_function))
        
        def optimize_model(initial_params, xs, y, y_uncertainties, constraints):
            #Why this works slow?
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
                    loss, grads = loss_and_grad(params, xs, y, y_uncertainties) #value_and_grad(loss_function)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, opt_state), loss

                (final_params, _), loss_history = lax.scan(
                    step_fn, (initial_params, opt_state), None, length=num_steps
                )

            return final_params, loss_history
        optimize_model = jit(optimize_model) #powerfull when we apply montecarlo-in in 1-2 objects sample not much impact +3 sec
        return loss_function, optimize_model

    @staticmethod
    def minimization_function2(
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

        loss_function = jit(build_varpro_loss_function(func,weighted,penalty_function,penalty_weight,param_converter,curvature_weight,smoothness_weight,max_weight,))
        #loss_function = jit(loss_function)
        loss_and_grad = jit(value_and_grad(loss_function))
        
        def optimize_model(initial_params, xs, y, y_uncertainties, constraints):
            #Why this works slow?
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
                    loss, grads = loss_and_grad(params, xs, y, y_uncertainties) #value_and_grad(loss_function)
                    updates, opt_state = optimizer.update(grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return (params, opt_state), loss

                (final_params, _), loss_history = lax.scan(
                    step_fn, (initial_params, opt_state), None, length=num_steps
                )

            return final_params, loss_history
        optimize_model = jit(optimize_model) #powerfull when we apply montecarlo-in in 1-2 objects sample not much impact +3 sec
        return loss_function, optimize_model


    

class Minimizer_:
    """
    Handles constrained optimization for a given model function using JAX and Optax.
    #TODO maybe for one object remove the JIT
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
        print("The experimental one")
        self.func = func
        self.non_optimize_in_axis = non_optimize_in_axis
        self.num_steps = num_steps
        self.learning_rate = learning_rate or 1e-2
        self.list_dependencies = list_dependencies
        self.param_converter = param_converter
        self.method = method.lower()
        self.lbfgs_options = lbfgs_options or {}
        #self.optimizer = kwargs.get("optimizer", optax.adam(self.learning_rate))
        #print(method,penalty_weight,curvature_weight,smoothness_weight,max_weight)
        #self.parsed_dependencies_tuple = parse_dependencies(self.list_dependencies)

        self.nonlinear_raw_idx = self.func.nonlinear_param_indices
        self.loss_fn, self.solve_fn = build_varpro_loss_from_profile_and_params_obj(fused_profile=self.func, params_obj=self.param_converter, nonlinear_raw_idx=self.nonlinear_raw_idx, weighted=True,lambda_reg=0.0,reg_matrix=None,)
        
        self.loss_function, self.optimize_model = Minimizer.minimization_function(self.func, weighted=weighted, penalty_function=penalty_function, penalty_weight=penalty_weight,param_converter=self.param_converter,
            curvature_weight=curvature_weight, learning_rate = learning_rate, smoothness_weight=smoothness_weight, max_weight=max_weight,
            method=self.method, lbfgs_options=self.lbfgs_options, num_steps = num_steps,loss_fn=self.loss_fn)

    def __call__(self, initial_params, y, x, yerror, constraints,):
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
        optimize_in_axis = ((None, 0, 0, 0, None) if self.non_optimize_in_axis == 3 else (0, 0, 0, 0, None))
        
        
        vmap_optimize_model = vmap(self.optimize_model, in_axes=optimize_in_axis, out_axes=0)
        vmap_solve_fn = vmap(self.solve_fn, in_axes=(0, 0, 0, 0), out_axes=0)
        print(initial_params.shape)
        if self.param_converter:
            raw0_full = jnp.asarray(self.param_converter.phys_to_raw(initial_params))
            initial_params = raw0_full.at[:,jnp.array([*self.nonlinear_raw_idx ])].get()
            #initial_params = self.param_converter.phys_to_raw(initial_params)
            raw_params,loss = vmap_optimize_model(initial_params,y,x,yerror,constraints,)
            phys_best, _, _, _, _, _ = vmap_solve_fn(raw_params,y,x,yerror )
            return phys_best,loss
        else:
            #print warning sayng about no param class is defined
            return vmap_optimize_model(initial_params,y,x,yerror,constraints,)

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
        num_steps,loss_fn
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
        #build_varpro_loss_function
        loss_function = jit(loss_fn)
        #loss_function = jit(build_varpro_loss_function(func,weighted,penalty_function,penalty_weight,param_converter,curvature_weight,smoothness_weight,max_weight,))
        #loss_function = jit(loss_function)
        loss_and_grad = jit(value_and_grad(loss_function))
        
        def optimize_model(initial_params, xs, y, y_uncertainties, constraints):
            #Why this works slow?
            loss_history = []

            #else:  # adam
                #here should go a way to choose as a dictionary the name of the optimizer.
            optimizer = optax.adam(learning_rate=learning_rate)
            opt_state = optimizer.init(initial_params)

            def step_fn(carry, _):
                params, opt_state = carry
                loss, grads = loss_and_grad(params, xs, y, y_uncertainties) #value_and_grad(loss_function)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), loss

            (final_params, _), loss_history = lax.scan(
                step_fn, (initial_params, opt_state), None, length=num_steps
            )

            return final_params, loss_history
        optimize_model = jit(optimize_model) #powerfull when we apply montecarlo-in in 1-2 objects sample not much impact +3 sec
        return loss_function, optimize_model