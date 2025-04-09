import jax
import jax.numpy as jnp
from jax import jit,vmap
import optax
from typing import Callable, Dict, Tuple,List,Optional
from .utils import project_params,parse_dependencies



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
    
    def __init__(self,func: Callable,non_optimize_in_axis: int = 3,constraints: Optional[Callable] = None,
                num_steps: int = 1000,optimizer: optax.GradientTransformation = None,learning_rate=None,list_dependencies=[],weighted=True,**kwargs):
        self.func = func #TODO desing the function class 
        self.non_optimize_in_axis = non_optimize_in_axis #axis in where is require enter data of same dimension
        self.num_steps = num_steps
        self.learning_rate = learning_rate  or 1e-1
        self.list_dependencies = list_dependencies
        self.parsed_dependencies_tuple = parse_dependencies(self.list_dependencies)
        self.optimizer = kwargs.get("optimizer",optax.adabelief(self.learning_rate)) 
        #print('optimizer:',self.optimizer)
        
        self.loss_function, self.optimize_model, self.residuals = MasterMinimizer.minimization_function(self.func,weighted=weighted)
        
        self.vmap_func = vmap(self.func, in_axes=(0, 0), out_axes=0) #?
    def __call__(self,initial_params,y,x,yerror,constraints,learning_rate=None,num_steps=None,optimizer=None,non_optimize_in_axis=None,list_dependencies=None):
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
        self.default_args = (self.parsed_dependencies_tuple,
                        self.learning_rate,
                        self.num_steps,
                        self.optimizer,
                        False)
        
        print('learning_rate:',self.learning_rate)
        print('optimizer:',optax.adabelief.__name__)
        print('num_steps:',self.num_steps)
        
        if non_optimize_in_axis==3:
            #print("vmap Optimize over y,x,yerror")
            optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None)
        elif non_optimize_in_axis==4:
            #print("vmap Optimize over init_val,y,x,yerror")
            optimize_in_axis = (0, 0, 0, 0, None, None, None, None, None, None)
        # elif non_optimize_in_axis==5:
        #     #means the first values will be arrays
        #     optimize_in_axis = (0, 0, 0, 0, None, None, None, None, None, None)
        else:
            print("This value of non_optimize_in_axis not cover it will replace for 3")
            #print("vmap Optimize over y,x,yerror")
            non_optimize_in_axis = 3
            optimize_in_axis = (None, 0, 0, 0, None, None, None, None, None, None)
        self.optimize_in_axis = optimize_in_axis
        vmap_optimize_model = vmap(self.optimize_model,in_axes=optimize_in_axis, out_axes=0)
        
        return vmap_optimize_model(initial_params,y,x,yerror,constraints,*self.default_args)
        
        
        
    @staticmethod
    def minimization_function(
    func: Callable[[List[jnp.ndarray], jnp.ndarray], jnp.ndarray]
    ,weighted:bool =True) -> Callable[..., jnp.ndarray]:
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
        @jit
        def residuals(params: jnp.ndarray, xs: List[jnp.ndarray], y: jnp.ndarray, y_uncertainties: jnp.ndarray):
            predictions = func(xs, params)
            
            return jnp.abs(y - predictions) / y_uncertainties

        if weighted:
            @jit
            def loss_function(
                params: jnp.ndarray,
                xs: List[jnp.ndarray],
                y: jnp.ndarray,
                y_uncertainties: jnp.ndarray
                ) -> jnp.ndarray:
                
                y_pred = func(xs, params)
                weights = 1.0 / y_uncertainties**2
                loss = jnp.log(jnp.cosh(y_pred - y))
                wmse = jnp.nansum(weights * loss) / jnp.nansum(weights)
                return wmse
        else:
            @jit
            def loss_function(
                params: jnp.ndarray,
                xs: List[jnp.ndarray],
                y: jnp.ndarray,
                y_uncertainties: jnp.ndarray
                ) -> jnp.ndarray:
                y_pred = func(xs, params)
                loss = jnp.log(jnp.cosh(y_pred - y))
                wmse = jnp.nansum(loss) # For xshooter this looks like the only good option 
                return wmse
        
        def optimize_model(
            initial_params: jnp.ndarray,
            xs: List[jnp.ndarray],
            y: jnp.ndarray,
            y_uncertainties: jnp.ndarray,
            constraints: jnp.ndarray= None,
            parsed_dependencies = None,
            learning_rate: float = 1e-2,
            num_steps: int = 1000,
            optimizer = None,
            verbose: bool = False) -> Tuple[jnp.ndarray, list]:
            # Initialize parameters and optimizer state
            params = initial_params
            optimizer = optimizer or optax.adabelief(learning_rate)
            opt_state = optimizer.init(params)
            loss_history = []
            
            if constraints is None:
                constraints = jnp.array([[-1e41,1e41]] * params.shape[0])
            # Define the step function with constraints captured via closure
            @jit
            def step(params, opt_state, xs, y):
                # Compute loss and gradients
                loss, grads = jax.value_and_grad(loss_function)(
                    params, jnp.nan_to_num(xs), jnp.nan_to_num(y), y_uncertainties)

                # Compute parameter updates
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                # Project parameters to enforce constraints
                #combination = xs.T*100*params
                #negatives_per_column = jnp.nansum(combination < 0, axis=0)
                #params = jnp.where(negatives_per_column>1000,1e-3,params)
                
                params = project_params(
                    params,
                    constraints,parsed_dependencies)
                #extra projection
                
                    
                return params, opt_state, loss

            # Optimization loop
            for step_num in range(num_steps):
                params, opt_state, loss = step(params, opt_state, xs, y)
                loss_history.append(loss)
                if step_num % 100 == 0 and verbose:
                    print(f"Step {step_num}, Loss: {loss:.4f}")

            return params, loss_history

        return loss_function, optimize_model, residuals

