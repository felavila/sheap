from typing import Callable, Dict, Tuple, Optional
import jax.numpy as jnp
from jax import jit,vmap
import jax 
from SHEAP.fitting.utils import param_count

#maybe move to linear without this parameter? in the end will be require add 1e-3
#what is the best option ?
@jit
@param_count(2)
def linear(x,params):
    return params[0] * (x/1000) + params[1]
@jit
@param_count(2)
def loglinear(x,params):
    return params[0] * x + params[1]

# @jit
# @param_count(2)
# def linear2(x,params):
#     return x*10**(params[0]) + params[1]
@jit
def linear_combination(eieigenvectors,params):
    return jnp.nansum(eieigenvectors.T*100*params,axis=1)
# @jit
# def linear_combinationv2(eieigenvectors,params):
#     combination = eieigenvectors.T*100*params
#     negatives_per_column = jnp.nansum(combination < 0, axis=0)
#     params=jnp.where(negatives_per_column>1000,0,params)
#     return jnp.nansum(eieigenvectors.T*100*params,axis=1),params
# @jit
# def linear_combination(eieigenvectors,params):
#     combination = eieigenvectors.T*100*params
#     negatives_per_column = jnp.nansum(combination < 0, axis=0)
#     col_mask = negatives_per_column > 100
#     combination_zeroed_cols = jnp.where(col_mask[None, :], jnp.nan, combination)
#     return jnp.nansum(combination_zeroed_cols, axis=1)
#jnp.nansum(jnp.where(combination<0,0.0,combination),axis=1)
polyfit_vmap = vmap(jnp.polyfit,in_axes=(0,0,None,None,None,0),out_axes=0)

polyval_vmap = vmap(jnp.polyval,in_axes=(0,None),out_axes=0)

@jit
@param_count(3)
def gaussian_func(x,params):
    amplitude,center,width = params
    return  amplitude * jnp.exp(-0.5 * ((x - center) / width) ** 2)
@jit
@param_count(3)
def lorentzian_func(x,params):
    amplitude,center,gamma = params
    return amplitude/(1+((x-center)/gamma)**2) 
@jit
@param_count(2)
def power_law(x,params):
    return  (x/1000)**(params[0]) + params[1]

class GaussianSum:
    def __init__(self, n, constraints=None, inequalities=None):
        """
        Initialize the GaussianSum with parameter constraints.

        Parameters:
        - n (int): Number of Gaussian functions.
        - constraints (dict): Optional equality constraints on parameters.
            Example:
                {
                    'amp': [('amp0', 'amp1')],  # amp0 == amp1
                    'mu': [('mu2', 'mu3')],
                    'sigma': [('sigma1', 'sigma2')]
                }
        - inequalities (dict): Optional inequality constraints on parameters.
            Example:
                {
                    'sigma': [('sigma1', 'sigma2')]  # sigma2 > sigma1
                }
        """
        self.n = n
        self.constraints = constraints or {}
        self.inequalities = inequalities or {}
        # Determine free parameters based on constraints
        self.param_mapping = self._build_param_mapping()
        # Calculate the number of free parameters
        self.num_free_params = self._count_free_params()
        # Build the JIT-compiled Gaussian sum function
        self.sum_gaussians_jit = self._build_gaussian_sum()

    def _build_param_mapping(self):
        """
        Build a mapping from free parameters to all parameters,
        applying constraints as specified.
        """
        # Initialize mappings: each parameter maps to itself initially
        mapping = {
            'amp': list(range(self.n)),
            'mu': list(range(self.n)),
            'sigma': list(range(self.n))
        }

        # Apply equality constraints
        for param_type, pairs in self.constraints.items():
            for (p1, p2) in pairs:
                idx1 = int(p1.replace(param_type, ''))
                idx2 = int(p2.replace(param_type, ''))
                mapping[param_type][idx2] = mapping[param_type][idx1]

        return mapping

    def _count_free_params(self):
        """
        Count the number of free parameters after applying constraints.
        """
        free_amp = len(set(self.param_mapping['amp']))
        free_mu = len(set(self.param_mapping['mu']))
        free_sigma = len(set(self.param_mapping['sigma']))
        return free_amp + free_mu + free_sigma + self._count_inequality_free_params()

    def _count_inequality_free_params(self):
        """
        Count additional free parameters required for inequality constraints.
        For each inequality, an extra free parameter is needed to define the offset.
        """
        count = 0
        for param_type, pairs in self.inequalities.items():
            count += len(pairs)
        return count

    def _apply_constraints(self, params):
        """
        Apply equality constraints to the parameter vector to obtain full parameter sets.

        Parameters:
        - params (jnp.ndarray): Free parameters vector.

        Returns:
        - amps, mus, sigmas (tuple of jnp.ndarray): Full parameter sets.
        """
        free_amp = self.param_mapping['amp']
        free_mu = self.param_mapping['mu']
        free_sigma = self.param_mapping['sigma']

        num_free_amp = len(set(free_amp))
        num_free_mu = len(set(free_mu))
        num_free_sigma = len(set(free_sigma))

        # Extract free parameters
        idx = 0
        amps_free = params[idx:idx + num_free_amp]
        idx += num_free_amp
        mus_free = params[idx:idx + num_free_mu]
        idx += num_free_mu
        sigmas_free = params[idx:idx + num_free_sigma]
        idx += num_free_sigma

        # Map free parameters to all parameters using the mapping
        amps = jnp.array([amps_free[i] for i in self.param_mapping['amp']])
        mus = jnp.array([mus_free[i] for i in self.param_mapping['mu']])
        sigmas = jnp.array([sigmas_free[i] for i in self.param_mapping['sigma']])

        return amps, mus, sigmas

    def _apply_inequality_constraints(self, sigmas, params):
        """
        Apply inequality constraints to sigmas.

        For example, enforce sigma2 > sigma1 by setting sigma2 = sigma1 + softplus(delta)

        Parameters:
        - sigmas (jnp.ndarray): Current sigma parameters.
        - params (jnp.ndarray): Remaining parameters for inequality transformations.

        Returns:
        - jnp.ndarray: Transformed sigma parameters satisfying inequalities.
        """
        if not self.inequalities:
            return sigmas

        # Assuming all inequality constraints are on 'sigma'
        for (s1, s2) in self.inequalities.get('sigma', []):
            idx1 = int(s1.replace('sigma', ''))
            idx2 = int(s2.replace('sigma', ''))
            delta = params[0]
            params = params[1:]
            transformed_sigma2 = sigmas[idx1] + jax.nn.softplus(delta)
            sigmas = sigmas.at[idx2].set(transformed_sigma2)
        return sigmas

    def _build_gaussian_sum(self):
        """
        Build the JIT-compiled Gaussian sum function.

        Returns:
        - sum_gaussians_jit (function): JIT-compiled function.
        """
        def gaussian(x, amp, mu, sigma):
            return amp * jnp.exp(-0.5 * ((x - mu) / sigma) ** 2)


        def sum_gaussians(x, params):
            # Validate parameter length
            if params.shape[0] != self.num_free_params:
                raise ValueError(f"Expected {self.num_free_params} parameters, got {params.shape[0]}.")

            # Apply equality constraints
            amps, mus, sigmas = self._apply_constraints(params)
            
            # Apply inequality constraints if any
            if self.inequalities:
                # Extract deltas for inequalities
                delta_params = params[-len(self.inequalities.get('sigma', [])):]
                sigmas = self._apply_inequality_constraints(sigmas, delta_params)

            # Use a lambda to fix 'x' while vectorizing over amp, mu, sigma
            gaussians = vmap(lambda amp, mu, sigma: gaussian(x, amp, mu, sigma))(amps, mus, sigmas)
            
            return jnp.sum(gaussians, axis=0)
        self.n_params = self.num_free_params
        return jit(sum_gaussians)

    def __call__(self, x, params):
        """
        Compute the sum of Gaussians at points x with given parameters.

        Parameters:
        - x (jnp.ndarray): Points at which to evaluate the sum.
        - params (jnp.ndarray): Free parameters vector.

        Returns:
        - jnp.ndarray: Sum of Gaussians evaluated at x.
        """
        
        return self.sum_gaussians_jit(x, params)
