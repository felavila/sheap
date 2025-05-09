from typing import Any, Dict, List, Optional, Tuple, Union,Callable
import os

import jax
import jax.numpy as jnp
from jax import jit, vmap
import numpy as np 
import jax.scipy as jsp

from sheap.FunctionsMinimize.utils import param_count
from sheap.tools.others import kms_to_wl


templates_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"suport_data","templates")

fe_template_OP_file = os.path.join(templates_path,'fe2_Op.dat')
fe_template_OP = jnp.array(np.loadtxt(fe_template_OP_file,comments='#').transpose()) # y units?

fe_template_UV_file = os.path.join(templates_path,'fe2_UV02.dat')
fe_template_UV = jnp.array(np.loadtxt(fe_template_UV_file,comments='#').transpose()) # y units?

#maybe move to linear without this parameter? in the end will be require add 1e-3
#what is the best option ?
@jit
@param_count(2)
def linear(x,params):
    return params[0] * (x/1000.0) + params[1]

@jit
@param_count(2)
def powerlaw(x,params):
    x = jnp.nan_to_num(x)
    return  params[1] * jax.lax.pow(x / 1000.,params[0]) #+ params[1]

@jit
@param_count(2)
def loglinear(x,params):
    return params[0] * x + params[1]


@jit
def linear_combination(eieigenvectors,params):
    return jnp.nansum(eieigenvectors.T*100*params,axis=1)

@jit
@param_count(3)
def balmerconti(x,pars):
    """
    Compute the Balmer continuum (Dietrich+02) in pure JAX.

    Parameters
    ----------
    x : array-like
        Wavelengths in Angstrom.
    pars : array-like, shape (3,)
        pars[0] = A (amplitude)
        pars[1] = T (temperature in K)
        pars[2] = τ0 (optical‐depth scale)
   

    Returns
    -------
    result : ndarray
        Balmer continuum flux in the same shape as x.
    """
    # Constants
    h   = 6.62607015e-34   # Planck’s constant, J·s
    c   = 2.99792458e8     # Speed of light, m/s
    k_B = 1.380649e-23     # Boltzmann constant, J/K

    # Edge
    lambda_BE = 3646.0  # Å

    # Convert Å → m
    lam_m = x * 1e-10

    # Planck function B_λ(lam_m, T) [SI units]
    T = pars[1]
    exponent = h * c / (lam_m * k_B * T)
    B_lambda = (2.0 * h * c**2) / (lam_m**5 * (jnp.exp(exponent) - 1.0))

    # Apply the same “scale=10000” factor as in astropy’s BlackBody
    B_lambda *= 1e4

    # Optical depth τ(λ)
    tau = pars[2] * (x / lambda_BE)**3

    # Balmer-continuum formula
    result = pars[0] * B_lambda * (1.0 - jnp.exp(-tau))

    # Zero above the Balmer edge
    result = jnp.where(x > lambda_BE, 0.0, result)/1e18 #factor the normalisacion

    return result

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

@param_count(3)
def fitFeOP(x, params):
    "Fit the optical FeII on the continuum from 3686 to 7484 A based on Vestergaard & Wilkes 2001"
    log_FWHM_broad, shift_, scale = params
    central_wl = 4650.0  # Reference wavelength

    # Compute FWHM and convert to sigma (Gaussian dispersion)
    FWHM_broad = 10**log_FWHM_broad
    sigma_model = FWHM_broad / 2.355

    # Apply scaling to the shift parameter.
    shift = 10 * shift_

    # Replace NaNs in the wavelength array to ensure stability
    x = jnp.nan_to_num(x)

    # Read the normalized Fe emission template (assumes fe_template_read is loaded globally)
    x_fe_template_norm, y_fe_template_norm = fe_template_OP

    # Define intrinsic sigma of the template (from its FWHM of 900, for example)
    sigmatemplate = 900.0 / 2.355

    # Ensure the model sigma is not smaller than the template sigma to avoid sqrt of negative.
    safe_sigma_model = jnp.maximum(sigma_model, sigmatemplate + 1e-6)
    delta_sigma = jnp.sqrt(safe_sigma_model ** 2 - sigmatemplate ** 2)

    # Instead of a complex index, assume uniform wavelength array.
    # Use the first two elements to compute the step size.
    dl = x[1] - x[0]
    # Safety: if dl somehow is zero, assign a minimum value.
    dl = jnp.where(dl == 0, 1e-6, dl)

    # Convert the additional broadening from km/s to wavelength units.
    sigma_wl = kms_to_wl(delta_sigma, central_wl) / dl

    # Define the convolution kernel radius; ensure that it is at least 1.
    radius = jnp.maximum(jnp.round(4 * sigma_wl).astype(jnp.int32), 1)

    # Create a local grid for the convolution kernel.
    max_radius = 1000  # Over-dimension the grid for JIT compilation compatibility.
    x_local = jnp.arange(-max_radius, max_radius + 1)

    # Create a mask with a robust approach: only values within [-radius, radius] are nonzero.
    mask = jnp.where(jnp.abs(x_local) <= radius, 1.0, 0.0)
    # Compute the Gaussian kernel on the entire grid.
    kernel = jsp.stats.norm.pdf(x_local, scale=sigma_wl) * mask
    kernel = kernel / jnp.sum(kernel)

    # Convolve the normalized Fe template with the Gaussian kernel using FFT.
    broad_template = jsp.signal.convolve(y_fe_template_norm, kernel, mode='same', method='fft')

    # Shift the template's wavelength grid.
    shifted_wl = x_fe_template_norm + shift

    # Interpolate the convolved (broadened) template back onto the input wavelength grid.
    # Make sure _interp_jax is differentiable for gradient-based optimization.
    interpolated_broad_scaled_template = scale * jnp.interp(
        x, shifted_wl, broad_template, left=None, right=None
    )

    return interpolated_broad_scaled_template

@param_count(3)
#2795
def fitFeUV(x,params):
    "Fit the UV FeII component on the continuum from 1200 to 3500 A based on Boroson & Green 1992."
    log_FWHM_broad, shift_, scale = params
    central_wl = 2795  # Reference wavelength

    # Compute FWHM and convert to sigma (Gaussian dispersion)
    FWHM_broad = 10**log_FWHM_broad
    sigma_model = FWHM_broad / 2.355

    # Apply scaling to the shift parameter.
    shift = 10 * shift_

    # Replace NaNs in the wavelength array to ensure stability
    x = jnp.nan_to_num(x)

    # Read the normalized Fe emission template (assumes fe_template_read is loaded globally)
    x_fe_template_norm, y_fe_template_norm = fe_template_UV

    # Define intrinsic sigma of the template (from its FWHM of 900, for example)
    sigmatemplate = 900.0 / 2.355

    # Ensure the model sigma is not smaller than the template sigma to avoid sqrt of negative.
    safe_sigma_model = jnp.maximum(sigma_model, sigmatemplate + 1e-6)
    delta_sigma = jnp.sqrt(safe_sigma_model ** 2 - sigmatemplate ** 2)

    # Instead of a complex index, assume uniform wavelength array.
    # Use the first two elements to compute the step size.
    dl = x[1] - x[0]
    # Safety: if dl somehow is zero, assign a minimum value.
    dl = jnp.where(dl == 0, 1e-6, dl)

    # Convert the additional broadening from km/s to wavelength units.
    sigma_wl = kms_to_wl(delta_sigma, central_wl) / dl

    # Define the convolution kernel radius; ensure that it is at least 1.
    radius = jnp.maximum(jnp.round(4 * sigma_wl).astype(jnp.int32), 1)

    # Create a local grid for the convolution kernel.
    max_radius = 1000  # Over-dimension the grid for JIT compilation compatibility.
    x_local = jnp.arange(-max_radius, max_radius + 1)
    
    # Create a mask with a robust approach: only values within [-radius, radius] are nonzero.
    mask = jnp.where(jnp.abs(x_local) <= radius, 1.0, 0.0)
    # Compute the Gaussian kernel on the entire grid.
    kernel = jsp.stats.norm.pdf(x_local, scale=sigma_wl) * mask
    kernel = kernel / jnp.sum(kernel)

    # Convolve the normalized Fe template with the Gaussian kernel using FFT.
    broad_template = jsp.signal.convolve(y_fe_template_norm, kernel, mode='same', method='fft')

    # Shift the template's wavelength grid.
    shifted_wl = x_fe_template_norm + shift

    # Interpolate the convolved (broadened) template back onto the input wavelength grid.
    interpolated_broad_scaled_template = scale * jnp.interp(
        x, shifted_wl, broad_template, left=None, right=None
    )

    return interpolated_broad_scaled_template




def Gsum_model(centers, amplitudes):
    """
    Returns a Gaussian sum model function with shared sigma and shift.

    Args:
        centers (array): Array of Gaussian centers.
        amplitudes (array): Array of Gaussian amplitudes (same shape as centers).

    Returns:
        callable: Function G(x, params) with params = [amplitude, delta, width].
    """
    centers = jnp.array(centers)
    amplitudes = jnp.array(amplitudes)

    @jit
    @param_count(3)
    def G(x, params):
        amplitude = params[0]
        delta = params[1]
        width = params[2]
        shifted_centers = centers + delta
        dx = jnp.expand_dims(x, 0) - jnp.expand_dims(shifted_centers, 1)
        gaussians = amplitude * amplitudes[:, None] * jnp.exp(-0.5 * (dx / width) ** 2)
        return jnp.sum(gaussians, axis=0)

    return G
    



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
#polyfit_vmap = vmap(jnp.polyfit,in_axes=(0,0,None,None,None,0),out_axes=0)

#polyval_vmap = vmap(jnp.polyval,in_axes=(0,None),out_axes=0)