from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np 

import jax.numpy as jnp
from jax import vmap,grad,jit


#from sheap.Mappers.LineMapper import LineSelectionResult

#This is more than utils 

def trapz_jax(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    dx = x[1:] - x[:-1]
    return jnp.sum((y[1:] + y[:-1]) * dx / 2)

def integrate_function_error_single(function, x, p, sigma_p):
    y_int = trapz_jax(function(x, p), x)
    grad_f = grad(lambda pp: trapz_jax(function(x, pp), x))(p)
    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2))
    return y_int, sigma_f

# batched version via flatten–vmap–reshape
def integrate_batch(function, x, p, sigma_p):
    # p, sigma_p shape = (n, lines, params)  e.g. (2,19,3)
    n, lines, params = p.shape
    # 1) flatten
    p_flat     = p.reshape((n * lines, params))
    sigma_flat = sigma_p.reshape((n * lines, params))

    # 2) one vmap over the flattened batch axis
    batched_integrator = vmap(
        lambda pp, sp: integrate_function_error_single(function, x, pp, sp),
        in_axes=(0, 0),
        out_axes=(0, 0),
    )
    y_flat, sigma_flat_out = batched_integrator(p_flat, sigma_flat)

    # 3) reshape back to (n, lines)
    y_batch     = y_flat.reshape((n, lines))
    sigma_batch = sigma_flat_out.reshape((n, lines))
    return y_batch, sigma_batch

def integrate_function_error(function, x: jnp.ndarray, p: jnp.ndarray, sigma_p: jnp.ndarray = None):
    """
    Computes the integral of a function and propagates the error on the parameters.

    Parameters:
    -----------
    function : Callable
        Function to evaluate: function(x, p)
    x : jnp.ndarray
        Grid over which to integrate.
    p : jnp.ndarray
        Parameters for the function.
    sigma_p : jnp.ndarray, optional
        Standard deviation (uncertainty) for each parameter. Defaults to zero.

    Returns:
    --------
    y_int : float
        The integral of the function over `x`.
    sigma_f : float
        Propagated uncertainty on the integral due to `sigma_p`.
    """
    p = jnp.atleast_1d(p)
    sigma_p = jnp.zeros_like(p) if sigma_p is None else jnp.atleast_1d(sigma_p)

    def int_function(p_):
        return trapz_jax(function(x, p_), x)

    y_int = int_function(p)
    grad_f = grad(int_function)(p)

    
    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2))
    return y_int, sigma_f

def integrate_function_error_single(function, x, p, sigma_p):
    def int_function(p_):
        return trapz_jax(function(x, p_), x)

    y_int = int_function(p)
    grad_f = grad(int_function)(p)
    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2))
    return y_int, sigma_f


@jit
def combine_fast(
    params_broad: jnp.ndarray,
    params_narrow: jnp.ndarray,
    limit_velocity: float = 150.0,
    c: float = 299_792.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Combine any number of broad Gaussians + a narrow Gaussian per object,
    returning only (fwhm_final, amp_final, mu_final).

    Inputs
    ------
    params_broad : (N, 3*n_broad) array: [amp_i, mu_i, fwhm_i,...].
    params_narrow: (N, 3) array: [amp_n, mu_n, fwhm_n] but only mu_n used.
    limit_velocity : velocity threshold for virial filtering.
    c              : speed of light (same units as velocities).

    Returns
    -------
    fwhm_final : (N,) — chosen FWHM (in same units as input).
    amp_final  : (N,) — chosen amplitude.
    mu_final   : (N,) — chosen center.
    """
    N = params_broad.shape[0]
    n_broad = params_broad.shape[1] // 3
    broad = params_broad.reshape(N, n_broad, 3)
    amp_b, mu_b, fwhm_b = broad[..., 0], broad[..., 1], broad[..., 2]

    # 1) Weighted mean center & moment‐based FWHM
    total_amp = jnp.sum(amp_b, axis=1)                      # (N,)
    mu_eff    = jnp.sum(amp_b * mu_b, axis=1) / total_amp

    invf = 1.0 / 2.35482
    var_i   = (fwhm_b * invf) ** 2                          # variance per component
    dif2    = (mu_b - mu_eff[:, None]) ** 2
    var_eff = jnp.sum(amp_b * (var_i + dif2), axis=1) / total_amp
    fwhm_eff= jnp.sqrt(var_eff) * 2.35482                   # (N,)

    # 2) Closest‐to‐narrow component
    mu_nar   = params_narrow[:, 1]
    rel_vel  = jnp.abs((mu_b - mu_nar[:, None]) / mu_nar[:, None]) * c
    idx_near = jnp.argmin(rel_vel, axis=1)

    sel = lambda arr: arr[jnp.arange(N), idx_near]
    fwhm_nb  = sel(fwhm_b)
    amp_nb   = sel(amp_b)
    mu_nb    = sel(mu_b)

    # 3) Amplitude‐ratio mask
    amp_ratio = jnp.min(amp_b, axis=1) / jnp.max(amp_b, axis=1)
    mask_amp  = amp_ratio > 0.1

    fwhm_choice = jnp.where(mask_amp, fwhm_eff, fwhm_nb)
    amp_choice  = jnp.where(mask_amp, total_amp, amp_nb)
    mu_choice   = jnp.where(mask_amp, mu_eff, mu_nb)

    # 4) Virial filter
    mask_vir = jnp.min(rel_vel, axis=1) >= limit_velocity
    fwhm_final = jnp.where(mask_vir, fwhm_nb,    fwhm_choice)
    amp_final  = jnp.where(mask_vir, amp_nb,     amp_choice)
    mu_final   = jnp.where(mask_vir, mu_nb,      mu_choice)

    return fwhm_final, amp_final, mu_final

def summarize_samples(samples) -> Dict[str, np.ndarray]:
    """Compute 16/50/84 percentiles and return a summary dict using NumPy."""
    if isinstance(samples, jnp.ndarray):
        samples = np.asarray(samples)
    samples = np.atleast_2d(samples).T
    if np.isnan(samples).sum() / samples.size > 0.2:
        warnings.warn("High fraction of NaNs; uncertainty estimates may be biased.")
    if samples.shape[1]<=1:
        q = np.nanpercentile(samples, [16, 50, 84], axis=0)
    else:
        q = np.nanpercentile(samples, [16, 50, 84], axis=1)
    #else:
    
    return {
        "median": q[1],
        "err_minus": q[1] - q[0],
        "err_plus": q[2] - q[1]
    }
    
    
def summarize_nested_samples(d: dict) -> dict:
    """
    Recursively walk through a dictionary and apply summarize_samples_numpy
    to any array-like values.
    """
    summarized = {}
    for k, v in d.items():
        if isinstance(v, dict):
            summarized[k] = summarize_nested_samples(v)
        elif isinstance(v, (np.ndarray, jnp.ndarray)) and np.ndim(v) >= 1 and k!='component':
            summarized[k] = summarize_samples(v)
        else:
            summarized[k] = v
    return summarized



def evaluate_with_error(function, 
                        x: jnp.ndarray, 
                        p: jnp.ndarray, 
                        sigma_p: jnp.ndarray = None
                       ):
    """
    Evaluates `function(x, p)` and propagates the 1σ uncertainties in p
    to give an error on the result.

    Parameters
    ----------
    function : Callable
        Must have signature function(x, p) -> scalar (or array with last axis scalar).
    x : jnp.ndarray
        The “independent variable” at which to evaluate.
    p : jnp.ndarray, shape (..., P)
        Parameter vectors.  The leading “...” axes are treated as batch dims.
    sigma_p : jnp.ndarray, same shape as p, optional
        1σ uncertainties on each parameter.  If None, assumed zero.

    Returns
    -------
    f_val : jnp.ndarray, shape (...)
        The function evaluated at each batch of parameters.
    sigma_f : jnp.ndarray, shape (...)
        The propagated 1σ uncertainty on f_val.
    """
    # ensure arrays
    p = jnp.atleast_2d(p) if p.ndim == 1 else p
    if sigma_p is None:
        sigma_p = jnp.zeros_like(p)
    else:
        sigma_p = jnp.atleast_2d(sigma_p) if sigma_p.ndim == 1 else sigma_p

    # make a scalar-to-scalar function of only the parameter vector
    def f_of_p(params):
        return function(x, params)

    # vectorize over any leading batch dims
    #   - grad_f(p) has the same leading shape, with last axis = P
    grad_f = vmap(grad(f_of_p))(p)
    f_val  = vmap(f_of_p)(p)

    # error propagation: σ_f = sqrt( Σ_i (∂f/∂p_i · σ_{p_i})^2 )
    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2, axis=-1))

    return f_val, sigma_f


def batched_evaluate(function, x, p, sigma_p):
    # p, sigma_p: shape (n, lines, P)
    n, lines, P = p.shape

    # 1) flatten the batch dims
    p_flat     = p.reshape((n*lines, P))
    sigma_flat = sigma_p.reshape((n*lines, P))

    # 2) vectorize evaluate_with_error over that flat batch
    single_eval = lambda pp, sp: evaluate_with_error(function, x, pp, sp)
    f_flat, err_flat = vmap(single_eval, in_axes=(0,0), out_axes=(0,0))(p_flat, sigma_flat)

    # 3) reshape back to (n, lines)
    f_batch   = f_flat.reshape((n, lines))
    err_batch = err_flat.reshape((n, lines))
    return f_batch, err_batch



def evaluate_with_error(function,
                        x:         jnp.ndarray,       # shape = (n, lines)
                        p:         jnp.ndarray,       # shape = (n, P)
                        sigma1:    jnp.ndarray = None, # either x‐uncertainty or p‐uncertainty
                        sigma2:    jnp.ndarray = None  # the other one
                       ):
    """
    Evaluate f(x, p) and propagate 1σ errors in BOTH x and p.
    The two optional sigmas can be passed in either order; we'll
    auto–detect which is which by shape.

    Parameters
    ----------
    function : Callable
        f(x, p) → scalar per (x,p).
    x : jnp.ndarray, shape (n, lines)
    p : jnp.ndarray, shape (n, P)
    sigma1, sigma2 : jnp.ndarray or None
        Exactly one should match x.shape, the other should match p.shape.
        If you pass only one sigma, it will be applied to whichever it matches;
        the other is assumed zero.
    Returns
    -------
    y : jnp.ndarray, shape (n, lines)
    yerr : jnp.ndarray, shape (n, lines)
    """

    # figure out which sigma is for x and which for p
    sx = None
    sp = None
    for arr in (sigma1, sigma2):
        if arr is None:
            continue
        if arr.shape == x.shape:
            sx = arr
        elif arr.shape == p.shape:
            sp = arr
        else:
            raise ValueError(f"Unexpected sigma shape {arr.shape}; must match x{ x.shape } or p{ p.shape }")

    # default any missing one to zero
    if sx is None:
        sx = jnp.zeros_like(x)
    if sp is None:
        sp = jnp.zeros_like(p)

    n, lines = x.shape
    _, P      = p.shape

    # broadcast p and its sigma along the 'lines' axis
    p_exp   = jnp.broadcast_to(p[:, None, :],      (n, lines, P))
    sp_exp  = jnp.broadcast_to(sp[:, None, :],     (n, lines, P))

    # flatten everything to a single batch of size n*lines
    flat_size = n * lines
    x_flat   = x.reshape((flat_size,))
    sx_flat  = sx.reshape((flat_size,))
    p_flat   = p_exp.reshape((flat_size, P))
    sp_flat  = sp_exp.reshape((flat_size, P))

    # single‐point eval + error propagation
    def single_eval(xv, pv, sxv, spv):
        y   = function(xv, pv)
        dyx = grad(function, argnums=0)(xv, pv)
        dyp = grad(function, argnums=1)(xv, pv)
        var = (dyx * sxv)**2 + jnp.sum((dyp * spv)**2)
        return y, jnp.sqrt(var)

    # one vmap over the flat batch
    y_flat, err_flat = vmap(
        single_eval, in_axes=(0,0,0,0), out_axes=(0,0)
    )(x_flat, p_flat, sx_flat, sp_flat)

    # reshape back to (n, lines)
    y_batch   = y_flat.reshape((n, lines))
    err_batch = err_flat.reshape((n, lines))
    return y_batch, err_batch