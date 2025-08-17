"""This module handles basic operations."""
__author__ = 'felavila'


__all__ = [
    "batched_evaluate",
    "evaluate_with_error",
    "integrate_batch_with_error",
    "integrate_function_error",
    "integrate_function_error_single",
    "trapz_jax",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
import numpy as np 

import jax.numpy as jnp
from jax import vmap,grad,jit


#afterfitprofilehelpers This is the extended name of this. This function exist in Profiles?
def trapz_jax(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    dx = x[1:] - x[:-1]
    return jnp.sum((y[1:] + y[:-1]) * dx / 2)

def integrate_function_error_single(function, x, p, sigma_p):
    y_int = trapz_jax(function(x, p), x)
    grad_f = grad(lambda pp: trapz_jax(function(x, pp), x))(p)
    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2))
    return y_int, sigma_f


def integrate_batch_with_error(function, x, p, sigma_p):
    n, lines, params = p.shape
    p_flat     = p.reshape((n * lines, params))
    sigma_flat = sigma_p.reshape((n * lines, params))

    batched_integrator = vmap(
        lambda pp, sp: integrate_function_error_single(function, x, pp, sp),
        in_axes=(0, 0),
        out_axes=(0, 0),
    )
    y_flat, sigma_flat_out = batched_integrator(p_flat, sigma_flat)

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
    p = jnp.atleast_2d(p) if p.ndim == 1 else p
    if sigma_p is None:
        sigma_p = jnp.zeros_like(p)
    else:
        sigma_p = jnp.atleast_2d(sigma_p) if sigma_p.ndim == 1 else sigma_p

    def f_of_p(params):
        return function(x, params)

    grad_f = vmap(grad(f_of_p))(p)
    f_val  = vmap(f_of_p)(p)

    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2, axis=-1))

    return f_val, sigma_f


def batched_evaluate(function, x, p, sigma_p):
    n, lines, P = p.shape

    p_flat     = p.reshape((n*lines, P))
    sigma_flat = sigma_p.reshape((n*lines, P))

    single_eval = lambda pp, sp: evaluate_with_error(function, x, pp, sp)
    f_flat, err_flat = vmap(single_eval, in_axes=(0,0), out_axes=(0,0))(p_flat, sigma_flat)

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

    if sx is None:
        sx = jnp.zeros_like(x)
    if sp is None:
        sp = jnp.zeros_like(p)

    n, lines = x.shape
    _, P      = p.shape

    p_exp   = jnp.broadcast_to(p[:, None, :],      (n, lines, P))
    sp_exp  = jnp.broadcast_to(sp[:, None, :],     (n, lines, P))

    flat_size = n * lines
    x_flat   = x.reshape((flat_size,))
    sx_flat  = sx.reshape((flat_size,))
    p_flat   = p_exp.reshape((flat_size, P))
    sp_flat  = sp_exp.reshape((flat_size, P))

    def single_eval(xv, pv, sxv, spv):
        y   = function(xv, pv)
        dyx = grad(function, argnums=0)(xv, pv)
        dyp = grad(function, argnums=1)(xv, pv)
        var = (dyx * sxv)**2 + jnp.sum((dyp * spv)**2)
        return y, jnp.sqrt(var)

    y_flat, err_flat = vmap(
        single_eval, in_axes=(0,0,0,0), out_axes=(0,0)
    )(x_flat, p_flat, sx_flat, sp_flat)

    y_batch   = y_flat.reshape((n, lines))
    err_batch = err_flat.reshape((n, lines))
    return y_batch, err_batch


