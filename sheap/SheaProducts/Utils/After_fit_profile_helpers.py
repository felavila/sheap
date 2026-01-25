"""
After-Fit Profile Helpers
=========================

This module provides helper routines to evaluate and integrate
spectral line profiles while propagating parameter and/or grid
uncertainties. These are used primarily in the ``ComplexParams``
pipeline when computing derived quantities (flux, FWHM, luminosity,
etc.) from fitted or sampled parameter sets.

Main Features
-------------
- Numerical integration of profile functions with uncertainty
  propagation via JAX autodiff.
- Batched integration and evaluation for multiple lines/objects.
- Support for error propagation from both parameter uncertainties
  and wavelength (x) uncertainties.
- JAX-compatible (vectorized with ``vmap``, differentiable).

Public API
----------
- :func:`trapz_jax`
- :func:`integrate_function_error`
- :func:`integrate_function_error_single`
- :func:`integrate_batch_with_error`
- :func:`evaluate_with_error`
- :func:`batched_evaluate`
"""

__author__ = 'felavila'

__all__ = [
    "trapz_jax",
    "integrate_function_error",
    "integrate_function_error_single",
    "integrate_batch_with_error",
    "evaluate_with_error",
    "batched_evaluate",
]

from typing import Callable, Tuple
import jax.numpy as jnp
from jax import vmap, grad, jit


def trapz_jax(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Trapezoidal integration along a 1D grid using JAX.

    Parameters
    ----------
    y : jnp.ndarray
        Function values on the grid ``x``.
    x : jnp.ndarray
        Monotonic grid points.

    Returns
    -------
    jnp.ndarray
        Scalar integral :math:`\\int y(x) \\, dx` approximated with the trapezoid rule.
    """
    dx = x[1:] - x[:-1]
    return jnp.sum((y[1:] + y[:-1]) * dx / 2)


def integrate_function_error_single(function: Callable, x: jnp.ndarray,
                                    p: jnp.ndarray, sigma_p: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Integrate a single profile and propagate parameter errors.

    .. math::
        F = \\int f(\\lambda; p) \\, d\\lambda

    with uncertainty propagation via linearization.

    Parameters
    ----------
    function : Callable
        Profile function with signature ``function(x, p)``.
    x : jnp.ndarray
        1D integration grid.
    p : jnp.ndarray
        Parameter vector.
    sigma_p : jnp.ndarray
        1σ uncertainty per parameter.

    Returns
    -------
    y_int : jnp.ndarray
        Integrated value.
    sigma_f : jnp.ndarray
        Propagated 1σ uncertainty.
    """
    def int_function(pp):
        return trapz_jax(function(x, pp), x)

    y_int = int_function(p)
    grad_f = grad(int_function)(p)
    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2))
    return y_int, sigma_f


def integrate_batch_with_error(function: Callable, x: jnp.ndarray,
                            p: jnp.ndarray, sigma_p: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Batched integration with parameter uncertainty propagation.

    Parameters
    ----------
    function : Callable
        Profile function.
    x : jnp.ndarray
        1D integration grid.
    p : jnp.ndarray
        Parameters, shape (N, L, P).
    sigma_p : jnp.ndarray
        Uncertainties, shape (N, L, P).

    Returns
    -------
    y_batch : jnp.ndarray
        Integrated values, shape (N, L).
    sigma_batch : jnp.ndarray
        Propagated uncertainties, shape (N, L).
    """
    n, lines, params = p.shape
    p_flat = p.reshape((n * lines, params))
    sigma_flat = sigma_p.reshape((n * lines, params))

    batched_integrator = vmap(
        lambda pp, sp: integrate_function_error_single(function, x, pp, sp),
        in_axes=(0, 0), out_axes=(0, 0)
    )
    y_flat, sigma_flat_out = batched_integrator(p_flat, sigma_flat)
    return y_flat.reshape((n, lines)), sigma_flat_out.reshape((n, lines))


def integrate_function_error(function: Callable, x: jnp.ndarray,
                            p: jnp.ndarray, sigma_p: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Integrate a profile and propagate parameter uncertainties.

    .. math::
        F = \\int f(\\lambda; p) \\, d\\lambda

    Parameters
    ----------
    function : Callable
        Profile function ``f(x, p)``.
    x : jnp.ndarray
        Grid over which to integrate.
    p : jnp.ndarray
        Parameters.
    sigma_p : jnp.ndarray, optional
        1σ parameter uncertainties. If None, treated as zero.

    Returns
    -------
    y_int : jnp.ndarray
        Integral value.
    sigma_f : jnp.ndarray
        Propagated uncertainty.
    """
    p = jnp.atleast_1d(p)
    sigma_p = jnp.zeros_like(p) if sigma_p is None else jnp.atleast_1d(sigma_p)

    def int_function(pp):
        return trapz_jax(function(x, pp), x)

    y_int = int_function(p)
    grad_f = grad(int_function)(p)
    sigma_f = jnp.sqrt(jnp.sum((grad_f * sigma_p) ** 2))
    return y_int, sigma_f


def evaluate_with_error(function: Callable,
                        x: jnp.ndarray,
                        p: jnp.ndarray,
                        sigma_x: jnp.ndarray = None,
                        sigma_p: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Evaluate a profile and propagate 1σ errors in both x and p.

    .. math::
        \\sigma_y^2 = \\left( \\frac{\\partial f}{\\partial x} \\sigma_x \\right)^2
        + \\sum_i \\left( \\frac{\\partial f}{\\partial p_i} \\sigma_{p_i} \\right)^2

    Parameters
    ----------
    function : Callable
        Profile function ``f(x, p)``.
    x : jnp.ndarray
        Grid, shape (N, L).
    p : jnp.ndarray
        Parameters, shape (N, P).
    sigma_x : jnp.ndarray, optional
        Uncertainty on x, shape (N, L).
    sigma_p : jnp.ndarray, optional
        Uncertainty on p, shape (N, P).

    Returns
    -------
    y : jnp.ndarray
        Function values.
    yerr : jnp.ndarray
        Propagated uncertainties.
    """
    if sigma_x is None:
        sigma_x = jnp.zeros_like(x)
    if sigma_p is None:
        sigma_p = jnp.zeros_like(p)

    n, lines = x.shape
    _, P = p.shape

    p_exp = jnp.broadcast_to(p[:, None, :], (n, lines, P))
    sp_exp = jnp.broadcast_to(sigma_p[:, None, :], (n, lines, P))

    flat_size = n * lines
    x_flat = x.reshape((flat_size,))
    sx_flat = sigma_x.reshape((flat_size,))
    p_flat = p_exp.reshape((flat_size, P))
    sp_flat = sp_exp.reshape((flat_size, P))

    def single_eval(xv, pv, sxv, spv):
        y = function(xv, pv)
        dyx = grad(function, argnums=0)(xv, pv)
        dyp = grad(function, argnums=1)(xv, pv)
        var = (dyx * sxv)**2 + jnp.sum((dyp * spv)**2)
        return y, jnp.sqrt(var)

    y_flat, err_flat = vmap(single_eval, in_axes=(0, 0, 0, 0), out_axes=(0, 0))(
        x_flat, p_flat, sx_flat, sp_flat
    )
    return y_flat.reshape((n, lines)), err_flat.reshape((n, lines))


def batched_evaluate(function: Callable, x: jnp.ndarray,
                    p: jnp.ndarray, sigma_p: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Batched evaluation with parameter uncertainties only.

    Parameters
    ----------
    function : Callable
        Profile function.
    x : jnp.ndarray
        Independent variable(s).
    p : jnp.ndarray
        Parameters, shape (N, L, P).
    sigma_p : jnp.ndarray
        Parameter uncertainties, shape (N, L, P).

    Returns
    -------
    f_batch : jnp.ndarray
        Function values, shape (N, L).
    err_batch : jnp.ndarray
        Propagated errors, shape (N, L).
    """
    n, lines, P = p.shape
    p_flat = p.reshape((n * lines, P))
    sigma_flat = sigma_p.reshape((n * lines, P))

    def single_eval(pp, sp):
        f_val, f_err = evaluate_with_error(function, x, pp[None], sigma_p=sp[None])
        return f_val.squeeze(), f_err.squeeze()

    f_flat, err_flat = vmap(single_eval, in_axes=(0, 0), out_axes=(0, 0))(p_flat, sigma_flat)
    return f_flat.reshape((n, lines)), err_flat.reshape((n, lines))
