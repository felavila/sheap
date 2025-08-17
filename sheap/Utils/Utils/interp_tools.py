"""This module handles basic operations.
    The idea will be keep here all the tools for interpolation
"""
__author__ = 'felavila'

__all__ = [
    "cubic_spline_coefficients",
    "interpolate_nans",
    "replaze_nan_interpolation",
    "spline_eval",
    "vmap_interp",
]

import warnings
from copy import deepcopy
import functools as ft

import jax.numpy as jnp
from jax import jit, lax, vmap

@jit
def cubic_spline_coefficients(x, y):
    n = x.shape[0]
    h = x[1:] - x[:-1]  # Compute intervals h_i

    # Compute the alphas
    alpha = jnp.zeros(n)
    alpha = alpha.at[1:-1].set(3 / h[1:] * (y[2:] - y[1:-1]) - 3 / h[:-1] * (y[1:-1] - y[:-2]))

    # Initialize arrays
    l = jnp.zeros(n)
    mu = jnp.zeros(n)
    z = jnp.zeros(n)

    l = l.at[0].set(1.0)
    mu = mu.at[0].set(0.0)
    z = z.at[0].set(0.0)

    # Forward sweep
    def loop_body1(i, vals):
        l, mu, z = vals
        l = l.at[i].set(2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1])
        mu = mu.at[i].set(h[i] / l[i])
        z = z.at[i].set((alpha[i] - h[i - 1] * z[i - 1]) / l[i])
        return l, mu, z

    l, mu, z = lax.fori_loop(1, n - 1, loop_body1, (l, mu, z))

    l = l.at[n - 1].set(1.0)
    z = z.at[n - 1].set(0.0)

    c = jnp.zeros(n)
    b = jnp.zeros(n - 1)
    d = jnp.zeros(n - 1)

    c = c.at[n - 1].set(0.0)

    # Back substitution
    def loop_body2(j_rev, c_b_d):
        c, b, d = c_b_d
        j = n - 2 - j_rev
        c = c.at[j].set(z[j] - mu[j] * c[j + 1])
        b = b.at[j].set((y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3)
        d = d.at[j].set((c[j + 1] - c[j]) / (3 * h[j]))
        return c, b, d

    c, b, d = lax.fori_loop(0, n - 1, loop_body2, (c, b, d))

    return y[:-1], b, c[:-1], d  # Return coefficients y_i, b_i, c_i, d_i


@jit
def spline_eval(x_new, xk, yk, bk, ck, dk):
    # Find the interval xk_i <= x_new < xk_i+1
    inds = jnp.searchsorted(xk, x_new) - 1
    inds = jnp.clip(inds, 0, len(xk) - 2)
    dx = x_new - xk[inds]
    y_new = yk[inds] + bk[inds] * dx + ck[inds] * dx**2 + dk[inds] * dx**3
    return y_new


@jit
def interpolate_nans(x):
    """
    Dosent work
    Interpolates NaN values in a 1D JAX array using linear interpolation.

    Parameters:
    x (jnp.ndarray): Input 1D array with possible NaN values.

    Returns:
    jnp.ndarray: Array with NaNs replaced by interpolated values.
    """
    warnings.warn(
        "interpolate_nans is deprecated and will be removed in a future release. ",
        DeprecationWarning,
        stacklevel=2,  # Ensures the warning points to the user's call site
    )
    N = x.shape[0]
    indices = jnp.arange(N)
    not_nan = jnp.isfinite(x)

    # Forward scan to find the last valid index before each position
    def forward_step(carry, elem):
        idx, valid = elem
        new_last = jnp.where(valid, idx, carry)
        return new_last, new_last

    last_valid, _ = lax.scan(forward_step, -1, (indices, not_nan))

    # Reverse scan to find the next valid index after each position
    reversed_indices = jnp.flip(indices)
    reversed_not_nan = jnp.flip(not_nan)

    def reverse_step(carry, elem):
        idx, valid = elem
        new_next = jnp.where(valid, idx, carry)
        return new_next, new_next

    next_valid_reversed, _ = lax.scan(reverse_step, -1, (reversed_indices, reversed_not_nan))
    next_valid = jnp.flip(next_valid_reversed)

    # Handle boundary cases:
    # If no previous valid index, set to 0
    # If no next valid index, set to N-1
    last_valid = jnp.where(last_valid >= 0, last_valid, 0)
    next_valid = jnp.where(next_valid >= 0, next_valid, N - 1)

    # Gather the corresponding values
    last_valid_val = x[last_valid]
    next_valid_val = x[next_valid]

    # Compute the interpolation factor
    denom = next_valid - last_valid
    denom = jnp.where(denom == 0, 1, denom)  # Prevent division by zero
    prop = (indices - last_valid) / denom

    # Compute the interpolated values
    interpolated = last_valid_val + (next_valid_val - last_valid_val) * prop
    # interpolated = jnp.interp(x_a, xp, fp)
    # Replace NaNs with interpolated values
    y = jnp.where(not_nan, x, interpolated)

    return y


def replaze_nan_interpolation(y):
    A = deepcopy(y)

    ok = ~np.isnan(A)
    xp = ok.ravel().nonzero()[0]
    fp = A[~np.isnan(A)]
    x_a = np.isnan(A).ravel().nonzero()[0]

    A[np.isnan(A)] = np.interp(x_a, xp, fp)
    return A


@jit
def _interp_jax(x, xp, fp, left=None, right=None, period=None):
    """
    from pyspckit
    https://github.com/pyspeckit/pyspeckit/blob/4e1ed1c9c4759728cea04197d00d5c5f867b43f9/pyspeckit/spectrum/interpolation.py#L20
    Overrides numpy's interp function, which fails to check for
    we can thing is the same for jax.numpy
    increasingness....
    """
    indices = jnp.argsort(xp)
    xp = jnp.array(xp)[indices]
    fp = jnp.array(fp)[indices]
    return jnp.interp(x, xp, fp, left=left, right=right, period=period)




@ft.partial(vmap, in_axes=(None, None, 0), out_axes=0)
def vmap_interp(wavelength, wavelength_xp, flux_xp):
    return jnp.interp(wavelength, wavelength_xp, flux_xp)
