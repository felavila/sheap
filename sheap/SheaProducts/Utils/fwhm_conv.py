r"""
FWHM utilities for line profiles
================================

Helpers to compute the full width at half maximum (FWHM) for different
profile families, with optional uncertainty propagation and batched
(vmap) evaluation.

Notes
-----
- For some analytic profiles, the FWHM is read directly from named
  shape parameters. For example:

  - Gaussian / Lorentzian:
    .. math::
       \mathrm{FWHM} = \text{fwhm}

  - Top-hat:
    .. math::
       \mathrm{FWHM} = \text{width}

  - Pseudo-Voigt:
    .. math::
       \mathrm{FWHM} \approx 0.5346\,\text{FWHM}_L
       + \sqrt{0.2166\,\text{FWHM}_L^2 + \text{FWHM}_G^2}

- For other profiles (e.g., skewed shapes), a numeric half‑maximum
  search is performed around the peak.
"""

__author__ = 'felavila'

__all__ = [
    "compute_fwhm_split",
    "compute_fwhm_split_with_error",
    "make_batch_fwhm_split",
    "make_batch_fwhm_split_with_error",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import warnings
from functools import partial
from jax import vmap,jit
import jax.numpy as jnp
from sheap.Profiles.Profiles import PROFILE_LINE_FUNC_MAP

from jax import jacfwd

def compute_fwhm_split(profile: str,
                       amp:   jnp.ndarray,
                       center:jnp.ndarray,
                       extras:jnp.ndarray) -> jnp.ndarray:
    r"""
    Compute the FWHM of a single line component for a given profile.

    The function uses analytic formulas when available (Gaussian,
    Lorentzian, Top-hat, Pseudo‑Voigt). Otherwise, it estimates the
    half‑maximum width numerically around the peak.

    Parameters
    ----------
    profile : str
        Profile key (must exist in ``PROFILE_LINE_FUNC_MAP``),
        e.g. ``"gaussian"``, ``"lorentzian"``, ``"top_hat"``,
        ``"voigt_pseudo"``, etc.
    amp : jnp.ndarray
        Peak amplitude (scalar).
    center : jnp.ndarray
        Line center (scalar).
    extras : jnp.ndarray
        Remaining shape parameters in the order required by the profile,
        i.e. they correspond to ``param_names[2:]``.

    Returns
    -------
    jnp.ndarray
        The full width at half maximum for the component (scalar).

    Notes
    -----
    - Pseudo‑Voigt approximation:
      .. math::
         \mathrm{FWHM} \approx 0.5346\,\text{FWHM}_L
         + \sqrt{0.2166\,\text{FWHM}_L^2 + \text{FWHM}_G^2}
    - Numeric fallback scans a symmetric grid around the center and finds
      the left/right half‑max crossings.
    """
    func = PROFILE_LINE_FUNC_MAP[profile]

    # build the named‐param dict on‐the‐fly:
    # we know extras corresponds to param_names[2:]
    names = func.param_names
    p = { names[0]: amp,
          names[1]: center }
    for i,name in enumerate(names[2:]):
        p[name] = extras[i]

    # analytic cases:
    if profile == "gaussian" or profile == "lorentzian":
        return p["fwhm"]
    if profile == "top_hat":
        return p["width"]
    if profile == "voigt_pseudo":
        fg = p["fwhm_g"]; fl = p["fwhm_l"]
        return 0.5346*fl + jnp.sqrt(0.2166*fl*fl + fg*fg)

    # numeric‐fallback (e.g. skewed, EMG)
    half = amp/2.0
    def shape_fn(x):
        return func(x, jnp.concatenate([jnp.array([amp,center]), extras]))
    guess = p.get("fwhm", p.get("width",
                jnp.maximum(p.get("fwhm_g",0), p.get("fwhm_l",0))))
    lo,hi = center-5*guess, center+5*guess
    xs = jnp.linspace(lo, hi, 2001)
    ys = shape_fn(xs)

    maskL = (xs<center)&(ys<=half)
    maskR = (xs> center)&(ys<=half)
    xL = jnp.max(jnp.where(maskL, xs, lo))
    xR = jnp.min(jnp.where(maskR, xs, hi))
    return xR - xL


def compute_fwhm_split_with_error(
    profile: str,
    amp: jnp.ndarray,
    center: jnp.ndarray,
    extras: jnp.ndarray,
    amp_err: jnp.ndarray,
    center_err: jnp.ndarray,
    extras_err: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    r"""
    Compute FWHM and its 1σ uncertainty for a single component.

    Uncertainty is propagated via the Jacobian of FWHM with respect to
    all input parameters (``amp``, ``center``, ``extras``):

    .. math::
       \sigma_{\mathrm{FWHM}}^2
       = \sum_i \left( \frac{\partial\,\mathrm{FWHM}}{\partial p_i} \,
                       \sigma_{p_i} \right)^2

    Parameters
    ----------
    profile : str
        Profile key for :func:`compute_fwhm_split`.
    amp, center, extras : jnp.ndarray
        Scalar amplitude, scalar center, and extras vector for the profile.
    amp_err, center_err, extras_err : jnp.ndarray
        Matching 1σ uncertainties for the corresponding parameters.

    Returns
    -------
    fwhm_val : jnp.ndarray
        Estimated FWHM (scalar).
    fwhm_uncertainty : jnp.ndarray
        Propagated 1σ uncertainty (scalar).

    Notes
    -----
    Uses :func:`jax.jacfwd` to compute the gradient of the FWHM function
    with respect to the concatenated parameter vector.
    """

    fwhm_fn = lambda amp, center, extras: compute_fwhm_split(profile, amp, center, extras)
    fwhm_val = fwhm_fn(amp, center, extras)

    # Build parameter vector
    all_params = jnp.concatenate([amp[None], center[None], extras])
    all_errors = jnp.concatenate([amp_err[None], center_err[None], extras_err])

    # Compute gradient
    grad_fwhm = jacfwd(lambda p: fwhm_fn(p[0], p[1], p[2:]))(all_params)

    # Propagate uncertainty
    fwhm_uncertainty = jnp.sqrt(jnp.sum((grad_fwhm * all_errors) ** 2))
    return fwhm_val, fwhm_uncertainty



def make_batch_fwhm_split_with_error(profile: str):
    """
    Vectorized (batched) FWHM + uncertainty evaluator for a profile.

    Returns a function that accepts batched inputs for values and their
    uncertainties and computes both FWHM and its propagated 1σ error
    using two levels of ``vmap`` (over lines, then over batch).

    Parameters
    ----------
    profile : str
        Profile key for :func:`compute_fwhm_split_with_error`.

    Returns
    -------
    Callable
        A function
        ``batcher(amp, center, extras, amp_err, center_err, extras_err)``
        that returns ``(fwhm_val, fwhm_uncertainty)`` with shapes matching
        the leading batch dimensions of the inputs.
    """
    single = partial(compute_fwhm_split_with_error, profile)
    over_lines = vmap(single, in_axes=(0, 0, 0, 0, 0, 0))
    batcher = vmap(over_lines, in_axes=(0, 0, 0, 0, 0, 0))
    return batcher

def make_batch_fwhm_split(profile: str):
    """
    Create a batched FWHM evaluator for a given profile.

    This returns a function that computes the full width at half maximum (FWHM)
    for multiple objects and multiple line components in parallel, using JAX’s
    :func:`vmap` for vectorization.

    Parameters
    ----------
    profile : str
        Profile name (must exist in ``PROFILE_LINE_FUNC_MAP``),
        e.g. ``"gaussian"``, ``"lorentzian"``, ``"voigt_pseudo"``, etc.

    Returns
    -------
    callable
        A function with signature::

            fwhm_batch(amp, center, extras) -> jnp.ndarray

        where
        - ``amp`` has shape (n_objects, n_lines),
        - ``center`` has shape (n_objects, n_lines),
        - ``extras`` has shape (n_objects, n_lines, n_extras),

        and the result is a ``(n_objects, n_lines)`` array of FWHM values.

    Notes
    -----
    - Analytic shortcuts are used for common profiles
      (Gaussian, Lorentzian, Top-hat, pseudo-Voigt).
    - For other profiles, a numeric search is performed around the line center
      to locate the half-maximum crossing.
    """
    single = partial(compute_fwhm_split, profile)
    over_lines = vmap(single, in_axes=(0, 0, 0))
    batcher    = vmap(over_lines, in_axes=(0, 0, 0))

    return batcher
