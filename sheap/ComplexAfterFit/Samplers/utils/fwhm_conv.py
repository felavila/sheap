"""Tools to move line profiles to fwhm. This should be moved to profiles"""
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
from sheap.Profiles.profiles import PROFILE_LINE_FUNC_MAP

from jax import jacfwd


def compute_fwhm_split(profile: str,
                       amp:   jnp.ndarray,
                       center:jnp.ndarray,
                       extras:jnp.ndarray) -> jnp.ndarray:
    #i guess here it is important to add a helper to move the profile to 
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

def make_batch_fwhm_split(profile: str):
    
    single = partial(compute_fwhm_split, profile)
    over_lines = vmap(single, in_axes=(0, 0, 0))
    batcher    = vmap(over_lines, in_axes=(0, 0, 0))

    return batcher


def compute_fwhm_split_with_error(
    profile: str,
    amp: jnp.ndarray,
    center: jnp.ndarray,
    extras: jnp.ndarray,
    amp_err: jnp.ndarray,
    center_err: jnp.ndarray,
    extras_err: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes FWHM and its uncertainty given profile name and parameter errors.
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
    single = partial(compute_fwhm_split_with_error, profile)
    over_lines = vmap(single, in_axes=(0, 0, 0, 0, 0, 0))
    batcher = vmap(over_lines, in_axes=(0, 0, 0, 0, 0, 0))
    return batcher


