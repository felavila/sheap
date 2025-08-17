"""This module ?."""
__author__ = 'felavila'

# Auto-generated __all__
__all__ = [
    "deredshift",
    "unred",
]

"""Based in the implementation in numpy from
https://github.com/sczesla/PyAstronomy/blob/93f6f0668d6b5aa77d281981e13ba1bf6ded38cd/src/pyasl/asl/unred.py#L4
"""

import functools as ft

import jax
import jax.numpy as jnp
from jax import jit, vmap

from sheap.Utils.Utils.interp_tools import cubic_spline_coefficients, spline_eval


#basic correction for reddening and redshift.

#@jit
@ft.partial(vmap, in_axes=(0, 0, 0), out_axes=0)
def unred(wave, flux, ebv, R_V=3.1, LMC2=False, AVGLMC=False):
    LMC2, AVGLMC = False, False
    x = 10000.0 / wave  # Convert to inverse microns
    # Set default values
    x0 = 4.596
    gamma = 0.99
    c3 = 3.23
    c4 = 0.41
    c2 = -0.824 + 4.717 / R_V
    c1 = 2.030 - 3.007 * c2
    # Update coefficients based on LMC2 or AVGLMC flags
    if LMC2:
        x0 = 4.626
        gamma = 1.05
        c4 = 0.42
        c3 = 1.92
        c2 = 1.31
        c1 = -2.16
    elif AVGLMC:
        x0 = 4.596
        gamma = 0.91
        c4 = 0.64
        c3 = 2.73
        c2 = 1.11
        c1 = -1.28
    xcutuv = 10000.0 / 2700.0
    xspluv = 10000.0 / jnp.array([2700.0, 2600.0])
    XUV = jnp.concatenate((xspluv, jnp.where(x >= xcutuv, x, jnp.zeros_like(x))))
    yuv_ = c1 + c2 * XUV
    yuv_ += c3 * XUV**2 / ((XUV**2 - x0**2) ** 2 + (XUV * gamma) ** 2)
    yuv_ += c4 * (
        0.5392 * (jnp.maximum(XUV, 5.9) - 5.9) ** 2
        + 0.05644 * (jnp.maximum(XUV, 5.9) - 5.9) ** 3
    )
    yuv_ += R_V
    yspluv_ = yuv_[0:2]  # save spline points
    XSPLOPIR = jnp.concatenate(
        (
            jnp.array([0]),
            10000.0 / jnp.array([26500.0, 12200.0, 6000.0, 5470.0, 4670.0, 4110.0]),
        )
    )
    ysplir_ = jnp.array([0.0, 0.26469, 0.82925]) * R_V / 3.1
    ysplop_ = jnp.array(
        [
            jnp.polyval(jnp.array([-4.22809e-01, 1.00270, 2.13572e-04][::-1]), R_V),
            jnp.polyval(jnp.array([-5.13540e-02, 1.00216, -7.35778e-05][::-1]), R_V),
            jnp.polyval(jnp.array([7.00127e-01, 1.00184, -3.32598e-05][::-1]), R_V),
            jnp.polyval(
                jnp.array([1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05][::-1]),
                R_V,
            ),
        ]
    )
    ysplopir_ = jnp.concatenate((ysplir_, ysplop_))

    # interpol = jnp.interp(x,jnp.concatenate((XSPLOPIR,xspluv)),jnp.concatenate((ysplopir_,yspluv_)))
    yk, bk, ck, dk = cubic_spline_coefficients(
        jnp.concatenate((XSPLOPIR, xspluv)), jnp.concatenate((ysplopir_, yspluv_))
    )
    interpol = spline_eval(x, jnp.concatenate((XSPLOPIR, xspluv)), yk, bk, ck, dk)

    # curve = jnp.zeros_like(x)
    # interpol = jnp.interp(x,jnp.concatenate((XSPLOPIR,xspluv)),jnp.concatenate((ysplopir_,yspluv_)))
    curve = jnp.where(x < xcutuv, interpol, jnp.zeros_like(x))
    curve = jnp.where(x >= xcutuv, yuv_[2::], curve)
    return flux * 10.0 ** (0.4 * curve * ebv)


@ft.partial(vmap, in_axes=(0, 0), out_axes=0)
def deredshift(spectra, z):
    # PyQSO DR16 pass the results in redshift
    spectra = spectra.at[[1, 2], :].multiply(1 + z[jnp.newaxis, jnp.newaxis])
    spectra = spectra.at[0, :].divide(1 + z[jnp.newaxis])
    return spectra
