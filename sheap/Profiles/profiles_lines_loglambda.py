from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
from jax import jit, vmap,lax 
from jax.scipy.special import erfc
from jax.scipy.stats import norm #maybe dosent exist xd

from sheap.Profiles.Utils import with_param_names,trapz_jax


C_KMS = 299_792.458
FWHM_TO_SIGMA = 1.0 / 2.355

@with_param_names(["amplitude", "vshift_kms", "fwhm_v_kms", "lambda0"])
def gaussian_fwhm_loglambda(x_lambda, params):
    """
    Velocity-symmetric Gaussian in log(lambda) space.

    Parameters
    ----------
    x_lambda : jnp.ndarray
        Wavelength grid.
    params : [amp, vshift_kms, fwhm_v_kms, lambda0]
        amp : Linear amplitude
        vshift_kms : Centroid velocity shift [km/s]
        fwhm_v_kms : log10(FWHM [km/s])
        lambda0 : Rest wavelength of the line (required, fixed)
    """
    amp, vshift_kms, fwhm_v_kms, lambda0 = params

    ratio = jnp.clip(x_lambda / lambda0, a_min=jnp.finfo(x_lambda.dtype).tiny, a_max=jnp.inf)
    y = C_KMS * jnp.log(ratio)

    fwhm_linear = 10.0 ** fwhm_v_kms
    sigma_v = fwhm_linear * FWHM_TO_SIGMA

    z = (y - vshift_kms) / sigma_v
    return amp * jnp.exp(-0.5 * z * z)