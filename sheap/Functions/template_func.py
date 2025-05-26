from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path


import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from sheap.Functions.utils import param_count
from sheap.Tools.spectral_basic import kms_to_wl

#TEMPLATE FAMILIES SHOULD BE ADDED 

templates_path = Path(__file__).resolve().parent.parent.parent / "SuportData" / "templates"


fe_template_OP_file =   templates_path / 'fe2_Op.dat'

fe_template_OP = jnp.array(
    np.loadtxt(fe_template_OP_file, comments='#').transpose()
)  # y units?

fe_template_UV_file = templates_path / 'fe2_UV02.dat'

fe_template_UV = jnp.array(
    np.loadtxt(fe_template_UV_file, comments='#').transpose()
) 

class FeIITemplateModel:
    def __init__(self, template: Tuple[jnp.ndarray, jnp.ndarray], central_wl: float,sigmatemplate: float):
        self.wl, self.flux = template
        self.central_wl = central_wl #Ang
        self.sigmatemplate =sigmatemplate #km/s

    def __call__(self, x: jnp.ndarray, log_FWHM: float, shift_: float, scale: float):
        FWHM = 10 ** log_FWHM
        sigma_model = FWHM/ 2.355
        shift = 10 * shift_

        delta_sigma = jnp.sqrt(jnp.maximum(sigma_model**2 - self.sigmatemplate**2, 1e-12))
        dl = jnp.maximum(x[1] - x[0], 1e-6)
        sigma_wl = kms_to_wl(delta_sigma, self.central_wl) / dl

        max_radius = 1000
        x_local = jnp.arange(-max_radius, max_radius + 1)
        mask = jnp.where(jnp.abs(x_local) <= jnp.round(4 * sigma_wl), 1.0, 0.0)
        kernel = jsp.stats.norm.pdf(x_local, scale=sigma_wl) * mask
        kernel /= jnp.sum(kernel)

        broadened = jsp.signal.convolve(self.flux, kernel, mode='same', method='fft')
        shifted = self.wl + shift
        return scale * jnp.interp(x, shifted, broadened, left=None, right=None)

fitFeOP_model = FeIITemplateModel(template=fe_template_OP, central_wl=4650.0,sigmatemplate=900.0 / 2.355)
fitFeUV_model = FeIITemplateModel(template=fe_template_UV, central_wl=2795.0,sigmatemplate=900.0 / 2.355)

@param_count(3)
def fitFeOP(x, params): return fitFeOP_model(x, *params)

@param_count(3)
def fitFeUV(x, params): return fitFeUV_model(x, *params)