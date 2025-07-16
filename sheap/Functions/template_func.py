from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from sheap.Functions.utils import param_count
from sheap.Tools.spectral_basic import kms_to_wl

# -- lazy loading of templates and models to avoid import-time execution --

templates_path = Path(__file__).resolve().parent.parent / "SuportData" / "templates"

_fe_template_OP = None
_fe_template_UV = None
_fitFeOP_model = None
_fitFeUV_model = None


def load_fe_template_OP() -> Tuple[jnp.ndarray, jnp.ndarray]:
    global _fe_template_OP
    if _fe_template_OP is None:
        file = templates_path / 'fe2_Op.dat'
        arr = np.loadtxt(file, comments='#').T
        wl, flux = arr[:, 0], arr[:, 1]
        flux = flux / np.max(flux)
        _fe_template_OP = (jnp.array(wl), jnp.array(flux))
    return _fe_template_OP


def load_fe_template_UV() -> Tuple[jnp.ndarray, jnp.ndarray]:
    global _fe_template_UV
    if _fe_template_UV is None:
        file = templates_path / 'fe2_UV02.dat'
        arr = np.loadtxt(file, comments='#').T
        wl, flux = arr[:, 0], arr[:, 1]
        flux = flux / np.max(flux)
        _fe_template_UV = (jnp.array(wl), jnp.array(flux))
    return _fe_template_UV


class FeIITemplateModel:
    def __init__(self, template: Tuple[jnp.ndarray, jnp.ndarray], central_wl: float, sigmatemplate: float):
        self.wl, self.flux = template
        self.central_wl = central_wl
        self.sigmatemplate = sigmatemplate

    def __call__(self,
                 x: jnp.ndarray,
                 log_FWHM: float,
                 shift: float,
                 scale: float) -> jnp.ndarray:
        FWHM = 10 ** log_FWHM
        sigma_model = FWHM / 2.355
        delta_sigma = jnp.sqrt(jnp.maximum(sigma_model**2 - self.sigmatemplate**2, 1e-12))
        sigma_lambda = kms_to_wl(delta_sigma, self.central_wl)
        dl = jnp.maximum(x[1] - x[0], 1e-6)
        n_pix = self.flux.shape[0]
        freq = jnp.fft.fftfreq(n_pix, d=dl)
        gauss_tf = jnp.exp(-2 * (jnp.pi * freq * sigma_lambda)**2)
        flux_fft = jnp.fft.fft(self.flux)
        broadened = jnp.real(jnp.fft.ifft(flux_fft * gauss_tf))
        shifted_wl = self.wl + shift
        return scale * jnp.interp(x, shifted_wl, broadened, left=0.0, right=0.0)


def get_fitFeOP_model() -> FeIITemplateModel:
    global _fitFeOP_model
    if _fitFeOP_model is None:
        template = load_fe_template_OP()
        _fitFeOP_model = FeIITemplateModel(template, central_wl=4650.0, sigmatemplate=900.0/2.355)
    return _fitFeOP_model


def get_fitFeUV_model() -> FeIITemplateModel:
    global _fitFeUV_model
    if _fitFeUV_model is None:
        template = load_fe_template_UV()
        _fitFeUV_model = FeIITemplateModel(template, central_wl=2795.0, sigmatemplate=900.0/2.355)
    return _fitFeUV_model


@param_count(3)
def fitFeOP(x: jnp.ndarray, params: Tuple[float, float, float]) -> jnp.ndarray:
    return get_fitFeOP_model()(x, *params)

@param_count(3)
def fitFeUV(x: jnp.ndarray, params: Tuple[float, float, float]) -> jnp.ndarray:
    return get_fitFeUV_model()(x, *params)


# Additional fixed-dispersion model can be refactored similarly if needed.
