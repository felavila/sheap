"""This module ?."""
__author__ = 'felavila'

__all__ = [
    "TEMPLATES_PATH",
    "make_feii_template_function",
    "make_host_function",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path


import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from sheap.Profiles.utils import with_param_names


TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "SuportData" / "templates"

FEII_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "feop": {
        "file": TEMPLATES_PATH / "fe2_Op.dat",
        "central_wl": 4650.0,
        "sigmatemplate": 900.0 / 2.355,
        "fixed_dispersion": None,
    },
    "feuv": {
        "file": TEMPLATES_PATH / "fe2_UV02.dat",
        "central_wl": 2795.0,
        "sigmatemplate": 900.0 / 2.355,
        "fixed_dispersion": 106.3, 
    },
    "feuvop":{"file": TEMPLATES_PATH / "uvofeii1000kms.txt",
        "central_wl": 4570.0,
        "sigmatemplate": 1000.0 / 2.355},
    "BalHiOrd":{"file": TEMPLATES_PATH / "BalHiOrd_FWHM1000.dat",
                "sigmatemplate": 1000.0 / 2.355,
                "central_wl": 3675.0
                }
}

def make_feii_template_function(
    name: str,
    x_min: Optional[float] = None,  # Angstroms (linear)
    x_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Factory for a FeII template model by name, with optional wavelength cuts.

    Looks up path, central_wl, sigmatemplate, and optional fixed_dispersion in FEII_TEMPLATES.

    If x_min/x_max are provided, the template spectrum is cut to [x_min, x_max]
    with a ±50 Å guard band to reduce boundary artifacts in the FFT broadening, and
    then re-normalized to unit sum.

    Returns:
      {
        'model': Callable(x, params) -> flux,  # has .param_names, .n_params
        'template_info': {
            'name', 'file', 'central_wl', 'sigmatemplate',
            'fixed_dispersion', 'x_min', 'x_max', 'dl'
        }
      }
    """
    cfg = FEII_TEMPLATES.get(name)
    if cfg is None:
        raise KeyError(f"No such FeII template: {name}")

    path          = cfg["file"]
    central_wl    = cfg["central_wl"]
    sigmatemplate = cfg["sigmatemplate"]
    user_fd       = cfg.get("fixed_dispersion", None)

    data = np.loadtxt(path, comments="#").T
    wl   = np.array(data[0], dtype=np.float64)
    flux = np.array(data[1], dtype=np.float64)

    # Optional wavelength cut with ±50 Å margin (like host model)
    if x_min is not None or x_max is not None:
        print("cutting between",x_min,x_max)
        mask = np.ones_like(wl, dtype=bool)
        if x_min is not None:
            mask &= wl >= max(x_min - 50.0, wl.min())
        if x_max is not None:
            mask &= wl <= min(x_max + 50.0, wl.max())
        if not np.any(mask):
            raise ValueError("No wavelength values left after applying x_min/x_max cut.")
        wl   = wl[mask]
        flux = flux[mask]

    # Ensure equally spaced grid
    if wl.size < 3:
        raise ValueError("Template too short after cutting; need at least 3 points.")
    dl = float(wl[1] - wl[0])

    # Re-normalize to unit sum AFTER any cut
    unit_flux = flux / np.clip(np.sum(flux), 1e-10, np.inf)

    # km/s per pixel ≈ (dλ/λ) * c  (c ~ 3e5 km/s)
    if user_fd is None:
        fixed_dispersion = (dl / central_wl) * 3e5
    else:
        fixed_dispersion = float(user_fd)

    param_names = ["logamp", "logFWHM", "shift"]

    @with_param_names(param_names)
    def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        logamp, logFWHM, shift = params
        amp   = 10.0 ** logamp
        FWHM  = 10.0 ** logFWHM
        sigma_model = FWHM / 2.355
        # quadratic subtraction of the template's intrinsic sigma
        delta_sigma = jnp.sqrt(jnp.maximum(sigma_model**2 - sigmatemplate**2, 1e-12))

        # Convert km/s → Å (at central λ), then → pixels
        sigma_lambda = delta_sigma * central_wl / 3e5  # Å
        sigma_pix    = sigma_lambda / dl               # pixels

        n_pix = unit_flux.shape[0]
        # FFT-based Gaussian broadening
        freq     = jnp.fft.fftfreq(n_pix, d=dl)
        gauss_tf = jnp.exp(-2.0 * (jnp.pi * freq * sigma_pix) ** 2)
        spec_fft = jnp.fft.fft(jnp.asarray(unit_flux))
        broadened = jnp.real(jnp.fft.ifft(spec_fft * gauss_tf))

        # Apply shift in Å and interpolate onto x
        shifted_wl = jnp.asarray(wl) + shift
        interp = jnp.interp(x, shifted_wl, broadened, left=0.0, right=0.0)
        return amp * interp

    return {
        "model": model,
        "template_info": {
            "name": name,
            "file": str(path),
            "central_wl": float(central_wl),
            "sigmatemplate": float(sigmatemplate),
            "fixed_dispersion": float(fixed_dispersion),
            "x_min": None if x_min is None else float(x_min),
            "x_max": None if x_max is None else float(x_max),
            "dl": dl,
        },
    }

# def make_feii_template_function(
#     name: str
# ) -> Dict[str, Any]:
#     """
#     Factory for a FeII template model by name.
#     Looks up path, central_wl, sigmatemplate, and optional fixed_dispersion in FEII_TEMPLATES.

#     Returns:
#       {
#         'model': Callable(x, params) -> flux,  # has .param_names, .n_params
#         'template_info': { file, central_wl, sigmatemplate,
#                            fixed_dispersion, unit_sum }
#       }
#     """
#     cfg = FEII_TEMPLATES.get(name)
#     if cfg is None:
#         raise KeyError(f"No such FeII template: {name}")

#     path          = cfg["file"]
#     central_wl    = cfg["central_wl"]
#     sigmatemplate = cfg["sigmatemplate"]
#     user_fd       = cfg.get("fixed_dispersion", None)

#     data = np.loadtxt(path, comments="#").T
#     wl   = np.array(data[0])
#     flux = np.array(data[1])
#     unit_flux = flux / np.clip(jnp.sum(flux), 1e-10, jnp.inf)

#     dl = float(wl[1] - wl[0])  # Å per pixel
#     if user_fd is None:
#         # km/s per pixel ≈ (dλ/λ) * c
#         fixed_dispersion = (dl / central_wl) * 3e5
#     else:
#         fixed_dispersion = user_fd

#     param_names = ["logamp","logFWHM","shift"]

#     @with_param_names(param_names)
#     def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
#         logamp, logFWHM, shift = params
#         amp = 10** logamp
#         FWHM = 10 ** logFWHM
#         sigma_model = FWHM / 2.355
#         delta_sigma = jnp.sqrt(jnp.maximum(sigma_model**2 - sigmatemplate**2, 1e-12))

#         # convert km/s → Å at central λ, then to pixel units
#         sigma_lambda = delta_sigma * central_wl / 3e5   # Å
#         sigma_pix    = sigma_lambda / dl                # pixels

#         n_pix = unit_flux.shape[0]
#         freq  = jnp.fft.fftfreq(n_pix, d=dl)
#         gauss_tf = jnp.exp(-2 * (jnp.pi * freq * sigma_pix)**2)
#         spec_fft = jnp.fft.fft(unit_flux)
#         broadened = jnp.real(jnp.fft.ifft(spec_fft * gauss_tf))

#         # shift & scale
#         shifted_wl = wl + shift
#         interp = jnp.interp(x, shifted_wl, broadened, left=0.0, right=0.0)
#         return amp * interp

#     return {
#         "model": model,
#         "template_info": {
#             "name": name,
#             "central_wl": central_wl,
#             "sigmatemplate": sigmatemplate,
#             "fixed_dispersion": fixed_dispersion,
#         },
#     }
    

    
 
def make_host_function(
    filename: str = TEMPLATES_PATH / "miles_cube_log.npz",
    #miles_cube_log_old_hightres.npz,"miles_cube_log.npz
    z_include: Optional[Union[tuple[float, float], list[float]]] = [-0.7, 0.22],
    age_include: Optional[Union[tuple[float, float], list[float]]] = [0.1, 10.0],
    x_min: Optional[float] = None,  # in Angstroms (linear)
    x_max: Optional[float] = None,
    **kwargs,
) -> dict:
    """
    Load host model from a .npz cube file and return a functional host model interface.
    Allows sub-selection of Z, age, and wavelength (via x_min, x_max in Angstroms).

    Returns:
    {
        'model': Callable[[x, params], flux], with attributes .param_names, .n_params
        'host_info': dict[str, np.ndarray]
    }
    3 x 41 it can 
    7 x 50, it cant
    """
    data = np.load(filename)

    cube = data["cube_log"]
    wave = data["wave_log"]
    all_ages = data["ages_sub"]
    all_zs = data["zs_sub"]
    sigmatemplate = float(data["sigmatemplate"])
    fixed_dispersion = float(data["fixed_dispersion"])

    #print(f"cube.sum(): {cube.sum()}, cube.shape:{cube.shape}")
    if z_include is not None:
        z_min, z_max = np.min(z_include), np.max(z_include)
        z_mask = (all_zs >= z_min) & (all_zs <= z_max)
        if not np.any(z_mask):
            raise ValueError(f"No metallicities in range {z_min} to {z_max}")
        zs = all_zs[z_mask]
        cube = cube[z_mask, :, :]
    else:
        zs = all_zs

    
    if age_include is not None:
        a_min, a_max = np.min(age_include), np.max(age_include)
        a_mask = (all_ages >= a_min) & (all_ages <= a_max)
        if not np.any(a_mask):
            raise ValueError(f"No ages in range {a_min} to {a_max}")
        ages = all_ages[a_mask]
        cube = cube[:, a_mask, :]
    else:
        ages = all_ages

    if x_min is not None or x_max is not None:
        mask = np.ones_like(wave, dtype=bool)
        #to avoid border issues ?
        if x_min is not None:
            mask &= wave >= max([x_min - 50, min(wave)])
        if x_max is not None:
            mask &= wave <= min([x_max + 50,max(wave)])

        if not np.any(mask):
            raise ValueError("No wavelength values left after applying x_min/x_max cut.")

        wave = wave[mask]
        cube = cube[:, :, mask].astype(np.float32)
    
    dx = wave[1] - wave[0]
    n_Z, n_age, n_pix = cube.shape
    #print(f"Host added with n_Z: {n_Z} and n_age: {n_age}")
    templates_flat = cube.reshape(-1, n_pix)
    grid_metadata = [(float(Z), float(age)) for Z in zs for age in ages]

    param_names = ["logamp", "logFWHM", "shift"]
    for Z, age in grid_metadata:
        zstr = str(Z).replace(".", "p")
        astr = str(age).replace(".", "p")
        param_names.append(f"weight_Z{zstr}_age{astr}")
    
    templates_flat_jax = jnp.asarray(templates_flat)
    templates_fft = jnp.fft.fft(templates_flat_jax, axis=1)
    @with_param_names(param_names)
    def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        logamp = params[0]
        amplitude = 10**logamp
        logFWHM = params[1]
        shift_A = params[2]
        weights = params[3:]

        FWHM = 10 ** logFWHM
        sigma_model = FWHM / 2.355
        delta_sigma = jnp.sqrt(jnp.maximum(sigma_model**2 - sigmatemplate**2, 1e-12))
        sigma_pix = delta_sigma / fixed_dispersion
        sigma_lambda = sigma_pix * dx

        freq = jnp.fft.fftfreq(n_pix, d=dx)
        gauss_tf = jnp.exp(-2 * (jnp.pi * freq * sigma_lambda) ** 2)
        #templates_flat_jax = jnp.asarray(templates_flat)  # or do this earlier once
        #templates_fft = jnp.fft.fft(templates_flat_jax, axis=1)
        convolved = jnp.real(jnp.fft.ifft(templates_fft * gauss_tf, axis=1))

        model_flux = jnp.sum(weights[:, None] * convolved, axis=0)
        shifted = wave + shift_A
        return amplitude * jnp.interp(x, shifted, model_flux, left=0.0, right=0.0)

    return {
        "model": model,
        "host_info": {
            "z_include": zs,
            "age_include": ages,
            "n_Z": n_Z,
            "n_age": n_age,
            "x_min": x_min,
            "x_max": x_max,
        },
    }
   
