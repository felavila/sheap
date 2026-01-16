"""
Template-Based Profiles
=======================

This module provides template-driven spectral components used in *sheap*:

- **Fe II templates** (UV, optical, combined) read from ASCII files and broadened
    via FFT convolution.
- **Balmer high-order blends** represented as fixed templates.
- **Host galaxy templates** based on E-MILES SSP cubes, sub-selected in metallicity,
    age, and wavelength, and combined with free weights.

Functions
---------
- ``make_feii_template_function`` :
    Factory for Fe II template models by name. Supports optional wavelength cuts
    and returns a JAX-ready profile function plus template metadata.
- ``make_host_function`` :
    Factory for host galaxy models from a precomputed SSP cube. Uses efficient
    memory mapping and a single FFT-based convolution of the weighted template sum.

Constants
---------
- ``TEMPLATES_PATH`` : Path to the bundled template data directory.
- ``FEII_TEMPLATES`` : Registry of available Fe II template definitions.

Notes
-----
- All returned models are decorated with ``@with_param_names`` and are JAX-compatible.
- FFT-based Gaussian broadening quadratically subtracts the intrinsic template
    resolution before applying user-defined FWHM.
- Host models build parameter names dynamically as ``weight_Z{Z}_age{age}``
    for each included SSP grid point.

Todo
----
Rename ``make_feii_template_function`` for a more general function. 
"""

__author__ = 'felavila'

__all__ = [
    "TEMPLATES_PATH",
    "make_feii_template_function",
    "make_host_function",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path

import jax 
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from sheap.Profiles.Utils import with_param_names
from sheap.Utils.Constants import C_KMS

TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "SuportData" / "templates"

TEMPLATES: Dict[str, Dict[str, Any]] = {
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

def make_template_function(
    name: str,
    x_min: Optional[float] = None,  # Angstroms (linear)
    x_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Factory for a FeII template model by name, with optional wavelength cuts.

    Looks up path, central_wl, sigmatemplate, and optional fixed_dispersion in TEMPLATES.

    If x_min/x_max are provided, the template spectrum is cut to [x_min, x_max]
    with a ±50 Å guard band to reduce boundary artifacts in the FFT broadening, and
    then re-normalized to unit sum.

    Notes
    -----
    The third parameter is **vshift_kms** (velocity shift in km/s), applied as a
    multiplicative stretch of the wavelength grid:
        wl_shifted = wl * (1 + vshift_kms / c_kms)

    Returns
    -------
    dict
        {
          'model': Callable(x, params) -> flux,  # has .param_names, .n_params
          'template_info': {
              'name', 'file', 'central_wl', 'sigmatemplate',
              'fixed_dispersion', 'x_min', 'x_max', 'dl'
          }
        }
    """
    cfg = TEMPLATES.get(name)
    if cfg is None:
        raise KeyError(f"No such template: {name}")

    path          = cfg["file"]
    central_wl    = cfg["central_wl"]
    sigmatemplate = cfg["sigmatemplate"]
    user_fd       = cfg.get("fixed_dispersion", None)

    data = np.loadtxt(path, comments="#").T
    wl   = np.array(data[0], dtype=np.float64)
    flux = np.array(data[1], dtype=np.float64)

    # Optional wavelength cut with ±50 Å margin
    if x_min is not None or x_max is not None:
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

    if user_fd is None:
        fixed_dispersion = (dl / central_wl) * C_KMS
    else:
        fixed_dispersion = float(user_fd)

    param_names = ["logamp", "logFWHM", "vshift_kms"]

    # Pre-pack constants as JAX arrays once
    wl_jax = jnp.asarray(wl)
    unit_flux_jax = jnp.asarray(unit_flux)

    @with_param_names(param_names)
    def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        logamp, logFWHM, vshift_kms = params

        amp  = 10.0 ** logamp
        FWHM = 10.0 ** logFWHM              # km/s
        sigma_model = FWHM / 2.355          # km/s

        # Quadratic subtraction of template intrinsic sigma (km/s), made safe
        diff_sq = sigma_model**2 - sigmatemplate**2
        diff_sq_safe = jax.nn.softplus(diff_sq / 10.0) * 10.0 + 1e-12
        delta_sigma = jnp.sqrt(diff_sq_safe)  # km/s

        # Convert km/s broadening -> pixels (fixed_dispersion is km/s per pixel)
        sigma_pix = delta_sigma / fixed_dispersion

        n_pix = unit_flux_jax.shape[0]
        freq = jnp.fft.fftfreq(n_pix, d=1.0)  # pixel-frequency
        gauss_tf = jnp.exp(-2.0 * (jnp.pi * freq * sigma_pix) ** 2)

        spec_fft = jnp.fft.fft(unit_flux_jax)
        broadened = jnp.real(jnp.fft.ifft(spec_fft * gauss_tf))

        # --- velocity shift (positive -> redder features) ---
        beta = vshift_kms / C_KMS
        xp = wl_jax * (1.0 + beta)

        interp = jnp.interp(x, xp, broadened, left=0.0, right=0.0)
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



def make_host_function(
    filename: str = TEMPLATES_PATH / "miles_cube_log.npz",
    #filename: str = TEMPLATES_PATH / "xsl_cube_log.npz",
    z_include: Optional[Union[tuple[float, float], list[float]]] = [-0.7, 0.22],
    age_include: Optional[Union[tuple[float, float], list[float]]] = [0.1, 10.0],
    xmin: Optional[float] = None, 
    xmax: Optional[float] = None,
    verbose: Optional[bool] = None,
    **kwargs,
) -> dict:
    """
    Memory-lean host model:
      - sums weighted templates first, then does a single FFT-based convolution
      - np.load(..., mmap_mode='r') to reduce RAM pressure
      - keeps arrays in float32

    Parameters
    ----------
    The third parameter is vshift_kms: a velocity shift in km/s.
    """
    #f = 1.0
    #print(filename)
    #z_source = 2.16
    #z_lens = 0.905
    #f = (1 + z_source) / (1 + z_lens)
    f = 1.
    data = np.load(filename, mmap_mode="r")

    cube = np.asarray(data["cube_log"], dtype=np.float32)   # (n_Z, n_age, n_pix)
    wave = np.asarray(data["wave_log"], dtype=np.float32)
    all_ages = np.asarray(data["ages_sub"], dtype=np.float32)
    all_zs = np.asarray(data["zs_sub"], dtype=np.float32)
    sigmatemplate = float(data["sigmatemplate"])
    fixed_dispersion = float(data["fixed_dispersion"])

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

    if xmin is not None or xmax is not None:
        mask = np.ones_like(wave, dtype=bool)
        if xmin is not None:
            mask &= wave >= max([xmin * f - 50.0, float(wave.min())])
        if xmax is not None:
            mask &= wave <= min([xmax * f + 50.0, float(wave.max())])
        if not np.any(mask):
            raise ValueError("No wavelength values left after applying x_min/x_max cut.")
        wave = wave[mask].astype(np.float32, copy=False)
        cube = cube[:, :, mask].astype(np.float32, copy=False)
    #print(wave)
    dx = float(wave[1] - wave[0])
    n_Z, n_age, n_pix = cube.shape
    if verbose:
        print(f"Host added with n_Z: {n_Z} and n_age: {n_age}")
    
    eps = 1e-30
    flux_int = np.nansum(cube, axis=-1, keepdims=True)  
    cube = cube / (flux_int + eps)
    
    templates_flat = cube.reshape(-1, n_pix)                # numpy array
    grid_metadata = [(float(Z), float(age)) for Z in zs for age in ages]

   
    param_names = ["logamp", "logFWHM", "vshift_kms"]
    for Z, age in grid_metadata:
        zstr = str(Z).replace(".", "p")
        astr = str(age).replace(".", "p")
        param_names.append(f"weight_Z{zstr}_age{astr}")

    templates_jax = jnp.asarray(templates_flat)             # (N, P) float32
    wave_jax = jnp.asarray(wave)
    #print(wave_jax)
    freq = jnp.fft.fftfreq(n_pix, d=dx).astype(jnp.float32) # (P,)


    @with_param_names(param_names)
    def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        logamp = params[0]
        amplitude = 10.0 ** logamp

        logFWHM = params[1]
        vshift_kms = params[2]
        weights = params[3:]                                 # (N,)

        base = jnp.tensordot(weights, templates_jax, axes=(0, 0))  # (P,)

        # --- broadening ---
        FWHM = 10.0 ** logFWHM                  # total FWHM in km/s
        sigma_model = FWHM / 2.355              # km/s

        diff_sq = sigma_model**2 - sigmatemplate**2
        diff_sq_safe = jax.nn.softplus(diff_sq / 10.0) * 10.0 + 1e-12
        delta_sigma = jnp.sqrt(diff_sq_safe)    # km/s

        sigma_pix = delta_sigma / fixed_dispersion
        sigma_lambda = sigma_pix * dx

        gauss_tf = jnp.exp(-2.0 * (jnp.pi * freq * sigma_lambda) ** 2)
        base_fft = jnp.fft.fft(base)
        conv = jnp.real(jnp.fft.ifft(base_fft * gauss_tf))


        beta = vshift_kms / C_KMS
        xp = wave_jax * (1.0 + beta)  

        return amplitude * jnp.interp(x*f, xp, conv, left=0.0, right=0.0)

    return {
        "model": model,
        "host_info": {
            "z_include": zs,
            "age_include": ages,
            "n_Z": n_Z,
            "n_age": n_age,
            "xmin": xmin,
            "xmax": xmax,
        },
    }
    