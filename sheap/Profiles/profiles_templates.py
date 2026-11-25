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
from scipy.fft import rfft, irfft, rfftfreq, next_fast_len

import jax 
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from sheap.Profiles.Utils import with_param_names
from sheap.Utils.Constants import DEFAULT_C_KMS

TEMPLATES_PATH = Path(__file__).resolve().parent.parent / "SuportData" / "templates"

TEMPLATES: Dict[str, Dict[str, Any]] = {"feop": { "file": TEMPLATES_PATH / "fe2_Op.dat", "central_wl": 4650.0, "sigmatemplate": 900.0 / 2.355,"fixed_dispersion": None,},
    "feuv": {"file": TEMPLATES_PATH / "fe2_UV02.dat","central_wl": 2795.0,"sigmatemplate": 900.0 / 2.355,"fixed_dispersion": 106.3, },
    "feuvop":{"file": TEMPLATES_PATH / "uvofeii1000kms.txt", "central_wl": 4570.0, "sigmatemplate": 1000.0 / 2.355},
    "BalHiOrd":{"file": TEMPLATES_PATH / "BalHiOrd_FWHM1000.dat", "sigmatemplate": 1000.0 / 2.355, "central_wl": 3675.0}}

def make_template_function(name, x_min=None, x_max=None,TEMPLATES=TEMPLATES):
    """Create template profile with JAX and NumPy float64 implementations.

    Parameters (physical SHEAP order): [logamp, logFWHM, vshift_kms].
    All three are nonlinear in the current parameterization.
    The design builder therefore returns (full_flux, empty [n_pix, 0]).
    """
    cfg = TEMPLATES.get(name)
    if cfg is None:
        raise KeyError(f"No such template: {name}")

    path = cfg["file"]
    central_wl = float(cfg["central_wl"])
    sigma_template64 = float(cfg["sigmatemplate"])
    user_fd = cfg.get("fixed_dispersion", None)

    data = np.loadtxt(path, comments="#")
    wl64 = np.asarray(data[:, 0], dtype=np.float64)
    flux64 = np.asarray(data[:, 1], dtype=np.float64)

    # Keep the original SHEAP wavelength-selection rule for parity testing.
    if x_min is not None or x_max is not None:
        mask = np.ones(wl64.shape, dtype=bool)
        if x_min is not None:
            mask &= wl64 >= max(float(x_min) - 50.0, float(wl64.min()))
        if x_max is not None:
            mask &= wl64 <= min(float(x_max) + 50.0, float(wl64.max()))
        if not np.any(mask):
            raise ValueError(f"No template wavelengths left for {name}")
        wl64, flux64 = wl64[mask], flux64[mask]

    if wl64.size < 3 or not np.all(np.isfinite(wl64)) or np.any(np.diff(wl64) <= 0):
        raise ValueError(f"Invalid template wavelength grid for {name}")

    dl = float(wl64[1] - wl64[0])
    fixed_dispersion64 = (
        dl / central_wl * float(DEFAULT_C_KMS)
        if user_fd is None else float(user_fd)
    )
    if fixed_dispersion64 <= 0:
        raise ValueError("Template fixed_dispersion must be > 0")

    # Preserve float64 before converting to float32 for JAX.
    unit_flux64 = flux64 / np.clip(np.sum(flux64), 1e-10, np.inf)
    unit_flux32 = np.asarray(unit_flux64, dtype=np.float32)
    n_pix = len(unit_flux64)

    # Precompute Fourier representations once for each backend.
    fft64 = rfft(unit_flux64)
    freq64 = rfftfreq(n_pix, d=1.0)
    fft32 = jnp.asarray(rfft(unit_flux32), dtype=jnp.complex64)
    freq32 = jnp.asarray(rfftfreq(n_pix, d=1.0), dtype=jnp.float32)
    wl32 = jnp.asarray(wl64, dtype=jnp.float32)

    c64 = float(DEFAULT_C_KMS)
    c32 = jnp.float32(c64)
    sigma_template32 = jnp.float32(sigma_template64)
    dispersion32 = jnp.float32(fixed_dispersion64)

    def jax_build_shape(x, logfwhm, vshift):
        x = jnp.asarray(x, dtype=jnp.float32)
        fwhm = jnp.power(jnp.float32(10.0), logfwhm)
        sigma_model = fwhm / jnp.float32(2.355)
        delta_sq = sigma_model**2 - sigma_template32**2
        delta_sigma = jnp.sqrt(jax.nn.softplus(delta_sq / jnp.float32(10.0)) * jnp.float32(10.0) + jnp.float32(1e-12))
        sigma_pix = delta_sigma / dispersion32
        tf = jnp.exp(-jnp.float32(2.0) * (jnp.pi * freq32 * sigma_pix)**2)
        broadened = jnp.fft.irfft(fft32 * tf, n=n_pix)
        # Preserve the original template velocity convention: (1 + v/c).
        shifted_wl = wl32 * (jnp.float32(1.0) + vshift / c32)
        return jnp.interp(x, shifted_wl, broadened, left=0.0, right=0.0)

    param_names = ["logamp", "logFWHM", "vshift_kms"]
    @with_param_names(param_names, profile_name=name)
    def model(x, params):
        p = jnp.asarray(params, dtype=jnp.float32)
        return jnp.power(jnp.float32(10.0), p[0]) * jax_build_shape(x, p[1], p[2])

    def design_builder(x, params):
        y = model(x, params)
        return y, jnp.empty((y.shape[0], 0), dtype=jnp.float32)

    model.design_builder = design_builder

    info = {
        "name": name,
        "file": str(path),
        "central_wl": central_wl,
        "sigmatemplate": sigma_template64,
        "fixed_dispersion": fixed_dispersion64,
        "x_min": None if x_min is None else float(x_min),
        "x_max": None if x_max is None else float(x_max),
        "dl": dl,
        "n_template_pixels": n_pix,
    }
    def numpy_shape(x, logfwhm, vshift):
        x = np.asarray(x, dtype=np.float64)
        fwhm = 10.0 ** float(logfwhm)
        sigma_model = fwhm / 2.355
        delta_sq = sigma_model**2 - sigma_template64**2
        # Equivalent to jax.nn.softplus for the same precision.
        delta_sigma = np.sqrt(10.0 * np.logaddexp(0.0, delta_sq / 10.0) + 1e-12)
        sigma_pix = delta_sigma / fixed_dispersion64
        tf = np.exp(-2.0 * (np.pi * freq64 * sigma_pix)**2)
        broadened = irfft(fft64 * tf, n=n_pix)
        shifted_wl = wl64 * (1.0 + float(vshift) / c64)
        return np.interp(x, shifted_wl, broadened, left=0.0, right=0.0)

    def numpy_model(x, params):
        p = np.asarray(params, dtype=np.float64)
        if p.shape != (3,):
            raise ValueError(f"{name}: expected 3 physical parameters, got {p.shape}")
        return 10.0**p[0] * numpy_shape(x, p[1], p[2])

    def numpy_design_builder(x, params):
        flux = np.atleast_1d(numpy_model(x, params))
        # logamp is nonlinear, so there are no linear columns yet.
        return flux, np.empty((flux.size, 0), dtype=np.float64)

    model.numpy_model = numpy_model
    model.numpy_design_builder = numpy_design_builder

    model.template_info = info
    return {"model": model, "template_info": info}


def make_host_function(
    filename: str | Path | None = "miles",
    z_include=(-0.7, 0.22),
    age_include=(0.1, 10.0),
    xmin=None,
    xmax=None,
    verbose=False,
    **kwargs,):

    """Create a host SSP model with matching JAX and NumPy interfaces.

    The returned model exposes:
        model(x, params)                    # JAX, float32
        model.design_builder(x, params)     # JAX, float32
        model.numpy_model(x, params)        # NumPy, float64
        model.numpy_design_builder(x, params)  # NumPy, float64

    All NumPy methods are attached ONLY to the model, not to the
    returned result dictionary or its host_info metadata.

    Local physical parameter order:
        [logamp, logFWHM, vshift_kms, weight_0, ..., weight_(N-1)]

    A design builder returns (offset[n_data], columns[n_data, n_templates]).

    Notes
    -----
    This version intentionally preserves the original wavelength cut
    (+/- 50 Angstrom), FFT convolution, normalization and Doppler convention.
    It does not yet implement the separate boundary-safe host improvement.
    """
    external = kwargs.get("data", None)

    if external is not None:
        cube = np.asarray(external["cube_log"], dtype=np.float64)
        wave = np.asarray(external["wave_log"], dtype=np.float64)
        sigmatemplate = float(external["sigmatemplate"])
        n_Z, n_age = cube.shape[:2]
        zs = np.asarray(external.get("zs_sub", np.arange(n_Z)), dtype=float)
        ages = np.asarray(external.get("ages_sub", np.arange(n_age)), dtype=float)
        source = str(external.get("filename", "external"))
        # External cubes intentionally do not use the default z/age selection.
    else:
        if filename is None or filename == "miles":
            filename = TEMPLATES_PATH / "miles_cube_log.npz"
        elif filename == "xsl":
            filename = TEMPLATES_PATH / "xsl_cube_log.npz"
        filename = Path(filename)
        if not filename.exists():
            raise FileNotFoundError(f"Host template file not found: {filename}")
        source = str(filename)

        with np.load(filename, mmap_mode="r") as data:
            cube = np.asarray(data["cube_log"], dtype=np.float64)
            wave = np.asarray(data["wave_log"], dtype=np.float64)
            zs = np.asarray(data["zs_sub"], dtype=np.float64)
            ages = np.asarray(data["ages_sub"], dtype=np.float64)
            sigmatemplate = float(data["sigmatemplate"])

        if z_include is not None:
            lo, hi = np.min(z_include), np.max(z_include)
            z_mask = (zs >= lo) & (zs <= hi)
            if not np.any(z_mask):
                raise ValueError("No metallicities remain after z_include")
            cube, zs = cube[z_mask, :, :], zs[z_mask]

        if age_include is not None:
            lo, hi = np.min(age_include), np.max(age_include)
            a_mask = (ages >= lo) & (ages <= hi)
            if not np.any(a_mask):
                raise ValueError("No ages remain after age_include")
            cube, ages = cube[:, a_mask, :], ages[a_mask]

    if cube.ndim != 3 or wave.ndim != 1 or cube.shape[-1] != wave.size:
        raise ValueError("Expected cube_log (n_Z,n_age,n_pix) and wave_log (n_pix,)")
    if (wave.size < 4 or not np.all(np.isfinite(wave))
            or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
        raise ValueError("wave_log must have >=4 strictly increasing, positive, finite entries")
    if len(zs) != cube.shape[0] or len(ages) != cube.shape[1]:
        raise ValueError("zs_sub and ages_sub must match the cube axes")

    if xmin is not None or xmax is not None:
        mask = np.ones_like(wave, dtype=bool)
        if xmin is not None:
            mask &= wave >= max(float(xmin) - 50.0, float(wave.min()))
        if xmax is not None:
            mask &= wave <= min(float(xmax) + 50.0, float(wave.max()))
        if not np.any(mask):
            raise ValueError("No host wavelengths remain after requested cut")
        wave, cube = wave[mask], cube[:, :, mask]

    n_Z, n_age, n_pix = cube.shape
    n_templates = n_Z * n_age
    if verbose:
        print(f"Host: n_Z={n_Z}, n_age={n_age}, n_templates={n_templates}, n_pix={n_pix}")

    cube = cube / (np.nansum(cube, axis=-1, keepdims=True) + 1e-30)
    templates64 = np.ascontiguousarray(cube.reshape(n_templates, n_pix), dtype=np.float64)
    templates32 = np.ascontiguousarray(templates64.astype(np.float32))

    dln = float(np.mean(np.diff(np.log(wave))))
    freq32 = jnp.asarray(rfftfreq(n_pix, d=dln), dtype=jnp.float32)
    wave32 = jnp.asarray(wave, dtype=jnp.float32)
    templates_jax = jnp.asarray(templates32)
    templates_rfft32 = jnp.asarray(rfft(templates32, axis=-1), dtype=jnp.complex64)

    # Float64 frequency grid and cached FFT, available unconditionally.
    freq64 = rfftfreq(n_pix, d=dln)
    templates_rfft64 = rfft(templates64, axis=-1)

    metadata = [(float(Z), float(age)) for Z in zs for age in ages]
    param_names = ["logamp", "logFWHM", "vshift_kms"]
    linear_names = []
    for Z, age in metadata:
        zstr = str(Z).replace(".", "p")
        astr = str(age).replace(".", "p")
        name = f"weight_Z{zstr}_age{astr}"
        param_names.append(name)
        linear_names.append(name)

    c32 = jnp.asarray(DEFAULT_C_KMS, dtype=jnp.float32)
    sigma_template32 = jnp.asarray(sigmatemplate, dtype=jnp.float32)
    c64 = float(DEFAULT_C_KMS)

    def jax_transfer(logfwhm):
        fwhm = jnp.power(jnp.float32(10.0), logfwhm)
        sigma_model = fwhm / jnp.float32(2.355)
        diff_sq = sigma_model**2 - sigma_template32**2
        delta_sigma = jnp.sqrt(
            jax.nn.softplus(diff_sq / jnp.float32(10.0)) * jnp.float32(10.0)
            + jnp.float32(1e-12)
        )
        sigma_y = delta_sigma / c32
        return jnp.exp(-jnp.float32(2.0) * (jnp.pi * freq32 * sigma_y)**2)

    def numpy_transfer(logfwhm):
        fwhm = 10.0**float(logfwhm)
        sigma_model = fwhm / 2.355
        diff_sq = sigma_model**2 - sigmatemplate**2
        delta_sigma = np.sqrt(10.0 * np.logaddexp(0.0, diff_sq / 10.0) + 1e-12)
        sigma_y = delta_sigma / c64
        return np.exp(-2.0 * (np.pi * freq64 * sigma_y)**2)

    @with_param_names(
        param_names,
        linear_param_names=linear_names,
        profile_name="host",
    )
    def model(x, params):
        x = jnp.asarray(x, dtype=jnp.float32)
        params = jnp.asarray(params, dtype=jnp.float32)
        logamp, logfwhm, vshift = params[:3]
        weights = params[3:]

        # Forward path: weighted sum, then ONE convolution.
        base = jnp.tensordot(weights, templates_jax, axes=(0, 0))
        broadened = jnp.fft.irfft(jnp.fft.rfft(base) * jax_transfer(logfwhm), n=n_pix)
        xp = wave32 * jnp.exp(vshift / c32)
        return jnp.power(jnp.float32(10.0), logamp) * jnp.interp(x, xp, broadened, left=0.0, right=0.0)

    def design_builder(x, params):
        x = jnp.asarray(x, dtype=jnp.float32)
        params = jnp.asarray(params, dtype=jnp.float32)
        logamp, logfwhm, vshift = params[:3]

        broadened = jnp.fft.irfft(
            templates_rfft32 * jax_transfer(logfwhm)[None, :],
            n=n_pix, axis=-1,
        )
        xp = wave32 * jnp.exp(vshift / c32)
        right = jnp.clip(jnp.searchsorted(xp, x, side="right"), 1, n_pix - 1)
        left = right - 1
        fraction = (x - xp[left]) / (xp[right] - xp[left])
        left_flux = jnp.take(broadened, left, axis=1)
        right_flux = jnp.take(broadened, right, axis=1)
        columns = (
            (1.0 - fraction)[None, :] * left_flux
            + fraction[None, :] * right_flux
        ).T
        inside = (x >= xp[0]) & (x <= xp[-1])
        columns = jnp.where(inside[:, None], columns, jnp.float32(0.0))
        columns = jnp.power(jnp.float32(10.0), logamp) * columns
        return jnp.zeros_like(x), columns

    def numpy_model(x, params):
        x = np.asarray(x, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        logamp, logfwhm, vshift = params[:3]
        weights = params[3:]

        # Match JAX: weight the templates before a single FFT/IFFT.
        base = weights @ templates64
        broadened = irfft(rfft(base) * numpy_transfer(logfwhm), n=n_pix)
        xp = wave * np.exp(vshift / c64)
        return 10.0**logamp * np.interp(x, xp, broadened, left=0.0, right=0.0)

    def numpy_design_builder(x, params):
        x = np.asarray(x, dtype=np.float64)
        params = np.asarray(params, dtype=np.float64)
        logamp, logfwhm, vshift = params[:3]

        broadened = irfft(
            templates_rfft64 * numpy_transfer(logfwhm)[None, :],
            n=n_pix, axis=-1,
        )
        xp = wave * np.exp(vshift / c64)
        right = np.clip(np.searchsorted(xp, x, side="right"), 1, n_pix - 1)
        left = right - 1
        fraction = (x - xp[left]) / (xp[right] - xp[left])
        columns = (
            (1.0 - fraction)[None, :] * broadened[:, left]
            + fraction[None, :] * broadened[:, right]
        ).T
        inside = (x >= xp[0]) & (x <= xp[-1])
        columns[~inside, :] = 0.0
        columns *= 10.0**logamp
        return np.zeros_like(x), columns

    # Attach everything only to the profile callable.
    model.design_builder = design_builder
    model.numpy_model = numpy_model
    model.numpy_design_builder = numpy_design_builder

    return {"model": model,
        "host_info": {"z_include": zs,"age_include": ages,
            "n_Z": n_Z, "n_age": n_age, "xmin": xmin, "xmax": xmax,
            "file_name": source,
            "dln": dln, "dv_pix_kms": float(DEFAULT_C_KMS * dln),"sigmatemplate": sigmatemplate,
            "data": external,"cached_template_rfft": True,},}







# def make_template_function_clasic(
#     name: str,
#     x_min: Optional[float] = None,
#     x_max: Optional[float] = None,
# ) -> Dict[str, Any]:
#     """
#     Factory for a template model with optional wavelength cuts.

#     Parameters
#     ----------
#     name
#         Template name in TEMPLATES.

#     x_min, x_max
#         Optional wavelength limits in Angstrom.

#     Returns
#     -------
#     dict
#         {
#             "model": model,
#             "design_builder": design_builder,
#             "template_info": {...},
#         }

#     Notes
#     -----
#     Parameters remain fully nonlinear:

#         [logamp, logFWHM, vshift_kms]

#     Therefore the design builder returns:

#         offset  = complete template model
#         columns = empty array with shape (n_pixels, 0)

#     The intrinsic template rFFT is precomputed once, but the
#     FWHM-dependent broadening kernel and velocity shift are
#     recomputed for every parameter evaluation.
#     """

#     # ============================================================
#     # Template configuration
#     # ============================================================

#     cfg = TEMPLATES.get(name)

#     if cfg is None:
#         raise KeyError(
#             f"No such template: {name}"
#         )

#     path = cfg["file"]
#     central_wl = cfg["central_wl"]
#     sigmatemplate = cfg["sigmatemplate"]

#     user_fd = cfg.get(
#         "fixed_dispersion",
#         None,
#     )

#     # ============================================================
#     # Read template
#     # ============================================================

#     data = np.loadtxt(
#         path,
#         comments="#",
#     ).T

#     wl = np.asarray(
#         data[0],
#         dtype=np.float64,
#     )

#     flux = np.asarray(
#         data[1],
#         dtype=np.float64,
#     )

#     # ============================================================
#     # Optional wavelength cut
#     # ============================================================

#     if (
#         x_min is not None
#         or x_max is not None
#     ):

#         mask = np.ones_like(
#             wl,
#             dtype=bool,
#         )

#         if x_min is not None:
#             mask &= (
#                 wl
#                 >= max(
#                     float(x_min) - 50.0,
#                     float(wl.min()),
#                 )
#             )

#         if x_max is not None:
#             mask &= (
#                 wl
#                 <= min(
#                     float(x_max) + 50.0,
#                     float(wl.max()),
#                 )
#             )

#         if not np.any(mask):
#             raise ValueError(
#                 f"No wavelength values left after "
#                 f"applying x_min/x_max cut for {name}"
#             )

#         wl = wl[mask]

#         flux = flux[mask]

#     # ============================================================
#     # Validate wavelength grid
#     # ============================================================

#     if wl.size < 3:
#         raise ValueError(
#             "Template too short after cutting; "
#             "need at least 3 points."
#         )

#     dl = float(
#         wl[1] - wl[0]
#     )

#     # ============================================================
#     # Normalize template
#     # ============================================================

#     unit_flux = (
#         flux
#         / np.clip(
#             np.sum(flux),
#             1e-10,
#             np.inf,
#         )
#     ).astype(
#         np.float32
#     )

#     # ============================================================
#     # Intrinsic template dispersion
#     # ============================================================

#     if user_fd is None:

#         fixed_dispersion = (
#             dl
#             / central_wl
#             * DEFAULT_C_KMS
#         )

#     else:

#         fixed_dispersion = float(
#             user_fd
#         )

#     # ============================================================
#     # Parameters
#     #
#     # ALL remain nonlinear.
#     # ============================================================

#     param_names = [
#         "logamp",
#         "logFWHM",
#         "vshift_kms",
#     ]

#     # ============================================================
#     # Fixed arrays / constants
#     # ============================================================

#     wl_jax = jnp.asarray(
#         wl,
#         dtype=jnp.float32,
#     )

#     c_kms = jnp.asarray(
#         DEFAULT_C_KMS,
#         dtype=jnp.float32,
#     )

#     sigma_template = jnp.asarray(
#         sigmatemplate,
#         dtype=jnp.float32,
#     )

#     dispersion = jnp.asarray(
#         fixed_dispersion,
#         dtype=jnp.float32,
#     )

#     ten = jnp.asarray(
#         10.0,
#         dtype=jnp.float32,
#     )

#     two = jnp.asarray(
#         2.0,
#         dtype=jnp.float32,
#     )

#     two355 = jnp.asarray(
#         2.355,
#         dtype=jnp.float32,
#     )

#     tiny = jnp.asarray(
#         1e-12,
#         dtype=jnp.float32,
#     )

#     # ============================================================
#     # PRECOMPUTE intrinsic template rFFT
#     #
#     # This is fixed because unit_flux never changes.
#     # ============================================================

#     n_pix = unit_flux.shape[0]

#     template_rfft_np = np.fft.rfft(
#         unit_flux
#     ).astype(
#         np.complex64
#     )

#     template_rfft = jnp.asarray(
#         template_rfft_np,
#         dtype=jnp.complex64,
#     )

#     # Frequencies corresponding to the rFFT
#     freq = jnp.asarray(
#         np.fft.rfftfreq(
#             n_pix,
#             d=1.0,
#         ),
#         dtype=jnp.float32,
#     )

#     # ============================================================
#     # Shared nonlinear template shape
#     # ============================================================

#     def build_profile(
#         x,
#         logFWHM,
#         vshift_kms,
#     ):
#         """
#         Generate broadened and shifted unit-amplitude template.

#         Depends only on nonlinear shape parameters:
#             logFWHM
#             vshift_kms
#         """

#         x = jnp.asarray(
#             x,
#             dtype=jnp.float32,
#         )

#         # --------------------------------------------------------
#         # Fitted FWHM
#         # --------------------------------------------------------

#         FWHM = jnp.power(
#             ten,
#             logFWHM,
#         )

#         sigma_model = (
#             FWHM
#             / two355
#         )

#         # --------------------------------------------------------
#         # Remove intrinsic template resolution
#         # --------------------------------------------------------

#         diff_sq = (
#             sigma_model**2
#             - sigma_template**2
#         )

#         diff_sq_safe = (
#             jax.nn.softplus(
#                 diff_sq / ten
#             )
#             * ten
#             + tiny
#         )

#         delta_sigma = jnp.sqrt(
#             diff_sq_safe
#         )

#         # --------------------------------------------------------
#         # Broadening in template pixels
#         # --------------------------------------------------------

#         sigma_pix = (
#             delta_sigma
#             / dispersion
#         )

#         # --------------------------------------------------------
#         # Current Gaussian transfer function
#         #
#         # THIS changes with logFWHM.
#         # --------------------------------------------------------

#         gauss_tf = jnp.exp(
#             -two
#             * (
#                 jnp.pi
#                 * freq
#                 * sigma_pix
#             ) ** 2
#         )

#         # --------------------------------------------------------
#         # Apply broadening to precomputed template FFT
#         # --------------------------------------------------------

#         broadened = jnp.fft.irfft(
#             template_rfft
#             * gauss_tf,
#             n=n_pix,
#         ).astype(
#             jnp.float32
#         )

#         # --------------------------------------------------------
#         # Current fitted velocity shift
#         # --------------------------------------------------------

#         beta = (
#             vshift_kms
#             / c_kms
#         )

#         xp = (
#             wl_jax
#             * (
#                 jnp.float32(1.0)
#                 + beta
#             )
#         )

#         # --------------------------------------------------------
#         # Interpolate onto requested wavelength grid
#         # --------------------------------------------------------

#         profile = jnp.interp(
#             x,
#             xp,
#             broadened,
#             left=jnp.float32(0.0),
#             right=jnp.float32(0.0),
#         )

#         return profile.astype(
#             jnp.float32
#         )

#     # ============================================================
#     # Normal model
#     # ============================================================

#     @with_param_names(param_names,profile_name=name,)
#     def model(
#         x: jnp.ndarray,
#         params: jnp.ndarray,
#     ) -> jnp.ndarray:

#         params = jnp.asarray(
#             params,
#             dtype=jnp.float32,
#         )

#         logamp = params[0]
#         logFWHM = params[1]
#         vshift_kms = params[2]

#         # --------------------------------------------------------
#         # Nonlinear amplitude
#         # --------------------------------------------------------

#         amplitude = jnp.power(
#             ten,
#             logamp,
#         )

#         # --------------------------------------------------------
#         # Nonlinear shape
#         # --------------------------------------------------------

#         profile = build_profile(
#             x,
#             logFWHM,
#             vshift_kms,
#         )

#         return (
#             amplitude
#             * profile
#         ).astype(
#             jnp.float32
#         )

#     # ============================================================
#     # Design builder
#     #
#     # There are NO linear parameters in this profile.
#     # Therefore the entire model is the offset.
#     # ============================================================

#     def design_builder(
#         x: jnp.ndarray,
#         params: jnp.ndarray,
#     ):

#         x = jnp.asarray(
#             x,
#             dtype=jnp.float32,
#         )

#         params = jnp.asarray(
#             params,
#             dtype=jnp.float32,
#         )

#         logamp = params[0]
#         logFWHM = params[1]
#         vshift_kms = params[2]

#         amplitude = jnp.power(
#             ten,
#             logamp,
#         )

#         profile = build_profile(
#             x,
#             logFWHM,
#             vshift_kms,
#         )

#         # --------------------------------------------------------
#         # Complete nonlinear contribution
#         # --------------------------------------------------------

#         offset = (
#             amplitude
#             * profile
#         ).astype(
#             jnp.float32
#         )

#         # --------------------------------------------------------
#         # Zero linear columns
#         #
#         # Shape:
#         #     (n_data_pixels, 0)
#         # --------------------------------------------------------

#         columns = jnp.empty(
#             (
#                 x.shape[0],
#                 0,
#             ),
#             dtype=jnp.float32,
#         )

#         return (
#             offset,
#             columns,
#         )

#     # ============================================================
#     # Attach builder
#     # ============================================================

#     model.design_builder = (
#         design_builder
#     )

#     # ============================================================
#     # Return
#     # ============================================================

#     return {
#         "model": model,

#         "design_builder": design_builder,

#         "template_info": {
#             "name": name,

#             "file": str(path),

#             "central_wl": float(
#                 central_wl
#             ),

#             "sigmatemplate": float(
#                 sigmatemplate
#             ),

#             "fixed_dispersion": float(
#                 fixed_dispersion
#             ),

#             "x_min": (
#                 None
#                 if x_min is None
#                 else float(x_min)
#             ),

#             "x_max": (
#                 None
#                 if x_max is None
#                 else float(x_max)
#             ),

#             "dl": dl,
#         },
#     }

# def make_template_function_(
#     name: str,
#     x_min: Optional[float] = None,  # Angstroms (linear)
#     x_max: Optional[float] = None,
# ) -> Dict[str, Any]:
#     """
#     Factory for a FeII template model by name, with optional wavelength cuts.

#     Looks up path, central_wl, sigmatemplate, and optional fixed_dispersion in TEMPLATES.

#     If x_min/x_max are provided, the template spectrum is cut to [x_min, x_max]
#     with a ±50 Å guard band to reduce boundary artifacts in the FFT broadening, and
#     then re-normalized to unit sum.

#     Notes
#     -----
#     The third parameter is **vshift_kms** (velocity shift in km/s), applied as a
#     multiplicative stretch of the wavelength grid:
#         wl_shifted = wl * (1 + vshift_kms / DEFAULT_C_KMS)

#     Returns
#     -------
#     dict
#         {
#           'model': Callable(x, params) -> flux,  # has .param_names, .n_params
#           'template_info': {
#               'name', 'file', 'central_wl', 'sigmatemplate',
#               'fixed_dispersion', 'x_min', 'x_max', 'dl'
#           }
#         }
#     """
#     cfg = TEMPLATES.get(name)
#     if cfg is None:
#         raise KeyError(f"No such template: {name}")

#     path          = cfg["file"]
#     central_wl    = cfg["central_wl"]
#     sigmatemplate = cfg["sigmatemplate"]
#     user_fd       = cfg.get("fixed_dispersion", None)

#     data = np.loadtxt(path, comments="#").T
#     wl   = np.array(data[0], dtype=np.float64)
#     flux = np.array(data[1], dtype=np.float64)

#     # Optional wavelength cut with ±50 Å margin
#     if x_min is not None or x_max is not None:
#         mask = np.ones_like(wl, dtype=bool)
#         if x_min is not None:
#             mask &= wl >= max(x_min - 50.0, wl.min())
#         if x_max is not None:
#             mask &= wl <= min(x_max + 50.0, wl.max())
#         if not np.any(mask):
#             raise ValueError(f"No wavelength values left after applying x_min/x_max cut for {name}")
#         wl   = wl[mask]
#         flux = flux[mask]

#     # Ensure equally spaced grid
#     if wl.size < 3:
#         raise ValueError("Template too short after cutting; need at least 3 points.")
#     dl = float(wl[1] - wl[0])

#     # Re-normalize to unit sum AFTER any cut
#     unit_flux = flux / np.clip(np.sum(flux), 1e-10, np.inf)

#     if user_fd is None:
#         fixed_dispersion = (dl / central_wl) * DEFAULT_C_KMS
#     else:
#         fixed_dispersion = float(user_fd)

#     param_names = ["logamp", "logFWHM", "vshift_kms"]
   
#     # Pre-pack constants as JAX arrays once
#     wl_jax = jnp.asarray(wl, dtype=np.float32)
#     unit_flux_jax = jnp.asarray(unit_flux, dtype=np.float32)

#     @with_param_names(param_names)
#     def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
#         # Enforce single precision for all model calculations
#         x = jnp.asarray(x, dtype=jnp.float32)
#         params = jnp.asarray(params, dtype=jnp.float32)

#         logamp, logFWHM, vshift_kms = params

#         c_kms = jnp.asarray(DEFAULT_C_KMS, dtype=jnp.float32)
#         sigma_template = jnp.asarray(sigmatemplate, dtype=jnp.float32)
#         dispersion = jnp.asarray(fixed_dispersion, dtype=jnp.float32)

#         amp = jnp.power(jnp.float32(10.0), logamp)
#         FWHM = jnp.power(jnp.float32(10.0), logFWHM)
#         sigma_model = FWHM / jnp.float32(2.355)

#         # Quadratic subtraction
#         diff_sq = sigma_model**2 - sigma_template**2
#         diff_sq_safe = (
#             jax.nn.softplus(diff_sq / jnp.float32(10.0))
#             * jnp.float32(10.0)
#             + jnp.float32(1e-12)
#         )
#         delta_sigma = jnp.sqrt(diff_sq_safe)

#         # Broadening in pixels
#         sigma_pix = delta_sigma / dispersion

#         n_pix = unit_flux_jax.shape[0]
#         freq = jnp.fft.fftfreq(n_pix, d=1.0).astype(jnp.float32)

#         gauss_tf = jnp.exp(jnp.float32(-2.0)* (jnp.pi * freq * sigma_pix) ** 2)

#         # FFT of float32 input is complex64; preserve the imaginary component
#         spec_fft = jnp.fft.fft(unit_flux_jax).astype(jnp.complex64)
#         broadened = jnp.fft.ifft(
#             spec_fft * gauss_tf.astype(jnp.complex64)
#         ).real.astype(jnp.float32)

#         # Positive velocity means redder features
#         beta = vshift_kms / c_kms
#         xp = wl_jax * (jnp.float32(1.0) + beta)

#         interpolated = jnp.interp(
#             x,
#             xp,
#             broadened,
#             left=jnp.float32(0.0),
#             right=jnp.float32(0.0),
#         )

#         return (amp * interpolated).astype(jnp.float32)

#     return {
#         "model": model,
#         "template_info": {
#             "name": name,
#             "file": str(path),
#             "central_wl": float(central_wl),
#             "sigmatemplate": float(sigmatemplate),
#             "fixed_dispersion": float(fixed_dispersion),
#             "x_min": None if x_min is None else float(x_min),
#             "x_max": None if x_max is None else float(x_max),
#             "dl": dl,
#         },
#     }


# def make_host_function_old(
#     filename: str | Path = TEMPLATES_PATH / "miles_cube_log.npz",
#     z_include: Optional[Union[tuple[float, float], list[float]]] = [-0.7, 0.22],
#     age_include: Optional[Union[tuple[float, float], list[float]]] = [0.1, 10.0],
#     xmin: Optional[float] = None,
#     xmax: Optional[float] = None,
#     verbose: Optional[bool] = None,
#     **kwargs,
# ) -> dict:
#     r"""
#     Factory for host-galaxy SSP models stored on a log-sampled wavelength grid.

#     The returned model builds a weighted sum of SSP templates, then applies a single
#     FFT-based Gaussian broadening in **velocity space** (LOSVD-like), and finally
#     interpolates to the user grid ``x``.

#     Parameters
#     ----------
#     filename
#         Either ``"miles"``, ``"xsl"``, a path to a ``*_cube_log.npz`` file, or ``None``
#         (defaults to MILES).
#     z_include
#         Metallicity selection; either (min,max) or list/array.
#     age_include
#         Age selection; either (min,max) or list/array.
#     xmin, xmax
#         Optional wavelength window in Angstrom. A ±50 Å guard band is applied.
#     verbose
#         Print selected grid size.

#     Returns
#     -------
#     dict
#         ``{"model": callable, "host_info": dict}``

#     Notes
#     -----
#     - **Important:** The template wavelength axis in the NPZ file must be sampled
#       approximately uniformly in :math:`\ln(\lambda)` (i.e. *log-spaced in wavelength*).
#       The convolution kernel is constructed in :math:`\ln(\lambda)`, so the model
#       assumes a constant velocity step per pixel.
#     - The input ``x`` passed to ``model(x, params)`` is in **Angstrom**, but should be
#       sampled on a grid that is also approximately uniform in :math:`\ln(\lambda)` if you
#       want the broadening to behave exactly like a Gaussian LOSVD on your evaluation grid.
#       (Interpolation to arbitrary ``x`` is allowed, but the kernel is defined on the
#       template log-grid.)
#     - Velocity shift is applied as :math:`\lambda \rightarrow \lambda \exp(v/c)`.

#     The parameter order is:
#     ``[logamp, logFWHM, vshift_kms, weight_0, weight_1, ...]``.
#     """
#     #print("Makehost test function")
#     allowed = {"miles": TEMPLATES_PATH / "miles_cube_log.npz",
#                "xsl":   TEMPLATES_PATH / "xsl_cube_log.npz",
#                None:    TEMPLATES_PATH / "miles_cube_log.npz"}
#     if filename in allowed:
#         filename = allowed[filename]
#     else:
#         filename = Path(filename)
#         if not filename.exists():
#             raise KeyError(
#                 f"file_name '{filename}' is not available. Available: ['miles', 'xsl'] or a valid path."
#             )

#     data = np.load(filename, mmap_mode="r")

    
#     cube = np.asarray(data["cube_log"], dtype=np.float32)   # (n_Z, n_age, n_pix)
#     wave = np.asarray(data["wave_log"], dtype=np.float32)   # (n_pix,) in Angstrom, log-sampled
#     all_ages = np.asarray(data["ages_sub"], dtype=np.float32)
#     all_zs = np.asarray(data["zs_sub"], dtype=np.float32)
#     sigmatemplate = float(data["sigmatemplate"])            # km/s
#     #fixed_dispersion = float(data["fixed_dispersion"])      # km/s per pixel (on the log grid)


#     if wave.size < 4:
#         raise ValueError("Template wavelength grid too short.")
#     if not np.all(np.isfinite(wave)) or np.any(wave <= 0):
#         raise ValueError("wave_log must be finite and strictly > 0 (Angstrom).")
#     if np.any(np.diff(wave) <= 0):
#         raise ValueError("wave_log must be strictly increasing.")

#     if z_include is not None:
#         z_min, z_max = float(np.min(z_include)), float(np.max(z_include))
#         z_mask = (all_zs >= z_min) & (all_zs <= z_max)
#         if not np.any(z_mask):
#             raise ValueError(f"No metallicities in range {z_min} to {z_max}")
#         zs = all_zs[z_mask]
#         cube = cube[z_mask, :, :]
#     else:
#         zs = all_zs

#     if age_include is not None:
#         a_min, a_max = float(np.min(age_include)), float(np.max(age_include))
#         a_mask = (all_ages >= a_min) & (all_ages <= a_max)
#         if not np.any(a_mask):
#             raise ValueError(f"No ages in range {a_min} to {a_max}")
#         ages = all_ages[a_mask]
#         cube = cube[:, a_mask, :]
#     else:
#         ages = all_ages
#     grid_metadata = [(float(Z), float(age)) for Z in zs for age in ages]
#     if kwargs.get("data"):
#        print("We will use external template experimental")
#        data = kwargs["data"]
#        filename = kwargs["data"]["filename"]
#        cube =  np.asarray(data["cube_log"], dtype=np.float32)
#        wave = np.asarray(data["wave_log"], dtype=np.float32)   # (n_pix,) in Angstrom, log-sampled
#        assert wave.shape[0] == cube.shape[-1], "check the shapes of wave_log and cube_log"
#        sigmatemplate = data["sigmatemplate"]
#        n_Z, n_age = cube.shape[:-1]
#        grid_metadata = [(float(Z), float(age)) for Z in np.arange(n_Z) for age in  np.arange(n_age)]
#     f = 1.0  # keep if you later want external scaling
    
       
#     if xmin is not None or xmax is not None:
#         mask = np.ones_like(wave, dtype=bool)
#         if xmin is not None:
#             mask &= wave >= max(float(xmin) * f - 50.0, float(wave.min()))
#         if xmax is not None:
#             mask &= wave <= min(float(xmax) * f + 50.0, float(wave.max()))
#         if not np.any(mask):
#             raise ValueError("No wavelength values left after applying xmin/xmax.")
#         wave = wave[mask].astype(np.float32, copy=False)
#         cube = cube[:, :, mask].astype(np.float32, copy=False)

#     n_Z, n_age, n_pix = cube.shape
#     if verbose:
#         print(f"Host added with n_Z={n_Z}, n_age={n_age}, n_pix={n_pix}")
    
#     eps = 1e-30
#     flux_int = np.nansum(cube, axis=-1, keepdims=True)
#     cube = cube / (flux_int + eps)

#     templates_flat = cube.reshape(-1, n_pix)  # (n_Z*n_age, n_pix)
#     param_names = ["logamp", "logFWHM", "vshift_kms"]
#     linear_params = []
#     for Z, age in grid_metadata:
#         zstr = str(Z).replace(".", "p")
#         astr = str(age).replace(".", "p")
#         param_names.append(f"weight_Z{zstr}_age{astr}")
#         linear_params.append(f"weight_Z{zstr}_age{astr}")
#     templates_jax = jnp.asarray(templates_flat)  # (Ntemp, n_pix) float32
#     wave_jax = jnp.asarray(wave)                 # (n_pix,) float32

#     # build convolution kernel in ln(lambda) (LOSVD-correct)
#     # dln is ~constant if wave is log-sampled in Angstrom
#     dln = float(np.mean(np.gradient(np.log(wave.astype(np.float64)))))
#     freq = jnp.fft.fftfreq(n_pix, d=dln).astype(jnp.float32)
#     @with_param_names(param_names,linear_param_names=linear_params,profile_name="host")
#     def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
#         logamp = params[0]
#         logFWHM = params[1]
#         vshift_kms = params[2]
#         weights = params[3:]  # (Ntemp,)

#         amplitude = 10.0 ** logamp

#         base = jnp.tensordot(weights, templates_jax, axes=(0, 0))  # (n_pix,)

#         FWHM = 10.0 ** logFWHM
#         sigma_model = FWHM / 2.355  # km/s

#         diff_sq = sigma_model**2 - sigmatemplate**2
#         diff_sq_safe = jax.nn.softplus(diff_sq / 10.0) * 10.0 + 1e-12
#         delta_sigma = jnp.sqrt(diff_sq_safe)  # km/s

#         sigma_y = (delta_sigma / DEFAULT_C_KMS).astype(jnp.float32)

#         gauss_tf = jnp.exp(-2.0 * (jnp.pi * freq * sigma_y) ** 2)
#         conv = jnp.real(jnp.fft.ifft(jnp.fft.fft(base) * gauss_tf))

#         xp = wave_jax * jnp.exp(vshift_kms / DEFAULT_C_KMS)

#         return amplitude * jnp.interp(x * f, xp, conv, left=0.0, right=0.0)

#     def design_builder(x: jnp.ndarray,params: jnp.ndarray,):
#         """
#         Build the profiled linear representation of the host model.

#         Parameters
#         ----------
#         x : (n_pix_data,)
#         params :
#             [logamp, logFWHM, vshift_kms, weight_0, weight_1, ...]

#         Returns
#         -------
#         offset : (n_pix_data,)
#             Zero for this model because all host flux is carried
#             by the linear SSP weights.

#         columns : (n_pix_data, n_templates)
#             One broadened/shifted SSP template per linear weight.
#         """

#         x = jnp.asarray(x, dtype=jnp.float32)
#         params = jnp.asarray(params, dtype=jnp.float32)

#         logamp = params[0]
#         logFWHM = params[1]
#         vshift_kms = params[2]

#         # IMPORTANT:
#         # params[3:] are linear coefficients and are NOT used here.

#         amplitude = jnp.power(jnp.float32(10.0),logamp,)

#         FWHM = jnp.power(jnp.float32(10.0),logFWHM,)

#         sigma_model = FWHM / jnp.float32(2.355)

#         diff_sq = (sigma_model**2 - jnp.float32(sigmatemplate)**2)

#         diff_sq_safe = (jax.nn.softplus(diff_sq / jnp.float32(10.0)) * jnp.float32(10.0) + jnp.float32(1e-12))

#         delta_sigma = jnp.sqrt(diff_sq_safe)

#         sigma_y = (delta_sigma / jnp.float32(DEFAULT_C_KMS))

#         gauss_tf = jnp.exp(-jnp.float32(2.0)* (jnp.pi* freq* sigma_y) ** 2)

#         # --------------------------------------------------
#         # Broaden ALL templates simultaneously
#         #
#         # templates_jax:
#         #     (n_templates, n_template_pixels)
#         # --------------------------------------------------

#         templates_fft = jnp.fft.fft(templates_jax,axis=-1,)

#         broadened_templates = jnp.fft.ifft(templates_fft* gauss_tf[None, :],axis=-1,).real.astype(jnp.float32)

#         # --------------------------------------------------
#         # Common velocity shift
#         # --------------------------------------------------

#         xp = (wave_jax* jnp.exp(vshift_kms/ jnp.float32(DEFAULT_C_KMS)))

#         # --------------------------------------------------
#         # Interpolate every template to x
#         # --------------------------------------------------

#         def interpolate_one(template):
#             return jnp.interp(
#                 x * jnp.float32(f),
#                 xp,
#                 template,
#                 left=jnp.float32(0.0),
#                 right=jnp.float32(0.0),
#             )

#         columns = jax.vmap(interpolate_one)(broadened_templates)

#         # currently:
#         # (n_templates, n_data_pixels)
#         #
#         # design matrix wants:
#         # (n_data_pixels, n_templates)

#         columns = jnp.swapaxes(columns,0,1,)

#         columns = (amplitude* columns)

#         offset = jnp.zeros_like(x,dtype=jnp.float32,)

#         return offset, columns

#     model.design_builder = design_builder

#     return {
#         "model": model,
#         "host_info": {
#             "z_include": zs,
#             "age_include": ages,
#             "n_Z": n_Z,
#             "n_age": n_age,
#             "xmin": xmin,
#             "xmax": xmax,
#             "file_name": str(filename),
#             "dln": dln,
#             "dv_pix_kms": float(DEFAULT_C_KMS * dln),
#             "sigmatemplate": sigmatemplate,
#             "data":kwargs.get("data",None)
#         },
#     }

# # #xsl_cube_log_
# def make_host_function_classic(
#     filename: str = TEMPLATES_PATH / "miles_cube_log.npz",
#     #filename: str = TEMPLATES_PATH / "xsl_cube_log.npz",
#     z_include: Optional[Union[tuple[float, float], list[float]]] = [-0.7, 0.22],
#     age_include: Optional[Union[tuple[float, float], list[float]]] = [0.1, 10.0],
#     xmin: Optional[float] = None, 
#     xmax: Optional[float] = None,
#     verbose: Optional[bool] = None,
#     **kwargs,
# ) -> dict:
#     """
#     Memory-lean host model:
#       - sums weighted templates first, then does a single FFT-based convolution
#       - np.load(..., mmap_mode='r') to reduce RAM pressure
#       - keeps arrays in float32

#     Parameters
#     ----------
#     The third parameter is vshift_kms: a velocity shift in km/s.
#     """
#     #f = 1.0
#     #print(filename)
#     #z_source = 2.16
#     #z_lens = 0.905
#     #f = (1 + z_source) / (1 + z_lens)
#     if filename not in ["miles","xsl",TEMPLATES_PATH / "miles_cube_log.npz",TEMPLATES_PATH / "xsl_cube_log.npz",None]:
#         raise KeyError(
#             f"file_name '{filename}' is not available file_name."
#             f"Available parameters: {['miles','xsl']}")
#     f = 1.
#     filename = {"miles":TEMPLATES_PATH / "miles_cube_log.npz","xsl":TEMPLATES_PATH / "xsl_cube_log.npz"}.get(filename,TEMPLATES_PATH / "miles_cube_log.npz")
#     #print(filename)
#     data = np.load(filename, mmap_mode="r")

#     cube = np.asarray(data["cube_log"], dtype=np.float32)   # (n_Z, n_age, n_pix)
#     wave = np.asarray(data["wave_log"], dtype=np.float32)
#     all_ages = np.asarray(data["ages_sub"], dtype=np.float32)
#     all_zs = np.asarray(data["zs_sub"], dtype=np.float32)
#     sigmatemplate = float(data["sigmatemplate"])
#     fixed_dispersion = float(data["fixed_dispersion"])

#     if z_include is not None:
#         z_min, z_max = np.min(z_include), np.max(z_include)
#         z_mask = (all_zs >= z_min) & (all_zs <= z_max)
#         if not np.any(z_mask):
#             raise ValueError(f"No metallicities in range {z_min} to {z_max}")
#         zs = all_zs[z_mask]
#         cube = cube[z_mask, :, :]
#     else:
#         zs = all_zs

#     if age_include is not None:
#         a_min, a_max = np.min(age_include), np.max(age_include)
#         a_mask = (all_ages >= a_min) & (all_ages <= a_max)
#         if not np.any(a_mask):
#             raise ValueError(f"No ages in range {a_min} to {a_max}")
#         ages = all_ages[a_mask]
#         cube = cube[:, a_mask, :]
#     else:
#         ages = all_ages

#     if xmin is not None or xmax is not None:
#         mask = np.ones_like(wave, dtype=bool)
#         if xmin is not None:
#             mask &= wave >= max([xmin * f - 50.0, float(wave.min())])
#         if xmax is not None:
#             mask &= wave <= min([xmax * f + 50.0, float(wave.max())])
#         if not np.any(mask):
#             raise ValueError("No wavelength values left after applying x_min/x_max cut.")
#         wave = wave[mask].astype(np.float32, copy=False)
#         cube = cube[:, :, mask].astype(np.float32, copy=False)
#     #print(wave)
#     dx = float(wave[1] - wave[0])
#     n_Z, n_age, n_pix = cube.shape
#     if verbose:
#         print(f"Host added with n_Z: {n_Z} and n_age: {n_age}")
    
#     eps = 1e-30
#     flux_int = np.nansum(cube, axis=-1, keepdims=True)  
#     cube = cube / (flux_int + eps)
    
#     templates_flat = cube.reshape(-1, n_pix)                # numpy array
#     grid_metadata = [(float(Z), float(age)) for Z in zs for age in ages]

   
#     param_names = ["logamp", "logFWHM", "vshift_kms"]
#     linear_params = []
#     for Z, age in grid_metadata:
#         zstr = str(Z).replace(".", "p")
#         astr = str(age).replace(".", "p")
#         param_names.append(f"weight_Z{zstr}_age{astr}")
#         linear_params.append(f"weight_Z{zstr}_age{astr}")
#     templates_jax = jnp.asarray(templates_flat)             # (N, P) float32
#     wave_jax = jnp.asarray(wave)
#     #print(wave_jax)
#     freq = jnp.fft.fftfreq(n_pix, d=dx).astype(jnp.float32) # (P,)

#     print(linear_params)
#     @with_param_names(param_names,linear_param_names=linear_params)
#     def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
#         logamp = params[0]
#         amplitude = 10.0 ** logamp

#         logFWHM = params[1]
#         vshift_kms = params[2]
#         weights = params[3:]                                 # (N,)

#         base = jnp.tensordot(weights, templates_jax, axes=(0, 0))  # (P,)

#         # --- broadening ---
#         FWHM = 10.0 ** logFWHM                  # total FWHM in km/s
#         sigma_model = FWHM / 2.355              # km/s

#         diff_sq = sigma_model**2 - sigmatemplate**2
#         diff_sq_safe = jax.nn.softplus(diff_sq / 10.0) * 10.0 + 1e-12
#         delta_sigma = jnp.sqrt(diff_sq_safe)    # km/s

#         sigma_pix = delta_sigma / fixed_dispersion
#         sigma_lambda = sigma_pix * dx

#         gauss_tf = jnp.exp(-2.0 * (jnp.pi * freq * sigma_lambda) ** 2)
#         base_fft = jnp.fft.fft(base)
#         conv = jnp.real(jnp.fft.ifft(base_fft * gauss_tf))


#         beta = vshift_kms / DEFAULT_C_KMS
#         xp = wave_jax * (1.0 + beta)  

#         return amplitude * jnp.interp(x*f, xp, conv, left=0.0, right=0.0)

#     return {
#         "model": model,
#         "host_info": {
#             "z_include": zs,
#             "age_include": ages,
#             "n_Z": n_Z,
#             "n_age": n_age,
#             "xmin": xmin,
#             "xmax": xmax,
#             "file_name":filename
#         },
#     }





# # """Improved SHEAP host SSP factory.

# # Drop ``make_host_function`` into ``sheap/Profiles/profiles_templates.py``
# # or import this function directly. Maintains SHEAP's model + design_builder API.

# # Key changes:
# #   * optional velocity-aware cropping (disabled by default, for safety);
# #   * zero-padded, cached FFTs for each host design column;
# #   * a short *post-convolution* edge taper to make zero-outside interpolation
# #     continuous in velocity even at a genuinely unsupported library edge;
# #   * JAX and optional NumPy/SciPy float64 implementations sharing selected SSPs;
# #   * metadata describing the guaranteed interior wavelength coverage.

# # A taper is a numerical boundary treatment, NOT physical extrapolation.
# # For scientific fitting, mask outside host_info['safe_wave_min/max'] if the
# # native SSP range does not cover the entire fitted spectrum.
# # """

# # from pathlib import Path
# # import warnings
# # import numpy as np
# # import jax
# # import jax.numpy as jnp
# # from scipy.fft import rfft, irfft, rfftfreq, next_fast_len
# # from sheap.Profiles.Utils import with_param_names
# # from sheap.Utils.Constants import DEFAULT_C_KMS
# # from sheap.Profiles.profiles_templates import TEMPLATES_PATH


# def make_host_functionv2(
#     filename: str | Path | None = "miles",
#     z_include=(-0.7, 0.22),
#     age_include=(0.1, 10.0),
#     xmin=None,
#     xmax=None,
#     verbose=False,
#     add_numpy_backend=True,
#     *,
#     crop_templates=False,
#     velocity_bounds=(-500.0, 500.0),
#     max_logfwhm=3.0,
#     n_sigma_margin=5.0,
#     edge_taper_pixels=8,
#     min_fft_pad_pixels=8,
#     **kwargs,
# ):
#     """Return a compatible JAX host model, plus optional NumPy float64 backend.

#     Parameters
#     ----------
#     crop_templates : bool, default False
#         False: keep the *full native SSP wavelength grid*, ignoring xmin/xmax
#         for the SSP crop. This is safest if ``x`` later has wider coverage
#         than the ``xmin/xmax`` passed by SHEAP.
#         True: crop using a velocity/FWHM-dependent margin around xmin/xmax.
#         For this mode xmin/xmax MUST cover all pixels used by the host fit.
#     velocity_bounds : tuple of km/s
#         Host velocity shift limits used for the crop and safe-region estimate.
#     max_logfwhm : float
#         Upper log10(FWHM/km/s) permitted for the stellar host.
#     n_sigma_margin : float
#         Convolution support reserved on either side (5 sigma by default).
#     edge_taper_pixels : int
#         Smoothly taper the *broadened output* to zero over the first/last
#         few SSP pixels. This prevents a discrete jump when a shifted pixel
#         crosses the end of the available library. Set 0 to disable.
#         Tapering is not physical extrapolation.
#     min_fft_pad_pixels : int
#         Minimum left/right padding for FFT-based (non-circular) broadening.

#     Parameter order:
#         [logamp, logFWHM, vshift_kms, weight_0, ..., weight_(N-1)]

#     Design builders return:
#         (offset (n_data,), columns (n_data, n_templates))
#     """
#     c64 = float(DEFAULT_C_KMS)
#     external = kwargs.get("data", None)
#     vmin, vmax = map(float, velocity_bounds)
#     if vmin > vmax:
#         raise ValueError("velocity_bounds must be (min_velocity, max_velocity)")
#     if n_sigma_margin <= 0:
#         raise ValueError("n_sigma_margin must be positive")

#     if external is not None:
#         cube = np.asarray(external["cube_log"], dtype=np.float64)
#         wave = np.asarray(external["wave_log"], dtype=np.float64)
#         sigmatemplate = float(external["sigmatemplate"])
#         if cube.ndim != 3:
#             raise ValueError("External cube_log must be (n_Z,n_age,n_pix)")
#         n_Z, n_age = cube.shape[:2]
#         zs = np.asarray(external.get("zs_sub", np.arange(n_Z)), dtype=float)
#         ages = np.asarray(external.get("ages_sub", np.arange(n_age)), dtype=float)
#         source = str(external.get("filename", "external"))
#         # External data is assumed already selected in age and metallicity.
#     else:
#         if filename is None or filename == "miles":
#             filename = TEMPLATES_PATH / "miles_cube_log.npz"
#         elif filename == "xsl":
#             filename = TEMPLATES_PATH / "xsl_cube_log.npz"
#         filename = Path(filename)
#         if not filename.exists():
#             raise FileNotFoundError(f"Host template not found: {filename}")
#         source = str(filename)
#         with np.load(filename, mmap_mode="r") as data:
#             cube = np.asarray(data["cube_log"], dtype=np.float64)
#             wave = np.asarray(data["wave_log"], dtype=np.float64)
#             zs = np.asarray(data["zs_sub"], dtype=np.float64)
#             ages = np.asarray(data["ages_sub"], dtype=np.float64)
#             sigmatemplate = float(data["sigmatemplate"])

#         if z_include is not None:
#             zlo, zhi = float(np.min(z_include)), float(np.max(z_include))
#             mz = (zs >= zlo) & (zs <= zhi)
#             if not mz.any():
#                 raise ValueError("No metallicities remain after z_include")
#             cube, zs = cube[mz], zs[mz]
#         if age_include is not None:
#             alo, ahi = float(np.min(age_include)), float(np.max(age_include))
#             ma = (ages >= alo) & (ages <= ahi)
#             if not ma.any():
#                 raise ValueError("No ages remain after age_include")
#             cube, ages = cube[:, ma], ages[ma]

#     if (cube.ndim != 3 or wave.ndim != 1 or cube.shape[-1] != wave.size
#             or cube.shape[0] != len(zs) or cube.shape[1] != len(ages)):
#         raise ValueError("Inconsistent cube_log/wave_log/zs_sub/ages_sub shapes")
#     if (wave.size < 4 or not np.all(np.isfinite(wave))
#             or np.any(wave <= 0) or np.any(np.diff(wave) <= 0)):
#         raise ValueError("wave_log must be strictly increasing, positive, finite")

#     native_wave_min, native_wave_max = float(wave[0]), float(wave[-1])
#     # A crop must reserve space for both the Gaussian kernel and the
#     # final edge taper, otherwise the taper can intrude into the fit.
#     native_dln = float(np.mean(np.diff(np.log(wave))))
#     native_dv_pix = c64 * native_dln
#     taper_extra_kms = max(int(edge_taper_pixels), 0) * native_dv_pix
#     sigma_max = 10.0**float(max_logfwhm) / 2.355
#     # The resolution-matching kernel gets smaller if the SSP resolution
#     # is already broader than the target model's total sigma.
#     max_kernel_sigma = np.sqrt(max(sigma_max**2 - sigmatemplate**2, 0.0))
#     kernel_margin_kms = float(n_sigma_margin) * max_kernel_sigma
#     velocity_margin_kms = kernel_margin_kms + taper_extra_kms

#     requested_min = None if xmin is None else float(xmin)
#     requested_max = None if xmax is None else float(xmax)
#     if requested_min is not None and requested_max is not None and requested_min >= requested_max:
#         raise ValueError("xmin must be smaller than xmax")

#     if crop_templates and (requested_min is not None or requested_max is not None):
#         # Rest wavelengths needed to model *every* trial in the allowed
#         # stellar velocity and broadened-width ranges.
#         required_min = (native_wave_min if requested_min is None
#                         else requested_min * np.exp(-(vmax + velocity_margin_kms) / c64))
#         required_max = (native_wave_max if requested_max is None
#                         else requested_max * np.exp(-(vmin - velocity_margin_kms) / c64))
#         first = max(0, np.searchsorted(wave, required_min, side="left") - 2)
#         stop = min(wave.size, np.searchsorted(wave, required_max, side="right") + 2)
#         if stop - first < 4:
#             raise ValueError("No sufficient host wavelength coverage after crop")
#         wave = wave[first:stop]
#         cube = cube[:, :, first:stop]
#     else:
#         required_min = native_wave_min
#         required_max = native_wave_max

#     n_Z, n_age, n_pix = cube.shape
#     n_templates = n_Z * n_age
#     wave = np.ascontiguousarray(wave, dtype=np.float64)
#     dlog = np.diff(np.log(wave))
#     dln = float(np.mean(dlog))
#     if np.max(np.abs(dlog - dln)) > 1e-3 * dln:
#         raise ValueError("SSP wave_log grid must be nearly uniform in ln(wavelength)")
#     dv_pix = c64 * dln

#     # This matches the user's existing normalization convention. Changing
#     # the cropped wavelength region changes template scaling: refit weights.
#     totals = np.nansum(cube, axis=-1, keepdims=True)
#     if not np.all(np.isfinite(totals)):
#         raise ValueError("Nonfinite template normalization")
#     cube = cube / (totals + 1e-30)
#     templates64 = np.ascontiguousarray(cube.reshape(n_templates, n_pix), dtype=np.float64)
#     if not np.all(np.isfinite(templates64)):
#         raise ValueError("Nonfinite values in normalized stellar templates")
#     templates32 = np.ascontiguousarray(templates64.astype(np.float32))

#     # Non-circular convolution up to n_sigma_margin kernel sigmas.
#     pad_pix = max(int(min_fft_pad_pixels),
#                   int(np.ceil(kernel_margin_kms / dv_pix)))
#     n_fft = int(next_fast_len(n_pix + 2 * pad_pix))
#     # np.pad adds zero flux beyond available wavelengths. It is a
#     # convolution boundary condition, not invented stellar-template data.
#     padded64 = np.pad(templates64, ((0, 0), (pad_pix, pad_pix)))
#     padded32 = np.pad(templates32, ((0, 0), (pad_pix, pad_pix)))
#     fft64 = rfft(padded64, n=n_fft, axis=-1)
#     fft32 = jnp.asarray(rfft(padded32, n=n_fft, axis=-1), dtype=jnp.complex64)
#     freq64 = rfftfreq(n_fft, d=dln)
#     freq32 = jnp.asarray(freq64, dtype=jnp.float32)
#     templates_jax = jnp.asarray(templates32)
#     wave32 = jnp.asarray(wave, dtype=jnp.float32)
#     c32 = jnp.asarray(c64, dtype=jnp.float32)
#     sigma_template32 = jnp.asarray(sigmatemplate, dtype=jnp.float32)

#     # Taper the BROADENED spectrum (after convolution), not the SSPs before
#     # convolution, so both forward and design paths remain linear/identical.
#     taper = np.ones(n_pix, dtype=np.float64)
#     k = min(max(int(edge_taper_pixels), 0), n_pix // 2)
#     if k >= 2:
#         t = np.linspace(0.0, 1.0, k)
#         roll = 0.5 - 0.5 * np.cos(np.pi * t)
#         taper[:k] = roll
#         taper[-k:] = roll[::-1]
#     elif k == 1:
#         taper[0] = 0.0
#         taper[-1] = 0.0
#     taper32 = jnp.asarray(taper, dtype=jnp.float32)

#     metadata = [(float(Z), float(age)) for Z in zs for age in ages]
#     linear_names = [
#         f"weight_Z{str(Z).replace('.', 'p')}_age{str(age).replace('.', 'p')}"
#         for Z, age in metadata
#     ]
#     param_names = ["logamp", "logFWHM", "vshift_kms"] + linear_names

#     # Precompute once, keep physical boundaries based on full selected grid.
#     safe_wave_min = float(wave[0] * np.exp((vmax + velocity_margin_kms) / c64))
#     safe_wave_max = float(wave[-1] * np.exp((vmin - velocity_margin_kms) / c64))
#     # Also exclude post-convolution taper region from physically safe fit.
#     if k:
#         safe_wave_min = max(safe_wave_min, float(wave[k - 1] * np.exp(vmax / c64)))
#         safe_wave_max = min(safe_wave_max, float(wave[-k] * np.exp(vmin / c64)))

#     if crop_templates and requested_min is not None and requested_min < safe_wave_min:
#         warnings.warn("Selected SSPs do not provide the full required blue-side margin; "
#                       "mask unsupported host pixels or disable cropping.", RuntimeWarning)
#     if crop_templates and requested_max is not None and requested_max > safe_wave_max:
#         warnings.warn("Selected SSPs do not provide the full required red-side margin; "
#                       "mask unsupported host pixels or disable cropping.", RuntimeWarning)
#     if (not crop_templates and requested_min is not None and requested_min < safe_wave_min):
#         warnings.warn("Native SSP library lacks safe blue-side coverage of xmin.", RuntimeWarning)
#     if (not crop_templates and requested_max is not None and requested_max > safe_wave_max):
#         warnings.warn("Native SSP library lacks safe red-side coverage of xmax.", RuntimeWarning)

#     def jax_transfer(logfwhm):
#         fwhm = jnp.power(jnp.float32(10.0), logfwhm)
#         sigma_model = fwhm / jnp.float32(2.355)
#         diff_sq = sigma_model**2 - sigma_template32**2
#         delta_sigma = jnp.sqrt(
#             jax.nn.softplus(diff_sq / jnp.float32(10.0)) * jnp.float32(10.0)
#             + jnp.float32(1e-12)
#         )
#         sigma_y = delta_sigma / c32
#         return jnp.exp(-jnp.float32(2.0) * (jnp.pi * freq32 * sigma_y)**2)

#     def numpy_transfer(logfwhm):
#         sigma_model = 10.0**logfwhm / 2.355
#         diff_sq = sigma_model**2 - sigmatemplate**2
#         delta_sigma = np.sqrt(10.0 * np.logaddexp(0.0, diff_sq / 10.0) + 1e-12)
#         sigma_y = delta_sigma / c64
#         return np.exp(-2.0 * (np.pi * freq64 * sigma_y)**2)

#     def _check_params(params):
#         if params.ndim != 1 or params.shape[0] != n_templates + 3:
#             raise ValueError(
#                 f"Host expects 3+{n_templates}={n_templates+3} parameters; "
#                 f"received {params.shape}"
#             )

#     @with_param_names(param_names, linear_param_names=linear_names, profile_name="host")
#     def model(x, params):
#         x = jnp.asarray(x, dtype=jnp.float32)
#         params = jnp.asarray(params, dtype=jnp.float32)
#         _check_params(params)
#         logamp, logfwhm, vshift = params[:3]
#         weights = params[3:]
#         base = weights @ templates_jax
#         base_pad = jnp.pad(base, (pad_pix, pad_pix))
#         broadened_full = jnp.fft.irfft(
#             jnp.fft.rfft(base_pad, n=n_fft) * jax_transfer(logfwhm), n=n_fft
#         )
#         broadened = broadened_full[pad_pix:pad_pix+n_pix] * taper32
#         xp = wave32 * jnp.exp(vshift / c32)
#         return jnp.power(jnp.float32(10.0), logamp) * jnp.interp(
#             x, xp, broadened, left=0.0, right=0.0
#         )

#     def design_builder(x, params):
#         x = jnp.asarray(x, dtype=jnp.float32)
#         params = jnp.asarray(params, dtype=jnp.float32)
#         _check_params(params)
#         logamp, logfwhm, vshift = params[:3]
#         broadened_full = jnp.fft.irfft(
#             fft32 * jax_transfer(logfwhm)[None, :],
#             n=n_fft, axis=-1
#         )
#         broadened = broadened_full[:, pad_pix:pad_pix+n_pix] * taper32[None, :]
#         xp = wave32 * jnp.exp(vshift / c32)
#         # On the user's machine the original vmapped interpolation was much
#         # faster than the explicit take/searchsorted interpolation.
#         columns = jax.vmap(
#             lambda template: jnp.interp(x, xp, template, left=0.0, right=0.0)
#         )(broadened).T
#         columns = jnp.power(jnp.float32(10.0), logamp) * columns
#         return jnp.zeros_like(x), columns

#     def numpy_model(x, params):
#         x = np.asarray(x, dtype=np.float64)
#         params = np.asarray(params, dtype=np.float64)
#         _check_params(params)
#         logamp, logfwhm, vshift = params[:3]
#         weights = params[3:]
#         base = weights @ templates64
#         base_pad = np.pad(base, (pad_pix, pad_pix))
#         broadened_full = irfft(rfft(base_pad, n=n_fft) * numpy_transfer(logfwhm), n=n_fft)
#         broadened = broadened_full[pad_pix:pad_pix+n_pix] * taper
#         xp = wave * np.exp(vshift / c64)
#         return 10.0**logamp * np.interp(x, xp, broadened, left=0.0, right=0.0)

#     def numpy_design_builder(x, params):
#         x = np.asarray(x, dtype=np.float64)
#         params = np.asarray(params, dtype=np.float64)
#         _check_params(params)
#         logamp, logfwhm, vshift = params[:3]
#         broadened_full = irfft(
#             fft64 * numpy_transfer(logfwhm)[None, :], n=n_fft, axis=-1
#         )
#         broadened = broadened_full[:, pad_pix:pad_pix+n_pix] * taper[None, :]
#         xp = wave * np.exp(vshift / c64)
#         idx_r = np.clip(np.searchsorted(xp, x, side="right"), 1, n_pix - 1)
#         idx_l = idx_r - 1
#         fraction = (x - xp[idx_l]) / (xp[idx_r] - xp[idx_l])
#         cols = ((1.0 - fraction)[None, :] * broadened[:, idx_l]
#                 + fraction[None, :] * broadened[:, idx_r]).T
#         outside = (x < xp[0]) | (x > xp[-1])
#         cols[outside, :] = 0.0
#         cols *= 10.0**logamp
#         return np.zeros_like(x), cols

#     def host_safe_mask(x):
#         """Conservative, *fixed* wavelength-support mask for the host."""
#         x = np.asarray(x)
#         return (x >= safe_wave_min) & (x <= safe_wave_max)

#     host_info = {
#         "z_include": zs,
#         "age_include": ages,
#         "n_Z": n_Z,
#         "n_age": n_age,
#         "xmin": xmin,
#         "xmax": xmax,
#         "file_name": source,
#         "sigmatemplate": sigmatemplate,
#         "dln": dln,
#         "dv_pix_kms": dv_pix,
#         "data": external,
#         "crop_templates": bool(crop_templates),
#         "native_wave_min": native_wave_min,
#         "native_wave_max": native_wave_max,
#         "selected_wave_min": float(wave[0]),
#         "selected_wave_max": float(wave[-1]),
#         "safe_wave_min": safe_wave_min,
#         "safe_wave_max": safe_wave_max,
#         "host_safe_mask": host_safe_mask,
#         "velocity_bounds": (vmin, vmax),
#         "max_logfwhm": float(max_logfwhm),
#         "n_sigma_margin": float(n_sigma_margin),
#         "edge_taper_pixels": int(k),
#         "pad_pixels": int(pad_pix),
#         "n_fft": n_fft,
#         "cached_template_rfft": True,
#         "n_templates": n_templates,
#     }
#     model.design_builder = design_builder
#     model.host_info = host_info
#     model.template_info = host_info
#     result = {"model": model, "design_builder": design_builder, "host_info": host_info}
#     if add_numpy_backend:
#         model.numpy_model = numpy_model
#         model.numpy_design_builder = numpy_design_builder
#         # Both locations supported: callers use host_result['numpy_*']
#         # and SHEAP has also used ``line.template_info['numpy_*']``.
#         host_info.update({
#             "numpy_model": numpy_model,
#             "numpy_design_builder": numpy_design_builder,
#         })
#         result.update({
#             "numpy_model": numpy_model,
#             "numpy_design_builder": numpy_design_builder,
#         })

#     if verbose:
#         print(f"Host: {n_Z} Z x {n_age} ages = {n_templates} SSPs, {n_pix} native pixels")
#         print(f"  Library coverage: {native_wave_min:.1f} - {native_wave_max:.1f} A")
#         print(f"  Selected coverage: {wave[0]:.1f} - {wave[-1]:.1f} A")
#         print(f"  Conservative safe fit: {safe_wave_min:.1f} - {safe_wave_max:.1f} A")
#         print(f"  Zero-padded FFT: n_fft={n_fft}, padding={pad_pix} px each side")

#     return result
