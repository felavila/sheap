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
from sheap.Utils.Constants import DEFAULT_C_KMS

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
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Factory for a template model with optional wavelength cuts.

    Parameters
    ----------
    name
        Template name in TEMPLATES.

    x_min, x_max
        Optional wavelength limits in Angstrom.

    Returns
    -------
    dict
        {
            "model": model,
            "design_builder": design_builder,
            "template_info": {...},
        }

    Notes
    -----
    Parameters remain fully nonlinear:

        [logamp, logFWHM, vshift_kms]

    Therefore the design builder returns:

        offset  = complete template model
        columns = empty array with shape (n_pixels, 0)

    The intrinsic template rFFT is precomputed once, but the
    FWHM-dependent broadening kernel and velocity shift are
    recomputed for every parameter evaluation.
    """

    # ============================================================
    # Template configuration
    # ============================================================

    cfg = TEMPLATES.get(name)

    if cfg is None:
        raise KeyError(
            f"No such template: {name}"
        )

    path = cfg["file"]
    central_wl = cfg["central_wl"]
    sigmatemplate = cfg["sigmatemplate"]

    user_fd = cfg.get(
        "fixed_dispersion",
        None,
    )

    # ============================================================
    # Read template
    # ============================================================

    data = np.loadtxt(
        path,
        comments="#",
    ).T

    wl = np.asarray(
        data[0],
        dtype=np.float64,
    )

    flux = np.asarray(
        data[1],
        dtype=np.float64,
    )

    # ============================================================
    # Optional wavelength cut
    # ============================================================

    if (
        x_min is not None
        or x_max is not None
    ):

        mask = np.ones_like(
            wl,
            dtype=bool,
        )

        if x_min is not None:
            mask &= (
                wl
                >= max(
                    float(x_min) - 50.0,
                    float(wl.min()),
                )
            )

        if x_max is not None:
            mask &= (
                wl
                <= min(
                    float(x_max) + 50.0,
                    float(wl.max()),
                )
            )

        if not np.any(mask):
            raise ValueError(
                f"No wavelength values left after "
                f"applying x_min/x_max cut for {name}"
            )

        wl = wl[mask]

        flux = flux[mask]

    # ============================================================
    # Validate wavelength grid
    # ============================================================

    if wl.size < 3:
        raise ValueError(
            "Template too short after cutting; "
            "need at least 3 points."
        )

    dl = float(
        wl[1] - wl[0]
    )

    # ============================================================
    # Normalize template
    # ============================================================

    unit_flux = (
        flux
        / np.clip(
            np.sum(flux),
            1e-10,
            np.inf,
        )
    ).astype(
        np.float32
    )

    # ============================================================
    # Intrinsic template dispersion
    # ============================================================

    if user_fd is None:

        fixed_dispersion = (
            dl
            / central_wl
            * DEFAULT_C_KMS
        )

    else:

        fixed_dispersion = float(
            user_fd
        )

    # ============================================================
    # Parameters
    #
    # ALL remain nonlinear.
    # ============================================================

    param_names = [
        "logamp",
        "logFWHM",
        "vshift_kms",
    ]

    # ============================================================
    # Fixed arrays / constants
    # ============================================================

    wl_jax = jnp.asarray(
        wl,
        dtype=jnp.float32,
    )

    c_kms = jnp.asarray(
        DEFAULT_C_KMS,
        dtype=jnp.float32,
    )

    sigma_template = jnp.asarray(
        sigmatemplate,
        dtype=jnp.float32,
    )

    dispersion = jnp.asarray(
        fixed_dispersion,
        dtype=jnp.float32,
    )

    ten = jnp.asarray(
        10.0,
        dtype=jnp.float32,
    )

    two = jnp.asarray(
        2.0,
        dtype=jnp.float32,
    )

    two355 = jnp.asarray(
        2.355,
        dtype=jnp.float32,
    )

    tiny = jnp.asarray(
        1e-12,
        dtype=jnp.float32,
    )

    # ============================================================
    # PRECOMPUTE intrinsic template rFFT
    #
    # This is fixed because unit_flux never changes.
    # ============================================================

    n_pix = unit_flux.shape[0]

    template_rfft_np = np.fft.rfft(
        unit_flux
    ).astype(
        np.complex64
    )

    template_rfft = jnp.asarray(
        template_rfft_np,
        dtype=jnp.complex64,
    )

    # Frequencies corresponding to the rFFT
    freq = jnp.asarray(
        np.fft.rfftfreq(
            n_pix,
            d=1.0,
        ),
        dtype=jnp.float32,
    )

    # ============================================================
    # Shared nonlinear template shape
    # ============================================================

    def build_profile(
        x,
        logFWHM,
        vshift_kms,
    ):
        """
        Generate broadened and shifted unit-amplitude template.

        Depends only on nonlinear shape parameters:
            logFWHM
            vshift_kms
        """

        x = jnp.asarray(
            x,
            dtype=jnp.float32,
        )

        # --------------------------------------------------------
        # Fitted FWHM
        # --------------------------------------------------------

        FWHM = jnp.power(
            ten,
            logFWHM,
        )

        sigma_model = (
            FWHM
            / two355
        )

        # --------------------------------------------------------
        # Remove intrinsic template resolution
        # --------------------------------------------------------

        diff_sq = (
            sigma_model**2
            - sigma_template**2
        )

        diff_sq_safe = (
            jax.nn.softplus(
                diff_sq / ten
            )
            * ten
            + tiny
        )

        delta_sigma = jnp.sqrt(
            diff_sq_safe
        )

        # --------------------------------------------------------
        # Broadening in template pixels
        # --------------------------------------------------------

        sigma_pix = (
            delta_sigma
            / dispersion
        )

        # --------------------------------------------------------
        # Current Gaussian transfer function
        #
        # THIS changes with logFWHM.
        # --------------------------------------------------------

        gauss_tf = jnp.exp(
            -two
            * (
                jnp.pi
                * freq
                * sigma_pix
            ) ** 2
        )

        # --------------------------------------------------------
        # Apply broadening to precomputed template FFT
        # --------------------------------------------------------

        broadened = jnp.fft.irfft(
            template_rfft
            * gauss_tf,
            n=n_pix,
        ).astype(
            jnp.float32
        )

        # --------------------------------------------------------
        # Current fitted velocity shift
        # --------------------------------------------------------

        beta = (
            vshift_kms
            / c_kms
        )

        xp = (
            wl_jax
            * (
                jnp.float32(1.0)
                + beta
            )
        )

        # --------------------------------------------------------
        # Interpolate onto requested wavelength grid
        # --------------------------------------------------------

        profile = jnp.interp(
            x,
            xp,
            broadened,
            left=jnp.float32(0.0),
            right=jnp.float32(0.0),
        )

        return profile.astype(
            jnp.float32
        )

    # ============================================================
    # Normal model
    # ============================================================

    @with_param_names(param_names,profile_name=name,)
    def model(
        x: jnp.ndarray,
        params: jnp.ndarray,
    ) -> jnp.ndarray:

        params = jnp.asarray(
            params,
            dtype=jnp.float32,
        )

        logamp = params[0]
        logFWHM = params[1]
        vshift_kms = params[2]

        # --------------------------------------------------------
        # Nonlinear amplitude
        # --------------------------------------------------------

        amplitude = jnp.power(
            ten,
            logamp,
        )

        # --------------------------------------------------------
        # Nonlinear shape
        # --------------------------------------------------------

        profile = build_profile(
            x,
            logFWHM,
            vshift_kms,
        )

        return (
            amplitude
            * profile
        ).astype(
            jnp.float32
        )

    # ============================================================
    # Design builder
    #
    # There are NO linear parameters in this profile.
    # Therefore the entire model is the offset.
    # ============================================================

    def design_builder(
        x: jnp.ndarray,
        params: jnp.ndarray,
    ):

        x = jnp.asarray(
            x,
            dtype=jnp.float32,
        )

        params = jnp.asarray(
            params,
            dtype=jnp.float32,
        )

        logamp = params[0]
        logFWHM = params[1]
        vshift_kms = params[2]

        amplitude = jnp.power(
            ten,
            logamp,
        )

        profile = build_profile(
            x,
            logFWHM,
            vshift_kms,
        )

        # --------------------------------------------------------
        # Complete nonlinear contribution
        # --------------------------------------------------------

        offset = (
            amplitude
            * profile
        ).astype(
            jnp.float32
        )

        # --------------------------------------------------------
        # Zero linear columns
        #
        # Shape:
        #     (n_data_pixels, 0)
        # --------------------------------------------------------

        columns = jnp.empty(
            (
                x.shape[0],
                0,
            ),
            dtype=jnp.float32,
        )

        return (
            offset,
            columns,
        )

    # ============================================================
    # Attach builder
    # ============================================================

    model.design_builder = (
        design_builder
    )

    # ============================================================
    # Return
    # ============================================================

    return {
        "model": model,

        "design_builder": design_builder,

        "template_info": {
            "name": name,

            "file": str(path),

            "central_wl": float(
                central_wl
            ),

            "sigmatemplate": float(
                sigmatemplate
            ),

            "fixed_dispersion": float(
                fixed_dispersion
            ),

            "x_min": (
                None
                if x_min is None
                else float(x_min)
            ),

            "x_max": (
                None
                if x_max is None
                else float(x_max)
            ),

            "dl": dl,
        },
    }

def make_template_function_(
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
        wl_shifted = wl * (1 + vshift_kms / DEFAULT_C_KMS)

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
            raise ValueError(f"No wavelength values left after applying x_min/x_max cut for {name}")
        wl   = wl[mask]
        flux = flux[mask]

    # Ensure equally spaced grid
    if wl.size < 3:
        raise ValueError("Template too short after cutting; need at least 3 points.")
    dl = float(wl[1] - wl[0])

    # Re-normalize to unit sum AFTER any cut
    unit_flux = flux / np.clip(np.sum(flux), 1e-10, np.inf)

    if user_fd is None:
        fixed_dispersion = (dl / central_wl) * DEFAULT_C_KMS
    else:
        fixed_dispersion = float(user_fd)

    param_names = ["logamp", "logFWHM", "vshift_kms"]
   
    # Pre-pack constants as JAX arrays once
    wl_jax = jnp.asarray(wl, dtype=np.float32)
    unit_flux_jax = jnp.asarray(unit_flux, dtype=np.float32)

    @with_param_names(param_names)
    def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        # Enforce single precision for all model calculations
        x = jnp.asarray(x, dtype=jnp.float32)
        params = jnp.asarray(params, dtype=jnp.float32)

        logamp, logFWHM, vshift_kms = params

        c_kms = jnp.asarray(DEFAULT_C_KMS, dtype=jnp.float32)
        sigma_template = jnp.asarray(sigmatemplate, dtype=jnp.float32)
        dispersion = jnp.asarray(fixed_dispersion, dtype=jnp.float32)

        amp = jnp.power(jnp.float32(10.0), logamp)
        FWHM = jnp.power(jnp.float32(10.0), logFWHM)
        sigma_model = FWHM / jnp.float32(2.355)

        # Quadratic subtraction
        diff_sq = sigma_model**2 - sigma_template**2
        diff_sq_safe = (
            jax.nn.softplus(diff_sq / jnp.float32(10.0))
            * jnp.float32(10.0)
            + jnp.float32(1e-12)
        )
        delta_sigma = jnp.sqrt(diff_sq_safe)

        # Broadening in pixels
        sigma_pix = delta_sigma / dispersion

        n_pix = unit_flux_jax.shape[0]
        freq = jnp.fft.fftfreq(n_pix, d=1.0).astype(jnp.float32)

        gauss_tf = jnp.exp(jnp.float32(-2.0)* (jnp.pi * freq * sigma_pix) ** 2)

        # FFT of float32 input is complex64; preserve the imaginary component
        spec_fft = jnp.fft.fft(unit_flux_jax).astype(jnp.complex64)
        broadened = jnp.fft.ifft(
            spec_fft * gauss_tf.astype(jnp.complex64)
        ).real.astype(jnp.float32)

        # Positive velocity means redder features
        beta = vshift_kms / c_kms
        xp = wl_jax * (jnp.float32(1.0) + beta)

        interpolated = jnp.interp(
            x,
            xp,
            broadened,
            left=jnp.float32(0.0),
            right=jnp.float32(0.0),
        )

        return (amp * interpolated).astype(jnp.float32)

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
    filename: str | Path = TEMPLATES_PATH / "miles_cube_log.npz",
    z_include: Optional[Union[tuple[float, float], list[float]]] = [-0.7, 0.22],
    age_include: Optional[Union[tuple[float, float], list[float]]] = [0.1, 10.0],
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    verbose: Optional[bool] = None,
    **kwargs,
) -> dict:
    r"""
    Factory for host-galaxy SSP models stored on a log-sampled wavelength grid.

    The returned model builds a weighted sum of SSP templates, then applies a single
    FFT-based Gaussian broadening in **velocity space** (LOSVD-like), and finally
    interpolates to the user grid ``x``.

    Parameters
    ----------
    filename
        Either ``"miles"``, ``"xsl"``, a path to a ``*_cube_log.npz`` file, or ``None``
        (defaults to MILES).
    z_include
        Metallicity selection; either (min,max) or list/array.
    age_include
        Age selection; either (min,max) or list/array.
    xmin, xmax
        Optional wavelength window in Angstrom. A ±50 Å guard band is applied.
    verbose
        Print selected grid size.

    Returns
    -------
    dict
        ``{"model": callable, "host_info": dict}``

    Notes
    -----
    - **Important:** The template wavelength axis in the NPZ file must be sampled
      approximately uniformly in :math:`\ln(\lambda)` (i.e. *log-spaced in wavelength*).
      The convolution kernel is constructed in :math:`\ln(\lambda)`, so the model
      assumes a constant velocity step per pixel.
    - The input ``x`` passed to ``model(x, params)`` is in **Angstrom**, but should be
      sampled on a grid that is also approximately uniform in :math:`\ln(\lambda)` if you
      want the broadening to behave exactly like a Gaussian LOSVD on your evaluation grid.
      (Interpolation to arbitrary ``x`` is allowed, but the kernel is defined on the
      template log-grid.)
    - Velocity shift is applied as :math:`\lambda \rightarrow \lambda \exp(v/c)`.

    The parameter order is:
    ``[logamp, logFWHM, vshift_kms, weight_0, weight_1, ...]``.
    """
    #print("Makehost test function")
    allowed = {"miles": TEMPLATES_PATH / "miles_cube_log.npz",
               "xsl":   TEMPLATES_PATH / "xsl_cube_log.npz",
               None:    TEMPLATES_PATH / "miles_cube_log.npz"}
    if filename in allowed:
        filename = allowed[filename]
    else:
        filename = Path(filename)
        if not filename.exists():
            raise KeyError(
                f"file_name '{filename}' is not available. Available: ['miles', 'xsl'] or a valid path."
            )

    data = np.load(filename, mmap_mode="r")

    
    cube = np.asarray(data["cube_log"], dtype=np.float32)   # (n_Z, n_age, n_pix)
    wave = np.asarray(data["wave_log"], dtype=np.float32)   # (n_pix,) in Angstrom, log-sampled
    all_ages = np.asarray(data["ages_sub"], dtype=np.float32)
    all_zs = np.asarray(data["zs_sub"], dtype=np.float32)
    sigmatemplate = float(data["sigmatemplate"])            # km/s
    #fixed_dispersion = float(data["fixed_dispersion"])      # km/s per pixel (on the log grid)


    if wave.size < 4:
        raise ValueError("Template wavelength grid too short.")
    if not np.all(np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError("wave_log must be finite and strictly > 0 (Angstrom).")
    if np.any(np.diff(wave) <= 0):
        raise ValueError("wave_log must be strictly increasing.")

    if z_include is not None:
        z_min, z_max = float(np.min(z_include)), float(np.max(z_include))
        z_mask = (all_zs >= z_min) & (all_zs <= z_max)
        if not np.any(z_mask):
            raise ValueError(f"No metallicities in range {z_min} to {z_max}")
        zs = all_zs[z_mask]
        cube = cube[z_mask, :, :]
    else:
        zs = all_zs

    if age_include is not None:
        a_min, a_max = float(np.min(age_include)), float(np.max(age_include))
        a_mask = (all_ages >= a_min) & (all_ages <= a_max)
        if not np.any(a_mask):
            raise ValueError(f"No ages in range {a_min} to {a_max}")
        ages = all_ages[a_mask]
        cube = cube[:, a_mask, :]
    else:
        ages = all_ages
    grid_metadata = [(float(Z), float(age)) for Z in zs for age in ages]
    if kwargs.get("data"):
       print("We will use external template experimental")
       data = kwargs["data"]
       filename = kwargs["data"]["filename"]
       cube =  np.asarray(data["cube_log"], dtype=np.float32)
       wave = np.asarray(data["wave_log"], dtype=np.float32)   # (n_pix,) in Angstrom, log-sampled
       assert wave.shape[0] == cube.shape[-1], "check the shapes of wave_log and cube_log"
       sigmatemplate = data["sigmatemplate"]
       n_Z, n_age = cube.shape[:-1]
       grid_metadata = [(float(Z), float(age)) for Z in np.arange(n_Z) for age in  np.arange(n_age)]
    f = 1.0  # keep if you later want external scaling
    
       
    if xmin is not None or xmax is not None:
        mask = np.ones_like(wave, dtype=bool)
        if xmin is not None:
            mask &= wave >= max(float(xmin) * f - 50.0, float(wave.min()))
        if xmax is not None:
            mask &= wave <= min(float(xmax) * f + 50.0, float(wave.max()))
        if not np.any(mask):
            raise ValueError("No wavelength values left after applying xmin/xmax.")
        wave = wave[mask].astype(np.float32, copy=False)
        cube = cube[:, :, mask].astype(np.float32, copy=False)

    n_Z, n_age, n_pix = cube.shape
    if verbose:
        print(f"Host added with n_Z={n_Z}, n_age={n_age}, n_pix={n_pix}")
    
    eps = 1e-30
    flux_int = np.nansum(cube, axis=-1, keepdims=True)
    cube = cube / (flux_int + eps)

    templates_flat = cube.reshape(-1, n_pix)  # (n_Z*n_age, n_pix)
    param_names = ["logamp", "logFWHM", "vshift_kms"]
    linear_params = []
    for Z, age in grid_metadata:
        zstr = str(Z).replace(".", "p")
        astr = str(age).replace(".", "p")
        param_names.append(f"weight_Z{zstr}_age{astr}")
        linear_params.append(f"weight_Z{zstr}_age{astr}")
    templates_jax = jnp.asarray(templates_flat)  # (Ntemp, n_pix) float32
    wave_jax = jnp.asarray(wave)                 # (n_pix,) float32

    # build convolution kernel in ln(lambda) (LOSVD-correct)
    # dln is ~constant if wave is log-sampled in Angstrom
    dln = float(np.mean(np.gradient(np.log(wave.astype(np.float64)))))
    freq = jnp.fft.fftfreq(n_pix, d=dln).astype(jnp.float32)
    @with_param_names(param_names,linear_param_names=linear_params,profile_name="host")
    def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
        logamp = params[0]
        logFWHM = params[1]
        vshift_kms = params[2]
        weights = params[3:]  # (Ntemp,)

        amplitude = 10.0 ** logamp

        base = jnp.tensordot(weights, templates_jax, axes=(0, 0))  # (n_pix,)

        FWHM = 10.0 ** logFWHM
        sigma_model = FWHM / 2.355  # km/s

        diff_sq = sigma_model**2 - sigmatemplate**2
        diff_sq_safe = jax.nn.softplus(diff_sq / 10.0) * 10.0 + 1e-12
        delta_sigma = jnp.sqrt(diff_sq_safe)  # km/s

        sigma_y = (delta_sigma / DEFAULT_C_KMS).astype(jnp.float32)

        gauss_tf = jnp.exp(-2.0 * (jnp.pi * freq * sigma_y) ** 2)
        conv = jnp.real(jnp.fft.ifft(jnp.fft.fft(base) * gauss_tf))

        xp = wave_jax * jnp.exp(vshift_kms / DEFAULT_C_KMS)

        return amplitude * jnp.interp(x * f, xp, conv, left=0.0, right=0.0)

    def design_builder(x: jnp.ndarray,params: jnp.ndarray,):
        """
        Build the profiled linear representation of the host model.

        Parameters
        ----------
        x : (n_pix_data,)
        params :
            [logamp, logFWHM, vshift_kms, weight_0, weight_1, ...]

        Returns
        -------
        offset : (n_pix_data,)
            Zero for this model because all host flux is carried
            by the linear SSP weights.

        columns : (n_pix_data, n_templates)
            One broadened/shifted SSP template per linear weight.
        """

        x = jnp.asarray(x, dtype=jnp.float32)
        params = jnp.asarray(params, dtype=jnp.float32)

        logamp = params[0]
        logFWHM = params[1]
        vshift_kms = params[2]

        # IMPORTANT:
        # params[3:] are linear coefficients and are NOT used here.

        amplitude = jnp.power(jnp.float32(10.0),logamp,)

        FWHM = jnp.power(jnp.float32(10.0),logFWHM,)

        sigma_model = FWHM / jnp.float32(2.355)

        diff_sq = (sigma_model**2 - jnp.float32(sigmatemplate)**2)

        diff_sq_safe = (jax.nn.softplus(diff_sq / jnp.float32(10.0)) * jnp.float32(10.0) + jnp.float32(1e-12))

        delta_sigma = jnp.sqrt(diff_sq_safe)

        sigma_y = (delta_sigma / jnp.float32(DEFAULT_C_KMS))

        gauss_tf = jnp.exp(-jnp.float32(2.0)* (jnp.pi* freq* sigma_y) ** 2)

        # --------------------------------------------------
        # Broaden ALL templates simultaneously
        #
        # templates_jax:
        #     (n_templates, n_template_pixels)
        # --------------------------------------------------

        templates_fft = jnp.fft.fft(templates_jax,axis=-1,)

        broadened_templates = jnp.fft.ifft(templates_fft* gauss_tf[None, :],axis=-1,).real.astype(jnp.float32)

        # --------------------------------------------------
        # Common velocity shift
        # --------------------------------------------------

        xp = (wave_jax* jnp.exp(vshift_kms/ jnp.float32(DEFAULT_C_KMS)))

        # --------------------------------------------------
        # Interpolate every template to x
        # --------------------------------------------------

        def interpolate_one(template):
            return jnp.interp(
                x * jnp.float32(f),
                xp,
                template,
                left=jnp.float32(0.0),
                right=jnp.float32(0.0),
            )

        columns = jax.vmap(interpolate_one)(broadened_templates)

        # currently:
        # (n_templates, n_data_pixels)
        #
        # design matrix wants:
        # (n_data_pixels, n_templates)

        columns = jnp.swapaxes(columns,0,1,)

        columns = (amplitude* columns)

        offset = jnp.zeros_like(x,dtype=jnp.float32,)

        return offset, columns

    model.design_builder = design_builder

    return {
        "model": model,
        "host_info": {
            "z_include": zs,
            "age_include": ages,
            "n_Z": n_Z,
            "n_age": n_age,
            "xmin": xmin,
            "xmax": xmax,
            "file_name": str(filename),
            "dln": dln,
            "dv_pix_kms": float(DEFAULT_C_KMS * dln),
            "sigmatemplate": sigmatemplate,
            "data":kwargs.get("data",None)
        },
    }

# #xsl_cube_log_
def make_host_function_classic(
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
    if filename not in ["miles","xsl",TEMPLATES_PATH / "miles_cube_log.npz",TEMPLATES_PATH / "xsl_cube_log.npz",None]:
        raise KeyError(
            f"file_name '{filename}' is not available file_name."
            f"Available parameters: {['miles','xsl']}")
    f = 1.
    filename = {"miles":TEMPLATES_PATH / "miles_cube_log.npz","xsl":TEMPLATES_PATH / "xsl_cube_log.npz"}.get(filename,TEMPLATES_PATH / "miles_cube_log.npz")
    #print(filename)
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
    linear_params = []
    for Z, age in grid_metadata:
        zstr = str(Z).replace(".", "p")
        astr = str(age).replace(".", "p")
        param_names.append(f"weight_Z{zstr}_age{astr}")
        linear_params.append(f"weight_Z{zstr}_age{astr}")
    templates_jax = jnp.asarray(templates_flat)             # (N, P) float32
    wave_jax = jnp.asarray(wave)
    #print(wave_jax)
    freq = jnp.fft.fftfreq(n_pix, d=dx).astype(jnp.float32) # (P,)

    print(linear_params)
    @with_param_names(param_names,linear_param_names=linear_params)
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


        beta = vshift_kms / DEFAULT_C_KMS
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
            "file_name":filename
        },
    }