# from pathlib import Path
# import numpy as np
# import jax.numpy as jnp
# from typing import Callable, Optional, Union,Sequence

# from sheap.Functions.utils import with_param_names


# templates_path = Path(__file__).resolve().parent.parent / "SuportData" / "templates"

# def make_host_function(
#     filename: str = templates_path / "miles_cube_log.npz",
#     z_include: Optional[Union[tuple[float, float], list[float]]] = [-0.7, 0.22],
#     age_include: Optional[Union[tuple[float, float], list[float]]] = [0.1, 10.0],
#     x_min: Optional[float] = None,  # in Angstroms (linear)
#     x_max: Optional[float] = None,
#     **kwargs,
# ) -> dict:
#     """
#     Load host model from a .npz cube file and return a functional host model interface.
#     Allows sub-selection of Z, age, and wavelength (via x_min, x_max in Angstroms).

#     Returns:
#     {
#         'model': Callable[[x, params], flux], with attributes .param_names, .n_params
#         'host_info': dict[str, np.ndarray]
#     }
#     """
#     data = np.load(filename)

#     cube = data["cube_log"]
#     wave = data["wave_log"]
#     all_ages = data["ages_sub"]
#     all_zs = data["zs_sub"]
#     sigmatemplate = float(data["sigmatemplate"])
#     fixed_dispersion = float(data["fixed_dispersion"])

#     print(f"cube.sum(): {cube.sum()}, cube.shape:{cube.shape}")
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

#     if x_min is not None or x_max is not None:
#         mask = np.ones_like(wave, dtype=bool)
#         #to avoid border issues ?
#         if x_min is not None:
#             mask &= wave >= max([x_min - 50, min(wave)])
#         if x_max is not None:
#             mask &= wave <= min([x_max + 50,max(wave)])

#         if not np.any(mask):
#             raise ValueError("No wavelength values left after applying x_min/x_max cut.")

#         wave = wave[mask]
#         cube = cube[:, :, mask]

#     # Global normalization of cube to sum=1
#     #cube = cube
#     #cube /= np.clip(np.sum(cube), 1e-10, np.inf)
#     #print(f"cube.sum(): {cube.sum()}, cube.shape:{cube.shape}")
    
#     dx = wave[1] - wave[0]
#     n_Z, n_age, n_pix = cube.shape
#     print(f"Will be added n_Z: {n_Z} and n_age: {n_age}")
#     templates_flat = cube.reshape(-1, n_pix)
#     grid_metadata = [(float(Z), float(age)) for Z in zs for age in ages]

#     param_names = ["log_amp", "log_FWHM", "shift"]
#     for Z, age in grid_metadata:
#         zstr = str(Z).replace(".", "p")
#         astr = str(age).replace(".", "p")
#         param_names.append(f"weight_Z{zstr}_age{astr}")

#     @with_param_names(param_names)
#     def model(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
#         log_amp = params[0]
#         amplitude = 10**log_amp
#         log_FWHM = params[1]
#         shift_A = params[2]
#         weights = params[3:]

#         FWHM = 10 ** log_FWHM
#         sigma_model = FWHM / 2.355
#         delta_sigma = jnp.sqrt(jnp.maximum(sigma_model**2 - sigmatemplate**2, 1e-12))
#         sigma_pix = delta_sigma / fixed_dispersion
#         sigma_lambda = sigma_pix * dx

#         freq = jnp.fft.fftfreq(n_pix, d=dx)
#         gauss_tf = jnp.exp(-2 * (jnp.pi * freq * sigma_lambda) ** 2)

#         templates_fft = jnp.fft.fft(templates_flat, axis=1)
#         convolved = jnp.real(jnp.fft.ifft(templates_fft * gauss_tf, axis=1))

#         model_flux = jnp.sum(weights[:, None] * convolved, axis=0)
#         shifted = wave + shift_A
#         return amplitude * jnp.interp(x, shifted, model_flux, left=0.0, right=0.0)

#     return {
#         "model": model,
#         "host_info": {
#             "z_include": zs,
#             "age_include": ages,
#             "n_Z": n_Z,
#             "n_age": n_age,
#             "x_min": x_min,
#             "x_max": x_max,
#         },
#     }


# #n_Z, n_age, n_pix = cube.shape