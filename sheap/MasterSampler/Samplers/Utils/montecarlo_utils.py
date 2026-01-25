
"""
Monte Carlo Sampler utils
===================
?
"""


__author__ = 'felavila'

__all__ = ["phys_trust_region_inits","resample_spec_all"]

import jax.numpy as jnp
from jax import jit , random
import jax.numpy as jnp



def phys_trust_region_inits(key, *, params_class, best_params, phys_bounds, num_samples=100, sigma_phys=None, frac_box_sigma=0.05, k_sigma= 0.5 ):
    key = random.PRNGKey(key) if isinstance(key, int) else key

    lo = jnp.array([b[0] for b in phys_bounds], dtype=jnp.float32)
    hi = jnp.array([b[1] for b in phys_bounds], dtype=jnp.float32)
    width = hi - lo

    if sigma_phys is None:
        sigma_phys = jnp.where(width > 0, frac_box_sigma * width, 0.0)

    keys = random.split(key, num_samples)
    draws_phys = []
    for ki in keys:
        step = k_sigma * sigma_phys * random.normal(ki, shape=best_params.shape)
        phys = best_params + step
        phys = jnp.clip(phys, lo, hi)
        draws_phys.append(phys)

    draws_phys = jnp.stack(draws_phys)
    
    draws_raw = jnp.stack([params_class.phys_to_raw(p) for p in draws_phys])
    return draws_raw, draws_phys

def resample_spec_all(key, spec):
    """
    Resample flux for all objects in `spec` using their per-pixel errors.

    Assumes `spec` has shape (C, N_obj, X) with:
      spec[0, :, :] = wavelength (unchanged)
      spec[1, :, :] = flux (resampled)
      spec[2, :, :] = 1-sigma error (used for noise)

    Parameters
    ----------
    key : jax.random.PRNGKey
    spec : array-like, shape (3, N_obj, X)

    Returns
    -------
    spec_out : jnp.ndarray, shape (3, N_obj, X), dtype float32
        Same as input but with resampled flux channel.
    """
    spec = jnp.asarray(spec, dtype=jnp.float32)

    #wave  = spec[0]  
    flux  = spec[1]  
    sigma = spec[2] 


    eps = random.normal(key, shape=flux.shape, dtype=jnp.float32)

    flux_new = flux + sigma * eps

    spec_out = spec.at[1].set(flux_new)

    return spec_out



# def phys_trust_region_inits(
#     key, *,
#     params_class,
#     best_params,
#     phys_bounds,
#     num_samples=100,
#     sigma_raw=None,        # std in raw space
#     frac_box_sigma=0.05,
#     k_sigma=0.5,
# ):
#     key = random.PRNGKey(key) if isinstance(key, int) else key

#     lo = jnp.array([b[0] for b in phys_bounds], dtype=jnp.float32)
#     hi = jnp.array([b[1] for b in phys_bounds], dtype=jnp.float32)

#     # map MAP -> raw
#     phys_map = best_params
#     raw_map = params_class.phys_to_raw(phys_map)

#     if sigma_raw is None:
#         # approximate raw sigma via a physical width mapped through the transform
#         width = hi - lo
#         sigma_phys = jnp.where(width > 0, frac_box_sigma * width, 0.0)
#         # finite-diff local jacobian diag: d(raw)/d(phys)
#         eps = 1e-4
#         raw_plus  = params_class.phys_to_raw(jnp.clip(phys_map + eps, lo, hi))
#         raw_minus = params_class.phys_to_raw(jnp.clip(phys_map - eps, lo, hi))
#         jac_diag = (raw_plus - raw_minus) / (2 * eps)
#         sigma_raw = jnp.abs(jac_diag) * sigma_phys

#     keys = random.split(key, num_samples)
#     draws_raw = []
#     for ki in keys:
#         step = k_sigma * sigma_raw * random.normal(ki, shape=raw_map.shape)
#         r = raw_map + step
#         # convert back to phys and enforce bounds
#         p = jnp.clip(params_class.raw_to_phys(r), lo, hi)
#         draws_raw.append(params_class.phys_to_raw(p))

#     draws_raw = jnp.stack(draws_raw)
#     draws_phys = jnp.stack([params_class.raw_to_phys(r) for r in draws_raw])
#     return draws_raw, draws_phys
