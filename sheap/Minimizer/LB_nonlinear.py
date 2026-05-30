from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp


dtype = jnp.float32

# varpro.py  — optimized
from functools import partial
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp

dtype = jnp.float32


def _solve_wlls(
    A: jnp.ndarray,
    y: jnp.ndarray,
    yerr: jnp.ndarray,
    lambda_reg: float = 0.0,
    reg_matrix: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Weighted linear least-squares. Single implementation — dead duplicate removed."""
    A    = jnp.asarray(A,    dtype=dtype)
    y    = jnp.asarray(y,    dtype=dtype).reshape(-1)
    yerr = jnp.asarray(yerr, dtype=dtype).reshape(-1)
    n_lin = A.shape[1]

    if A.shape[0] == 0:
        return jnp.full((n_lin,), jnp.nan, dtype=dtype)

    w  = 1.0 / jnp.clip(yerr, 1e-10)
    Aw = A * w[:, None]
    yw = y * w

    if lambda_reg > 0.0:
        R = jnp.eye(n_lin, dtype=dtype) if reg_matrix is None \
            else jnp.asarray(reg_matrix, dtype=dtype)
        # append regularization rows in one shot — avoids two separate vstack calls
        sqrt_lam = jnp.sqrt(jnp.asarray(lambda_reg, dtype=dtype))
        Aw = jnp.vstack([Aw, sqrt_lam * R])
        yw = jnp.concatenate([yw, jnp.zeros(R.shape[0], dtype=dtype)])

    x, *_ = jnp.linalg.lstsq(Aw, yw, rcond=None)
    return jnp.asarray(x, dtype=dtype)


def make_build_linear_system_from_phys_profile(fused_profile: Callable):
    """Same semantics; column builder is now vmapped over a static arange so
    JAX can compile it once and cache the result."""
    linear_phys_idx    = jnp.array(fused_profile.linear_param_indices,    dtype=jnp.int32)
    nonlinear_phys_idx = jnp.array(fused_profile.nonlinear_param_indices, dtype=jnp.int32)
    n_params = int(fused_profile.n_params)
    n_linear = int(linear_phys_idx.shape[0])
    col_idx  = jnp.arange(n_linear, dtype=jnp.int32)  # static — computed once

    def build_linear_system(
        x: jnp.ndarray,
        params_phys: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x           = jnp.asarray(x,           dtype=dtype)
        params_phys = jnp.asarray(params_phys, dtype=dtype).reshape(-1)

        params_base = params_phys.at[linear_phys_idx].set(0.0)
        y_base      = jnp.asarray(fused_profile(x, params_base), dtype=dtype).reshape(-1)

        if n_linear == 0:
            A = jnp.zeros((y_base.shape[0], 0), dtype=dtype)
            return A, y_base, linear_phys_idx, nonlinear_phys_idx

        def one_column(j):
            params_j = params_base.at[linear_phys_idx[j]].set(1.0)
            return jnp.asarray(fused_profile(x, params_j), dtype=dtype).reshape(-1) - y_base

        # col_idx is a captured static array — vmap traces this once per unique n_linear
        A = jax.vmap(one_column)(col_idx).T
        return A, y_base, linear_phys_idx, nonlinear_phys_idx

    return build_linear_system


def build_varpro_loss_from_profile_and_params_obj(
    fused_profile: Callable,
    params_obj,
    nonlinear_raw_idx: jnp.ndarray,
    linear_phys_idx: Optional[jnp.ndarray] = None,
    weighted: bool = True,
    lambda_reg: float = 0.0,
    reg_matrix: Optional[jnp.ndarray] = None,
):
    nonlinear_raw_idx = jnp.asarray(nonlinear_raw_idx, dtype=jnp.int32)

    if linear_phys_idx is None:
        linear_phys_idx = jnp.asarray(fused_profile.linear_param_indices, dtype=jnp.int32)
    else:
        linear_phys_idx = jnp.asarray(linear_phys_idx, dtype=jnp.int32)

    n_params          = int(fused_profile.n_params)
    build_linear_system = make_build_linear_system_from_phys_profile(fused_profile)

    # unit-error array reused across calls — avoids re-allocation in the unweighted branch
    _ones_cache: dict = {}

    def _get_ones(n: int):
        if n not in _ones_cache:
            _ones_cache[n] = jnp.ones((n,), dtype=dtype)
        return _ones_cache[n]

    def unpack_raw_nonlinear(raw_nonlinear):
        raw_nonlinear = jnp.asarray(raw_nonlinear, dtype=dtype).reshape(-1)
        raw_full      = jnp.zeros((n_params,), dtype=dtype)
        return raw_full.at[nonlinear_raw_idx].set(raw_nonlinear)

    def build_from_raw_nonlinear(raw_nonlinear, x):
        raw_full  = unpack_raw_nonlinear(raw_nonlinear)
        phys_full = jnp.asarray(params_obj.raw_to_phys(raw_full), dtype=dtype).reshape(-1)
        A, y_base, _, _ = build_linear_system(x, phys_full)
        return raw_full, phys_full, A, y_base

    def solve_full_parameters(raw_nonlinear, x, y, yerr):
        x    = jnp.asarray(x,    dtype=dtype)
        y    = jnp.asarray(y,    dtype=dtype).reshape(-1)
        yerr = jnp.asarray(yerr, dtype=dtype).reshape(-1)

        raw_full, phys_full_nonlin, A, y_base = build_from_raw_nonlinear(raw_nonlinear, x)
        y_target = y - y_base

        _yerr = yerr if weighted else _get_ones(y.shape[0])
        a_star  = _solve_wlls(A, y_target, _yerr, lambda_reg=lambda_reg, reg_matrix=reg_matrix)
        y_model = y_base + A @ a_star

        phys_full_best = phys_full_nonlin.at[linear_phys_idx].set(a_star)

        if hasattr(params_obj, "phys_to_raw"):
            raw_full_best = jnp.asarray(params_obj.phys_to_raw(phys_full_best), dtype=dtype).reshape(-1)
        else:
            raw_full_best = raw_full

        return phys_full_best, raw_full_best, a_star, y_model, y_base, A

    # ── loss is JIT-compiled once; shape changes retrace but that's unavoidable ──
    @jax.jit
    def loss(raw_nonlinear, x, y, yerr):
        x    = jnp.asarray(x,    dtype=dtype)
        y    = jnp.asarray(y,    dtype=dtype).reshape(-1)
        yerr = jnp.asarray(yerr, dtype=dtype).reshape(-1)

        _, _, _, y_model, _, _ = solve_full_parameters(raw_nonlinear, x, y, yerr)

        if weighted:
            resid = (y_model - y) / jnp.clip(yerr, 1e-8)
        else:
            resid = y_model - y

        return jnp.mean(resid ** 2)

    return loss, solve_full_parameters