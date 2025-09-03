"""
Pseudo Monte Carlo Sampler
==========================

.. note::
   the docs require update now.

?
"""
from __future__ import annotations
__author__ = 'felavila'

__all__ = ["PseudoMonteCarloSampler"]

from typing import Callable, Optional, Tuple, Any, Dict, Dict, List
#import warning 

import jax.numpy as jnp
from jax import vmap, random,jacobian
import jax

import numpy as np 

from typing import Any, Callable, Dict, Optional, Tuple


from sheap.Assistants.parser_mapper import descale_amp,scale_amp,apply_tied_and_fixed_params,make_get_param_coord_value,build_tied,parse_dependencies,flatten_tied_map
from sheap.Assistants.Parameters import build_Parameters
from sheap.ComplexParams.ComplexParams import ComplexParams


class PseudoMonteCarloSampler:
    """
    Laplace (pseudo Monte Carlo) sampler in PHYSICAL space, robust and batched.
    ?
    """

    def __init__(self, estimator: Any, dtype=jnp.float32):
        self.estimator = estimator
        self.dtype = dtype
        self.complexparams = ComplexParams(estimator)
        self.obj_params = getattr(estimator, "obj_params", None) or getattr(estimator, "params", None)
        self.names = getattr(estimator, "names", None)
        self.constraints = getattr(estimator, "constraints", None)
        self.params_dict = getattr(estimator, "params_dict", None)
        self.scale = getattr(estimator, "scale", None)
        #self.constraints = 
        
        if self.obj_params is None or isinstance(self.obj_params, jnp.ndarray):
            
            self.initial_params = getattr(estimator, "initial_params", None)
            
            #self.constraints
            self.get_param_coord_value = make_get_param_coord_value(self.params_dict, self.initial_params)
            self.fitkwargs = getattr(estimator, "fitkwargs", None)
            self.obj_params = self._make_params_obj()

        if not (hasattr(self.obj_params, "raw_to_phys") and hasattr(self.obj_params, "phys_to_raw")):
            raise TypeError("obj_params must provide raw_to_phys(x) and phys_to_raw(x).")

        params_phys = getattr(estimator.result, "params", None)
        #params_dict, params, scale
        #(params_dict, params, scale)
        params_phys = descale_amp(self.params_dict,params_phys,self.scale)
        if params_phys is None:
            raise RuntimeError("estimator.result.params not found (expected physical mode).")

        mode = jnp.asarray(params_phys, dtype=self.dtype)
        self.mode_phys, self.is_batched = self._ensure_batched(mode)  # (N, D), bool

    def sample_params(
        self,
        num_samples: int,
        key_seed: int = 0,
        cov_phys: Optional[jnp.ndarray] = None,
        residuals_fn_phys: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
        eps: float = 1e-8,
        summarize: bool = True,  # kept for compatibility; not used here
    ) -> Dict[str, Any]:
        """
        Returns
        -------
        dict with keys:
          - "samples_raw":  (S, D) or (N, S, D)
          - "samples_phys": (S, D) or (N, S, D)  (always included)
          - "products":     postprocess_fn(samples_phys) if provided
        """
        key = random.PRNGKey(key_seed)

        lo_b, hi_b = self._get_bounds_phys_broadcast()                        # (N, D)
        covP = self._get_cov_phys_broadcast(cov_phys, residuals_fn_phys)      # (N, D, D)

        phys_samples = self._laplace_phys_samples_safe_batched(
            mean_phys=self.mode_phys,
            cov_phys=covP,
            lo_phys=lo_b,
            hi_phys=hi_b,
            key=key,
            num_samples=num_samples,
            eps=eps,)
        #bottle neck. 
        dic_posterior_params = {}
        for n,name_i in enumerate(self.names):
          if n % 100 == 0:
            print(f"{n} of {len(self.names)}")
          full_samples = scale_amp(self.params_dict,phys_samples[n],np.array(self.scale[n]))
          dic_posterior_params[name_i] = self.complexparams.extract_params(full_samples,n,summarize=summarize)
          dic_posterior_params[name_i].update({"samples_phys":full_samples})
        
        return dic_posterior_params
        
    def _get_cov_phys_broadcast(
        self,
        cov_phys: Optional[jnp.ndarray],
        residuals_fn_phys: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
    ) -> jnp.ndarray:
        """
        Returns (N, D, D). Broadcasts shared (D, D) to all N.
        """
        if cov_phys is not None:
            cov = jnp.asarray(cov_phys, dtype=self.dtype)
        else:
            maybe_cov = getattr(self.estimator.result, "cov_phys", None)
            cov = jnp.asarray(maybe_cov, dtype=self.dtype) if maybe_cov is not None else None

        if cov is not None:
            if cov.ndim == 2:
                cov = jnp.broadcast_to(cov, (self.mode_phys.shape[0],) + cov.shape)  # (N, D, D)
            elif cov.ndim == 3:
                if cov.shape[0] != self.mode_phys.shape[0]:
                    raise ValueError(f"cov_phys batch dim {cov.shape[0]} != N {self.mode_phys.shape[0]}")
            else:
                raise ValueError("cov_phys must be (D,D) or (N,D,D)")
            return self._symmetrize_jitter_batched(cov)

        # Fallback: compute via Gauss–Newton per object
        if residuals_fn_phys is None:
            residuals_fn_phys = getattr(self.estimator, "residuals_fn_phys", None)

        if residuals_fn_phys is None:
            # Tiny-diagonal per object
            N, D = self.mode_phys.shape
            diag_vecs = jnp.ones((N, D), dtype=self.dtype) * 1e-6
            cov = vmap(jnp.diag, in_axes=0, out_axes=0)(diag_vecs)  # (N, D, D)
            return self._symmetrize_jitter_batched(cov)

        def _gn(x_phys: jnp.ndarray) -> jnp.ndarray:
            J = jacobian(residuals_fn_phys)(x_phys)     # (L, D)
            H = J.T @ J
            H = 0.5 * (H + H.T) + jnp.eye(H.shape[0], dtype=H.dtype) * 1e-8
            return jnp.linalg.pinv(H, rcond=1e-12)

        cov = vmap(_gn, in_axes=0)(self.mode_phys)      # (N, D, D)
        return self._symmetrize_jitter_batched(cov)

    def _symmetrize_jitter_batched(self, cov: jnp.ndarray, jitter: float = 0.0) -> jnp.ndarray:
        cov = 0.5 * (cov + jnp.swapaxes(cov, -1, -2))
        if jitter > 0:
            I = jnp.eye(cov.shape[-1], dtype=cov.dtype)
            cov = cov + I[None, ...] * jitter
        return cov

    def _raw_to_phys(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.obj_params.raw_to_phys(x)

    def _phys_to_raw(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.obj_params.phys_to_raw(x)

    def _get_bounds_phys(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Build (lo, hi) in physical space from Parameters.specs (preferred),
        else from estimator.result.constraints, else unbounded.
        """
        if hasattr(self.obj_params, "specs") and self.obj_params.specs is not None:
            specs = self.obj_params.specs
            los, his = [], []
            for s in specs:
                if isinstance(s, dict):
                    lo = s.get("min", None)
                    hi = s.get("max", None)
                else:
                    lo = getattr(s, "min", None)
                    hi = getattr(s, "max", None)
                los.append(-jnp.inf if lo is None else lo)
                his.append( jnp.inf if hi is None else hi)
            return jnp.asarray(los, self.dtype), jnp.asarray(his, self.dtype)

        res_constraints = getattr(self.estimator.result, "constraints", None)
        names = getattr(self.obj_params, "names", None)
        if (res_constraints is not None) and (names is not None):
            los, his = [], []
            for nm in names:
                c = res_constraints.get(nm, {}) if isinstance(res_constraints, dict) else {}
                lo = c.get("min", None)
                hi = c.get("max", None)
                los.append(-jnp.inf if lo is None else lo)
                his.append( jnp.inf if hi is None else hi)
            return jnp.asarray(los, self.dtype), jnp.asarray(his, self.dtype)

        D = self.mode_phys.shape[-1]
        return (jnp.full((D,), -jnp.inf, dtype=self.dtype),
                jnp.full((D,),  jnp.inf, dtype=self.dtype))

    def _get_bounds_phys_broadcast(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        lo, hi = self._get_bounds_phys()        # (D,), (D,)
        N, D = self.mode_phys.shape
        return jnp.broadcast_to(lo, (N, D)), jnp.broadcast_to(hi, (N, D))


    @staticmethod
    def _make_psd(cov: jnp.ndarray, min_eig: float = 1e-12) -> jnp.ndarray:
        cov = 0.5 * (cov + jnp.swapaxes(cov, -1, -2))
        w, V = jnp.linalg.eigh(cov)                         # w: (N,D), V: (N,D,D)
        cov_psd = (V * w[..., None, :]) @ jnp.swapaxes(V, -1, -2)
        return cov_psd

    @classmethod
    def _safe_cholesky_batched(cls, cov: jnp.ndarray, jitter0: float = 1e-10, max_tries: int = 6) -> jnp.ndarray:
        def chol_one(C: jnp.ndarray) -> jnp.ndarray:
            C = 0.5 * (C + C.T)
            # try direct
            try:
                return jnp.linalg.cholesky(C)
            except Exception:
                pass
            # jitter backoff
            j = jitter0
            for _ in range(max_tries):
                try:
                    return jnp.linalg.cholesky(C + j * jnp.eye(C.shape[0], dtype=C.dtype))
                except Exception:
                    j *= 10.0
            C_psd = PseudoMonteCarloSampler._make_psd(C[None, ...])[0]
            return jnp.linalg.cholesky(C_psd + 1e-8 * jnp.eye(C.shape[0], dtype=C.dtype))

        return vmap(chol_one, in_axes=0)(cov)  # (N, D, D)

    @classmethod
    def _laplace_phys_samples_safe_batched(
        cls,
        mean_phys: jnp.ndarray,   # (N, D)
        cov_phys: jnp.ndarray,    # (N, D, D)
        lo_phys: jnp.ndarray,     # (N, D)
        hi_phys: jnp.ndarray,     # (N, D)
        key: jax.Array,
        num_samples: int,
        eps: float = 1e-8,
    ) -> jnp.ndarray:
        N, D = mean_phys.shape
        L = cls._safe_cholesky_batched(cov_phys)                    # (N, D, D)

        # Random normals per object → (N, S, D)
        subkeys = random.split(key, N)                           # (N, 2)
        def draw_one(k: jax.Array, mu: jnp.ndarray, Lm: jnp.ndarray) -> jnp.ndarray:
            z = random.normal(k, (num_samples, D), dtype=mu.dtype)
            return mu[None, :] + z @ jnp.swapaxes(Lm, -1, -2)

        phys = vmap(draw_one, in_axes=(0, 0, 0))(subkeys, mean_phys, L)  # (N, S, D)

        # Enforce bounds per object
        reflect = vmap(PseudoMonteCarloSampler._reflect_into_bounds, in_axes=(0, 0, 0))
        push    = vmap(PseudoMonteCarloSampler._push_interior,      in_axes=(0, 0, 0, None))
        sanitize= vmap(PseudoMonteCarloSampler._sanitize_nonfinite, in_axes=(0, 0))

        phys = reflect(phys, lo_phys[:, None, :], hi_phys[:, None, :])       # (N, S, D)
        phys = push(phys,    lo_phys[:, None, :], hi_phys[:, None, :], eps)  # NOTE: eps is NOT mapped
        phys = sanitize(phys, mean_phys[:, None, :])
        return phys

    @staticmethod
    def _reflect_into_bounds(x: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray) -> jnp.ndarray:
        finite_lo = jnp.isfinite(lo)
        finite_hi = jnp.isfinite(hi)
        width = hi - lo
        finite_interval = jnp.logical_and(finite_lo, finite_hi) & (width > 0)

        t = (x - lo) / jnp.where(finite_interval, width, 1.0)
        t_wrapped = t - jnp.floor(t / 2.0) * 2.0
        x_reflected = jnp.where(
            t_wrapped <= 1.0,
            lo + t_wrapped * width,
            hi - (t_wrapped - 1.0) * width,
        )

        x = jnp.where(finite_interval, x_reflected, x)
        x = jnp.where(jnp.logical_and(finite_lo, ~finite_hi), jnp.maximum(x, lo), x)
        x = jnp.where(jnp.logical_and(~finite_lo, finite_hi), jnp.minimum(x, hi), x)
        return x

    @staticmethod
    def _push_interior(x: jnp.ndarray, lo: jnp.ndarray, hi: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
        finite_lo = jnp.isfinite(lo)
        finite_hi = jnp.isfinite(hi)
        both = jnp.logical_and(finite_lo, finite_hi)
        lo_i = jnp.where(both, lo + eps, lo)
        hi_i = jnp.where(both, hi - eps, hi)
        return jnp.minimum(jnp.maximum(x, lo_i), hi_i)

    @staticmethod
    def _sanitize_nonfinite(x: jnp.ndarray, fallback: jnp.ndarray) -> jnp.ndarray:
        bad = ~jnp.isfinite(x)
        return jnp.where(bad, fallback, x)

    @staticmethod
    def _ensure_batched(mode_phys: jnp.ndarray) -> Tuple[jnp.ndarray, bool]:
        if mode_phys.ndim == 1:
            return mode_phys[None, :], False
        if mode_phys.ndim == 2:
            return mode_phys, True
        raise ValueError("mode_phys must be (D,) or (N, D)")

    def _make_params_obj(self):
        list_dependencies = parse_dependencies(
            build_tied(self.fitkwargs[-1]["tied"], self.get_param_coord_value)
        )
        tied_map = {T[1]: T[2:] for T in list_dependencies}
        tied_map = flatten_tied_map(tied_map)
        return build_Parameters(tied_map, self.params_dict, self.initial_params, self.constraints)

    
    
  


