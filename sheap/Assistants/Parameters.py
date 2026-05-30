"""
Parameter / Parameters with built-in support for *shared* (global) free parameters.

Key behavior
------------
- If you do NOT use shared params:
    raw_init() returns:
        - (n_free,) for single spectrum
        - (n_spec, n_free) for batched spectra (same as your original code)

- If you DO use shared params (any Parameter with shared=True and not fixed and not tied):
    raw_init() returns a *packed 1D vector*:
        raw_packed = [ raw_shared (n_shared,) | raw_local_flat (n_spec * n_local,) ]

    raw_to_phys(raw_packed) returns:
        - (n_spec, n_total)  (batched physical vectors, with shared params identical across spectra)

    phys_to_raw(phys) returns the same packed 1D vector.

Notes
-----
- In shared-mode, at least ONE local (shared=False) parameter should carry a vector value
  of shape (n_spec,) so the class can infer n_spec.
- Shared free parameters should usually be provided as scalars (or 0-d arrays).
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable

import jax
import jax.numpy as jnp


default_inf = float("inf")
# self.value = jnp.asarray(value, dtype=jnp.float32)
# self.min = jnp.float32(min)
# self.max = jnp.float32(max)

class Parameter:
    """
    Represents a single fit parameter with optional bounds, ties, fixed status,
    and optional shared behavior across a batch (shared=True).
    """
    def __init__(
        self,
        name: str,
        value: Union[float, jnp.ndarray, List[float], Tuple[float, ...]],
        *,
        min: float = -default_inf,
        max: float = default_inf,
        tie: Optional[Tuple[str, str, str, float]] = None,
        fixed: bool = False,
        shared: bool = False,
        #linear_param: bool = None 
    ):
        self.name = name
        if isinstance(value, (jnp.ndarray, list, tuple)):
            self.value = jnp.array(value, dtype=jnp.float32)
        else:
            self.value = float(value)

        self.min = jnp.float32(min)
        self.max = jnp.float32(max)
        self.tie = tie
        self.fixed = bool(fixed)
        self.shared = bool(shared)
        self.linear_param = False
        if "amplitude" in self.name or "weight" in self.name:#maybe is best change weight for amplitude? 
            self.linear_param = True
        if math.isfinite(self.min) and math.isfinite(self.max):
            self.transform = "logistic"
        elif math.isfinite(self.min):
            self.transform = "lower_bound_square"
        elif math.isfinite(self.max):
            self.transform = "upper_bound_square"
        else:
            self.transform = "linear"


class Parameters:
    r"""
    Container for managing a list of `Parameter` instances.

    Supports:
    - fixed params (excluded from optimization)
    - tied params (computed from sources)
    - bounded transforms (logistic / square / linear)
    - batched parameters (values as arrays)
    - shared free parameters (shared=True) as global hyper-parameters for batched fits,
      with packing/unpacking handled internally.

    Notes
    -----
    - If *any* parameter has an array value with shape (n_spec,), we treat the container as batched.
    - In shared-mode (at least one free shared parameter), raw space is a packed 1D vector:
        [raw_shared..., raw_local(spec0)... raw_local(spec1)...]
      and `raw_to_phys` returns a full physical array of shape (n_spec, n_total).

    - IMPORTANT: In shared-mode, `raw_to_phys` broadcasts shared parameters across spectra in the
      returned `phys` array, so `phys[:, idx_shared]` is always shape (n_spec,) and can be fed into
      `vmap(lambda A,m,s: ...)` directly (shared means identical values across the batch).
    """

    def __init__(self):
        self._list: List[Parameter] = []

        self._jit_raw_to_phys = None
        self._jit_phys_to_raw = None
        self._jit_raw_to_phys_mixed = None
        self._jit_phys_to_raw_mixed = None

        # caches set in _finalize
        self._raw_list = None
        self._tied_list = None
        self._fixed_list = None
        self._raw_shared_list = None
        self._raw_local_list = None
        self._n_spectra = None

    # -------------------------
    # mutation / cache invalidation
    # -------------------------
    def add(
        self,
        name: str,
        value: Union[float, jnp.ndarray, List[float], Tuple[float, ...]],
        *,
        min: Optional[float] = None,
        max: Optional[float] = None,
        tie: Optional[Tuple[str, str, str, float]] = None,
        fixed: bool = False,
        shared: bool = False,
    ):
        lo = -jnp.inf if min is None else float(min)
        hi =  jnp.inf if max is None else float(max)

        self._list.append(
            Parameter(
                name=name,
                value=value,
                min=lo,
                max=hi,
                tie=tie,
                fixed=fixed,
                shared=shared,
            )
        )
        self._invalidate()

    def _invalidate(self):
        self._jit_raw_to_phys = None
        self._jit_phys_to_raw = None
        self._jit_raw_to_phys_mixed = None
        self._jit_phys_to_raw_mixed = None

        self._raw_list = None
        self._tied_list = None
        self._fixed_list = None
        self._raw_shared_list = None
        self._raw_local_list = None
        self._n_spectra = None

    # -------------------------
    # shape inference / finalize
    # -------------------------
    def _infer_n_spectra(self) -> int:
        lens = []
        for p in self._list:
            v = p.value
            if isinstance(v, jnp.ndarray) and v.ndim > 0:
                lens.append(int(v.shape[0]))
        if not lens:
            return 1
        if len(set(lens)) != 1:
            raise ValueError(f"Inconsistent batch lengths in parameters: {sorted(set(lens))}")
        return lens[0]

    @property
    def names(self) -> List[str]:
        return [p.name for p in self._list]

    def _finalize(self):
        # partition params by role
        self._raw_list   = [p for p in self._list if p.tie is None and not p.fixed]
        self._tied_list  = [p for p in self._list if p.tie is not None and not p.fixed]
        self._fixed_list = [p for p in self._list if p.fixed]

        # split free params into shared vs local
        self._raw_shared_list = [p for p in self._raw_list if p.shared]
        self._raw_local_list  = [p for p in self._raw_list if not p.shared]

        # infer batch size from ANY vector-valued parameter
        self._n_spectra = self._infer_n_spectra()

        # validations that prevent confusing behavior
        if self._n_spectra == 1 and len(self._raw_shared_list) > 0:
            # shared free params in a single-spectrum run is not harmful,
            # but "mixed packing" would be pointless; still allow it.
            pass

        # JIT compile both "classic" and "mixed" cores
        self._jit_raw_to_phys = jax.jit(self._raw_to_phys_core)
        self._jit_phys_to_raw = jax.jit(self._phys_to_raw_core)
        self._jit_raw_to_phys_mixed = jax.jit(self._raw_to_phys_core_mixed)
        self._jit_phys_to_raw_mixed = jax.jit(self._phys_to_raw_core_mixed)

    def _ensure_finalized(self):
        if self._jit_raw_to_phys is None:
            self._finalize()

    def _has_shared_free(self) -> bool:
        self._ensure_finalized()
        return len(self._raw_shared_list) > 0

    def _mixed_sizes(self) -> Tuple[int, int, int]:
        """
        Returns (n_spec, n_shared_free, n_local_free) for shared-mode.
        """
        self._ensure_finalized()
        n_spec   = int(self._n_spectra) if self._n_spectra is not None else 1
        n_shared = len(self._raw_shared_list)
        n_local  = len(self._raw_local_list)
        return n_spec, n_shared, n_local

    # -------------------------
    # transforms (helpers)
    # -------------------------
    @staticmethod
    def _apply_transform(p: Parameter, rv: jnp.ndarray) -> jnp.ndarray:
        if p.transform == "logistic":
            return p.min + (p.max - p.min) * jax.nn.sigmoid(rv)
        elif p.transform == "lower_bound_square":
            return p.min + rv**2
        elif p.transform == "upper_bound_square":
            return p.max - rv**2
        else:
            return rv

    @staticmethod
    def _inv_transform(p: Parameter, vv: jnp.ndarray) -> jnp.ndarray:
        if p.transform == "logistic":
            frac = (vv - p.min) / (p.max - p.min)
            frac = jnp.clip(frac, 1e-6, 1 - 1e-6)
            return jnp.log(frac / (1 - frac))
        elif p.transform == "lower_bound_square":
            return jnp.sqrt(jnp.maximum(vv - p.min, 0))
        elif p.transform == "upper_bound_square":
            return jnp.sqrt(jnp.maximum(p.max - vv, 0))
        else:
            return vv

    # -------------------------
    # public API
    # -------------------------
    def raw_init(self) -> jnp.ndarray:
        """
        Initial raw vector from stored physical values.
        """
        self._ensure_finalized()
        init_phys = self.phys_init()

        # Dispatch to the right inverse-map
        if not self._has_shared_free():
            return self._jit_phys_to_raw(init_phys)

        if init_phys.ndim == 1:
            init_phys = init_phys[None, :]
        return self._jit_phys_to_raw_mixed(init_phys)
    
    def phys_init(self) -> jnp.ndarray:
        """
        Build the initial physical parameter array from stored `Parameter.value`.

        Returns
        -------
        jnp.ndarray
            - If n_spec == 1: shape (n_total,)
            - If n_spec > 1:  shape (n_spec, n_total)

        Notes
        -----
        - Scalar values are broadcast across spectra.
        - Vector values (shape (n_spec,)) are taken per spectrum.
        """
        self._ensure_finalized()
        n_spec = int(self._n_spectra)

        if n_spec == 1:
            return jnp.array([p.value for p in self._list])

        init_phys_list = []
        for i in range(n_spec):
            spec_values = []
            for p in self._list:
                v = p.value
                if isinstance(v, jnp.ndarray) and v.ndim > 0:
                    spec_values.append(v[i])
                else:
                    spec_values.append(v)
            init_phys_list.append(jnp.array(spec_values))
        return jnp.stack(init_phys_list)  # (n_spec, n_total)
    
    def raw_to_phys(self, raw_params: jnp.ndarray) -> jnp.ndarray:
        """
        Convert raw parameter vector(s) to physical space.

        - No shared free params: classic behavior.
        - Shared free params: expects packed 1D raw and returns (n_spec, n_total),
          with shared params broadcast across spectra in the returned `phys`.
        """
        self._ensure_finalized()
        if not self._has_shared_free():
            return self._jit_raw_to_phys(raw_params)
        return self._jit_raw_to_phys_mixed(raw_params)

    def phys_to_raw(self, phys_params: jnp.ndarray) -> jnp.ndarray:
        """
        Convert physical parameter vector(s) to raw space.

        - No shared free params: classic behavior.
        - Shared free params: expects (n_spec, n_total) and returns packed 1D raw.
        """
        self._ensure_finalized()
        if not self._has_shared_free():
            return self._jit_phys_to_raw(phys_params)

        if phys_params.ndim == 1:
            phys_params = phys_params[None, :]
        return self._jit_phys_to_raw_mixed(phys_params)


    def _raw_to_phys_core(self, raw: jnp.ndarray) -> jnp.ndarray:
        def convert_one(r_vec, spec_idx):
            ctx: Dict[str, jnp.ndarray] = {}
            idx = 0

            for p in self._raw_list:
                ctx[p.name] = self._apply_transform(p, r_vec[idx])
                idx += 1

            for p in self._fixed_list:
                v = p.value
                ctx[p.name] = v[spec_idx] if isinstance(v, jnp.ndarray) else v

            op_map = {"*": jnp.multiply, "+": jnp.add, "-": jnp.subtract, "/": jnp.divide}
            for p in self._tied_list:
                tgt, src, op, operand = p.tie
                ctx[tgt] = op_map[op](ctx[src], operand)

            return jnp.stack([ctx[p.name] for p in self._list])

        if raw.ndim == 1:
            return convert_one(raw, 0)
        else:
            N = raw.shape[0]
            idxs = jnp.arange(N)
            return jax.vmap(convert_one, in_axes=(0, 0))(raw, idxs)

    def _phys_to_raw_core(self, phys: jnp.ndarray) -> jnp.ndarray:
        def invert_one(v_vec):
            ctx = {p.name: v_vec[i] for i, p in enumerate(self._list)}
            raws: List[jnp.ndarray] = []
            for p in self._raw_list:
                raws.append(self._inv_transform(p, ctx[p.name]))
            return jnp.stack(raws)

        if phys.ndim == 1:
            return invert_one(phys)
        else:
            return jax.vmap(invert_one)(phys)

    # -------------------------
    # mixed cores (shared + local packed raw, internal)
    # -------------------------
    def _raw_to_phys_core_mixed(self, raw_packed: jnp.ndarray) -> jnp.ndarray:
        """
        Shared-mode raw->phys.

        Input
        -----
        raw_packed : shape (n_shared + n_spec*n_local,)

        Output
        ------
        phys : shape (n_spec, n_total)

        Guarantee
        ---------
        In the returned `phys`, any shared parameter column is broadcast to all spectra,
        so `phys[:, idx_shared]` has shape (n_spec,) and identical values.
        """
        n_spec, n_shared, n_local = self._mixed_sizes()

        if n_shared == 0:
            return self._raw_to_phys_core(raw_packed)

        raw_shared = raw_packed[:n_shared]                       # (n_shared,)
        raw_local  = raw_packed[n_shared:].reshape(n_spec, n_local)  # (n_spec, n_local)

        # Compute shared phys scalars once, then broadcast when building each spectrum row
        shared_vals: Dict[str, jnp.ndarray] = {}
        for j, p in enumerate(self._raw_shared_list):
            v = self._apply_transform(p, raw_shared[j])          # scalar
            shared_vals[p.name] = v

        op_map = {"*": jnp.multiply, "+": jnp.add, "-": jnp.subtract, "/": jnp.divide}

        def convert_one(r_loc, spec_idx):
            ctx: Dict[str, jnp.ndarray] = {}

            # shared free params: scalar in ctx for this spectrum row
            for p in self._raw_shared_list:
                ctx[p.name] = shared_vals[p.name]

            # local free params for this spec
            for j, p in enumerate(self._raw_local_list):
                ctx[p.name] = self._apply_transform(p, r_loc[j])

            # fixed params (may be per-spec arrays)
            for p in self._fixed_list:
                v = p.value
                ctx[p.name] = v[spec_idx] if isinstance(v, jnp.ndarray) else v

            # tied params
            for p in self._tied_list:
                tgt, src, op, operand = p.tie
                ctx[tgt] = op_map[op](ctx[src], operand)

            return jnp.stack([ctx[p.name] for p in self._list])

        idxs = jnp.arange(n_spec)
        phys = jax.vmap(convert_one, in_axes=(0, 0))(raw_local, idxs)  # (n_spec, n_total)

        # --- broadcast shared columns explicitly to guarantee `phys[:, idx_shared]` is (n_spec,)
        # and identical across rows (even if later logic changes).
        if self._raw_shared_list:
            for p in self._raw_shared_list:
                col = self.names.index(p.name)
                phys = phys.at[:, col].set(jnp.full((n_spec,), shared_vals[p.name]))
        return phys

    def _phys_to_raw_core_mixed(self, phys: jnp.ndarray) -> jnp.ndarray:
        """
        Shared-mode phys->raw (inverse of packed layout).
        """
        n_spec, n_shared, n_local = self._mixed_sizes()

        if n_shared == 0:
            return self._phys_to_raw_core(phys)

        def ctx_from_phys(v_vec):
            return {p.name: v_vec[i] for i, p in enumerate(self._list)}

        # shared raws: read from first spectrum
        ctx0 = ctx_from_phys(phys[0])
        shared_raws = []
        for p in self._raw_shared_list:
            shared_raws.append(self._inv_transform(p, ctx0[p.name]))
        shared_raws = jnp.stack(shared_raws) if shared_raws else jnp.zeros((0,), dtype=phys.dtype)

        # local raws per spectrum
        def local_raws_one(v_vec):
            ctx = ctx_from_phys(v_vec)
            raws = []
            for p in self._raw_local_list:
                raws.append(self._inv_transform(p, ctx[p.name]))
            return jnp.stack(raws) if raws else jnp.zeros((0,), dtype=phys.dtype)

        local_raws = jax.vmap(local_raws_one)(phys)  # (n_spec, n_local)
        return jnp.concatenate([shared_raws, local_raws.ravel()])
    def linear_phys_indices(self) -> jnp.ndarray:
        """
        Return the indices in physical parameter space corresponding to linear parameters.

        Physical-space indices always follow the full parameter ordering in `self._list`,
        so fixed and tied parameters are included if they are marked as linear.

        Returns
        -------
        jnp.ndarray
            Integer array with the indices of linear parameters in the physical vector.
        """
        return jnp.array(
            [i for i, p in enumerate(self._list) if p.linear_param],
            dtype=jnp.int32,
        )

    def nonlinear_raw_indices(self) -> jnp.ndarray:
        """
        Return the indices in raw optimization space corresponding to non-linear free parameters.

        Notes
        -----
        - Raw space only contains free parameters (`tie is None` and `fixed is False`).
        - Parameters with `linear_param=True` are excluded.
        - In classic mode (no shared free parameters), raw indices follow `self._raw_list`.
        - In shared-mode, raw packing is:
              [raw_shared..., raw_local(spec0)... raw_local(spec1)...]
          so local non-linear indices are repeated for each spectrum.
        - Shared non-linear parameters appear only once at the beginning of the packed raw vector.

        Returns
        -------
        jnp.ndarray
            Integer array with the indices of non-linear free parameters in raw space.
        """
        self._ensure_finalized()

        # Classic mode
        if not self._has_shared_free():
            return jnp.array(
                [i for i, p in enumerate(self._raw_list) if not p.linear_param],
                dtype=jnp.int32,
            )

        # Shared/local packed mode
        n_spec, n_shared, n_local = self._mixed_sizes()

        shared_idx = [
            i
            for i, p in enumerate(self._raw_shared_list)
            if not p.linear_param
        ]

        local_base_idx = [
            i
            for i, p in enumerate(self._raw_local_list)
            if not p.linear_param
        ]

        local_idx = []
        offset0 = n_shared
        for k in range(n_spec):
            offset = offset0 + k * n_local
            local_idx.extend([offset + i for i in local_base_idx])

        return jnp.array(shared_idx + local_idx, dtype=jnp.int32)
    # -------------------------
    # convenience
    # -------------------------
    @property
    def specs(self) -> List[Tuple[str, Any, float, float, str, bool, bool]]:
        """
        (name, value, min, max, transform, fixed, shared)
        """
        return [(p.name, p.value, p.min, p.max, p.transform, p.fixed, p.shared,p.linear_param) for p in self._list]

    @property
    def summary(self) -> List[Dict[str, Any]]:
        """
        User-facing summary of parameters and definitions.
        """
        self._ensure_finalized()

        rows: List[Dict[str, Any]] = []
        for p in self._list:
            if p.fixed:
                status = "fixed"
            elif p.tie is not None:
                status = "tied"
            else:
                status = "free"

            v = p.value
            if isinstance(v, jnp.ndarray):
                v_out: Any = {
                    "shape": tuple(v.shape),
                    "dtype": str(v.dtype),
                    "preview": v[:5] if (v.ndim == 1 and v.size > 5) else v,
                }
            else:
                v_out = float(v)

            rows.append(
                {
                    "name": p.name,
                    "value": v_out,
                    "min": float(p.min),
                    "max": float(p.max),
                    "transform": p.transform,
                    "fixed": bool(p.fixed),
                    "shared": bool(p.shared),
                    "tie": p.tie,
                    "status": status,
                    "linear_param" :p.linear_param
                }
            )
        return rows



def build_Parameters(
    tied_map: Dict[int, Tuple[int, str, float]],
    params_dict: Dict[str, int],
    initial_params: Iterable[float],
    constraints: jnp.ndarray,
) -> Parameters:
    r"""
    Construct a :class:`Parameters` object from initialization arrays, constraints,
    and tie definitions.

    This helper builds a container of :class:`Parameter` instances ready for fitting,
    applying bounds, fixed values, and tied relationships.

    Parameters
    ----------
    tied_map : dict[int, tuple[int, str, float]]
        Mapping of parameter indices to tie definitions.
        Each entry is of the form
        ``idx_target -> (idx_source, op, operand)``, where:
          * ``idx_target`` is the index of the tied parameter,
          * ``idx_source`` is the index of the source parameter,
          * ``op`` is an arithmetic operator string (``'*'``, ``'/'``, ``'+'``, ``'-'``),
          * ``operand`` is a numeric factor or offset.
    params_dict : dict[str, int]
        Dictionary mapping parameter names to their index positions
        in the parameter vector.
    initial_params : array-like, shape (n_params,)
        Initial physical parameter values.
    constraints : array-like, shape (n_params, 2)
        Lower and upper bounds per parameter.

    Returns
    -------
    Parameters
        A populated container with names, values, bounds, and tie definitions.

    Notes
    -----
    * Tied parameters are added with their ``tie`` attribute and are not optimized
        directly; their values are reconstructed from the source parameter during
        raw→physical mapping.
    * Untied parameters are added with their initial value and bounds taken from
        ``constraints``.
    * Fixed parameters are not assigned here; add them via
        :meth:`Parameters.add(..., fixed=True) <Parameters.add>` if needed.
    * Typically called by higher-level fitting routines (e.g., :class:`RegionFitting`)
        when preparing parameter sets.
    """
    params_obj = Parameters()
    for name, idx in params_dict.items():
        val = jnp.atleast_2d(initial_params)[:,idx]
        min_val, max_val = constraints[idx]
        
        if idx in tied_map.keys():
            src_idx, op, operand = tied_map[idx]
            src_name = list(params_dict.keys())[src_idx]
            tie = (name, src_name, op, operand)
            params_obj.add(name, val, min=min_val, max=max_val, tie=tie)
        else:
            params_obj.add(name, val, min=min_val, max=max_val)
    params_obj.n_obj = initial_params.shape[0]
    return params_obj