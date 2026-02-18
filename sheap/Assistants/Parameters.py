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


class Parameter:
    """
    Represents a single fit parameter with optional bounds, ties, fixed status,
    and an optional 'shared' flag for global (hyper-)parameters across a batch.
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
    ):
        self.name = name

        # allow scalar or array initial values
        if isinstance(value, (jnp.ndarray, list, tuple)):
            self.value = jnp.array(value)
        else:
            self.value = float(value)

        self.min = float(min)
        self.max = float(max)
        self.tie = tie  # (target, source, op, operand)
        self.fixed = bool(fixed)
        self.shared = bool(shared)

        # Choose transform based on bounds (ignored if fixed=True in mapping)
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
    - NEW: shared free parameters (shared=True) as global hyper-parameters for batched fits,
      with packing/unpacking handled internally.
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

    def add(
        self,
        name: str,
        value: Union[float, jnp.ndarray, List[float], Tuple[float, ...]],
        *,
        min: Optional[float] = None,
        max: Optional[float] = None,
        tie: Optional[Tuple[str, str, str, float]] = None,
        fixed: bool = False,
        shared: bool = False
    ):
        lo = -jnp.inf if min is None else min
        hi = jnp.inf if max is None else max

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
        self._raw_list = [p for p in self._list if p.tie is None and not p.fixed]
        self._tied_list = [p for p in self._list if p.tie is not None and not p.fixed]
        self._fixed_list = [p for p in self._list if p.fixed]

        # split free params into shared vs local
        self._raw_shared_list = [p for p in self._raw_list if getattr(p, "shared", False)]
        self._raw_local_list = [p for p in self._raw_list if not getattr(p, "shared", False)]

        # infer n_spectra from ANY vector-valued param (fixed or free), but in shared-mode
        # we require at least one local param to be vector-valued.
        self._n_spectra = self._infer_n_spectra() #?
        for p in self._list:
            v = p.value
            if isinstance(v, jnp.ndarray) and v.ndim > 0:
                self._n_spectra = int(v.shape[0])
                break

        # jit compile both "classic" and "mixed" cores
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
        n_shared = len(self._raw_shared_list)
        n_local = len(self._raw_local_list)
        n_spec = self._n_spectra if self._n_spectra is not None else 1
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
        if self._jit_phys_to_raw is None:
            self._finalize()

        n_spec = self._n_spectra

        if n_spec == 1:
            init_phys = jnp.array([p.value for p in self._list])
        else:
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
            init_phys = jnp.stack(init_phys_list)

        if not self._has_shared_free():
            return self._jit_phys_to_raw(init_phys)

        # shared-mode expects batch axis
        if init_phys.ndim == 1:
            init_phys = init_phys[None, :]
        return self._jit_phys_to_raw_mixed(init_phys)


    def raw_to_phys(self, raw_params: jnp.ndarray) -> jnp.ndarray:
        """
        Convert raw parameter vector(s) to physical space.

        - no shared free params:
            same as original.

        - shared free params exist:
            expects packed 1D raw and returns (n_spec, n_total).
        """
        self._ensure_finalized()

        if not self._has_shared_free():
            return self._jit_raw_to_phys(raw_params)

        # shared-mode: accept packed 1D raw
        return self._jit_raw_to_phys_mixed(raw_params)

    def phys_to_raw(self, phys_params: jnp.ndarray) -> jnp.ndarray:
        """
        Convert physical parameter vector(s) to raw space.

        - no shared free params:
            same as original.

        - shared free params exist:
            returns packed 1D raw.
        """
        self._ensure_finalized()

        if not self._has_shared_free():
            return self._jit_phys_to_raw(phys_params)

        if phys_params.ndim == 1:
            phys_params = phys_params[None, :]
        return self._jit_phys_to_raw_mixed(phys_params)

    # -------------------------
    # classic cores (your original behavior)
    # -------------------------
    def _raw_to_phys_core(self, raw: jnp.ndarray) -> jnp.ndarray:
        def convert_one(r_vec, spec_idx):
            ctx: Dict[str, jnp.ndarray] = {}
            idx = 0

            for p in self._raw_list:
                rv = r_vec[idx]
                ctx[p.name] = self._apply_transform(p, rv)
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
            raws: List[jnp.ndarray] = []
            ctx = {p.name: v_vec[i] for i, p in enumerate(self._list)}

            for p in self._raw_list:
                vv = ctx[p.name]
                raws.append(self._inv_transform(p, vv))

            return jnp.stack(raws)

        if phys.ndim == 1:
            return invert_one(phys)
        else:
            return jax.vmap(invert_one)(phys)

    # -------------------------
    # mixed cores (shared + local packed raw, internal)
    # -------------------------
    def _raw_to_phys_core_mixed(self, raw_packed: jnp.ndarray) -> jnp.ndarray:
        n_spec, n_shared, n_local = self._mixed_sizes()

        if n_shared == 0:
            # fall back (should not happen if dispatch is correct)
            return self._raw_to_phys_core(raw_packed)

        raw_shared = raw_packed[:n_shared]  # (n_shared,)
        raw_local = raw_packed[n_shared:].reshape(n_spec, n_local)  # (n_spec, n_local)

        # shared ctx once
        shared_ctx: Dict[str, jnp.ndarray] = {}
        for j, p in enumerate(self._raw_shared_list):
            shared_ctx[p.name] = self._apply_transform(p, raw_shared[j])

        op_map = {"*": jnp.multiply, "+": jnp.add, "-": jnp.subtract, "/": jnp.divide}

        def convert_one(r_loc, spec_idx):
            ctx = dict(shared_ctx)

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
        return jax.vmap(convert_one, in_axes=(0, 0))(raw_local, idxs)  # (n_spec, n_total)

    def _phys_to_raw_core_mixed(self, phys: jnp.ndarray) -> jnp.ndarray:
        # phys is (n_spec, n_total)
        n_spec, n_shared, n_local = self._mixed_sizes()

        if n_shared == 0:
            # fall back (should not happen if dispatch is correct)
            return self._phys_to_raw_core(phys)

        def ctx_from_phys(v_vec):
            return {p.name: v_vec[i] for i, p in enumerate(self._list)}

        # shared raws: read from first spectrum (shared params are identical conceptually)
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

    # -------------------------
    # optional convenience
    # -------------------------
    @property
    def specs(self) -> List[Tuple[str, Any, float, float, str, bool, bool]]:
        """
        (name, value, min, max, transform, fixed, shared)
        """
        return [(p.name, p.value, p.min, p.max, p.transform, p.fixed, p.shared) for p in self._list]

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
    
    return params_obj