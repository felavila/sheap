from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import jax.numpy as jnp
import jax
import math




class Parameter:
    """
    Represents a fit parameter with optional bounds or ties, plus a transform
    determined by its min/max, or held fixed.
    """
    def __init__(
        self,
        name: str,
        value: float,
        *,
        min: float = -jnp.inf,
        max: float = jnp.inf,
        tie: Optional[Tuple[str, str, str, float]] = None,
        fixed: bool = False,
    ):
        self.name  = name
        self.value = float(value)
        self.min   = float(min)
        self.max   = float(max)
        self.tie   = tie   # (target, source, op, operand)
        self.fixed = fixed

        # Choose transform based on bounds (ignored if fixed=True)
        if math.isfinite(self.min) and math.isfinite(self.max):
            self.transform = 'logistic'
        elif math.isfinite(self.min):
            self.transform = 'lower_bound_square'
        elif math.isfinite(self.max):
            self.transform = 'upper_bound_square'
        else:
            self.transform = 'linear'


class Parameters:
    def __init__(self):
        self._list = []                 # all parameters
        self._jit_raw_to_phys = None
        self._jit_phys_to_raw = None

    def add(
        self,
        name: str,
        value: float,
        *,
        min: Optional[float] = None,
        max: Optional[float] = None,
        tie: Optional[Tuple[str, str, str, float]] = None,
        fixed: bool = False,
    ):
        lo = -jnp.inf if min is None else min
        hi =  jnp.inf if max is None else max
        self._list.append(Parameter(
            name=name, value=value, min=lo, max=hi,
            tie=tie, fixed=fixed
        ))
        # Invalidate compiled kernels
        self._jit_raw_to_phys = None
        self._jit_phys_to_raw = None

    @property
    def names(self) -> List[str]:
        return [p.name for p in self._list]

    def _finalize(self):
        # free (untied, unfixed) params go into the raw vector
        self._raw_list   = [p for p in self._list if p.tie is None   and not p.fixed]
        # tied params are computed from others
        self._tied_list  = [p for p in self._list if p.tie is not None and not p.fixed]
        # fixed params sit out of transforms entirely
        self._fixed_list = [p for p in self._list if p.fixed]

        # compile
        self._jit_raw_to_phys = jax.jit(self._raw_to_phys_core)
        self._jit_phys_to_raw = jax.jit(self._phys_to_raw_core)

    def raw_init(self) -> jnp.ndarray:
        if self._jit_phys_to_raw is None:
            self._finalize()
        init_phys = jnp.array([p.value for p in self._raw_list])
        return self._jit_phys_to_raw(init_phys)

    def raw_to_phys(self, raw_params: jnp.ndarray) -> jnp.ndarray:
        if self._jit_raw_to_phys is None:
            self._finalize()
        return self._jit_raw_to_phys(raw_params)

    def phys_to_raw(self, phys_params: jnp.ndarray) -> jnp.ndarray:
        if self._jit_phys_to_raw is None:
            self._finalize()
        return self._jit_phys_to_raw(phys_params)

    def _raw_to_phys_core(self, raw: jnp.ndarray) -> jnp.ndarray:
        def convert_one(r_vec):
            ctx: Dict[str, jnp.ndarray] = {}
            idx = 0

            # 1) free params from raw → phys
            for p in self._raw_list:
                rv = r_vec[idx]
                if p.transform == 'logistic':
                    val = p.min + (p.max - p.min) * jax.nn.sigmoid(rv)
                elif p.transform == 'lower_bound_square':
                    val = p.min + rv**2
                elif p.transform == 'upper_bound_square':
                    val = p.max - rv**2
                else:
                    val = rv
                ctx[p.name] = val
                idx += 1

            # 2) tied params
            op_map = {'*': jnp.multiply, '+': jnp.add,
                      '-': jnp.subtract, '/': jnp.divide}
            for p in self._tied_list:
                tgt, src, op, operand = p.tie
                ctx[tgt] = op_map[op](ctx[src], operand)

            # 3) fixed params
            for p in self._fixed_list:
                ctx[p.name] = p.value

            # stack in original order
            return jnp.stack([ctx[p.name] for p in self._list])

        if raw.ndim == 1:
            return convert_one(raw)
        else:
            return jax.vmap(convert_one)(raw)

    def _phys_to_raw_core(self, phys: jnp.ndarray) -> jnp.ndarray:
        def invert_one(v_vec):
            raws: List[jnp.ndarray] = []
            idx = 0

            # only invert free params
            for p in self._raw_list:
                vv = v_vec[idx]
                if p.transform == 'logistic':
                    frac = (vv - p.min) / (p.max - p.min)
                    frac = jnp.clip(frac, 1e-6, 1 - 1e-6)
                    raws.append(jnp.log(frac / (1 - frac)))
                elif p.transform == 'lower_bound_square':
                    raws.append(jnp.sqrt(jnp.maximum(vv - p.min, 0)))
                elif p.transform == 'upper_bound_square':
                    raws.append(jnp.sqrt(jnp.maximum(p.max - vv, 0)))
                else:
                    raws.append(vv)
                idx += 1

            return jnp.stack(raws)

        if phys.ndim == 1:
            return invert_one(phys)
        else:
            return jax.vmap(invert_one)(phys)

    @property
    def specs(self) -> List[Tuple[str, float, float, float, str, bool]]:
        # name, init, min, max, transform, fixed
        return [
            (p.name, p.value, p.min, p.max, p.transform, p.fixed)
            for p in self._list
        ]