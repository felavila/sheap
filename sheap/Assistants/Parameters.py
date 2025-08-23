"""
Parameters
=================

This module defines the `Parameter` and `Parameters` classes and a helper
`build_Parameters` to construct constraint‑aware parameter sets for fitting.

It handles:
- Reparameterization between optimizer (raw) and physical spaces
- Per‑parameter bounds, fixed values, and arithmetic ties
- Batched evaluation for multiple spectra via JAX vmap

Notes
-----
- Tied parameters are reconstructed from their sources during raw→physical mapping.
- Only *free* (untied, unfixed) parameters live in the raw vector.
"""

__author__ = 'felavila'


__all__ = [
    "Parameter",
    "Parameters",
    "build_Parameters",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union,Iterable


import jax.numpy as jnp
import jax
import math


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp
    default_inf = jnp.inf
else:
    default_inf = float("inf")

class Parameter:
    """
    Represents a single fit parameter with optional bounds, ties, and fixed status.

    This class encapsulates metadata about the parameter, including transformation
    rules for optimization based on bounds or constraints.

    Attributes
    ----------
    name : str
        Name of the parameter (e.g., "amplitude_Halpha_1_broad").
    value : float or jnp.ndarray
        Initial value(s) for the parameter. Can be scalar or array.
    min : float
        Lower bound for the parameter.
    max : float
        Upper bound for the parameter.
    tie : tuple, optional
        A tuple specifying a tied relationship (target, source, operation, operand).
    fixed : bool
        If True, the parameter is excluded from optimization.
    transform : str
        Type of transformation used: 'logistic', 'lower_bound_square',
        'upper_bound_square', or 'linear'.
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
    ):
        self.name  = name
        # allow scalar or array initial values for fixed parameters
        if isinstance(value, (jnp.ndarray, list, tuple)):
            self.value = jnp.array(value)
        else:
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
    """
    Container for managing a list of `Parameter` instances for fitting models.

    This class handles the declaration, transformation, and synchronization
    between raw and physical parameter spaces. It supports automatic handling
    of fixed, tied, and bounded parameters, including vectorization with `vmap`.

    Attributes
    ----------
    _list : list of Parameter
        All declared parameters in order of definition.
    _jit_raw_to_phys : callable
        JIT-compiled function that maps raw parameters to physical space.
    _jit_phys_to_raw : callable
        JIT-compiled function that maps physical parameters to raw space.
    """

    def __init__(self):
        self._list = []
        self._jit_raw_to_phys = None
        self._jit_phys_to_raw = None

    def add(
        self,
        name: str,
        value: Union[float, jnp.ndarray, List[float], Tuple[float, ...]],
        *,
        min: Optional[float] = None,
        max: Optional[float] = None,
        tie: Optional[Tuple[str, str, str, float]] = None,
        fixed: bool = False,
    ):
        """
        Add a parameter to the collection.

        Parameters
        ----------
        name : str
            Name of the parameter.
        value : float or array-like
            Initial value.
        min : float, optional
            Lower bound; defaults to -inf if not set.
        max : float, optional
            Upper bound; defaults to +inf if not set.
        tie : tuple, optional
            Constraint as a tuple (target, source, op, operand).
        fixed : bool, default=False
            Whether the parameter is fixed during fitting.
        """
        lo = -jnp.inf if min is None else min
        hi = jnp.inf if max is None else max
        self._list.append(Parameter(
            name=name, value=value, min=lo, max=hi,
            tie=tie, fixed=fixed
        ))
        self._jit_raw_to_phys = None
        self._jit_phys_to_raw = None

    @property
    def names(self) -> List[str]:
        """
        Names of all parameters in declaration order.

        Returns
        -------
        List[str]
            Parameter names.
        """
        return [p.name for p in self._list]

    def _finalize(self):
        self._raw_list = [p for p in self._list if p.tie is None and not p.fixed]
        self._tied_list = [p for p in self._list if p.tie is not None and not p.fixed]
        self._fixed_list = [p for p in self._list if p.fixed]
        self._jit_raw_to_phys = jax.jit(self._raw_to_phys_core)
        self._jit_phys_to_raw = jax.jit(self._phys_to_raw_core)

    def raw_init(self) -> jnp.ndarray:
        """
        Generate the initial raw parameter vector from physical values.

        Returns
        -------
        jnp.ndarray
            Raw parameter array suitable for optimization.
        """
        if self._jit_phys_to_raw is None:
            self._finalize()
        init_phys = jnp.array([p.value for p in self._raw_list])
        return self._jit_phys_to_raw(init_phys)

    def raw_to_phys(self, raw_params: jnp.ndarray) -> jnp.ndarray:
        """
        Convert raw parameter vector(s) to physical space.

        Parameters
        ----------
        raw_params : jnp.ndarray
            Raw input array of shape (n_params,) or (n_samples, n_params).

        Returns
        -------
        jnp.ndarray
            Corresponding physical parameter array(s).
        """
        if self._jit_raw_to_phys is None:
            self._finalize()
        return self._jit_raw_to_phys(raw_params)

    def phys_to_raw(self, phys_params: jnp.ndarray) -> jnp.ndarray:
        """
        Convert physical parameter vector(s) to raw space.

        Parameters
        ----------
        phys_params : jnp.ndarray
            Physical input array.

        Returns
        -------
        jnp.ndarray
            Raw parameter array suitable for optimization.
        """
        if self._jit_phys_to_raw is None:
            self._finalize()
        return self._jit_phys_to_raw(phys_params)

    def _raw_to_phys_core(self, raw: jnp.ndarray) -> jnp.ndarray:
        """
        Convert from raw vector(s) to full physical parameter vector(s).

        Handles transformation of free, tied, and fixed parameters and returns
        them in the original declaration order.

        Parameters
        ----------
        raw : jnp.ndarray
            Raw parameter array(s), shape (n_free,) or (n_batch, n_free).

        Returns
        -------
        jnp.ndarray
            Physical parameters in full vector form, shape (n_total,) or (n_batch, n_total).
        """
        def convert_one(r_vec, spec_idx):
            ctx: Dict[str, jnp.ndarray] = {}
            idx = 0
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
            op_map = {'*': jnp.multiply, '+': jnp.add, '-': jnp.subtract, '/': jnp.divide}
            for p in self._tied_list:
                tgt, src, op, operand = p.tie
                ctx[tgt] = op_map[op](ctx[src], operand)
            for p in self._fixed_list:
                v = p.value
                ctx[p.name] = v[spec_idx] if isinstance(v, jnp.ndarray) else v
            return jnp.stack([ctx[p.name] for p in self._list])

        if raw.ndim == 1:
            return convert_one(raw, 0)
        else:
            N = raw.shape[0]
            idxs = jnp.arange(N)
            return jax.vmap(convert_one, in_axes=(0, 0))(raw, idxs)

    def _phys_to_raw_core(self, phys: jnp.ndarray) -> jnp.ndarray:
        """
        Inverse mapping from physical to raw parameter space.

        Parameters
        ----------
        phys : jnp.ndarray
            Physical parameter array(s), shape (n_total,) or (n_batch, n_total).

        Returns
        -------
        jnp.ndarray
            Corresponding raw parameter array(s), shape (n_free,) or (n_batch, n_free).
        """
        def invert_one(v_vec):
            raws: List[jnp.ndarray] = []
            idx = 0
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
        """
        Get summary of each parameter's definition.

        Returns
        -------
        List[Tuple[str, float, float, float, str, bool]]
            Each entry contains (name, value, min, max, transform, fixed).
        """
        return [
            (p.name, p.value, p.min, p.max, p.transform, p.fixed)
            for p in self._list
        ]



def build_Parameters(
    tied_map: Dict[int, Tuple[int, str, float]],
    params_dict: Dict[str, int],
    initial_params: Iterable[float],
    constraints: jnp.ndarray,
) -> Parameters:
    """
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
        min,max = constraints[idx]
        #if name in ["amplitude_slope_linear_0_continuum","amplitude_intercept_linear_0_continuum"] and iteration_number==10:
         #   params_obj.add(name, val, fixed=True)
        if idx in tied_map.keys():
            src_idx, op, operand = tied_map[idx]
            src_name = list(params_dict.keys())[src_idx]
            tie = (name, src_name, op, operand)
            params_obj.add(name, val, min=min, max=max, tie=tie)
        else:
            val = initial_params[idx]
            min,max = constraints[idx]
            params_obj.add(name, val, min=min, max=max)
    return params_obj




