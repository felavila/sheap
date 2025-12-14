r"""
Params Class 
==========

This module defines the :class:`Parameter` and :class:`Parameters`
classes and a helper :func:`build_Parameters` to construct
constraint-aware parameter sets for fitting.

It handles
----------
- Reparameterization between optimizer (raw) and physical spaces
- Per-parameter bounds, fixed values, and arithmetic ties
- Batched evaluation for multiple spectra via JAX ``vmap``

Notes
-----
- Tied parameters are reconstructed from their sources during
    raw→physical mapping.
- Only **free** (untied, unfixed) parameters live in the raw vector.

"""

__author__ = 'felavila'


__all__ = [
    "Parameter",
    "Parameters",
    "build_Parameters",
]

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Iterable


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
        is_linear: Optional[bool] = None, 
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

        if is_linear is None:
            lname = name.lower()
            is_linear = (
                ("amp"    in lname) or
                ("weight" in lname)
            )
        self.is_linear = bool(is_linear)
        
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
    r"""
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

        self._linear_raw_idx = None
        self._nonlinear_raw_idx = None
        
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
        
        
        linear_raw_idx = []
        nonlinear_raw_idx = []
        for i, p in enumerate(self._raw_list):
            if getattr(p, "is_linear", False):
                linear_raw_idx.append(i)
            else:
                nonlinear_raw_idx.append(i)

        self._linear_raw_idx    = jnp.array(linear_raw_idx, dtype=int)
        self._nonlinear_raw_idx = jnp.array(nonlinear_raw_idx, dtype=int)
        
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
        
        # Check if we have batched parameters
        first_val = self._list[0].value
        if isinstance(first_val, jnp.ndarray) and first_val.ndim > 0:
            # Batched: each parameter has multiple values
            n_spectra = len(first_val)
            init_phys_list = []
            for i in range(n_spectra):
                spec_values = []
                for p in self._list:
                    if isinstance(p.value, jnp.ndarray):
                        spec_values.append(p.value[i])
                    else:
                        spec_values.append(p.value)
                init_phys_list.append(jnp.array(spec_values))
            init_phys = jnp.stack(init_phys_list)
        else:
            # Single spectrum
            init_phys = jnp.array([p.value for p in self._list])
        
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
            
            # First, process all raw (free) parameters
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
            
            # Then, process fixed parameters (they may have per-spectrum values)
            for p in self._fixed_list:
                v = p.value
                ctx[p.name] = v[spec_idx] if isinstance(v, jnp.ndarray) else v
            
            # Finally, process tied parameters (AFTER fixed parameters are in ctx)
            op_map = {'*': jnp.multiply, '+': jnp.add, '-': jnp.subtract, '/': jnp.divide}
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
            # Build a mapping from param name to value
            ctx = {p.name: v_vec[i] for i, p in enumerate(self._list)}
            
            # Extract only the raw (free) parameters
            for p in self._raw_list:
                vv = ctx[p.name]
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
            return jnp.stack(raws)

        if phys.ndim == 1:
            return invert_one(phys)
        else:
            return jax.vmap(invert_one)(phys)
        
    def split_raw_linear_nonlinear(self, raw: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Split raw parameter vector(s) into non-linear and linear parts.

        Parameters
        ----------
        raw : jnp.ndarray
            Raw parameter array of shape (n_free,) or (n_batch, n_free).

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            (raw_nonlinear, raw_linear)
        """
        if self._jit_raw_to_phys is None:
            self._finalize()

        lin_idx = self._linear_raw_idx
        nl_idx  = self._nonlinear_raw_idx

        if raw.ndim == 1:
            raw_lin = raw[lin_idx]
            raw_nl  = raw[nl_idx]
        else:
            raw_lin = raw[:, lin_idx]
            raw_nl  = raw[:, nl_idx]
        return raw_nl, raw_lin

    def merge_raw_linear_nonlinear(
        self, raw_nl: jnp.ndarray, raw_lin: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Merge separate non-linear and linear raw parameter vectors back into a single vector.

        Parameters
        ----------
        raw_nl : jnp.ndarray
            Non-linear raw parameters, shape (n_nl,) or (n_batch, n_nl).
        raw_lin : jnp.ndarray
            Linear raw parameters, shape (n_lin,) or (n_batch, n_lin).

        Returns
        -------
        jnp.ndarray
            Full raw parameter array, shape (n_free,) or (n_batch, n_free).
        """
        if self._jit_raw_to_phys is None:
            self._finalize()

        n_free = len(self._raw_list)
        lin_idx = self._linear_raw_idx
        nl_idx  = self._nonlinear_raw_idx

        if raw_nl.ndim == 1:
            full = jnp.zeros((n_free,), dtype=raw_nl.dtype)
            full = full.at[nl_idx].set(raw_nl)
            full = full.at[lin_idx].set(raw_lin)
        else:
            n_batch = raw_nl.shape[0]
            full = jnp.zeros((n_batch, n_free), dtype=raw_nl.dtype)
            full = full.at[:, nl_idx].set(raw_nl)
            full = full.at[:, lin_idx].set(raw_lin)
        return full

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
    
    @property
    def linear_raw_indices(self) -> jnp.ndarray:
        """Indices in the raw free-parameter vector that are linear-in-the-model."""
        if self._jit_raw_to_phys is None:
            self._finalize()
        return self._linear_raw_idx

    @property
    def nonlinear_raw_indices(self) -> jnp.ndarray:
        """Indices in the raw free-parameter vector that are non-linear-in-the-model."""
        if self._jit_raw_to_phys is None:
            self._finalize()
        return self._nonlinear_raw_idx

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