from typing import Any, Callable, Dict, List, Optional, Tuple, Union


import jax.numpy as jnp
import jax
import math


from sheap.DataClass.DataClass import ConstraintSet, FittingLimits, SpectralLine
from sheap.Tools.spectral_basic import kms_to_wl
from sheap.Functions.profiles import PROFILE_FUNC_MAP,PROFILE_LINE_FUNC_MAP,PROFILE_CONTINUUM_FUNC_MAP


# import jax
# import jax.numpy as jnp




CANONICAL_WAVELENGTHS = {
    'broad': 4861.0,    # Hbeta
    'narrow': 5007.0,   # [OIII]
    'outflow': 5007.0,  # [OIII]
    'fe': 4570.0,       # Mean FeII blend
    'nlr': 6583.0,       # [NII]
    "winds": 5007.0} #why?


DEFAULT_LIMITS = {
    'broad': dict(
        upper_fwhm=11775.0,  # FWHM ~ 1000–10000 km/s for broad lines
        lower_fwhm=1000.875,
        center_shift=5000.0,
        max_amplitude=10.0,
        # Ref: Sulentic+2000, Shen+2011
    ),
    'narrow': dict(
        upper_fwhm=471.0,   # FWHM ~ 200–1000 km/s typical for NLR
        lower_fwhm=117.75,
        center_shift=2500.0,
        max_amplitude=10.0,
        # Ref: Osterbrock & Ferland 2006, Véron-Cetty+2001
    ),
    #not sure about this 
    'outflow': dict(
        upper_fwhm=11775.0,   # FWHM for blueshifted or broad outflowing components
        lower_fwhm=5000.875,
        center_shift=3000.0,
        max_amplitude=10.0,
        # Ref: Bischetti+2017, Perrotta+2019
    ),
    'fe': dict(
        upper_fwhm=7065.0,   # Typical Fe II FWHM from 800 to 2500 km/s
        lower_fwhm=494.55,
        center_shift=2500.0,
        max_amplitude=0.07,
        # Ref: Kovačević+2010, Ilic+2022
    ),
    'nlr': dict(
        upper_fwhm=2355.0,   # NLR lines are narrow; similar to 'narrow' but possibly less broadened
        lower_fwhm=117.75,
        center_shift=1500.0,
        max_amplitude=10.0,),
        # Ref: Bennert+2006, Hainline+2013
    "winds": dict(upper_fwhm   = 15000.0,   # up to 15 000 km/s for very fast winds
    lower_fwhm   = 5000.0,    # minimum ~ 5 000 km/s
    center_shift = 8000.0,    # allow blueshifts up to ~8 000 km/s
    max_amplitude= 10.0,      # same cap as your other lines
        )
}


def make_constraints(
    sp: SpectralLine,
    limits: FittingLimits,
    subprofile: Optional[str] = None
) -> ConstraintSet:
    
    """
    Compute initial values and bounds for the profile parameters of a spectral line.

    Args:
        cfg: SpectralLine configuration.
        limits: Kinematic constraints (FWHM and center shift in km/s).
        profile: Default profile if cfg.profile is None.
        subprofile: Sub-profile function to use within compound models like SPAF.

    Returns:
        ConstraintSet: Contains initial values, bounds, profile type, and parameter names.
    """
   
    selected_profile = sp.profile
    if selected_profile not in PROFILE_FUNC_MAP:
        raise ValueError(
            f"Profile '{selected_profile}' is not defined. "
        f"Available for continuum are : {list(PROFILE_CONTINUUM_FUNC_MAP.keys())+["balmercontinuum"]} and for the profiles are {list(PROFILE_LINE_FUNC_MAP.keys())+ ["SPAF"]}")
    if selected_profile == "SPAF":
        # ---- SPAF: Sum of Profiles with Free Amplitudes ----
        if not subprofile:
            raise ValueError(f"SPAF profile requires a defined subprofile avalaible options are {list(PROFILE_LINE_FUNC_MAP.keys())}.")
        if not isinstance(sp.amplitude, list):
            raise ValueError("SPAF profile requires cfg.amplitude to be a list of amplitudes.")
        if sp.region not in CANONICAL_WAVELENGTHS:
            raise KeyError(f"Missing canonical wavelength for region='{sp.region}' in CANONICAL_WAVELENGTHS.")
   
    if sp.region.lower() == 'fe' and sp.how == 'template':
        if not sp.which_template:
            #here we can change the "must define for a waring"
            raise ValueError("Fe template must define 'which_template' (e.g., 'OP', 'UV')")
        
        return ConstraintSet(
            init=[3.045, 0.0, 1.0],
            upper=[3.8, 100.0, 100.0],
            lower=[2.7, -100.0, 0.0],
            profile='fitFe' + sp.which_template,
            param_names=['logFWHM', 'shift', 'amplitude'],
        )
        
    if selected_profile == "balmercontinuum":
        return ConstraintSet(
            init=[1.0, 10000.0, 1.0],
            upper=[10.0, 50000.0, 2.0],
            lower=[0.0, 5000.0, 0.01],
            profile = selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names,)

    if selected_profile == 'powerlaw':
         return ConstraintSet(
             init=[-1.1, 0.0],
             upper=[-1.0, 10.0],
             lower=[-3.0, 0.0],
             profile=selected_profile,
             param_names=PROFILE_FUNC_MAP.get(selected_profile).param_names)#['index', 'scale'],

    if selected_profile == 'linear':
         return ConstraintSet(
             init=[0.1e-4, 0.5],
             upper=[10.0, 10.0],
             lower=[-3.0, 0.0],
             profile=selected_profile,
            param_names=PROFILE_FUNC_MAP.get(selected_profile).param_names)
    if selected_profile == "brokenpowerlaw":
         return ConstraintSet(
             init=[0.1,-1.7, 0.0, 5500.0],
             upper=[10.0,0.0, 1.0, 7000.0],
             lower=[0.0,-3.0, -1.0, 4000.0],
             profile=selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names)
    if selected_profile == "logparabola":
         #should be testted
         return ConstraintSet(
             init=[ 1.0,1.5, 0.1],
            upper=[10,3.0, 1.0, 10.0],
            lower=[0.0,0.0, 0.0],
             profile=selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names)
    if selected_profile == "exp_cutoff":
         #should be testted
         return ConstraintSet(
             init=[1.0,1.5,5000.0],
            upper=[10,3.0, 1.0, 1e5],
            lower=[0.0,0.0, 0.0],
             profile=selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names)
    if selected_profile == "polynomial":
         #should be testted
         return ConstraintSet(
             init=[1.0,1.0,1.0,1.0,1.0],
            upper=[10.0,10.0,10.0,10.0,10.0],
            lower=[0.0,0.0,0.0,0.0,0.0],
            profile=selected_profile,
            param_names= PROFILE_FUNC_MAP.get(selected_profile).param_names)
   
    if selected_profile in PROFILE_LINE_FUNC_MAP:
        func = PROFILE_LINE_FUNC_MAP[selected_profile]
        names = func.param_names 
        # base kinematics
        center0   = sp.center
        shift0    = -5.0 if sp.region in ["outflow", "winds"] else 0.0
        cen_up    = center0 + kms_to_wl(limits.center_shift, center0)
        cen_lo    = center0 - kms_to_wl(limits.center_shift, center0)
        fwhm_lo   = kms_to_wl(limits.lower_fwhm,    center0)
        fwhm_up   = kms_to_wl(limits.upper_fwhm,    center0)
        fwhm_init = fwhm_lo * (2.0 if sp.region in ["outflow", "winds"] else 1.0)
        amp_init  = float(sp.amplitude) / 10.0

        init, upper, lower = [], [], []

        for p in names:
            if p == "amplitude":
                init.append(amp_init)
                upper.append(limits.max_amplitude)
                lower.append(0.0)

            elif p == "center":
                init.append(center0 + shift0)
                upper.append(cen_up)
                lower.append(cen_lo)

            elif p in ("fwhm", "width", "fwhm_g", "fwhm_l"):
                # both Gaussian & Lorentzian widths share same kinematic bounds
                init.append(fwhm_init)
                upper.append(fwhm_up)
                lower.append(fwhm_lo)

            elif p == "alpha":
                # skewness parameter: start symmetric, allow ±5
                init.append(0.0)
                upper.append(5.0)
                lower.append(-5.0)

            elif p in ("lambda", "lambda_"):
                # EMG decay: start at 1, allow up to 1/tau ~ 1e3
                init.append(1.0)
                upper.append(1e3)
                lower.append(0.0)

            else:
                raise ValueError(f"Unknown profile parameter '{p}' for '{selected_profile}'")
        return ConstraintSet(
            init=init,
            upper=upper,
            lower=lower,
            profile=selected_profile,
            param_names=names,
        )
        
    if selected_profile == "SPAF":
        func = PROFILE_LINE_FUNC_MAP[subprofile]
        amp_list = sp.amplitude
        #print(amp_list)
        #print("n_free_amps",len(amp_list))
        #amp_upper = [1.0] * len(amp_list)
        names = [f"amplitude{n}" for n in range(len(amp_list))] +["shift"]+ func.param_names[2:]
        # base kinematics
        lambda0 = CANONICAL_WAVELENGTHS[sp.region]
        shift_init = 0.0 if sp.component == 1 else (-2.0) ** sp.component
        shift_upper = kms_to_wl(limits.center_shift, lambda0)
        fwhm_lo   = kms_to_wl(limits.lower_fwhm,    lambda0)
        fwhm_up   = kms_to_wl(limits.upper_fwhm,    lambda0)
        fwhm_init = (fwhm_lo) * (2.0 if sp.region in ["outflow", "winds"] else 1.0)

        init, upper, lower = [], [], []

        for _,p in enumerate(names):
            #print(p)
            if "amplitude" in p:
                init.append(5.0)
                upper.append(10.0)
                lower.append(0.0)

            elif p == "shift":
                init.append(shift_init)
                upper.append(shift_upper)
                lower.append(-shift_upper)

            elif p in ("fwhm", "width", "fwhm_g", "fwhm_l"):
                # both Gaussian & Lorentzian widths share same kinematic bounds
                init.append(0.0)
                upper.append(fwhm_up)
                lower.append(fwhm_lo)

            elif p == "alpha":
                # skewness parameter: start symmetric, allow ±5
                init.append(0.0)
                upper.append(5.0)
                lower.append(-5.0)

            elif p in ("lambda", "lambda_"):
                # EMG decay: start at 1, allow up to 1/tau ~ 1e3
                init.append(1.0)
                upper.append(1e3)
                lower.append(0.0)
        #print("n total params",len(init))
        if not (len(init) == len(upper) == len(lower) == len(names)):
            raise RuntimeError(f"Builder mismatch for '{selected_profile}_{subprofile}': {names}")
        
        return ConstraintSet(
            init=init,
            upper=upper,
            lower=lower,
            profile=f"{selected_profile}_{subprofile}",
            param_names=names,
        )






def make_get_param_coord_value(
    params_dict: Dict[str, int], initial_params: jnp.ndarray
) -> Callable[[str, str, Union[str, int], str, bool], Tuple[int, float, str]]:
    """
    Returns a function to retrieve the index, value, and name of a parameter based on its key parts.

    Args:
        params_dict: Mapping of parameter keys to indices.
        initial_params: Initial parameter array.

    Returns:
        A function to get parameter info by name, line name, component, and region.
    """

    def get_param_coord_value(
        param: str,
        line_name: str,
        component: Union[str, int],
        region: str,
        verbose: bool = False,
    ) -> Tuple[int, float, str]:
        key = f"{param}_{line_name}_{component}_{region}"
        pos = params_dict.get(key)
        if pos is None:
            raise KeyError(f"Key '{key}' not found in params_dict.")
        if verbose:
            print(f"{key}: value = {initial_params[pos]}")
        return pos, float(initial_params[pos]), param

    return get_param_coord_value


from typing import List, Optional, Tuple, Dict
import math
import jax
import jax.numpy as jnp

class Parameter:
    """
    Represents a fit parameter with optional bounds or ties, plus a transform
    determined by its min/max.
    """
    def __init__(
        self,
        name: str,
        value: float,
        *,
        min: float = -jnp.inf,
        max: float = jnp.inf,
        tie: Optional[Tuple[str, str, str, float]] = None
    ):
        self.name = name
        self.value = float(value)
        self.min = float(min)
        self.max = float(max)
        self.tie = tie  # (target, source, op, operand)

        # Determine transform based on bounds
        if math.isfinite(self.min) and math.isfinite(self.max):
            self.transform = 'logistic'          # map via sigmoid into [min,max]
        elif math.isfinite(self.min) and not math.isfinite(self.max):
            self.transform = 'lower_bound_square'  # val = min + r^2, so val>=min exactly
        elif not math.isfinite(self.min) and math.isfinite(self.max):
            self.transform = 'upper_bound_square'  # val = max - r^2, so val<=max exactly
        else:
            self.transform = 'linear'             # val = r (unbounded)

class Parameters:
    def __init__(self):
        self._list: List[Parameter] = []
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
    ):
        lo = -jnp.inf if min is None else min
        hi = jnp.inf if max is None else max
        self._list.append(Parameter(name=name, value=value, min=lo, max=hi, tie=tie))
        self._jit_raw_to_phys = None
        self._jit_phys_to_raw = None

    @property
    def names(self) -> List[str]:
        return [p.name for p in self._list]

    def _finalize(self):
        self._raw_list = [p for p in self._list if p.tie is None]
        self._tied_list = [p for p in self._list if p.tie is not None]
        self._jit_raw_to_phys = jax.jit(self._raw_to_phys_core)
        self._jit_phys_to_raw = jax.jit(self._phys_to_raw_core)

    def raw_init(self) -> jnp.ndarray:
        if self._jit_phys_to_raw is None:
            self._finalize()
        vals = jnp.array([p.value for p in self._raw_list])
        return self._jit_phys_to_raw(vals)

    def raw_to_phys(self, raw_params: jnp.ndarray) -> jnp.ndarray:
        if self._jit_raw_to_phys is None:
            self._finalize()
        return self._jit_raw_to_phys(raw_params)

    def phys_to_raw(self, phys_params: jnp.ndarray) -> jnp.ndarray:
        if self._jit_phys_to_raw is None:
            self._finalize()
        return self._jit_phys_to_raw(phys_params)

    def _raw_to_phys_core(self, raw: jnp.ndarray) -> jnp.ndarray:
        def convert_one(r_row):
            ctx: Dict[str, jnp.ndarray] = {}
            idx = 0
            for p in self._raw_list:
                if p.transform == 'logistic':
                    val = p.min + (p.max - p.min) * jax.nn.sigmoid(r_row[idx])
                elif p.transform == 'lower_bound_square':
                    val = p.min + r_row[idx]**2
                elif p.transform == 'upper_bound_square':
                    val = p.max - r_row[idx]**2
                else:
                    val = r_row[idx]
                ctx[p.name] = val
                idx += 1
            # apply ties
            op_map = {'*': jnp.multiply, '+': jnp.add, '-': jnp.subtract, '/': jnp.divide}
            for p in self._tied_list:
                tgt, src, op, operand = p.tie
                ctx[tgt] = op_map[op](ctx[src], operand)
            return jnp.stack([ctx[p.name] for p in self._list])

        if raw.ndim == 1:
            return convert_one(raw)
        return jax.vmap(convert_one)(raw)

    def _phys_to_raw_core(self, phys: jnp.ndarray) -> jnp.ndarray:
        def invert_one(v_row):
            raw_vals: List[jnp.ndarray] = []
            idx = 0
            for p in self._raw_list:
                if p.transform == 'logistic':
                    frac = (v_row[idx] - p.min) / (p.max - p.min)
                    frac = jnp.clip(frac, 1e-6, 1 - 1e-6)
                    raw_vals.append(jnp.log(frac / (1 - frac)))
                elif p.transform == 'lower_bound_square':
                    raw_vals.append(jnp.sqrt(jnp.maximum(v_row[idx] - p.min, 0)))
                elif p.transform == 'upper_bound_square':
                    raw_vals.append(jnp.sqrt(jnp.maximum(p.max - v_row[idx], 0)))
                else:
                    raw_vals.append(v_row[idx])
                idx += 1
            return jnp.stack(raw_vals)

        if phys.ndim == 1:
            return invert_one(phys)
        return jax.vmap(invert_one)(phys)

    @property
    def specs(self) -> List[Tuple[str, float, float, float, str]]:
        return [(p.name, p.value, p.min, p.max, p.transform) for p in self._list]
