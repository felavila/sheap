from __future__ import annotations

import logging
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np

from sheap.DataClass.DataClass import ComplexRegion, SpectralLine,FitResult
from sheap.FunctionsMinimize.profiles import PROFILE_FUNC_MAP

from sheap.RegionFitting.RegionFitting import RegionFitting

# from sfdmap2 import sfdmap
from sheap.RegionHandler.RegionBuilder import RegionBuilder
from sheap.LineMapper.LineMapper import mapping_params
from sheap.utils import prepare_uncertainties  # ?
from sheap.Plotting.SheapPlot import SheapPlot
logger = logging.getLogger(__name__)

module_dir = os.path.dirname(os.path.abspath(__file__))


ArrayLike = Union[np.ndarray, jnp.ndarray]


# list of SpectralLine
def make_g(list):
    amplitudes, centers = list.amplitude, list.center
    return PROFILE_FUNC_MAP["Gsum_model"](centers, amplitudes)


# TODO Add multiple models to the reading.
def pad_error_channel(spectra: ArrayLike, frac: float = 0.01) -> ArrayLike:
    """Ensure *spectra* has a third channel (error) by padding with *frac* × signal."""
    if spectra.shape[1] != 2:
        return spectra  # already 3‑channel
    signal = spectra[:, 1, :]
    error = jnp.expand_dims(signal * frac, axis=1)
    return jnp.concatenate((spectra, error), axis=1)


class Sheapectral:
    # the units of the flux are not important (I think) meanwhile all the wavelenght dependece are in A
    def __init__(
        self,
        spectra: Union[str, jnp.ndarray],
        z: Optional[Union[float, jnp.ndarray]] = None,
        coords: Optional[jnp.ndarray] = None,
        ebv: Optional[jnp.ndarray] = None,
        names: Optional[list[str]] = None,
        extinction_correction: str = "pending",  # this only can be pending or done
        redshift_correction: str = "pending",  # this only can be pending or done
        **kwargs,
    ):
        self.log = logging.getLogger(self.__class__.__name__)
        # self.cfg = config or SheapConfig()
        self.extinction_correction = extinction_correction
        self.redshift_correction = redshift_correction
        spec_arr = self._load_spectra(spectra)
        spec_arr = pad_error_channel(spec_arr)
        self.spectra = spec_arr.astype(jnp.float64)
        # self.in_spectra = spec_arr
        self.coords = coords  # may be None – handle carefully downstream
        self.ebv = ebv
        self.z = self._prepare_z(z, self.spectra.shape[0])

        self.names = (
            names if names is not None else np.arange(self.spectra.shape[0]).astype(str)
        )

        if self.extinction_correction == "pending" and (
            self.coords is not None or self.ebv is not None
        ):
            print(
                "extinction correction will be do it, change 'extinction_correction' to done if you want to avoid this step"
            )
            self._apply_extinction()
            self.extinction_correction = "done"

        if self.redshift_correction == "pending" and self.z is not None:
            print(
                "redshift correction will be do it, change 'redshift_correction' to done if you want to avoid this step"
            )
            self._apply_redshift()
            self.redshift_correction = "done"

        # Stage bookkeeping
        self.sheap_set_up()
        # self.host_subtraction = host_subtraction

    def _load_spectra(self, spectra: Union[str, ArrayLike]) -> jnp.ndarray:
        if isinstance(spectra, (str, Path)):
            arr = np.loadtxt(spectra)
            return jnp.array(arr).T  # ensure (c, λ) then transpose later
        elif isinstance(spectra, np.ndarray):
            return jnp.array(spectra)
        elif isinstance(spectra, jnp.ndarray):
            return spectra
        raise TypeError("spectra must be a path or ndarray")

    def _prepare_z(
        self, z: Optional[Union[float, ArrayLike]], nobj: int
    ) -> Optional[jnp.ndarray]:
        if z is None:
            return None
        if isinstance(z, (int, float)):
            return jnp.repeat(z, nobj)
        return jnp.array(z)

    def _apply_extinction(self) -> None:
        """Cardelli 1989 – uses *sfdmap* if coords are available."""
        from sfdmap2 import sfdmap  # lazy import to avoid heavy deps if unused

        from sheap.Tools.unred import unred

        ebv = self.ebv
        if self.coords is not None:
            self.coords = jnp.array(self.coords)
            l, b = self.coords.T  # type: ignore[union-attr]
            ebv_func = sfdmap.SFDMap(os.path.join(module_dir, "SuportData", "sfddata/")).ebv
            ebv = ebv_func(l, b)
        corrected = unred(*np.swapaxes(self.spectra[:, [0, 1], :], 0, 1), ebv)
        # propagate to error channel proportionally as pyqso
        ratio = corrected / self.spectra[:, 1, :]
        self.spectra = self.spectra.at[:, 1, :].set(corrected)
        self.spectra = self.spectra.at[:, 2, :].multiply(ratio)

    def _apply_redshift(self) -> None:
        from sheap.Tools.others import _deredshift

        self.spectra = _deredshift(self.spectra, self.z)

    def sheap_set_up(self):
        if len(self.spectra.shape) <= 2:
            self.spectra = self.spectra[jnp.newaxis, :]
        self.spectra_shape = self.spectra.shape  # ?
        self.spectra_nans = jnp.isnan(self.spectra)
        self.spectra_exp = jnp.round(
            jnp.log10(jnp.nanmedian(self.spectra[:, 1, :], axis=1))
        )  # * 0
        # maybe add a filter here to see whats going on?

    def build_region(
        self,
        xmin: float,
        xmax: float,
        n_narrow: int = 1,
        n_broad: int = 1,
        tied_narrow_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        tied_broad_to: Optional[Union[str, Dict[int, Dict[str, int]]]] = None,
        fe_regions: List[str] = ['fe_uv', "feii_IZw1", "feii_forbidden", "feii_coronal"],
        fe_mode: str = "template",
        add_outflow: bool = False,
        add_narrowplus: bool = False,
        by_region: bool = False,
        force_linear: bool = False,
        add_balmercontiniumm: bool = False,
        fe_tied_params: Union[tuple, list] = ('center', 'width'),
        add_NLR = False,
        powerlaw_profile = "powerlaw"
    ):
        self.builded_region = RegionBuilder(
            xmin=xmin,
            xmax=xmax,
            n_narrow=n_narrow,
            n_broad=n_broad,
            tied_narrow_to=tied_narrow_to,
            tied_broad_to=tied_broad_to,
            fe_regions=fe_regions,
            fe_mode=fe_mode,
            add_outflow=add_outflow,
            add_narrowplus=add_narrowplus,
            by_region=by_region,
            force_linear=force_linear,
            add_balmercontiniumm=add_balmercontiniumm,
            fe_tied_params=fe_tied_params,
            add_NLR = add_NLR,
            powerlaw_profile = powerlaw_profile
        )
        
        self.fitting_rutine = self.builded_region()
        self.complex_region = self.builded_region.complex_region
    
    def fit_region(self, num_steps_list=[3000, 3000], add_step=True, tied_fe=False):
        spectra = self.spectra.at[:, [1, 2], :].multiply(
            10 ** (-1 * self.spectra_exp[:, jnp.newaxis, jnp.newaxis])
        )  # apply escale to 0-20 max
        if not hasattr(self, "builded_region"):
            raise RuntimeError("build_region() must be called before fit()")
        self.fitting_rutine = self.builded_region(add_step=add_step, tied_fe=tied_fe, num_steps_list=num_steps_list)
        fittingclass = RegionFitting(self.fitting_rutine)
        self.result = fittingclass(spectra, do_return=True)
        # Rescale amplitudes
        scaled = 10**self.spectra_exp
        idxs = mapping_params(self.result.params_dict, [["amplitude"], ["scale"]])
        self.result.max_flux = self.result.max_flux * scaled
        self.result.params = self.result.params.at[:, idxs].multiply(scaled[:, None])
        self.result.uncertainty_params = self.result.uncertainty_params.at[:, idxs].multiply(scaled[:, None])
        self.model_keywords = self.fitting_rutine.get("model_keywords")
        self._plotter = SheapPlot(sheap=self)
        
    @property
    def modelplot(self):
        if not hasattr(self, "_plotter"):
            if hasattr(self, "result"):
                self._plotter = SheapPlot(sheap=self)
            else:
                raise RuntimeError("No fit result found. Run `fit_region()` first.")
        return self._plotter

    
    # def fit_region(self, num_steps_list=[3000, 3000], add_step=True, tied_fe=False):

    #     spectra = self.spectra.at[:, [1, 2], :].multiply(
    #         10 ** (-1 * self.spectra_exp[:, jnp.newaxis, jnp.newaxis])
    #     )  # apply escale to 0-20 max
    #     if not hasattr(self, "builded_region"):
    #         raise RuntimeError("build_region() must be called before fit()")
        
    #     self.fitting_rutine = self.builded_region(add_step=add_step, tied_fe=tied_fe, num_steps_list=num_steps_list)
        
        
    #     fittingclass = RegionFitting(self.fitting_rutine)
    #     result = fittingclass(spectra, do_return=True)
       
    #     # Unpack result to self
    #     self.params = result.params
    #     self.uncertainty_params = result.uncertainty_params
    #     self.outer_limits = result.outer_limits
    #     self.inner_limits = result.inner_limits
    #     self.loss = result.loss
    #     self.mask = result.mask
    #     self.step = result.step
    #     self.params_dict = result.params_dict
    #     self.initial_params = result.initial_params
    #     self.profile_params_index_list = result.profile_params_index_list
    #     self.profile_functions = result.profile_functions
    #     self.profile_names = result.profile_names
    #     self.complex_region = result.complex_region
    #     self.model_keywords = self.fitting_rutine.get("model_keywords")

    #     # Rescale amplitudes
    #     scaled = 10**self.spectra_exp
    #     idxs = mapping_params(self.params_dict, [["amplitude"], ["scale"]])
    #     self.max_flux = result.max_flux * scaled
    #     self.params = self.params.at[:, idxs].multiply(scaled[:, None])
    #     self.uncertainty_params = self.uncertainty_params.at[:, idxs].multiply(scaled[:, None])
        
    #     self._complex_region()

    @classmethod
    def from_pickle(cls, filepath: Union[str, Path]) -> Sheapectral:
        filepath = Path(filepath)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            spectra=data["spectra"],
            z=data["z"],
            names=data["names"],
            coords=data["coords"],
            extinction_correction=data["extinction_correction"],
            redshift_correction=data["redshift_correction"],
        )

        obj.params = jnp.array(data.get("params"))
        obj.uncertainty_params = jnp.array(
            data.get("uncertainty_params", jnp.zeros_like(obj.params))
        )
        obj.params_dict = data.get("params_dict")
        obj.mask = jnp.array(data.get("mask"))
        obj.profile_params_index_list = data.get("profile_params_index_list")
        obj.profile_names = data.get("profile_names")
        obj.fitting_rutine = {"fitting_rutine": data.get("fitting_rutine")}
        obj.model_keywords = data.get("model_keywords")
        region_defs = data.get("complex_region")
        obj.outer_limits = data.get(
            "outer_limits"
        )  # region extension comming as a more proper name
        if isinstance(region_defs, list):
            obj.complex_region = [SpectralLine(**i) for i in region_defs]
            # obj.complex_region = [SpectralLine(**d) for d in region_defs]
            #
        if obj.profile_names is not None:
            obj.profile_functions = [
                (
                    PROFILE_FUNC_MAP.get(i)
                    if i != "combine_gaussian"
                    else make_g(obj.complex_region[idx])
                )
                for idx, i in enumerate(obj.profile_names)
            ]
        obj._complex_region()
        return obj

    def _save(self):
        _region_defs = [
            i.to_dict() for i in self.complex_region
        ]  # to dict so it can be read anyway
        dic_ = {
            "names": self.names,
            "spectra": np.array(self.spectra),
            "coords": np.array(self.coords),
            "z": np.array(self.z),  # array
            "extinction_correction": self.extinction_correction,
            "redshift_correction": self.redshift_correction,
            # this should be a dataclass
            "params": np.array(self.params),  # array
            "uncertainty_params": np.array(self.uncertainty_params),  # array
            "params_dict": self.params_dict,  # array
            "mask": np.array(self.mask),  # array
            "complex_region": _region_defs,  # dic
            "profile_params_index_list": self.profile_params_index_list,  # array
            "profile_names": self.profile_names,  # list str
            "fitting_rutine": self.fitting_rutine["fitting_rutine"],  # dict list
            "outer_limits": self.outer_limits,
            "model_keywords": self.model_keywords,
        }
        estimated_size = sys.getsizeof(pickle.dumps(dic_))
        print(f"Estimated pickle size: {estimated_size / 1024:.2f} KB")
        return dic_

    def save_to_pickle(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self._save(), f)

    def quicklook(self, idx: int, ax=None, xlim=None, ylim=None):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FixedLocator

        lam, flux, err = self.spectra[idx]

        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 5))

        ax.errorbar(lam, flux, yerr=err, ecolor='dimgray', color="black", zorder=1)

        # Default xlim and ylim if not provided
        if xlim is None:
            xlim = (jnp.nanmin(lam), jnp.nanmax(lam))
        if ylim is None:
            ylim = (0, jnp.nanmax(flux) * 1.02)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Flux [arb]")

        # Plot ID label outside main plot area, above-left
        ax.text(
            0.0,
            1.05,
            f"ID {self.names[idx]} ({idx})",
            fontsize=10,
            transform=ax.transAxes,
            ha='left',
            va='bottom',
        )

        ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))

        return ax

    def _complex_region(self):
        self.complex_region_class = ComplexRegion(
            complex_region=self.complex_region,
            params=self.params,
            uncertainty_params=self.uncertainty_params,
            profile_functions=self.profile_functions,
            params_dict=self.params_dict,
            profile_names=self.profile_names,
            profile_params_index_list=self.profile_params_index_list,
        )

        # complex_region: List[SpectralLine]
        # profile_functions: List[Callable]
        # params:np.ndarray
        # uncertainty_params:np.ndarray
        # profile_params_index_list: np.ndarray
        # params_dict: Dict
        # profile_names: List[str]

