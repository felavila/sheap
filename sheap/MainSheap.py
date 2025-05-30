from __future__ import annotations

import logging
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import jax.numpy as jnp
import numpy as np

from sheap.DataClass.DataClass import SpectralLine,FitResult
from sheap.Functions.profiles import PROFILE_FUNC_MAP,make_g
from sheap.HostSubtraction.HostSubtraction import HostSubtraction
from sheap.RegionFitting.RegionFitting import RegionFitting
from sheap.RegionHandler.RegionBuilder import RegionBuilder
from sheap.Plotting.SheapPlot import SheapPlot
from sheap.Tools.setup_utils import pad_error_channel,ArrayLike


logger = logging.getLogger(__name__)



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
            sfd_path = Path(__file__).resolve().parent / "SuportData" / "sfddata/"
            ebv_func = sfdmap.SFDMap(sfd_path).ebv
            ebv = ebv_func(l, b)
        corrected = unred(*np.swapaxes(self.spectra[:, [0, 1], :], 0, 1), ebv)
        # propagate to error channel proportionally as pyqso
        ratio = corrected / self.spectra[:, 1, :]
        self.spectra = self.spectra.at[:, 1, :].set(corrected)
        self.spectra = self.spectra.at[:, 2, :].multiply(ratio)

    def _apply_redshift(self) -> None:
        from sheap.Tools.spectral_basic import _deredshift
        self.spectra = _deredshift(self.spectra, self.z)

    def sheap_set_up(self):
        if len(self.spectra.shape) <= 2:
            self.spectra = self.spectra[jnp.newaxis, :]
        self.spectra_shape = self.spectra.shape  # ?
        self.spectra_nans = jnp.isnan(self.spectra)

    def _apply_hostsubstraction(self,learning_rate=1e-1,n_galaxies=5,n_qso=10) -> None:
        "Experimental feature"
        hostsubstraction = HostSubtraction(self.spectra,learning_rate=learning_rate,n_galaxies=n_galaxies,n_qso=n_qso)    
        hostsubstraction._run_substraction(num_steps=50_000)
        return hostsubstraction
    
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
        fe_tied_params: Union[tuple, list] = ('center', 'fwhm'),
        add_NLR = False,
        powerlaw_profile = "powerlaw", #can be broken power law also we have to add that to the main class
        no_fe = False
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
            powerlaw_profile = powerlaw_profile,
            no_fe = no_fe
        )
        
        self.fitting_rutine = self.builded_region()
        self.complex_region = self.builded_region.complex_region
    
    def fit_region(self, num_steps_list=[3000, 3000], add_step=True, tied_fe=False):
        
        if not hasattr(self, "builded_region"):
            raise RuntimeError("build_region() must be called before fit_region()")

        fitting_rutine = self.builded_region(add_step=add_step, tied_fe=tied_fe, num_steps_list=num_steps_list)
        fitting_class = RegionFitting(fitting_rutine)

        fit_output = fitting_class(self.spectra, do_return=True)

        #fit_output.initial_params = fitting_class.initial_params #This also have to be "re-scale"
        fit_output.source = "computed"
        
        # Store result using FitResult directly
        self.result = FitResult(
            params=fit_output.params,
            uncertainty_params=fit_output.uncertainty_params,
            mask=fit_output.mask,
            profile_functions=fit_output.profile_functions,
            profile_names=fit_output.profile_names,
            #loss=fit_output.loss,
            profile_params_index_list=fit_output.profile_params_index_list,
            initial_params=fit_output.initial_params,
            max_flux=fit_output.max_flux,
            params_dict=fit_output.params_dict,
            complex_region=fit_output.complex_region,
            outer_limits=fit_output.outer_limits,
            inner_limits=fit_output.inner_limits,
            model_keywords= fit_output.model_keywords,
            fitting_rutine = fit_output.fitting_rutine,
            constraints = fit_output.constraints,
            source=fit_output.source,
            dependencies=fit_output.dependencies
        )

        self._plotter = SheapPlot(sheap=self)

        
    
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

        complex_region = data.get("complex_region", [])
        obj.complex_region = [SpectralLine(**i) for i in complex_region]

        profile_names = data.get("profile_names", [])
        obj.profile_functions = [
            PROFILE_FUNC_MAP.get(name, make_g(obj.complex_region[idx]) if name == "combine_gaussian" else None)
            for idx, name in enumerate(profile_names)
        ]

        obj.result = FitResult(
            params=jnp.array(data.get("params")),
            uncertainty_params=jnp.array(data.get("uncertainty_params", jnp.zeros_like(data.get("params")))),
            initial_params=jnp.array(data.get("initial_params")),
            mask=jnp.array(data.get("mask")),
            profile_functions=obj.profile_functions,
            profile_names=profile_names,
            loss=None,  # Not saved currently, could be added if needed
            profile_params_index_list=data.get("profile_params_index_list"),
            max_flux=data.get("max_flux"),  # Not saved currently, could be added if needed
            params_dict=data.get("params_dict"),
            complex_region=obj.complex_region,
            outer_limits=data.get("outer_limits"),
            inner_limits=data.get("inner_limits"),
            model_keywords=data.get("model_keywords"),
            source=data.get("source", "pickle"),
            constraints = data.get('constraints'),
            fitting_rutine = data.get("fitting_rutine")
        )
        obj._plotter = SheapPlot(sheap=obj)
        return obj
    
    def _save(self):
        _complex_region = [i.to_dict() for i in self.complex_region]

        dic_ = {
            "names": self.names,
            "spectra": np.array(self.spectra),
            "coords": np.array(self.coords),
            "z": np.array(self.z),
            "extinction_correction": self.extinction_correction,
            "redshift_correction": self.redshift_correction,
            "params": np.array(self.result.params),
            "uncertainty_params": np.array(self.result.uncertainty_params),
            "initial_params": np.array(self.result.initial_params),  # explicitly saved
            "params_dict": self.result.params_dict,
            "mask": np.array(self.result.mask),
            "complex_region": _complex_region,
            "profile_params_index_list": self.result.profile_params_index_list,
            "profile_names": self.result.profile_names,
            "fitting_rutine": self.fitting_rutine["fitting_rutine"],
            "outer_limits": self.result.outer_limits,
            "inner_limits": self.result.inner_limits,
            "model_keywords": self.result.model_keywords,
            "source": self.result.source,
            "max_flux":np.array(self.result.max_flux),
            'constraints':np.array(self.result.constraints)
        }

        estimated_size = sys.getsizeof(pickle.dumps(dic_))
        print(f"Estimated pickle size: {estimated_size / 1024:.2f} KB")

        return dic_

    def save_to_pickle(self, filepath: Union[str, Path]):
        ".pkl"
        filepath = Path(filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self._save(), f)

    @property
    def modelplot(self):
        if not hasattr(self, "_plotter"):
            if hasattr(self, "result"):
                self._plotter = SheapPlot(sheap=self)
            else:
                raise RuntimeError("No fit result found. Run `fit_region()` first.")
        return self._plotter
    def result_dict(self, n: int) -> Dict[str, List[float]]:
        return {
            key: [self.result.params[n][i], self.result.uncertainty_params[n][i]]
            for key, i in self.result.params_dict.items()
    }
    
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


