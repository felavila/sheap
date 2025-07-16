from typing import Dict, Optional, Union, List, Tuple
from pathlib import Path
import jax.numpy as jnp
import numpy as np

from sheap.RegionHandler.RegionBuilder import RegionBuilder
from sheap.RegionFitting.RegionFitting import RegionFitting
from sheap.DataClass import FitResult
from sheap.Tools.unred import unred
from sheap.Tools.spectral_basic import _deredshift


class SheapSpectralList:
    """
    Automatically manages RegionFitting instances for different spectral resolutions (npix).

    It uses a single RegionBuilder definition and caches RegionFitting objects keyed by npix.
    This avoids repeated JIT compilation while respecting JAX's static shape requirements.
    """

    def __init__(
        self,
        region_builder: RegionBuilder,
        num_steps_list: List[int] = [1000, 1000],
        add_step: bool = True,
        tied_fe: bool = False,
        sigma_params: bool = True,
    ):
        self.region_builder = region_builder
        self.num_steps_list = num_steps_list
        self.add_step = add_step
        self.tied_fe = tied_fe
        self.sigma_params = sigma_params
        self.cache: Dict = {}
        self.rutine = self.region_builder(
                add_step=self.add_step,
                tied_fe=self.tied_fe,
                num_steps_list=self.num_steps_list
            )
    def get(self, npix: int):
        if npix not in self.cache:
            self.cache[npix] = RegionFitting(self.rutine)
            #RegionFitting(self.rutine)
        return self.cache[npix]

    def _apply_extinction(self, spectra: jnp.ndarray, coords: Optional[jnp.ndarray]) -> jnp.ndarray:
        from sfdmap2 import sfdmap
        if coords is not None:
            coords = jnp.array(coords)
            l, b = coords.T
            sfd_path = Path(__file__).resolve().parent / "SuportData" / "sfddata/"
            ebv_func = sfdmap.SFDMap(sfd_path).ebv
            ebv = ebv_func(l, b)
            corrected = unred(*jnp.swapaxes(spectra[:, [0, 1], :], 0, 1), ebv)
            ratio = corrected / spectra[:, 1, :]
            spectra = spectra.at[:, 1, :].set(corrected)
            spectra = spectra.at[:, 2, :].multiply(ratio)
        return spectra

    def _apply_redshift(self, spectra: jnp.ndarray, z: Optional[jnp.ndarray]) -> jnp.ndarray:
        return _deredshift(spectra, z)


    def fit_loop(self,spectral_dict,early_result=True ):
        spectral_result = {}
        for npix,dic_v in spectral_dict.items():
            fitting_class= self.get(npix)
            spectra,coords,z = dic_v.get("spectra"), dic_v.get("coords"), dic_v.get("z")
            names =  dic_v.get("name",[f"{i}_{npix}" for i in range(coords.shape[0])])
            spectra = self._apply_extinction(spectra, coords)
            spectra = self._apply_redshift(spectra, z)
            #do correction?
            fit_output = fitting_class(spectra, do_return=True)   
            spectral_result[npix] = fit_output
            if early_result:
                return spectral_result
        return spectral_result
            #return spectral_result
            #= RegionFitting(fitting_rutine)