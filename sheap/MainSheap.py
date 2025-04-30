import os
from typing import Union, Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator
from sfdmap2 import sfdmap

from sheap.tools.others import _deredshift

#from .sfdmap import SFDMap_2
from sheap.tools.unred import unred
from sheap.utils import prepare_uncertainties  # ?

module_dir = os.path.dirname(os.path.abspath(__file__))

class Sheapectral:
    #the units of the flux are not important (I think) meanwhile all the wavelenght dependece are in A 
    #TODO normalization ? a good option or not?
    #TODO I have to take in consideration a think i never thing before the sdss spectras posses some 0 inside the errors the logic will be give to those a really big error in compensation
    def __init__(self,
    spectra: Union[str, jnp.ndarray],
    z: Optional[Union[float, jnp.ndarray]] = None,
    coords: Optional[jnp.ndarray] = None,
    names: Optional[list[str]] = None,
    host_subtraction: bool = True,
    **kwargs
):
        self.spectra = self._load_spectra(spectra)
        if self.spectra.shape[1]==2:
            print("Warning SHEAP works with arrays (n,3,X); if your array is (n,2,X) it will add an array equal to 1% of signal")
            extra_slice = jnp.expand_dims(self.spectra[:, 1, :] / 100, axis=1)
            self.spectra = jnp.concatenate((self.spectra, extra_slice), axis=1)
        #self.input_spectra = jnp.copy(self.spectra)
        if coords is not None:
            self.coords = coords
            self.f_ebv = sfdmap.SFDMap(os.path.join(module_dir,"suport_data","sfddata/")).ebv
            self.ebv = self.f_ebv(*self.coords.T)
            spectra_unred = unred(*np.swapaxes(self.spectra[:,[0,1],:],0,1),self.ebv)
            self.spectra = self.spectra.at[:,2,:].multiply(spectra_unred/self.spectra[:,1,:]) #error that uses pyqso
            self.spectra = self.spectra.at[:,1,:].set(spectra_unred)
        else:
            print("Warning no coords define the code will not correct for extinction")
        self.sheap_set_up()
        self.host_subtraction = host_subtraction
        self.z: Optional[jnp.ndarray] = None  # helps mypy know the type

        if z is not None:
            if isinstance(z, (int, float)):
                print("Assuming same redshift for all the objects ")
                self.z = jnp.repeat(z, self.spectra.shape[0])
            else:
                self.z = jnp.array(z)

            self.spectra = _deredshift(self.spectra, self.z)
            
        if names is None:
            self.names = np.arange(len(spectra)).astype(str)
            
    def _load_spectra(self, spectra: Union[str, jnp.ndarray]) -> jnp.ndarray:
        if isinstance(spectra, str):
            return jnp.array(np.loadtxt(spectra), dtype='float').transpose()  # this will be removed in future iterations
        elif isinstance(spectra, jnp.ndarray):
            return spectra
        elif isinstance(spectra, np.ndarray):
            return jnp.array(spectra)
        else:
            raise ValueError("Invalid spectra type")
    
    def sheap_set_up(self):
        if len(self.spectra.shape)<=2:
            self.spectra = self.spectra[jnp.newaxis,:]
        self.spectra_shape = self.spectra.shape#?
        self.spectra_nans = jnp.isnan(self.spectra)
        self.spectra_exp_ = jnp.round(jnp.log10(jnp.nanmedian(self.spectra[:,1, :],axis=1))) #* 0
        #maybe add a filter here to see whats going on? 
        self.spectra = self.spectra.at[:,[1,2],:].multiply(10 ** (-1 * self.spectra_exp_[:,jnp.newaxis,jnp.newaxis]))
    @property
    def spectra_exp(self):
        return -1 * self.spectra_exp_
    # def run_host_subtraction(self,method="star_method"):
    #     return self.spectra
    
    # def mask_not_fitted_region(self,min_r,max_r,fill=jnp.nan):
    #     #this function should be use in order to remove the external parts of the spectra 
    #     #I mean the parts that will not be use in the fit
    #     mask = (self.spectra[:,0, :] >= min_r) & (self.spectra[:,0, :] <= max_r)
    #     expanded_mask = mask[:, jnp.newaxis, :]  # Shape: (19, 1, 4595)
    #     masked_spectra = jnp.where(expanded_mask, self.spectra, fill)
    #     central_wl = jnp.nanmedian(masked_spectra[:,0, :],axis=1)
    #     return masked_spectra,central_wl
    
    # def fit_region(self,region):
    #     # region have to be a kind of dictionary or think like that and then we build a function to make it an array in the way that sheap can read it 
    #     min_r,max_r = 4400.11083984,5448.33105469
    #     min_e,max_e = 4750,5040
    #     masked_spectra,central_wl = self.mask_not_fitted_region(min_r,max_r,fill=jnp.nan)
    #     mask_emission = (self.spectra[:,0,:] >= min_e) & (self.spectra[:,0 ,:]  <= max_e)
    #     #mask_emission = mask_emission #[:, jnp.newaxis, :]  # Shape: (19, 1, 4595)
    #     masked_emission_spectra = masked_spectra.at[:,2,:].set(jnp.where(masked_spectra[:,2,:],jnp.nan,mask_emission))
    #     y_uncertainties = prepare_uncertainties(masked_emission_spectra[:,2,:],masked_emission_spectra[:,2,:])
    #     X = [masked_emission_spectra]
    #     fe_template_op_norm = jnp.zeros_like(mask_emission)
    #     if region in ["Hbeta"]:
    #         X.append(self.fe_template_op_norm)
    #     ##mag_corrected_agn_to_mini = mag_corrected_agn.copy()
    #     ##mag_corrected_agn_to_mini = jnp.array(mag_corrected_agn_to_mini).at[2,:].set(jnp.where(mask_emission_region,jnp.nan,mag_corrected_agn_to_mini[2,:]))
    #     #y_uncertainties = prepare_uncertainties(None,mag_corrected_agn_to_mini[2,:])
    #     #X = [mag_corrected_agn_to_mini,fe_template_norm,mag_corrected_agn_to_mini[0,:]]
    #     return masked_emission_spectra,mask_emission,y_uncertainties

    # def regions(self):
    #     return 
    
    def plot_(self,n=0,region=None):
        # This have to be more complex than this 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(35, 15), sharex=True, gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})
        ax1.plot(*self.spectra[n,[0,1],:])
        # ax1.plot(mag_corrected_agn[0,:],fit_continium(X,params))
        # ax2.plot(mag_corrected_agn[0,:],mag_corrected_agn[1,:]-fit_continium(X,params))
        ax2.axhline(0, ls='--', c='k')
        
        if isinstance(region,list):
            plt.xlim(region[0], region[1])
        elif region:
            print(ax1.get_ylim(), self.spectra[n,0,:].shape, self.mask_emission[0].shape)
            y1, y2 = ax1.get_ylim()    # Get the y-limits for the fill
            ax1.fill_between(self.spectra[n, 0, :], y1, y2, where=self.mask_emission[n], color="grey", alpha=0.1, label="mask", zorder=10)
        else:
            plt.xlim(jnp.nanmin(self.spectra[n,0,:]), jnp.nanmax(self.spectra[n,0,:]))
        # ax.set_xlabel(r"$R_S$ (lt-day)", fontsize=20)
        # ax.set_ylabel('Number of lenses', fontsize=20)
        # Remove corner y-axis tick values
        y_ticks = ax1.get_yticks()
        ax1.yaxis.set_major_locator(FixedLocator(y_ticks))
        y_tick_labels = ["" if i == 0 or i == len(y_ticks) - 1 else label.get_text() for i, label in enumerate(ax1.get_yticklabels())]
        ax1.set_yticklabels(y_tick_labels)
        
        y_ticks = ax2.get_yticks()
        ax2.yaxis.set_major_locator(FixedLocator(y_ticks))
        y_tick_labels = ["" if i == 0 or i == len(y_ticks) - 1 else label.get_text() for i, label in enumerate(ax2.get_yticklabels())]
        ax2.set_yticklabels(y_tick_labels)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax2.tick_params(axis='both', which='major', labelsize=20)
        plt.show()