from typing import Callable, Dict, Tuple, Optional
import jax.numpy as jnp
from jax import  lax,jit,vmap
import functools as ft
import numpy as np 





@jit
@ft.partial(vmap, in_axes=(0,0), out_axes=0)
def _deredshift(spectra,z):
    #PyQSO DR16 pass the results in redshift
    spectra = spectra.at[[1,2],:].multiply(1+z[jnp.newaxis,jnp.newaxis])
    spectra = spectra.at[0,:].divide(1+z[jnp.newaxis])
    return spectra

@jit
def kms_to_wl(kms,line_center,c= 2.99792458e5):
    """
    Convert a velocity in km/s to a wavelength shift based on the line center.

    Parameters:
    -----------
    kms : float or array-like
        The velocity value(s) in kilometers per second.
    line_center : float
        The central (reference) wavelength of the spectral line.
    c : float, optional
        The speed of light in km/s. The default value is 2.99792458e5 km/s.

    Returns:
    --------
    wl : float or array-like
        The calculated wavelength shift corresponding to the input velocity.
    """
    wl=kms*line_center/c
    return wl
@jit
def wl_to_kms(wl,line_center,c= 2.99792458e5):
    """
    Convert a velocity in km/s to a wavelength shift based on the line center.

    Parameters:
    -----------
    wl : float or array-like
        The calculated wavelength shift corresponding to the input velocity.
    
    line_center : float
        The central (reference) wavelength of the spectral line.
    c : float, optional
        The speed of light in km/s. The default value is 2.99792458e5 km/s.

    Returns:
    --------
    kms : float or array-like
        The velocity value(s) in kilometers per second.
    """
    kms = (wl*c)/line_center
    return kms

def vac_to_air(lam_vac):
    """
    Convert vacuum to air wavelengths

    :param lam_vac - Wavelength in Angstroms
    :return: lam_air - Wavelength in Angstroms

    """
    lam = np.asarray(lam_vac)
    sigma2 = (1e4/lam)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return lam_vac/fact

def get_EQW(flux_to_mini,baseline):
    #based as ussual in https://github.com/pyspeckit/pyspeckit/blob/4e1ed1c9c4759728cea04197d00d5c5f867b43f9/pyspeckit/spectrum/fitters.py#L357
    #sp_star_model.specfit.EQW(plot=True, plotcolor='g', fitted=False, components=False, annotate=True, loc='lower left')
    #take care of the units can be use normalize but you should be aware of that 
    #array of differences between x
    #baseline fited line of the continium
    _dxarr = jnp.concatenate([jnp.diff(flux_to_mini[0, :]),jnp.diff(flux_to_mini[0, :])[-1:]])
    diffspec = jnp.nansum(jnp.where(baseline == 0,jnp.nan,baseline-flux_to_mini[1, :])*jnp.nanmedian(_dxarr))
    continuum  = jnp.nanmedian(jnp.where(baseline == 0,jnp.nan,baseline))
    return diffspec/continuum#,diffspec,continuum,_dxarr

def get_EQW_with_mask(flux_to_mini,baseline,mask):
    #based as ussual in https://github.com/pyspeckit/pyspeckit/blob/4e1ed1c9c4759728cea04197d00d5c5f867b43f9/pyspeckit/spectrum/fitters.py#L357
    #sp_star_model.specfit.EQW(plot=True, plotcolor='g', fitted=False, components=False, annotate=True, loc='lower left')
    #take care of the units can be use normalize but you should be aware of that 
    #array of differences between x
    #baseline fited line of the continium
    #the mask should be a boolean array with false the values out side the region of interest
    #flux_over_cont = (1 - flux_to_mini[1, :]/baseline) * jnp.concatenate([jnp.diff(flux_to_mini[0, :]),jnp.diff(flux_to_mini[0, :])[-1:]])
    #return jnp.nansum(jnp.where(mask,0,flux_over_cont))
    baseline = baseline * (~mask).astype(jnp.float64)
    
    _dxarr = jnp.concatenate([jnp.diff(flux_to_mini[0, :]),jnp.diff(flux_to_mini[0, :])[-1:]])
    diffspec = jnp.nansum(jnp.where(baseline == 0,jnp.nan,baseline-flux_to_mini[1, :])*jnp.nanmedian(_dxarr))
    continuum  = jnp.nanmedian(jnp.where(baseline == 0,jnp.nan,baseline))
    return diffspec/continuum#,diffspec,continuum,_dxarr



vmap_get_EQW = vmap(get_EQW,in_axes=(0,0))

vmap_get_EQW_mask = vmap(get_EQW_with_mask,in_axes=(0,0,0))