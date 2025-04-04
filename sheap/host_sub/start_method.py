import jax.numpy as jnp
import jax.scipy as jsp
from SHEAP.utils import mask_builder
from SHEAP.fitting.main_fitting_I import MasterMinimizer
from SHEAP.fitting.functions import linear,GaussianSum
from SHEAP.fitting.utils import combine_auto
from SHEAP.tools.others import vmap_get_EQW_mask
import os 
from pathlib import Path
import numpy as np
from jax import vmap,jit
"""code to substarct the galactic component of an AGN, asuming that a template of a K-giant star is sufficient.
Following the method explained in Greene & Ho 2005b and Kim & Ho 2006 (https://ui.adsabs.harvard.edu/abs/2006ApJ...642..702K/abstract).
translation from Paula Sanchez's code . 
"""

module_dir = Path(__file__).resolve().parent.parent
#TODO steal appears systems that contain negative values in the host substation are like 5 in the santiago sample looking for solutions or assume the limit in the method maybe 

def host_flux(flux_ew,EWfin,waveq):
    """
    Parameters:
        flux_ew_star
    """
    flux_ew_star = 4000.2491299225276
    EW_star_final =  -16 #arbitrary original
    wave_kstar,flux_kstar,err_kstar =jnp.array(np.loadtxt(os.path.join(module_dir,'suport_data/templates/Kstar.txt'),dtype='float').transpose())
    spectra_exp = jnp.round(jnp.log10(jnp.mean(flux_kstar)))
    log_flux_kstar,log_err_kstar = flux_kstar * 10**(-spectra_exp),err_kstar * 10**(-spectra_exp)
    EW_star_final,flux_ew_star = -23.28276937,0.39887047 #sheap minimization using limits outer_limits=[3908,3960],inner_limits=[3915,3950]
    h = (flux_ew/flux_ew_star)*(EWfin/EW_star_final)*jnp.interp(waveq,wave_kstar,log_flux_kstar)
    h = jnp.where(waveq<min(wave_kstar),0.0,h)
    return h

vmap_host_flux = vmap(host_flux,in_axes=(0,0,0),out_axes=0)



def Extract_host_star(Spectra: jnp.array,c=2.99792458e5,outer_limits=[3908,3960],inner_limits=[3915,3950],vel_resolution= 69., vel_limit = 1000.,signal_noise_region_limit = 3,AN_limit = 3, EWfin_limit = 0,
                          constraints=jnp.array([[-1e41,0.0],[3908,3960],[0,1e41],[-1e41,+1e41],[-1e41,1e41]]),num_steps=100,developing=False):
    """
    Processes a set of spectra data.

    Parameters:
    Spectra (jax.numpy.ndarray): A JAX array of shape (X, 3, N) where:
        - X: Number of spectra.
        - 3: Represents [wavelength, flux, error].
        - N: Number of pixels in each spectrum.
    c speed of light on km/h
    Returns:
    None: Modify this as needed based on what the function should return.
    """
    g_c = jit(combine_auto([GaussianSum(n=1, constraints={}),linear]))
    _, _,_,mask_fit = mask_builder(Spectra,inner_limits=inner_limits,outer_limits=outer_limits)
    c_ca =sum(outer_limits)/2# where to measure the continuum
    fit_region_g, masked_uncertainties_g,_,mask_fit_g = mask_builder(Spectra,outer_limits=outer_limits)
    signal_noise_region = jnp.nanmedian(jnp.where(mask_fit_g,jnp.nan,Spectra[:,1,:]/Spectra[:,2,:]),axis=1)
    min_value = jnp.nanmin(jnp.where(mask_fit_g, jnp.inf, Spectra[:, 1, :]),axis=1)
    median_region = jnp.nanmedian(jnp.where(mask_fit,jnp.nan,Spectra[:,1,:]),axis=1)
    initial_params_g = jnp.array([(min_value-median_region)*1.2,jnp.ones(min_value.shape)*c_ca,jnp.ones(min_value.shape)*4,0.*jnp.ones(Spectra.shape[0]),median_region]).T
    #
    MasterFit = MasterMinimizer(g_c, non_optimize_in_axis=4,num_steps=num_steps)
    params_g,_ = MasterFit.vmap_optimize_model(initial_params_g,fit_region_g[:, 0, :],fit_region_g[:, 1, :],masked_uncertainties_g,constraints,*MasterFit.default_args) 
    line_center,sigma_jax = params_g[:,[1,2]].T
    params_linear = params_g[:,-2:]
    #
    vmap_linear = vmap(linear, in_axes=(0, 0), out_axes=0) 
    Baselines = vmap_linear(fit_region_g[:, 0, :],params_linear)
    #
    flux_ew = vmap_linear(line_center,params_linear)
    amplitude_star_jax = params_g[:,0] + flux_ew #params_g is negative
    #
    EWfin = jnp.where(jnp.isnan(line_center),10,-1.0 * vmap_get_EQW_mask(Spectra,Baselines,mask_fit_g))
    ####
    vel = c*sigma_jax/c_ca
    AN  =   jnp.abs(amplitude_star_jax/jnp.nanmedian(jnp.where(mask_fit_g,jnp.nan,masked_uncertainties_g),axis=1))
    host_detected = jnp.where(((vel>=vel_resolution) & (vel<=vel_limit) & (EWfin<=EWfin_limit) & (signal_noise_region_limit<=signal_noise_region)) == True )[0]
    # jnp.vstack([EWfin,vel,signal_noise_region,AN]).T parameters
    host_flux = jnp.zeros_like(Spectra[:,0,:]) # make a host flux array of shape equal to the flux array
    host_flux = host_flux.at[host_detected].set(vmap_host_flux(flux_ew[host_detected],EWfin[host_detected],Spectra[host_detected,0,:]))
    if developing:
        return host_detected,fit_region_g,mask_fit,masked_uncertainties_g,outer_limits,MasterFit,params_g,mask_fit_g,Baselines,AN,EWfin,vel,signal_noise_region,params_linear,initial_params_g,host_flux
    return host_flux

    
    

# def Extract_host_star_old(Spectra: jnp.array,c=2.99792458e5,outer_limits=[3908,3960],inner_limits=[3915,3950]):
#     """
#     Processes a set of spectra data.

#     Parameters:
#     Spectra (jax.numpy.ndarray): A JAX array of shape (X, 3, N) where:
#         - X: Number of spectra.
#         - 3: Represents [wavelength, flux, error].
#         - N: Number of pixels in each spectrum.
#     c speed of light on km/h
#     Returns:
#     None: Modify this as needed based on what the function should return.
#     """
#     c_ca =sum(outer_limits)/2# where to measure the continuum
#     fit_region, masked_uncertainties,_,mask_fit = mask_builder(Spectra,inner_limits=inner_limits,outer_limits=outer_limits)
#     ###
#     median_region = jnp.nanmedian(jnp.where(mask_fit,jnp.nan,Spectra[:,1,:]),axis=1)
#     initial_params = jnp.array([jnp.zeros(Spectra.shape[0])*1,median_region]).T
#     Master_Linear = MasterMinimizer(linear, non_optimize_in_axis=4)
#     constraints = None
#     params_linear,_ = Master_Linear.vmap_optimize_model(initial_params,fit_region[:, 0, :],fit_region[:, 1, :],masked_uncertainties,constraints,*Master_Linear.default_args)
#     Baselines = Master_Linear.vmap_func(Spectra[:,0,:],params_linear)
#     ###
#     fit_region_g, masked_uncertainties_g,_,mask_fit_g = mask_builder(Spectra,outer_limits=outer_limits)
#     signal_noise_region = jnp.nanmedian(jnp.where(mask_fit_g,jnp.nan,Spectra[:,1,:]/Spectra[:,2,:]),axis=1)
#     min_value = jnp.min(jnp.where(mask_fit_g, jnp.inf, Spectra[:, 1, :] - Baselines),axis=1)
#     initial_params_g = jnp.array([(min_value-median_region)*jnp.sqrt(2*jnp.pi),jnp.ones(min_value.shape)*sum(outer_limits)/2,jnp.ones(min_value.shape)* 20]).T
#     constraints=jnp.array([[-1e41,0.0],outer_limits,[0,1e41]])
#     Master_Gaussian = MasterMinimizer(GaussianSum(n=1, constraints={}), non_optimize_in_axis=4)
#     params_g,_ = Master_Gaussian.vmap_optimize_model(initial_params_g,fit_region_g[:, 0, :],fit_region_g[:, 1, :] - Baselines,masked_uncertainties_g,constraints,*Master_Gaussian.default_args) 
#     ###
#     line_center,sigma_jax = params_g[:,[1,2]].T
#     flux_ew = Master_Linear.vmap_func(line_center,params_linear)
#     amplitude_star_jax = ((params_g[:,0])/(jnp.sqrt(2*jnp.pi*params_g[:,2]**2))) + flux_ew
#     #full_model = Master_Gaussian.vmap_func(Spectra[:,0,:],params_g) + Baselines #gaussian+baseline
#     EQW_jax_c = jnp.where(jnp.isnan(line_center),0,-1.0 * vmap_get_EQW_mask(Spectra,Baselines,mask_fit_g))
#     ####
#     vel = c*sigma_jax/c_ca
#     AN  =   jnp.abs(amplitude_star_jax/jnp.nanmedian(jnp.where(mask_fit_g,jnp.nan,masked_uncertainties_g),axis=1))
#     #EWint= jnp.where(jnp.isnan(line_center),0,-1.0*jsp.integrate.trapezoid(jnp.where(mask_fit_g,0,1-(full_model)/Baselines),x=jnp.where(mask_fit_g,0,Spectra[:,0,:]),axis=1)) # this implementation could be not the best
#     #EWfin=jnp.maximum(EQW_jax_c,EWint)
#     #EWfin = EQW_jax_c
#     #what is AN?
#     #constants
#     vel_resolution = 69.
#     vel_limit = 690.
#     signal_noise_region_limit = 5#10
#     AN_limit = 3
#     EWfin_limit = 0.0 #limit that came from the original code
#     #(AN>AN_limit) &
#     #index = jnp.where(((vel>=vel_resolution) & (vel<=vel_limit) & (EQW_jax_c<=EWfin_limit)) == True)[0]
#     index = jnp.where(((EQW_jax_c<=EWfin_limit) & (vel>=vel_resolution) & (vel<=vel_limit)) == True)[0]
#     return index,fit_region_g,mask_fit,masked_uncertainties_g,outer_limits,Master_Gaussian,params_g,mask_fit_g,Baselines,AN,EQW_jax_c,vel,signal_noise_region,params_linear,_
#     #return index
    
#     #return 
#     host_flux = jnp.zeros_like(Spectra[:,0,:]) # make a host flux array of shape equal to the flux array
#     host_flux = host_flux.at[index].set(vmap_host_flux(flux_ew[index],EWfin[index],Spectra[index,0,:]))
#     #spectras  = test_clase.spectra.at[:,1,:].subtract(host_flux)
#     return host_flux

#index = jnp.where(((AN>3) & (EWfin<=-1.5) & (vel>=vel_resolution) & (vel<=vel_limit) & (signal_noise_region>10)) == True)[0]
# if (np.abs(AN)>3) and (EWfin<=-1.5) and (vel>=vel_resolution) and (vel<=vel_lim):
#         #the host galaxy component is estimated
#         flux_gal=(flux_ew/flux_ew_star)*(EWfin/EW_star_final)*np.interp(waveq,wavek,fluxk)
#         flux_gal[np.where(waveq<minwavek)]=0.0