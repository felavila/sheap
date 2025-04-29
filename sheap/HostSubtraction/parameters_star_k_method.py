import os
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from SHEAP.fitting.functions import GaussianSum, combine_auto, linear2
from SHEAP.fitting.main_fitting_I import MasterMinimizer
from SHEAP.host_sub.start_method import Extract_host_star_new
from SHEAP.tools.others import vmap_get_EQW_mask
from SHEAP.utils import mask_builder

module_dir = Path(__file__).resolve().parent.parent
#Small code that can use to check how we got the values in the star method 

wave_kstar,flux_kstar,err_kstar =jnp.array(np.loadtxt(os.path.join(module_dir,'suport_data/templates/Kstar.txt'),dtype='float').transpose())
spectra_exp = jnp.round(jnp.log10(jnp.mean(flux_kstar)))
log_flux_kstar,log_err_kstar = flux_kstar * 10**(-spectra_exp),err_kstar * 10**(-spectra_exp)
spectra = jnp.array([wave_kstar,log_flux_kstar,log_err_kstar])
if len(spectra.shape)==2:
    spectra = spectra[jnp.newaxis,:]
g_c = combine_auto([GaussianSum(n=1, constraints={}),linear2])
outer_limits,inner_limits=[3908,3960],[3915,3950]
_, _,_,mask_fit = mask_builder(spectra,inner_limits=inner_limits,outer_limits=outer_limits)
c_ca =sum(outer_limits)/2# where to measure the continuum
fit_region_g, masked_uncertainties_g,_,mask_fit_g = mask_builder(spectra,outer_limits=outer_limits)
signal_noise_region = jnp.nanmedian(jnp.where(mask_fit_g,jnp.nan,spectra[:,1,:]/spectra[:,2,:]),axis=1)
min_value = jnp.nanmin(jnp.where(mask_fit_g, jnp.inf, spectra[:, 1, :]),axis=1)
median_region = jnp.nanmedian(jnp.where(mask_fit,jnp.nan,spectra[:,1,:]),axis=1)
constraints=jnp.array([[-1e41,0.0],[3908,3960],[0,1e41],[-1e41,-3],[-1e41,1e41]])
initial_params_g = jnp.array([(min_value-median_region)*1.2,jnp.ones(min_value.shape)*c_ca,jnp.ones(min_value.shape)*4,-10.*jnp.ones(spectra.shape[0]),median_region]).T
num_steps = 1000 
Master_Gaussian = MasterMinimizer(g_c, non_optimize_in_axis=4,num_steps=num_steps)
params_g,_ = Master_Gaussian.vmap_optimize_model(initial_params_g,fit_region_g[:, 0, :],fit_region_g[:, 1, :],masked_uncertainties_g,constraints,*Master_Gaussian.default_args) 
line_center,sigma_jax = params_g[:,[1,2]].T
params_linear = params_g[:,-2:]
vmap_linear = vmap(linear2, in_axes=(0, 0), out_axes=0) 
Baselines = vmap_linear(fit_region_g[:, 0, :],params_linear)
flux_ew = vmap_linear(line_center,params_linear)
amplitude_star_jax = params_g[:,0] + flux_ew #params_g is negative
EWfin = jnp.where(jnp.isnan(line_center),10,-1.0 * vmap_get_EQW_mask(spectra,Baselines,mask_fit_g))
print(f"flux_ew:{flux_ew},EWfin:{EWfin}")
plt.plot(*spectra[0][[0,1],:])
plt.plot(jnp.linspace(*outer_limits,1000),g_c(jnp.linspace(*outer_limits,1000),params_g[0]))
plt.xlim(outer_limits)
plt.show()