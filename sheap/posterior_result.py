import os 
import yaml
from astropy.cosmology import FlatLambdaCDM
import jax.numpy as jnp 

#from SHEAP.numpy.line_handling import line_decomposition_measurements,line_parameters
#from SHEAP.numpy.monte_carlo import monte_carlo


module_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(module_dir,"suport_data/tabuled_values/dictionary_values.yaml"), 'r') as file:
    #this should be done more eficiently in some way
    #TODO add references to this 
    tabuled_values = yaml.safe_load(file)
    continuum_bands,logK,alpha,slope,A,B = tabuled_values.values()
    

#fwhm,fwhm_low,fwhm_up, luminosity,EW,EW_low,EW_up,dlambda,dlambda_low,dlambda_up,lambda0,c_total,conti,varr,xarr,std,ske,kurto,dlmax,dl50,dl90,dl95,dlt = line_parameters(results,model_lines,line_dictionary,line_name =line_name)
#fwmin,fwmax,l1,l2,dv1,dv2,fwhmine,fwhmaxe,l1e,l2e,dv1e,dv2e=line_decomposition_measurements(model_lines,line_dictionary,line_name =line_name)
    #iterations_per_object = 100

cm_per_mpc = 3.08568e+24

class ParameterEstimation:
    #TODO big how to combine distributions 
    def __init__(self,RegionClass,fluxnorm=None,z=None,d=None,cosmo=None,multi_comp=None):
        """_summary_
        Dimensional reduction in this step could be 2hard maybe we can move the combination for later steps
        Args:
            RegionClass (_type_): _description_
            fluxnorm (_type_, optional): _description_. Defaults to None.
            z (_type_, optional): _description_. Defaults to None.
            d (_type_, optional): _description_. Defaults to None.
            cosmo (_type_, optional): _description_. Defaults to None.
        """
        ####
        self.lines = RegionClass.lines
        self.multi_comp = list(RegionClass.multi_comp)
        self.multi_comp_index = RegionClass.mapping_params(self.multi_comp)
        ####
        self.sigma = RegionClass.params.at[:,RegionClass.mapping_params(["width"])].get()
        self.norm_amplitude = RegionClass.params.at[:,RegionClass.mapping_params(["amplitude"])].get()
        self.center = RegionClass.params.at[:,RegionClass.mapping_params(["center"])].get()
        ####
        self.d = None
        if d is not None:
            self.d = d
        if z is not None:
            if cosmo is None:
                self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
            self.d = self.cosmo.luminosity_distance(z) * cm_per_mpc
        if fluxnorm is None:
            self.fluxnorm = jnp.ones_like(self.d)
        else:
            self.fluxnorm = fluxnorm
        
        
    def __call__(self):
        """
        Calculate luminosity and FWHM for all spectral lines.
        """
        self.flux = self.compute_flux()
        self.fwhm = self.compute_fwhm() 
        self.amplitude = self.compute_amplitude()
        self.luminosity = self.compute_luminosity()
            
            # for line in self.lines.keys():
            #     if self.debug:
            #         print("Computing parameters for line %s" % line)

            #     self.lines[line]['fwhm'] = self.compute_fwhm(self.lines[line]['modelpars'])
            #     self.lines[line]['flux'] = self.compute_flux(self.lines[line]['modelpars'])
            #     self.lines[line]['amp'] = self.compute_amplitude(self.lines[line]['modelpars'])
            #     self.lines[line]['pos'] = self.lines[line]['modelpars'][1]
    
    def compute_flux(self):
        """
        Calculate integrated flux of emission line Unnormalized.
        """
            
        flux =  jnp.sqrt(2. * jnp.pi) * self.norm_amplitude * self.sigma 
        return flux * self.fluxnorm[:,None]

    def compute_amplitude(self):
        """
        Calculate amplitude of emission line.  Should be easy - add multiple components if they exist.
        Currently assumes multiple components have the same centroid.
        """
        return self.norm_amplitude * self.fluxnorm[:,None]

    def compute_luminosity(self):
        """
        Determine luminosity of line (need distance and flux units).
        """
        return 4. * jnp.pi * self.d[:,None]**2 * self.flux

    def compute_fwhm(self):
        """
        Determine full-width at half maximum in A
        
        """
        return 2. * jnp.sqrt(2. * jnp.log(2.)) * self.sigma
    
    
    
    
    
    
    
    
    
    
    
    
# def calculate_FWHM(region_class,line,params_region,limit_velocity=150,limit_ratio=0.1,broad_number=2,c=299792):
#     """
#     TODO ERRORS 
#     TODO bugs in some systems 
#     Depende, si es para calcular la masa del agujero negro debes considerar lo siguiente:
#     - si la diferencia en velocidad entre las dos componentes es importante (>150km/s), solo usa la que está en la más cercana a la velocidad de las narrow
#     - si las dos están en la misma o similar velocidad, y el flujo de la menos intensa es por lo menos >10% de la más intensa, ajusta las dos componentes como una sola gaussiana y usa ese FWHM. En caso de que el flujo sea  <10%, solo usa el FWHM de la más intensa
#     """
#     index_params_broad = region_class.mapping_params(["broad",line])
#     index_center_broad = region_class.mapping_params(["center","broad"])
#     index_amplitude_broad = region_class.mapping_params(["amplitude","broad",line])
#     index_center_narrow =region_class.mapping_params(["center","narrow",line])
#     delta_center = ((params_region[:,index_center_broad] - params_region[:,index_center_narrow].ravel()[:,None])/params_region[:,index_center_narrow].ravel()[:,None])
#     relative_velocity = abs(jnp.diff(delta_center,axis=1).ravel())*c
#     non_virialized_index = jnp.where(relative_velocity>=limit_velocity)[0]
#     virialized_index = jnp.where(relative_velocity<limit_velocity)[0]
#     #####
#     params_non_virialized = params_region[jnp.ix_(non_virialized_index, index_params_broad)]
#     params_non_virialized = params_non_virialized.reshape(params_non_virialized.shape[0], broad_number, -1)
#     argmin_center = jnp.argmin(abs(delta_center[non_virialized_index]),axis=1).ravel()
#     row_indices = jnp.arange(argmin_center.shape[0])
#     params_non_virialized = params_non_virialized[row_indices,argmin_center]
#     FWHM_non_virialized = params_non_virialized[:,2]*2.35482/params_non_virialized[:,1] # here is assume the position of sigma
#     params_amplitude_virialized = params_region[jnp.ix_(virialized_index, index_amplitude_broad)]
#     argmax = jnp.argmax(params_amplitude_virialized,axis=1).ravel()
#     argmin = jnp.argmin(params_amplitude_virialized,axis=1).ravel()
#     row_indices = jnp.arange(params_amplitude_virialized.shape[0])
#     max_vals = params_amplitude_virialized[row_indices, argmax]
#     min_vals = params_amplitude_virialized[row_indices, argmin]
#     ratio = min_vals / max_vals
#     print(jnp.where(jnp.isnan(ratio)))
#     #index_to_comb = jnp.where(ratio>limit_ratio)[0]
#     params_virialized = params_region[jnp.ix_(virialized_index, index_params_broad)]
#     params_virialized_r = params_virialized.reshape(params_virialized.shape[0], broad_number, -1)
#     params_virialized_r = params_virialized_r[row_indices,argmax]    
#     FWHM_virialized = jnp.where(ratio>limit_ratio,effective_fwhm(params_virialized.T),params_virialized_r[:,2]*2.35482/params_virialized_r[:,1])    
#     #FWHM_virialized = jnp.where(jnp.isnan(ratio>limit_ratio),0,FWHM_virialized)  
#     FWHM = jnp.zeros_like(relative_velocity)
#     FWHM = FWHM.at[non_virialized_index].set(FWHM_non_virialized)
#     FWHM = FWHM.at[virialized_index].set(FWHM_virialized)
#     return FWHM*c

# def effective_fwhm(params):
#     """
#     Estimate the FWHM of a combined two-Gaussian profile using moment analysis.
    
#     params: tuple or array-like with 6 elements:
#         (amp1, mu1, sigma1, amp2, mu2, sigma2)
#     Returns:
#         Effective FWHM computed as 2*sqrt(2*ln2)*sigma_eff.
#     """
#     amp1, mu1, sigma1, amp2, mu2, sigma2 = params

#     # Compute the effective mean
#     total_amp = amp1 + amp2
#     mu_eff = (amp1 * mu1 + amp2 * mu2) / total_amp

#     # Compute the effective variance (includes the variance from the spread of the means)
#     var_eff = (amp1 * (sigma1**2 + (mu1 - mu_eff)**2) +
#                amp2 * (sigma2**2 + (mu2 - mu_eff)**2)) / total_amp

#     sigma_eff = jnp.sqrt(var_eff)
#     fwhm = 2.35482 * sigma_eff
#     return fwhm/mu_eff