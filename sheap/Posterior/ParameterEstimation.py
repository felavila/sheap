import os
from pathlib import Path

from astropy.cosmology import FlatLambdaCDM
from auto_uncertainties import Uncertainty
from jax import vmap,grad
import jax.numpy as jnp

import numpy as np
import pandas as pd
import yaml





from sheap.Posterior.utils import combine
from sheap.tools.others import vmap_get_EQW_mask
from sheap.SuportFunctions.functions import LineMapper,mapping_params
from sheap.FunctionsMinimize.utils import combine_auto
#from SHEAP.numpy.line_handling import line_decomposition_measurements,line_parameters
#from SHEAP.numpy.monte_carlo import monte_carlo
# this file’s directory: mypackage/submodule
here = Path(__file__).resolve().parent
data_file = here.parent / "SuportData" / "tabuled_values"/ "dictionary_values.yaml"

with open(data_file, 'r') as file:
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
    def __init__(self,RegionClass,fluxnorm=None,z=None,d=None,cosmo=None,multi_comp=None,c = 299792.458):
        """_summary_
        i think the best is if you are calling this you already have your flux in flux units "classical"
        Dimensional reduction in this step could be 2hard maybe we can move the combination for later steps
        Args:
            RegionClass (_type_): _description_
            fluxnorm (_type_, optional): _description_. Defaults to None.
            z (_type_, optional): _description_. Defaults to None.
            d (_type_, optional): _description_. Defaults to None.
            cosmo (_type_, optional): _description_. Defaults to None.
            c speed of light in km/s
        """
        ####
        #self.lines = RegionClass.lines
        #self.multi_comp = list(RegionClass.multi_comp)
        #self.multi_comp_index = RegionClass.mapping_params(self.multi_comp)
        ####
        self.c = c
        ####
        #self.RegionClass = 
        self.mask = RegionClass.mask #mmm
        self.spectra = RegionClass.spectra #mmm
        self.z = RegionClass.z
        ###
        complex_region_class = RegionClass.complex_region_class
        self.RegionMap = LineMapper(**complex_region_class.to_dict())
        for key, value in vars(complex_region_class).items():
             setattr(self, key, value)
        if cosmo is None:
             self.cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
        if fluxnorm is None:
             self.fluxnorm = np.ones(self.spectra.shape[0])
        # #warning is self.z non?
        self.d = self.cosmo.luminosity_distance(self.z) * cm_per_mpc
        # #aja e.e
        self.kinds_map = {}
        for k in self.kind_list:
            self.kinds_map[k] = self.RegionMap._get("kind",k)
    #
    def compute_params_wu(self):
        "ok. TODO: FWHMkm_s ADD THE fwhm in km/s"
        dict_ = {}
        for k,k_map in self.kinds_map.items():
            idx_amplitude = mapping_params(k_map.params_names,"amplitude")
            idx_width = mapping_params(k_map.params_names,"width")
            idx_center = mapping_params(k_map.params_names,"center")
            
            params = k_map.params
            uncertainty_params = k_map.uncertainty_params
            
            norm_amplitude = params[:,idx_amplitude]
            norm_amplitude_u = uncertainty_params[:,idx_amplitude]
            
            width = params[:,idx_width]
            width_u = uncertainty_params[:,idx_width]
            
            center = params[:,idx_center]
            center_u = uncertainty_params[:,idx_center]
            
            norm_amplitude = Uncertainty(norm_amplitude, norm_amplitude_u)
            width = Uncertainty(width, width_u)
            center = Uncertainty(center, center_u)
            
            fwhm = 2. * np.sqrt(2. * np.log(2.)) * width
            flux =   np.sqrt(2. * np.pi) *norm_amplitude * width * self.fluxnorm[:,None]
            L =  4. *np.pi* np.array(self.d[:,None]**2 )*flux*center
            
            dict_[k] = {'L':{'value':L.value, 'error':L.error}
                        ,'flux':{'value':flux.value, 'error':flux.error},
                        'fwhm':{'value':fwhm.value, 'error':fwhm.error}
                        ,'lines':k_map.line_name}
        self.dict_basic_params = dict_
        
    def compute_Luminosity_w(self,wavelenghts=[1350.,1450.,3000.,5100.,6200.]):
        map_cont = self.kinds_map['continuum']
        profile_func = map_cont.profile_functions_combine
        params = map_cont.params.T
        uncertainty_params = map_cont.uncertainty_params.T
        L_w = {}
        for w in wavelenghts:
            hits = jnp.isclose(self.spectra[:,0,:], w, atol=1)
            valid = (hits & (~self.mask)).any(axis=1, keepdims=True)       # only the un-masked 5100’ 
            grad_f = grad(lambda p: jnp.sum(profile_func(jnp.array([w]), p)))(params)
            flux = jnp.where(valid.squeeze(),profile_func(jnp.array([w]),params),0)
            sigma_f = jnp.where(valid.squeeze(),jnp.sqrt(jnp.sum((grad_f * uncertainty_params)**2,axis=0)),0) 
            flux = Uncertainty(np.array(flux), np.array(sigma_f))
            l = (w * 4. * np.pi * np.array(self.d**2) * flux).ravel()
            L_w[str(w)] = {'value':l.value,"error":l.error}
        return L_w
    def compute_bolometric_luminosity(self, monochromatic_lums=None, method="default"):
        """
         Estimate bolometric luminosity using bolometric correction factors.
            Common correction factors (e.g., Richards et al. 2006 or Netzer 2019) look for this references
         Args:
             monochromatic_lums: Dict of monochromatic luminosities [erg/s].
             method: Which correction to apply (default: empirical).
         Returns:
             Dict with bolometric luminosities for each wavelength.
         """
        if monochromatic_lums is None: #and hasattr(self, 'compute_Luminosity_w'):
            monochromatic_lums = self.compute_Luminosity_w()

        bolometric_corrections = {
                    "1350.0": 3.81,
                    "1450.0": 3.81,
                    "3000.0": 5.15,
                    "5100.0": 9.26,
                    "6200.0": 9.26
                }
        L_bol = {}
        
        for wave, L in monochromatic_lums.items():
            corr = bolometric_corrections.get(wave, 0.0)  # fallback if not listed
            L_bol[wave] = {'value':L['value'] * corr,'error':L['error'] * corr}
        return L_bol
    #def _calculate_Fe_flux(self, measure_range, pp):(https://github.com/legolason/PyQSOFit/blob/master/src/pyqsofit/PyQSOFit.py)
    
    def compute_black_hole_mass(self,f=1.0,non_combine=False):
        """
        Compute black hole mass estimates using various broad emission lines with standard virial estimators.

        Returns:
            Dictionary of black hole mass estimates [Msun] per line.
        """
        # Virial estimator parameters: log(M_BH/Msun) = a + b*log10(L/10^44 erg/s) + 2*log10(FWHM/1000 km/s)
        estimators = {
            "Hbeta":     {"wavelength": "5100.0", "a": 6.91, "b": 0.5, "f": f},  # Vestergaard & Peterson 2006
            "MgII":   {"wavelength": "3000.0", "a": 6.86, "b": 0.5, "f": f},  # Shen et al. 2011
            "CIV":    {"wavelength": "1350.0", "a": 6.66, "b": 0.53, "f": f}, # Vestergaard & Peterson 2006
            "Halpha":     {"wavelength": "6200.0", "a": 6.98, "b": 0.5, "f": f},  # Greene & Ho 2005
        }

        # Compute continuum luminosities and FWHMs
        dict_broad = self.compute_params_wu()['broad']
        L_w = dict_broad['L']
        fwhm = dict_broad['fwhm'] #in A
        #fwhm_kms = self.FWHMkm_s()
        masses = {}
        return 
        # for line_name, params in estimators.items():
        #     if line_name not in self.linelist:
        #         continue  # skip if line not fitted
        #     wave = params["wavelength"]
        #     if wave not in L_w:
        #         continue  # skip if continuum not available
        #     mapline_broad = self.linemap._get(["line_name","kind"],[line_name,"broad"])
        #     if self.n_broad>1 and not non_combine:
        #         print("combine")
        #         mapline_narrow = self.linemap._get(["line_name","kind"],[line_name,"narrow"])
        #         params_broad = mapline_broad["params"]
        #         params_narrow = mapline_narrow["params"]
        #         fwhm_kms,amplitude_final,center_final,sigma_final = combine(params_broad,params_narrow,limit_velocity=150.0,c=self.c)
        #     #lambda0 = mapline["center"]
        #     else:
        #         params_line = mapline_broad["params"]
        #         idx_widhts = [idx for idx,_  in enumerate(mapline_broad.get("params_names")) if "width" in _]
        #         idx_center = [idx for idx,_  in enumerate(mapline_broad.get("params_names")) if "center" in _]
        #         fwhm = 2. * jnp.sqrt(2. * jnp.log(2.)) * params_line[:,idx_widhts]
        #         fwhm_kms = jnp.nanmax((fwhm/jnp.array(params_line[:,idx_center]))*self.c,axis=1) #in km/s
        #     a, b, f = params["a"], params["b"], params["f"]
        #     log_L = jnp.log10(L_w[wave])  # continuum luminosity in erg/s
        #     log_FWHM = jnp.log10(fwhm_kms) - 3  # convert FWHM to 10^3 km/s
        #     log_M_BH = a + b * (log_L - 44.0) + 2 * log_FWHM
        #     masses[line_name] = (10 ** log_M_BH) / f  # in Msun
        #     masses[line_name] = log_M_BH
        # return masses
    
    # def compute_EW(self):
    #     profile_index_list = self.RegionClass.profile_index_list
    #     x_axis = self.RegionClass.region_to_fit[:,0,:]
    #     mask = self.RegionClass.mask_region
    #     #this is based on the idea about the continium is the last profile added to the code maybe could be a good idea have the exacto position of it 
    #     EW = []
    #     for i,profile in enumerate(self.RegionClass.profile_list):
    #         profile_func = vmap(self.RegionClass.profile_function_list[i],in_axes=(0, 0))#maybe ask is the previous one is other proflie?
    #         min_,max_ = profile_index_list[i]
    #         values = self.RegionClass.params[:,min_:max_]
    #         if profile != "linear" and "Fe" not in profile:
    #             #we filter after the fe      
    #             emission_line = profile_func(x_axis,values) 
    #             c = jnp.stack([x_axis, emission_line], axis=1)
    #             ew = vmap_get_EQW_mask(c,self.baselines,mask)
    #             EW.append(ew)
    #     else:
    #         pass 
    #     EW = jnp.stack(EW, axis=1)
    #     return EW
    
    # #def _calculate_Fe_flux(self, measure_range, pp):(https://github.com/legolason/PyQSOFit/blob/master/src/pyqsofit/PyQSOFit.py)
    # #important to know we have to separate lines from continums from Fe
    # def compute_black_hole_mass(self,f=1.0,non_combine=False):
    #     """
    #     Compute black hole mass estimates using various broad emission lines with standard virial estimators.

    #     Returns:
    #         Dictionary of black hole mass estimates [Msun] per line.
    #     """
    #     # Virial estimator parameters: log(M_BH/Msun) = a + b*log10(L/10^44 erg/s) + 2*log10(FWHM/1000 km/s)
    #     estimators = {
    #         "Hbeta":     {"wavelength": "5100.0", "a": 6.91, "b": 0.5, "f": f},  # Vestergaard & Peterson 2006
    #         "MgII":   {"wavelength": "3000.0", "a": 6.86, "b": 0.5, "f": f},  # Shen et al. 2011
    #         "CIV":    {"wavelength": "1350.0", "a": 6.66, "b": 0.53, "f": f}, # Vestergaard & Peterson 2006
    #         "Halpha":     {"wavelength": "6200.0", "a": 6.98, "b": 0.5, "f": f},  # Greene & Ho 2005
    #     }

    #     # Compute continuum luminosities and FWHMs
    #     L_w = self.compute_Luminosity_w()
    #     #fwhm_kms = self.FWHMkm_s()
    #     masses = {}

    #     for line_name, params in estimators.items():
    #         if line_name not in self.linelist:
    #             continue  # skip if line not fitted
    #         wave = params["wavelength"]
    #         if wave not in L_w:
    #             continue  # skip if continuum not available
    #         mapline_broad = self.linemap._get(["line_name","kind"],[line_name,"broad"])
    #         if self.n_broad>1 and not non_combine:
    #             print("combine")
    #             mapline_narrow = self.linemap._get(["line_name","kind"],[line_name,"narrow"])
    #             params_broad = mapline_broad["params"]
    #             params_narrow = mapline_narrow["params"]
    #             fwhm_kms,amplitude_final,center_final,sigma_final = combine(params_broad,params_narrow,limit_velocity=150.0,c=self.c)
    #         #lambda0 = mapline["center"]
    #         else:
    #             params_line = mapline_broad["params"]
    #             idx_widhts = [idx for idx,_  in enumerate(mapline_broad.get("params_names")) if "width" in _]
    #             idx_center = [idx for idx,_  in enumerate(mapline_broad.get("params_names")) if "center" in _]
    #             fwhm = 2. * jnp.sqrt(2. * jnp.log(2.)) * params_line[:,idx_widhts]
    #             fwhm_kms = jnp.nanmax((fwhm/jnp.array(params_line[:,idx_center]))*self.c,axis=1) #in km/s
    #         a, b, f = params["a"], params["b"], params["f"]
    #         log_L = jnp.log10(L_w[wave])  # continuum luminosity in erg/s
    #         log_FWHM = jnp.log10(fwhm_kms) - 3  # convert FWHM to 10^3 km/s
    #         log_M_BH = a + b * (log_L - 44.0) + 2 * log_FWHM
    #         masses[line_name] = (10 ** log_M_BH) / f  # in Msun
    #         masses[line_name] = log_M_BH
    #     return masses
    
    #     #     wave = params["wavelength"]
    #     #     if wave not in L_w:
    #     #         continue  # skip if continuum not available

    #     #     a, b, f = params["a"], params["b"], params["f"]
    #     #     idx = self.RegionClass.lines_list.index(line_name)
    #     #     log_L = jnp.log10(L_w[wave])  # continuum luminosity in erg/s
    #     #     log_FWHM = jnp.log10(fwhm_kms[:, idx]) - 3  # convert FWHM to 10^3 km/s

    #     #     log_M_BH = a + b * (log_L - 44.0) + 2 * log_FWHM
    #     #     masses[line_name] = (10 ** log_M_BH) / f  # in Msun

    #     # return masses

    
    
    # # def FWHMkm_s(self):
    # #     if not hasattr(self, 'fwhm'):
    # #         self.fwhm = self.compute_fwhm()
    # #     lambda0 = self.RegionClass.initial_params.at[self.RegionClass.mapping_params(["center"])].get()
    # #     fwhmkms =  (self.fwhm*self.c)/lambda0
    # #     return fwhmkms
    
    # # def velocityshift(self):
    # #     #mmm
    # #     lambda0 = self.RegionClass.initial_params.at[self.RegionClass.mapping_params(["center"])].get()
    # #     velocityshift_ = ((self.center-lambda0)/lambda0)*self.c
    # #     return velocityshift_
    
    # @property
    # def panda_fwhm(self):
    #     if not hasattr(self, 'fwhm'):
    #         self.fwhm = self.compute_fwhm()
    #     return pd.DataFrame(self.fwhm,columns=self.RegionClass.lines_list) 
    
    # @property
    # def panda_luminosity(self):
    #     if not hasattr(self, 'luminosity'):
    #         self.luminosity = self.compute_luminosity()
    #     return pd.DataFrame(self.luminosity, columns=self.RegionClass.lines_list)
    
    # @property
    # def panda_flux(self):
    #     if not hasattr(self, 'flux'):
    #         self.flux = self.compute_flux()
    #     return pd.DataFrame(self.flux,columns=self.RegionClass.lines_list)
    
    # @property 
    # def panda_EW(self):
    #     if not hasattr(self,"EW"):
    #         self.EW = self.compute_EW()
    #     return pd.DataFrame(self.EW,columns=self.RegionClass.lines_list)
    
    # @property
    # def panda_fwhmkm_s(self):
    #     fwhmkms = self.FWHMkm_s()
    #     return pd.DataFrame(fwhmkms,columns=self.RegionClass.lines_list)
    
    # @property
    # def panda_velocityshift(self):
    #     return pd.DataFrame(self.velocityshift(),columns=self.RegionClass.lines_list) 
    
    
    
    
    

    
    
    
    
    
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