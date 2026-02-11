"""
Master combine Profile
====================================
#TODO cleaning 

"""

__author__ = 'felavila'

__all__ = ["combine_components","combine_fast","combine_fast_with_jacobian"]

from typing import Any, Dict, List, Union,Iterable
import numpy as np
import jax.numpy as jnp
from jax import vmap,jit,jacfwd,lax
from uncertainties import unumpy



from sheap.SheaProducts.Utils.Physical_functions import calc_flux,calc_luminosity
from sheap.Utils.Constants import DEFAULT_C_KMS
from sheap.Profiles.Utils import GaussianSum
from sheap.SheaProducts.Utils.Sample_handlers import concat_dicts_combine


class MasterCombineProfile:
	def __init__(self,LINES_TO_COMBINE=("Halpha", "Hbeta","MgII","CIV"), limit_velocity=150.0,C_KMS=DEFAULT_C_KMS,full_cont_profile=None,ucont_params = None
              ,full_cont_profile_NONE=None):
		
		#here the class should know if we are working with sapled values or just -> 1D
		#this works by object
		#here we need a "master key" that separate the single with the sampled values.
		#TODO all the part for all the objects at the same time.
		self.LINES_TO_COMBINE = LINES_TO_COMBINE
		self.limit_velocity = limit_velocity
		self.C_KMS = C_KMS
		#this is the continuum of all the seudo continuum.
		self.continuum_func = None 
		self.ucont_params  = ucont_params
		self.full_cont_profile = full_cont_profile
		self.full_cont_profile_NONE = full_cont_profile_NONE
  
	def combine_both(self,basic_params,distances,full_cont_params):
		combine_k = self.combine_kinematic_all(basic_params,distances,full_cont_params)
		combine_c = self.combine_classical_all(basic_params,distances,full_cont_params)

		return {"basic_params_combined_kinematic": concat_dicts_combine(combine_k),"basic_params_combined_classical": concat_dicts_combine(combine_c)}
 
	def combine_kinematic_all(self,basic_params,distances,full_cont_params):
		out = {}
		for line in self.LINES_TO_COMBINE:
			L = combine_kinematic(basic_params=basic_params,line=line,limit_velocity=self.limit_velocity,full_cont_profile=self.full_cont_profile,full_cont_params=full_cont_params,C_KMS=self.C_KMS, distances=distances, ucont_params=self.ucont_params,)	
			if L is not None:
				out[line] = L
		return out
	
	def combine_classical_all(self,basic_params,distances,full_cont_params):
		out = {}
		for line in self.LINES_TO_COMBINE:
			L = combine_classical(basic_params=basic_params,line=line, 
                         C_KMS=self.C_KMS, distances=distances,full_cont_profile=self.full_cont_profile_NONE,full_cont_params=full_cont_params,)
			if L is not None:
				out[line] = L
		return out

	def _get_index(self,basic_params):
		broad_lines = basic_params.get("broad", {}).get("lines",[])
		narrow_lines = basic_params.get("narrow", {}).get("lines",[])
		dic_idx_broad = {line:[i for i,L in enumerate(broad_lines) if L.lower() == line.lower()] for line in self.LINES_TO_COMBINE}
		dic_idx_narrow = {line:[i for i,L in enumerate(narrow_lines) if L.lower() == line.lower()] for line in self.LINES_TO_COMBINE}
		return dic_idx_broad,dic_idx_narrow

	def _from_any(self, src: object) -> None:
		for name in self._BASE_REQUIRED:
			setattr(self, name, getattr(src, name, None))

		if hasattr(src, "BOL_CORRECTIONS"):
			self.BOL_CORRECTIONS = src.BOL_CORRECTIONS
		if hasattr(src, "SINGLE_EPOCH_ESTIMATORS"):
			self.SINGLE_EPOCH_ESTIMATORS = src.SINGLE_EPOCH_ESTIMATORS
		if hasattr(src, "C_KMS"):
			self.C_KMS = src.C_KMS

	def _require(self, names: Iterable[str]) -> None:
		missing = [n for n in names if getattr(self, n, None) is None]
		if missing:
			raise ValueError(f"SheaProducts is missing required fields: {missing}")
 
 

def combine_classical(basic_params,line,distances=0,full_cont_profile=None,full_cont_params=None,
                      C_KMS=DEFAULT_C_KMS,DEFAULT_lambda_ref={"Halpha": 6564.61,  "Hbeta": 4862.68,"MgII": 2798.75,"CIV": 1549.48}):
	#This method rellay in belibe in the redshift
	#re-interpretation of https://github.com/legolason/PyQSOFit/issues/?
	#TODO single etc etc 
	#continuum_func_1D -> is one 1D because we are using it over the (0,None) axis
	#This method dosent have eerrors because dosent make sence.
	lambda_ref = DEFAULT_lambda_ref[line]
	
	broad_params = basic_params.get("broad", {})
	idx_broad = [i for i,L in enumerate(broad_params.get("lines",[])) if L.lower() == line.lower()]
 
	if len(idx_broad) < 2:
		return
	broad_params = basic_params["broad"]
	components =  np.array(broad_params["component"])[idx_broad]
	gg = GaussianSum(len(idx_broad))
	#check in comming iterations if this generate overcharge <-
	b_mu = jnp.asarray(unumpy.nominal_values(broad_params["center"]))[:,idx_broad].astype(jnp.float32)
	b_sigma = jnp.asarray(unumpy.nominal_values(broad_params["fwhm"]))[:,idx_broad].astype(jnp.float32) /  (2.0 * np.sqrt(2.0 * np.log(2.0)))
	b_amp   = jnp.asarray(unumpy.nominal_values(broad_params["amplitude"]))[:,idx_broad].astype(jnp.float32)    # (Nobj, NB)
	
	_ = np.stack([b_amp, b_mu,b_sigma], axis=2)
	line_params = jnp.array(_.transpose(0, 2, 1).reshape(_.shape[0], -1)).astype(jnp.float32)
	left = np.min(b_mu - 3*b_sigma,axis=1)
	right = np.max(b_mu + 3*b_sigma,axis=1)

	disp = 1.e-4 #hyperparam 
	npix = 50_000 #int(max((right-left)/disp))  #(maybe it is 2 much)

	wave = jnp.linspace(np.min(left), np.max(right), npix, dtype=jnp.float32)
	model_sum = vmap(gg,in_axes=(None,0))(wave,line_params).astype(jnp.float32)
	
	i_peak     = jnp.argmax(model_sum, axis=1)            
	#peak_A     = wave[i_peak]                         
	half       = 0.5 * jnp.max(model_sum, axis=1)     
	f          = model_sum - half[:, None]                 
							   
	Nlam   = wave.shape[0]
	idxs   = jnp.arange(Nlam - 1)                    
	eps = 1e-30

	def interp_at(k, f_row):
		# linear interpolation of zero crossing between k and k+1
		x0, x1 = wave[k],   wave[k + 1]
		y0, y1 = f_row[k],  f_row[k + 1]
		t = -y0 / (y1 - y0 + eps)
		return x0 + t * (x1 - x0)

	def row_fwhm(f_row, i_peak_i):
		s_row      = jnp.sign(f_row)                  # (Nlam,)
		cross_mask = (s_row[:-1] * s_row[1:] ) < 0    # (Nlam-1,)

		left_cand  = jnp.where((idxs < i_peak_i) & cross_mask, idxs, -1)
		left_idx   = jnp.max(left_cand)               # -1 if none

		right_cand = jnp.where((idxs >= i_peak_i) & cross_mask, idxs, Nlam)
		right_idx  = jnp.min(right_cand)              # Nlam if none

		has_left   = left_idx  >= 0
		has_right  = right_idx <= (Nlam - 2)

		lam_L = jnp.where(has_left,  interp_at(left_idx,  f_row), jnp.nan)
		lam_R = jnp.where(has_right, interp_at(right_idx, f_row), jnp.nan)

		return lam_L, lam_R

	lam_L, lam_R = vmap(row_fwhm, in_axes=(0, 0))(f.astype(jnp.float32), i_peak.astype(jnp.float32))   # (Nobj,), (Nobj,)

	fwhm_kms = ((lam_R - lam_L) / lambda_ref) * C_KMS   
	sigma_kms = fwhm_kms / (2.0 * np.sqrt(2.0 * np.log(2.0)))

	flux  = np.trapezoid(model_sum, wave, axis=1)
	luminosity = calc_luminosity(jnp.array(distances), flux)#just to be consisten with our self
	eqw = np.zeros_like(flux)
	if full_cont_profile:
		continuum_vals = full_cont_profile(wave,full_cont_params)
		cont_safe  = jnp.maximum(continuum_vals, 1e-30)
		eqw       = jnp.trapezoid(model_sum / cont_safe, wave, axis=1) 
	combined = {"component":components,"flux":flux,"sigma_kms":sigma_kms,"fwhm_kms":fwhm_kms,"luminosity":luminosity,"eqw":eqw} #more to add?
	return combined 

def combine_kinematic(basic_params,line, limit_velocity: float, C_KMS: float,full_cont_profile=None,full_cont_params=None,distances = 0,ucont_params=None):
	"""
	continuum_func -> a function that already has the continuum params integrated
 	"""
	broad_params = basic_params.get("broad", {})
	narrow_params = basic_params.get("narrow", {})
	
	idx_broad = [i for i,L in enumerate(broad_params.get("lines",[])) if L.lower() == line.lower()]
	idx_narrow = [i for i,L in enumerate(narrow_params.get("lines",[])) if L.lower() == line.lower()]
 
	if len(idx_broad) < 2 and  len(idx_narrow) != 1:
		return
	components =  np.array(broad_params["component"])[idx_broad]
	#broad
	amp_b = broad_params["amplitude"][:, idx_broad]
	mu_b = broad_params["center"][:, idx_broad]
	fwhm_kms_b = broad_params["fwhm_kms"][:, idx_broad]
	#narrow
	amp_n = narrow_params["amplitude"][:, idx_narrow]
	mu_n = narrow_params["center"][:, idx_narrow]
	fwhm_kms_n = narrow_params["fwhm_kms"][:, idx_narrow]
	#print(distances)
	if amp_b.dtype == 'O': #uncertainty rutines
		from sheap.SheaProducts.Utils.After_fit_profile_helpers import evaluate_with_error
		from sheap.SheaProducts.Utils.fwhm_conv import combine_fast_with_jacobian
		
		fwhm_c, amp_c, mu_c = combine_fast_with_jacobian(amp_b, mu_b, fwhm_kms_b,amp_n, mu_n, fwhm_kms_n,limit_velocity=limit_velocity,C_KMS=C_KMS)
		
		if fwhm_c.ndim==1:
			fwhm_c, amp_c, mu_c = fwhm_c.reshape(-1, 1), amp_c.reshape(-1, 1), mu_c.reshape(-1, 1)
		fwhm_A = (fwhm_c / C_KMS) * mu_c
		flux_c = calc_flux(amp_c, fwhm_A)
		L_line = calc_luminosity(np.array(distances)[:,None], flux_c)
		cont_c = unumpy.uarray(*np.array(evaluate_with_error(full_cont_profile,unumpy.nominal_values(mu_c), full_cont_params,unumpy.std_devs(mu_c), ucont_params)))
		eqw_c = flux_c / cont_c
	else:
		from sheap.SheaProducts.Utils.fwhm_conv import combine_fast	
		
		N = amp_b.shape[0]
		params_broad = jnp.stack([amp_b, mu_b, fwhm_kms_b], axis=-1).reshape(N, -1)
		params_narrow = jnp.concatenate([amp_n, mu_n, fwhm_kms_n], axis=1)

		fwhm_c, amp_c, mu_c = combine_fast(params_broad, params_narrow, limit_velocity=limit_velocity, C_KMS=C_KMS)
		if fwhm_c.ndim==1:
			fwhm_c, amp_c, mu_c = fwhm_c.reshape(-1, 1), amp_c.reshape(-1, 1), mu_c.reshape(-1, 1) #i have a helper to do this?

		fwhm_A = (fwhm_c / C_KMS) * mu_c
		flux_c = calc_flux(jnp.array(amp_c), jnp.array(fwhm_A))
		L_line = calc_luminosity(jnp.array(distances), flux_c)
		if full_cont_profile:
			cont_c = full_cont_profile(mu_c,full_cont_params)
			eqw_c = flux_c / cont_c
		else:
			eqw_c = jnp.zeros_like(flux_c)
	#L_line = 0 
	combined = {"component": components,"flux":flux_c,"fwhm":fwhm_A,"fwhm_kms":fwhm_c,"center":mu_c,"amplitude":amp_c,"luminosity":L_line,"eqw":eqw_c}
	
	return combined
	
#cont_c = cont_group(mu_c, cont_params)
