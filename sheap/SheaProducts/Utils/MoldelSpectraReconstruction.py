"""
?
This requiere alot of cleaning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

#import pandas as pd
#from collections.abc import Mapping
import jax.numpy as jnp
from jax import jit, vmap


from sheap.Profiles.Utils import make_fused_profiles


__author__ = 'felavila'

__all__ = ["RegionInfo","MoldelSpectraReconstruction"]

@dataclass
class RegionInfo:
	"""Lightweight region registry entry."""
	combined_profile: Any
	idx_global: jnp.ndarray

#TODO this one and Sheaproducts have to be in the same place.
class MoldelSpectraReconstruction:
	"""
	Evaluate fused model + per-region components for best-fit params and posterior samples/draws.

	On init, the class tries to automatically load posterior arrays and keep them cached:
	  - self.samples : (N_obj, N_samp, N_global) or None
	  - self.draws   : (N_obj, N_draw, N_global) or None

	Defaults
	--------
	posterior_group = "montecarlo"
	posterior_key   = "posterior_result"
	samples_field   = "samples_phys"
	draws_field     = "draws_phys"
	"""

	def __init__(
		self,
		sheapspectral_template: Any,
		*,
		jit_compile: bool = True,
		posterior_group: str = "montecarlo",
		posterior_key: str = "posterior_result",
		samples_field: str = "samples_phys",
		draws_field: str = "draws_phys",
		autoload_posterior: bool = True,
	):
		self.obj = sheapspectral_template
		self._jit_compile = bool(jit_compile)

		# posterior config
		self._posterior_group = posterior_group
		self._posterior_key = posterior_key
		self._samples_field = samples_field
		self._draws_field = draws_field

		# cached posterior
		self.samples: Optional[jnp.ndarray] = None
		self.draws: Optional[jnp.ndarray] = None

		# registry + models
		self._registry: Dict[str, RegionInfo] = {}
		self._model = None
		self._batched_model = None
		self._batched_region: Dict[str, Any] = {}

		self._build_registry_and_model()
		self._build_batched_evalers()

		if autoload_posterior:
			self._try_autoload_posterior()

	# ----------------------------- setup ---------------------------------

	def _build_registry_and_model(self) -> None:
		res = self.obj.result
		grouped = res.sheapmodel.group_by("region")

		self._registry.clear()
		for region_name in grouped.keys():
			self._registry[region_name] = RegionInfo(
				combined_profile=grouped[region_name].combined_profile,
				idx_global=jnp.array(grouped[region_name].flat_param_indices_global),
			)

		fused = make_fused_profiles(res.profile_functions)
		self._model = jit(fused) if self._jit_compile else fused

	def _build_batched_evalers(self) -> None:
		self._batched_model = vmap(vmap(self._model, in_axes=(None, 0)), in_axes=(None, 0))

		self._batched_region.clear()
		for region_name, info in self._registry.items():
			f = info.combined_profile
			self._batched_region[region_name] = vmap(vmap(f, in_axes=(None, 0)), in_axes=(None, 0))

	# ----------------------------- posterior --------------------------------

	def _posterior_dict(self) -> Dict[str, Any]:
		return self.obj.result.posterior[self._posterior_group][self._posterior_key]

	def _collect_field(self, field: str) -> jnp.ndarray:
		d = self._posterior_dict()
		all_list = []
		for _, spectral_values in d.items():
			if field not in spectral_values:
				raise KeyError(f"Field '{field}' not present in posterior_result entry keys: {list(spectral_values.keys())}")
			all_list.append(spectral_values[field])
		return jnp.array(all_list)

	def _try_autoload_posterior(self) -> None:
		# samples
		try:
			self.samples = self._collect_field(self._samples_field)
		except Exception:
			self.samples = None

		# draws (optional)
		try:
			self.draws = self._collect_field(self._draws_field)
		except Exception:
			self.draws = None

	def reload_posterior(self) -> None:
		"""Force re-read of samples/draws from result.posterior using current config."""
		self._try_autoload_posterior()

	# ----------------------------- helpers --------------------------------

	@property
	def region_names(self) -> Tuple[str, ...]:
		return tuple(self._registry.keys())

	def _wl_array(self, wavelength: float) -> jnp.ndarray:
		return jnp.array([float(wavelength)], dtype=jnp.float32)

	def get_region_info(self, region_name: str) -> RegionInfo:
		if region_name not in self._registry:
			raise KeyError(f"Region '{region_name}' not found. Available: {self.region_names}")
		return self._registry[region_name]

	def _require_samples(self, all_samples: Optional[jnp.ndarray]) -> jnp.ndarray:
		if all_samples is not None:
			return all_samples
		if self.samples is None:
			raise ValueError(
				"No samples provided and self.samples is None. "
				"Either pass all_samples=... or set autoload_posterior=True and ensure posterior exists."
			)
		return self.samples

	def _require_draws(self, all_draws: Optional[jnp.ndarray]) -> jnp.ndarray:
		if all_draws is not None:
			return all_draws
		if self.draws is None:
			raise ValueError(
				"No draws provided and self.draws is None. "
				"Either pass all_draws=... or ensure draws exist in posterior."
			)
		return self.draws

	# ----------------------------- best-fit evaluation ---------------------

	def eval_bestfit_model(self, wavelength: float) -> jnp.ndarray:
		wl = self._wl_array(wavelength)
		params = self.obj.result.params
		return vmap(self._model, in_axes=(None, 0))(wl, params)

	def eval_bestfit_region(self, region_name: str, wavelength: float) -> jnp.ndarray:
		wl = self._wl_array(wavelength)
		info = self.get_region_info(region_name)
		params = self.obj.result.params
		p_reg = params[:, info.idx_global]
		f = info.combined_profile
		return vmap(f, in_axes=(None, 0))(wl, p_reg)

	# ----------------------------- batched evaluation ----------------------

	def eval_batched_model(self, wavelength: float, all_samples: Optional[jnp.ndarray] = None) -> jnp.ndarray:
		wl = self._wl_array(wavelength)
		S = self._require_samples(all_samples)
		return self._batched_model(wl, S)

	def eval_batched_region(
		self,
		region_name: str,
		wavelength: float,
		all_samples: Optional[jnp.ndarray] = None,
	) -> jnp.ndarray:
		wl = self._wl_array(wavelength)
		info = self.get_region_info(region_name)
		S = self._require_samples(all_samples)

		reg_params = S[:, :, info.idx_global]
		f_batched = self._batched_region[region_name]
		return f_batched(wl, reg_params)

	def eval_batched_components(
		self,
		wavelength: float,
		all_samples: Optional[jnp.ndarray] = None,
		*,
		regions: Optional[Iterable[str]] = None,
		include_model: bool = True,
	) -> Dict[str, jnp.ndarray]:
		if regions is None:
			regions = self.region_names

		out: Dict[str, jnp.ndarray] = {}

		if include_model:
			out["model"] = self.eval_batched_model(wavelength, all_samples)

		for r in regions:
			out[r] = self.eval_batched_region(r, wavelength, all_samples)

		return out

	# ----------------------------- derived quantities ----------------------

	def stars_cont_ratio(
		self,
		wavelength: float,
		all_samples: Optional[jnp.ndarray] = None,
		*,
		host_region: str = "host",
		subtract_regions: Tuple[str, ...] = ("narrow", "balmer", "fe", "broad"),
		squeeze: bool = True,
	) -> jnp.ndarray:
		comps = self.eval_batched_components(
			wavelength,
			all_samples,
			regions=(host_region,) + subtract_regions,
			include_model=True,
		)
		host = comps[host_region]
		denom = comps["model"]
		for r in subtract_regions:
			denom = denom - comps[r]

		ratio = host / denom
		return jnp.squeeze(ratio) if squeeze else ratio

	@property
	def stars_Cont_5100(self):
		"""
		Uses cached samples by default.

		Example
		-------
		ra = ResultAnalysis(sheap)
		stars = ra.stars_Cont_5100   # (N_obj, N_samp) if wl dim=1
		"""
		return self.stars_cont_ratio(5100.0, all_samples=None)
	
	def stars_cont_ratio_bestfit(
		self,
		wavelength: float,
		*,
		host_region: str = "host",
		subtract_regions: Tuple[str, ...] = ("narrow", "balmer", "fe", "broad"),
		squeeze: bool = True,
	) -> jnp.ndarray:
		"""
		Best-fit version of:
		  host / (model - narrow - balmer - fe - broad)

		Uses self.obj.result.params (best-fit global params), NOT posterior samples.
		"""
		# evaluate fused model at best-fit
		model = self.eval_bestfit_model(wavelength)  # (N_obj, 1, ...)

		# evaluate host + subtract regions at best-fit
		host = self.eval_bestfit_region(host_region, wavelength)
		denom = model
		for r in subtract_regions:
			denom = denom - self.eval_bestfit_region(r, wavelength)

		ratio = host / denom
		return jnp.squeeze(ratio) if squeeze else ratio

	@property
	def stars_Cont_5100_bestfit(self) -> jnp.ndarray:
		"""Convenience: best-fit stars/cont ratio at 5100 Å."""
		return self.stars_cont_ratio_bestfit(5100.0)

	# ----------------------------- single-object reproduce -----------------
	def reproduce_one_object(
		self,
		n_obj: int,
		*,
		x: Optional[jnp.ndarray] = None,
		samples: Optional[jnp.ndarray] = None,      # (N_samp, N_global)
		draws: Optional[jnp.ndarray] = None,        # (N_draw, N_global)
	) -> Dict[str, Any]:
		"""
		Reconstruct (evaluate) the model decomposition for a single object on its wavelength grid.

		This returns:
		- per-region component fluxes (for posterior samples, best-fit, and optionally draws)
		- per-region parameter vectors (same)
		- full fused-model flux (same)
		- a "region_sum_*" flux built by summing all region components (same)

		If `samples`/`draws` are not provided, it tries to use cached `self.samples`/`self.draws`.

		Parameters
		----------
		n_obj : int
			Object index.
		x : jnp.ndarray, optional
			Wavelength grid to evaluate on. If None, uses `self.obj.spectra[n_obj, 0, :]`.
		samples : jnp.ndarray, optional
			Posterior samples in *global* parameter space for this object, shape (N_samp, N_global).
		draws : jnp.ndarray, optional
			Posterior draws in *global* parameter space for this object, shape (N_draw, N_global).

		Returns
		-------
		dict
			Keys are named to be explicit about:
			- what is being evaluated (flux vs params)
			- which parameter set (samples vs bestfit vs draws)
			- whether it's per-region or whole-model

			Structure:
			{
				"flux_by_region_samples": {region: (N_samp, N_wave, ...), ...},
				"flux_by_region_bestfit": {region: (N_wave, ...), ...},
				"flux_by_region_draws":   {region: (N_draw, N_wave, ...), ...} or None,

				"params_by_region_samples": {region: (N_samp, N_reg), ...},
				"params_by_region_bestfit": {region: (N_reg,), ...},
				"params_by_region_draws":   {region: (N_draw, N_reg), ...} or None,

				"flux_full_model_samples": (N_samp, N_wave, ...),
				"flux_full_model_bestfit": (N_wave, ...),
				"flux_full_model_draws":   (N_draw, N_wave, ...) or None,

				"flux_region_sum_samples": (N_samp, N_wave, ...),
				"flux_region_sum_bestfit": (N_wave, ...),
				"flux_region_sum_draws":   (N_draw, N_wave, ...) or None,
			}
		"""
		if x is None:
			x = self.obj.spectra[n_obj, 0, :]

		# choose samples
		if samples is None:
			if self.samples is None:
				raise ValueError("No samples given and self.samples is None.")
			samples = self.samples[n_obj]

		# choose draws (optional)
		if draws is None and self.draws is not None:
			draws = self.draws[n_obj]

		bestfit_global = self.obj.result.params[n_obj]  # (N_global,)

		out: Dict[str, Any] = {
			# per-region fluxes
			"flux_by_region_samples": {},
			"flux_by_region_bestfit": {},
			"flux_by_region_draws": {} if draws is not None else None,

			# per-region params
			"params_by_region_samples": {},
			"params_by_region_bestfit": {},
			"params_by_region_draws": {} if draws is not None else None,

			# whole-model fluxes
			"flux_full_model_samples": vmap(self._model, in_axes=(None, 0))(x, samples),
			"flux_full_model_bestfit": self._model(x, bestfit_global),
			"flux_full_model_draws": (
				vmap(self._model, in_axes=(None, 0))(x, draws) if draws is not None else None
			),
		}

		# We'll accumulate region sums as we go
		region_sum_samples = None
		region_sum_bestfit = None
		region_sum_draws = None if draws is not None else None

		for region_name, info in self._registry.items():
			f = info.combined_profile
			idx = info.idx_global

			# slice global -> region params
			p_samp = samples[:, idx]
			p_best = bestfit_global[idx]

			out["params_by_region_samples"][region_name] = p_samp
			out["params_by_region_bestfit"][region_name] = p_best

			# evaluate region fluxes
			flux_samp = vmap(f, in_axes=(None, 0))(x, p_samp)
			flux_best = f(x, p_best)

			out["flux_by_region_samples"][region_name] = flux_samp
			out["flux_by_region_bestfit"][region_name] = flux_best

			# accumulate sums
			region_sum_samples = flux_samp if region_sum_samples is None else (region_sum_samples + flux_samp)
			region_sum_bestfit = flux_best if region_sum_bestfit is None else (region_sum_bestfit + flux_best)

			if draws is not None:
				p_draw = draws[:, idx]
				out["params_by_region_draws"][region_name] = p_draw

				flux_draw = vmap(f, in_axes=(None, 0))(x, p_draw)
				out["flux_by_region_draws"][region_name] = flux_draw

				region_sum_draws = flux_draw if region_sum_draws is None else (region_sum_draws + flux_draw)

		out["flux_region_sum_samples"] = region_sum_samples
		out["flux_region_sum_bestfit"] = region_sum_bestfit
		out["flux_region_sum_draws"] = region_sum_draws

		return out

	def fe_integrated_flux(
		self,
		*,
		x_min: float = 2250,
		x_max: float = 2650,
		n_grid: int = 2_000,
		region_name: str = "fe",
		all_samples: Optional[jnp.ndarray] = None,
		include_bestfit: bool = True,
		attach_to_posterior: bool = False,
		attach_component: str = "broad",
		attach_key: str = "R_Fe",
	) -> Dict[str, Any]:
		"""
		Integrate the Fe-region component flux over a wavelength window.

		This evaluates the region's `combined_profile` on a linear wavelength grid and
		integrates with a trapezoidal rule:

		    I_Fe = ∫ F_Fe(λ) dλ

		By default, this is computed for posterior samples of all objects:
		    samples -> shape (N_obj, N_samp)

		Optionally, it also computes the best-fit integral:
		    bestfit -> shape (N_obj,)

		Optionally, it can attach the per-object sample integrals into the posterior
		dict under:
		    posterior_result[obj_key]["basic_params"][attach_component]["extras"][attach_key]
        TODO add a coment on the values from Pan+25
		Parameters
		----------
		x_min, x_max : float
			Integration bounds in Å.
		n_grid : int
			Number of wavelength points for the integration grid.
		region_name : str
			Name of the region in the registry (default "fe").
		all_samples : jnp.ndarray, optional
			Posterior samples array (N_obj, N_samp, N_global). If None, uses cached
			`self.samples`.
		include_bestfit : bool
			Also compute best-fit integral from `self.obj.result.params`.
		attach_to_posterior : bool
			If True, store the sample integrals in the posterior dict.
		attach_component : str
			Which basic_params component to attach to (default "broad").
		attach_key : str
			Key name inside extras (default "R_Fe").

		Returns
		-------
		dict
			{
			  "wavelength_grid": (N_wave,),
			  "samples": (N_obj, N_samp),
			  "bestfit": (N_obj,) or None,
			}
		"""
		# grid (float32, consistent with your other helpers)
		x_grid = jnp.linspace(
			float(x_min), float(x_max), int(n_grid), dtype=jnp.float32
		)

		# region info + slice samples to region param space
		info = self.get_region_info(region_name)
		S = self._require_samples(all_samples)  # (N_obj, N_samp, N_global)
		reg_params = S[:, :, info.idx_global]   # (N_obj, N_samp, N_reg)

		# batched eval over (obj, samp) -> flux(λ)
		f = info.combined_profile
		f_batched = vmap(vmap(f, in_axes=(None, 0)), in_axes=(None, 0))
		flux = f_batched(x_grid, reg_params)  # (N_obj, N_samp, N_wave)

		# integrate over wavelength axis
		int_samples = jnp.trapezoid(flux, x_grid, axis=-1)  # (N_obj, N_samp)

		out: Dict[str, Any] = {
			"wavelength_grid": x_grid,
			"samples": int_samples,
			"bestfit": None,
		}

		if include_bestfit:
			p_best = self.obj.result.params[:, info.idx_global]  # (N_obj, N_reg)
			flux_best = vmap(f, in_axes=(None, 0))(x_grid, p_best)  # (N_obj, N_wave)
			out["bestfit"] = jnp.trapezoid(flux_best, x_grid, axis=-1)   # (N_obj,)

		if attach_to_posterior:
			# attach sample integrals per object into posterior dict
			d = self._posterior_dict()
			keys = list(d.keys())  # keep same iteration order as _collect_field
			for i, k in enumerate(keys):
				entry = d[k]
				basic = entry.setdefault("basic_params", {})
				comp = basic.setdefault(attach_component, {})
				extras = comp.setdefault("extras", {})
				extras[attach_key] = int_samples[i]  # (N_samp,)
			# note: if you want bestfit attached too, you can store out["bestfit"][i] similarly.

		return out

