from typing import Any, Dict, List, Union
import numpy as np
import jax.numpy as jnp
from jax import vmap,jit
import warnings
from functools import partial

from sheap.Functions.utils import make_integrator
from .functions import calc_flux,calc_fwhm_kms,calc_luminosity,calc_monochromatic_luminosity,calc_bolometric_luminosity,calc_black_hole_mass
from .constants import BOL_CORRECTIONS,SINGLE_EPOCH_ESTIMATORS,c
from .utils import combine_fast

from sheap.Functions.profiles import PROFILE_LINE_FUNC_MAP,PROFILE_FUNC_MAP


def summarize_samples(samples) -> Dict[str, np.ndarray]:
    """Compute 16/50/84 percentiles and return a summary dict using NumPy."""
    if isinstance(samples, jnp.ndarray):
        samples = np.asarray(samples)
    samples = np.atleast_2d(samples).T
    if np.isnan(samples).sum() / samples.size > 0.2:
        warnings.warn("High fraction of NaNs; uncertainty estimates may be biased.")
    if samples.shape[1]<=1:
        q = np.nanpercentile(samples, [16, 50, 84], axis=0)
    else:
        q = np.nanpercentile(samples, [16, 50, 84], axis=1)
    #else:
    
    return {
        "median": q[1],
        "err_minus": q[1] - q[0],
        "err_plus": q[2] - q[1]
    }


def summarize_nested_samples(d: dict) -> dict:
    """
    Recursively walk through a dictionary and apply summarize_samples_numpy
    to any array-like values.
    """
    summarized = {}
    for k, v in d.items():
        if isinstance(v, dict):
            summarized[k] = summarize_nested_samples(v)
        elif isinstance(v, (np.ndarray, jnp.ndarray)) and np.ndim(v) >= 1 and k!='component':
            summarized[k] = summarize_samples(v)
        else:
            summarized[k] = v
    return summarized


def compute_fwhm_split(profile: str,
                       amp:   jnp.ndarray,
                       center:jnp.ndarray,
                       extras:jnp.ndarray) -> jnp.ndarray:
    func = PROFILE_LINE_FUNC_MAP[profile]

    # build the named‐param dict on‐the‐fly:
    # we know extras corresponds to param_names[2:]
    names = func.param_names
    p = { names[0]: amp,
          names[1]: center }
    for i,name in enumerate(names[2:]):
        p[name] = extras[i]

    # analytic cases:
    if profile == "gaussian" or profile == "lorentzian":
        return p["fwhm"]
    if profile == "top_hat":
        return p["width"]
    if profile == "voigt_pseudo":
        fg = p["fwhm_g"]; fl = p["fwhm_l"]
        return 0.5346*fl + jnp.sqrt(0.2166*fl*fl + fg*fg)

    # numeric‐fallback (e.g. skewed, EMG)
    half = amp/2.0
    def shape_fn(x):
        return func(x, jnp.concatenate([jnp.array([amp,center]), extras]))
    guess = p.get("fwhm", p.get("width",
                jnp.maximum(p.get("fwhm_g",0), p.get("fwhm_l",0))))
    lo,hi = center-5*guess, center+5*guess
    xs = jnp.linspace(lo, hi, 2001)
    ys = shape_fn(xs)

    maskL = (xs<center)&(ys<=half)
    maskR = (xs> center)&(ys<=half)
    xL = jnp.max(jnp.where(maskL, xs, lo))
    xR = jnp.min(jnp.where(maskR, xs, hi))
    return xR - xL

def make_batch_fwhm_split(profile: str):
    # bind away `profile`; single(amp, center, extras) -> scalar fwhm
    single = partial(compute_fwhm_split, profile)

    # 1⃣ map over the line‐index (axis=0 of amp: (6,), center: (6,), extras: (6,1))
    over_lines = vmap(single, in_axes=(0, 0, 0))

    # 2⃣ map that over objects  (axis=0 of amps:  (2000,6), centers: (2000,6), extras: (2000,6,1))
    batcher    = vmap(over_lines, in_axes=(0, 0, 0))

    return batcher

def extract_basic_line_parameters(
    full_samples: np.ndarray,
    region_group: Any, #we already have a class for this 
    distances: np.ndarray,
    c: float,
    wavelength_grid: jnp.ndarray = jnp.linspace(0, 20_000, 20_000),
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract continuum‐corrected flux, FWHM, FWHM (km/s), center, amplitude,
    equivalent width and luminosity for each emission line, grouped by 'kind'.

    Returns a dict mapping each kind → dict with keys:
      'lines', 'component',
      'flux', 'fwhm', 'fwhm_kms',
      'center', 'amplitude',
      'eqw', 'luminosity'
    """
    # Precompute continuum params
    cont_group = region_group.group_by("kind")["continuum"]
    cont_idx   = cont_group.flat_param_indices_global
    cont_params= full_samples[:, cont_idx]

    basic_params: Dict[str, Dict[str, np.ndarray]] = {}

    for kind, kind_group in region_group.group_by("kind").items():
        if kind in ("fe", "continuum"):
            continue

        line_names, components = [], []
        flux_parts, fwhm_parts = [], []
        fwhm_kms_parts, center_parts = [], []
        amp_parts, eqw_parts, lum_parts = [], [], []

        for profile_name, prof_group in kind_group.group_by("profile_name").items():

            # Determine integrator and handle sub-profiles
            if "_" in profile_name:
                _, subprof = profile_name.split("_", 1)
                profile_fn = PROFILE_FUNC_MAP[subprof]
                batch_fwhm = make_batch_fwhm_split(subprof)  # jitted on first call
                integrator = make_integrator(profile_fn, method="vmap")

                for sp, param_idxs in zip(
                    prof_group.lines, prof_group.global_profile_params_index_list
                ):
                    params      = full_samples[:, param_idxs]
                    names       = np.array(prof_group._master_param_names)[param_idxs]
                    amp_pos     = np.where(["amplitude" in n for n in names])[0]
                    shift_idx   = amp_pos.max() + 1

                    # build per-line stacks
                    stacks = []
                    for _, factor, src_id in sp.amplitude_relations:
                        amp   = params[:, amp_pos[np.where(
                                     [src_id == rel[2] for rel in sp.amplitude_relations]
                                 )[0][0]]] * factor
                        cen   = sp.center + params[:, shift_idx]
                        extra = params[:, shift_idx+1:]
                        stacks.append(np.stack([amp, cen, *[extra]], axis=-1))

                    full_params = jnp.stack(stacks, axis=1)
                    flux        = integrator(wavelength_grid, full_params)
                    fwhm        =  jnp.atleast_3d(extra[:,:,-1])
                    
                    centers     = full_params[:, :, 1]
                    amps        = full_params[:, :, 0]
                    print(amps.shape,centers.shape,fwhm.shape)
                    fwhm = batch_fwhm(amps, centers, fwhm)         # → (1000,20)
                    fwhm_kms    = jnp.abs(calc_fwhm_kms(fwhm, c, centers))
                    cont_vals   = vmap(cont_group.combined_profile, in_axes=(0,0))(
                                      centers, cont_params
                                  )
                    lum_vals    = calc_luminosity(distances[:, None], flux, centers)
                    eqw        = flux / cont_vals

                    nsub = flux.shape[1]
                    line_names.extend(sp.region_lines)
                    components.extend([sp.component]*nsub)
                    flux_parts.append(np.array(flux))
                    fwhm_parts.append(np.array(fwhm))
                    fwhm_kms_parts.append(np.array(fwhm_kms))
                    center_parts.append(np.array(centers))
                    amp_parts.append(np.array(amps))
                    eqw_parts.append(np.array(eqw))
                    lum_parts.append(np.array(lum_vals))

            else:
                profile_fn = PROFILE_FUNC_MAP[profile_name]
                batch_fwhm = make_batch_fwhm_split(profile_name)  # jitted on first call
                integrator = make_integrator(profile_fn, method="vmap")
                idxs       = prof_group.flat_param_indices_global
                params     = full_samples[:, idxs]
                names      = list(prof_group.params_dict.keys())

                amp_idx = [i for i,n in enumerate(names) if "amplitude" in n]
                cen_idx = [i for i,n in enumerate(names) if "center" in n]
                other   = [i for i in range(params.shape[1]) 
                           if i not in amp_idx + cen_idx]

                reshaped = params.reshape(params.shape[0], -1, profile_fn.n_params)
                flux     = integrator(wavelength_grid, reshaped)
                fwhm     = jnp.atleast_3d(jnp.abs(params[:, other]))
                centers  = params[:, cen_idx]
                amps     = params[:, amp_idx]
                #print(amps.shape,centers.shape,fwhm.shape)
                fwhm = batch_fwhm(amps, centers, fwhm)         # → (1000,20)
                fwhm_kms = jnp.abs(calc_fwhm_kms(fwhm, c, centers))
                cont_vals= vmap(cont_group.combined_profile, in_axes=(0,0))(
                              centers, cont_params
                          )
                #print(distances.shape,flux.shape,centers.shape)
                lum_vals = calc_luminosity(distances[:, None], flux, centers)
                eqw      = flux / cont_vals

                line_names.extend([l.line_name for l in prof_group.lines])
                components.extend([l.component for l in prof_group.lines])
                flux_parts.append(np.array(flux))
                fwhm_parts.append(np.array(fwhm))
                fwhm_kms_parts.append(np.array(fwhm_kms))
                center_parts.append(np.array(centers))
                amp_parts.append(np.array(amps))
                eqw_parts.append(np.array(eqw))
                lum_parts.append(np.array(lum_vals))

        basic_params[kind] = {
            "lines":      line_names,
            "component":  components,
            "flux":       np.concatenate(flux_parts,     axis=1),
            "fwhm":       np.concatenate(fwhm_parts,     axis=1),
            "fwhm_kms":   np.concatenate(fwhm_kms_parts, axis=1),
            "center":     np.concatenate(center_parts,    axis=1),
            "amplitude":  np.concatenate(amp_parts,       axis=1),
            "eqw":        np.concatenate(eqw_parts,       axis=1),
            "luminosity": np.concatenate(lum_parts,       axis=1),
        }

    return basic_params


def posterior_physical_parameters(
    wl_i: np.ndarray,
    flux_i: np.ndarray,
    yerr_i: np.ndarray,
    mask_i: np.ndarray,
    full_samples: np.ndarray,
    region_group: Any,
    distances: np.ndarray,
    BOL_CORRECTIONS: Dict[str, float] = BOL_CORRECTIONS,
    SINGLE_EPOCH_ESTIMATORS: Dict[str, Dict[str, Any]] =SINGLE_EPOCH_ESTIMATORS ,
    c: float = c,
    summarize: bool = False,
    LINES_TO_COMBINE = ["halpha", "hbeta"],
    combine_components = True,
    limit_velocity = 150.0,
) -> Dict[str, Any]:
    """
    Master routine: from MCMC samples → basic line params, monochromatic & bolometric
    luminosities, single-epoch BH masses, Eddington L, and accretion rates.
    """
    
    basic_params = extract_basic_line_parameters(
        full_samples=full_samples,
        region_group=region_group,
        distances=distances,
        c=c,
    )
    cont_group = region_group.group_by("kind")["continuum"]
    cont_idx   = cont_group.flat_param_indices_global
    cont_params= full_samples[:, cont_idx]
    cont_fun   = cont_group.combined_profile
    
    if combine_components and 'broad' in basic_params and 'narrow' in basic_params:
        combined = {}
        for line in LINES_TO_COMBINE:
            # find all the broad‐component indices for this line
            broad_lines = basic_params["broad"]["lines"]
            idx_broad   = [i for i, L in enumerate(broad_lines) if L.lower() == line]
            # find the single narrow index (if any)
            narrow_lines = basic_params["narrow"]["lines"]
            idx_narrow   = [i for i, L in enumerate(narrow_lines) if L.lower() == line]

            # only combine if we actually have ≥2 broad and exactly one narrow
            if len(idx_broad) >= 2 and len(idx_narrow) == 1:
                N = full_samples.shape[0]

                # pull out amps & centers
                amps = basic_params["broad"]["amplitude"][:, idx_broad]   # (N, n_broad)
                mus  = basic_params["broad"]["center"][:, idx_broad]      # (N, n_broad)
                fwhms_kms = basic_params["broad"]["fwhm_kms"][:, idx_broad]  # (N, n_broad)

                # stack into (N, 3*n_broad)
                params_broad = jnp.stack([amps, mus, fwhms_kms], axis=-1).reshape(N, -1)

                # narrow triplet (N,3)
                amp_n     = basic_params["narrow"]["amplitude"][:, idx_narrow]
                mu_n      = basic_params["narrow"]["center"][:, idx_narrow]
                fwhm_nkms = basic_params["narrow"]["fwhm_kms"][:, idx_narrow]
                params_narrow = jnp.concatenate([amp_n, mu_n, fwhm_nkms], axis=1)

                fwhm_c, amp_c, mu_c = combine_fast(
                    params_broad, params_narrow,
                    limit_velocity=limit_velocity, c=c
                )

                fwhm_A = (fwhm_c / c) * mu_c 

                flux_c = calc_flux(np.array(amp_c), np.array(fwhm_A))

                fwhm_A = (fwhm_c / c) * mu_c       # shape (N,)

                flux_c = calc_flux(np.array(amp_c), np.array(fwhm_A))  # (N,)

                cont_c = vmap(cont_group.combined_profile)(mu_c, cont_params)  # (N,)

                L_line = calc_luminosity(distances, flux_c, mu_c)  # (N,)

                eqw_c = flux_c / cont_c

                combined[line] = {
                    "amplitude":  np.array(amp_c),    
                    "center":     np.array(mu_c),     
                    "fwhm_kms":   np.array(fwhm_c),   
                    "fwhm":     np.array(fwhm_A),   
                    "flux":       np.array(flux_c),   
                    "luminosity": np.array(L_line),   
                    "eqw":        np.array(eqw_c),    
                }
    L_w, L_bol = {}, {}
    

    for wave in map(float, BOL_CORRECTIONS.keys()):
        wstr = str(int(wave))
        if (jnp.isclose(wl_i, wave, atol=1) & ~mask_i).any():
            Fcont   = vmap(cont_fun, in_axes=(None, 0))(jnp.array([wave]), cont_params).squeeze()
            Lmono   = calc_monochromatic_luminosity(distances, Fcont, wave)
            Lbolval = calc_bolometric_luminosity(Lmono, BOL_CORRECTIONS[wstr])
            L_w[wstr], L_bol[wstr] = np.array(Lmono), np.array(Lbolval)

    # 3) single‐epoch mass estimates (broad lines)
    masses: Dict[str, Dict[str, np.ndarray]] = {}
    broad = basic_params.get("broad")
    if broad:
        fwhm_kms_all = broad["fwhm_kms"]
        line_list     = np.array(broad["lines"])
        for line_name, est in SINGLE_EPOCH_ESTIMATORS.items():
            lam = est["wavelength"]
            wstr = str(int(lam))
            if line_name in line_list and wstr in L_w:
                idxs      = np.where(line_list == line_name)[0]
                fkm       = fwhm_kms_all[:, idxs].squeeze()
                Lmono     = L_w[wstr]
                Lbolval   = L_bol[wstr]
                # broadcast dims if needed
                if fkm.ndim == 2:
                    Lmono   = Lmono[..., None]
                    Lbolval = Lbolval[..., None]

                mbh_samp = calc_black_hole_mass(Lmono, fkm, est)
                L_edd    = 1.26e38 * mbh_samp # erg/s this is assuming that the SMBH is in solar mass
                eta      = 0.1
                c_cm     = c * 1e5 # the code uses c in km/s we move it to cm
                M_sun_g  = 1.98847e33 # Solar mass in grams
                sec_yr   = 3.15576e7 # 
                mdot_gs  = Lbolval / (eta * c_cm**2) #  # g/s
                mdot_yr  = mdot_gs / M_sun_g * sec_yr

                masses[line_name] = {
                    "Lwave":             Lmono,
                    "Lbol":              Lbolval,
                    "fwhm_kms":          fkm,
                    "log10_smbh":        np.log10(mbh_samp),
                    "Ledd":              L_edd,
                    "mdot_msun_per_year":mdot_yr,
                }
                        # add extra parameters to combined

    result = {
        "basic_params": basic_params,
        "L_w":           L_w,
        "L_bol":         L_bol,
        "extras":        masses,
    }
    if len(combined.keys())>0:
        result["combined"] = combined
    if summarize:
        result = summarize_nested_samples(result)

    return result
