
"""
Extract Extra Params
============================

?
"""

__author__ = 'felavila'

__all__ = []

import numpy as np 
from copy import deepcopy

from sheap.Utils.BasicFunctions import calc_black_hole_mass,calc_monochromatic_luminosity,calc_bolometric_luminosity,calc_luminosity
from sheap.Utils.Constants import DEFAULT_BOL_CORRECTIONS, DEFAULT_SINGLE_EPOCH_ESTIMATORS,DEFAULT_C_KMS




def extra_params_functions(params, L_w, L_bol, estimators, C_KMS,R_Fe=None,eta = 0.1,M_sun_g = 1.98847e33,sec_yr = 3.15576e7):
    r"""
    Compute derived parameters (BH masses, Eddington ratios, accretion rates).

    This routine applies single-epoch (SE) virial estimators to broad-line
    measurements, combining continuum or line luminosities with velocity widths
    to derive black hole masses and accretion-related quantities.
    TODO add R_Fe rutine  
    Parameters
    ----------
    params : dict
        Dictionary of broad-line properties (e.g., ``fwhm_kms``, ``luminosity``).
    L_w : dict
        Monochromatic luminosities keyed by wavelength.
    L_bol : dict
        Bolometric luminosities keyed by wavelength.
    estimators : dict
        Single-epoch estimators for both continuum and line calibrations.
    C_KMS : float
        Speed of light in km/s.
    extras : dict, optional
        Extra quantities for corrections (e.g., ``sigma_kms``, ``R_Fe``).

    Returns
    -------
    dict
        Nested dictionary of derived parameters per line and calibration.

    Notes
    -----
    The general single-epoch black hole mass relation is:

    .. math::
    \log M_\mathrm{BH} =
    a
    + b \cdot (\log L - \log L_0)
    + \beta \cdot (\log V - \log V_0)
    + \log f \;,

    where:

    - :math:`L` is either a monochromatic continuum luminosity or a line luminosity
    - :math:`V` is the velocity width (FWHM or :math:`\sigma`)
    - :math:`(a, b, \beta, f)` are the calibration parameters
    - :math:`L_0, V_0` are the pivot values from the calibration

    Special cases
    -------------

    **Continuum-based estimators**  
    Use monochromatic luminosities :math:`L_\lambda` at a given wavelength
    with a bolometric correction:

    .. math::
    L_\mathrm{bol} = BC_\lambda \cdot (\lambda L_\lambda)

    From this, the Eddington ratio and accretion rate are derived:

    .. math::
    L_\mathrm{Edd} = 1.26 \times 10^{38} \; 
    \left( \frac{M_\mathrm{BH}}{M_\odot} \right)
    \; [\mathrm{erg\,s^{-1}}]

    .. math::
    \dot{M} = \frac{L_\mathrm{bol}}{\eta \, c^2}

    with :math:`\eta = 0.1` by default.

    **Line-based estimators**  
    Use the integrated line luminosity:

    .. math::
    \log M_\mathrm{BH} =
    a + b \cdot (\log L_\mathrm{line} - \log L_0)
    + \beta \cdot (\log V - \log V_0)

    Corrections supported
    ---------------------

    - **Le20 shape term**: additional dependence on the FWHM-to-σ ratio.  
    - **Pan25 iron term**: additional correction proportional to :math:`R_\mathrm{Fe}`.
    """

    #Nsamples,lineas
    broad_params = params.get("broad",None)
    if not broad_params:
        return {}
    out = {}
    Combined = np.atleast_1d(broad_params.get("combined"))
    fwhm_all = np.atleast_2d(broad_params.get("fwhm_kms"))
    lum_all  = np.atleast_2d(broad_params.get("luminosity"))
    sigma_all = broad_params.get("sigma_kms", None)
    idx_by_name = broad_params.get("idx_by_name")
    if sigma_all is not None:
        sigma_all = np.atleast_2d(sigma_all)

    comps = np.asarray(broad_params.get("component", []))
    
    c_cm = C_KMS * 1e5

    for calib_key, est in estimators.items():
        line_name = est.get("line")
        kind = est.get("kind", "continuum")
        width_def = str(est.get("width_def", "fwhm")).lower()
        idxs = idx_by_name.get(line_name,[])
        if "Pan25" in calib_key or "Le20" in calib_key or len(idxs)==0:
            #print(f"TODO implement {calib_key}")
            continue
        combined = Combined[idxs]
        comp_here = comps[idxs]
        if width_def == "sigma":
            if sigma_all is not None:
                Vwidth = sigma_all[:, idxs]
            else:
                print("no sigma available")
                continue  # 
        else:
            Vwidth = fwhm_all[:, idxs]
        L_line = lum_all[:, idxs]
        
        if kind == "continuum":
            lam = est.get("wavelength", None)
            if lam is None:
                continue
            wkey = str(int(lam))
            if wkey not in L_w:
                continue
            lw = L_w[wkey]
            if np.shape(lw) == ():
                lw = np.array([lw])
            #print(lw)
            Lmono = np.repeat(lw[:, None], Vwidth.shape[1], axis=1) #np.tile(L_w[wkey],)
            #print(line_name,Lmono.shape,Vwidth.shape,combined)
            MBH = calc_black_hole_mass(Lmono, Vwidth, est, extras=None)

            # Ledd + mdot (only for continuum, and only if L_bol available)
            Ledd = 1.26e38 * MBH
            mdot_yr = None
            Lbol = None
            if wkey in L_bol:
                Lbol = np.atleast_2d(L_bol[wkey]).T
                #print("Lbol",Lbol.shape)
                mdot_gs = Lbol / (eta * c_cm**2)
                mdot_yr = mdot_gs / M_sun_g * sec_yr

            out.setdefault(line_name, {})[calib_key] = {
                "method": "continuum",
                #"wavelength": lam,
                "vwidth_def": width_def,
                "vwidth_kms": Vwidth,
                "log10_smbh": np.log10(MBH),
                "Lwave": Lmono,
                "Lbol": Lbol,
                "Ledd": Ledd,
                "mdot_msun_per_year": mdot_yr,
                "component": comp_here,
            "combined":combined}

        elif kind == "line":

            #L_line = lum_all[:, idxs]
            MBH = calc_black_hole_mass(L_line, Vwidth, est, extras=None)
            Ledd = 1.26e38 * MBH

            out.setdefault(line_name, {})[calib_key] = {
                "method": "line",
                "vwidth_def": width_def,
                "vwidth_kms": Vwidth,
                "Lline": L_line,
                "log10_smbh": np.log10(MBH),
                "Ledd": Ledd,
                "component": comp_here,
                "combined":combined}

    return out


def process_luminosity(d, dic_posterior_params,BOL_CORRECTIONS=DEFAULT_BOL_CORRECTIONS):
    dic_posterior_params = deepcopy(dic_posterior_params)
    dic_posterior_params["Lw"] = {}
    dic_posterior_params["Lbol"] = {}
    for key in list(dic_posterior_params.keys()):
        if "Fcont" in key:
            for wave, Fcont in dic_posterior_params[key].items():
                L_w = calc_monochromatic_luminosity(d, Fcont, int(wave))
                dic_posterior_params["Lw"][wave] = L_w
                dic_posterior_params["Lbol"][wave] = calc_bolometric_luminosity(L_w, BOL_CORRECTIONS[wave])
        
        if key in ['basic_params_combined_kinematic', 'basic_params_combined_classical', 'basic_params']:
            for region, values in dic_posterior_params[key].items():
                flux = values["flux"]
                dic_posterior_params[key][region]["luminosity"] = calc_luminosity(np.asarray(d), flux)
        else:
            continue 
    return dic_posterior_params

def process_smbh(dic_posterior_params, SINGLE_EPOCH_ESTIMATORS=DEFAULT_SINGLE_EPOCH_ESTIMATORS,C_KMS=DEFAULT_C_KMS,):
    dic_posterior_params = deepcopy(dic_posterior_params)

    L_w = dic_posterior_params["Lw"]
    L_bol = dic_posterior_params["Lbol"]

    for key, local_result in list(dic_posterior_params.items()):
        if "basic_params" in key:
            new_key = key.replace("basic_params", "extra_params")
            dic_posterior_params[new_key] = extra_params_functions(local_result, L_w, L_bol, SINGLE_EPOCH_ESTIMATORS,C_KMS,)

    return dic_posterior_params



