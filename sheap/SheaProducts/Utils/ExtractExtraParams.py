
"""
Extract Extra Params
============================

?
"""

__author__ = 'felavila'

__all__ = []

import numpy as np 

from sheap.Utils.BasicFunctions import log10

#TODO implemented Rfe 

def _col(x):
    """
    Ensure input is a 2D column vector repetead.
    TODO helper? .
    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    array-like
        If input is 1D, reshaped to (N, 1).
    """
#
    x = np.asarray(x)
    return x[None,:] if x.ndim == 1 else x


def calc_black_hole_mass(L_in, vwidth_kms, estimator, extras=None):
    r"""
    Unified single-epoch (SE) black-hole mass estimator.

    This function keeps the classical behavior of the SE mass formula while providing
    clear documentation for continuum-based and line-based calibrations. It also supports
    optional shape terms and Fe II strength corrections.

    Parameters
    ----------
    L_in : array-like or float
        Luminosity used by the calibration:
        - For ``kind="continuum"``: monochromatic luminosity :math:`L_\lambda \cdot \lambda`
        (erg s\ :sup:`-1`).
        - For ``kind="line"``: line luminosity :math:`L_\text{line}` (erg s\ :sup:`-1`).
    vwidth_kms : array-like or float
        Velocity width in km/s. Defaults to FWHM, but you can set
        ``width_def="sigma"`` in ``estimator`` to use :math:`\sigma`.
    estimator : dict
        Calibration dictionary. Required keys:

        - ``kind``: "continuum" or "line"
        - ``a``: intercept term (dimensionless)
        - ``b``: luminosity slope
        - ``f``: virial factor (applied multiplicatively to the mass)
        - ``fwhm_factor`` (alias ``vel_exp``): velocity-width exponent (default 2.0)
        - ``pivots``: dict with reference values (e.g., ``{"L": 1e44, "FWHM": 1e3}`` for continuum
        or ``{"L": 1e42, "FWHM": 1e3}`` for line)

        Optional:
        - ``width_def``: "fwhm" (default) or "sigma"
        - ``extras``: nested dict with optional switches:
            * ``le20_shape``: If True and ``width_def="fwhm"``, adds a shape term using
            :math:`\sigma`.
            * ``pan25_gamma``: Slope for Fe II strength correction (default :math:`-0.34`).

    extras : dict, optional
        Runtime extras for optional terms:
        - ``sigma_kms``: second velocity measure (km/s) for the Le20-like shape term.
        - ``R_Fe``: Fe II strength (e.g., :math:`R_\mathrm{FeII}`).

    Returns
    -------
    numpy-like
        :math:`M_\mathrm{BH}` in solar masses (:math:`M_\odot`), with the virial factor ``f``
        already applied.

    Notes
    -----
    Base (log) mass relation, valid for both continuum- and line-based inputs:

    .. math::
    \log_{10} M_\mathrm{BH} = 
    \log_{10} f + a + b \left[ \log_{10} L - \log_{10} L_0 \right]
    + \beta \left[ \log_{10} V - \log_{10} V_0 \right] \;,

    where:

    - :math:`L` is :math:`L_\lambda \cdot \lambda` (continuum) or :math:`L_\text{line}` (line),
    - :math:`V` is the velocity width (FWHM or :math:`\sigma`) in km/s,
    - :math:`L_0` and :math:`V_0` are the pivot luminosity and velocity from ``pivots``,
    - :math:`\beta` is ``fwhm_factor`` (or ``vel_exp``), by default 2.0,
    - :math:`f` is the virial factor.

    If ``width_def="fwhm"`` and ``extras["le20_shape"]`` is True (Leighly+20-like term),
    and a second velocity measure :math:`\sigma` is provided via ``extras["sigma_kms"]``:

    .. math::
    \Delta \log_{10} M_\mathrm{BH} =
    -1.14 \left[ \log_{10}(\mathrm{FWHM}) - \log_{10}(\sigma) \right] + 0.33 \;.

    If ``extras["R_Fe"]`` is provided, a Panessa+25-like correction is added:

    .. math::
    \Delta \log_{10} M_\mathrm{BH} = \gamma \, R_\mathrm{Fe} \;,

    with :math:`\gamma =` ``estimator["extras"]["pan25_gamma"]`` (default :math:`-0.34`).

    Examples
    --------
    Classical continuum-based recipe with FWHM:

    >>> est = {
    ...     "kind": "continuum",
    ...     "a": 0.0, "b": 0.5, "f": 1.0,
    ...     "fwhm_factor": 2.0,
    ...     "pivots": {"L": 1e44, "FWHM": 1e3},
    ...     "width_def": "fwhm",
    ... }
    >>> MBH = calc_black_hole_mass(L_5100, FWHM_kms, est)

    Same but with the Le20 shape term:

    >>> extras = {"sigma_kms": sigma_kms}
    >>> est["extras"] = {"le20_shape": True}
    >>> MBH = calc_black_hole_mass(L_5100, FWHM_kms, est, extras=extras)

    Line-based calibration:

    >>> est_line = {
    ...     "kind": "line",
    ...     "a": 6.57, "b": 0.47, "f": 1.0,
    ...     "fwhm_factor": 2.06,
    ...     "pivots": {"L": 1e42, "FWHM": 1e3},
    ...     "width_def": "fwhm",
    ... }
    >>> MBH = calc_black_hole_mass(L_Halpha, FWHM_kms, est_line)
    """

    if extras is None:
        extras = {}

    kind = str(estimator.get("kind", "continuum")).lower()
    width_def = str(estimator.get("width_def", "fwhm")).lower()

    piv = estimator.get("pivots", {})
    L0 = float(piv.get("L", 1e42 if kind == "line" else 1e44))
    V0 = float(piv.get("FWHM", 1e3))
    #print(V0,type(V0))
    a = estimator["a"]
    b = estimator["b"]
    beta = estimator.get("fwhm_factor", estimator.get("vel_exp", 2.0))
    f = estimator.get("f", 1.0)
    
    L = _col(L_in)
    V = _col(vwidth_kms)
    #print(type(f),type(L),type(L0),type(beta),type(V),type(V0))
    #logM = log10(f) + a + b * (log10(L) - log10(L0)) + beta * (log10(V) - log10(V0))
    logM = log10(f) + a  + b    * (log10(L) - log10(L0)) + beta * (log10(V) - log10(V0))
    # Le20 shape (only if baseline uses FWHM)
    if width_def == "fwhm" and estimator.get("extras", {}).get("le20_shape", False):
        sigma = extras.get("sigma_kms", None)
        if sigma is not None:
            sigma = _col(sigma)
            logM += (-1.14 * (log10(V) - log10(sigma)) + 0.33)

    # Pan25 iron term
    if "R_Fe" in extras:
        gamma = estimator.get("extras", {}).get("pan25_gamma", -0.21)#-0.34)
        RFe = _col(extras["R_Fe"])
        
        #logM += gamma * RFe  # broadcasts across components
    return (10.0 ** logM)



def extra_params_functions(params, L_w, L_bol, estimators, C_KMS,R_Fe=None,eta = 0.1):
    r"""
    Compute derived parameters (BH masses, Eddington ratios, accretion rates).

    This routine applies single-epoch (SE) virial estimators to broad-line
    measurements, combining continuum or line luminosities with velocity widths
    to derive black hole masses and accretion-related quantities.

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

    #if extras is None:
    broad_params = params.get("broad",None)
    combined = params.get("combined",False)
    if not broad_params and combined:
        broad_params = params #jeje
    elif not broad_params and not combined:
        print("No broad component")
        return {}
    out = {}
    fwhm_all = np.atleast_2d(_col(broad_params.get("fwhm_kms")))
    lum_all  = np.atleast_2d(_col(broad_params.get("luminosity")))
    sigma_all = broad_params.get("sigma_kms", None)
    idx_by_name = broad_params.get("idx_by_name")
    #flux_all = broad_params.get("flux", None)
    if sigma_all is not None:
        sigma_all = np.atleast_2d(_col(sigma_all))

    lines = np.asarray(broad_params.get("lines", []))
    comps = np.asarray(broad_params.get("component", []))
    c_cm = C_KMS * 1e5
    M_sun_g = 1.98847e33
    sec_yr = 3.15576e7

    for calib_key, est in estimators.items():
        #print(calib_key)
        line_name = est.get("line")
        kind = est.get("kind", "continuum")
        width_def = str(est.get("width_def", "fwhm")).lower()
        idxs = idx_by_name.get(line_name,[])
        if len(idxs)==0:
            continue
        if "Pan25" in calib_key or "Le20" in calib_key:
            #print(f"TODO implement {calib_key}")
            continue 
        comp_here = comps[idxs]
        if width_def == "sigma":
            if sigma_all is not None:
                Vwidth = sigma_all[:, idxs]
            # elif "sigma_kms" in extras:
            #     sig = _col(extras["sigma_kms"])
            #     Vwidth = sig[:, idxs] if sig.ndim == 2 else sig
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

            Lmono = _col(L_w[wkey])
            MBH = calc_black_hole_mass(Lmono, Vwidth, est, extras=None)

            # Ledd + mdot (only for continuum, and only if L_bol available)
            Ledd = 1.26e38 * MBH
            mdot_yr = None
            Lbol = None
            if wkey in L_bol:
                Lbol = _col(L_bol[wkey])
                mdot_gs = Lbol / (eta * c_cm**2)
                mdot_yr = mdot_gs / M_sun_g * sec_yr

            out.setdefault(line_name, {})[calib_key] = {
                "method": "continuum",
                "wavelength": lam,
                "vwidth_def": width_def,
                "vwidth_kms": Vwidth,
                "log10_smbh": log10(MBH),
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
                "log10_smbh": log10(MBH),
                "Ledd": Ledd,
                "component": comp_here,
            "combined":combined}

    return out