"""
Docstring for sheap.Utils.Paper

This requiere alot of cleaning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import os

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

import pandas as pd
from collections.abc import Mapping



def median_with_errors(x, low=0.16, high=0.84, ignore_nan=True, axis=None):
	x = np.asarray(x, dtype=float)
	if ignore_nan:
		x = x[~np.isnan(x)]
	if x.size == 0:
		return np.nan, np.nan, np.nan
	p_lo, p_med, p_hi = np.percentile(x, [100 * low, 50, 100 * high])
	return p_med, p_med - p_lo, p_hi - p_med


def posterior_extraction(
	sheapspectral,
	posterior_idx: int = 1,
	extra_key: str = "extra_combined_params",
	method="montecarlo",
	low=0.16,
	high=0.84,
) -> pd.DataFrame:

	rows = []
	posterior = sheapspectral.result.posterior["montecarlo"]["posterior_result"]

	# ⬇ enumerate to track object position
	for n_obj, (obj_name, values) in enumerate(posterior.items()):
		extra = values.get(extra_key, {})
		if not extra:
			continue

		for line, line_dict in extra.items():
			for combo, combo_dict in line_dict.items():
				meta = {}
				quantities = {}

				for key, val in combo_dict.items():
					if isinstance(val, Mapping) and "median" in val:
						quantities[key] = ("stats_dict", val)

					elif isinstance(val, (np.ndarray, list, tuple)):
						arr = np.asarray(val)
						if arr.size > 0 and np.issubdtype(arr.dtype, np.number):
							quantities[key] = ("samples", arr)
						else:
							meta[key] = val
					else:
						meta[key] = val

				for quantity_name, (qkind, payload) in quantities.items():
					row = {
						"n_obj": n_obj,          # ✅ object index
						"name": obj_name,        # object name
						"line": line,
						"SMBHEstimator": combo,
						"quantity": quantity_name,
					}

					# metadata
					for m_key, m_val in meta.items():
						if isinstance(m_val, (np.ndarray, list, tuple)):
							arr = np.asarray(m_val)
							row[m_key] = arr.item() if arr.size == 1 else m_val
						else:
							row[m_key] = m_val

					# statistics
					if qkind == "stats_dict":
						for stat_name, stat_val in payload.items():
							row[stat_name] = np.asarray(stat_val).squeeze()
					else:
						samples = payload
						med, em, ep = median_with_errors(samples, low=low, high=high)
						row["median"] = med
						row["err_minus"] = em
						row["err_plus"] = ep
						row["low_q"] = low
						row["high_q"] = high
						row["nsamp"] = int(np.size(samples))

					rows.append(row)

	df = pd.DataFrame(rows)

	non_numeric = {
		"n_obj", "name", "line", "SMBHEstimator", "quantity",
		"method", "vwidth_def", "component"
	}
	for col in df.columns:
		if col not in non_numeric:
			df[col] = pd.to_numeric(df[col], errors="ignore")

	return df

def mad_std(x):
    # Robust sigma estimate from MAD
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def median_abs_deviation(a):
    med = np.median(a)
    return np.median(np.abs(a - med))


def concordance_ccc(x, y):
    # Lin's concordance correlation coefficient
    x = np.asarray(x); y = np.asarray(y)
    mx, my = np.mean(x), np.mean(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return (2 * sxy) / (sx2 + sy2 + (mx - my) ** 2)
# ---------- helpers ----------
def mad_std(x):
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def concordance_ccc(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    sxy = np.cov(x, y, ddof=1)[0, 1]
    return (2 * sxy) / (vx + vy + (mx - my) ** 2)

def band_stats(x, y, band=0.3):
    m = np.isfinite(x) & np.isfinite(y)
    if not np.any(m):
        return dict(n=0, n_in=0, pct03=0.0, fr01=0.0, fr02=0.0,
                    bias=np.nan, sigmaR=np.nan, rmse=np.nan, ccc=np.nan)
    x, y = x[m], y[m]
    d = y - x
    n = d.size
    return dict(
        n=n,
        n_in=int((np.abs(d) <= band).sum()),
        pct03=100.0 * (np.abs(d) <= band).sum() / n,
        fr01=100.0 * (np.abs(d) <= 0.1).sum() / n,
        fr02=100.0 * (np.abs(d) <= 0.2).sum() / n,
        bias=float(np.mean(d)),
        sigmaR=float(mad_std(d)),
        rmse=float(np.sqrt(np.mean(d**2))),
        ccc=float(concordance_ccc(x, y)) if n > 2 else np.nan
    )

def summarize(name, S):
    print(f"{name}: N={S['n']}")
    print(f"  |Δ| ≤ 0.1 / 0.2 / 0.3 dex : {S['fr01']:.1f}% / {S['fr02']:.1f}% / {S['pct03']:.1f}%")
    print(f"  bias (mean Δ)             : {S['bias']:.3f} dex")
    print(f"  robust σ (MAD×1.4826)     : {S['sigmaR']:.3f} dex")
    print(f"  RMSE                      : {S['rmse']:.3f} dex")
    print(f"  CCC                       : {S['ccc']:.3f}")



def _finite_xy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def concordance_corrcoef(x, y):
    # Lin's CCC
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    return (2 * cov) / (vx + vy + (mx - my)**2)

def agreement_stats(x, y, ci=True, n_boot=5000, rng=None):
    """Bias, robust scatter, CCC, and fractions within common dex windows."""
    x, y = _finite_xy(x, y)
    d = y - x
    bias = d.mean()
    # robust sigma ~ 1-sigma if residuals ~ normal
    sigma_rob = 1.4826 * median_abs_deviation(d)

    sigma = d.std(ddof=1)
    # Lin's concordance (agreement with 1:1)
    ccc = concordance_corrcoef(x, y)
    # convenience fractions (dex windows commonly quoted)
    frac_01 = np.mean(np.abs(d) <= 0.1)
    frac_03 = np.mean(np.abs(d) <= 0.3)
    frac_05 = np.mean(np.abs(d) <= 0.5)
    # 95% limits of agreement (Bland–Altman)
    loa_lo = bias - 1.96 * sigma
    loa_hi = bias + 1.96 * sigma

    out = dict(bias=bias, sigma=sigma, sigma_rob=sigma_rob, ccc=ccc,
               frac_01=frac_01, frac_03=frac_03, frac_05=frac_05,
               loa=(loa_lo, loa_hi))

    if not ci:
        return out

    rng = np.random.default_rng(None if rng is None else rng)
    n = x.size
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        db = yb - xb
        bias_b = db.mean()
        sig_b = db.std(ddof=1)
        sigrob_b = 1.4826 * median_abs_deviation(db)
        ccc_b = concordance_corrcoef(xb, yb)
        boots.append((bias_b, sig_b, sigrob_b, ccc_b))
    boots = np.array(boots)
    q = lambda col: np.percentile(boots[:, col], [2.5, 50, 97.5])

    out["ci_bias"]      = q(0)
    out["ci_sigma"]     = q(1)
    out["ci_sigma_rob"] = q(2)
    out["ci_ccc"]       = q(3)
    return out

def _pretty_ykey(yk):
    """
    Allow tuple keys like ('SHEAP', 'Hα') but show just 'SHEAP' in the legend.
    Extend as you like.
    """
    if isinstance(yk, tuple):
        # ('SHEAP', 'Hα') -> "SHEAP"
        return yk[0]
    return yk


def plot_logdex_agreement_xd(
    x_dict,
    y_dict,
    xlabel=r'$\log_{10}(\mathrm{FWHM}_{\mathrm{ref}}\ [\mathrm{km\ s^{-1}}])$',
    ylabel=r'$\log_{10}(\mathrm{FWHM}_{\mathrm{SHEAP}}\ [\mathrm{km\ s^{-1}}])$',
    band=0.3,
    lims="auto",
    lims_pad=0.05,
    pair_mode="auto",              # "auto" | "zip" | "product"
    save_path=None,                # directory or full path; if dir, auto-filename
    dpi=300,
    save_format="pdf",             # preferred format: "pdf" | "png" | "jpg" | "jpeg"
    markers=('o', '*', 'X', 'D', '^', 'v', 'P', 's'),
    colors =  (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
),
    markersize=10,
    alpha=0.9,
    legend_fontsize=30,
    label_fontsize=30,
    tick_fontsize=30,
    what = "",
    label_mode = None,
    add_numbers = False
):
    """
    Plot y vs x in log10 space with a 1:1 line and a ±band (dex) region.

    Returns
    -------
    fig, ax, stats, saved_file
        stats[(x_key, y_key)] = dict(n_in, n_tot, pct, band, x_key, y_key, idx_out)
        saved_file is the path used for saving, or None if not saved.
    """
    import os
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from itertools import product, zip_longest
    
    if label_mode:
       xlabel, ylabel = {
                            "fwhm": (
                                r'$\log_{10}(\mathrm{FWHM}_{\mathrm{Literature}}\ [\mathrm{km\ s^{-1}}])$',
                                r'$\log_{10}(\mathrm{FWHM}_{\mathrm{SHEAP}}\ [\mathrm{km\ s^{-1}}])$'
                            ),
                            "lcont": (
                                r'$\log_{10}(\lambda L_{\lambda,\mathrm{Literature}}\ [\mathrm{erg\ s^{-1}}])$',
                                r'$\log_{10}(\lambda L_{\lambda,\mathrm{SHEAP}}\ [\mathrm{erg\ s^{-1}}])$'
                            ),
                            "lline": (
                                r'$\log_{10}(L_{\mathrm{line,Literature}}\ [\mathrm{erg\ s^{-1}}])$',
                                r'$\log_{10}(L_{\mathrm{line,SHEAP}}\ [\mathrm{erg\ s^{-1}}])$'
                            ),
                            "smbh": (
                                r'$\log_{10}(M_{\mathrm{BH,Literature}}\ [M_{\odot}])$',
                                r'$\log_{10}(M_{\mathrm{BH,SHEAP}}\ [M_{\odot}])$'),
                             "smbh_c": (
                                r'$\log_{10}(M_{\mathrm{BH,line}}\ [M_{\odot}])$',
                                r'$\log_{10}(M_{\mathrm{BH,continuum}}\ [M_{\odot}])$'),
                             
                            "rfe": (r'$R_{\mathrm{FeII,Literature}}$ (dimensionless)',r'$R_{\mathrm{FeII,SHEAP}}$ (dimensionless)')
                        }.get(label_mode.lower())
    # ---------- Build pairs ----------
    x_keys = list(x_dict.keys())
    y_keys = list(y_dict.keys())

    if pair_mode not in {"auto", "zip", "product"}:
        raise ValueError("pair_mode must be 'auto', 'zip', or 'product'.")

    if pair_mode == "auto":
        pair_mode = "product" if len(x_keys) != len(y_keys) else "zip"

    if pair_mode == "zip":
        pairs = []
        for xk, yk in zip_longest(x_keys, y_keys, fillvalue=None):
            if xk is None or yk is None:
                continue
            pairs.append((xk, yk))
    else:  # product
        pairs = list(product(x_keys, y_keys))

    # ---------- Auto limits from all data (ignore NaNs/Infs) ----------
    def _finite_log_values(d):
        vals = []
        for arr in d.values():
            arr = np.asarray(arr)
            with np.errstate(divide="ignore", invalid="ignore"):
                lv = np.log10(arr)
            vals.append(lv[np.isfinite(lv)])
        if len(vals) == 0:
            return np.array([])
        return np.concatenate(vals) if len(vals) > 1 else vals[0]

    if lims == "auto" or lims is None:
        all_x = _finite_log_values(x_dict)
        all_y = _finite_log_values(y_dict)
        both = np.concatenate([all_x, all_y]) if all_x.size and all_y.size else (all_x if all_x.size else all_y)

        if both.size:
            dmin, dmax = float(np.min(both)), float(np.max(both))
            if not np.isfinite(dmin) or not np.isfinite(dmax):
                lims_use = (2.5, 4.5)
            else:
                rng = dmax - dmin
                if rng == 0:
                    lims_use = (dmin - 0.1, dmax + 0.1)
                else:
                    pad = lims_pad * rng
                    lims_use = (dmin - pad, dmax + pad)
        else:
            lims_use = (2.5, 4.5)
    else:
        lims_use = lims

    # ---------- Figure ----------
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lims_use); ax.set_ylim(lims_use)

    # 1:1 line and band
    x_fill = np.linspace(lims_use[0], lims_use[1], 200)
    ax.fill_between(x_fill, x_fill - band, x_fill + band, alpha=0.10, color='gray', label=f'±{band} dex band')
    ax.plot(lims_use, lims_use, 'k--', linewidth=1.8, label='1:1 line')

    # Styling
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Cycle helpers
    def cyc(seq):
        while True:
            for item in seq:
                yield item
    marker_cyc = cyc(markers)
    color_cyc  = cyc(colors)

    # Stats container and legend proxies
    stats = {}
    legend_handles = [mlines.Line2D([], [], linestyle='--', color='k', label='1:1 line'),
                      mpatches.Patch(facecolor='gray', alpha=0.10, label=f'±{band} dex band')]

    # ---------- Plot each pair and compute stats ----------
    for (xk, yk) in pairs:
        x = np.asarray(x_dict[xk])
        y = np.asarray(y_dict[yk])

        with np.errstate(divide="ignore", invalid="ignore"):
            x_log = np.log10(x)
            y_log = np.log10(y)

        m = np.isfinite(x_log) & np.isfinite(y_log)

        if m.sum() == 0:
            n_in = 0; n_tot = 0; pct = 0.0
            idx_out = []
        else:
            res = y_log[m] - x_log[m]
            n_tot = int(m.sum())
            n_in = int((np.abs(res) <= band).sum())
            pct = 100.0 * n_in / n_tot if n_tot > 0 else 0.0

            # indices in the ORIGINAL arrays where |Δ| > band
            idx_all  = np.where(m)[0]
            idx_out = idx_all[np.abs(res) > band].tolist()

            mk = next(marker_cyc)
            col = next(color_cyc)

            ax.errorbar(x_log[m], y_log[m],
                        fmt=mk, capsize=0, color=col,
                        markersize=markersize, markeredgewidth=1.5, elinewidth=1.5, alpha=alpha)
            if add_numbers:
                for nn, (xx, yy,_is) in enumerate(zip(x_log, y_log,m)):
                    if _is:
                        ax.text(xx, yy, str(nn), fontsize=10, ha='left', va='bottom')
            #series_label = rf"{xk} vs {yk}" #(|Δ|≤{band} dex: {n_in}/{n_tot}, {pct:.0f}%)"
            series_label = rf"{xk} vs {_pretty_ykey(yk)}"
            legend_handles.append(
                mlines.Line2D([], [], linestyle='none', marker=mk, markersize=markersize,
                              markeredgewidth=1.5, color=col, label=series_label)
            )

        stats[(xk, yk)] = dict(
            x_key=xk, y_key=yk, n_in=n_in, n_tot=n_tot, pct=pct, band=band, idx_out=idx_out
        )

    # Optional: drop the first y tick
    yticks = ax.get_yticks()
    if len(yticks) > 1:
        ax.set_yticks(yticks[1:])

    ax.legend(handles=legend_handles, fontsize=legend_fontsize, frameon=False, markerscale=1.0, ncol=1)

    # ---------- Tight layout and save ----------
    plt.tight_layout()

    saved_file = None
    if save_path is not None:
        # Decide filename
        if os.path.isdir(save_path):
            # Build an informative name from keys
            xname = "_".join(str(k) for k in x_keys) if x_keys else "x"
            yname = "_".join(str(k) for k in y_keys) if y_keys else "y"
            base = f"{yname}_vs_{xname}_logdex_" + what
            saved_file = os.path.join(save_path, base)
        else:
            saved_file = save_path

        # Ensure extension
        ext = f".{save_format.lower()}"
        if not saved_file.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            saved_file = saved_file + ext

        # Ensure directory exists
        os.makedirs(os.path.dirname(saved_file), exist_ok=True)

        # Save with tight bbox
        fig.savefig(saved_file, dpi=dpi, bbox_inches='tight', pad_inches=0.01)

    plt.show()

    # Console summary
    for (xk, yk), s in stats.items():
        print(f"{yk} vs {xk}: |Δ|≤{band} dex -> {s['n_in']}/{s['n_tot']} ({s['pct']:.1f}%), "
              f"out_idx={s['idx_out'][:5]}{'...' if len(s['idx_out'])>5 else ''}")

    return fig, ax, stats, saved_file

def extract_data(arr):
        """
        Extract values and errors from array.
        Returns: values, xerr_lower, xerr_upper (all 1D arrays or None for errors)
        
        Supported shapes:
        - (N,): values only, no errors
        - (N, 1): values only (squeezed), no errors
        - (N, 2): values and symmetric errors
        - (N, 3): values, positive error, negative error
        """
        arr = np.asarray(arr)
        
        if arr.ndim == 1:
            # Shape (N,): just values, no errors
            return arr, None, None
        elif arr.ndim == 2:
            if arr.shape[1] == 1:
                # Shape (N, 1): squeeze to 1D, no errors
                return arr[:, 0], None, None
            elif arr.shape[1] == 2:
                # Shape (N, 2): values and symmetric errors
                return arr[:, 0], arr[:, 1], arr[:, 1]
            elif arr.shape[1] == 3:
                # Shape (N, 3): values, positive error, negative error
                return arr[:, 0], arr[:, 2], arr[:, 1]  # lower=neg, upper=pos
            else:
                # Unexpected shape, use first column only
                print(f"Warning: unexpected shape {arr.shape}, using only first column")
                return arr[:, 0], None, None
        else:
            # Higher dimensions, flatten to 1D
            print(f"Warning: array has {arr.ndim} dimensions, flattening")
            return arr.flatten(), None, None
    
    # ========== HELPER: Convert errors to log space ==========
def errors_to_logspace(values, err_lower, err_upper):
    """
    Convert linear errors to logarithmic errors.
    For log10(x ± σ), the error in log space is approximately:
    Δlog10(x) = σ / (x * ln(10))
    
    Returns: err_lower_log, err_upper_log (or None if no errors)
    """
    if err_lower is None or err_upper is None:
        return None, None
    
    # Avoid division by zero
    values_safe = np.where(values > 0, values, np.nan)
    
    # Convert to log space: Δlog ≈ Δx / (x * ln(10))
    err_lower_log = err_lower / (values_safe * np.log(10))
    err_upper_log = err_upper / (values_safe * np.log(10))
    
    return err_lower_log, err_upper_log
    
    
def plot_logdex_agreement(
    x_dict,
    y_dict,
    xlabel=r'$\log_{10}(\mathrm{FWHM}_{\mathrm{Bernal+25}}\ [\mathrm{km\ s^{-1}}])$',
    ylabel=r'$\log_{10}(\mathrm{FWHM}_{\mathrm{SHEAP}}\ [\mathrm{km\ s^{-1}}])$',
    band=0.3,
    lims="auto",
    lims_pad=0.05,
    pair_mode="auto",
    save_path=None,
    dpi=300,
    save_format="pdf",
    markers=('o', '*', 'X', 'D', '^', 'v', 'P', 's'),
    colors=(
        "#d62728", "#6b67bd", "#2ca02c", "#d62728",
        "#6b67bd", "#8c564b", "#e377c2", "#7f7f7f",
    ),
    markersize=10,
    alpha=0.9,
    legend_fontsize=30,
    label_fontsize=30,
    tick_fontsize=30,
    what="",
    label_mode=None,
    add_numbers=False
):
    """
    Plot y vs x in log10 space with a 1:1 line and a ±band (dex) region.
    
    Handles multi-dimensional data for error bars:
    - (N,): values only, no errors
    - (N, 2): values and symmetric errors
    - (N, 3): values, positive errors, negative errors
    
    Returns
    -------
    fig, ax, stats, saved_file
        stats[(x_key, y_key)] = dict(n_in, n_tot, pct, band, x_key, y_key, idx_out)
        saved_file is the path used for saving, or None if not saved.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from itertools import product, zip_longest
    
    # ========== HELPER: Extract values and errors ==========
    # ========== Label mode handling ==========
    if label_mode:
        xlabel, ylabel = {
            "fwhm": (
                r'$\log_{10}(\mathrm{FWHM}_{\mathrm{ref}}\ [\mathrm{km\ s^{-1}}])$',
                r'$\log_{10}(\mathrm{FWHM}_{\mathrm{SHEAP}}\ [\mathrm{km\ s^{-1}}])$'
            ),
            "lcont": (
                r'$\log_{10}(\lambda L_{\lambda,\mathrm{ref}}\ [\mathrm{erg\ s^{-1}}])$',
                r'$\log_{10}(\lambda L_{\lambda,\mathrm{SHEAP}}\ [\mathrm{erg\ s^{-1}}])$'
            ),
            "lline": (
                r'$\log_{10}(L_{\mathrm{line,ref}}\ [\mathrm{erg\ s^{-1}}])$',
                r'$\log_{10}(L_{\mathrm{line,SHEAP}}\ [\mathrm{erg\ s^{-1}}])$'
            ),
            "smbh": (
                r'$\log_{10}(M_{\mathrm{BH,ref}}\ [M_{\odot}])$',
                r'$\log_{10}(M_{\mathrm{BH,SHEAP}}\ [M_{\odot}])$'
            ),
            "smbh_c": (
                r'$\log_{10}(M_{\mathrm{BH,line}}\ [M_{\odot}])$',
                r'$\log_{10}(M_{\mathrm{BH,continuum}}\ [M_{\odot}])$'
            ),
            "rfe": (
                r'$R_{\mathrm{FeII,ref}}$ (dimensionless)',
                r'$R_{\mathrm{FeII,SHEAP}}$ (dimensionless)'
            )
        }.get(label_mode.lower())
    

    x_keys = list(x_dict.keys())
    y_keys = list(y_dict.keys())
    
    if pair_mode not in {"auto", "zip", "product"}:
        raise ValueError("pair_mode must be 'auto', 'zip', or 'product'.")
    
    if pair_mode == "auto":
        pair_mode = "product" if len(x_keys) != len(y_keys) else "zip"
    
    if pair_mode == "zip":
        pairs = []
        for xk, yk in zip_longest(x_keys, y_keys, fillvalue=None):
            if xk is None or yk is None:
                continue
            pairs.append((xk, yk))
    else:
        pairs = list(product(x_keys, y_keys))
    
    def _finite_log_values(d):
        vals = []
        for arr in d.values():
            values, _, _ = extract_data(arr)
            with np.errstate(divide="ignore", invalid="ignore"):
                lv = np.log10(values)
            vals.append(lv[np.isfinite(lv)])
        if len(vals) == 0:
            return np.array([])
        return np.concatenate(vals) if len(vals) > 1 else vals[0]
    
    if lims == "auto" or lims is None:
        all_x = _finite_log_values(x_dict)
        all_y = _finite_log_values(y_dict)
        both = np.concatenate([all_x, all_y]) if all_x.size and all_y.size else (all_x if all_x.size else all_y)
        
        if both.size:
            dmin, dmax = float(np.min(both)), float(np.max(both))
            if not np.isfinite(dmin) or not np.isfinite(dmax):
                lims_use = (2.5, 4.5)
            else:
                rng = dmax - dmin
                if rng == 0:
                    lims_use = (dmin - 0.1, dmax + 0.1)
                else:
                    pad = lims_pad * rng
                    lims_use = (dmin - pad, dmax + pad)
        else:
            lims_use = (2.5, 4.5)
    else:
        lims_use = lims
    
    # ========== Figure setup ==========
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(lims_use)
    ax.set_ylim(lims_use)
    
    # 1:1 line and band
    x_fill = np.linspace(lims_use[0], lims_use[1], 200)
    ax.fill_between(x_fill, x_fill - band, x_fill + band, alpha=0.10, color='gray', label=f'±{band} dex band')
    ax.plot(lims_use, lims_use, 'k--', linewidth=1.8, label='1:1 line',zorder=10)
    
    # Styling
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    
    # Cycle helpers
    def cyc(seq):
        while True:
            for item in seq:
                yield item
    
    marker_cyc = cyc(markers)
    color_cyc = cyc(colors)
    
    # Stats and legend
    stats = {}
    legend_handles = [
        mlines.Line2D([], [], linestyle='--', color='k', label='1:1 line'),
        mpatches.Patch(facecolor='gray', alpha=0.10, label=f'±{band} dex band')
    ]
    
    # ========== Plot each pair ==========
    for (xk, yk) in pairs:
        # Extract data and errors
        x_vals, x_err_lower, x_err_upper = extract_data(x_dict[xk])
        y_vals, y_err_lower, y_err_upper = extract_data(y_dict[yk])
        
        # Convert to log space
        with np.errstate(divide="ignore", invalid="ignore"):
            x_log = np.log10(x_vals)
            y_log = np.log10(y_vals)
        
        # Convert errors to log space
        x_err_lower_log, x_err_upper_log = errors_to_logspace(x_vals, x_err_lower, x_err_upper)
        y_err_lower_log, y_err_upper_log = errors_to_logspace(y_vals, y_err_lower, y_err_upper)
        
        # Mask for finite values
        m = np.isfinite(x_log) & np.isfinite(y_log)
        
        if m.sum() == 0:
            n_in = 0
            n_tot = 0
            pct = 0.0
            idx_out = []
        else:
            res = y_log[m] - x_log[m]
            n_tot = int(m.sum())
            n_in = int((np.abs(res) <= band).sum())
            pct = 100.0 * n_in / n_tot if n_tot > 0 else 0.0
            
            # Indices where |Δ| > band
            idx_all = np.where(m)[0]
            idx_out = idx_all[np.abs(res) > band].tolist()
            
            # Plotting
            mk = next(marker_cyc)
            col = next(color_cyc)
            
            # Prepare error bars
            xerr = None
            yerr = None
            
            if x_err_lower_log is not None and x_err_upper_log is not None:
                xerr = [x_err_lower_log[m], x_err_upper_log[m]]
            
            if y_err_lower_log is not None and y_err_upper_log is not None:
                yerr = [y_err_lower_log[m], y_err_upper_log[m]]
            
            # Plot with error bars
            ax.errorbar(x_log[m], y_log[m],
                       xerr=xerr, yerr=yerr,
                       fmt=mk, capsize=3, color=col,
                       markersize=markersize, markeredgewidth=1.5, 
                       elinewidth=1.5, alpha=alpha)
            
            # Optional numbering
            if add_numbers:
                for nn, (xx, yy, _is) in enumerate(zip(x_log, y_log, m)):
                    if _is:
                        ax.text(xx, yy, str(nn), fontsize=10, ha='left', va='bottom')
            
            # Legend label
            series_label = f"{xk} vs {yk}"
            legend_handles.append(
                mlines.Line2D([], [], linestyle='none', marker=mk, markersize=markersize,
                             markeredgewidth=1.5, color=col, label=series_label)
            )
        
        stats[(xk, yk)] = dict(
            x_key=xk, y_key=yk, n_in=n_in, n_tot=n_tot, pct=pct, band=band, idx_out=idx_out
        )
    
    # Optional: drop first y tick
    yticks = ax.get_yticks()
    if len(yticks) > 1:
        ax.set_yticks(yticks[1:])
    
    ax.legend(handles=legend_handles, fontsize=legend_fontsize, frameon=False, markerscale=1.0, ncol=1, loc='lower right')
    
    plt.tight_layout()
    
    # ========== Save figure ==========
    saved_file = None
    if save_path is not None:
        if os.path.isdir(save_path):
            xname = "_".join(str(k) for k in x_keys) if x_keys else "x"
            yname = "_".join(str(k) for k in y_keys) if y_keys else "y"
            base = f"{yname}_vs_{xname}_logdex_" + what
            saved_file = os.path.join(save_path, base)
        else:
            saved_file = save_path
        
        ext = f".{save_format.lower()}"
        if not saved_file.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
            saved_file = saved_file + ext
        
        os.makedirs(os.path.dirname(saved_file), exist_ok=True)
        fig.savefig(saved_file, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    
    plt.show()
    
    # Console summary
    for (xk, yk), s in stats.items():
        print(f"{yk} vs {xk}: |Δ|≤{band} dex -> {s['n_in']}/{s['n_tot']} ({s['pct']:.1f}%), "
              f"out_idx={s['idx_out'][:5]}{'...' if len(s['idx_out'])>5 else ''}")
    
    return fig, ax, stats, saved_file



def errors_to_logspace(vals, err_minus, err_plus, *, fill_value=0.30103):
    """
    Convert linear errors on vals to log10-space errors.

    If the log-error is undefined (e.g. v - err_minus <= 0),
    replace NaN with a conservative 100% error (log10(2) ≈ 0.301 dex).

    Parameters
    ----------
    vals : array-like
        Central values (linear space).
    err_minus, err_plus : array-like
        Lower and upper 1-sigma errors (linear space).
    fill_value : float, optional
        Replacement value in log10-space for invalid errors.
        Default is log10(2) ≈ 0.301 dex.

    Returns
    -------
    err_minus_log, err_plus_log : ndarray
        Log10-space asymmetric errors.
    """
    if err_minus is None or err_plus is None:
        return None, None

    v  = np.asarray(vals, dtype=float)
    em = np.asarray(err_minus, dtype=float)
    ep = np.asarray(err_plus, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        base = np.log10(v)

        err_m = base - np.log10(v - em)
        err_p = np.log10(v + ep) - base

    # replace invalid values with "100% error"
    bad_plus = np.where(~np.isfinite(err_p))[0]
    bad_minus = np.where(~np.isfinite(err_m))[0]
    if len(bad_plus)>0:
        print(f"Bad errors plus in index {bad_plus}, replacing per 100% error")
    if len(bad_minus)>0:
        print(f"Bad errors minus in index {bad_minus} 100% error")
    err_m = np.where(np.isfinite(err_m), err_m, fill_value)
    err_p = np.where(np.isfinite(err_p), err_p, fill_value)

    return err_m, err_p


def plot_logdex_agreement_v2(
    data_dict,
    xlabel=r'$\log_{10}(\mathrm{FWHM}_{\mathrm{ref}}\ [\mathrm{km\ s^{-1}}])$',
    ylabel=r'$\log_{10}(\mathrm{FWHM}_{\mathrm{SHEAP}}\ [\mathrm{km\ s^{-1}}])$',
    ref_label="ref",   # <<< NEW
    band=0.3,
    lims="auto",
    lims_pad=0.05,
    save_file=None,
    dpi=300,
    save_format="pdf",
    markers=('o', '*', 'X', 'D', '^', 'v', 'P', 's'),
    colors=(
        "#d62728", "#6b67bd", "#2ca02c", "#d62728",
        "#6b67bd", "#8c564b", "#e377c2", "#7f7f7f",
    ),
    markersize=10,
    alpha=0.9,
    legend_fontsize=30,
    label_fontsize=30,
    tick_fontsize=30,
    what="",
    label_mode=None,
    add_numbers=False,
    name_line ="line",
    legend_loc = "lower right",
):
    """
    Plot y vs x in log10 space with a 1:1 line and a ±band (dex) region, for
    multiple series defined by a single dictionary.

    Parameters
    ----------
    data_dict : dict
        Dictionary defining series. Each key is the legend label for a series.
        Each value must be a dict with:
            {
              "x": array_like,
              "y": array_like
            }

        Array conventions (for BOTH x and y)
        -----------------------------------
        Accepted shapes:
        - (N,) or (1, N): values only (no errors)
        - (2, N): [values, symmetric_error]
        - (3, N): [values, err_plus, err_minus]
                 NOTE: here err_plus and err_minus are *positive magnitudes*
                 (not signed). i.e. value+err_plus, value-err_minus.

        Any of the above can also be provided as (N, K) with K in {1,2,3};
        it will be interpreted accordingly after transpose if needed.

    Other parameters
    ----------------
    xlabel, ylabel : str
        Axis labels.
    band : float
        Dex agreement band: |log10(y) - log10(x)| <= band.
    lims : "auto" or (min, max)
        Axis limits in log10 space.
    lims_pad : float
        Fractional padding applied when lims="auto".
    save_path : str or None
        Directory or full filepath to save.
    save_format : str
        File format, e.g. "pdf" or "png".
    what : str
        Extra string appended to output filename (if save_path is a directory).
    add_numbers : bool
        If True, annotate points with their index (within each series).

    Returns
    -------
    fig, ax, stats, saved_file
        stats[label] = dict(n_in, n_tot, pct, band, idx_out)
        saved_file is the path used for saving, or None if not saved.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    # ---------------------- label presets ----------------------
    if label_mode:
        _lm = label_mode.lower()
        presets = {
                    "fwhm": (
                        rf'$\log_{{10}}(\mathrm{{FWHM}}_{{\mathrm{{{ref_label}}}}}\ [\mathrm{{km\ s^{{-1}}}}])$',
                        r'$\log_{10}(\mathrm{FWHM}_{\mathrm{SHEAP}}\ [\mathrm{km\ s^{-1}}])$'
                    ),
                    "lcont": (
                        rf'$\log_{{10}}(\lambda L_{{\lambda,\mathrm{{{ref_label}}}}}\ [\mathrm{{erg\ s^{{-1}}}}])$',
                        r'$\log_{10}(\lambda L_{\lambda,\mathrm{SHEAP}}\ [\mathrm{erg\ s^{-1}}])$'
                    ),
                    "lline": (
                        rf'$\log_{{10}}(L_{{\mathrm{{{name_line},{ref_label}}}}}\ [\mathrm{{erg\ s^{{-1}}}}])$',
                        r'$\log_{10}(L_{\mathrm{line,SHEAP}}\ [\mathrm{erg\ s^{-1}}])$'
                    ),
                    "smbh": (
                        rf'$\log_{{10}}(M_{{\mathrm{{BH,{ref_label}}}}}\ [M_\odot])$',
                        r'$\log_{10}(M_{\mathrm{BH,SHEAP}}\ [M_\odot])$'
                    ),
                    "smbh_c": (
                        rf'$\log_{{10}}(M_{{\mathrm{{BH,line}}}}\ [M_\odot])$',
                        rf'$\log_{{10}}(M_{{\mathrm{{BH,continuum}}}}\ [M_\odot])$'
                    ),
                    "rfe": (
                        rf'$R_{{\mathrm{{FeII,{ref_label}}}}}$ (dimensionless)',
                        r'$R_{\mathrm{FeII,SHEAP}}$ (dimensionless)'
                    ),}
        if _lm in presets:
            xlabel, ylabel = presets[_lm]

    # ---------------------- helpers ----------------------
    def _as_2d(a):
        a = np.asarray(a)
        if a.ndim == 1:
            return a[None, :]  # (1, N)
        return a

    def extract_data(arr):
        """
        Return (values, err_minus, err_plus) in linear space.

        err_minus/err_plus are 1D arrays of positive magnitudes, or None.
        """
        a = _as_2d(arr)

        # accept (N, K) where K in {1,2,3} -> transpose to (K, N)
        if a.shape[0] not in (1, 2, 3) and a.shape[1] in (1, 2, 3):
            a = a.T

        if a.shape[0] not in (1, 2, 3):
            raise ValueError(
                f"Expected shape (N,), (1,N), (2,N), (3,N) (or transposed). Got {a.shape}."
            )

        vals = np.asarray(a[0], dtype=float)

        if a.shape[0] == 1:
            return vals, None, None

        if a.shape[0] == 2:
            e = np.asarray(a[1], dtype=float)
            e = np.abs(e)
            return vals, e, e

        # a.shape[0] == 3: [values, err_plus, err_minus]
        e_plus  = np.asarray(a[1], dtype=float)
        e_minus = np.asarray(a[2], dtype=float)
        e_plus  = np.abs(e_plus)
        e_minus = np.abs(e_minus)
        return vals, e_minus, e_plus

   

    def _finite_log_values_from_series(series):
        xv, _, _ = extract_data(series["x"])
        yv, _, _ = extract_data(series["y"])
        with np.errstate(divide="ignore", invalid="ignore"):
            xl = np.log10(xv)
            yl = np.log10(yv)
        both = np.concatenate([xl[np.isfinite(xl)], yl[np.isfinite(yl)]])
        return both

    # ---------------------- determine limits ----------------------
    if lims == "auto" or lims is None:
        vals = []
        for _, series in data_dict.items():
            both = _finite_log_values_from_series(series)
            if both.size:
                vals.append(both)
        if len(vals):
            both = np.concatenate(vals)
            dmin, dmax = float(np.nanmin(both)), float(np.nanmax(both))
            if np.isfinite(dmin) and np.isfinite(dmax):
                rng = dmax - dmin
                if rng == 0:
                    lims_use = (dmin - 0.1, dmax + 0.1)
                else:
                    pad = lims_pad * rng
                    lims_use = (dmin - pad, dmax + pad)
            else:
                lims_use = (2.5, 4.5)
        else:
            lims_use = (2.5, 4.5)
    else:
        lims_use = lims

    # ---------------------- figure ----------------------
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lims_use)
    ax.set_ylim(lims_use)

    x_fill = np.linspace(lims_use[0], lims_use[1], 200)
    ax.fill_between(
        x_fill, x_fill - band, x_fill + band,
        alpha=0.10, color="gray", label=rf"$\pm {band}$ dex band"
    )
    ax.plot(lims_use, lims_use, "k--", linewidth=1.8, label="1:1 line", zorder=10)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)

    # cycle
    def cyc(seq):
        while True:
            for item in seq:
                yield item

    marker_cyc = cyc(markers)
    color_cyc  = cyc(colors)

    stats = {}
    legend_handles = [
        mlines.Line2D([], [], linestyle="--", color="k", label="1:1 line"),
        mpatches.Patch(facecolor="gray", alpha=0.10, label=rf"$\pm {band}$ dex band"),
    ]

    # ---------------------- plot each series ----------------------
    for label, series in data_dict.items():
        x_vals, x_err_m, x_err_p = extract_data(series["x"])
        y_vals, y_err_m, y_err_p = extract_data(series["y"])

        with np.errstate(divide="ignore", invalid="ignore"):
            x_log = np.log10(x_vals)
            y_log = np.log10(y_vals)

        x_err_m_log, x_err_p_log = errors_to_logspace(x_vals, x_err_m, x_err_p)
        y_err_m_log, y_err_p_log = errors_to_logspace(y_vals, y_err_m, y_err_p)

        m = np.isfinite(x_log) & np.isfinite(y_log)

        if m.sum() == 0:
            stats[label] = dict(n_in=0, n_tot=0, pct=0.0, band=band, idx_out=[])
            continue

        res = y_log[m] - x_log[m]
        n_tot = int(m.sum())
        n_in  = int((np.abs(res) <= band).sum())
        pct   = 100.0 * n_in / n_tot if n_tot > 0 else 0.0

        idx_all = np.where(m)[0]
        idx_out = idx_all[np.abs(res) > band].tolist()

        mk  = next(marker_cyc)
        col = next(color_cyc)

        xerr = None
        yerr = None
        if x_err_m_log is not None and x_err_p_log is not None:
            xerr = [x_err_m_log[m], x_err_p_log[m]]
        if y_err_m_log is not None and y_err_p_log is not None:
            yerr = [y_err_m_log[m], y_err_p_log[m]]
        
        
        ax.errorbar(
            x_log[m], y_log[m],
            xerr=xerr, yerr=yerr,
            fmt=mk, capsize=3, color=col,
            markersize=markersize, markeredgewidth=1.5,
            elinewidth=2, alpha=alpha
        )

        if add_numbers:
            for i, (xx, yy, ok) in enumerate(zip(x_log, y_log, m)):
                if ok:
                    ax.text(xx, yy, str(i), fontsize=10, ha="left", va="bottom")

        legend_handles.append(
            mlines.Line2D([], [], linestyle="none", marker=mk, markersize=markersize,
                          markeredgewidth=1.5, color=col, label=label)
        )

        stats[label] = dict(n_in=n_in, n_tot=n_tot, pct=pct, band=band, idx_out=idx_out)

    # optional: drop first y tick
    yticks = ax.get_yticks()
    if len(yticks) > 1:
        ax.set_yticks(yticks[1:])

    ax.legend(
        handles=legend_handles,
        fontsize=legend_fontsize,
        frameon=False,
        markerscale=1.0,
        ncol=1,
        loc=legend_loc,
    )
    if lims:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
    plt.tight_layout()
    

    # ---------------------- save ----------------------
    # saved_file = None
    # if save_path is not None:
    #     if os.path.isdir(save_path):
    #         base = f"logdex_agreement_{what}" if what else "logdex_agreement"
    #         saved_file = os.path.join(save_path, base)
    #     else:
    #         saved_file = save_path

    #ext = f".{save_format.lower()}"
    # if not saved_file.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
    #     saved_file = saved_file + ext

    #     os.makedirs(os.path.dirname(saved_file), exist_ok=True)
    if save_file:
        fig.savefig(save_file, dpi=dpi, bbox_inches="tight", pad_inches=0.01)

    plt.show()

    # console summary
    for label, s in stats.items():
        print(
            f"{label}: |Δ|≤{band} dex -> {s['n_in']}/{s['n_tot']} ({s['pct']:.1f}%), "
            f"out_idx={s['idx_out'][:5]}{'...' if len(s['idx_out'])>5 else ''}"
        )

    return fig, ax, stats





def plot_ratio_with_sn(
	x, y,
	xerr_low=None, xerr_up=None,
	yerr_low=None, yerr_up=None,
	sn=None,
	*,
	lims=(-0.1, 1.1),
	label_template="",
	cmap="viridis",
	ms=70,              # scatter marker size (points^2)
	alpha=0.8,
	label_colorbar=r'$\log_{10}~\mathrm{S/N}$',
    xlabel = r'$\mathrm{Stars/Cont.~ratio~at~5100~\AA~(Bernal+2025)}$',
    ylabel = r'$\mathrm{Stars/Cont.~ratio~at~5100~\AA~(SHEAP)}$',
	err_alpha=0.65,
	err_lw=1.2,
	capsize=2,
):
	"""
	Plot y vs x with asymmetric errorbars and encode S/N as point color.

	This version draws *per-point* errorbars with the same color as the marker.

	Parameters
	----------
	x, y : array-like, shape (N,)
		Values to plot (e.g. Bernal+2025 vs SHEAP).
	xerr_low, xerr_up, yerr_low, yerr_up : array-like, shape (N,), optional
		Asymmetric errors. If provided, they are used as [low, up].
	sn : array-like, shape (N,)
		Per-point S/N values (already in log10 if you want log10 on the colorbar).
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	n = x.size

	def _asarr(a):
		if a is None:
			return None
		a = np.asarray(a, dtype=float)
		if a.size != n:
			raise ValueError(f"Array has size {a.size}, expected {n}.")
		return a

	xerr_low = _asarr(xerr_low)
	xerr_up  = _asarr(xerr_up)
	yerr_low = _asarr(yerr_low)
	yerr_up  = _asarr(yerr_up)

	# Build asymmetric error arrays (2, N)
	xerr = None if (xerr_low is None or xerr_up is None) else np.vstack([xerr_low, xerr_up])
	yerr = None if (yerr_low is None or yerr_up is None) else np.vstack([yerr_low, yerr_up])

	if sn is None:
		raise ValueError("sn must be provided (array of length N).")
	sn = np.asarray(sn, dtype=float)
	if sn.size != n:
		raise ValueError(f"sn has size {sn.size}, expected {n}.")

	fig, ax = plt.subplots(figsize=(12, 12))

	# Colormap normalization used for BOTH scatter and colored errorbars
	norm = Normalize(vmin=np.nanmin(sn), vmax=np.nanmax(sn))
	cmap_obj = plt.get_cmap(cmap)

	# Errorbars (colored per point to match marker)
	if (xerr is not None) or (yerr is not None):
		for i in range(n):
			color_i = cmap_obj(norm(sn[i]))

			xerri = None if xerr is None else np.array([[xerr[0, i]], [xerr[1, i]]])
			yerri = None if yerr is None else np.array([[yerr[0, i]], [yerr[1, i]]])

			ax.errorbar(
				x[i], y[i],
				xerr=xerri, yerr=yerri,
				fmt="none",
				ecolor=color_i,
				elinewidth=err_lw,
				alpha=err_alpha,
				capsize=capsize,
				zorder=1,
			)

	# Scatter with S/N color encoding (keep this so colorbar is correct)
	sc = ax.scatter(
		x, y,
		c=sn,
		s=ms,
		cmap=cmap_obj,
		norm=norm,
		alpha=alpha,
		marker="*",
		edgecolors="none",
		zorder=2,
		label=label_template,
	)

	# 1:1 line
	ax.plot(lims, lims, "k--", linewidth=1.8, label="1:1 line", zorder=20)
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlim(lims)
	ax.set_ylim(lims)

	# Labels
	ax.set_xlabel(xlabel,
		fontsize=25,
	)
	ax.set_ylabel(ylabel,
		fontsize=25,
	)
	ax.tick_params(axis="both", which="major", labelsize=25)

	# Colorbar with matched height
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="4%", pad=0.08)

	cbar = plt.colorbar(sc, cax=cax)
	cbar.set_label(label_colorbar, fontsize=25)
	cbar.ax.tick_params(labelsize=20)

	ax.legend(
		fontsize=20,
		frameon=False,
		markerscale=1.6,
		handlelength=2.6,
	)

	plt.tight_layout()

	return fig, ax, sn

def _as_2d(a):
        a = np.asarray(a)
        if a.ndim == 1:
            return a[None, :]
        return a

def extract_data(arr):
    """
    Return (values, err_minus, err_plus) in linear space.
    err_minus/err_plus are positive magnitudes, or None.
    """
    a = _as_2d(arr)

    # accept (N, K) where K in {1,2,3} -> transpose to (K, N)
    if a.shape[0] not in (1, 2, 3) and a.shape[1] in (1, 2, 3):
        a = a.T

    if a.shape[0] not in (1, 2, 3):
        raise ValueError(
            f"Expected shape (N,), (1,N), (2,N), (3,N) (or transposed). Got {a.shape}."
        )

    vals = np.asarray(a[0], dtype=float)

    if a.shape[0] == 1:
        return vals, None, None

    if a.shape[0] == 2:
        e = np.abs(np.asarray(a[1], dtype=float))
        return vals, e, e

    e_plus  = np.abs(np.asarray(a[1], dtype=float))
    e_minus = np.abs(np.asarray(a[2], dtype=float))
    return vals, e_minus, e_plus

def _finite_log_values_from_series(series):
    xv, _, _ = extract_data(series["x"])
    yv, _, _ = extract_data(series["y"])
    with np.errstate(divide="ignore", invalid="ignore"):
        xl = np.log10(xv)
        yl = np.log10(yv)
    both = np.concatenate([xl[np.isfinite(xl)], yl[np.isfinite(yl)]])
    return both

def plot_logdex_agreement_v3(
    data_dict,
    *,
    sn=None,  # <<< NEW: common S/N array for ALL series
    xlabel=r'$\log_{10}(\mathrm{FWHM}_{\mathrm{ref}}\ [\mathrm{km\ s^{-1}}])$',
    ylabel=r'$\log_{10}(\mathrm{FWHM}_{\mathrm{SHEAP}}\ [\mathrm{km\ s^{-1}}])$',
    ref_label="ref",
    band=0.3,
    lims="auto",
    lims_pad=0.05,
    save_file=None,
    dpi=300,
    save_format="pdf",
    markers=('o', '*', 'X', 'D', '^', 'v', 'P', 's'),
    colors=(
        "#d62728", "#6b67bd", "#2ca02c", "#d62728",
        "#6b67bd", "#8c564b", "#e377c2", "#7f7f7f",
    ),
    markersize=10,
    alpha=0.9,
    legend_fontsize=30,
    label_fontsize=30,
    tick_fontsize=30,
    what="",
    label_mode=None,
    add_numbers=False,
    name_line="line",
    legend_loc="lower right",
    # --- NEW colormap knobs ---
    cmap="viridis",
    label_colorbar=r'$\log_{10}\,\mathrm{S/N}$',
    colorbar=True,
    colorbar_pad=0.08,
    colorbar_size="4%",
    err_alpha=0.65,
    err_lw=2.0,
    capsize=3,
    ref_wavelenght = 3000,
    
):
    """
    Same as your v2, but now point color is driven by a *common* S/N array shared
    across all series in data_dict. Markers still distinguish series.
    """

    if label_mode:
        _lm = label_mode.lower()
        presets = {
            "fwhm": (
                rf'$\log_{{10}}(\mathrm{{FWHM}}_{{\mathrm{{{ref_label}}}}}\ [\mathrm{{km\ s^{{-1}}}}])$',
                r'$\log_{10}(\mathrm{FWHM}_{\mathrm{SHEAP}}\ [\mathrm{km\ s^{-1}}])$'
            ),
            "lcont": (
                rf'$\log_{{10}}(\lambda L_{{\lambda,\mathrm{{{ref_label}}}}} ({ref_wavelenght} Å)\ [\mathrm{{erg\ s^{{-1}}}}])$',
                 rf'$\log_{{10}}(\lambda L_{{\lambda,\mathrm{{{"SHEAP"}}}}} ({ref_wavelenght} Å)\ [\mathrm{{erg\ s^{{-1}}}}])$',
                
            ),
            "lline": (
                rf'$\log_{{10}}(L_{{\mathrm{{{name_line},{ref_label}}}}} \ [\mathrm{{erg\ s^{{-1}}}}])$',
                r'$\log_{10}(L_{\mathrm{line,SHEAP}}\ [\mathrm{erg\ s^{-1}}])$'
            ),
            "smbh": (
                rf'$\log_{{10}}(M_{{\mathrm{{BH,{ref_label}}}}}\ [M_\odot])$',
                r'$\log_{10}(M_{\mathrm{BH,SHEAP}}\ [M_\odot])$'
            ),
            "smbh_c": (
                rf'$\log_{{10}}(M_{{\mathrm{{BH,line}}}}\ [M_\odot])$',
                rf'$\log_{{10}}(M_{{\mathrm{{BH,continuum}}}}\ [M_\odot])$'
            ),
            "rfe": (
                rf'$R_{{\mathrm{{FeII,{ref_label}}}}}$ (dimensionless)',
                r'$R_{\mathrm{FeII,SHEAP}}$ (dimensionless)'
            ),
        }
        if _lm in presets:
            xlabel, ylabel = presets[_lm]    

    # ---------------------- determine limits ----------------------
    if lims == "auto" or lims is None:
        vals = []
        for _, series in data_dict.items():
            both = _finite_log_values_from_series(series)
            if both.size:
                vals.append(both)
        if len(vals):
            both = np.concatenate(vals)
            dmin, dmax = float(np.nanmin(both)), float(np.nanmax(both))
            if np.isfinite(dmin) and np.isfinite(dmax):
                rng = dmax - dmin
                if rng == 0:
                    lims_use = (dmin - 0.1, dmax + 0.1)
                else:
                    pad = lims_pad * rng
                    lims_use = (dmin - pad, dmax + pad)
    else:
        lims_use = lims
    # ---------------------- validate common S/N ----------------------
    if sn is not None:
        sn = np.asarray(sn, dtype=float)
        # infer N from the first series x
        first_series = next(iter(data_dict.values()))
        xv0, _, _ = extract_data(first_series["x"])
        if sn.size != xv0.size:
            raise ValueError(
                f"sn must have the same length as the underlying arrays. "
                f"Got sn.size={sn.size}, expected {xv0.size}.")

    # ---------------------- figure ----------------------
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal", adjustable="box")
   
    x_fill = np.linspace(lims_use[0], lims_use[1], 200)
    ax.fill_between(x_fill, x_fill - band, x_fill + band,alpha=0.10, color="gray", label=rf"$\pm {band}$ dex band")
    
    ax.plot(lims_use, lims_use, "k--", linewidth=1.8, label="1:1 line", zorder=10)
    def cyc(seq):
        while True:
            for item in seq:
                yield item

    marker_cyc = cyc(markers)
    color_cyc  = cyc(colors)

    stats = {}
    legend_handles = [
        mlines.Line2D([], [], linestyle="--", color="k", label="1:1 line"),
        mpatches.Patch(facecolor="gray", alpha=0.10, label=rf"$\pm {band}$ dex band"),]

    # --- Colormap setup (shared across series) ---
    cmap_obj = plt.get_cmap(cmap)
    norm = None
    if sn is not None:
        norm = Normalize(vmin=np.nanmin(sn), vmax=np.nanmax(sn))

    # keep one mappable for the colorbar
    sc_for_cbar = None

    # ---------------------- plot each series ----------------------
    for label, series in data_dict.items():
        x_vals, x_err_m, x_err_p = extract_data(series["x"])
        y_vals, y_err_m, y_err_p = extract_data(series["y"])

        with np.errstate(divide="ignore", invalid="ignore"):
            x_log = np.log10(x_vals)
            y_log = np.log10(y_vals)

        # NOTE: your code uses this function; keep as-is
        x_err_m_log, x_err_p_log = errors_to_logspace(x_vals, x_err_m, x_err_p)
        y_err_m_log, y_err_p_log = errors_to_logspace(y_vals, y_err_m, y_err_p)

        m = np.isfinite(x_log) & np.isfinite(y_log)

        if m.sum() == 0:
            stats[label] = dict(n_in=0, n_tot=0, pct=0.0, band=band, idx_out=[])
            continue

        res = y_log[m] - x_log[m]
        n_tot = int(m.sum())
        n_in  = int((np.abs(res) <= band).sum())
        pct   = 100.0 * n_in / n_tot if n_tot > 0 else 0.0

        idx_all = np.where(m)[0]
        idx_out = idx_all[np.abs(res) > band].tolist()

        mk  = next(marker_cyc)
        col = next(color_cyc)

        # Build asymmetric errors in log space (2, N_masked)
        xerr = None
        yerr = None
        if x_err_m_log is not None and x_err_p_log is not None:
            xerr = np.vstack([x_err_m_log[m], x_err_p_log[m]])
        if y_err_m_log is not None and y_err_p_log is not None:
            yerr = np.vstack([y_err_m_log[m], y_err_p_log[m]])

        # --- Colors per point from common sn (same indexing as the original arrays) ---
        if sn is not None:
            sn_m = sn[m]
            facecols = cmap_obj(norm(sn_m))
        else:
            facecols = None  # fallback: use series color

        # --- Draw per-point errorbars so they match marker color ---
        if (xerr is not None) or (yerr is not None):
            xm = x_log[m]
            ym = y_log[m]
            for j in range(xm.size):
                ecolor_j = col if facecols is None else facecols[j]

                xerr_j = None
                yerr_j = None
                if xerr is not None:
                    xerr_j = np.array([[xerr[0, j]], [xerr[1, j]]])
                if yerr is not None:
                    yerr_j = np.array([[yerr[0, j]], [yerr[1, j]]])
                
                ax.errorbar(
                    xm[j], ym[j],
                    xerr=xerr_j, yerr=yerr_j,
                    fmt=mk,
                    ecolor=ecolor_j,
                    color = ecolor_j,
                    elinewidth=err_lw,
                    alpha=err_alpha,
                    capsize=capsize,
                    zorder=1,
                )
        linewidths = 0.5
        
        #  ax.errorbar(
        #     x_log[m], y_log[m],
        #     xerr=xerr, yerr=yerr,
        #     fmt=mk, capsize=3, color=col,
        #     markersize=markersize, markeredgewidth=1.5,
        #     elinewidth=2, alpha=alpha
        # )
        # # --- Scatter: facecolor from sn, edgecolor by series (so you still see groups) ---
        # if facecols is None:
        #     sc = ax.scatter(
        #         x_log[m], y_log[m],
        #         marker=mk,
        #         s=markersize**2 / 2.0,  # convert "markersize-like" to scatter area
        #         c=col,
        #         alpha=alpha,
        #         edgecolors=col,
        #         linewidths=linewidths,
        #         zorder=2,
        #         label=label,
        #     )
        # else:
        #     sc = ax.scatter(
        #         x_log[m], y_log[m],
        #         marker=mk,
        #         s=markersize**2 / 2.0,
        #         facecolors=facecols,
        #         edgecolors=col,
        #         linewidths=linewidths,
        #         alpha=alpha,
        #         zorder=2,
        #         label=label,
        #     )
        #     if sc_for_cbar is None:
        #         sc_for_cbar = ax.scatter(
        #             [], [], c=[], cmap=cmap_obj, norm=norm
        #         )  # dummy mappable for colorbar
        #         # make sure it carries the same cmap/norm
        #         sc_for_cbar.set_cmap(cmap_obj)
        #         sc_for_cbar.set_norm(norm)

        if add_numbers:
            for i, (xx, yy, ok) in enumerate(zip(x_log, y_log, m)):
                if ok:
                    ax.text(xx, yy, str(i), fontsize=10, ha="left", va="bottom")

        # legend handle for series (marker only; colorbar explains facecolor)
        legend_handles.append(
    mlines.Line2D(
        [], [], linestyle="none", marker=mk,
        markersize=markersize,
        markeredgewidth=3.0,
        markerfacecolor="white" if sn is not None else col,  # better than "none" in PDFs
        markeredgecolor="k" if sn is not None else col,
        label=label,
    )
)

        stats[label] = dict(n_in=n_in, n_tot=n_tot, pct=pct, band=band, idx_out=idx_out)



    

    if colorbar and (sn is not None):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=colorbar_size, pad=colorbar_pad)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        mappable.set_array(sn)  # for matplotlib compatibility
        cbar = plt.colorbar(mappable, cax=cax)
        cbar.set_label(label_colorbar, fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=tick_fontsize)
    
    ax.legend(handles=legend_handles,fontsize=legend_fontsize,frameon=False,markerscale=1.0,ncol=1,loc=legend_loc,)
    
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    
    ax.set_xlim(lims_use)
    ax.set_ylim(lims_use)
    plt.tight_layout()
    ticks = ax.get_xticks()
    ax.set_yticks(ticks)
    if min(ticks) == min(lims_use): # 
        ax.set_yticks(ticks[1:])
        ax.set_xticks(ticks[1:])
    

    saved_file = None
    if save_file is not None:
        saved_file = save_file
        fig.savefig(saved_file, dpi=dpi, format=save_format, bbox_inches="tight")

    return fig, ax, stats, saved_file


def log10_to_linear(
	logval,
	logerr,
):
	"""
	Convert log10(x) with uncertainty to linear x with asymmetric errors.

	Parameters
	----------
	logval : float or ndarray
		log10(x)
	logerr : float or ndarray
		1-sigma uncertainty in log10 space

	Returns
	-------
	val : ndarray
		Linear value x
	err_minus : ndarray
		Lower uncertainty (x - x_low)
	err_plus : ndarray
		Upper uncertainty (x_high - x)
	"""
	logval = np.asarray(logval, dtype=float)
	logerr = np.asarray(logerr, dtype=float)

	val = 10.0 ** logval
	err_plus = 10.0 ** (logval + logerr) - val
	err_minus = val - 10.0 ** (logval - logerr)

	return val, err_minus, err_plus
