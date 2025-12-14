import numpy as np 
import matplotlib.pyplot as plt 


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
                                r'$\log_{10}(M_{\mathrm{BH,SHEAP}}\ [M_{\odot}])$'),
                             "smbh_c": (
                                r'$\log_{10}(M_{\mathrm{BH,line}}\ [M_{\odot}])$',
                                r'$\log_{10}(M_{\mathrm{BH,continuum}}\ [M_{\odot}])$'),
                             
                            "rfe": (r'$R_{\mathrm{FeII,ref}}$ (dimensionless)',r'$R_{\mathrm{FeII,SHEAP}}$ (dimensionless)')
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


def plot_logdex_agreement(
    x_dict,
    y_dict,
    xlabel=r'$\log_{10}(\mathrm{FWHM}_{\mathrm{ref}}\ [\mathrm{km\ s^{-1}}])$',
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
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
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
    
    ax.legend(handles=legend_handles, fontsize=legend_fontsize, frameon=False, markerscale=1.0, ncol=1)
    
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