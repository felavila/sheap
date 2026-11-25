import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import pandas as pd
import numpy as np 

def quicklook(spectra, idx: list, ax=None, xlim=None, ylim=None,add_text=True,figsize= (15, 5)):
    """
    Produce a quick errorbar plot of flux vs. wavelength.

    Parameters
    ----------
    idx : int
        Spectrum index to plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into (creates new if None).
    xlim : tuple, optional
        X-axis limits as (xmin, xmax).
    ylim : tuple, optional
        Y-axis limits as (ymin, ymax).

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if isinstance(idx,int):
        idx = [idx]
    
    for i in idx:
        lam, flux, err = spectra[i]
        line, caplines, barlinecols = ax.errorbar(lam,flux, yerr=err,zorder=1,label=names[i],)

        for barline in barlinecols:
            barline.set_alpha(0.1)

    # Default xlim and ylim if not provided
    if xlim is None:
        xlim = (jnp.nanmin(lam), jnp.nanmax(lam))
    if ylim is None:
        ylim = (0, jnp.nanmax(flux) * 1.02)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_xlabel("Rest wavelength [Å]", fontsize=25)
    
    flux_unit=r"$\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$"
    ax.set_ylabel(f"Flux [{flux_unit}]", fontsize=25)
    ax.tick_params(axis='both', labelsize=25)
    ax.yaxis.offsetText.set_fontsize(25)
    
    # Plot ID label outside main plot area, above-left
    if add_text:
        if len(idx) == 1:
            ax.text(
                0.7,
                1.0,
                f"ID {self.names[idx[0]]} ({idx[0]}) \nz = {self.z[idx[0]]} ",
                fontsize=20,
                transform=ax.transAxes,
                ha='left',
                va='bottom',
            )
        else:
            ax.legend(fontsize=14, frameon=False)
    ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))

    return ax

    #@property
def plot_redshift_signal2noise_distribution(spectra,z):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    axes[0].hist(z, bins=20, color="#214994", edgecolor='black', alpha=0.8)
    axes[0].set_xlabel("Redshift (z)", fontsize=16)
    axes[0].set_ylabel("Number of objects", fontsize=16)
    #axes[0].set_title("Redshift Distribution", fontsize=18)
    axes[0].tick_params(axis='both', which='major', labelsize=14)

    snr = np.nanmean(spectra[:, 1, :] / spectra[:, 2, :], axis=1)
    axes[1].hist(snr, bins=20, color="#f5b041", edgecolor='black', alpha=0.8)
    axes[1].set_xlabel("Mean Signal-to-Noise Ratio", fontsize=16)
    #axes[1].set_title("S/N Distribution", fontsize=18)
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    return axes
    
def plot_chi2(chi2_red):
    #if not hasattr(self, "result"):
     #   raise RuntimeError("self.result should exist to run this.")
    #chi2_model = self.result.chi2_red  # or whatever your model object is
    chi2_model = np.asarray(chi2_model)

    nan_mask = ~np.isfinite(chi2_model)
    nan_idx = np.where(nan_mask)[0]
    chi2_model = chi2_model[~nan_mask]
    print(f"NaN / non-finite entries at indices: {nan_idx.tolist()}")
    print(f"Number of NaNs / non-finite values: {nan_idx.size}")
    #mask_range_model = (chi2_model > 0.) & (chi2_model < 5.)
    #frac_model_0_5 = np.nanmean(mask_range_model) * 100.0

    # --- Define bins from model only ---
    chi2_min = chi2_model.min()
    chi2_max = chi2_model.max()
    bins = np.linspace(chi2_min, chi2_max, 40)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(chi2_model,bins=bins,alpha=0.7,color="#d62728",)

    ax.set_xlabel(r"Reduced $\chi^2$", fontsize=18)
    ax.set_ylabel("Number of spectra", fontsize=18)
    
    ax.tick_params(axis="both", labelsize=14)

    ax.axvline(np.nanmedian(chi2_model), linestyle="--",label=fr"Reduced $\chi^2$ median = {np.nanmedian(chi2_model):.2f}",c="b")
    ax.axvline(1.0, linestyle="--",label=fr"Reduced $\chi^2$ = 1",c="k")
    ax.set_xlim(0.1,ax.get_xlim()[-1])
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout()
    plt.show()

# def plot_param_distribution(self,param_name,no_log=False):
#     #TODO what about this plot but for posterior->
#     params_keys = self.result.params_dict.keys()
#     if param_name not in params_keys:
#         raise KeyError(f"param_name '{param_name}' is not available. " f"Available param names: {list(params_keys)}")
    
#     #data = []

#     scale = self.result.scale
#     param_index = self.result.params_dict[param_name]
#     param_values = self.result.params[:,param_index]
#     init_value = float(self.result.initial_params[param_index])
#     if "amplitude" in param_name:
#             param_values /= scale
#     elif "logamp" in param_name:
#             param_values -= np.log10(scale)
    

#     constraints = self.result.constraints[param_index]
#     if no_log:
#         constraints = 10**constraints
#         #constraints[1] = 10**constraints[1]
#         init_value = 10**init_value
#         param_values = 10**param_values
#         param_name = param_name.replace("log","")
#     bins = np.linspace(np.min(param_values),np.max(param_values), 40)
#     fig, ax = plt.subplots(figsize=(10, 6))

#     ax.hist(
#         param_values,
#         bins=bins,
#         alpha=0.7,
#         color="#2781d65c",)

#     ax.set_xlabel(param_name, fontsize=18)
#     ax.set_ylabel("Number of spectra", fontsize=18)
#     ax.set_title(f"param number {param_index}")
#     ax.tick_params(axis="both", labelsize=14)
#     ax.axvline(init_value, linestyle="--",label="init value",c="r")
#     ax.axvline(constraints[0], linestyle="--",label="min value",c="k")
#     ax.axvline(constraints[1], linestyle="--",label="max value",c="k")
#     ax.legend(fontsize=14, frameon=False)
#     fig.tight_layout()
#     plt.show()
#     return param_values