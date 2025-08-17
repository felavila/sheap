"""This module handles ?."""
__author__ = 'felavila'

__all__ = [
    "SheapPlot_old",
    "plot_a_spectra",
    "plot_region",
]

# import seaborn as sns
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#from sheap.Minimizer.functions import linear_combination


# class Sheap_pca_ploting:
#     import numpy as np

#     def __init__(
#         self, test_clase, masked_uncertainties, fit_array, eigenvectors, params_linear
#     ):
#         self.test_clase = test_clase
#         self.masked_uncertainties = masked_uncertainties
#         self.fit_array = fit_array
#         self.eigenvectors = eigenvectors
#         self.params_linear = params_linear
#         self.combination = self.eigenvectors.T * 100 * self.params_linear.T
#         self.negatives_per_column = jnp.nansum(self.combination < 0, axis=0).T

#     def plot(self, n, save="", **kwargs):
#         # save = False
#         # for n in range(len(dr_filtered)):

#         # Create subplots with shared x-axis
#         # if save and os.path.isfile(f"images/images_pca/{n}.jpg"):
#         # continue
#         fig, (ax1, ax2) = plt.subplots(
#             2, 1, sharex=True, figsize=(35, 15), gridspec_kw={'height_ratios': [2, 1]}
#         )

#         # Set axis labels and their font sizes
#         ax1.set_ylabel("Flux", fontsize=20)
#         # ax1.set_xlabel("Wavelength", fontsize=40)  # Even though ax1 and ax2 share x-axis, you can label ax1
#         ax2.set_xlabel("wavelength A", fontsize=20)
#         ax2.set_ylabel("Normalized Residuals", fontsize=20)

#         # Define the x-axis based on the length of the spectrum
#         x_axis_pix = np.arange(len(self.test_clase.spectra[n, 0, :]))
#         x_limit_pix = x_axis_pix[self.masked_uncertainties[n] != 1e11][[0, -1]]
#         n_pixels = x_axis_pix.shape[0]
#         # Create an array of pixel indices
#         indices = jnp.arange(n_pixels)

#         # Create a boolean mask for indices outside the desired range
#         mask = (indices < x_limit_pix[0]) | (indices > x_limit_pix[1])
#         x_axis = self.test_clase.spectra[n, 0, :]
#         x_limit = [np.nanmin(x_axis), np.nanmax(x_axis)]
#         x_limit = [x_axis[x_limit_pix[0]], x_axis[x_limit_pix[-1]]]
#         obj = self.fit_array[:, 1, :][n]
#         linear_model = linear_combination(self.eigenvectors[n], self.params_linear[n])

#         residual = (self.fit_array[:, 1, :][n] - linear_model) / self.fit_array[:, 2, :][n]
#         model_qso = jnp.nansum(
#             self.eigenvectors[n][10:].T * 100 * self.params_linear[n][10:], axis=1
#         )
#         model_galaxy = jnp.nansum(
#             self.eigenvectors[n][:10].T * 100 * self.params_linear[n][:10], axis=1
#         )
#         linear_model = linear_model.at[mask].set(jnp.nan)
#         model_galaxy = model_galaxy.at[mask].set(jnp.nan)
#         model_qso = model_qso.at[mask].set(jnp.nan)
#         residual = residual.at[mask].set(jnp.nan)
#         maxs = [
#             np.nanmax(obj),
#             np.nanmax(linear_model),
#             np.nanmax(model_qso),
#             np.nanmax(model_galaxy),
#         ]
#         minx = [
#             np.nanmin(obj),
#             np.nanmin(linear_model),
#             np.nanmin(model_qso),
#             np.nanmin(model_galaxy),
#         ]
#         # Compute the model using your linear combination function

#         # Plot the observed object spectrum
#         ax1.plot(x_axis, obj, alpha=1, label=f"object {n}", color='grey')
#         # Plot the model spectrum
#         ax1.plot(x_axis, linear_model, label="model", color='r')
#         # Plot the PCA components
#         ax1.plot(x_axis, model_qso, label="pca_qso")
#         ax1.plot(x_axis, model_galaxy, label="pca_galaxy", color="g")

#         ax1.fill_between(
#             x_axis,
#             0,
#             max(maxs),
#             where=self.masked_uncertainties[n] != 1e11,
#             color="grey",
#             alpha=0.1,
#             zorder=10,
#             label="eigenvalues coverage",
#         )
#         # ax2.fill_between(x_axis, -0.5, 0.5,where=masked_uncertainties[n] != 1e11, color="grey", alpha=0.5,zorder=1, label="eigenvalues coverage")
#         ax1.axhline(0, ls="--", linewidth=5, c="k")
#         # Set the x-axis limits based on the non-masked region
#         ax1.set_xlim(x_limit)
#         ax2.set_xlim(x_limit)
#         ax1.set_ylim(min(minx), max(maxs))
#         # Place the legend for ax1 outside the plot area
#         ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=30)

#         # For ax2, plot the normalized residuals (observed - model)/error
#         ax2.scatter(x_axis, residual, alpha=0.5, zorder=10)
#         ax2.axhline(0, ls="--", linewidth=5, c="k")

#         ax2.text(
#             0.05,
#             0.95,
#             rf'{(sum(jnp.where(abs(residual)<=0.4,True,False))/sum(~jnp.isnan(residual)))*100:.3f} % between abs(0.4)',
#             transform=ax2.transAxes,
#             fontsize=30,
#             verticalalignment='top',
#         )
#         ax2.set_ylim(-0.4, 0.4)
#         plt.tight_layout(rect=[0, 0, 0.85, 1])
#         # Save or show figure
#         if save:
#             plt.savefig(f"images/{save}.jpg", dpi=300, bbox_inches='tight')
#             plt.close()
#         else:
#             plt.show()

#     def plot_valeu(self, n):
#         plt.plot(self.params_linear[n])
#         plt.axhline(0)
#         plt.axvline(10)
#         plt.axvspan(0, 10, alpha=0.2, color="r", label="galaxy linear paramters")
#         plt.axvspan(10, 60, alpha=0.2, label="qso linear paramters")
#         plt.xlim(0, 59)
#         plt.ylabel("parameter valeu")
#         plt.xlabel("parameter number")
#         plt.legend()
#         plt.show()

#     def plot_n_negatives(self, n):
#         # combination = self.eigenvectors[n].T*100*self.params_linear[n]
#         # negatives_per_column = jnp.nansum(combination < 0, axis=0)
#         plt.plot(self.negatives_per_column[n])
#         plt.axhline(0, ls="--", alpha=0.5)
#         plt.axvline(10)
#         plt.axvspan(0, 10, alpha=0.2, color="r", label="galaxy linear paramters")
#         plt.axvspan(10, 60, alpha=0.2, label="qso linear paramters")
#         plt.xlim(0, 59)
#         plt.ylabel("number of negatives by parameter x eigvector")
#         plt.xlabel("parameter number")
#         plt.legend()
#         plt.show()

#     def sep_componentes(self, n):
#         # combination = eigenvectors[0].T * 100 * params_linear[0]
#         fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20, 10))
#         combination = self.combination[:, :, n]
#         for i, spec in enumerate(combination.T):
#             if i < 10:
#                 # Plot on the left subplot
#                 ax1.plot(spec, c="r", alpha=0.2)
#             else:
#                 # Plot on the right subplot
#                 ax2.plot(spec, c="b", alpha=0.1)

#         # (Optional) adjust x-limits or other axes properties if you wish:
#         ax1.set_xlim(left=0, right=combination.shape[0])  # Or omit for auto-limits
#         ax2.set_xlim(left=0, right=combination.shape[0])  # Or omit for auto-limits

#         ax1.set_title("galaxy(10)")
#         ax2.set_title("qso(50)")

#         plt.tight_layout()
#         plt.show()

#     # if save:
#     #     plt.savefig(f"images/images_pca/{n}.jpg", dpi=300, bbox_inches='tight')
#     #     plt.close()
#     # else:
#     #     plt.show()
#     # brea


def plot_region(x, function, region, save=''):
    fig, ax1 = plt.subplots(1, 1, figsize=(35, 15))
    ax1.plot(x, function)
    min_y, max_y = ax1.get_ylim()

    # Dictionary to keep track of how many times a line_name has been plotted
    text_offsets = {}
    for i, line in enumerate(region):
        # Unpack line details assuming the dictionary has these keys
        center = line.get("center")
        kind = line.get("kind")
        amplitude = line.get("amplitude")
        line_name = line.get("line_name")

        # Plot vertical line with a dashed style
        ax1.axvline(center, linestyle="--", color="red", linewidth=2, alpha=0.5)

        # Compute offset for the text label if the same line_name is already plotted
        if line_name in text_offsets:
            offset = text_offsets[line_name]
            text_offsets[line_name] += 2
        else:
            offset = 0
            text_offsets[line_name] = 2
        n = 1
        if "h" in line_name:
            n = 6
        # Adjust text position: for each duplicate, move the text downwards by 5% of max_y.
        text_y = max_y / n - (offset * 0.05 * max_y)
        ax1.text(
            center,
            text_y,
            f" {line_name}\n {kind}",
            fontsize=16,
            rotation=0,
            verticalalignment="top",
            color="k",
            zorder=10,
            horizontalalignment="left",
        )

    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(0.0, max_y + max_y / 10)

    # Add labels to x and y axes with larger fonts
    ax1.set_xlabel("rest-wavelength", fontsize=20)
    ax1.set_ylabel("Flux", fontsize=20)
    if save:
        plt.savefig(f"{save}.jpg")
    plt.show()


class SheapPlot_old:
    """This aim to be the main class to plot the results from sheap."""

    def __init__(
        self,
        test_clase,
        fit_region_g,
        mask_fit,
        mask_fit_g,
        masked_uncertainties_g,
        Master_Gaussian,
        params_g,
        Baselines,
        outer_limits,
        AN,
        EWfin,
        signal_noise_region,
        host_detected,
        host_flux,
    ):
        self.test_clase = test_clase
        self.fit_region_g = fit_region_g
        self.mask_fit = mask_fit
        self.masked_uncertainties_g = masked_uncertainties_g
        self.Master_Gaussian = Master_Gaussian
        self.params_g = params_g
        self.Baselines = Baselines
        self.outer_limits = outer_limits
        self.AN = AN
        self.EWfin = EWfin
        self.signal_noise_region = signal_noise_region
        self.host_detected = host_detected  # index
        self.mask_fit_g = mask_fit_g
        self.host_flux = host_flux

    def plot_combined(self, n, save='', add_baseline=True, pandas=None, **kwargs):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(35, 15), gridspec_kw={'height_ratios': [2, 1]}
        )

        # Load necessary data
        x_axis = self.test_clase.spectra[n, 0, :]  # Already redshift corrected
        y_spectrum = self.test_clase.spectra[n, 1, :]
        y_err = self.test_clase.spectra[n, 2, :]
        colors = plt.cm.Pastel1(np.linspace(0, 1, 9))
        if not jnp.all(self.host_flux == 0):
            add_all = kwargs.get("add_all", False)
            spectros = {}
            if add_all:
                for i, key in enumerate(
                    ['Spectra redshift corrected', "host", "AGN(spectra-host)"]
                ):
                    if key == 'Spectra redshift corrected':
                        y = y_spectrum
                        yerr = y_err
                    elif key == "host":
                        y = self.host_flux[n]
                        yerr = None
                    elif key == 'AGN(spectra-host)':
                        y = y_spectrum - self.host_flux[n]
                        yerr = None
                    ax1.errorbar(
                        x_axis,
                        y,
                        yerr=yerr,
                        ecolor='lightskyblue',
                        label=key,
                        zorder=3,
                        alpha=0.8,
                    )

            else:
                ax1.errorbar(
                    x_axis,
                    y_spectrum,
                    yerr=y_err,
                    color='b',
                    ecolor='lightskyblue',
                    label='Spectra redshift corrected',
                    zorder=1,
                )

        ax1.set_ylabel(
            r'$ f_{\lambda}$ ($\rm 10^{-17} {\rm erg\;s^{-1}\;cm^{-2}\;\AA^{-1}}$)',
            fontsize=40,
        )
        ax1.set_xlabel(r'$\rm Rest \ Wavelength$ ($\rm \AA$)', fontsize=40)
        ax1.tick_params(which="both", length=10, width=2, labelsize=35)

        # Set axis limits
        ax1.set_xlim(np.nanmin(x_axis), np.nanmax(x_axis))
        if "xlim_sheap" in kwargs:
            ax1.set_xlim(*kwargs["xlim_sheap"])
        if "ylim_sheap" in kwargs:
            ax1.set_ylim(*kwargs["ylim_sheap"])

        ylimit = ax1.get_ylim()

        # Apply masking efficiently
        if "mask" in kwargs:
            mask = kwargs["mask"]
            mask_x = np.logical_and(x_axis >= min(mask), x_axis <= max(mask))
            ax1.fill_between(
                x_axis, *ylimit, where=mask_x, color='grey', alpha=0.5, label='Mask', zorder=1
            )

        ax1.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1),
            fancybox=True,
            shadow=False,
            ncol=4,
            fontsize=30,
        )
        ax1.set_ylim(*ylimit)

        # Plot Local Spectra
        fit_x = self.fit_region_g[n][0][~self.mask_fit[n]]
        fit_y = self.fit_region_g[n][1][~self.mask_fit[n]]
        fit_err = self.masked_uncertainties_g[n][~self.mask_fit[n]]

        ax2.plot(x_axis, y_spectrum, color='red', label='Spectra')
        ax2.errorbar(
            fit_x,
            fit_y,
            yerr=fit_err,
            fmt='o',
            color='red',
            alpha=0.5,
            label='Fit Region with Uncertainties',
        )

        baseline_plus_gaussian = self.Master_Gaussian.func(x_axis, self.params_g[n])
        if add_baseline:
            baseline_plus_gaussian += self.Baselines[n]
        ax2.plot(x_axis, baseline_plus_gaussian, label='Baseline + Gaussian Fit')

        ax2.fill_between(
            x_axis,
            0,
            np.nanmax(y_spectrum),
            where=self.mask_fit[n],
            color='grey',
            alpha=0.5,
            label='Mask for Linear Fit',
            zorder=10,
        )
        ax2.fill_between(
            x_axis,
            0,
            np.nanmax(y_spectrum),
            where=self.mask_fit_g[n],
            color='green',
            alpha=0.5,
            label='Mask for Gaussian Fit',
            zorder=10,
        )

        ax2.set_xlim(self.outer_limits)
        median_val = np.median(fit_y)
        if "xlim_local" in kwargs:
            arg_min = np.nanargmin(abs(x_axis - min(kwargs["xlim_local"])))
            arg_max = np.nanargmin(abs(x_axis - max(kwargs["xlim_local"])))
            median_val = np.nanmedian(y_spectrum[arg_min:arg_max])
            ax2.set_xlim(*kwargs["xlim_local"])

        if np.isnan(median_val):
            median_val = np.nanmedian(y_spectrum)

        ax2.set_ylim([median_val * 0.1, median_val * 1.9])
        if "ylim_local" in kwargs:
            ax2.set_ylim(*kwargs["ylim_local"])
        if isinstance(pandas, pd.DataFrame):
            text_string = ""
            for k in [
                "EWfin",
                "vel",
                "host_detected",
                "AN",
                "signal_noise_region",
                "rAGN_stars_5100",
                "rAGN_Cont_5100",
                "stars_Cont_5100",
                "agn_slope",
            ]:
                if k == "host_detected":
                    text_string += f"{k} :{bool(pandas.iloc[n][k])}\n"
                else:
                    text_string += f"{k} :{pandas.iloc[n][k]:.2f}\n"
            # = pandas.iloc[n][["EWfin","rAGN_Cont_5100","vel","index","AN","rAGN_stars_5100"]].values
        # matched_df
        else:
            text_string = (
                f"AN: {self.AN[n]:.2f}\n"
                f"EWfin: {self.EWfin[n]:.2f}\n"
                f"vel: {self.vel[n]:.2f}\n"
                f"Signal noise region: {self.signal_noise_region[n]:.2f}\n"
                f"is the host detected?: {n in self.host_detected}"
            )

        fig.text(
            0.71,
            0.7,
            text_string,
            fontsize=20,
            color='blue',
            verticalalignment='center',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
        )

        ax2.set_title(f"Local Spectra for n={n}", fontsize=20)
        ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.0, fontsize=15)

        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Cleanup large arrays
        del baseline_plus_gaussian, fit_x, fit_y, fit_err, y_spectrum, y_err

        # Save or show figure
        if save:
            plt.savefig(f"images/{save}.jpg", dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def plot_a_spectra(spectra, save=None, **kwargs):
    # Unpack the spectra data
    x_axis, y_axis, yerr = spectra
    # Determine x-axis limits (either passed in or computed from data)
    xlim = kwargs.get("xlim", [np.nanmin(x_axis), np.nanmax(x_axis)])

    # Optional object name to annotate; default if not provided
    object_name = kwargs.get("object_name", "Unknown Object")

    # Define spectral regions with wavelength ranges.
    regions = {
        'La': [1000, 1500],
        'CIV,cIII': [1100, 2000],
        'MgII': [2500, 3000],
        'Hbeta': [4400, 5600],
        'Halpha': [5600, 7300],
    }

    # Filter regions that are completely inside the main x-axis limits.
    regions_to_plot = []
    for region, (start, end) in regions.items():
        if xlim[0] <= start and xlim[1] >= end:
            regions_to_plot.append((region, start, end))
    n_regions = len(regions_to_plot)
    # print(n_regions)
    # If we have regions to show, create a two-row layout: top for regions, bottom for main plot.
    if n_regions > 0:
        # Create a figure with overall size (20,10)
        fig = plt.figure(figsize=(20, 10))
        # Define a GridSpec with 2 rows:
        # - top row: one column per region (each region gets its own subplot)
        # - bottom row: main plot spans all columns.
        # Adjust height ratios to give more space to the main plot.
        gs = gridspec.GridSpec(2, n_regions, height_ratios=[1, 2], hspace=0.1)

        # Create subplots for each region in the top row.
        for i, (region, start, end) in enumerate(regions_to_plot):
            ax = fig.add_subplot(gs[0, i])
            # Extract data in this spectral region.
            mask = (x_axis >= start) & (x_axis <= end)
            if np.any(mask):
                if yerr is not None:
                    ax.errorbar(
                        x_axis[mask], y_axis[mask], yerr=yerr[mask], fmt='-', lw=1, c="k"
                    )
                else:
                    ax.plot(x_axis[mask], y_axis[mask], '-', lw=1, c="k")
                # Optionally adjust y-limits based on the data within the region.
                ax.set_ylim(np.nanmin(y_axis[mask]), np.nanmax(y_axis[mask]) * 1.1)
            ax.set_xlim(start, end)
            # Add annotation text in the top left corner of each region subplot.
            annotation = f"Region: {region}"
            ax.text(
                0.05,
                0.95,
                annotation,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
            )
            ax.set_xlabel("Wavelength", fontsize=10)
            ax.set_ylabel("Flux", fontsize=10)
            ax.tick_params(axis='both', labelsize=8)

        # Create the main plot on the bottom row spanning all top-row columns.
        ax_main = fig.add_subplot(gs[1, :])
    else:
        # If no regions qualify, just create a single axis.
        fig, ax_main = plt.subplots(figsize=(20, 10))

    # Plot the full spectrum on the main axis.
    # (x_axis,y_axis,yerr=yerr,ecolor='dimgray',c="k",zorder=1)
    if yerr is not None:
        ax_main.errorbar(x_axis, y_axis, yerr=yerr, c="k", fmt='-', ecolor='dimgray', lw=1)
    else:
        ax_main.plot(x_axis, y_axis, c="k", fmt='-', lw=1)
    ax_main.set_xlim(xlim)
    ax_main.set_xlabel('Wavelength', fontsize=12)
    ax_main.set_ylabel('Flux', fontsize=12)
    # Add annotation text in the top right corner of the main plot.
    ax_main.text(
        0.15,
        0.95,
        f"Object: {object_name}",
        transform=ax_main.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'),
    )

    # ax_main.legend(fontsize=10)
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

