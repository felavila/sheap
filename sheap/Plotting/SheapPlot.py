"""This module handles ?."""

__author__ = 'felavila'
__all__ = [
    "SheapPlot",
]

from typing import Optional, List, Any
from dataclasses import dataclass
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from jax import jit
 
from sheap.Profiles.utils import make_fused_profiles



class SheapPlot:
    def __init__(
        self,
        sheap: Optional["Sheapectral"] = None,
        fit_result: Optional["FitResult"] = None,
        spectra: Optional[jnp.ndarray] = None,
    ):
        """
        Initialize SheapPlot using:
          - a full Sheapectral object (preferred), or
          - a FitResult + spectra.
        """
        if sheap is not None:
            self._from_sheap(sheap)
        elif fit_result is not None and spectra is not None:
            self._from_fit_result(fit_result, spectra)
        else:
            raise ValueError("Provide either `sheap` or (`fit_result` + `spectra`).")

    def _from_sheap(self, sheap):
        self.spec = sheap.spectra
        #self.max_flux = sheap.max_flux
        self.result = sheap.result  # keep reference if needed

        result = sheap.result  # for convenience

        self.params = result.params
        self.scale = result.scale
        self.uncertainty_params = result.uncertainty_params
        self.profile_params_index_list = result.profile_params_index_list
        self.profile_functions = result.profile_functions
        self.profile_names = result.profile_names
        self.complex_region = result.complex_region
        self.xlim = result.outer_limits
        self.mask = result.mask
        self.names = sheap.names
        self.model_keywords = result.model_keywords or {}
        self.z = sheap.z
        #self.fe_mode = self.model_keywords.get("fe_mode")
        self.model = jit(make_fused_profiles(self.profile_functions))
        

    def _from_fit_result(self, result, spectra):
        self.spec = spectra
        self.scale = jnp.nanmax(spectra[:, 1, :], axis=1)
        self.params = result.params
        self.uncertainty_params = result.uncertainty_params
        self.profile_params_index_list = result.profile_params_index_list
        self.profile_functions = result.profile_functions
        self.profile_names = result.profile_names
        self.complex_region = result.complex_region
        self.xlim = result.outer_limits
        self.mask = result.mask
        self.names = [str(i) for i in range(self.params.shape[0])]
        self.model_keywords = result.model_keywords or {}
        self.z = result.z
        #self.fe_mode = self.model_keywords.get("fe_mode")
        self.model = jit(make_fused_profiles(self.profile_functions))

    def plot(self, n, save=None, add_lines_name=False, residual=True,params=None,line=None,flux_unit=r"$10^{-17}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$", **kwargs):
        """Plot spectrum, model components, and residuals for a given index `n`."""
        # Setup and defaults
        default_colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        filtered_colors = [
            c for c in default_colors if c not in ['black', 'red', 'grey', '#7f7f7f',"blue","green"]
        ] * 50

        ylim = kwargs.get("ylim", [0,self.scale[n]])
        xlim = kwargs.get("xlim", self.xlim)

        x_axis, y_axis, yerr = self.spec[n, :]

        params = params if params is not None else self.params[n]
        mask = self.mask[n]
        fit_y = self.model(x_axis, params)

        if residual:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(35, 15),
                gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1},
            )
        else:
            fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(35, 15))

        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        
        for i, (profile_name, profile_func, region, idxs) in enumerate(zip(self.profile_names,self.profile_functions,self.complex_region,self.profile_params_index_list,)
        ):
            #print(profile_name, profile_func, region, idxs)
            values = params[idxs]
            component_y = profile_func(x_axis, values)

            if region.region == "continuum":
                ax1.plot(x_axis, component_y, ls='-.', zorder=3, color="blue",label="Continuum")
            elif "Fe" in profile_name or "fe" in region.region.lower() or region.region == "fe":
                ax1.plot(x_axis, component_y, ls='-.', zorder=3, color="grey",label="Fe")
            elif "host" in region.region.lower():
                ax1.plot(x_axis, component_y, ls='-.', zorder=3, color="green",label="Host")
            else:
                ax1.plot(x_axis, component_y, ls='-.', zorder=3, color=filtered_colors[i],label=region.region.capitalize())
                ax1.axvline(values[1], ls="--", linewidth=1, color="k")
                if add_lines_name and isinstance(region.region_lines,list):
                    import numpy as np 
                    idx_shift = [n for n,i in enumerate(profile_func.param_names) if "shift" in i ]
                    centers = np.array(region.center) + params[*idx_shift]#This is only true for gaussian
                    #print(centers)
                    for ii,c in enumerate(centers):
                        #ax1.axvline(c)
                        if min(xlim) < c < max(xlim):
                            label = f"- {region.region_lines[ii]}_{region.region}_{region.component}".replace("_", " ")
                            ypos = 0.25 if "broad" in label else 0.75
                            ax1.text(
                            c,
                            ypos,
                            label,
                            transform=trans,
                            rotation=90,
                            fontsize=20,
                            zorder=10,
                            ha = "center"
                        )
                elif add_lines_name and min(xlim) < values[1] < max(xlim):
                    label = f"- {region.line_name}_{region.region}_{region.component}".replace(
                        "_", " "
                    )
                    ypos = 0.25 if "broad" in label else 0.75
                    ax1.text(
                        values[1],
                        ypos,
                        label,
                        transform=trans,
                        rotation=90,
                        fontsize=20,
                        zorder=10,
                        ha = "center"
                    )

        ax1.plot(x_axis, fit_y, linewidth=3, zorder=2, color="red",label="Complex")#
        ax1.errorbar(x_axis, y_axis, yerr=yerr, ecolor='dimgray', color="black", zorder=1,label="Data")
        ax1.fill_between(x_axis, *ylim, where=mask, color="grey", alpha=0.3, zorder=10)
        if line:
            ax1.axhline(line)
        ax1.set_ylabel(f"Flux [{flux_unit}]", fontsize=35)
        ax1.set_ylim(ylim)
        ax1.set_xlim(xlim)
        ax1.text(
            0.75,
            1.05,
            f"ID {self.names[n]} ({n}) \n z = {self.z[n]}",
            fontsize=35,
            transform=ax1.transAxes,
            ha='left',
            va='bottom',
        )
        #font_legend =
        ax1.tick_params(axis='both', labelsize=35)
        ax1.yaxis.offsetText.set_fontsize(35)
        ax1.legend(fontsize=30, markerscale=0.8, labelspacing=0.5,frameon=False)
        if residual:
            residuals = (fit_y - y_axis) / yerr
            residuals = residuals.at[mask].set(0.0)
            ax2.axhline(0, ls="--", linewidth=5, color="black")
            ax2.scatter(x_axis, residuals, alpha=0.9, zorder=10,c="black")
            ax2.set_ylabel("Normalized \n residuals", fontsize=35)
            ax2.set_xlabel("Wavelength [Å] (rest frame)", fontsize=35)
            ax2.tick_params(axis='both', labelsize=35, pad=10)
        else:
            ax1.set_xlabel("Wavelength [Å] (rest frame)", fontsize=35)

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

