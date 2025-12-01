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
 
from sheap.Profiles.Utils import make_fused_profiles



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

    def plot(self, n, save=None, add_lines_name=False, residual=True,params=None,add_xline=None,
             flux_unit=r"$\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$",add_legend=True, **kwargs):
        """Plot spectrum, model components, and residuals for a given index `n`."""
        # TODO is time to update this. 
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
                figsize=(30, 8),
                gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.1},
            )
        else:
            fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(35, 15))

        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        
        colors_by_region = {"model":"#d62728","broad":"#0f6fb4","narrow":"#559e46","outflow":"#bcbd22","winds":"#17becf","fe":"#8f220c","host":"#9467bd",
                            "continuum":"#000000","data":"#1B1B1B","balmer":"#2C2424","bal":"#803939"}
        component_ls = {1: "-",2: "--",3: "-.",4: ":", 5: (0, (5, 5)), 6: (0, (3, 5, 1, 5)), 7: (0, (1, 5))}
        cont_counter = 1 
        cont_names = {"balmercontinuum":"Balmer Cont.","balmerhighorder":"Higher-order Balmer"}
        for i, (profile_name, profile_func, region, idxs) in enumerate(zip(self.profile_names,self.profile_functions,self.complex_region,self.profile_params_index_list,)):
            #print(profile_name, profile_func, region, idxs)
            values = params[idxs]
            #print(profile_name)
            
            if region.region == "continuum" or region.region=="balmer":
                #print(region.line_name)
                component_y = profile_func(x_axis, values)
                line_name = cont_names.get(region.line_name,"Cont.")
                #region.line_name
                #print(region)
                ax1.plot(x_axis, component_y, zorder=3, label = line_name, color= colors_by_region["continuum"],ls = component_ls[cont_counter])
                cont_counter += 1
            
            elif "Fe" in profile_name or "fe" in region.region.lower() or region.region == "fe":
                component_y = profile_func(x_axis, values)
                ax1.plot(x_axis, component_y, ls=component_ls[1], zorder=3, color=colors_by_region[region.region.lower()],label="Fe II", linewidth=3)
            
            elif "host" in region.region.lower():
                f = 1.0/0.6028481012658228
                component_y = profile_func(x_axis, values)
                ax1.plot(x_axis, component_y, ls=component_ls[1], zorder=3, color=colors_by_region["host"],label="Host", linewidth=3)
            else:
                component_y = profile_func(x_axis, values)
                label = region.region.capitalize()
                #print(region.component)
                zorder = 0
                if region.component>1:
                    label = f"{region.region.capitalize()} {region.component}"
                if "broad" == region.region:
                    zorder = 10
                ax1.plot(x_axis, component_y, ls=component_ls[region.component], zorder=zorder, color=colors_by_region[region.region], label=label, linewidth=3)
                #ax1.axvline(values[1], ls="--", linewidth=1, color="k")
                if add_lines_name and isinstance(region.region_lines,list):
                    import numpy as np 
                    idx_shift = np.where("vshift_kms" == np.array(profile_func.param_names))[0]
                    #print(idx_shift)
                    C_KMS = 299_792.458
                    centers = np.array(region.center) *(1+values[*idx_shift]/C_KMS)#This is only true for gaussian
                    
                    for ii,c in enumerate(centers):
                        #ax1.axvline(c)
                        if min(xlim) < c < max(xlim):
                            #print(f"- {region.region_lines[ii]}_{region.region}_{region.component}".replace("_", " "),c)
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
                            ha = "center")
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

        ax1.plot(x_axis, fit_y, linewidth=3, zorder=2, color=colors_by_region["model"],label="Model")#
        ax1.errorbar(x_axis, y_axis, yerr=yerr, ecolor='dimgray', color=colors_by_region["data"], zorder=1,label="Obs.")
        ax1.fill_between(x_axis, *ylim, where=mask, color="grey", alpha=0.3, zorder=10)
        if isinstance(add_xline,(float,int)):
            ax1.axvline(add_xline,c='#A020F0',linewidth=3)
        ax1.set_ylabel(f"Flux [{flux_unit}]", fontsize=25)
        ax1.set_ylim(ylim)
        ax1.set_xlim(xlim)
        ax1.text(
            0.75,
            1.0,
            f"ID {self.names[n]} ({n}) \n z = {self.z[n]}",
            fontsize=25,
            transform=ax1.transAxes,
            ha='left',
            va='bottom',
        )
        #font_legend =
        ax1.tick_params(axis='both', labelsize=25)
        ax1.yaxis.offsetText.set_fontsize(25)
        if add_legend:
            handles, labels = ax1.get_legend_handles_labels()
            # Remove duplicates while keeping order
            unique = {}
            for h, l in zip(handles, labels):
                if l not in unique:
                    unique[l] = h

            ax1.legend(handles=list(unique.values()),labels=list(unique.keys()),fontsize=25,
                markerscale=0.8,labelspacing=0.5,frameon=False,ncol=3,columnspacing=1.5,handletextpad=0.4,)
        if residual:
            residuals = (fit_y - y_axis) / yerr
            residuals = residuals.at[mask].set(0.0)
            ax2.axhline(0, ls="--", linewidth=5, color="black")
            ax2.scatter(x_axis, residuals, alpha=0.9, zorder=10,c="#4C72B0")
            ax2.set_ylabel("Norm. Res.", fontsize=25)
            ax2.set_xlabel("Rest wavelength [Å]", fontsize=25)
            ax2.tick_params(axis='both', labelsize=25, pad=10)
        else:
            ax1.set_xlabel("Rest wavelength [Å]", fontsize=25)

        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            #plt.close()
        else:
            plt.show()

