import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from jax import jit
from sheap.FunctionsMinimize.utils import combine_auto
#add constrains in the for example only plot broad,fe,thinks like that 

class SheapPlot:
    def __init__(self, ComplexRegion):
        """Initialize SheapPlot with a ComplexRegion object."""
        
        try:
            self.spec = ComplexRegion.spec  # Newer version
        except AttributeError:
            self.spec = ComplexRegion.spectra
        try:
            self.max_flux = ComplexRegion.max_flux
        except AttributeError:
            self.max_flux = jnp.nanmax(self.spec[:, 1, :], axis=1)

        self.params = ComplexRegion.params
        self.profile_params_index_list = ComplexRegion.profile_params_index_list
        self.profile_functions = ComplexRegion.profile_functions
        self.profile_names = ComplexRegion.profile_names
        self.complex_region = ComplexRegion.complex_region
        self.xlim = ComplexRegion.outer_limits
        self.mask = ComplexRegion.mask
        self.names = ComplexRegion.names
        self.model_keywords = ComplexRegion.model_keywords
        self.fe_mode = self.model_keywords.get("fe_mode")
        try:
            self.model = ComplexRegion.model
        except AttributeError:
            self.model = jit(combine_auto(self.profile_functions))

    def plot(self, n, save=None, add_name=False, residual=True, **kwargs):
        """Plot spectrum, model components, and residuals for a given index `n`."""
        # Setup and defaults
        default_colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        filtered_colors = [c for c in default_colors if c not in ['black', 'red', 'grey', '#7f7f7f']] * 50

        ylim = kwargs.get("ylim", [0, self.max_flux[n]])
        xlim = kwargs.get("xlim", self.xlim)

        x_axis, y_axis, yerr = self.spec[n, :]
        params = self.params[n]
        mask = self.mask[n]
        fit_y = self.model(x_axis, params)

        if residual:
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(35, 15),gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.05})
        else:
            fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(35, 15))

        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)

        # Plot model components
        for i, (profile_name, profile_func, region, idxs) in enumerate(zip(
            self.profile_names, self.profile_functions, self.complex_region, self.profile_params_index_list)):

            #if self.fe_mode=="model" and isinstance(region,list):
             #   continue
           
            values = params[idxs]
            component_y = profile_func(x_axis, values)
            if isinstance(region,list):
                ax1.plot(x_axis, component_y, ls='-.', zorder=3, color="grey")
            else:
                if region.region == "continuum" :
                    ax1.plot(x_axis, component_y, ls='-.', zorder=3, color=filtered_colors[i])
                elif "Fe" in profile_name or "fe" in region.region.lower():
                    ax1.plot(x_axis, component_y, ls='-.', zorder=3, color="grey")
                else:
                    ax1.plot(x_axis, component_y, ls='-.', zorder=3, color=filtered_colors[i])
                    ax1.axvline(values[1], ls="--", linewidth=1, color="k")

                    if add_name and min(xlim) < values[1] < max(xlim):
                        label = f"{region.line_name}_{region.kind}_{region.component}".replace("_", " ")
                        ypos = 0.25 if "broad" in label else 0.75
                        ax1.text(values[1], ypos, label, transform=trans, rotation=90, fontsize=20, zorder=10)

        # Plot main model and data
        ax1.plot(x_axis, fit_y, linewidth=3, zorder=2, ls="--", color="red")
        ax1.errorbar(x_axis, y_axis, yerr=yerr, ecolor='dimgray', color="black", zorder=1)
        ax1.fill_between(x_axis, *ylim, where=mask, color="grey", alpha=0.3, zorder=10)

        ax1.set_ylabel("Flux [arb]", fontsize=20)
        ax1.set_ylim(ylim)
        ax1.set_xlim(xlim)
        ax1.text(0.0, 1.05, f"ID {self.names[n]} ({n})", fontsize=20, transform=ax1.transAxes,
                ha='left', va='bottom')
        ax1.tick_params(axis='both', labelsize=20)
        ax1.yaxis.offsetText.set_fontsize(20)
        if residual:
            residuals = (fit_y - y_axis) / yerr
            residuals = residuals.at[mask].set(0.0)
            ax2.axhline(0, ls="--", linewidth=5, color="black")
            ax2.scatter(x_axis, residuals, alpha=0.9, zorder=10)
            ax2.set_ylabel("Normalized Residuals", fontsize=20)
            ax2.set_xlabel("Wavelength [Å]", fontsize=30)
            ax2.tick_params(axis='both', labelsize=20)
        else:
            ax1.set_xlabel("Wavelength", fontsize=30)

        # Save or display
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
