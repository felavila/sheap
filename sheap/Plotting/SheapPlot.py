import numpy as np 
import pandas as pd 
import jax.numpy as jnp 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms 

class SheapPlot:
    def __init__(self,ComplexRegion):
        """_summary_

        Args:
            ComplexRegion (_type_): _description_
        """
        #self.region_class = ComplexRegion
        
        self.spec = ComplexRegion.spec #new version
        self.max_flux = ComplexRegion.max_flux
        self.params = ComplexRegion.params
        self.profile_params_index = ComplexRegion.profile_params_index
        self.model = ComplexRegion.model
        self.profile_functions = ComplexRegion.profile_functions
        self.profile_names = ComplexRegion.profile_names
        self.region_defs = ComplexRegion.region_defs #this is the entire region 
        #self.pandas_r = pd.DataFrame(params,columns=list(region_class.params_dict.keys()))
        
    def plot(self,n,save=None,add_name=False,residual=True,**kwargs):
        default_colors = list(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        filtered_default_colors = [color for color in default_colors if color not in ['black', 'red',"grey","#7f7f7f"]]*50
        ylim = kwargs.get("ylim",[0,self.max_flux[n]])
        verbose = kwargs.get("verbose",True)
        x_axis,y_axis,yerr = self.spec[n,:]
        xlim =  kwargs.get("xlim",[jnp.nanmin(x_axis),jnp.nanmax(x_axis)])
        params = self.params[n]
        profile_index_list = self.profile_params_index
        profile_functions = self.profile_functions
        profile_names = self.profile_names
        region_defs = self.region_defs 
        fit_y =  self.model(x_axis,params)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(35, 15),gridspec_kw={'height_ratios': [2, 1]})
        if not residual:
            plt.close()
            fig, ax1 = plt.subplots(1, 1, sharex=True, figsize=(35, 15))
        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        for i,profile in enumerate(profile_names):
            #print(region_defs[i].region)
            min_,max_ = profile_index_list[i]
            profile_function = profile_functions[i]
            values_ = params[min_:max_] #mmm i am taking alot of adbantage of the "gaussianity of the equation"
            local_y = profile_function(x_axis,values_)
            c = "k"
            #ax1.text(values[1],max(ylim),text,verticalalignment="bottom",horizontalalignment="center")
            if profile != "linear" and "Fe" not in profile:
                if "fe" in region_defs[i].region.lower():
                    color = "grey"
                    c =  "grey"
                else:
                    color = filtered_default_colors[i]
                    ax1.axvline(values_[1],ls="--",linewidth=1,c=c)
                    #ylim = ax.get_ylim() 
                    if add_name and max(xlim)>values_[1]>min(xlim):
                        name_obj = region_defs[i].line_name +"_"+ region_defs[i].kind +"_"+ str(region_defs[i].component)
                        if "broad" in name_obj:
                            ax1.text(values_[1],0.25,name_obj.replace("_"," "),transform=trans,rotation=90, fontsize=20,zorder=10)
                        else:
                            ax1.text(values_[1],0.75,name_obj.replace("_"," "),transform=trans,rotation=90, fontsize=20,zorder=10)
                    #print(self.region_class.lines[i],color)
                ax1.plot(x_axis, local_y,zorder=3,ls='-.',color=color)
            elif "Fe" in profile:
                ax1.plot(x_axis, local_y,zorder=3,ls='-.',color="grey")
            
            elif profile == "linear":
                #ax1.axvline(values[1],ls="--",linewidth=1,c=c)
                ax1.plot(x_axis, local_y,zorder=3,ls='-.',color=filtered_default_colors[i])
            ax1.set_ylabel("Flux", fontsize=20)
        if not residual:
            ax1.plot(x_axis,fit_y,linewidth=3,zorder=2,ls="--",c="r")
            ax1.errorbar(x_axis,y_axis,yerr=yerr,ecolor='dimgray',c="k",zorder=1)
            #ax1.fill_between(x_axis, *ylim,where= yerr >= 1e11, color="grey", alpha=0.9,zorder=10)
            ax1.set_ylim(ylim)
            ax1.set_xlim(xlim)
            ax1.fill_between(x_axis, *ylim,where= yerr != 1e11, color="grey", alpha=0.1,zorder=10)
            ax1.text(ax1.get_xlim()[0],ax1.get_ylim()[1]*1.1,f"Obj {n}",fontsize=30)
            ax1.set_xlabel("Wavelength", fontsize=40)
            if save:
                plt.savefig(save, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        ax2.set_xlabel("Wavelength", fontsize=40)  # Even though ax1 and ax2 share x-axis, you can label ax1
        ax1.plot(x_axis,fit_y,linewidth=3,zorder=2,ls="--",c="r")
        ax1.errorbar(x_axis,y_axis,yerr=yerr,ecolor='dimgray',c="k",zorder=1)
        #ax1.fill_between(x_axis, *ylim,where= yerr >= 1e11, color="grey", alpha=0.9,zorder=10)
        residual = (fit_y - y_axis) / (yerr)
        ax2.axhline(0,ls="--",linewidth=5,c="k")
        ax2.scatter(x_axis, residual,alpha=0.9,zorder=10)
        #ax2.set_xlabel("wavelength A", fontsize=20)
        ax2.set_ylabel("Normalized Residuals", fontsize=20)
        ax1.set_ylim(ylim)
        ax1.set_xlim(xlim)
        ax1.text(ax1.get_xlim()[0],ax1.get_ylim()[1]*1.1,f"Obj {n}",fontsize=30)
        ax1.tick_params(axis='both', labelsize=20)
        ax2.tick_params(axis='both', labelsize=20)
        if save:
            plt.savefig(save, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
 