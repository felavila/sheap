from pathlib import Path
from typing import Optional, Tuple
import time

from astropy.io import fits 
import jax.numpy as jnp
from jax import vmap,jit
import matplotlib.pyplot as plt 

from sheap.utils import mask_builder, prepare_spectra
from sheap.tools.others import vac_to_air
from sheap.FunctionsMinimize.MasterMinimizer import MasterMinimizer
from .utils import make_penalty_func,linear_combination

module_dir = Path(__file__).resolve().parent.parent / "SuportData"/ "eigen" 
#print(module_dir)


class HostSubtraction:
    
    def __init__(self,Spectra:jnp.array,num_steps= 1000,learning_rate=1e-1,xmax=7300,xmin=4050,n_galaxies=10,n_qso=50,**kwargs):
        " - Spectra shape (K, N, M)"    
        self.Spectra = Spectra
        self.xmax = xmax
        self.xmin = xmin
        self.n_galaxies = n_galaxies
        self.n_qso = n_qso
        self._setup_eigenvectors()
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.names = kwargs.get("names")
        self.z = kwargs.get("z")
        
    def _setup_eigenvectors(self,n_galaxies=None,n_qso=None,xmin=None,xmax=None): #here add the constrain in the number of objects 
        
        #add limits in the number of n_galaxies and n_qso
        n_galaxies = n_galaxies or self.n_galaxies
        if n_galaxies>10:
            print("n_galaxies cant be bigger than 10")
            n_galaxies = 10
        n_qso = n_qso or self.n_qso
        if n_qso>50:
            print("n_galaxies cant be bigger than 50")
            n_qso = 50
        xmax = xmax or self.xmax
        xmin = xmin or self.xmin
        
        galaxies_eig = fits.open(f"{module_dir}/gal_eigenspec_Yip2004.fits")[1].data 
        qso_eig = fits.open(f"{module_dir}/qso_eigenspec_Yip2004_global.fits")[1].data
        galaxies_eig_wave =  jnp.array(vac_to_air(galaxies_eig["wave"].flatten()))
        qso_eig_wave= jnp.array(vac_to_air(qso_eig["wave"].flatten()))
        galaxies_eig_flux = jnp.array(galaxies_eig["pca"].reshape(galaxies_eig["pca"].shape[1], galaxies_eig["pca"].shape[2]))[:n_galaxies,:]
        qso_eig_flux = jnp.array(qso_eig["pca"].reshape(qso_eig["pca"].shape[1], qso_eig["pca"].shape[2]))[:n_qso,:]
       
        obj_spectra,_,_,_ = mask_builder(self.Spectra,outer_limits=[xmin,xmax])
        
        mask_qso = (obj_spectra[:, 0, :] >=xmin) & (obj_spectra[:, 0, :] <= xmax)
        mask_galaxies = (obj_spectra[:, 0, :] >= xmin) & (obj_spectra[:, 0, :] <= xmax)
        
        qso_eig_interp = interpolate_flux_array(obj_spectra[:, 0, :], qso_eig_wave, qso_eig_flux)
        galaxies_eig_interp = interpolate_flux_array(obj_spectra[:, 0, :], galaxies_eig_wave, galaxies_eig_flux)
        
        
        qso_eig = normalize(jnp.where(mask_qso[:, None, :], jnp.moveaxis(qso_eig_interp, 0, 1), 0.0))
        galaxies_eig = normalize(jnp.where(mask_galaxies[:, None, :], jnp.moveaxis(galaxies_eig_interp, 0, 1), 0.0))
        self.n_galaxies, self.n_qso,self.xmax, self.xmin = n_galaxies,n_qso,xmax,xmin
        self.eigenvectors = jnp.hstack((galaxies_eig,qso_eig))
        self.obj_spectra = obj_spectra
        
    def _run_substraction(self,num_steps= None,learning_rate=1e-1,penalty_function=None,penalty_weight=None
                          ,n_galaxies=None,n_qso=None,xmin=None,xmax=None):
        
        if any(eval(v) is not None or eval(v) != eval(f"self.{v}") for v in ("n_galaxies", "n_qso", "xmin", "xmax")):
            print("set up again")
            self. _setup_eigenvectors(n_galaxies=n_galaxies,n_qso=n_qso,xmin=xmin,xmax=xmax)
        
        eigenvectors = self.eigenvectors
        obj_spectra = self.obj_spectra
        num_steps = num_steps or self.num_steps
        learning_rate = learning_rate or self.learning_rate 
        penalty_weight = 1.0
        penalty_function = make_penalty_func(linear_combination, self.n_galaxies)
        initial_params = jnp.array([1.0]*self.n_galaxies + [1.0]*self.n_qso)
        constraints = jnp.array([[-1.e3, +1.e3]]*self.n_galaxies + [[-1e3, +1e3]]*self.n_qso) 
        start = time.perf_counter()
        minimizer = MasterMinimizer(linear_combination, optimize_in_axis=3,
                                    num_steps=num_steps,learning_rate=learning_rate,
                                    penalty_function = penalty_function,   # <--- Add this
                                    penalty_weight = penalty_weight  )                # <--- Add this)
        params, loss = minimizer(initial_params,eigenvectors,obj_spectra[:,1,:],eigenvectors[:,2,:],constraints)
        end = time.perf_counter()
        self.params = params
        self.time = end - start
        self.num_steps = num_steps
        self.loss = loss
        
    def _make_text(self,n,model_galaxy,):
        text = ""
        text2 = "Object "
        if self.z:
            text += f"Redshift {self.z[n]} \n"
        text += f"\n% negatives values {100*sum(model_galaxy<0)/model_galaxy.shape[0]:.2f} \nnum_steps {self.num_steps} \ntime {self.time:.3f}[s]"
        if self.names:
            text2 += f"self.names[n] {[n]}"
        else:
             text2 += f"{n}"
        return text,text2
        
    def _plot(self,n):
        
        eigenvectors = self.eigenvectors
        n_galaxies = self.n_galaxies
        n_qso = self.n_qso
        params = self.params
        x_axis, y_axis, yerr = self.obj_spectra[n, :]
        print(eigenvectors[n][n_galaxies:].shape)
        model_qso = jnp.nansum(eigenvectors[n][n_galaxies:].T* params[n][n_galaxies:], axis=1)
        model_galaxy = jnp.nansum(eigenvectors[n][:n_galaxies].T* params[n][:n_galaxies], axis=1)
        text1,text2 = self._make_text(n,model_galaxy)
        # Plot the components
        plt.figure(figsize=(20, 10))
        plt.plot(x_axis, y_axis, label='Observed')
        plt.plot(x_axis,model_qso, label='QSO Model')
        plt.plot(x_axis, model_galaxy, label='Galaxy Model')
        plt.plot(x_axis, model_qso + model_galaxy, label='Total Model')
        plt.axhline(0,ls="--",c="k")
        plt.text(0.05, 0.95, text1, transform=plt.gca().transAxes,
                fontsize=16, verticalalignment='top')
        plt.text(0.02, 1.03, text2, transform=plt.gca().transAxes,
                fontsize=16, verticalalignment='top')
        #text = f"Redshift {sheapclass.z[n]} \n% negatives values {100*sum(model_galaxy<0)/model_galaxy.shape[0]:.2f} \nnum_steps {results.get('num_steps')} \ntime {results.get('time(s)'):.3f}[s]"
        #plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
         #       fontsize=16, verticalalignment='top')
        #plt.text(0.02, 1.03, f"System {sheapclass.names[n]} [{n}]", transform=plt.gca().transAxes,
         #       fontsize=16, verticalalignment='top')
        plt.legend(fontsize=21,framealpha =0)
        plt.xlabel("Wavelength [Å]", fontsize=22)
        plt.ylabel("Flux", fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=18)  # major ticks
        plt.tick_params(axis='both', which='minor', labelsize=16)  # minor ticks if any
        plt.xlim(min(x_axis),max(x_axis))
        plt.tight_layout()
        plt.show()
