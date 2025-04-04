import jax.numpy as jnp
from jax import jit,vmap,value_and_grad
from SHEAP.utils import mask_builder
from SHEAP.tools.others import vac_to_air
from SHEAP.tools.interp_tools import vmap_interp
from SHEAP.fitting.functions import linear_combination
from SHEAP.fitting.utils import project_params
from SHEAP.fitting.main_fitting_I import MasterMinimizer
from pathlib import Path
from astropy.io import fits
import optax

#print("xaaxaxaas")
module_dir = Path(__file__).resolve().parent.parent

def Extract_host_pca(Spectra: jnp.array,c=2.99792458e5,num_steps=1000,learning_rate=1e-1):
    """
    Processes a set of spectra data.

    Parameters:
    Spectra (jax.numpy.ndarray): A JAX array of shape (X, 3, N) where:
        - X: Number of spectra.
        - 3: Represents [wavelength, flux, error].
        - N: Number of pixels in each spectrum.
    c speed of light on km/h
    Returns:
    None: Modify this as needed based on what the function should return.
    """
    fit_array,masked_uncertainties,_,_ = mask_builder(Spectra,outer_limits=[3450,7190])
    
    
    glx = fits.open(f"{module_dir}/suport_data/eigen/gal_eigenspec_Yip2004.fits")
    glx = glx[1].data
    gl_wave = glx["wave"].flatten()
    gl_wave=jnp.array(vac_to_air(gl_wave))
    gl_flux = jnp.array(glx["pca"].reshape(glx["pca"].shape[1], glx["pca"].shape[2]))

    qso = fits.open(f"{module_dir}/suport_data/eigen/qso_eigenspec_Yip2004_global.fits")

    qso = qso[1].data
    qso_wave = qso["wave"].flatten()
    qso_wave= jnp.array(vac_to_air(qso_wave))
    qso_flux = jnp.array(qso["pca"].reshape(qso["pca"].shape[1], qso["pca"].shape[2]))
    qso_=jnp.moveaxis(vmap_interp(fit_array[:,0],qso_wave,qso_flux),0,1) #spectra templates flux
    gal_=jnp.moveaxis(vmap_interp(fit_array[:,0],gl_wave,gl_flux),0,1)#
    eigenvectors = jnp.hstack((gal_,qso_)) #why it takes so long ? xd from 0 to 10 gal rest galaxy
    initial_params = jnp.array([1.0]*60)
    constraints= jnp.array([[0,+1e41]]*60)
    master_interp = MasterMinimizer(linear_combination, optimize_in_axis=3,num_steps=num_steps,learning_rate=learning_rate)
    params_linear,loss_curves_linear = master_interp.vmap_optimize_model(initial_params,eigenvectors,fit_array[:, 1, :],masked_uncertainties,constraints,*master_interp.default_args)
    #combination = eigenvectors*100*params_linear[:,:,None]
    #negatives_per_column = jnp.nansum(combination < 0, axis=2)
    #params_linear_init = jnp.where(negatives_per_column>1000,0,params_linear)
    
    return eigenvectors,params_linear,loss_curves_linear,masked_uncertainties,fit_array


def Extract_host_pca_new(Spectra: jnp.array,c=2.99792458e5,num_steps=1000):
    """
    Processes a set of spectra data.

    Parameters:
    Spectra (jax.numpy.ndarray): A JAX array of shape (X, 3, N) where:
        - X: Number of spectra.
        - 3: Represents [wavelength, flux, error].
        - N: Number of pixels in each spectrum.
    c speed of light on km/h
    Returns:
    None: Modify this as needed based on what the function should return.
    """
    fit_array,masked_uncertainties,_,_ = mask_builder(Spectra,outer_limits=[3450,7190])
    
    
    glx = fits.open(f"{module_dir}/suport_data/eigen/gal_eigenspec_Yip2004.fits")
    glx = glx[1].data
    gl_wave = glx["wave"].flatten()
    gl_wave=jnp.array(vac_to_air(gl_wave))
    gl_flux = jnp.array(glx["pca"].reshape(glx["pca"].shape[1], glx["pca"].shape[2]))

    qso = fits.open(f"{module_dir}/suport_data/eigen/qso_eigenspec_Yip2004_global.fits")

    qso = qso[1].data
    qso_wave = qso["wave"].flatten()
    qso_wave= jnp.array(vac_to_air(qso_wave))
    qso_flux = jnp.array(qso["pca"].reshape(qso["pca"].shape[1], qso["pca"].shape[2]))
    
    #####################
    qso_ = jnp.moveaxis(vmap_interp(fit_array[:,0],qso_wave,qso_flux),0,1) #spectra templates flux
    gal_=jnp.moveaxis(vmap_interp(fit_array[:,0],gl_wave,gl_flux),0,1)#
    del glx,gl_wave,gl_flux,qso,qso_wave,qso_flux
    eigenvectors = jnp.hstack((gal_,qso_)) #why it takes so long ? xd from 0 to 10 gal rest galaxy
    initial_params = jnp.array([1.0]*60)
    constraints= jnp.array([[0,+1e41]]*60)
    @jit
    def loss_function(
        params: jnp.ndarray,
        xs: List[jnp.ndarray],
        y: jnp.ndarray,
        y_uncertainties: jnp.ndarray
        ) -> jnp.ndarray:
        
        y_pred = linear_combination(xs, params)
        weights = 1.0 / y_uncertainties**2
        wmse = jnp.nansum(weights * (y - y_pred)**2) / jnp.nansum(weights)
        
        return wmse 
    
    def optimize_model(
            initial_params: jnp.ndarray,
            xs: List[jnp.ndarray],
            y: jnp.ndarray,
            y_uncertainties: jnp.ndarray,
            constraints: jnp.ndarray= None,
            learning_rate: float = 1e-2,
            num_steps: int = 1000,
            optimizer = None,
            verbose: bool = False,
        parsed_dependencies = None) -> Tuple[jnp.ndarray, list]:
            # Initialize parameters and optimizer state
            params = initial_params
            optimizer = optimizer or optax.adabelief(learning_rate)
            opt_state = optimizer.init(params)
            loss_history = []
            
            if constraints is None:
                constraints = jnp.array([[-1e41,1e41]] * params.shape[0])
            # Define the step function with constraints captured via closure
            @jit
            def step(params, opt_state, xs, y):
                # Compute loss and gradients
                loss, grads = value_and_grad(loss_function)(
                    params, jnp.nan_to_num(xs), jnp.nan_to_num(y), y_uncertainties)

                # Compute parameter updates
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                
                # Project parameters to enforce constraints
                params = project_params(
                    params,
                    constraints,parsed_dependencies)
                
                return params, opt_state, loss

            # Optimization loop
            for step_num in range(num_steps):
                params, opt_state, loss = step(params, opt_state, xs, y)
                
                loss_history.append(loss)
                if step_num % 100 == 0 and verbose:
                    print(f"Step {step_num}, Loss: {loss:.4f}")

            return params, loss_history

    
    
    
    #master_interp = MasterMinimizer(linear_combination, optimize_in_axis=3,num_steps=num_steps)
    #params_linear,loss_curves_linear = master_interp.vmap_optimize_model(initial_params,eigenvectors,fit_array[:, 1, :],masked_uncertainties,constraints,*master_interp.default_args)
    #return eigenvectors,params_linear,loss_curves_linear,masked_uncertainties,fit_array