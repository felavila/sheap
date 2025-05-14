
# def Extract_host_pca(Spectra: jnp.array,c=2.99792458e5,num_steps=1000,learning_rate=1e-1):
    
#     # limits fantasy xmin=4050, xmax=7300
#     glx = fits.open(f"{module_dir}/gal_eigenspec_Yip2004.fits")
#     glx = glx[1].data
#     gl_wave = glx["wave"].flatten()
#     gl_wave = jnp.array(vac_to_air(gl_wave))
#     gl_flux = jnp.array(glx["pca"].reshape(glx["pca"].shape[1], glx["pca"].shape[2]))
#     xmax = 7300 #min(qso_wave[-1],gl_wave[-1])
#     xmin = 4050 #max(qso_wave[0],gl_wave[0])
#     qso = fits.open(f"{module_dir}/qso_eigenspec_Yip2004_global.fits")

#     qso = qso[1].data
#     qso_wave = qso["wave"].flatten()
#     qso_wave= jnp.array(vac_to_air(qso_wave))
#     qso_flux = jnp.array(qso["pca"].reshape(qso["pca"].shape[1], qso["pca"].shape[2]))
   
#     # Define coverage masks for observed wavelengths within PCA wavelength ranges
#     array,masked_uncertainties,_,_ = mask_builder(Spectra,outer_limits=[xmin,xmax])
#     mask_qso = (array[:, 0, :] >=xmin) & (array[:, 0, :] <= xmax)
#     mask_glx = (array[:, 0, :] >= xmin) & (array[:, 0, :] <= xmax)
    
    
#     # Interpolate and move axis
#     qso_interp = interpolate_flux_array(array[:, 0, :], qso_wave, qso_flux)
#     glx_interp = interpolate_flux_array(array[:, 0, :], gl_wave, gl_flux)

#     # Apply mask: set values outside the PCA template wavelength range to zero
#     qso_ = jnp.where(mask_qso[:, None, :], jnp.moveaxis(qso_interp, 0, 1), 0.0)
#     gl_ = jnp.where(mask_glx[:, None, :], jnp.moveaxis(glx_interp, 0, 1), 0.0)

#     # Normalize after masking to avoid NaNs
#     gl_ = normalize(gl_)
#     qso_ = normalize(qso_)
#     #gal_=jnp.moveaxis(vmap_interp(fit_array[:,0],gl_wave,gl_flux),0,1)#
#     eigenvectors = jnp.hstack((gl_,qso_)) #why it takes so long ? xd from 0 to 10 gal rest galaxy
    
#     initial_params = jnp.array([1.0]*10 + [1.0]*50)
#     constraints = jnp.array([[-1.e3, +1.e3]]*10 + [[-1e3, +1e3]]*50)
    
#     start = time.perf_counter()
#     minimizer = MasterMinimizer(linear_combination, optimize_in_axis=3,
#                                 num_steps=num_steps,learning_rate=learning_rate,
#                                 penalty_function=galaxy_negative_penalty,   # <--- Add this
#                                 penalty_weight = 1.0   )                # <--- Add this)
#     params, loss = minimizer(initial_params,eigenvectors,array[:,1,:],array[:,2,:],constraints)
#     end = time.perf_counter()
#     #master_interp.vmap_optimize_model(initial_params,eigenvectors,fit_array[:, 1, :],masked_uncertainties,constraints,*master_interp.default_args)
#     return {"qso":qso_,"gl":gl_,"params":params,"eigenvectors":eigenvectors,'num_steps':num_steps,"time(s)": end - start,"array":array}
