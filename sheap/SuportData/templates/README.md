ref
===

Reference for this files please cite the corresponding authors in case you use it.


- **Kstar.txt**: ?

- **fe2_template2022.txt**: Iron emission template from [Park et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJS..258...38P), available at [GitHub](https://github.com/DaeseongPark/Iron_Template).

- **fe2_OP.txt**: Iron emission template from [J. E. Mejía‑Restrepo et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016MNRAS.460..187M/abstract).

- **fe2_UV02.txt**: Iron emission template from [J. E. Mejía‑Restrepo et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016MNRAS.460..187M/abstract).

- **miles_cube_log.npz**: Compressed NumPy archive containing the SSP template cube and associated grids:

  - **cube_log**: 3D array of log‑flux SSP templates, shape (n_z, n_age, n_wave)  
  - **wave_log**: 1D array of log‑wavelength grid (Å)  
  - **ages_sub**: 1D array of ages (Gyr) corresponding to the second axis of `cube_log`  
  - **zs_sub**: 1D array of metallicities [M/H] corresponding to the first axis of `cube_log`  
  - **sigmatemplate**: Template velocity dispersion grid (km s⁻¹) used internally  
  - **fixed_dispersion**: Boolean flag indicating whether a constant dispersion was enforced  
  - **resolution**: Full‑width at half‑maximum (FWHM) resolution of the templates (Å)  
  - **dlam**: Wavelength increment per pixel (Å/pix)  
  - **wave0**: Starting wavelength (Å) of the linear grid  
  - **wave1**: Ending wavelength (Å) of the linear grid  

  These templates were generated using the MILES “Tune SSP Models” web‐tool at IAC with the following settings:

  Input parameters
  ----------------

  :SSP Models:         E-MILES  
  :Isochrone:          Padova+00  
  :Type of IMF:        un  
  :IMF slope:          2.30  

  :Metallicity [M/H]:  
    - All  

  :Age (Gyr):  
    - All  

  :[α/Fe]:  
    - baseFe  

  Output parameters
  -----------------

  :λ_initial (Å):       3541.40 
  :λ_final (Å):         8950.40
  :Δλ (Å/pix):          0.9  
  :Sampling:            Linear  
  :Redshift (z):        0.0  
  :Resolution (Å FWHM): 2.51 (3541.40 ,8950.40)
  :Format:              FITS  

  Reference
  ---------

  Generated with the “Tune SSP Models” tool:
  https://research.iac.es/proyecto/miles/pages/webtools/tune-ssp-models.php
