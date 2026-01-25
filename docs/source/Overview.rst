.. sheap: Spectral Handling and Estimation of AGN Parameters
.. =========================================================

sheap's Overview
================

Overview
--------

**sheap** is a Python library for modeling, fitting, and sampling astronomical spectra. Leveraging modern JAX-based numerical routines and probabilistic inference via NumPyro, sheap provides a flexible, high-performance framework for:

- **Pre-processing**: automated Galactic extinction and redshift corrections  
- **Region definition**: build spectral moodel with emission‐line regions (e.g. broad, narrow, outflow) Fe II templates, Balmer continuum, and more  
- **Deterministic fitting**: gradient-based optimization of continuum and multi-component line profiles (Gaussian, Lorentzian, linear, broken-powerlaw)  
- **Uncertainty estimation**: Monte Carlo sampling using per-pixel spectral uncertainties.
- **Bayesian sampling**: posterior sampling of line and continuum parameters using Hamiltonian Monte Carlo (NUTS)

Key Features
------------

- **High performance**: JIT-compiled flux modeling and optimization (via JAX & Optax)  
- **Modular API**: separate stages for region building, fitting, plotting, and sampling  
- **Flexible templates**: define custom line lists via YAML or Python dicts  
- **Extensible**: add new profile shapes, priors/constraints, and sampling strategies  

Quickstart
----------

1. **Install sheap**  

   .. code-block:: shell

      pip install sheap

2. **Load a spectrum**

   .. code-block:: python

      from sheap import Sheapectral
      spec = Sheapectral("my_spectrum.txt", z=0.5, ebv=0.02)

3. **Build a fitting region**

   .. code-block:: python
      xmin,xmax = 4500,5500
      spec.makemodel((xmin,xmax ), n_narrow=1, n_broad=1, fe_mode="template")

4. **Fit**

   .. code-block:: python

      spec.fitmodel()

5. **Inspect results**

   .. code-block:: python

      fig    = spec.plotter.plot(0)
      params = spec.result.params

6. **Obtain the extra products**

   .. code-block:: python

      spec.estimate_posteriors(sampling_method="montecarlo")
      spec.result.posterior[1]



Documentation
-------------

See the following modules for detailed API reference:

- :py:mod:`Sheapectral <sheap.Sheapectral.Sheapectral>`: core entry point, I/O, extinction & redshift correction  

- :py:mod:`SheapModelBuilder <sheap.SheapModelBuilder.SheapModelBuilder>`: construct line‐fitting templates from YAML & rules  

- :py:mod:`SheapModelFitting <sheap.SheapModelFitting.SheapModelFitting>`: perform JAX/Optax minimization with constraints  

- :py:mod:`Minimizer <sheap.Minimizer.Minimizer>`: low‐level optimizer wrapper  

- :py:mod:`MasterSampler <sheap.MasterSampler.MasterSampler>`: Posterior sampling 

- :py:mod:`SheaProducts <sheap.SheaProducts.SheaProducts>`: sheap extra products. 

.. Installation
.. ------------

.. :: 

..   pip install sheap

.. License
.. -------

.. * `GNU Affero General Public License v3.0 <https://www.gnu.org/licenses/agpl-3.0.html>`_
