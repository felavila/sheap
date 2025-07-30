SHEAP: Spectral Handling and Estimation of AGN Parameters
=========================================================

Overview
--------

**sheap** (Spectral Handling and Estimation of AGN Parameters) is a Python library for modeling, fitting, and sampling astronomical spectra. Leveraging modern JAX-based numerical routines and probabilistic inference via NumPyro, sheap provides a flexible, high-performance framework for:

- **Pre-processing**: automated Galactic extinction and redshift corrections  
- **Region definition**: build complex emission‐line regions with broad, narrow, outflow, Fe II templates, Balmer continuum, and more  
- **Deterministic fitting**: gradient-based optimization of continuum and multi-component line profiles (Gaussian, Lorentzian, linear, broken-powerlaw)  
- **Uncertainty estimation**: covariance estimation via error‐propagation loops  
- **Bayesian sampling**: posterior sampling of line and continuum parameters using Hamiltonian Monte Carlo (NUTS)

Key Features
------------

- **High performance**: JIT-compiled flux modeling and optimization (via JAX & Optax)  
- **Modular API**: separate stages for region building, fitting, plotting, and sampling  
- **Flexible templates**: define custom line lists via YAML or Python dicts  
- **Extensible**: add new profile shapes, priors/constraints, and sampling strategies  

Quickstart
----------

1. **Install SHEAP**  

   .. code-block:: shell

      pip install sheap

2. **Load a spectrum**

   .. code-block:: python

      from sheap.MainSheap import Sheapectral
      spec = Sheapectral("my_spectrum.txt", z=0.5, ebv=0.02)

3. **Build a fitting region**

   .. code-block:: python

      spec.makecomplex(xmin=4500, xmax=5500, n_narrow=1, n_broad=1, fe_mode="template")

4. **Fit**

   .. code-block:: python

      spec.fitcomplex()

5. **Inspect results**

   .. code-block:: python

      fig    = spec.plotter.plot(0)
      params = spec.result.params(0)

6. **Obtain the extra parameters**

   .. code-block:: python

      spec.afterfit()


Documentation
-------------

See the following modules for detailed API reference:

- :py:mod:`sheap.MainSheap`: core entry point, I/O, extinction & redshift correction  

- :py:mod:`sheap.ComplexBuilder`: construct line‐fitting templates from YAML & rules  

- :py:mod:`sheap.ComplexFitting`: perform JAX/Optax minimization with constraints  

- :py:mod:`sheap.Minimizer`: low‐level optimizer wrapper  

- :py:mod:`sheap.ComplexAfterFit`: Posterior sampling 

- **utils**: parameter projection, loss building, dependency parsing  

.. Installation
.. ------------

.. :: 

..   pip install sheap

.. License
.. -------

.. * `GNU Affero General Public License v3.0 <https://www.gnu.org/licenses/agpl-3.0.html>`_
