=====
Usage
=====

------------
Installation
------------

| **sheap** is available on PyPI, hence you can use `pip` to install it.

It is recommended to perform the installation in an isolated `python virtual environment` (env).
You can create and activate an `env` using any tool of your preference (e.g., `virtualenv`, `venv`, `pyenv`).

Assuming you have *activated* a `python virtual environment`:

.. code-block:: shell

  python -m pip install sheap


---------------
Simple Use Case
---------------

| A common use case for **sheap** is the spectral decomposition of AGN data cubes or 1D spectra to extract emission line parameters such as flux, FWHM, and equivalent width for scientific inference (e.g., black hole mass estimates, bolometric luminosity, diagnostic ratios).

| SHEAP is designed to work with spectra in shape `(N, 3, M)` where:
  - `N` is the number of spectra (e.g., galaxies, exposures, spaxels),
  - `3` represents `[wavelength, flux, uncertainty]`, and
  - `M` is the number of pixels per spectrum.

| The analysis workflow typically includes:
  - Building a region of interest (e.g., Hβ or MgII zone) using `RegionBuilder`.
  - Mapping line definitions and constraints to parameter arrays using `LineMapper`.
  - Fitting the model using `RegionFitting`, powered by JAX + Optax optimization.
  - Extracting quantities with uncertainty using `auto_uncertainties`.

| A full working example is under development and will be included in the upcoming releases. Please refer to the [examples/](https://github.com/your-repo/sheap/tree/main/examples) directory in the repository once available.

.. note::

   Development is active, and interface/API details may change slightly until the 1.0.0 release.
