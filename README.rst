.. image:: https://raw.githubusercontent.com/felavila/sheap/main/docs/source/_static/sheap_withname.png
   :alt: SHEAP Logo
   :align: left
   :width: 700


Spectral Handling and Estimation of AGN Parameters
=========================================================
|pypi_badge| |docs_badge|

**sheap** is a Python 3 package designed to analyze and estimate key parameters of Active Galactic Nuclei (AGN) from spectral data. This package provides tools to streamline the handling of spectral data and applies models to extract relevant AGN properties efficiently.

Features
========

- **Spectral Fitting**: Automatically fits AGN spectra to estimate key physical parameters.
- **Model Customization**: Allows flexible models for AGN spectra to suit a variety of use cases.
- **AGN Parameter Estimation**: Extract black hole mass from observed spectra.

Installation
============

You can install sheap locally using the following command:

.. code-block:: shell

    pip install -e .

Prerequisites
=============

You need to have Python (>=3.12) and the required dependencies installed. Dependencies are managed using Poetry or can be installed manually via `requirements.txt`.

References
==========

sheap is based on methodologies and models outlined in the following paper:

-  **Mejía-Restrepo, J. E., et al. (2016)**.
   *Active galactic nuclei at z ∼ 1.5 – II. Black hole mass estimation by means of broad emission lines.*
   Monthly Notices of the Royal Astronomical Society, **460**, 187.
   Available at: `ADS Abstract <https://ui.adsabs.harvard.edu/abs/2016MNRAS.460..187M/abstract>`_


License
=======

* `GNU Affero General Public License v3.0 <https://www.gnu.org/licenses/agpl-3.0.html>`_

.. |pypi_badge| image:: https://img.shields.io/pypi/v/sheap.svg
   :alt: PyPI version
   :target: https://pypi.org/project/sheap/

.. |docs_badge| image:: https://readthedocs.org/projects/sheap/badge/?version=latest
   :alt: Documentation Status
   :target: https://sheap.readthedocs.io/en/latest/?badge=latest

