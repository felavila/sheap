=========
Changelog
=========

0.0.1 (2025-04-04)
==================

| This is the first public release of the **sheap** Python package, developed as part of the **SHEAP** Project.
| The package is open-source and available at: https://github.com/favila/sheap
| It was scaffolded using the `Cookiecutter Python Package <https://github.com/boromir674/cookiecutter-python-package/tree/master/src/cookiecutter_python>`_ template.

Initial Features
----------------

- Modular codebase designed for the spectral analysis of AGN, with a focus on emission line decomposition and parameter estimation.
- Core components include:
  - `RegionBuilder`: constructs spectral fitting regions from YAML templates, including support for narrow, broad, outflow, and Fe II components.
  - `RegionFitting`: performs constrained optimization using JAX and Optax, with support for multi-step fitting routines and parameter tying.

CI/CD & Automation
------------------

- Continuous Integration pipeline using **GitHub Actions**: https://github.com/favila/sheap/actions
  - Test matrix covering:
    - Platforms: `ubuntu-latest`, `macos-latest`
    - Python versions: `3.12`, `3.13`
  - Automated unit testing with parallel execution across multiple CPUs
  - Code coverage tracking

- Development automation using `tox`:
  - Linting, type-checking, building, and publishing handled via a unified CLI interface

Miscellaneous
-------------

- Includes custom line typing logic, dynamic region templates, and advanced parameter constraints (e.g., tied parameters and template-based FeII fitting).
- Internal utilities for uncertainty propagation, parameter mapping, and multi-model composition (e.g., Gaussian, Lorentzian, power-law, and FeII templates).

Next Steps
----------

- Expand documentation and tutorials
- Improve test coverage and benchmark fitting performance
- Integrate additional physical models and observational constraints

