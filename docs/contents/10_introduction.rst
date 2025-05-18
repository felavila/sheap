============
Introduction
============

| This is **SHEAP**, a *Python package* designed to analyze and estimate key physical parameters of Active Galactic Nuclei (AGN) from spectral data using flexible, high-performance modeling techniques.

| The goal of this project is to provide a modular, efficient, and accurate tool for the decomposition and analysis of AGN spectra. It enables researchers to fit broad and narrow emission lines, Fe~II templates, and continuum components — accounting for both observational uncertainties and physical constraints. 

| Additionally, SHEAP is designed with scalability and scientific reproducibility in mind. It supports vectorized spectral fitting using JAX and Optax, incorporates prior knowledge via constraints, and allows batch processing of large spectral datasets. Features like GPU acceleration, automatic uncertainty propagation, and dynamic region building make it ideal for modern AGN spectroscopic surveys (e.g., 4MOST-ChAnGES).

| This documentation aims to help people understand the package's features and demonstrate
| how to leverage them for their use cases.
| It also presents the overall package design, from data ingestion and spectral region definition, to parameter fitting and scientific inference.
