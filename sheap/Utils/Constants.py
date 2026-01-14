"""
Constants and Configuration
===========================

This module defines physical constants and loads configuration data
from YAML files, such as parameter limits, single-epoch estimators, 
and bolometric corrections.

Attributes
----------
c : float
    Speed of light in km/s (``299792.458``).
cm_per_mpc : float
    Conversion factor from megaparsecs to centimeters (``3.08568e24``).
DEFAULT_LIMITS : dict
    Default parameter limits loaded from ``DefaultLimits.yaml``.
SINGLE_EPOCH_ESTIMATORS : dict
    Calibration recipes for single-epoch black hole mass estimation,
    loaded from ``SingleEpochEstimators.yaml``.
BOL_CORRECTIONS : dict
    Bolometric corrections for AGN luminosities,
    loaded from ``BolometricCorrections.yaml``.

Functions
---------
read_yaml(p: Path) -> dict
    Load and cache a YAML file into a Python dictionary.
"""

from __future__ import annotations
__author__ = 'felavila'

from pathlib import Path
from functools import lru_cache
import yaml

__all__ = [
    "BOL_CORRECTIONS",
    "DEFAULT_LIMITS",
    "SINGLE_EPOCH_ESTIMATORS",
    "c",
    "cm_per_mpc",
    "read_yaml",
    "DEFAULT_C_KMS",
    "C_KMS",
    "FWHM_TO_SIGMA"
]

# ---------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------
c: float = 299792.458       #: Speed of light in km/s
DEFAULT_C_KMS: float = 299_792.458       #: Speed of light in km/s
cm_per_mpc: float = 3.08568e24  #: Megaparsec in centimeters
C_KMS = 299_792.458
FWHM_TO_SIGMA = 1.0 / 2.355
# ---------------------------------------------------------------------
# Paths to YAML configuration files
# ---------------------------------------------------------------------
_DEFAULT_LIMITS = Path(__file__).resolve().parent.parent / "SuportData" / "DefaultLimits" / "DefaultLimits.yaml"
_SINGLE_EPOCH_ESTIMATORS = Path(__file__).resolve().parent.parent / "SuportData" / "SingleEpochEstimators" / "SingleEpochEstimators.yaml"
_BOL_CORRECTIONS = Path(__file__).resolve().parent.parent / "SuportData" / "BolometricCorrections" / "BolometricCorrections.yaml"

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _assert_exists(p: Path) -> None:
    """Raise if the given path does not exist as a file."""
    if not p.is_file():
        raise FileNotFoundError(f"YAML not found: {p}")

@lru_cache(maxsize=None)
def read_yaml(p: Path) -> dict:
    """
    Load a YAML file into a Python dictionary, with caching.

    Parameters
    ----------
    p : Path
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    _assert_exists(p)
    with p.open("r") as f:
        return yaml.safe_load(f)

# ---------------------------------------------------------------------
# Load configuration data
# ---------------------------------------------------------------------
DEFAULT_LIMITS = read_yaml(_DEFAULT_LIMITS)
DEFAULT_SINGLE_EPOCH_ESTIMATORS = read_yaml(_SINGLE_EPOCH_ESTIMATORS)
DEFAULT_BOL_CORRECTIONS = read_yaml(_BOL_CORRECTIONS)
