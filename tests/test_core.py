import pytest
import jax.numpy as jnp
import numpy as np
from sheap.Core import SpectralLine, ComplexRegion, ComplexResult, FittingLimits, ProfileConstraintSet


def test_spectral_line_to_dict():
    line = SpectralLine(line_name="Halpha", center=6563.0, region="narrow", component=1)
    d = line.to_dict()
    assert isinstance(d, dict)
    assert d["line_name"] == "Halpha"
    assert d["center"] == 6563.0


def test_fitting_limits_from_dict():
    d = {
        "upper_fwhm": 5000,
        "lower_fwhm": 100,
        "center_shift": 200,
        "v_shift": 300,
        "max_amplitude": 1e4
    }
    fl = FittingLimits.from_dict(d)
    assert fl.upper_fwhm == 5000
    assert fl.max_amplitude == 1e4


def test_profile_constraint_set_valid():
    pcs = ProfileConstraintSet(
        init=[1.0, 2.0],
        upper=[10.0, 20.0],
        lower=[0.0, 1.0],
        profile="gaussian",
        param_names=["amp", "center"]
    )
    assert pcs.init[0] == 1.0


def test_complex_region_init_and_df():
    lines = [SpectralLine("Halpha", center=6563.0, region="narrow", component=1)]
    cr = ComplexRegion(lines=lines)
    df = cr.as_df()
    assert "line_name" in df.columns
    assert df.iloc[0]["line_name"] == "Halpha"


def test_complex_result_to_dict():
    lines = [SpectralLine("Halpha", center=6563.0, region="narrow", component=1)]
    dummy_params = jnp.ones((1, 2))
    cr = ComplexResult(
        complex_region=lines,
        params=dummy_params,
        uncertainty_params=dummy_params * 0.1,
        mask=np.zeros((1, 2), dtype=bool),
        constraints=dummy_params[0][:, None].repeat(2, axis=1),
        profile_functions=[lambda x, p: x * p[0] + p[1]],
        profile_names=["linear"],
        loss=[0.01],
        profile_params_index_list=[[0, 1]],
        initial_params=dummy_params[0],
        scale=jnp.ones((1,)),
        params_dict={"amp_Halpha_1_narrow": 0, "center_Halpha_1_narrow": 1},
        outer_limits=[6400, 6700],
        inner_limits=[6500, 6600],
        model_keywords={"test": True},
        source={"type": "synthetic"},
        dependencies=[],
        free_params=jnp.array([1]),
        residuals=jnp.array([[0.0, 0.1]]),
        chi2_red=jnp.array([1.1])
    )
    assert isinstance(cr.to_dict(), dict)
    assert isinstance(cr.complex_class, ComplexRegion)