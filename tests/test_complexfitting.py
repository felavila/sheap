import pytest
import jax.numpy as jnp
import numpy as np
from sheap.ComplexBuilder import ComplexBuilder
from sheap.ComplexFitting import ComplexFitting
from sheap.Core import ComplexResult

@pytest.fixture
def dummy_spectrum():
    wl = jnp.linspace(4000, 5000, 300)
    flux = jnp.ones_like(wl)
    err = jnp.ones_like(wl) * 0.1
    return jnp.stack([wl, flux, err])[None, :, :]

def test_init_from_builder(dummy_spectrum):
    builder = ComplexBuilder(xmin=4000, xmax=5000)
    cf = ComplexFitting.from_builder(builder)
    assert isinstance(cf.model, type(jnp.sin))  # jitted function

def test_basic_fit_and_result(dummy_spectrum):
    builder = ComplexBuilder(xmin=4000, xmax=5000)
    cf = ComplexFitting.from_builder(builder)
    cf(dummy_spectrum, list_learning_rate=[1e-2], run_uncertainty_params=False)
    assert isinstance(cf.complexresult, ComplexResult)
    assert cf.params.shape[0] == 1

def test_profile_names_and_dict(dummy_spectrum):
    builder = ComplexBuilder(xmin=4000, xmax=5000)
    cf = ComplexFitting.from_builder(builder)
    assert len(cf.profile_names) > 0
    assert isinstance(cf.params_dict, dict)

def test_fit_with_penalty(dummy_spectrum):
    builder = ComplexBuilder(xmin=3500, xmax=7000, add_host_miles=True)
    cf = ComplexFitting.from_builder(builder)
    cf(dummy_spectrum, list_learning_rate=[1e-2], run_uncertainty_params=False, add_penalty_function=True)
    assert hasattr(cf, "params")

def test_to_result_call(dummy_spectrum):
    builder = ComplexBuilder(xmin=4000, xmax=5000)
    cf = ComplexFitting.from_builder(builder)
    cf(dummy_spectrum, list_learning_rate=[1e-2], run_uncertainty_params=False)
    result = cf.to_result()
    assert isinstance(cf.complexresult, ComplexResult)
