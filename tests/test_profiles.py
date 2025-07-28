import pytest
import jax.numpy as jnp
import numpy as np
from sheap.Profiles.profiles import (
    PROFILE_LINE_FUNC_MAP,
    PROFILE_CONTINUUM_FUNC_MAP,
    PROFILE_FUNC_MAP
)
from sheap.Profiles.profiles_templates import (
    make_feii_template_function,
    make_host_function
)

# ---------- BASIC PROFILES (Line + Continuum) ----------

@pytest.fixture(params=list(PROFILE_LINE_FUNC_MAP.items()) + list(PROFILE_CONTINUUM_FUNC_MAP.items()))
def profile_func_fixture(request):
    return request.param

def test_profile_signature(profile_func_fixture):
    name, profile = profile_func_fixture
    x = jnp.linspace(4000, 5000, 100)
    num_params = len(profile.param_names)
    params = jnp.ones((num_params,))
    result = profile(x, params)
    assert result.shape == x.shape, f"{name} output shape mismatch"
    assert result.dtype == x.dtype, f"{name} output dtype mismatch"

def test_profilefunc_type_compliance(profile_func_fixture):
    name, profile = profile_func_fixture
    assert callable(profile), f"{name} is not callable"
    assert hasattr(profile, "param_names"), f"{name} missing param_names attribute"

# ---------- TEMPLATE PROFILES: FeII ----------

@pytest.mark.parametrize("template_name", ["feop", "feuv"])
def test_feii_template_profile_runs(template_name):
    wrapper = make_feii_template_function(template_name)
    model = wrapper["model"]
    assert callable(model)
    x = jnp.linspace(1000, 8000, 1000)
    params = jnp.array([0.5, 2.0, 0.0])  # logamp, logFWHM, shift
    y = model(x, params)
    assert y.shape == x.shape
    assert y.dtype == x.dtype

# ---------- TEMPLATE PROFILE: Host Galaxy ----------

def test_host_template_profile_runs():
    wrapper = make_host_function()
    model = wrapper["model"]
    assert callable(model)
    x = jnp.linspace(3500, 7500, 1000)
    n_weights = len(model.param_names) - 3
    params = jnp.concatenate([
        jnp.array([0.5, 2.0, 0.0]),       # logamp, logFWHM, shift
        jnp.full((n_weights,), 0.001)     # weights
    ])
    y = model(x, params)
    assert y.shape == x.shape
    assert y.dtype == x.dtype
