import pytest
import jax.numpy as jnp
import numpy as np
from sheap.Sheapectral.Sheapectral import Sheapectral

#TODO test the different sub-rutines
@pytest.fixture
def dummy_spectrum():
    wl = jnp.linspace(4000, 5000, 300)
    flux = jnp.ones_like(wl)
    err = jnp.ones_like(wl) * 0.1
    return jnp.stack([wl, flux, err])[None, :, :]


def test_init_from_array(dummy_spectrum):
    SP = Sheapectral(dummy_spectrum, z=0.1)
    assert SP.spectra.shape == (1, 3, 300)
    assert SP.z.shape == (1,)
    assert SP.names[0] == "0"

def test_fit_region(dummy_spectrum):
    SP = Sheapectral(dummy_spectrum)
    SP.makemodel((4100, 4900))
    SP.fitmodel(list_num_steps=[5], list_learning_rate=[1e-2])
    assert hasattr(SP, "result")
    assert SP.result.params.shape[0] == 1


# def test_modelplot_property(dummy_spectrum):
#     SP = Sheapectral(dummy_spectrum)
#     SP.makemodel(4100, 4900)
#     SP.fitmodel(list_num_steps=[5], list_learning_rate=[1e-2])
#     plotter = SP.modelplot
#     assert plotter is not None


# def test_result_panda_structure(dummy_spectrum):
#     SP = Sheapectral(dummy_spectrum)
#     SP.makemodel(4100, 4900)
#     SP.fitmodel(list_num_steps=[5], list_learning_rate=[1e-2])
#     df = SP.result_panda(0)
#     assert set(df.columns) == {"value", "error", "max_constraint", "min_constraint"}
#     assert len(df) > 0


def test_save_and_load_pickle_roundtrip(tmp_path, dummy_spectrum):
    SP = Sheapectral(dummy_spectrum,extinction_correction="done")
    SP.makemodel((4100, 4900))
    SP.fitmodel(list_num_steps=[5], list_learning_rate=[1e-2])

    save_path = tmp_path / "test_sheap.pkl"
    SP.save_to_pickle(save_path)
    loaded = Sheapectral.from_pickle(save_path)

    assert np.allclose(SP.result.params, loaded.result.params, rtol=1e-5)
    assert SP.result.profile_names == loaded.result.profile_names


def test_quicklook_execution(dummy_spectrum):
    import matplotlib
    matplotlib.use("Agg")  # Prevent GUI usage
    SP = Sheapectral(dummy_spectrum)
    ax = SP.quicklook(0)
    assert ax is not None


def test_invalid_spectrum_type():
    with pytest.raises(TypeError):
        Sheapectral(12345)  # not a valid spectrum type