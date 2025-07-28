import pytest
import jax.numpy as jnp
import numpy as np
from sheap.MainSheap import Sheapectral


@pytest.fixture
def dummy_spectrum():
    wl = jnp.linspace(4000, 5000, 300)
    flux = jnp.ones_like(wl)
    err = jnp.ones_like(wl) * 0.1
    return jnp.stack([wl, flux, err])[None, :, :]


def test_init_from_array(dummy_spectrum):
    sheap = Sheapectral(dummy_spectrum, z=0.1)
    assert sheap.spectra.shape == (1, 3, 300)
    assert sheap.z.shape == (1,)
    assert sheap.names[0] == "0"


def test_make_region(dummy_spectrum):
    sheap = Sheapectral(dummy_spectrum)
    sheap.make_region(4100, 4900, n_narrow=1, n_broad=1)
    assert hasattr(sheap, "complexbuild")
    assert sheap.complexbuild.xmin == 4100
    assert sheap.complexbuild.n_narrow == 1


def test_fit_region(dummy_spectrum):
    sheap = Sheapectral(dummy_spectrum)
    sheap.make_region(4100, 4900)
    sheap.fit_region(list_num_steps=[5], list_learning_rate=[1e-2], run_uncertainty_params=False)
    assert hasattr(sheap, "result")
    assert sheap.result.params.shape[0] == 1


def test_modelplot_property(dummy_spectrum):
    sheap = Sheapectral(dummy_spectrum)
    sheap.make_region(4100, 4900)
    sheap.fit_region(list_num_steps=[5], list_learning_rate=[1e-2], run_uncertainty_params=False)
    plotter = sheap.modelplot
    assert plotter is not None


def test_result_panda_structure(dummy_spectrum):
    sheap = Sheapectral(dummy_spectrum)
    sheap.make_region(4100, 4900)
    sheap.fit_region(list_num_steps=[5], list_learning_rate=[1e-2], run_uncertainty_params=False)
    df = sheap.result_panda(0)
    assert set(df.columns) == {"value", "error", "max_constraint", "min_constraint"}
    assert len(df) > 0


def test_save_and_load_pickle_roundtrip(tmp_path, dummy_spectrum):
    sheap = Sheapectral(dummy_spectrum)
    sheap.make_region(4100, 4900)
    sheap.fit_region(list_num_steps=[5], list_learning_rate=[1e-2], run_uncertainty_params=False)

    save_path = tmp_path / "test_sheap.pkl"
    sheap.save_to_pickle(save_path)
    loaded = Sheapectral.from_pickle(save_path)

    assert np.allclose(sheap.result.params, loaded.result.params, rtol=1e-5)
    assert sheap.result.profile_names == loaded.result.profile_names


def test_quicklook_execution(dummy_spectrum):
    import matplotlib
    matplotlib.use("Agg")  # Prevent GUI usage
    sheap = Sheapectral(dummy_spectrum)
    ax = sheap.quicklook(0)
    assert ax is not None


def test_invalid_spectrum_type():
    with pytest.raises(TypeError):
        Sheapectral(12345)  # not a valid spectrum type