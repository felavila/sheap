import pytest
from sheap.SheapModelBuilder import SheapModelBuilder
from sheap.Core import SheapModel, SpectralLine


def test_basic_initialization():
    cb = SheapModelBuilder(xmin=4000, xmax=7000)
    assert isinstance(cb.lines_available, dict)
    assert cb.xmin == 4000
    assert cb.xmax == 7000
    assert cb.fe_mode in ["template", "model", "none"]


def test_region_creation_defaults():
    cb = SheapModelBuilder(xmin=4000, xmax=7000)
    assert isinstance(cb.sheapmodel, SheapModel)
    assert len(cb.sheapmodel.lines) > 0


def test_region_override_parameters():
    cb = SheapModelBuilder(xmin=4000, xmax=7000, fe_mode="none", add_balmer_continuum=True)
    cb.make_region(4200, 6800, n_narrow=2, n_broad=1)
    assert isinstance(cb.sheapmodel, SheapModel)
    names = [line.line_name for line in cb.sheapmodel.lines]
    assert any("balmer" in name for name in names)


def test_fitting_routine_structure():
    cb = SheapModelBuilder(xmin=4000, xmax=7000)
    routine = cb._make_fitting_routine(list_num_steps=[100, 100], list_learning_rate=[1e-2, 1e-3])
    assert "step1" in routine["fitting_routine"]
    assert "sheapmodel" in routine
    assert isinstance(routine["sheapmodel"], SheapModel)


def test_add_host_template():
    cb = SheapModelBuilder(xmin=3500, xmax=7500, add_host_miles=True)
    lines = cb.sheapmodel.lines
    assert any(line.profile == "hostmiles" for line in lines)
