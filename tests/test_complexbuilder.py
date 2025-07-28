import pytest
from sheap.ComplexBuilder import ComplexBuilder
from sheap.Core import ComplexRegion, SpectralLine


def test_basic_initialization():
    cb = ComplexBuilder(xmin=4000, xmax=7000)
    assert isinstance(cb.lines_available, dict)
    assert cb.xmin == 4000
    assert cb.xmax == 7000
    assert cb.fe_mode in ["template", "model", "none"]


def test_region_creation_defaults():
    cb = ComplexBuilder(xmin=4000, xmax=7000)
    assert isinstance(cb.complex_class, ComplexRegion)
    assert len(cb.complex_class.lines) > 0


def test_region_override_parameters():
    cb = ComplexBuilder(xmin=4000, xmax=7000, fe_mode="none", add_balmer_continuum=True)
    cb.make_region(4200, 6800, n_narrow=2, n_broad=1)
    assert isinstance(cb.complex_class, ComplexRegion)
    names = [line.line_name for line in cb.complex_class.lines]
    assert any("balmer" in name for name in names)


def test_fitting_routine_structure():
    cb = ComplexBuilder(xmin=4000, xmax=7000)
    routine = cb._make_fitting_routine(list_num_steps=[100, 100], list_learning_rate=[1e-2, 1e-3])
    assert "step1" in routine["fitting_routine"]
    assert "complex_class" in routine
    assert isinstance(routine["complex_class"], ComplexRegion)


def test_add_host_template():
    cb = ComplexBuilder(xmin=3500, xmax=7500, add_host_miles=True)
    lines = cb.complex_class.lines
    assert any(line.profile == "hostmiles" for line in lines)
