"""Tests for :mod:`spinanalysis.core` simulate and optimize workflows.

The external simulation packages (``mkl``, ``genetic_radpair``, ``oop_eseem``,
``teacups``, ``static_radical_pair``, ``PySpin``) are mocked via
``sys.modules`` so that ``spinanalysis.core`` and
``spinanalysis._interface_handler`` can be imported in a clean environment.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Pre-populate sys.modules with mock external dependencies.
# This must happen *before* importing spinanalysis.core or
# spinanalysis._interface_handler, both of which import these packages.
# ---------------------------------------------------------------------------
_mock_modules = {
    "mkl": MagicMock(),
    "genetic_radpair": MagicMock(),
    "genetic_radpair.genetic_classes": MagicMock(),
    "oop_eseem": MagicMock(),
    "oop_eseem.opossum": MagicMock(),
    "teacups": MagicMock(),
    "teacups.simulations": MagicMock(),
    "static_radical_pair": MagicMock(),
    "static_radical_pair.radpair": MagicMock(),
    "PySpin": MagicMock(),
    "PySpin.plotter": MagicMock(),
}
for _name, _mod in _mock_modules.items():
    sys.modules.setdefault(_name, _mod)

# Pre-import _interface_handler so it is available for patching.  Its
# top-level external imports resolve to the mock modules above.
import spinanalysis._interface_handler  # noqa: E402, F401
from spinanalysis.core import optimize, simulate  # noqa: E402
from spinanalysis.epr import (  # noqa: E402
    Experimental,
    FittingOptions,
    SimulationOptions,
    Spinsystem,
    Variation,
)


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset all mock modules before each test."""
    for mod in _mock_modules.values():
        mod.reset_mock()
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_objects(routine=""):
    """Return ``(Sys, Exp, SimOpt)`` with sensible defaults for simulate."""
    return (
        Spinsystem(),
        Experimental(),
        SimulationOptions(routine=routine),
    )


def _default_opt_objects(routine=""):
    """Return ``(Sys, Exp, SimOpt, FitOpt, Var)`` for optimize."""
    return (
        Spinsystem(),
        Experimental(),
        SimulationOptions(),
        FittingOptions(routine=routine),
        Variation(),
    )


# ---------------------------------------------------------------------------
# simulate – dispatch
# ---------------------------------------------------------------------------


class TestSimulateDispatch:
    """Each routine calls the correct external function with the right args."""

    def test_static_radpair(self):
        mock_fn = sys.modules["static_radical_pair.radpair"].do_simulation_multicore
        mock_fn.return_value = np.array([1.0, 2.0, 3.0])
        sys_obj, exp, simopt = _default_objects("static_radpair")
        simulate(sys_obj, exp, simopt)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt)

    def test_teacups(self):
        mock_fn = sys.modules["teacups.simulations"].teacups
        mock_fn.return_value = np.array([0.5, 1.0, 0.5])
        sys_obj, exp, simopt = _default_objects("teacups")
        simopt.eigval_mode = False
        simulate(sys_obj, exp, simopt)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt)

    def test_opossum(self):
        mock_fn = sys.modules["oop_eseem.opossum"].oop_eseem
        mock_fn.return_value = np.array([1.0, -1.0])
        sys_obj, exp, simopt = _default_objects("opossum")
        simulate(sys_obj, exp, simopt)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt)

    def test_didelphis(self):
        mock_fn = sys.modules["oop_eseem.opossum"].oop_eseem_distribution
        mock_fn.return_value = np.array([2.0, 4.0])
        sys_obj, exp, simopt = _default_objects("didelphis")
        simulate(sys_obj, exp, simopt)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt)

    def test_didelphis_tikhonov(self):
        mock_fn = sys.modules["oop_eseem.opossum"].oop_eseem_distance_distribution
        mock_fn.return_value = np.array([3.0, 6.0])
        sys_obj, exp, simopt = _default_objects("didelphis_tikhonov")
        simulate(sys_obj, exp, simopt)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt)

    def test_unknown_routine_raises(self):
        sys_obj, exp, simopt = _default_objects("nonexistent")
        with pytest.raises(ValueError, match="Can't find a routine named"):
            simulate(sys_obj, exp, simopt)

    def test_routine_case_insensitive(self):
        mock_fn = sys.modules["static_radical_pair.radpair"].do_simulation_multicore
        mock_fn.return_value = np.array([1.0])
        sys_obj, exp, simopt = _default_objects("STATIC_RADPAIR")
        simulate(sys_obj, exp, simopt)
        mock_fn.assert_called_once()


# ---------------------------------------------------------------------------
# simulate – teacups eigval_mode branch
# ---------------------------------------------------------------------------


class TestSimulateTeacupsBranch:
    """The teacups routine branches on ``SimOpt.eigval_mode``."""

    def test_eigval_mode_true_returns_ones(self):
        mock_fn = sys.modules["teacups.simulations"].teacups
        sys_obj, exp, simopt = _default_objects("teacups")
        simopt.eigval_mode = True
        exp.t_points = 5
        exp.B_z = np.linspace(300, 310, 10)
        result = simulate(sys_obj, exp, simopt)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt)
        assert result.shape == (5, 10)
        assert np.all(result == 1.0)

    def test_eigval_mode_false_calls_teacups_and_returns_its_value(self):
        mock_fn = sys.modules["teacups.simulations"].teacups
        mock_fn.return_value = np.array([2.0, 4.0, 8.0])
        sys_obj, exp, simopt = _default_objects("teacups")
        simopt.eigval_mode = False
        result = simulate(sys_obj, exp, simopt)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt)
        assert np.allclose(result, np.array([0.25, 0.5, 1.0]))

    def test_eigval_mode_true_does_not_set_fitting_mode(self):
        sys.modules["teacups.simulations"].teacups
        sys_obj, exp, simopt = _default_objects("teacups")
        simopt.eigval_mode = True
        simulate(sys_obj, exp, simopt)
        assert simopt.mode == "simulation"


# ---------------------------------------------------------------------------
# simulate – normalization
# ---------------------------------------------------------------------------


class TestSimulateNormalization:
    """The returned spectrum is normalized to a maximum absolute value of 1."""

    def test_positive_spectrum_normalized(self):
        mock_fn = sys.modules["static_radical_pair.radpair"].do_simulation_multicore
        mock_fn.return_value = np.array([1.0, 2.0, 4.0, 2.0, 1.0])
        sys_obj, exp, simopt = _default_objects("static_radpair")
        result = simulate(sys_obj, exp, simopt)
        assert np.isclose(abs(result).max(), 1.0)

    def test_negative_spectrum_normalized(self):
        mock_fn = sys.modules["static_radical_pair.radpair"].do_simulation_multicore
        mock_fn.return_value = np.array([-1.0, -3.0, -6.0, -3.0, -1.0])
        sys_obj, exp, simopt = _default_objects("static_radpair")
        result = simulate(sys_obj, exp, simopt)
        assert np.isclose(abs(result).max(), 1.0)
        assert np.isclose(result.min(), -1.0)

    def test_all_zero_spectrum_no_nan(self):
        mock_fn = sys.modules["static_radical_pair.radpair"].do_simulation_multicore
        mock_fn.return_value = np.zeros(100)
        sys_obj, exp, simopt = _default_objects("static_radpair")
        result = simulate(sys_obj, exp, simopt)
        assert not np.any(np.isnan(result))
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# simulate – side effects
# ---------------------------------------------------------------------------


class TestSimulateSideEffects:
    """``simulate`` mutates ``SimOpt.mode`` and ``Exp.spec_sim``."""

    def test_mode_set_to_simulation(self):
        sys.modules[
            "static_radical_pair.radpair"
        ].do_simulation_multicore.return_value = np.array([1.0])
        sys_obj, exp, simopt = _default_objects("static_radpair")
        simulate(sys_obj, exp, simopt)
        assert simopt.mode == "simulation"

    def test_spec_sim_set_to_returned_spectrum(self):
        mock_fn = sys.modules["static_radical_pair.radpair"].do_simulation_multicore
        mock_fn.return_value = np.array([3.0, 6.0, 3.0])
        sys_obj, exp, simopt = _default_objects("static_radpair")
        result = simulate(sys_obj, exp, simopt)
        assert np.array_equal(exp.spec_sim, result)

    def test_teacups_non_eigval_sets_mode_to_fitting(self):
        sys.modules["teacups.simulations"].teacups.return_value = np.array([1.0])
        sys_obj, exp, simopt = _default_objects("teacups")
        simopt.eigval_mode = False
        simulate(sys_obj, exp, simopt)
        assert simopt.mode == "fitting"


# ---------------------------------------------------------------------------
# simulate – return type
# ---------------------------------------------------------------------------


class TestSimulateReturnType:
    def test_returns_ndarray(self):
        sys.modules[
            "static_radical_pair.radpair"
        ].do_simulation_multicore.return_value = np.array([1.0, 2.0])
        sys_obj, exp, simopt = _default_objects("static_radpair")
        result = simulate(sys_obj, exp, simopt)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# optimize – dispatch
# ---------------------------------------------------------------------------


class TestOptimizeDispatch:
    """Each optimization routine calls the correct function with the right args."""

    def test_genetic(self):
        mock_gr = sys.modules["genetic_radpair.genetic_classes"].Genetic_Radpair
        mock_gr.return_value.best_spinsystem = "best_sys"
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects("genetic")
        result = optimize(sys_obj, exp, simopt, fitopt, var)
        mock_gr.assert_called_once_with(sys_obj, exp, simopt, fitopt, var)
        assert result == "best_sys"

    @patch("spinanalysis._interface_handler.dualannealing")
    def test_dual_annealing(self, mock_fn):
        mock_fn.return_value = "best_sys"
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects("dual_annealing")
        result = optimize(sys_obj, exp, simopt, fitopt, var)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt, fitopt, var)
        assert result == "best_sys"

    @patch("spinanalysis._interface_handler.shgo")
    def test_shgo(self, mock_fn):
        mock_fn.return_value = "best_sys"
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects("shgo")
        result = optimize(sys_obj, exp, simopt, fitopt, var)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt, fitopt, var)
        assert result == "best_sys"

    @patch("spinanalysis._interface_handler.differential_evolution")
    def test_differential_evolution(self, mock_fn):
        mock_fn.return_value = "best_sys"
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects(
            "differential_evolution"
        )
        result = optimize(sys_obj, exp, simopt, fitopt, var)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt, fitopt, var)
        assert result == "best_sys"

    @patch("spinanalysis._interface_handler.basinhopping")
    def test_basinhopping(self, mock_fn):
        mock_fn.return_value = "best_sys"
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects("basinhopping")
        result = optimize(sys_obj, exp, simopt, fitopt, var)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt, fitopt, var)
        assert result == "best_sys"

    @patch("spinanalysis._interface_handler.least_squares")
    def test_least_squares(self, mock_fn):
        mock_fn.return_value = "best_sys"
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects("least_squares")
        result = optimize(sys_obj, exp, simopt, fitopt, var)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt, fitopt, var)
        assert result == "best_sys"

    @patch("spinanalysis._interface_handler.minimize")
    def test_minimize(self, mock_fn):
        mock_fn.return_value = "best_sys"
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects("minimize")
        result = optimize(sys_obj, exp, simopt, fitopt, var)
        mock_fn.assert_called_once_with(sys_obj, exp, simopt, fitopt, var)
        assert result == "best_sys"

    def test_unknown_routine_raises(self):
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects("nonexistent")
        with pytest.raises(
            ValueError, match="Can't find an optimization routine named"
        ):
            optimize(sys_obj, exp, simopt, fitopt, var)

    def test_routine_case_insensitive(self):
        mock_gr = sys.modules["genetic_radpair.genetic_classes"].Genetic_Radpair
        mock_gr.return_value.best_spinsystem = "best_sys"
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects("GENETIC")
        result = optimize(sys_obj, exp, simopt, fitopt, var)
        mock_gr.assert_called_once()
        assert result == "best_sys"

    @patch("spinanalysis._interface_handler.minimize")
    def test_mode_set_to_fitting(self, mock_fn):
        mock_fn.return_value = "best_sys"
        sys_obj, exp, simopt, fitopt, var = _default_opt_objects("minimize")
        optimize(sys_obj, exp, simopt, fitopt, var)
        assert simopt.mode == "fitting"
