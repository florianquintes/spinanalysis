import datetime
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from spinanalysis.epr import (
    Experimental,
    FittingOptions,
    SimulationOptions,
    Spinsystem,
)
from spinanalysis.saving import (
    _get_authors,
    _resolve_output_dir,
    save_plot,
    save_simulation,
    write_out_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def epr_objects():
    return Spinsystem(), Experimental(), SimulationOptions(), FittingOptions()


@pytest.fixture
def fig():
    f, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    yield f
    plt.close(f)


@pytest.fixture
def fig_from_bes3t(test_loading_dir):
    """Create a figure from real BES3T data."""
    from spinanalysis.loading import load_epr_bruker_bes3t

    with pytest.warns(UserWarning, match="No axis data"):
        axis, data = load_epr_bruker_bes3t("cw_xband_100k")
    f, ax = plt.subplots()
    ax.plot(axis[0], data.real)
    yield f
    plt.close(f)


@pytest.fixture
def fig_from_transient(test_loading_dir):
    """Create a figure from real transient data."""
    from spinanalysis.loading import load_epr_ESP_transient

    axis, data = load_epr_ESP_transient("transient_xband")
    f, ax = plt.subplots()
    ax.plot(axis[1], data[:, 0].real)
    yield f
    plt.close(f)


# ---------------------------------------------------------------------------
# TestGetAuthors
# ---------------------------------------------------------------------------


class TestGetAuthors:
    def test_returns_non_empty(self):
        result = _get_authors()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_author_name(self):
        result = _get_authors()
        assert "Florian Quintes" in result

    def test_fallback_on_missing_metadata(self):
        with patch(
            "spinanalysis.saving._pkg_metadata",
            side_effect=Exception("not found"),
        ):
            assert _get_authors() == "spinanalysis"


# ---------------------------------------------------------------------------
# TestResolveOutputDir
# ---------------------------------------------------------------------------


class TestResolveOutputDir:
    def test_explicit_path(self, tmp_path):
        out = _resolve_output_dir(tmp_path / "output")
        assert out == tmp_path / "output"
        assert out.is_dir()

    def test_explicit_path_already_exists(self, tmp_path):
        existing = tmp_path / "existing"
        existing.mkdir()
        out = _resolve_output_dir(existing)
        assert out == existing

    def test_nested_path(self, tmp_path):
        out = _resolve_output_dir(tmp_path / "a" / "b" / "c")
        assert out.is_dir()

    @patch("spinanalysis.saving.datetime")
    def test_auto_naming_first(self, mock_dt, tmp_path, monkeypatch):
        mock_dt.datetime.now.return_value = datetime.datetime(2026, 1, 15)
        mock_dt.datetime.strftime = datetime.datetime.strftime
        monkeypatch.chdir(tmp_path)
        out = _resolve_output_dir(None)
        assert out == tmp_path / "spinanalysis_2026-01-15_1"
        assert out.is_dir()

    @patch("spinanalysis.saving.datetime")
    def test_auto_naming_skips_existing(self, mock_dt, tmp_path, monkeypatch):
        mock_dt.datetime.now.return_value = datetime.datetime(2026, 1, 15)
        mock_dt.datetime.strftime = datetime.datetime.strftime
        monkeypatch.chdir(tmp_path)
        (tmp_path / "spinanalysis_2026-01-15_1").mkdir()
        (tmp_path / "spinanalysis_2026-01-15_2").mkdir()
        out = _resolve_output_dir(None)
        assert out == tmp_path / "spinanalysis_2026-01-15_3"
        assert out.is_dir()


# ---------------------------------------------------------------------------
# TestSavePlot
# ---------------------------------------------------------------------------


class TestSavePlot:
    def test_single_figure(self, fig, tmp_path):
        save_plot("test_plot.png", fig, path=tmp_path)
        assert (tmp_path / "test_plot.png").is_file()

    def test_format_override(self, fig, tmp_path):
        save_plot("test_plot.png", fig, path=tmp_path, format="pdf")
        assert (tmp_path / "test_plot.pdf").is_file()
        assert not (tmp_path / "test_plot.png").exists()

    def test_multiple_figures(self, tmp_path):
        figs = []
        for i in range(3):
            f, ax = plt.subplots()
            ax.plot([1, 2, 3])
            figs.append(f)
        try:
            save_plot("multi.png", *figs, path=tmp_path)
        finally:
            for f in figs:
                plt.close(f)
        assert (tmp_path / "multi_1.png").is_file()
        assert (tmp_path / "multi_2.png").is_file()
        assert (tmp_path / "multi_3.png").is_file()

    def test_figures_as_list(self, fig, tmp_path):
        save_plot("list_plot.png", [fig], path=tmp_path)
        assert (tmp_path / "list_plot.png").is_file()

    def test_no_figures_raises(self, tmp_path):
        with pytest.raises(ValueError, match="At least one figure"):
            save_plot("empty.png", path=tmp_path)

    def test_plot_from_bes3t_data(self, fig_from_bes3t, tmp_path):
        save_plot("bes3t_plot.png", fig_from_bes3t, path=tmp_path)
        assert (tmp_path / "bes3t_plot.png").is_file()

    def test_plot_from_transient_data(self, fig_from_transient, tmp_path):
        save_plot("transient_plot.png", fig_from_transient, path=tmp_path)
        assert (tmp_path / "transient_plot.png").is_file()


# ---------------------------------------------------------------------------
# TestSaveSimulation
# ---------------------------------------------------------------------------


class TestSaveSimulation:
    def test_2d_data(self, tmp_path):
        x = np.linspace(0, 100, 50)
        intensity = np.random.rand(50)
        save_simulation("test_sim", x, intensity, path=tmp_path)
        assert (tmp_path / "test_sim" / "x_axis.txt").is_file()
        assert (tmp_path / "test_sim" / "intensity.txt").is_file()
        assert not (tmp_path / "test_sim" / "y_axis.txt").exists()

    def test_3d_data(self, tmp_path):
        x = np.linspace(0, 100, 50)
        y = np.linspace(0, 50, 30)
        intensity = np.random.rand(50, 30)
        save_simulation("test_sim3d", x, y, intensity, path=tmp_path)
        assert (tmp_path / "test_sim3d" / "x_axis.txt").is_file()
        assert (tmp_path / "test_sim3d" / "y_axis.txt").is_file()
        assert (tmp_path / "test_sim3d" / "intensity.txt").is_file()

    def test_strips_txt_extension(self, tmp_path):
        x = np.linspace(0, 10, 5)
        intensity = np.ones(5)
        save_simulation("data.txt", x, intensity, path=tmp_path)
        assert (tmp_path / "data" / "x_axis.txt").is_file()
        assert not (tmp_path / "data.txt").exists()

    def test_wrong_array_count(self, tmp_path):
        x = np.ones(5)
        with pytest.raises(ValueError, match="Need 2D or 3D"):
            save_simulation("bad", x, path=tmp_path)
        with pytest.raises(ValueError, match="Need 2D or 3D"):
            save_simulation("bad", x, x, x, x, path=tmp_path)

    def test_roundtrip_2d(self, tmp_path):
        x = np.linspace(0, 100, 50)
        intensity = np.random.rand(50) + 1j * np.random.rand(50)
        save_simulation("roundtrip", x, intensity, path=tmp_path)
        x_loaded = np.loadtxt(str(tmp_path / "roundtrip" / "x_axis.txt"))
        assert np.allclose(x_loaded, x)

    def test_custom_path_created(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        x = np.ones(3)
        intensity = np.ones(3)
        save_simulation("sim", x, intensity, path=nested)
        assert (nested / "sim" / "intensity.txt").is_file()


# ---------------------------------------------------------------------------
# TestWriteOutFile
# ---------------------------------------------------------------------------


class TestWriteOutFile:
    def test_simulation_mode(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        out = write_out_file(Sys, Exp, SimOpt, path=tmp_path)
        assert out == tmp_path / "teacups_result.spinanalysis"
        assert out.is_file()

    def test_fit_mode(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        FitOpt.routine = "scipy_optimize"
        out = write_out_file(Sys, Exp, SimOpt, FitOpt, path=tmp_path)
        assert out == tmp_path / "scipy_optimize_result.spinanalysis"
        assert out.is_file()

    def test_current_best(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        FitOpt.routine = "scipy_optimize"
        out = write_out_file(Sys, Exp, SimOpt, FitOpt, current_best=True, path=tmp_path)
        assert "current_best" in out.name
        assert out.is_file()

    def test_result_default(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        out = write_out_file(Sys, Exp, SimOpt, path=tmp_path)
        assert "result" in out.name

    def test_file_contains_section_headers(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        out = write_out_file(Sys, Exp, SimOpt, path=tmp_path)
        content = out.read_text()
        assert "SPINSYSTEM" in content
        assert "EXPERIMENTAL" in content
        assert "SIMULATION OPTIONS" in content

    def test_fit_mode_contains_fitting_section(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        FitOpt.routine = "scipy_optimize"
        out = write_out_file(Sys, Exp, SimOpt, FitOpt, path=tmp_path)
        content = out.read_text()
        assert "FITTING OPTIONS" in content

    def test_sim_mode_no_fitting_section(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        out = write_out_file(Sys, Exp, SimOpt, path=tmp_path)
        content = out.read_text()
        assert "FITTING OPTIONS" not in content

    def test_file_contains_author(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        out = write_out_file(Sys, Exp, SimOpt, path=tmp_path)
        content = out.read_text()
        assert "FLORIAN QUINTES" in content

    def test_file_contains_knots_not_grid_points(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        out = write_out_file(Sys, Exp, SimOpt, path=tmp_path)
        content = out.read_text()
        assert "knots" in content
        assert "grid_points" not in content

    def test_none_fitting_fields_skipped(self, epr_objects, tmp_path):
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        FitOpt.routine = "scipy_optimize"
        out = write_out_file(Sys, Exp, SimOpt, FitOpt, path=tmp_path)
        content = out.read_text()
        fit_section = content.split("FITTING OPTIONS")[1]
        assert "routine" in fit_section
        assert "method" not in fit_section
        assert "None" not in fit_section

    @patch("spinanalysis.saving.datetime")
    def test_auto_path(self, mock_dt, epr_objects, tmp_path, monkeypatch):
        mock_dt.datetime.now.return_value = datetime.datetime(2026, 1, 15)
        mock_dt.datetime.strftime = datetime.datetime.strftime
        monkeypatch.chdir(tmp_path)
        Sys, Exp, SimOpt, FitOpt = epr_objects
        SimOpt.routine = "teacups"
        out = write_out_file(Sys, Exp, SimOpt)
        assert out.parent == tmp_path / "spinanalysis_2026-01-15_1"
        assert out.is_file()
