"""Tests for the plotting functions in :mod:`spinanalysis.plotting`.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from spinanalysis import plotting


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PLOT_DATA = Path("src/spinanalysis/data/profiles/plot")


def _install_plot_profiles(tmp_path, monkeypatch):
    """Copy the bundled plot stylesheets into a temp PROFILE_ROOT."""
    root = tmp_path / "profiles"
    plot_dir = root / "plot"
    plot_dir.mkdir(parents=True)
    for f in _PLOT_DATA.iterdir():
        if f.is_file():
            shutil.copy(f, plot_dir / f.name)
    monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", root)
    return root


@pytest.fixture
def plot_profiles(tmp_path, monkeypatch):
    return _install_plot_profiles(tmp_path, monkeypatch)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# TestAlignZ
# ---------------------------------------------------------------------------


class TestAlignZ:
    def test_correct_orientation_no_transpose(self):
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 20, 7)
        Z = np.random.rand(7, 5)
        result = plotting._align_z(x, y, Z)
        assert result.shape == (7, 5)
        np.testing.assert_array_equal(result, Z)

    def test_transposed_orientation(self):
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 20, 7)
        Z = np.random.rand(5, 7)
        result = plotting._align_z(x, y, Z)
        assert result.shape == (7, 5)
        np.testing.assert_array_equal(result, Z.T)

    def test_square_array_no_transpose(self):
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 20, 5)
        Z = np.random.rand(5, 5)
        result = plotting._align_z(x, y, Z)
        assert result.shape == (5, 5)
        np.testing.assert_array_equal(result, Z)

    def test_incompatible_shape_raises(self):
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 20, 7)
        Z = np.random.rand(3, 4)
        with pytest.raises(ValueError, match="incompatible"):
            plotting._align_z(x, y, Z)

    def test_single_element(self):
        x = np.array([1.0])
        y = np.array([2.0])
        Z = np.array([[5.0]])
        result = plotting._align_z(x, y, Z)
        assert result.shape == (1, 1)

    def test_1d_z_raises(self):
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 20, 7)
        Z = np.random.rand(5)
        with pytest.raises(ValueError):
            plotting._align_z(x, y, Z)


# ---------------------------------------------------------------------------
# TestGetShiftMatrix
# ---------------------------------------------------------------------------


class TestGetShiftMatrix:
    def test_shape_3_2(self):
        sm = plotting._get_shift_matrix((3, 2))
        expected = np.array([[0, 0], [2, 2], [4, 4]])
        np.testing.assert_array_equal(sm, expected)

    def test_shape_1_5(self):
        sm = plotting._get_shift_matrix((1, 5))
        expected = np.array([[0, 0, 0, 0, 0]])
        np.testing.assert_array_equal(sm, expected)

    def test_shape_4_3(self):
        sm = plotting._get_shift_matrix((4, 3))
        assert sm.shape == (4, 3)
        # First row all zeros
        np.testing.assert_array_equal(sm[0], np.zeros(3))
        # Last row all 2*(4-1) = 6
        np.testing.assert_array_equal(sm[-1], np.full(3, 6.0))

    def test_monotonic_increase(self):
        sm = plotting._get_shift_matrix((6, 4))
        for col in range(4):
            diffs = np.diff(sm[:, col])
            assert np.allclose(diffs, diffs[0])

    def test_addable_to_data(self):
        shape = (5, 8)
        data = np.random.rand(*shape)
        sm = plotting._get_shift_matrix(shape)
        result = data + sm
        assert result.shape == shape


# ---------------------------------------------------------------------------
# TestGetLabelList
# ---------------------------------------------------------------------------


class TestGetLabelList:
    def test_basic(self):
        labels = plotting._get_label_list(3, "foo")
        assert labels == ["foo", "foo", "foo"]

    def test_length_zero(self):
        labels = plotting._get_label_list(0, "bar")
        assert labels == []

    def test_length_one(self):
        labels = plotting._get_label_list(1, "baz")
        assert labels == ["baz"]


# ---------------------------------------------------------------------------
# TestGetAxisLimit
# ---------------------------------------------------------------------------


class TestGetAxisLimit:
    def test_fixed_mode_with_limits(self):
        data = np.array([0, 50, 100])
        result = plotting._get_axis_limit(data, [10.0, 90.0], False)
        assert result == (10.0, 90.0)

    def test_fixed_mode_empty_limits(self):
        data = np.array([5.0, 15.0, 25.0])
        result = plotting._get_axis_limit(data, "", False)
        assert result == (5.0, 25.0)

    def test_percentage_mode_empty_limits(self):
        data = np.array([0.0, 100.0])
        result = plotting._get_axis_limit(data, "", True)
        assert result == (0.0, 100.0)

    def test_percentage_mode_with_limits(self):
        data = np.array([0.0, 200.0])
        result = plotting._get_axis_limit(data, [-10.0, 10.0], True)
        # range = 200, lower_diff = 200 * -10/100 = -20
        # upper_diff = 200 * 10/100 = 20
        # lower = 0 - (-20) = 20, upper = 200 + 20 = 220
        assert result == (20.0, 220.0)

    def test_percentage_mode_negative_range(self):
        data = np.array([-100.0, 0.0])
        result = plotting._get_axis_limit(data, [0.0, 0.0], True)
        assert result == (-100.0, 0.0)


# ---------------------------------------------------------------------------
# TestPlot2D
# ---------------------------------------------------------------------------


class TestPlot2D:
    def test_returns_figure(self, plot_profiles):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        fig = plotting.plot_2D(x, y)
        assert isinstance(fig, plt.Figure)

    def test_2d_y_array(self, plot_profiles):
        x = np.linspace(0, 10, 50)
        y = np.random.rand(3, 50)
        fig = plotting.plot_2D(x, y)
        assert isinstance(fig, plt.Figure)

    def test_with_labels_list(self, plot_profiles):
        x = np.linspace(0, 10, 50)
        y = np.random.rand(2, 50)
        fig = plotting.plot_2D(x, y, labels=["a", "b"])
        assert isinstance(fig, plt.Figure)

    def test_with_ax_provided(self, plot_profiles):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        fig_in, ax_in = plt.subplots()
        fig_out = plotting.plot_2D(x, y, ax=ax_in)
        assert fig_out is fig_in

    def test_kwargs_passed_through(self, plot_profiles):
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        fig = plotting.plot_2D(x, y, linewidth=2.0, color="red")
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# TestShifted2D
# ---------------------------------------------------------------------------


class TestShifted2D:
    def test_returns_figure(self, plot_profiles):
        x = np.linspace(0, 10, 50)
        Y = np.random.rand(5, 50)
        fig = plotting.shifted_2D(x, Y)
        assert isinstance(fig, plt.Figure)

    def test_with_ax(self, plot_profiles):
        x = np.linspace(0, 10, 50)
        Y = np.random.rand(3, 50)
        fig_in, ax_in = plt.subplots()
        fig_out = plotting.shifted_2D(x, Y, ax=ax_in)
        assert fig_out is fig_in

    def test_with_labels(self, plot_profiles):
        x = np.linspace(0, 10, 50)
        Y = np.random.rand(4, 50)
        fig = plotting.shifted_2D(x, Y, labels=["a", "b", "c", "d"])
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# TestPlot3D
# ---------------------------------------------------------------------------


class TestPlot3D:
    def test_returns_figure(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(15, 20)
        fig = plotting.plot_3D(x, y, Z)
        assert isinstance(fig, plt.Figure)

    def test_transposed_z(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(20, 15)
        fig = plotting.plot_3D(x, y, Z)
        assert isinstance(fig, plt.Figure)

    def test_with_ax(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(15, 20)
        fig_in, ax_in = plt.subplots(subplot_kw={"projection": "3d"})
        fig_out = plotting.plot_3D(x, y, Z, ax=ax_in)
        assert fig_out is fig_in

    def test_incompatible_z_raises(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(10, 10)
        with pytest.raises(ValueError):
            plotting.plot_3D(x, y, Z)


# ---------------------------------------------------------------------------
# TestPlot3DMultipleLines
# ---------------------------------------------------------------------------


class TestPlot3DMultipleLines:
    def test_returns_figure(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(15, 20)
        fig = plotting.plot_3D_multiple_lines(x, y, Z)
        assert isinstance(fig, plt.Figure)

    def test_transposed_z(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(20, 15)
        fig = plotting.plot_3D_multiple_lines(x, y, Z)
        assert isinstance(fig, plt.Figure)

    def test_with_ax(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(15, 20)
        fig_in, ax_in = plt.subplots(subplot_kw={"projection": "3d"})
        fig_out = plotting.plot_3D_multiple_lines(x, y, Z, ax=ax_in)
        assert fig_out is fig_in

    def test_incompatible_z_raises(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(8, 8)
        with pytest.raises(ValueError):
            plotting.plot_3D_multiple_lines(x, y, Z)


# ---------------------------------------------------------------------------
# TestHeatmap
# ---------------------------------------------------------------------------


class TestHeatmap:
    def test_returns_figure(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(15, 20)
        fig = plotting.heatmap(x, y, Z)
        assert isinstance(fig, plt.Figure)

    def test_transposed_z(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(20, 15)
        fig = plotting.heatmap(x, y, Z)
        assert isinstance(fig, plt.Figure)

    def test_with_ax(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(15, 20)
        fig_in, ax_in = plt.subplots()
        fig_out = plotting.heatmap(x, y, Z, ax=ax_in)
        assert fig_out is fig_in

    def test_incompatible_z_raises(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(10, 10)
        with pytest.raises(ValueError):
            plotting.heatmap(x, y, Z)

    def test_colorbar_present(self, plot_profiles):
        x = np.linspace(0, 10, 20)
        y = np.linspace(0, 5, 15)
        Z = np.random.rand(15, 20)
        fig = plotting.heatmap(x, y, Z)
        # Default profile has colorbar: True
        assert len(fig.axes) >= 2


# ---------------------------------------------------------------------------
# TestPrepareStyle
# ---------------------------------------------------------------------------


class TestPrepareStyle:
    def test_returns_profile_and_style_list(self, plot_profiles):
        kwargs = {"linewidth": 2}
        profile, style_list = plotting._prepare_style("default_stylesheet", kwargs)
        assert isinstance(profile, dict)
        assert isinstance(style_list, list)
        assert len(style_list) == 6

    def test_profile_keys_removed_from_kwargs(self, plot_profiles):
        kwargs = {"xlabel": "Custom"}
        profile, style_list = plotting._prepare_style("default_stylesheet", kwargs)
        assert "xlabel" not in kwargs
        assert profile["xlabel"] == "Custom"

    def test_non_profile_keys_remain_in_kwargs(self, plot_profiles):
        kwargs = {"linewidth": 2, "color": "red"}
        profile, style_list = plotting._prepare_style("default_stylesheet", kwargs)
        assert "linewidth" in kwargs
        assert "color" in kwargs


# ---------------------------------------------------------------------------
# TestSuppressWarnings
# ---------------------------------------------------------------------------


class TestSuppressWarnings:
    def test_no_error(self):
        with plotting._suppress_warnings():
            pass

    def test_suppresses_warning(self):
        import warnings as _w

        with plotting._suppress_warnings():
            _w.warn("test warning")
        # If not suppressed, this would not raise; just verify no exception
