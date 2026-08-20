#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render EPR spectra as 2D, 3D, heatmap, and shifted-line plots.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings
from contextlib import contextmanager
from spinanalysis import profiles


@contextmanager
def _suppress_warnings():
    """Suppress warnings in a context manager."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _prepare_style(mpl_stylesheet: str, kwargs: dict) -> tuple[dict, list]:
    """Load the plot profile and build the style list.

    Profile keys that also appear in *kwargs* are moved from *kwargs*
    into the returned profile dict, so that the caller's *kwargs*
    only contains keys meant for the matplotlib plotting function.

    Parameters
    ----------
    mpl_stylesheet : str
        Name of the matplotlib style sheet / plotting profile.
    kwargs : dict
        Keyword arguments from the caller; modified in place.

    Returns
    -------
    profile : dict
        Plotting profile with kwargs overrides applied.
    style_list : list
        List of style paths suitable for ``plt.style.context``.

    """
    profile = profiles.load_plot_profile(mpl_stylesheet)
    profile.update((k, kwargs[k]) for k in profile.keys() & kwargs.keys())
    for k in profile.keys() & kwargs.keys():
        kwargs.pop(k)
    style = _get_style_path(mpl_stylesheet)
    style_list = _load_default_styles()
    style_list.append(style)
    return profile, style_list


def plot_2D(
    x: np.ndarray,
    y: np.ndarray,
    mpl_stylesheet: str = "default_stylesheet",
    labels: list[str] | str = "no_label",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot the given y value(s) against the given x array.

    Using matplotlib.pylab.plot(). The plot can be configured via plot
    profiles.

    Parameters
    ----------
    x : np.ndarray
        Array with values for the x axis.
    y : np.ndarray
        1D-Array or 2D-Array with values for y axis.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    labels : list[str] or str, optional
        List of labels for the legend. If only one label is given, all labels
        will be the same. The default is 'no_label'.
    ax : plt.Axes, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : plt.Figure
        Figure object of matplotlib.pylab.

    """
    profile, style_list = _prepare_style(mpl_stylesheet, kwargs)

    with _suppress_warnings():
        with plt.style.context(style_list):
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            # [LABELS]
            if y.ndim > 1 and not isinstance(labels, list):
                labels = _get_label_list(y.shape[0], labels)

            # [PLOT DATA]
            if y.ndim == 1:
                ax.plot(x, y, label=labels, **kwargs)
            else:
                for y_set in range(len(y)):
                    ax.plot(x, y[y_set], label=labels[y_set], **kwargs)

            # [Axes]
            _set_axis(ax, profile, x, y)
            _set_figure(ax, profile, mpl_stylesheet=mpl_stylesheet)

    return fig


def shifted_2D(
    x: np.ndarray,
    Y: np.ndarray,
    mpl_stylesheet: str = "default_stylesheet",
    labels: list[str] | str = "no_label",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot multiples lines in 2D, shifted vertically.

    The plot can be configured via plot profiles.

    Parameters
    ----------
    x : np.ndarray
        Array with values for the x axis.
    Y : np.ndarray
        2D-Array with values for y axis.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    labels : list[str] or str, optional
        List of labels for the legend. If only one label is given, all labels
        will be the same. The default is 'no_label'.
    ax : plt.Axes, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : plt.Figure
        Figure object of matplotlib.pylab.

    """
    profile, style_list = _prepare_style(mpl_stylesheet, kwargs)

    with _suppress_warnings():
        with plt.style.context(style_list):
            # [NORMALIZE DATA]
            Y = Y / np.max(np.abs(Y))

            # [SHIFT DATA]
            shift_matrix = _get_shift_matrix(Y.shape)
            Y += shift_matrix

            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            # [LABELS]
            if not isinstance(labels, list):
                labels = _get_label_list(Y.shape[0], labels)

            # [PLOT DATA]
            for y_set in range(Y.shape[0]):
                ax.plot(x, Y[y_set], label=labels[y_set], **kwargs)

            # [AXES]
            _set_axis(ax, profile, x, Y, y_axis=False)
            _set_figure(ax, profile, mpl_stylesheet=mpl_stylesheet)

    return fig


def plot_3D(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    mpl_stylesheet: str = "default_stylesheet",
    labels: str = "no_label",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot 2D Data in 3D using matplotlib.pylab.plot_surface().

    The plot can be configured via plot profiles.

    Parameters
    ----------
    x : np.ndarray
        Array with values for the x axis.
    y : np.ndarray
        Array with values for the y axis.
    Z : np.ndarray
        2D-Array with intensities.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    labels : str, optional
        At the moment no function. The default is 'no_label'.
    ax : plt.Axes, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : plt.Figure
        Figure object of matplotlib.pylab.

    """
    profile, style_list = _prepare_style(mpl_stylesheet, kwargs)

    kwargs_ = {"cmap": "coolwarm", "antialiased": True, "linewidth": 0}
    kwargs_.update(kwargs)

    with _suppress_warnings():
        with plt.style.context(style_list):
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            else:
                fig = ax.figure

            # [GRID]
            X, Y = np.meshgrid(x, y)

            # [PLOT DATA]
            Z = _align_z(x, y, Z)
            ax.plot_surface(X, Y, Z, **kwargs_)

            # [AXES]
            _set_axis(ax, profile, X, Y, Z)
            _set_figure(ax, profile, mpl_stylesheet=mpl_stylesheet)

    return fig


def plot_3D_multiple_lines(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    mpl_stylesheet: str = "default_stylesheet",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot 2D Data in 3D using matplotlib.pylab.plot().

    Each y trace as a single line plot. The plot can be configured via plot
    profiles.

    Parameters
    ----------
    x : np.ndarray
        Array with values for the x axis.
    y : np.ndarray
        Array with values for the y axis.
    Z : np.ndarray
        2D-Array with intensities.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    ax : plt.Axes, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : plt.Figure
        Figure object of matplotlib.pylab.

    """
    profile, style_list = _prepare_style(mpl_stylesheet, kwargs)

    with _suppress_warnings():
        with plt.style.context(style_list):
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            else:
                fig = ax.figure

            # [PLOT DATA]
            Z = _align_z(x, y, Z)
            X = np.ones((len(y), len(x))) * x
            X = X.T
            for dataset in range(Z.shape[1]):
                ax.plot(X[dataset], y, Z[:, dataset], **kwargs)

            # [AXES]
            _set_axis(ax, profile, x, y, Z)
            _set_figure(ax, profile, mpl_stylesheet=mpl_stylesheet)

    return fig


def heatmap(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray,
    mpl_stylesheet: str = "default_stylesheet",
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot 2D Data as a heatmap using matplotlib.pylab.pcolormesh().

    The plot can be configured via plot profiles.

    Parameters
    ----------
    x : np.ndarray
        Array with values for the x axis.
    y : np.ndarray
        Array with values for the y axis.
    Z : np.ndarray
        2D-Array with intensities.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    ax : plt.Axes, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : plt.Figure
        Figure object of matplotlib.pylab.

    """
    profile, style_list = _prepare_style(mpl_stylesheet, kwargs)

    kwargs_ = {"cmap": "RdBu", "shading": "auto"}
    kwargs_.update(kwargs)

    with _suppress_warnings():
        with plt.style.context(style_list):
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.figure

            # [GRID]
            X, Y = np.meshgrid(x, y)

            # [PLOT DATA]
            Z = _align_z(x, y, Z)
            c = ax.pcolormesh(X, Y, Z, **kwargs_)

            # [PLOT COLORBAR]
            if profile["colorbar"]:
                fig.colorbar(c, ax=ax)

            # [AXES]
            _set_axis(ax, profile, X, Y)
            _set_figure(ax, profile, mpl_stylesheet=mpl_stylesheet)

    return fig


def _align_z(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Transpose Z if needed so that Z.shape == (len(y), len(x)).

    ``np.meshgrid(x, y)`` returns arrays with shape ``(len(y), len(x))``,
    so matplotlib plotting functions expect Z to match this orientation.
    If Z has the transposed shape ``(len(x), len(y))`` it is returned
    transposed.  Any other shape raises ``ValueError``.

    Parameters
    ----------
    x : np.ndarray
        Array with values for the x axis.
    y : np.ndarray
        Array with values for the y axis.
    Z : np.ndarray
        2D-Array with intensities.

    Returns
    -------
    np.ndarray
        Z oriented as (len(y), len(x)).

    """
    nx, ny = len(x), len(y)
    if Z.shape == (ny, nx):
        return Z
    if Z.shape == (nx, ny):
        return Z.T
    raise ValueError(
        "Z shape {} is incompatible with x (len={}) and y (len={}); "
        "expected ({}, {}) or ({}, {})".format(Z.shape, nx, ny, ny, nx, nx, ny)
    )


def _get_shift_matrix(shape: tuple[int, int]) -> np.ndarray:
    """
    Create a matrix to shift the single lines vertically.

    Needed for shifted_2D().

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the data which will be shifted. Use np.array.shape to get the
        shape from your data.

    Returns
    -------
    shift_matrix : np.ndarray
        Matrix to shift your data vertically by adding the shift matrix.
        E. g. : structure of the shift matrix for shape (3, 2):
        ::

            [[0, 0],
             [2, 2],
             [4, 4]]

    """
    shift_matrix = 2 * np.linspace(
        0, np.full((1, shape[1]), (shape[0] - 1))[0], shape[0]
    )

    return shift_matrix


def _set_axis(
    ax: plt.Axes, figure_par: dict, *axes: np.ndarray, y_axis: bool = True
) -> plt.Axes:
    """
    Set up all axis for a plot created by matplotlib.

    Parameters
    ----------
    ax : plt.Axes
        Axis object of matplotlib.pylab.
    figure_par : dict
        Dictionary containing all settings for the figure.
    *axes : np.ndarray
        All axes used for the plot. Used to determine the dimension an set up
        the z axis if needed.
    y_axis : bool, optional
        If a y axis is needed. If False, all y ticks will be removed. The
        default is True.

    Returns
    -------
    ax : plt.Axes
        Modified axis object of matplotlib.pylab.

    """
    ax.set_xlabel(figure_par["xlabel"])
    ax.set_ylabel(figure_par["ylabel"])
    ax.set_xlim(
        _get_axis_limit(axes[0], figure_par["xlim"], figure_par["percentage_mode"])
    )
    if y_axis:
        ax.set_ylim(
            _get_axis_limit(axes[1], figure_par["ylim"], figure_par["percentage_mode"])
        )
    else:
        ax.set_yticks([])

    if len(axes) == 3:
        ax.set_zlabel(figure_par["zlabel"])
        ax.set_zlim(
            _get_axis_limit(axes[2], figure_par["zlim"], figure_par["percentage_mode"])
        )

    return ax


def _set_figure(ax: plt.Axes, figure_par: dict, mpl_stylesheet: str = None) -> plt.Axes:
    """
    Set up the figure of a plot created by matplotlib.

    Parameters
    ----------
    ax : plt.Axes
        Axis object of matplotlib.pylab.
    figure_par : dict
        Dictionary containing all settings for the figure.
    mpl_stylesheet : str, optional
        Name of the stylesheet which should be used instead of a plotting
        profile. If no filename is given, the settings from the given plotting
        profile will be used. The default is None.

    Returns
    -------
    ax : plt.Axes
        Modified axis object of matplotlib.pylab.

    """
    if figure_par["show_title"]:
        ax.set_title(figure_par["title"])
    if figure_par["legend"]:
        ax.legend()

    return ax


def _get_style_path(mpl_stylesheet: str = None) -> str:
    """
    Get the path to the chosen mpl stylesheet.

    Parameters
    ----------
    mpl_stylesheet : str, optional
        Name of the mpl stylesheet. The default is None.

    Returns
    -------
    str
        Path to the chosen stylesheet.

    """
    if mpl_stylesheet is None:
        style = str(profiles.PROFILE_ROOT / "plot" / "default_stylesheet")
    elif mpl_stylesheet in plt.style.available:
        style = mpl_stylesheet
    else:
        style = str(profiles.PROFILE_ROOT / "plot" / mpl_stylesheet)

    return style


def _load_default_styles() -> list:
    """
    Load the paths for the default matplotlib styles.

    Returns
    -------
    default_styles : list
        Paths for the default style sheets.

    """
    default_styles = []
    for def_style in ["axis", "figure", "plot", "line", "text"]:
        def_style = def_style + "_style"
        default_styles.append(_get_style_path(def_style))

    return default_styles


def _get_label_list(length: int, label: str) -> list[str]:
    """
    Get a list with length times the label.

    Parameters
    ----------
    length : int
        Number of labels.
    label : str
        String which will be multiplied.

    Returns
    -------
    list[str]
        Length times label as a list. E. g.: ['label', 'label', 'label'].

    """
    labels = [label] * length

    return labels


def _get_axis_limit(
    data_axis: np.ndarray, ax_lim: list, percentage_mode: bool
) -> tuple[float, float]:
    """
    Get the plot limits for a given data axis.

    If no limit is specified, the minimal and maximal value of the given
    dataset will be taken.

    Parameters
    ----------
    data_axis : np.ndarray
        Data vector for an axis.
    ax_lim : list
        Limits for the data vector taken from a profile.
    percentage_mode : bool
        If True, the ax.lim will be handled as a percentage in context to the
        lowest and highest values with respect to the given axis.
        E. g.: x-vector with values [0, 200], ax_lim: [-2., 2.] -> [4, 204].

    Returns
    -------
    tuple[float, float]
        Limits for plotting.

    """
    if percentage_mode:
        if ax_lim == "":
            min_val = data_axis.min()
            max_val = data_axis.max()
            limits = [min_val, max_val]
        else:
            value_range = data_axis.max() - data_axis.min()
            lower_diff = value_range * ax_lim[0] / 100
            upper_diff = value_range * ax_lim[1] / 100
            lower_border = data_axis.min() - lower_diff
            upper_border = data_axis.max() + upper_diff
            limits = [lower_border, upper_border]
    else:
        if ax_lim == "":
            min_val = data_axis.min()
            max_val = data_axis.max()
            limits = [min_val, max_val]
        else:
            limits = ax_lim

    return limits[0], limits[1]
