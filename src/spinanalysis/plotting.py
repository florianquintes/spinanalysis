#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from spinanalysis import profiles


class HiddenPrints:
    """Supress Error Messages in a context manager."""

    def __enter__(self):
        """Deactivate the error stream when entering the context manager."""
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Activate the error stream when leaving the context manager."""
        sys.stderr.close()
        sys.stderr = self._original_stderr


def plot_2D(
    x: np.array,
    y: np.array,
    mpl_stylesheet: str = "default_stylesheet",
    labels: list[str] = "no_label",
    ax: object = None,
    **kwargs,
) -> object:
    """
    Plot the given y value(s) against the given x array.

    Using matplotlib.pylab.plot(). The plot can be configured via plot
    profiles.

    Parameters
    ----------
    x : np.array
        Array with values for the x axis.
    y : np.array
        1D-Array or 2D-Array with values for y axis.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    labels : list[str], optional
        List of labels for the legend. If only one label is given, all labels
        will be the same. The default is 'no_label'.
    ax : object, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : object
        Figure object of matplotlib.pylab.

    """
    profile = profiles.load_plot_profile(mpl_stylesheet)
    profile.update((k, kwargs[k]) for k in profile.keys() & kwargs.keys())
    for k in profile.keys() & kwargs.keys():
        kwargs.pop(k)
    style = _get_style_path(mpl_stylesheet)
    default_styles = _load_default_styles()
    default_styles.append(style)

    with HiddenPrints():
        with plt.style.context(default_styles):
            if ax is None:
                # [NEW FIGURE]
                fig, ax = plt.subplots()
            else:
                fig, _ = plt.subplots()

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
            _set_figure(ax, profile, mplstylesheet=mpl_stylesheet)

    return fig


def shifted_2D(
    x: np.array,
    Y: np.array,
    mpl_stylesheet: str = "default_stylesheet",
    labels: list[str] = "no_label",
    ax: object = None,
    **kwargs,
) -> object:
    """
    Plot multiples lines in 2D, shifted vertically.

    The plot can be configured via plot profiles.

    Parameters
    ----------
    x : np.array
        Array with values for the x axis.
    Y : np.array
        2D-Array with values for y axis.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    labels : list[str], optional
        List of labels for the legend. If only one label is given, all labels
        will be the same. The default is 'no_label'.
    ax : object, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : object
        Figure object of matplotlib.pylab.

    """
    profile = profiles.load_plot_profile(mpl_stylesheet)
    profile.update((k, kwargs[k]) for k in profile.keys() & kwargs.keys())
    for k in profile.keys() & kwargs.keys():
        kwargs.pop(k)
    style = _get_style_path(mpl_stylesheet)
    default_styles = _load_default_styles()
    default_styles.append(style)

    with HiddenPrints():
        with plt.style.context(default_styles):
            # [NORMALIZE DATA]
            Y = Y / np.max(np.abs(Y))

            # [SHIFT DATA]
            shift_matrix = _get_shift_matrix(Y.shape)
            Y += shift_matrix

            if ax is None:
                # [NEW FIGURE]
                fig, ax = plt.subplots()
            else:
                fig = None

            # [LABELS]
            if not isinstance(labels, list):
                labels = _get_label_list(Y.shape[0], labels)

            # [PLOT DATA]
            for y_set in range(Y.shape[0]):
                ax.plot(x, Y[y_set], label=labels[y_set], **kwargs)

            # [AXES]
            _set_axis(ax, profile, x, Y, y_axis=False)
            _set_figure(ax, profile, mplstylesheet=mpl_stylesheet)

    return fig


def plot_3D(
    x: np.array,
    y: np.array,
    Z: np.array,
    mpl_stylesheet: str = "default_stylesheet",
    labels: str = "no_label",
    ax: object = None,
    **kwargs,
) -> object:
    """
    Plot 2D Data in 3D using matplotlib.pylab.plot_surface().

    The plot can be configured via plot profiles.

    Parameters
    ----------
    x : np.array
        Array with values for the x axis.
    y : np.array
        Array with values for the y axis.
    Z : np.array
        2D-Array with intensities.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    labels : str, optional
        At the moment no function. The default is 'no_label'. # TODO
    ax : object, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : object
        Figure object of matplotlib.pylab.

    """
    profile = profiles.load_plot_profile(mpl_stylesheet)
    profile.update((k, kwargs[k]) for k in profile.keys() & kwargs.keys())
    for k in profile.keys() & kwargs.keys():
        kwargs.pop(k)
    style = _get_style_path(mpl_stylesheet)
    default_styles = _load_default_styles()
    default_styles.append(style)

    kwargs_ = {"cmap": cm.coolwarm, "antialiased": True, "linewidth": 0}
    kwargs_.update(kwargs)

    with HiddenPrints():
        with plt.style.context(default_styles):
            if ax is None:
                # [NEW FIGURE]
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            else:
                fig = None

            # [GRID]
            X, Y = np.meshgrid(x, y)

            # [PLOT DATA]
            try:
                ax.plot_surface(X, Y, Z, **kwargs_)
            except ValueError:
                ax.plot_surface(X, Y, Z.T, **kwargs_)

            # [AXES]
            _set_axis(ax, profile, X, Y, Z)
            _set_figure(ax, profile, mplstylesheet=mpl_stylesheet)

    return fig


def plot_3D_multiple_lines(
    x: np.array,
    y: np.array,
    Z: np.array,
    mpl_stylesheet: str = "default_stylesheet",
    ax: object = None,
    **kwargs,
) -> object:
    """
    Plot 2D Data in 3D using matplotlib.pylab.plot().

    Each y trace as a single line plot. The plot can be configured via plot
    profiles.

    Parameters
    ----------
    x : np.array
        Array with values for the x axis.
    y : np.array
        Array with values for the y axis.
    Z : np.array
        2D-Array with intensities.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    ax : object, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : object
        Figure object of matplotlib.pylab.

    """
    profile = profiles.load_plot_profile(mpl_stylesheet)
    profile.update((k, kwargs[k]) for k in profile.keys() & kwargs.keys())
    for k in profile.keys() & kwargs.keys():
        kwargs.pop(k)
    style = _get_style_path(mpl_stylesheet)
    default_styles = _load_default_styles()
    default_styles.append(style)

    with HiddenPrints():
        with plt.style.context(default_styles):
            if ax is None:
                # [NEW FIGURE]
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            else:
                fig = None

            # [PLOT DATA]
            # TODO: Hier muss drüber nachgedacht werden, wann Z transponiert werden
            # muss, damit man die x- und y-Achse in beliebiger Reihenfolge geben
            # kann.
            X = np.ones((len(y), len(x))) * x
            X = X.T
            for dataset in range(len(Z.T)):
                ax.plot(X[dataset], y, Z[:, dataset], **kwargs)

            # [AXES]
            _set_axis(ax, profile, x, y, Z)
            _set_figure(ax, profile, mplstylesheet=mpl_stylesheet)

    return fig


def heatmap(
    x: np.array,
    y: np.array,
    Z: np.array,
    mpl_stylesheet: str = "default_stylesheet",
    ax: object = None,
    **kwargs,
) -> object:
    """
    Plot 2D Data as a heatmap using matplotlib.pylab.pcolormesh().

    The plot can be configured via plot profiles.

    Parameters
    ----------
    x : np.array
        Array with values for the x axis.
    y : np.array
        Array with values for the y axis.
    Z : np.array
        2D-Array with intensities.
    mpl_stylesheet : str, optional
        Name of the matplotlib style sheet (see: matplotlib documentation).
        If no style sheet is given, the styles defined in the plotting profile
        will be used. The default is None.
    ax : object, optional
        Axes object, used for the PySpin GUI.
    **kwargs : optional
        Keyword arguments passed to the matplotlib plot function. Overrides the
        arguments given in the stylesheet.

    Returns
    -------
    fig : object
        Figure object of matplotlib.pylab.

    """
    profile = profiles.load_plot_profile(mpl_stylesheet)
    profile.update((k, kwargs[k]) for k in profile.keys() & kwargs.keys())
    for k in profile.keys() & kwargs.keys():
        kwargs.pop(k)
    style = _get_style_path(mpl_stylesheet)
    default_styles = _load_default_styles()
    default_styles.append(style)

    kwargs_ = {"cmap": "RdBu", "shading": "auto"}
    kwargs_.update(kwargs)

    with HiddenPrints():
        with plt.style.context(default_styles):
            if ax is None:
                # [NEW FIGURE]
                fig, ax = plt.subplots()
            else:
                fig = None

            # [GRID]
            X, Y = np.meshgrid(x, y)

            # [PLOT DATA]
            try:
                c = ax.pcolormesh(X, Y, Z, **kwargs_)
            except TypeError:
                c = ax.pcolormesh(X, Y, Z.T, **kwargs_)

            # [PLOT COLORBAR]
            if profile["colorbar"]:
                fig.colorbar(c, ax=ax)

            # [AXES]
            _set_axis(ax, profile, X, Y)
            _set_figure(ax, profile, mplstylesheet=mpl_stylesheet)

    return fig


def _get_shift_matrix(shape: tuple[int, int]) -> np.array:
    """
    Create a matrix to shift the single lines vertically.

    Needed for plot_shifted_2D().

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the data which will be shifted. Use np.array.shape to get the
        shape from your data.

    Returns
    -------
    shift_matrix : np.array
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
    ax: object, figure_par: dict, *axes: np.array, y_axis: bool = True
) -> object:
    """
    Set up all axis for a plot created by matplotlib.

    Parameters
    ----------
    ax : object
        Axis object of matplotlib.pylab.
    figure_par : dict
        Dictionary containing all settings for the figure.
    *axes : np.array
        All axes used for the plot. Used to determine the dimension an set up
        the z axis if needed.
    y_axis : bool, optional
        If a y axis is needed. If False, all y ticks will be removed. The
        default is True.

    Returns
    -------
    ax : object
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


def _set_figure(ax: object, figure_par: dict, mplstylesheet: str = None) -> object:
    """
    Set up the figure of a plot created by matplotlib.

    Parameters
    ----------
    ax : object
        Axis object of matplotlib.pylab.
    figure_par : dict
        Dictionary containing all settings for the figure.
    mplstylesheet : str, optional
        Name of the stylesheet which should be used instead of a plotting
        profile. If no filename is given, the settings from the given plotting
        profile will be used. The default is None.

    Returns
    -------
    ax : object
        Modified axis object of matplotlib.pylab.

    """
    if figure_par["show_title"]:
        ax.set_title(figure_par["title"])
    if figure_par["legend"]:
        ax.legend()

    return ax


def _get_style_path(mplstylesheet: str = None) -> str:
    """
    Get the path to the choosen mpl stylesheet.

    Parameters
    ----------
    mplstylesheet : str, optional
        Name of the mpl stylesheet. The default is None.

    Returns
    -------
    str
        Path to the choosen stylesheet.

    """
    if mplstylesheet is None:
        style = str(profiles.PROFILE_ROOT / "plot" / "default_stylesheet")
    elif mplstylesheet in plt.style.available:
        style = mplstylesheet
    else:
        style = str(profiles.PROFILE_ROOT / "plot" / mplstylesheet)

    return style


def _load_default_styles() -> list:
    """
    Load the pathes for the default matplotlib styles.

    Returns
    -------
    default_styles : list
        Pathes for the default style sheetss.

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
    data_axis: np.array, ax_lim: list, percentage_mode: bool
) -> [float, float]:
    """
    Get the plot limits for a given data axis.

    If no limit is specified, the minimal and maximal value of the given
    dataset will be taken.

    Parameters
    ----------
    data_axis : np.array
        Data vector for an axis.
    ax_lim : list
        Limits for the data vector taken from a profile.
    percentage_mode : bool
        If True, the ax.lim will be handled as a percentage in context to the
        lowest and highest values with respect to the given axis.
        E. g.: x-vector with values [0, 200], ax_lim: [-2., 2.] -> [4, 204].

    Returns
    -------
    [float, float]
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
