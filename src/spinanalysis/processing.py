#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transform EPR spectra through normalization, background correction, and reconstruction.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


def normalization(
    x: np.ndarray, mode: str | None = None, dx: float | None = None
) -> np.ndarray:
    r"""Normalize the given data.

    .. math::
        x_{\mathrm{norm}} = \frac{x_i - \min(x)}{\max(x)-\min(x)}

    Parameters
    ----------
    x : np.ndarray
        Unnormalized data.
    mode : str, optional
        If ``'area'``, the total AUC will be 1; if ``'value'``, the
        maximum absolute value will be 1; otherwise the formula above
        is used.  The default is ``None``.
    dx : float, optional
        Distance between two points on the x axis.  Only used for
        Simpson integration.  The default is ``None``.

    Returns
    -------
    np.ndarray
        Normalized data.
    """
    if mode is None:
        if x.dtype == "complex":
            x_norm = np.zeros(x.shape, dtype=np.complex128)
            x_norm.real = (x.real) / (x.real.max() - x.real.min())
            x_norm.imag = (x.imag) / (x.imag.max() - x.imag.min())
        else:
            x_norm = (x) / (x.max() - x.min())
    elif mode == "area":
        if x.dtype == "complex":
            x_norm = np.zeros(x.shape, dtype=np.complex128)
            x_norm.real = x.real / abs(integrate.simpson(abs(x), dx=dx))
            try:
                x_norm.imag = x.imag / abs(integrate.simpson(abs(x), dx=dx))
            except ZeroDivisionError:
                pass
        else:
            x_norm = x / abs(integrate.simpson(x, dx=dx))
    elif mode == "value":
        if x.dtype == "complex":
            x_norm = np.zeros(x.shape, dtype=np.complex128)
            x_norm.real = x.real / max(abs(x.real))
            try:
                x_norm.imag = x.imag / max(abs(x.imag))
            except ZeroDivisionError:
                pass
        else:
            x_norm = x / max(abs(x))

    return x_norm


def reduce_offset(x: np.ndarray) -> np.ndarray:
    """Eliminate the offset of the data by subtracting the mean of the last quarter.

    Parameters
    ----------
    x : np.ndarray
        Given data, e.g. measured intensities.

    Returns
    -------
    np.ndarray
        Shifted data without offset.
    """
    start = 3 * x.shape[0] // 4
    if x.dtype == "complex":
        x_shifted = np.zeros(x.shape, dtype=np.complex128)
        x_shifted.real = x.real - x.real[start:].mean()
        x_shifted.imag = x.imag - x.imag[start:].mean()
    else:
        x_shifted = x - x[start:].mean()

    return x_shifted


def background_corr(x: np.ndarray, y: np.ndarray, mode: str = "biexp") -> np.ndarray:
    """Perform a background correction of measured data.

    Available correction modes are: ``biexp``, ``exp``, ``lin``,
    ``poly2``, ``poly3`` and ``poly4``.  ``biexp`` and ``exp`` use
    exponential models for the background; ``lin`` and ``poly2-4`` are
    polynomial models of first to fourth order.

    .. warning::
        ``poly3`` and ``poly4`` can lead to overfitting!

    Parameters
    ----------
    x : np.ndarray
        x axis of the dataset.
    y : np.ndarray
        y data which will be background corrected.
    mode : str, optional
        Select the type of the background.  The default is ``'biexp'``.

    Returns
    -------
    np.ndarray
        Background corrected y data.
    """
    if mode == "biexp":
        p0 = [0.9, -0.002, 0.05, -0.0009, 0.0]
        popt, pcov = curve_fit(biexp_fun, x, y, p0=p0, maxfev=10000)
        y_corr = y - biexp_fun(x, *popt)
    elif mode == "exp":
        p0 = [1.0, -0.002, 0.0]
        popt, pcov = curve_fit(exp_fun, x, y, p0=p0, maxfev=10000)
        y_corr = y - exp_fun(x, *popt)
    elif mode == "lin":
        p0 = [1.0, -0.002]
        popt, pcov = curve_fit(lin_fun, x, y, p0=p0, maxfev=10000)
        y_corr = y - lin_fun(x, *popt)
    elif mode == "poly2":
        p0 = [1.0, -0.002, 0.0]
        popt, pcov = curve_fit(poly2_fun, x, y, p0=p0, maxfev=10000)
        y_corr = y - poly2_fun(x, *popt)
    elif mode == "poly3":
        p0 = [1.0, 1.0, -0.002, 0.0]
        popt, pcov = curve_fit(poly3_fun, x, y, p0=p0, maxfev=10000)
        y_corr = y - poly3_fun(x, *popt)
    elif mode == "poly4":
        p0 = [1.0, 1.0, 1.0, -0.002, -1.0]
        popt, pcov = curve_fit(poly4_fun, x, y, p0=p0, maxfev=10000)
        y_corr = y - poly4_fun(x, *popt)

    return y_corr


def exp_fun(x: np.ndarray, *coeff: float) -> np.ndarray:
    """Generalized monoexponential function for background correction.

    Parameters
    ----------
    x : np.ndarray
        x values used to calculate corresponding y values.
    *coeff : float
        Variables for the monoexponential function which will be fitted.

    Returns
    -------
    np.ndarray
        Calculated y values.
    """
    a, b, c = coeff
    y = a * np.exp(b * x) + c
    return y


def biexp_fun(x: np.ndarray, *coeff: float) -> np.ndarray:
    """Generalized biexponential function for background correction.

    Parameters
    ----------
    x : np.ndarray
        x values used to calculate corresponding y values.
    *coeff : float
        Variables for the biexponential function which will be fitted.

    Returns
    -------
    np.ndarray
        Calculated y values.
    """
    a, b, c, d, e = coeff
    y = a * np.exp(b * x) + c * np.exp(d * x) + e
    return y


def lin_fun(x: np.ndarray, *coeff: float) -> np.ndarray:
    """Generalized linear function for background correction.

    Parameters
    ----------
    x : np.ndarray
        x values used to calculate corresponding y values.
    *coeff : float
        Variables for the linear function which will be fitted.

    Returns
    -------
    np.ndarray
        Calculated y values.
    """
    a, b = coeff
    y = a * x + b
    return y


def poly2_fun(x: np.ndarray, *coeff: float) -> np.ndarray:
    """Generalized polynomial function of degree 2 for background correction.

    Parameters
    ----------
    x : np.ndarray
        x values used to calculate corresponding y values.
    *coeff : float
        Variables for the polynomial function of degree 2 which will be
        fitted.

    Returns
    -------
    np.ndarray
        Calculated y values.
    """
    a, b, c = coeff
    y = a * x**2 + b * x + c
    return y


def poly3_fun(x: np.ndarray, *coeff: float) -> np.ndarray:
    """Generalized polynomial function of degree 3 for background correction.

    Parameters
    ----------
    x : np.ndarray
        x values used to calculate corresponding y values.
    *coeff : float
        Variables for the polynomial function of degree 3 which will be
        fitted.

    Returns
    -------
    np.ndarray
        Calculated y values.
    """
    a, b, c, d = coeff
    y = a * x**3 + b * x**2 + c * x + d
    return y


def poly4_fun(x: np.ndarray, *coeff: float) -> np.ndarray:
    """Generalized polynomial function of degree 4 for background correction.

    Parameters
    ----------
    x : np.ndarray
        x values used to calculate corresponding y values.
    *coeff : float
        Variables for the polynomial function of degree 4 which will be
        fitted.

    Returns
    -------
    np.ndarray
        Calculated y values.
    """
    a, b, c, d, e = coeff
    y = a * x**4 + b * x**3 + c * x**2 + d * x + e
    return y


def reconstruct(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a time signal using the Yule-Walker algorithm.

    Parameters
    ----------
    x : np.ndarray
        x axis.
    y : np.ndarray
        Intensities.

    Returns
    -------
    x_new : np.ndarray
        Reconstructed x axis.
    y_new : np.ndarray
        Reconstructed intensities.
    """
    x_step = x[1] - x[0]
    if x[0] % x_step != 0:
        x_fill_points = int(x[0] / x_step) + 1
    else:
        x_fill_points = int(x[0] / x_step)
    x_new = np.concatenate(
        (np.linspace(x[0] - x_step * x_fill_points, x[0] - x_step, x_fill_points), x)
    )

    order = ar_select_order(y[::-1], maxlag=40)
    nlag = len(order.ar_lags)

    AutoRegFit = AutoReg(y[::-1], lags=order.ar_lags).fit()
    y_pred = AutoRegFit.predict(start=0, end=x_new.shape[0] + nlag - 1)

    y_pred = np.roll(y_pred[nlag:], nlag)
    y_flip = y_pred
    y_flip = np.concatenate((y[::-1], y_flip[len(y) :]))

    return x_new, y_flip[::-1]
