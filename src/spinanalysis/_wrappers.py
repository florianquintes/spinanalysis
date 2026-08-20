#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provide decorators and multicore helpers for simulation routines.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

from collections.abc import Callable
from copy import deepcopy
from itertools import repeat
from multiprocessing import Pool, cpu_count
from time import time
from typing import Any

import numpy as np


def timer(func: Callable) -> Callable:
    """Measure the wall-clock runtime of a single function call.

    Parameters
    ----------
    func : callable
        Function whose runtime will be measured.

    Returns
    -------
    callable
        Wrapper that prints the runtime and returns the original result.
    """

    def time_wrap(*args: Any, **kwargs: Any) -> Any:
        start = time()
        res = func(*args, **kwargs)
        runtime = time() - start
        print("The runtime of {} is {:.3f} s".format(func.__name__, runtime))
        return res

    return time_wrap


def function_benchmark(func: Callable, niter: int = 100) -> Callable:
    """Run *func* *niter* times and print best, worst, and average runtime.

    Parameters
    ----------
    func : callable
        Function which will be benchmarked.
    niter : int, optional
        Number of function calls. The default is 100.

    Returns
    -------
    callable
        Wrapper that runs the benchmark and prints timing statistics.
    """

    def benchmarked_function(*args: Any, **kwargs: Any) -> None:
        times = np.empty(niter)
        for i in range(times.shape[0]):
            start = time()
            func(*args, **kwargs)
            times[i] = time() - start

        print("Runned a benchmark of {}.".format(func.__name__))

        if times.min() > 1:
            unit = "s"
        elif times.min() > 1e-3:
            unit = "ms"
            times *= 1e3
        else:
            unit = "μs"
            times *= 1e6

        print("Average time: {:.3f} {}".format(times.mean(), unit))
        print("Best time: {:.3f} {}".format(times.min(), unit))
        print("Worst time: {:3f} {}".format(times.max(), unit))

    return benchmarked_function


def multicore(simulation: Callable) -> Callable:
    """Parallelise a simulation routine using :class:`multiprocessing.Pool`.

    The decorated function must accept ``(Sys, Exp, SimOpt)`` and is
    executed on ``SimOpt.cpu_cores`` processes, each handling a slice of
    the magnetic-field axis.

    Parameters
    ----------
    simulation : callable
        Simulation function with the signature ``(Sys, Exp, SimOpt)``.

    Returns
    -------
    callable
        Wrapper with signature ``(Sys, Exp, SimOpt)`` that distributes
        the work across CPU cores and returns the concatenated spectrum.
    """

    def multicore_wrapper(Sys: Any, Exp: Any, SimOpt: Any) -> np.ndarray:
        if SimOpt.cpu_cores == 0:
            SimOpt.cpu_cores = cpu_count()

        whole_spectrum = 1 * Exp.magnetic_field
        whole_B_z = 1 * Exp.B_z
        n_field_points = whole_spectrum.shape[0]

        points_per_core = n_field_points // SimOpt.cpu_cores

        Exp_list = np.empty(SimOpt.cpu_cores, dtype=object)
        for core in range(SimOpt.cpu_cores):
            Experimental = deepcopy(Exp)
            start = core * points_per_core
            if core + 1 < SimOpt.cpu_cores:
                end = (core + 1) * points_per_core
                Experimental.magnetic_field = whole_spectrum[start:end]
                Experimental.B_z = whole_B_z[start:end]
            else:
                Experimental.magnetic_field = whole_spectrum[start:]
                Experimental.B_z = whole_B_z[start:]
            Exp_list[core] = Experimental

        pool = Pool(processes=SimOpt.cpu_cores)
        single_intensities = pool.starmap(
            simulation, zip(repeat(Sys), Exp_list, repeat(SimOpt))
        )
        pool.close()
        pool.join()

        intens_arr_tuple = tuple(single_intensities)
        intensity = np.hstack(intens_arr_tuple)

        return intensity

    return multicore_wrapper
