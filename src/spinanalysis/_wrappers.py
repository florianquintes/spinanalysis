#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

from time import time
from copy import deepcopy
from multiprocessing import cpu_count, Pool
from itertools import repeat
import numpy as np


def timer(func: callable) -> callable:
    """
    Decorator function to measure time for one function call.

    Parameters
    ----------
    func : callable
        Function whose runtime will be measured.

    Returns
    -------
    res : any
        Result(s) of the function.

    """

    def time_wrap(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        runtime = time() - start
        print("The runtime of {} is {:.3f} s".format(func.__name__, runtime))
        return res

    return time_wrap


def function_benchmark(func: callable, niter: int = 100) -> callable:
    """
    This decorateur will run the given function niter times and print the best,
    the worst and the average runtime.

    Parameters
    ----------
    func : callable
        Function which will be benchmarked.
    niter : int
        Number of function calls.

    Returns
    -------
    callable
        Function with automatic benchmark.

    """

    def benchmarked_function(*args, **kwargs):
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


def multicore(simulation: callable) -> callable:
    """
    Using multiprocessing.Pool() with starmap() for parallel computing of
    various simulation routines using the easypairspin function interface
    simulation(Sys, Exp, SimOpt).

    Parameters
    ----------
    simulation : callable
        Simulation function which uses the easypairspin interface (Sys, Exp,
        SimOpt).

    Returns
    -------
    multicore_wrapper : callable
        The origin simulation callable as multicore version.

    """

    def simulation_with_queue(
        simulation: callable,
        Sys: object,
        Exp: object,
        SimOpt: object,
        queue: object,
        num: int,
    ) -> None:
        intensity = simulation(Sys, Exp, SimOpt)
        queue.put((num, intensity))

        return None

    def multicore_wrapper(Sys: object, Exp: object, SimOpt: object) -> np.array:
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

        # [Multi-Core Calculation]
        pool = Pool(processes=SimOpt.cpu_cores)
        single_intensities = pool.starmap(
            simulation, zip(repeat(Sys), Exp_list, repeat(SimOpt))
        )
        pool.close()
        pool.join()

        # queue = Queue()

        # processes = []
        # for i, Exp_i in enumerate(Exp_list):
        #     processes.append(Process(target=simulation_with_queue,
        #                              args=(simulation, Sys, Exp_i, SimOpt,
        #                                    queue, i)
        #                              )
        #                      )

        # for p in processes:
        #     p.start()

        # for p in processes:
        #     p.join()

        # intensities = [queue.get() for p in processes]
        # intensities.sort()

        # single_intensities = [intensity[1] for intensity in intensities]

        intens_arr_tuple = tuple(single_intensities)
        intensity = np.hstack(intens_arr_tuple)

        return intensity

    return multicore_wrapper
