#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge between spinanalysis objects and scipy.optimize routines.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from static_radical_pair.radpair import do_simulation_multicore, do_simulation
from teacups.simulations import teacups_multicore
from teacups.simulations import teacups
from oop_eseem.opossum import (
    oop_eseem,
    oop_eseem_distribution,
    _get_multi_gauss_distribution,
)

from PySpin.plotter import plot
import scipy.optimize as optimize
from copy import deepcopy
from functools import partial


def plot_callback(
    xk: NDArray[np.float64],
    *_,
    Sys: Any = None,
    Exp: Any = None,
    SimOpt: Any = None,
    FitOpt: Any = None,
    Var: Any = None,
    **_kwargs,
) -> None:
    """
    Plot the current optimization state in the graphical user interface.

    Parameters
    ----------
    xk : np.array
        Current best guess vector.
    *_ : arbitrary
        Unused arguments passed by some optimization routines to the callback
        function.
    Sys : object
        Reference spin-system object.
    Exp : object
        Experimental data object.
    SimOpt : object
        Simulation options object.
    Var : object
        Variation object describing the fitted parameters.
    FitOpt : object
        Fitting options object.
    *_kwargs : arbitrary
        Unused keyword arguments passed by some optimization routines to the
        callback function.

    Raises
    ------
    ValueError
        Raised if the selected simulation routine is unknown.

    Returns
    -------
    None.

    """
    current_Sys = guess2Sys(xk, Sys, Var, SimOpt)
    SimOpt.mode = "simulation"

    if SimOpt.routine.lower() == "static_radpair":
        simulated_spectra = do_simulation_multicore(current_Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "teacups":
        simulated_spectra = teacups_multicore(current_Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "opossum":
        simulated_spectra = oop_eseem(current_Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "didelphis":
        simulated_spectra = oop_eseem_distribution(current_Sys, Exp, SimOpt)
    else:
        raise ValueError("Can't find a routine named '{0}'!".format(SimOpt.routine))

    simulated_spectra /= (abs(simulated_spectra)).max()
    Exp.spec_sim = simulated_spectra
    data = np.array([Exp.spec_sim, Exp.int])
    label = ["current best Fit", "exp. Data"]
    if SimOpt.routine.lower() in ("static_radpair", "teacups"):
        plot(FitOpt.window, Exp.B_z, data, FitOpt.window.canvas_1, labels=label)
    else:
        plot(
            FitOpt.window,
            Exp.time_axis,
            data,
            FitOpt.window.canvas_1,
            labels=label,
        )

        plot(
            FitOpt.window,
            current_Sys.distribution[0],
            current_Sys.distribution[1],
            FitOpt.window.canvas_distribution,
            labels=["current Distribution"],
        )


def spinanalysis2scipy(x: NDArray[np.float64], *objects: Any) -> float:
    """
    Objective function for the scipy.optimize interface.

    Used by scipy.optimize routines.

    Parameters
    ----------
    x : np.array
        Current parameter vector of the optimizer.
    *objects : object
        Additional objects required for the simulation and fitting interface.

    Raises
    ------
    ValueError
        Raised if the selected simulation routine is unknown.

    Returns
    -------
    error : float
        Sum of squared residuals between experimental and simulated data.

    """
    Sys_def, Exp, SimOpt, Var = objects

    Sys = guess2Sys(x, Sys_def, Var, SimOpt)

    SimOpt.mode = "simulation"

    if SimOpt.routine.lower() == "static_radpair":
        simulated_spectra = do_simulation_multicore(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "teacups":
        simulated_spectra = teacups_multicore(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "opossum":
        simulated_spectra = oop_eseem(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "didelphis":
        simulated_spectra = oop_eseem_distribution(Sys, Exp, SimOpt)
    else:
        raise ValueError("Can't find a routine named '{0}'!".format(SimOpt.routine))

    simulated_spectra /= (abs(simulated_spectra)).max()
    Exp.spec_sim = simulated_spectra

    error = ((Exp.int.real - Exp.spec_sim.real) ** 2).sum()

    return error


def spinanalysis2scipy_singlecore(x: NDArray[np.float64], *objects: Any) -> float:
    """
    Objective function for the scipy.optimize interface.

    Used by scipy.optimize.differential_evolution.

    Parameters
    ----------
    x : np.array
        Current parameter vector of the optimizer.
    *objects : object
        Additional objects required for the simulation and fitting interface.

    Raises
    ------
    ValueError
        Raised if the selected simulation routine is unknown.

    Returns
    -------
    error : float
        Sum of squared residuals between experimental and simulated data.

    """
    Sys_def, Exp, SimOpt, Var = objects

    Sys = guess2Sys(x, Sys_def, Var, SimOpt)

    SimOpt.mode = "fitting"

    if SimOpt.routine.lower() == "static_radpair":
        simulated_spectra = do_simulation(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "teacups":
        simulated_spectra = teacups(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "opossum":
        simulated_spectra = oop_eseem(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "didelphis":
        simulated_spectra = oop_eseem_distribution(Sys, Exp, SimOpt)
    else:
        raise ValueError("Can't find a routine named '{0}'!".format(SimOpt.routine))

    simulated_spectra /= (abs(simulated_spectra)).max()
    Exp.spec_sim = simulated_spectra

    error = ((Exp.int.real - Exp.spec_sim.real) ** 2).sum()

    return error


def spinanalysis2scipy_res(
    x: NDArray[np.float64], *objects: Any
) -> NDArray[np.float64]:
    """
    Residual function for scipy.optimize least-squares algorithms.

    Returns the absolute residuals between simulation and experiment.

    Used by scipy.optimize routines.

    Parameters
    ----------
    x : np.array
        Current parameter vector of the optimizer.
    *objects : object
        Additional objects required for the simulation and fitting interface.

    Raises
    ------
    ValueError
        Raised if the selected simulation routine is unknown.

    Returns
    -------
    error : np.array
        One-dimensional array containing the absolute residuals.

    """
    Sys_def, Exp, SimOpt, Var = objects

    Sys = guess2Sys(x, Sys_def, Var, SimOpt)

    SimOpt.mode = "simulation"

    if SimOpt.routine.lower() == "static_radpair":
        simulated_spectra = do_simulation_multicore(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "teacups":
        simulated_spectra = teacups_multicore(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "opossum":
        simulated_spectra = oop_eseem(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "didelphis":
        simulated_spectra = oop_eseem_distribution(Sys, Exp, SimOpt)
    else:
        raise ValueError("Can't find a routine named '{0}'!".format(SimOpt.routine))

    simulated_spectra /= (abs(simulated_spectra)).max()
    Exp.spec_sim = simulated_spectra

    error = abs(Exp.int.real - Exp.spec_sim.real)

    return error


def guess2Sys(x: NDArray[np.float64], Sys: Any, Var: Any, SimOpt: Any) -> Any:
    """
    Create a spin-system object from the current optimizer vector.

    Parameters
    ----------
    x : np.array
        Current parameter vector of the optimizer.
    Sys : object
        Reference spin-system object.
    Var : object
        Variation object.
    SimOpt : object
        Simulation options object.

    Returns
    -------
    Sys_mod : object
        Spin-system object corresponding to the current optimizer vector.

    """
    Sys_mod = deepcopy(Sys)

    n = 0
    for key in Var.__dict__:
        if key in Var.non_vars:
            pass
        elif key in Var.single_vars:
            if getattr(Var, key) > 0:
                setattr(Sys_mod, key, x[n])
                n += 1
        else:
            var_values = getattr(Var, key)
            sys_values = getattr(Sys_mod, key)
            for i, parameter in enumerate(var_values):
                if parameter > 0:
                    sys_values[i] = x[n]
                    n += 1

    if hasattr(Var, "isotropic"):
        for par in Var.isotropic:
            arr = getattr(Sys_mod, par)
            arr[1] = arr[0]
            arr[2] = arr[0]

    for key in Sys.__dict__:
        if key.startswith("frame_group"):
            for i, frame in enumerate(getattr(Sys, key)):
                if not frame.endswith("_frame"):
                    frame = frame + "_frame"
                if i == 0:
                    reference_frame = getattr(Sys_mod, frame)
                else:
                    setattr(Sys_mod, frame, reference_frame)

    if Var.fit_distribution:
        pos = np.zeros(Sys.distribution_order)
        sigma = np.zeros(Sys.distribution_order)
        intens = np.zeros(Sys.distribution_order)
        for i in range(Sys.distribution_order):
            intens[i] = x[n + i * 3]
            sigma[i] = x[n + i * 3 + 1]
            pos[i] = x[n + i * 3 + 2]

        intens /= intens.sum()
        r_axis = np.linspace(SimOpt.min_r, SimOpt.max_r, SimOpt.r_points)
        mg_distribution = _get_multi_gauss_distribution(intens, sigma, pos, r_axis)
        Sys_mod.distribution = np.array([r_axis, mg_distribution])
        Sys._mgf_parameters = np.array([pos, intens, sigma])

    return Sys_mod


def get_random_x0(
    boundaries: Sequence[tuple[float, float]],
) -> NDArray[np.float64]:
    """
    Generate a random initial guess within the variation boundaries.

    Parameters
    ----------
    boundaries : list
        Lower and upper bounds of the fitted parameters.

    Returns
    -------
    x0 : np.array
        Randomly generated initial parameter vector.

    """
    x0 = np.empty(len(boundaries))

    for i, bounds in enumerate(boundaries):
        x0[i] = np.random.uniform(bounds[0], bounds[1])

    return x0


def dualannealing(Sys: Any, Exp: Any, SimOpt: Any, FitOpt: Any, Var: Any) -> Any:
    """
    Run scipy.optimize.dual_annealing for global optimization.

    Parameters
    ----------
    Sys : object
        Reference spin-system object.
    Exp : object
        Experimental data object.
    SimOpt : object
        Simulation options object.
    Var : object
        Variation object describing the fitted parameters.
    FitOpt : object
        Fitting options object.

    Returns
    -------
    best_Sys : object
        Best spin-system object found by the optimizer.

    """
    Var.get_boundaries(Sys)

    settings = dict(args=(Sys, Exp, SimOpt, Var), bounds=Var.boundaries)

    if FitOpt.gui:
        settings["callback"] = partial(
            plot_callback,
            Sys=Sys,
            Exp=Exp,
            SimOpt=SimOpt,
            FitOpt=FitOpt,
            Var=Var,
        )

    if FitOpt.x0 is not None:
        settings["x0"] = FitOpt.x0

    if FitOpt.method is not None:
        settings["minimizer_kwargs"] = dict()
        settings["minimizer_kwargs"]["method"] = FitOpt.method

        if FitOpt.maxiter_minimizer is not None:
            settings["minimizer_kwargs"]["maxiter"] = FitOpt.maxiter_minimizer

    if FitOpt.maxiter is not None:
        settings["maxiter"] = FitOpt.maxiter

    if FitOpt.initial_temp is not None:
        settings["initial_temp"] = FitOpt.initial_temp

    if FitOpt.restart_temp_ratio is not None:
        settings["restart_temp_ratio"] = FitOpt.restart_temp_ratio

    if FitOpt.visit is not None:
        settings["visit"] = FitOpt.visit

    if FitOpt.accept is not None:
        settings["accept"] = FitOpt.accept

    if FitOpt.maxfun is not None:
        settings["maxfun"] = FitOpt.maxfun

    if FitOpt.no_local_search is not None:
        settings["no_local_search"] = FitOpt.no_local_search

    results = optimize.dual_annealing(spinanalysis2scipy, **settings)

    best_Sys = guess2Sys(results["x"], Sys, Var, SimOpt)

    if FitOpt.gui:
        return best_Sys, results
    print(results)

    return best_Sys


def shgo(Sys: Any, Exp: Any, SimOpt: Any, FitOpt: Any, Var: Any) -> Any:
    """
    Run scipy.optimize.shgo for global optimization.

    Parameters
    ----------
    Sys : object
        Reference spin-system object.
    Exp : object
        Experimental data object.
    SimOpt : object
        Simulation options object.
    Var : object
        Variation object describing the fitted parameters.
    FitOpt : object
        Fitting options object.

    Returns
    -------
    best_Sys : object
        Best spin-system object found by the optimizer.

    """
    Var.get_boundaries(Sys)

    settings = dict(args=(Sys, Exp, SimOpt, Var), bounds=Var.boundaries)

    if FitOpt.gui:
        settings["callback"] = partial(
            plot_callback,
            Sys=Sys,
            Exp=Exp,
            SimOpt=SimOpt,
            FitOpt=FitOpt,
            Var=Var,
        )

    if FitOpt.method is not None:
        settings["minimizer_kwargs"] = dict()
        settings["minimizer_kwargs"]["method"] = FitOpt.method

        if FitOpt.maxiter_minimizer is not None:
            options = dict(maxiter=FitOpt.maxiter_minimizer)
            settings["minimizer_kwargs"]["options"] = options

    settings["options"] = dict()

    if FitOpt.maxiter is not None:
        settings["options"]["maxiter"] = FitOpt.maxiter

    if FitOpt.n is not None:
        settings["n"] = FitOpt.n

    if FitOpt.iters is not None:
        settings["iters"] = FitOpt.iters

    if FitOpt.maxfev is not None:
        settings["maxfev"] = FitOpt.maxfev

    if FitOpt.f_tol is not None:
        settings["options"]["f_tol"] = FitOpt.f_tol

    if FitOpt.maxev is not None:
        settings["options"]["maxev"] = FitOpt.maxev

    if FitOpt.maxtime is not None:
        settings["options"]["maxtime"] = FitOpt.maxtime

    if FitOpt.minimize_every_iter is not None:
        settings["options"]["minimize_every_iter"] = FitOpt.minimize_every_iter

    if FitOpt.local_iter is not None:
        settings["options"]["local_iter"] = FitOpt.local_iter

    if FitOpt.sampling_method is not None:
        settings["sampling_method"] = FitOpt.sampling_method

    results = optimize.shgo(spinanalysis2scipy, **settings)
    best_Sys = guess2Sys(results["x"], Sys, Var, SimOpt)

    if FitOpt.gui:
        return best_Sys, results
    print(results)

    return best_Sys


def differential_evolution(
    Sys: Any, Exp: Any, SimOpt: Any, FitOpt: Any, Var: Any
) -> Any:
    """
    Run scipy.optimize.differential_evolution for global optimization.

    Parameters
    ----------
    Sys : object
        Reference spin-system object.
    Exp : object
        Experimental data object.
    SimOpt : object
        Simulation options object.
    Var : object
        Variation object describing the fitted parameters.
    FitOpt : object
        Fitting options object.

    Returns
    -------
    best_Sys : object
        Best spin-system object found by the optimizer.

    """
    Var.get_boundaries(Sys)

    settings = dict(args=(Sys, Exp, SimOpt, Var), bounds=Var.boundaries)

    if FitOpt.gui:
        settings["callback"] = partial(
            plot_callback,
            Sys=Sys,
            Exp=Exp,
            SimOpt=SimOpt,
            FitOpt=FitOpt,
            Var=Var,
        )

    if FitOpt.strategy is not None:
        settings["strategy"] = FitOpt.strategy

    if FitOpt.maxiter is not None:
        settings["maxiter"] = FitOpt.maxiter

    if FitOpt.popsize is not None:
        settings["popsize"] = FitOpt.popsize

    if FitOpt.tol is not None:
        settings["tol"] = FitOpt.tol

    if FitOpt.mutation is not None:
        settings["mutation"] = FitOpt.mutation

    if FitOpt.recombination is not None:
        settings["recombination"] = FitOpt.recombination

    if FitOpt.seed is not None:
        settings["seed"] = FitOpt.seed

    if FitOpt.polish is not None:
        settings["polish"] = FitOpt.polish

    if FitOpt.init is not None:
        settings["init"] = FitOpt.init

    if FitOpt.atol is not None:
        settings["atol"] = FitOpt.atol

    if FitOpt.updating is not None:
        settings["updating"] = FitOpt.updating

    if FitOpt.x0 is not None:
        settings["x0"] = FitOpt.x0

    if FitOpt.cpu_cores == 0:
        settings["workers"] = -1
    else:
        settings["workers"] = FitOpt.cpu_cores

    results = optimize.differential_evolution(spinanalysis2scipy_singlecore, **settings)
    best_Sys = guess2Sys(results["x"], Sys, Var, SimOpt)

    if FitOpt.gui:
        return best_Sys, results
    print(results)

    return best_Sys


def basinhopping(Sys: Any, Exp: Any, SimOpt: Any, FitOpt: Any, Var: Any) -> Any:
    """
    Run scipy.optimize.basinhopping for global optimization.

    Parameters
    ----------
    Sys : object
        Reference spin-system object.
    Exp : object
        Experimental data object.
    SimOpt : object
        Simulation options object.
    Var : object
        Variation object describing the fitted parameters.
    FitOpt : object
        Fitting options object.

    Returns
    -------
    best_Sys : object
        Best spin-system object found by the optimizer.

    """
    Var.get_boundaries(Sys)

    settings = dict()

    if FitOpt.gui:
        settings["callback"] = partial(
            plot_callback,
            Sys=Sys,
            Exp=Exp,
            SimOpt=SimOpt,
            FitOpt=FitOpt,
            Var=Var,
        )

    if FitOpt.x0 is not None:
        settings["x0"] = FitOpt.x0
    else:
        settings["x0"] = get_random_x0(Var.boundaries)

    basinhopping_bounds = BasinhoppingBounds(Var)
    settings["accept_test"] = basinhopping_bounds

    settings["minimizer_kwargs"] = dict()
    settings["minimizer_kwargs"]["args"] = (Sys, Exp, SimOpt, Var)

    if FitOpt.method is not None:
        settings["minimizer_kwargs"]["method"] = FitOpt.method
        settings["minimizer_kwargs"]["bounds"] = Var.boundaries

    if FitOpt.T is not None:
        settings["T"] = FitOpt.T

    if FitOpt.niter is not None:
        settings["niter"] = FitOpt.niter

    if FitOpt.stepsize is not None:
        basinhopping_steps = BasinhoppingStep(Var, FitOpt.stepsize)
    else:
        basinhopping_steps = BasinhoppingStep(Var)
    settings["take_step"] = basinhopping_steps

    if FitOpt.interval is not None:
        settings["interval"] = FitOpt.interval

    if FitOpt.disp is not None:
        settings["disp"] = FitOpt.disp

    basinhopping_status = BasinhoppingStatus(Sys, Var, FitOpt.disp)
    settings["callback"] = basinhopping_status

    if FitOpt.niter_success is not None:
        settings["niter_success"] = FitOpt.niter_success

    if FitOpt.seed is not None:
        settings["seed"] = FitOpt.seed

    if FitOpt.target_accept_rate is not None:
        settings["target_accept_rate"] = FitOpt.target_accept_rate

    if FitOpt.stepwise_factor is not None:
        settings["stepwise_factor"] = FitOpt.stepwise_factor

    results = optimize.basinhopping(spinanalysis2scipy, **settings)
    best_Sys = guess2Sys(results["x"], Sys, Var, SimOpt)

    if FitOpt.gui:
        return best_Sys, results
    print(results)

    return best_Sys


def least_squares(Sys: Any, Exp: Any, SimOpt: Any, FitOpt: Any, Var: Any) -> Any:
    """
    Run scipy.optimize.least_squares for nonlinear optimization.

    Parameters
    ----------
    Sys : object
        Reference spin-system object.
    Exp : object
        Experimental data object.
    SimOpt : object
        Simulation options object.
    Var : object
        Variation object describing the fitted parameters.
    FitOpt : object
        Fitting options object.

    Returns
    -------
    best_Sys : object
        Best spin-system object found by the optimizer.

    """
    Var.get_boundaries(Sys)

    lb = []
    ub = []
    for bounds in Var.boundaries:
        lb.append(bounds[0])
        ub.append(bounds[1])

    lb = np.array(lb)
    ub = np.array(ub)

    settings = dict(args=(Sys, Exp, SimOpt, Var), bounds=(lb, ub))

    if FitOpt.x0 is not None:
        settings["x0"] = FitOpt.x0
    else:
        settings["x0"] = get_random_x0(Var.boundaries)

    if FitOpt.method is not None:
        settings["method"] = FitOpt.method

        if FitOpt.method == "lm":
            del settings["bounds"]

    if FitOpt.ftol is not None:
        settings["ftol"] = FitOpt.ftol

    if FitOpt.xtol is not None:
        settings["xtol"] = FitOpt.xtol

    if FitOpt.gtol is not None:
        settings["gtol"] = FitOpt.gtol

    if FitOpt.loss is not None:
        settings["loss"] = FitOpt.loss

    if FitOpt.f_scale is not None:
        settings["f_scale"] = FitOpt.f_scale

    if FitOpt.max_nfev is not None:
        settings["max_nfev"] = FitOpt.max_nfev

    if FitOpt.tr_solver is not None:
        settings["tr_solver"] = FitOpt.tr_solver

    if FitOpt.verbose is not None:
        settings["verbose"] = FitOpt.verbose

    results = optimize.least_squares(spinanalysis2scipy_res, **settings)
    best_Sys = guess2Sys(results["x"], Sys, Var, SimOpt)

    if FitOpt.gui:
        return best_Sys, results
    print(results)

    return best_Sys


def minimize(Sys: Any, Exp: Any, SimOpt: Any, FitOpt: Any, Var: Any) -> Any:
    """
    Run scipy.optimize.minimize for local optimization.

    Minimize provides multiple local optimization routines such as Nelder-Mead,
    COBYLA, Powell, CG and so on.

    Parameters
    ----------
    Sys : object
        Reference spin-system object.
    Exp : object
        Experimental data object.
    SimOpt : object
        Simulation options object.
    Var : object
        Variation object describing the fitted parameters.
    FitOpt : object
        Fitting options object.

    Returns
    -------
    best_Sys : object
        Best spin-system object found by the optimizer.
    results : str, optional
        Results of the scipy optimization. Only for the GUI.

    """
    Var.get_boundaries(Sys)

    settings = dict(args=(Sys, Exp, SimOpt, Var), bounds=Var.boundaries)
    if FitOpt.gui:
        settings["callback"] = partial(
            plot_callback,
            Sys=Sys,
            Exp=Exp,
            SimOpt=SimOpt,
            FitOpt=FitOpt,
            Var=Var,
        )

    if FitOpt.x0 is not None:
        settings["x0"] = FitOpt.x0
    else:
        settings["x0"] = get_random_x0(Var.boundaries)

    if FitOpt.method is not None:
        settings["method"] = FitOpt.method

    if FitOpt.maxiter is not None:
        settings["options"] = dict()
        settings["options"]["maxiter"] = FitOpt.maxiter

    results = optimize.minimize(spinanalysis2scipy, **settings)
    best_Sys = guess2Sys(results["x"], Sys, Var, SimOpt)

    if FitOpt.gui:
        return best_Sys, results
    print(results)

    return best_Sys


class BasinhoppingBounds:
    """
    Acceptance test for the scipy.optimize.basinhopping algorithm.

    Attributes
    ----------
    xmin : np.array
        Lower bounds for the varied parameters.
    xmax : np.array
        Upper bounds for the varied parameters.
    Var : object
        Object of class Variation from the epr_setup module.

    Methods
    -------
    __call__(**kwargs)
        Check if the current guess is within the bounds.

    """

    def __init__(self, Var: Any) -> None:
        self.xmin, self.xmax = np.array(Var.boundaries).T

        return None

    def __call__(self, **kwargs) -> bool:
        """
        Check if the current guess is within the bounds.

        Parameters
        ----------
        **kwargs : list
            Varied parameters.

        Returns
        -------
        bool
            True if the guess is within the bounds, False if not.

        """
        x = kwargs["x_new"]
        test_min = bool(np.all(x >= self.xmin))
        test_max = bool(np.all(x <= self.xmax))

        return test_max and test_min


class BasinhoppingStep:
    """
    Step generator for the scipy.optimize.basinhopping algorithm.

    Attributes
    ----------
    stepsize : float, optional
        Relative size of the random step with respect to the variation range.
    rng : object
        NumPy random number generator.
    Var : object
        Object of class Variation from the epr_setup module.
    bounds : np.array
        Parameter boundaries as a two-dimensional array.
    lb : np.array
        Lower bounds.
    ub : np.array
        Upper bounds.
    var_range : np.array
        Half the difference between lower and upper bounds.
    dim_var : int
        Number of variables.

    Methods
    -------
    __call__(x)
        Generate the next random step.

    """

    def __init__(self, Var: Any, stepsize: float = 0.75) -> None:
        self.stepsize = stepsize
        self.rng = np.random.default_rng()
        self.bounds = np.array(Var.boundaries)
        self.lb = self.bounds[:, 0]
        self.ub = self.bounds[:, 1]
        self.var_range = 0.5 * (self.ub - self.lb)
        self.dim_var = len(self.var_range)

        return None

    def __call__(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Generate the next random step.

        Parameters
        ----------
        x : np.array
            Current guess.

        Returns
        -------
        x : np.array
            Current parameter vector after applying a random step.

        """
        x = self.check_guess(x)

        step_range = self.stepsize * self.var_range

        mask_lb = (x - self.lb) >= step_range
        mask_ub = (self.ub - x) >= step_range

        l_range = np.zeros(self.dim_var)
        u_range = np.zeros(self.dim_var)

        l_range[mask_lb] = step_range[mask_lb]
        l_range[~mask_lb] = (x - self.lb)[~mask_lb]

        u_range[mask_ub] = step_range[mask_ub]
        u_range[~mask_ub] = (self.ub - x)[~mask_ub]

        x += self.rng.uniform(-l_range, u_range)

        return x

    def check_guess(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Check whether the current parameter vector is within the bounds.

        Out-of-bound values are replaced by randomly generated values inside the bounds.

        Parameters
        ----------
        x : np.array
            Current guess.

        Returns
        -------
        x : np.array
            Current guess.

        """
        choose_outliers = ~((x >= self.lb) & (x <= self.ub))

        if choose_outliers.sum() > 0:
            x_new = self.rng.uniform(self.lb, self.ub)
            x[choose_outliers] = x_new[choose_outliers]

        return x


class BasinhoppingStatus:
    """
    Status callback for the scipy.optimize.basinhopping algorithm.

    Attributes
    ----------
    best : float
        Best objective-function value found so far.
    def_Sys : object
        Reference spin-system object.
    Var : object
        Variation object containing the parameter ranges.
    verbose : bool, optional
        Controls whether status information is printed.
    xmin : np.array
        Lower bounds for the varied parameters.
    xmax : np.array
        Upper bounds for the varied parameters.

    Methods
    -------
    __call__(x, value, accepted)
        Print status information.

    """

    def __init__(self, Sys: Any, Var: Any, verbose: bool = False) -> None:
        self.best = np.inf
        self.def_Sys = deepcopy(Sys)
        self.Var = Var
        self.xmin, self.xmax = np.array(Var.boundaries).T

        if verbose is not None:
            self.verbose = verbose
        else:
            self.verbose = False

        return None

    def __call__(self, x: NDArray[np.float64], value: float, accepted: int) -> None:
        """
        Print status information about the basinhopping progress.

        Parameters
        ----------
        x : np.array
            Current guess.
        value : float
            Objective-function value of the current parameter vector.
        accepted : int
            Acceptance status returned by the basinhopping algorithm.

        Returns
        -------
        None
            No return value.

        """
        mes_1 = "The current minimum with an error of {:.4f} ".format(value)
        if self.check_bounds(x):
            mes_2 = "was accepted"

            if value < self.best:
                self.best = value
                self.save_best(x)
                mes_3 = " and is a new best minimum!"
            else:
                mes_3 = "."
        else:
            mes_2 = "was not accepted"
            mes_3 = "."

        if self.verbose:
            print(mes_1 + mes_2 + mes_3)

        return None

    def check_bounds(self, x: NDArray[np.float64]) -> bool:
        """
        Check if the current guess is within the bounds.

        Parameters
        ----------
        x : np.array
            Current guess.

        Returns
        -------
        bool
            True if the guess is within the bounds, False if not.

        """
        test_min = bool(np.all(x >= self.xmin))
        test_max = bool(np.all(x <= self.xmax))

        return test_min and test_max

    def save_best(self, x: NDArray[np.float64]) -> None:
        """
        Create a spin-system object from the best parameter vector and save it.

        Parameters
        ----------
        x : np.array
            Current guess.

        Returns
        -------
        None
            No return value.

        """
        Sys = guess2Sys(x, self.def_Sys, self.Var, self.SimOpt)
        Sys.save_spinsystem("basinhopping_current_best")

        return None
