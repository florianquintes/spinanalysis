#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import mkl
import spinanalysis._interface_handler as spo
from genetic_radpair.genetic_classes import Genetic_Radpair
from oop_eseem.opossum import (
    oop_eseem,
    oop_eseem_distance_distribution,
    oop_eseem_distribution,
)
from teacups.simulations import teacups
from static_radical_pair.radpair import do_simulation_multicore
import sys
import os
import datetime
import logging
import numpy as np

version = "v0.1.0"


mkl.set_num_threads(1)


def start_log() -> None:
    # TODO
    """
    Diese Funktion soll mal Log-Dateien während der Nutzung anlegen.
    Aktuell nicht nutzbar! Reine Baustelle!
    """
    date = datetime.datetime.now()

    date = date.strftime("%d_%m_%Y__%H:%M:%S")
    session_name = "EasyPairSpin_Session_at_" + date + ".log"
    log_path = os.path.join(sys.prefix, "easypairspin", "logs")

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logfile_name = os.path.join(log_path, session_name)

    fmt = "{asctime} - [{levelname:8}] - {module} {funcName} - {message}"
    dfmt = "%d/%m/%Y %H:%M:%S"

    logging.basicConfig(
        filename=logfile_name,
        style="{",
        format=fmt,
        datefmt=dfmt,
        level=logging.DEBUG,
    )
    logging.info("Log startet")
    return None


def simulate(Sys: object, Exp: object, SimOpt: object) -> np.ndarray:
    # TODO Hinschreiben, welche Simulationen möglich sind + Paper.
    """
    Do various simulations with (spinpolarized) radical pairs.

    Parameters
    ----------
    Sys : object
        Spinsystem object of module 'epr_setup'.
    Exp : object
        Experimental object of module 'epr_setup'. simulated_spectra will be
        saved in Exp.spec_sim.
    SimOpt : object
        SimulationOptions object of module 'epr_setup'.

    Raises
    ------
    ValueError
        Will be raised, if the given simulation routine in SimOpt.routine is an
        invalid string.

    Returns
    -------
    simulated_spectra : np.ndarray
        Simulated spectra as a numpy.ndarray.

    Examples
    --------

    Basic simulation using static_radical_pair:

    >>> from epr_setup import Spinsystem, Experimental, SimulationOptions
    >>> Sys = Spinsystem()
    >>> Exp = Experimental()
    >>> SimOpt = SimulationOptions()
    >>> SimOpt.routine = 'static_radpair'
    >>> simulate(Sys, Exp, SimOpt)

    Plot your result:

    >>> from plotting import plot_2D
    >>> plot_2D(Exp.B_z , Exp.spec_sim)

    """
    SimOpt.mode = "simulation"

    # TODO an match case anpassen PYTHON 3.10
    if SimOpt.routine.lower() == "static_radpair":
        simulated_spectra = do_simulation_multicore(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "teacups":
        if SimOpt.eigval_mode is True:
            teacups(Sys, Exp, SimOpt)
            simulated_spectra = np.ones((Exp.t_points, len(Exp.B_z)))
        else:
            SimOpt.mode = "fitting"
            simulated_spectra = teacups(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "opossum":
        simulated_spectra = oop_eseem(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "didelphis":
        simulated_spectra = oop_eseem_distribution(Sys, Exp, SimOpt)
    elif SimOpt.routine.lower() == "didelphis_tikhonov":
        simulated_spectra = oop_eseem_distance_distribution(Sys, Exp, SimOpt)
    else:
        raise ValueError("Can't find a routine named '{0}'!".format(SimOpt.routine))

    simulated_spectra /= (abs(simulated_spectra)).max()
    Exp.spec_sim = simulated_spectra

    return simulated_spectra


def optimize(
    Sys: object,
    Exp: object,
    SimOpt: object,
    FitOpt: object,
    Var: object,
) -> object:
    # Hinschreiben welche Optimierungen möglich sind + Paper.
    """
    Do various optimizations with all simulations available in
    'easypairspin()'.

    Parameters
    ----------
    Sys : object
        Spinsystem object of module 'epr_setup'.
    Exp : object
        Experimental object of module 'epr_setup'.
    SimOpt : object
        SimulationOptions object of module 'epr_setup'.
    FitOpt : object
        FittingOptions object of module 'epr_setup'.
    Var : object
        Variation object of module 'epr_setup'.

    Raises
    ------
    ValueError
        Will be raised, if the given simulation routine in FitOpt.routine is an
        invalid string.

    Returns
    -------
    best_Spinsystem: object
        The best Spinsystem found during optimization. Object is of class
        Spinsystem of module 'epr_setup'.

    Examples
    --------

    Basic optimization using genetic_radpair and static_radical_pair:

    >>> from epr_setup import Spinsystem, Experimental, SimulationOptions,
    ... FittingOptions, Variation
    >>> Sys = Spinsystem()
    >>> Exp = Experimental()
    >>> SimOpt = SimulationOptions()
    >>> SimOpt.routine = 'static_radpair'
    >>> FitOpt = FittingOptions()
    >>> Var = Variation()
    >>> Var.g1 = np.array([0.001, 0.003, 0.002])
    >>> best_Sys = optimize(Sys, Exp, SimOpt, FitOpt, Var)

    Plot your result:

    >>> from plotting import plot_2D
    >>> simulate(best_Sys, Exp, SimOpt)
    >>> plot_2D(Exp.B_z , Exp.spec_sim)


    """
    # TODO an match case anpassen PYTHON 3.10

    SimOpt.mode = "fitting"

    if FitOpt.routine.lower() == "genetic":
        Gen_Rad = Genetic_Radpair(Sys, Exp, SimOpt, FitOpt, Var)
        best_spinsystem = Gen_Rad.best_spinsystem
    elif FitOpt.routine.lower() == "dual_annealing":
        best_spinsystem = spo.dualannealing(Sys, Exp, SimOpt, FitOpt, Var)
    elif FitOpt.routine.lower() == "shgo":
        best_spinsystem = spo.shgo(Sys, Exp, SimOpt, FitOpt, Var)
    elif FitOpt.routine.lower() == "differential_evolution":
        best_spinsystem = spo.differential_evolution(Sys, Exp, SimOpt, FitOpt, Var)
    elif FitOpt.routine.lower() == "basinhopping":
        best_spinsystem = spo.basinhopping(Sys, Exp, SimOpt, FitOpt, Var)
    elif FitOpt.routine.lower() == "least_squares":
        best_spinsystem = spo.least_squares(Sys, Exp, SimOpt, FitOpt, Var)
    elif FitOpt.routine.lower() == "minimize":
        best_spinsystem = spo.minimize(Sys, Exp, SimOpt, FitOpt, Var)
    else:
        raise ValueError(
            "Can't find an optimization routine named  '{0}'".format(FitOpt.routine)
        )
    return best_spinsystem
