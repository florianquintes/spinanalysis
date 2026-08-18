#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provide spectrum simulation and spin-system optimization workflows.

The module exposes :func:`simulate` for generating spectra with the supported
simulation routines and :func:`optimize` for fitting spin-system parameters.

© M. Sc. Florian Quintes, 2026

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
import numpy as np

version = "v0.1.0"


mkl.set_num_threads(1)


def simulate(Sys: object, Exp: object, SimOpt: object) -> np.ndarray:
    """Simulate a spectrum using the selected simulation routine.

    Parameters
    ----------
    Sys : object
        Spin-system parameters from :mod:`spinanalysis.epr`.
    Exp : object
        Experimental parameters from :mod:`spinanalysis.epr`. The simulated
        spectrum is stored in ``Exp.spec_sim``.
    SimOpt : object
        Simulation options from :mod:`spinanalysis.epr`.

    Raises
    ------
    ValueError
        If ``SimOpt.routine`` is not supported.

    Returns
    -------
    numpy.ndarray
        The normalized simulated spectrum.
    """
    SimOpt.mode = "simulation"
    routine = SimOpt.routine.lower()

    match routine:
        case "static_radpair":
            simulated_spectra = do_simulation_multicore(Sys, Exp, SimOpt)
        case "teacups":
            match SimOpt.eigval_mode:
                case True:
                    teacups(Sys, Exp, SimOpt)
                    simulated_spectra = np.ones((Exp.t_points, len(Exp.B_z)))
                case _:
                    SimOpt.mode = "fitting"
                    simulated_spectra = teacups(Sys, Exp, SimOpt)
        case "opossum":
            simulated_spectra = oop_eseem(Sys, Exp, SimOpt)
        case "didelphis":
            simulated_spectra = oop_eseem_distribution(Sys, Exp, SimOpt)
        case "didelphis_tikhonov":
            simulated_spectra = oop_eseem_distance_distribution(Sys, Exp, SimOpt)
        case _:
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
    """Optimize spin-system parameters using the selected routine.

    Parameters
    ----------
    Sys : object
        Spin-system parameters from :mod:`spinanalysis.epr`.
    Exp : object
        Experimental parameters from :mod:`spinanalysis.epr`.
    SimOpt : object
        Simulation options from :mod:`spinanalysis.epr`.
    FitOpt : object
        Fitting options from :mod:`spinanalysis.epr`.
    Var : object
        Variation parameters from :mod:`spinanalysis.epr`.

    Raises
    ------
    ValueError
        If ``FitOpt.routine`` is not supported.

    Returns
    -------
    object
        The best spin-system parameters found during optimization.
    """
    SimOpt.mode = "fitting"
    routine = FitOpt.routine.lower()

    match routine:
        case "genetic":
            genetic_radpair = Genetic_Radpair(Sys, Exp, SimOpt, FitOpt, Var)
            best_spinsystem = genetic_radpair.best_spinsystem
        case "dual_annealing":
            best_spinsystem = spo.dualannealing(Sys, Exp, SimOpt, FitOpt, Var)
        case "shgo":
            best_spinsystem = spo.shgo(Sys, Exp, SimOpt, FitOpt, Var)
        case "differential_evolution":
            best_spinsystem = spo.differential_evolution(Sys, Exp, SimOpt, FitOpt, Var)
        case "basinhopping":
            best_spinsystem = spo.basinhopping(Sys, Exp, SimOpt, FitOpt, Var)
        case "least_squares":
            best_spinsystem = spo.least_squares(Sys, Exp, SimOpt, FitOpt, Var)
        case "minimize":
            best_spinsystem = spo.minimize(Sys, Exp, SimOpt, FitOpt, Var)
        case _:
            raise ValueError(
                "Can't find an optimization routine named  '{0}'".format(FitOpt.routine)
            )
    return best_spinsystem
