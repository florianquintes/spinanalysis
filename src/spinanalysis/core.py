#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provide spectrum simulation and spin-system optimization workflows.

The module exposes :func:`simulate` for generating spectra with the supported
simulation routines and :func:`optimize` for fitting spin-system parameters.

© M. Sc. Florian Quintes, 2021-2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import numpy as np

from spinanalysis.epr import (
    Experimental,
    FittingOptions,
    SimulationOptions,
    Spinsystem,
    Variation,
)


def _set_mkl_threads(n: int = 1) -> None:
    """Restrict MKL to *n* threads to avoid oversubscription in parallel runs."""
    try:
        import mkl

        mkl.set_num_threads(n)
    except ImportError:
        pass


_set_mkl_threads()


def simulate(
    Sys: Spinsystem,
    Exp: Experimental,
    SimOpt: SimulationOptions,
) -> np.ndarray:
    """Simulate a spectrum using the selected simulation routine.

    Parameters
    ----------
    Sys : Spinsystem
        Spin-system parameters from :mod:`spinanalysis.epr`.
    Exp : Experimental
        Experimental parameters from :mod:`spinanalysis.epr`. The simulated
        spectrum is stored in ``Exp.spec_sim``.
    SimOpt : SimulationOptions
        Simulation options from :mod:`spinanalysis.epr`.

    Side Effects
    ------------
    ``SimOpt.mode`` is set to ``"simulation"`` before dispatch.  For the
    ``teacups`` routine with ``eigval_mode`` *False* the mode is overridden to
    ``"fitting"`` because teacups internally switches between the two.
    ``Exp.spec_sim`` is always set to the returned (normalized) spectrum.

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
            from static_radical_pair.radpair import do_simulation_multicore

            simulated_spectra = do_simulation_multicore(Sys, Exp, SimOpt)
        case "teacups":
            from teacups.simulations import teacups

            if SimOpt.eigval_mode:
                teacups(Sys, Exp, SimOpt)
                simulated_spectra = np.ones((Exp.t_points, len(Exp.B_z)))
            else:
                SimOpt.mode = "fitting"
                simulated_spectra = teacups(Sys, Exp, SimOpt)
        case "opossum":
            from oop_eseem.opossum import oop_eseem

            simulated_spectra = oop_eseem(Sys, Exp, SimOpt)
        case "didelphis":
            from oop_eseem.opossum import oop_eseem_distribution

            simulated_spectra = oop_eseem_distribution(Sys, Exp, SimOpt)
        case "didelphis_tikhonov":
            from oop_eseem.opossum import oop_eseem_distance_distribution

            simulated_spectra = oop_eseem_distance_distribution(Sys, Exp, SimOpt)
        case _:
            raise ValueError("Can't find a routine named '{0}'!".format(SimOpt.routine))

    max_val = abs(simulated_spectra).max()
    if max_val > 0:
        simulated_spectra = simulated_spectra / max_val
    Exp.spec_sim = simulated_spectra

    return simulated_spectra


def optimize(
    Sys: Spinsystem,
    Exp: Experimental,
    SimOpt: SimulationOptions,
    FitOpt: FittingOptions,
    Var: Variation,
) -> Spinsystem:
    """Optimize spin-system parameters using the selected routine.

    Parameters
    ----------
    Sys : Spinsystem
        Spin-system parameters from :mod:`spinanalysis.epr`.
    Exp : Experimental
        Experimental parameters from :mod:`spinanalysis.epr`.
    SimOpt : SimulationOptions
        Simulation options from :mod:`spinanalysis.epr`.
    FitOpt : FittingOptions
        Fitting options from :mod:`spinanalysis.epr`.
    Var : Variation
        Variation parameters from :mod:`spinanalysis.epr`.

    Side Effects
    ------------
    ``SimOpt.mode`` is set to ``"fitting"`` before dispatch.

    Raises
    ------
    ValueError
        If ``FitOpt.routine`` is not supported.

    Returns
    -------
    Spinsystem
        The best spin-system parameters found during optimization.
    """
    SimOpt.mode = "fitting"
    routine = FitOpt.routine.lower()

    match routine:
        case "genetic":
            from genetic_radpair.genetic_classes import Genetic_Radpair

            genetic_radpair = Genetic_Radpair(Sys, Exp, SimOpt, FitOpt, Var)
            best_spinsystem = genetic_radpair.best_spinsystem
        case "dual_annealing":
            import spinanalysis._interface_handler as spo

            best_spinsystem = spo.dualannealing(Sys, Exp, SimOpt, FitOpt, Var)
        case "shgo":
            import spinanalysis._interface_handler as spo

            best_spinsystem = spo.shgo(Sys, Exp, SimOpt, FitOpt, Var)
        case "differential_evolution":
            import spinanalysis._interface_handler as spo

            best_spinsystem = spo.differential_evolution(Sys, Exp, SimOpt, FitOpt, Var)
        case "basinhopping":
            import spinanalysis._interface_handler as spo

            best_spinsystem = spo.basinhopping(Sys, Exp, SimOpt, FitOpt, Var)
        case "least_squares":
            import spinanalysis._interface_handler as spo

            best_spinsystem = spo.least_squares(Sys, Exp, SimOpt, FitOpt, Var)
        case "minimize":
            import spinanalysis._interface_handler as spo

            best_spinsystem = spo.minimize(Sys, Exp, SimOpt, FitOpt, Var)
        case _:
            raise ValueError(
                "Can't find an optimization routine named '{0}'!".format(FitOpt.routine)
            )
    return best_spinsystem
