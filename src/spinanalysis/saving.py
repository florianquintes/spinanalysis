#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import numpy as np
import os


def save_plot(fname: str, *figures: object, path: str = None, **kwargs: dict) -> None:
    """
    Save the figures plotted with matplotlib.

    Parameters
    ----------
    fname : str
        Filename for the figure(s). If multiple figures are given, '_[number]'
        will be append to the filename.
    *figures : object
        Matplotlib figure object(s).
    path : str, optional
        Absolute path where the figures will be stored. The default is
        '~/Results/Plots/'.
    **kwargs : dict
        Other keyword arguments. Will be passed to plt.savefig(). See
        matplotlib documentation for further informations.

    Returns
    -------
    None
        Nothing will be returned.

    """
    # [SETUP]
    if fname[-4] == ".":
        fmt = fname[-4:]
        fname = fname[:-4]
    elif fname[-5] == ".":
        fmt = fname[-5:]
        fname = fname[:-5]
    else:
        fmt = ""

    if type(figures[0]) == list:
        figures = tuple(*figures)

    if len(figures) == 1:
        multi_plot = False
    else:
        multi_plot = True

    if path is None:
        save_path = os.path.join(os.path.expanduser("~"), "Results", "Plots")
    else:
        save_path = path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # [SAVE]
    for n, fig in enumerate(figures):
        if multi_plot:
            save_name = os.path.join(save_path, fname + "_" + str(n + 1) + fmt)
        else:
            save_name = os.path.join(save_path, fname + fmt)

        if "format" in kwargs:
            save_name += "." + kwargs["format"]

        fig.savefig(save_name, **kwargs)

    return None


def save_simulation(name: str, *data: "np.array", path: str = None) -> None:
    """
    Save the simulated data at '[path]/[name]/[files]' using np.savetxt.

    Parameters
    ----------
    name : str
        Foldername for the dataset.
    *data : np.array
        Arrays with the simulated data. Must be 2d or 3d. If 2d: x_axis, int;
        if 3d: x_axis, y_axis, int.
    path : str, optional
        Full path where the data will be stored. The default is
        '~/Results/Simulated Data/[name]'.

    Raises
    ------
    ValueError
        Will be raised, if dimension of the data isn't 2d or 3d.

    Returns
    -------
    None
        Nothing will be returned.

    """
    # [SETUP]
    if name.lower().endswith(".txt"):
        name = name[:-4]
    if path is None:
        save_path = os.path.join(os.path.expanduser("~"), "Results", "Simulated Data")
    else:
        save_path = path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_folder = os.path.join(save_path, name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if len(data) == 2:
        x, intensity = data
    elif len(data) == 3:
        x, y, intensity = data
    else:
        raise ValueError("Can't handle {0}D Data. Need 2D or 3D.".format(len(data)))

    # [SAVE]

    np.savetxt(os.path.join(save_folder, "x_axis.txt"), x)
    np.savetxt(os.path.join(save_folder, "intensity.txt"), intensity)

    if len(data) == 3:
        np.savetxt(os.path.join(save_folder, "y_axis.txt"), y)

    return None


def write_out_file(
    Sys: object,
    Exp: object,
    SimOpt: object,
    *FitOpt: object,
    current_best: bool = False,
) -> None:
    """
    Write an output file with all datas from Sys, Exp, SimOpt and, if running
    in optimization mode, FitOpt.

    Parameters
    ----------
    Sys : object
        Spinsystem object of module 'epr_setup'.
    Exp : object
        Experimental object of module 'epr_setup'.
    SimOpt : object
        SimulationOptions object of module 'epr_setup'.
    *FitOpt : object
        FittingOptions object of module 'epr_setup'.
    current_best : bool, optional
        True if the given Sys and Exp are the current best while running in
        optimization mode. False if Sys and Exp are the final result / are the
        given Sys and Exp in normal simulation mode. The default is False.

    Returns
    -------
    None
        Nothing will be returned.

    """

    if current_best:
        title_part = "_current_best"
    else:
        title_part = "_result"

    # TODO Pfad überlegen um Ergebnisse zu speichern?
    if len(FitOpt) == 1:
        fit_mode = True
        out = open(FitOpt[0].routine + title_part + ".eps_out", "w")
    else:
        fit_mode = False
        out = open(SimOpt.routine + title_part + ".eps_out", "w")

    out.write("{:#^50s}\n".format(""))

    if current_best:
        out.write("{:#^50s}\n".format(" CURRENT BEST "))
    else:
        out.write("{:#^50s}\n".format(" OUTPUT-FILE "))

    if fit_mode:
        out.write("{:#^50s}\n".format(" " + FitOpt[0].routine + " "))
    else:
        out.write("{:#^50s}\n".format(" " + SimOpt.routine + " "))

    out.write("{:#^50s}\n".format(" AUTHOR: FLORIAN QUINTES "))
    out.write("{:#^50s}\n".format(""))
    out.write("\n")

    out.write("\n")
    out.write("\n")
    out.write("{:#^50s}\n".format(" SPINSYSTEM "))
    out.write("\n")
    out.write("\n")

    out.write("{: ^15s} {: ^20s}\n".format("g-Tensor", "Value"))
    np.set_printoptions(formatter={"float": "{:0.7f}".format})

    for key in vars(Sys):
        if key in ("g1", "g2"):
            out.write("{0:>14s}: {1}\n".format(key, vars(Sys)[key]))

    out.write("\n")
    out.write("{: ^15s} {: ^20s}\n".format("A-Tensor", "Value / MHz"))
    np.set_printoptions(formatter={"float": "{:2.2f}".format})

    for key in vars(Sys):
        if key.startswith("A") and len(key) < 3:
            out.write("{0:>14s}: {1}\n".format(key, vars(Sys)[key]))

    out.write("\n")
    out.write("{: ^15s} {: ^20s}\n".format("ZFS", "Value / MHz"))

    for key in vars(Sys):
        if key in ("D", "E", "J_ex"):
            out.write("{0:>14s}: {1}\n".format(key, vars(Sys)[key]))

    out.write("\n")
    out.write("{: ^15s} {: ^20s}\n".format("Orientations", "Angle / rad"))
    np.set_printoptions(formatter={"float": "{: 0.1f}".format})

    for key in vars(Sys):
        if key.endswith("_frame"):
            out.write("{0:>14s}: {1}\n".format(key, vars(Sys)[key]))

    out.write("\n")
    out.write("{: ^15s}|{: ^20s}\n".format("Nuclear spin", "Number of cores"))

    spins = []
    cores = []
    for key in vars(Sys):
        if (key.startswith("n") or key.startswith("I")) and len(key) < 3:
            if key.startswith("n"):
                cores.append(key)
            else:
                spins.append(key)
    cores.sort()
    spins.sort()

    atoms = zip(cores, spins)
    for core, spin in atoms:
        out.write(
            "{0:>7s}: {1:<6.1f}|{2:>9s}: {3:<9}\n".format(
                core, vars(Sys)[core], spin, vars(Sys)[spin]
            )
        )

    out.write("\n")
    out.write("{:>14s}: {:.2f} mT\n".format("width_gauss", Sys.width_gauss))

    out.write("\n")
    out.write("\n")
    out.write("{:#^50s}\n".format(" EXPERIMENTAL "))
    out.write("\n")
    out.write("\n")

    out.write("{0:>20s}: {1:<9.6f} mT\n".format("microwave amplitude", Exp.B_mw))
    out.write(
        "{0:>20s}: {1:<9.6f} GHz\n".format("microwave frequency", Exp.freq_mw / 1e9)
    )
    out.write("{0:>20s}: {1:<7.2f} mT\n".format("min B_z", Exp.B_z.min()))
    out.write("{0:>20s}: {1:<7.2f} mT\n".format("max B_z", Exp.B_z.max()))
    out.write("{0:>20s}: {1}\n".format("B_z points", len(Exp.B_z)))

    out.write("\n")
    out.write("\n")
    out.write("{:#^50s}\n".format(" SIMULATION OPTIONS "))
    out.write("\n")
    out.write("\n")

    out.write("{0:>20s}: {1:<20s}\n".format("routine", SimOpt.routine))
    out.write("{0:>20s}: {1} cores\n".format("run at", SimOpt.cpu_cores))
    out.write("{0:>20s}: {1:<20}\n".format("grid points", SimOpt.grid_points))
    out.write("{0:>20s}: {1} space\n".format("using", SimOpt.space))

    if fit_mode:
        out.write("\n")
        out.write("\n")
        out.write("{:#^50s}\n".format(" FITTING OPTIONS "))
        out.write("\n")
        out.write("\n")

        for key in vars(FitOpt[0]):
            if key is not None:
                out.write(
                    ("{0:>20s}: " + "{1:<20s}\n").format(key, str(vars(FitOpt[0])[key]))
                )

    out.close()

    return None
