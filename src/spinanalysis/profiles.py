#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

from configobj import ConfigObj
from matplotlib.pyplot import style
from spinanalysis._utils import strtobool
import validate
import os
import sys
from pathlib import Path
from zipfile import ZipFile


PROFILE_ROOT = Path(
    os.environ.get(
        "SPINANALYSIS_PROFILE_DIR",
        Path.home() / ".config" / "spinanalysis" / "profiles",
    )
)


def import_profiles(zipfile: str, override: bool = False) -> None:
    """
    Import profiles from a zip archive.

    Parameters
    ----------
    zipfile : str
        Path to the zip archive.
    override : bool, optional
        If True, existing profile with the same name will be overriden. The
        default is False.

    Returns
    -------
    None
        Nothing will be returned.

    """
    zipfile = os.path.abspath(zipfile)
    profiles_folder = os.path.join(sys.prefix, "easypairspin", "profiles")

    with ZipFile(zipfile, "r") as zipfile:
        for file in zipfile.namelist():
            if os.path.isfile(os.path.join(profiles_folder, file)) and not override:
                continue

            zipfile.extract(file, profiles_folder)

        zipfile.close()


def export(path: str = None, pkind: [str, list] = "all", pname: str = "all") -> None:
    """
    Export the choosen profile(s) as a zip archive.

    Parameters
    ----------
    path : str, optional
        Path where the zip file will be stored. If no path given, the zip file
        will be stored in the current working directory. The default is None.
    pkind : str, optional
        Define which kind(s) of profiles should be exported. Multiple kinds of
        profiles possible. Options are 'plot', 'spinsystem', 'optimization',
        'save', 'variation', 'simulation' and 'all'. The default is 'all'.
    pname : str, optional
        Give the basename of the profile. The default is 'all'.

    Returns
    -------
    None
        Nothing will be returned.

    """
    profiles_folder = os.path.join(sys.prefix, "easypairspin", "profiles")

    if pkind == "all":
        pkind = [
            "plot",
            "spinsystem",
            "optimization",
            "simulation",
            "variation",
            "save",
        ]
    else:
        if not isinstance(pkind, list):
            stack = []
            for el in pkind.lower().strip().split():
                if el in (
                    "plot",
                    "spinsystem",
                    "optimization",
                    "simulation",
                    "variation",
                    "save",
                ):
                    stack.append(el)
                elif el == "optimisation":
                    stack.append("optimization")
                else:
                    continue
            pkind = stack

        else:
            stack = []
            for el in pkind:
                if el.lower() in (
                    "plot",
                    "spinsystem",
                    "optimization",
                    "simulation",
                    "variation",
                    "save",
                ):
                    stack.append(el.lower())
                elif el.lower() == "optimisation":
                    stack.append("optimization")
                else:
                    continue
            pkind = stack

    file_paths = []
    for kind in pkind:
        file_paths.extend(_get_profile_paths(os.path.join(profiles_folder, kind)))

    if path is None:
        path = os.path.join(os.getcwd(), "profiles_easypairspin.zip")
    else:
        path = os.path.join(path, "profiles_easypairspin.zip")

    with ZipFile(path, "w") as zipfile:
        for file in file_paths:
            arcname = file[len(profiles_folder) + 1 :]
            zipfile.write(file, arcname)

        zipfile.close()


def _get_profile_paths(pfolder: str, pname: str = "all") -> None:
    """
    Get the pathes of all profile.

    Parameters
    ----------
    pfolder: : str
        Folder with profiles.
    pname : str, optional
        Basename of the profile. The default is "all".

    Returns
    -------
    None
        Nothing will be returned.

    """
    paths = []

    for root, directories, files in os.walk(pfolder):
        for file in files:
            if file.endswith("configspec.ini"):
                continue

            if pname.lower() == "all":
                paths.append(os.path.join(pfolder, file))
            elif pname.lower() == os.path.basename(file):
                paths.append(os.path.join(pfolder, file))
            else:
                continue

    return paths


def add_profile(profile: dict, pkind: str, pname: str = "") -> None:
    """
    Add a new profile for EasyPairSpin.

    Parameters
    ----------
    profile : dict
        Dictionary with all profile settings.
    pkind : str
        Give the kind of the kind of the profile. Not case sensitive. pkind can
        be 'plot', 'save', 'simulation', 'optimization', 'spinsystem' or
        'variation'.
    pname : str, optional
        Name of the profile.  If no profile name is given, a default one will
        be generated by _get_profile_name(). The default is ''.


    Raises
    ------
    ValueError
        Raised if pkind isn't 'plot', 'save', 'simulation', 'optimization',
        'spinsystem' or 'variation'.

    Returns
    -------
    None
        Nothing will be returned.

    Examples
    --------
    Creating and adding a new profile:

    >>> Sys = epr_setup.Spinsystem()
    >>> Sys_profile = new_spinsystem_profile()
    >>> Sys_profile['g_1'] = [2.0034, 2.00156, 2.00228]
    >>> prom.add_profile(Sys.profile, 'spinsystem', 'Sys_prof_1')

    """
    pkind = pkind.lower()
    if pkind == "optimisation":
        pkind = "optimization"

    if pkind not in (
        "plot",
        "save",
        "simulation",
        "optimization",
        "spinsystem",
        "variation",
    ):
        raise ValueError(
            "pkind must be 'plot', 'save', 'simulation',"
            " 'optimization', 'spinsystem' or 'variation'! "
        )

    if pname == "":
        pname = _get_profile_name(pkind)

    config_path = PROFILE_ROOT / pkind
    if pkind == "plot":
        config_name = config_path / pname
    else:
        config_name = config_path / (pname + ".ini")
        path_to_configspec = config_path / "configspec.ini"

        config_path.mkdir(parents=True, exist_ok=True)

        config = ConfigObj(str(config_name), configspec=str(path_to_configspec))

    # TODO Exception Handling, ob alle relevanten Werte (korrekt) gegeben sind
    # TODO match case Struktur einführen (wenn Python 3.10 möglich)
    if pkind == "plot":
        _save_config_plot(profile, config_name)
    elif pkind == "spinsystem":
        config = _set_configobj_spinsystem(profile, config)
    elif pkind == "variation":
        config = _set_configobj_variation(profile, config)
    elif pkind == "optimization":
        config = _set_configobj_optimization(profile, config)
    elif pkind == "save":
        config = _set_configobj_save(profile, config)
    elif pkind == "simulation":
        config = _set_configobj_simulation(profile, config)

    if not (pkind == "plot"):
        validator = validate.Validator()
        config.validate(validator)
        config.write()

    return None


def _save_config_plot(profile: dict, path: str) -> object:
    """
    Write profile parameters into the ConfigObject for a plot profile.

    Parameters
    ----------
    profile : dict
        Plot profile.
    path : str
        Path, where the profile will be stored.

    Returns
    -------
    object
        ConfigObj from module configobj.

    """
    preamble = "## {:*^76s}\n## {:*^76s}\n## {:*^76s}\n".format(
        "", " EASYPAIRSPIN SPECIAL SETTINGS ", ""
    )
    with open(path, "w") as f:
        f.write(preamble)
        f.write("\n")
        for key in profile:
            if key.endswith("lim"):
                line = "{}: {}, {}".format(
                    str(key), str(profile[key][0]), str(profile[key][1])
                )
            else:
                if isinstance(profile[key], (int, float, bool)):
                    line = "{}: {}".format(str(key), str(profile[key]))
                else:
                    line = '{}: "{}"'.format(str(key), str(profile[key]))
            f.write(line)
            f.write("\n")
        f.close()


def _set_configobj_spinsystem(profile: dict, ConObj: object) -> object:
    """
    Write profile parameters into the ConfigObject for a spinsystem profile.

    Parameters
    ----------
    profile : dict
        Spinsystem profile.
    ConObj : object
        ConfigObj from module configobj.

    Returns
    -------
    object
        ConfigObj from module configobj.

    """
    ConObj["main"] = profile["main"]

    return ConObj


def _set_configobj_variation(profile: dict, ConObj: object) -> object:
    """
    Write profile parameters into the ConfigObject for a variation profile.

    Parameters
    ----------
    profile : dict
        Variation profile.
    ConObj : object
        ConfigObj from module configobj.

    Returns
    -------
    object
        ConfigObj from module configobj.

    """
    ConObj["main"] = profile["main"]

    return ConObj


def _set_configobj_optimization(profile: dict, ConObj: object) -> object:
    """
    Write profile parameters into the ConfigObject for an optimization profile.

    Parameters
    ----------
    profile : dict
        Optimization profile.
    ConObj : object
        ConfigObj from module configobj.

    Raises
    ------
    ValueError
        If no valid optimization routine is given.

    Returns
    -------
    object
        ConfigObj from module configobj.

    """
    ConObj["main"] = profile["main"]

    routine = profile["main"]["routine"]
    if routine in [
        "genetic",
        "minimize",
        "dual_annealing",
        "shgo",
        "differential_evolution",
        "basinhopping",
        "least_squares",
    ]:
        ConObj[routine] = profile[routine]
    else:
        raise ValueError("{} is no valid optimization routine!".format(routine))

    return ConObj


def _set_configobj_simulation(profile: dict, ConObj: object) -> object:
    """
    Write profile parameters into the ConfigObject for a simulation  profile.

    Parameters
    ----------
    profile : dict
        Simulation profile.
    ConObj : object
        ConfigObj from module configobj.

    Raises
    ------
    ValueError
        If no valid simulation routine is given.

    Returns
    -------
    object
        ConfigObj from module configobj.

    """
    ConObj["main"] = profile["main"]

    routine = profile["main"]["routine"]
    if routine in ["static_radpair", "teacups", "opossum", "didelphis"]:
        ConObj[routine] = profile[routine]
    else:
        raise ValueError("{} is no valid simulation routine!".format(routine))

    return ConObj


def _set_configobj_save(profile: dict, ConObj: object) -> object:
    """
    Write profile parameters into the ConfigObject for a save profile.

    Parameters
    ----------
    profile : dict
        Save profile.
    ConObj : object
        ConfigObj from module configobj.

    Returns
    -------
    object
        ConfigObj from module configobj.

    """
    ConObj["main"] = profile["main"]

    return ConObj


def _get_profile_name(pkind: str) -> str:
    """
    Search for the smallest number available for the default profile name.

    Scheme for default profile name is 'profile_[number]'

    Parameters
    ----------
    pkind : str
        Give the kind of the kind of the profile. Not case sensitive. pkind can
        be 'plot', 'save', 'simulation', 'optimization', 'spinsystem' or
        'variation'.

    Returns
    -------
    name : str
        String with the available profile name e.g. 'profile_12'.

    """
    name = "profile_"
    suffix = "" if pkind == "plot" else ".ini"
    profile_number = 0
    profile_number_used = True

    while profile_number_used:
        profile_number += 1
        for root, dirs, files in os.walk(
            os.path.join(sys.prefix, "easypairspin", "profiles", pkind)
        ):
            counter = 0
            for file in files:
                if file.endswith("profile_" + str(profile_number) + suffix):
                    counter += 1

            profile_number_used = counter

    name += str(profile_number)

    return name


def load_profile(pname: str, pkind: str) -> dict:
    """
    Load a given plotting profile.

    Parameters
    ----------
    pname : str
        Name of the profile. Case sensitive. Either with .ini or not.
        E. g.: load_profile('test') or load_profile('test.ini').
    pkind : str
        Give the kind of the profile. Not case sensitive. pkind can be 'save',
        'simulation', 'optimization', 'spinsystem' or 'variation'.

    Returns
    -------
    profile: dict
        Loaded profile as a dictionary.

    """
    if pname.endswith(".ini"):
        pname = pname[:-4]

    path_to_profile = PROFILE_ROOT / pkind / (pname + ".ini")
    path_to_configspec = PROFILE_ROOT / pkind / "configspec.ini"

    config = ConfigObj(
        str(path_to_profile), configspec=str(path_to_configspec), file_error=True
    )
    validator = validate.Validator()
    config.validate(validator)

    profile = {}

    for section in config.sections:
        profile[section] = config[section]

    return profile


def load_plot_profile(pname: str) -> dict:
    """
    Load a plotting profile from a mplstylesheet.

    Parameters
    ----------
    pname : str
        Name of the profile. Case sensitive.

    Returns
    -------
    dict
        Contains the settings for the plotting functions.

    """
    if pname is None or pname in style.available:
        pname = "default_stylesheet"

    path_to_profile = os.path.join(
        sys.prefix, "easypairspin", "profiles", "plot", pname
    )

    profile = {}
    with open(path_to_profile, "r") as file:
        for line in file.readlines():
            if line.startswith("#"):
                continue
            else:
                line = line.strip().split(":")
                if line[0] in (
                    "percentage_mode",
                    "xlim",
                    "ylim",
                    "zlim",
                    "xlabel",
                    "ylabel",
                    "zlabel",
                    "show_title",
                    "title",
                    "legend",
                    "colorbar",
                ):
                    if "lim" in line[0] and len(line[0]) == 4:
                        bounds = line[1].split("#")[0].strip()
                        bounds = bounds.split(",")
                        lb = float(bounds[0])
                        ub = float(bounds[1])
                        profile[line[0]] = [lb, ub]
                    elif ("label" in line[0] and len(line[0]) == 6) or (
                        line[0] == "title"
                    ):
                        label = line[1].split("#")[0].strip()
                        label = label.strip('"').strip("'")
                        profile[line[0]] = label
                    else:
                        boolean = line[1].split("#")[0].strip()
                        boolean = boolean.strip('"').strip("'")
                        profile[line[0]] = bool(strtobool(boolean))

    return profile


def new_plot_profile() -> dict:
    """
    Get an empty plotting profile.

    Returns
    -------
    default_profile: dict
        Dictionary with default settings for plottings.

    """
    default_profile = {
        "percentage_mode": False,
        "xlim": [0, 0],
        "ylim": [0, 0],
        "zlim": [0, 0],
        "xlabel": "",
        "ylabel": "",
        "zlabel": "",
        "legend": "off",
        "show_title": "off",
        "title": "",
        "colorbar": "on",
    }

    return default_profile


def new_spinsystem_profile() -> dict:
    """
    Get a default spinsystem profile.

    Returns
    -------
    default_profile: dict
        Dictionary with default settings for a spinsystem.

    """
    default_profile = {
        "main": {
            "spin_system": "rp",
            "precursor": "triplet-zf",
            "population": [1.0, 0.0, 0.0],
            "g1": [2.002, 2.002, 2.002],
            "g2": [2.004, 2.004, 2.004],
            "g_tri": [2.002, 2.002, 2.002],
            "g": [2.004, 2.004, 2.004],
            "g1_frame": [0, 0, 0],
            "g2_frame": [0, 0, 0],
            "g_tri_frame": [0, 0, 0],
            "g_frame": [0, 0, 0],
            "width_gauss": 0.5,
            "acceptor_list": [1, 2, 3],
            "donor_list": [4, 5],
            "A1": [0, 0, 0],
            "A2": [0, 0, 0],
            "A3": [0, 0, 0],
            "A4": [0, 0, 0],
            "A5": [0, 0, 0],
            "A_eseem": 0.0,
            "omega_I": 0.0,
            "A1_frame": [0, 0, 0],
            "A2_frame": [0, 0, 0],
            "A3_frame": [0, 0, 0],
            "A4_frame": [0, 0, 0],
            "A5_frame": [0, 0, 0],
            "n1": 0,
            "I1": 0,
            "n2": 0,
            "I2": 0,
            "n3": 0,
            "I3": 0,
            "n4": 0,
            "I4": 0,
            "n5": 0,
            "I5": 0,
            "D": 0.0,
            "E": 0.0,
            "D_tri": 700.0,
            "E_tri": 0.0,
            "beta": 1.4,
            "J_0": 1e10,
            "J_ex": 0.1,
            "D_frame": [0, 0, 0],
            "D_tri_frame": [0, 0, 0],
            "T_relax_1": 0.0,
            "T_relax_2": 0.0,
            "decay": 0.0,
            "dynamics": [0.0, 0.0, 0.0, 0.0],
            "T_pm": 0.1,
            "amplitude": 0.0,
            "distribution": None,
            "distribution_order": 3,
        }
    }

    return default_profile


def new_variation_profile() -> dict:
    """
    Get a default variation profile.

    Returns
    -------
    default_profile: dict
        Dictionary with default settings for variation.

    """
    default_profile = {
        "main": {
            "g1": [0, 0, 0],
            "g2": [0, 0, 0],
            "g_tri": [0, 0, 0],
            "g": [0, 0, 0],
            "A1": [0, 0, 0],
            "A2": [0, 0, 0],
            "A3": [0, 0, 0],
            "A4": [0, 0, 0],
            "A5": [0, 0, 0],
            "A_eseem": 0.0,
            "omega_I": 0.0,
            "D": 0.0,
            "D_tri": 0.0,
            "E": 0.0,
            "E_tri": 0.0,
            "beta": 0.0,
            "J_0": 0.0,
            "J_ex": 0.0,
            "g1_frame": [0, 0, 0],
            "g2_frame": [0, 0, 0],
            "g_tri_frame": [0, 0, 0],
            "g_frame": [0, 0, 0],
            "A1_frame": [0, 0, 0],
            "A2_frame": [0, 0, 0],
            "A3_frame": [0, 0, 0],
            "A4_frame": [0, 0, 0],
            "A5_frame": [0, 0, 0],
            "D_frame": [0, 0, 0],
            "D_tri_frame": [0, 0, 0],
            "width_gauss": 0.5,
            "T_relax_1": 0.0,
            "T_relax_2": 0.0,
            "T_pm": 0.0,
            "population": [0.0, 0.0, 0.0],
            "freq_mw": 0.0,
            "amplitude": 0,
        }
    }

    return default_profile


def new_save_profile() -> dict:
    """
    Get a default save profile.

    Returns
    -------
    default_profile: dict
        Dictionary with default settings for saving.

    """
    default_profile = {"main": {}}
    # TODO Profil verwenden um Projektordner festzulegen -> Speicherort für
    # Bilder, Simulations/Optimierungsergebnisse, out-Files etc.

    return default_profile


def new_optimization_profile() -> dict:
    """
    Get a default optimization profile.

    Returns
    -------
    default_profile: dict
        Dictionary with default settings for optimization routines.

    """
    default_profile = {
        "main": {"routine": "", "method": None, "cpu_cores": 0},
        "genetic": {
            "GAVaPS": True,
            "representation": "",
            "lifetime_mode": "",
            "crossover_type": "",
            "mutation_type": "",
            "min_lifetime": 0,
            "max_lifetime": 10,
            "reproduction_ratio": 0.2,
            "p_c": 0.5,
            "p_m": 0.001,
            "pop_size": 500,
            "min_pop_size": 1,
            "max_pop_size": 100,
            "convergence": 0.02,
            "peak_prominence": 0.5,
            "error_weight": [1.0, 0.2],
            "max_generation": 1000,
            "show_status": True,
        },
        "minimize": {"maxiter": ""},
        "dual_annealing": {
            "maxiter": None,
            "maxiter_minimizer": None,
            "initial_temp": None,
            "restart_temp_ratio": None,
            "visit": None,
            "accept": None,
            "maxfun": None,
            "no_local_search": None,
        },
        "shgo": {
            "maxiter": None,
            "maxiter_minimizer": None,
            "n": None,
            "iters": None,
            "maxfev": None,
            "ftol": None,
            "maxev": None,
            "maxtime": None,
            "minimize_every_iter": None,
            "local_iter": None,
            "sampling_method": None,
        },
        "differential_evolution": {
            "strategy": None,
            "maxiter": None,
            "popsize": None,
            "tol": None,
            "mutation": None,
            "recombination": None,
            "seed": None,
            "atol": None,
            "polish": None,
            "init": None,
            "disp": None,
            "updating": None,
        },
        "basinhopping": {
            "T": None,
            "stepsize": None,
            "disp": None,
            "stepwise_factor": None,
            "maxiter_minimizer": None,
            "seed": None,
            "interval": None,
            "niter_success": None,
            "niter": None,
            "target_accept_rate": None,
        },
        "least_squares": {
            "ftol": None,
            "xtol": None,
            "gtol": None,
            "loss": None,
            "f_scale": None,
            "max_nfev": None,
            "tr_solver": None,
            "verbose": None,
        },
    }

    return default_profile


def new_simulation_profile() -> dict:
    """
    Get an empty simulation profile.

    Returns
    -------
    default_profile: dict
        Dictionary with default settings for simulation profiles.

    """
    default_profile = {
        "main": {"routine": "", "cpu_cores": 0},
        "static_radpair": {"knots": 20},
        "teacups": {
            "knots": 20,
            "space": "hilbert",
        },
        "opossum": {},
        "didelphis_tikhonov": {
            "min_r": 10,
            "max_r": 50,
            "r_points": 401,
            "force_cpu": False,
            "regularization_mode": 2,
        },
        "didelphis": {"min_r": 10, "max_r": 50, "r_points": 401},
    }

    return default_profile
