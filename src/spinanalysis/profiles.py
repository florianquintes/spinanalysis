#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

from configobj import ConfigObj
from matplotlib.pyplot import style
from spinanalysis._utils import strtobool
import os
import validate
from pathlib import Path
from zipfile import ZipFile


PROFILE_ROOT = Path(
    os.environ.get(
        "SPINANALYSIS_PROFILE_DIR",
        Path.home() / ".config" / "spinanalysis" / "profiles",
    )
)

_VALID_PROFILE_KINDS = (
    "plot",
    "simulation",
    "optimization",
    "spinsystem",
    "variation",
)

_VALID_OPTIMIZATION_ROUTINES = (
    "genetic",
    "minimize",
    "dual_annealing",
    "shgo",
    "differential_evolution",
    "basinhopping",
    "least_squares",
)

_VALID_SIMULATION_ROUTINES = (
    "static_radpair",
    "teacups",
    "opossum",
    "didelphis",
    "didelphis_tikhonov",
)


def _normalize_pkind(pkind: str) -> str:
    """Normalise a profile kind string to lower-case and map British spelling."""
    pkind = pkind.lower()
    if pkind == "optimisation":
        pkind = "optimization"
    if pkind not in _VALID_PROFILE_KINDS:
        raise ValueError(
            "pkind must be one of {}!".format(
                ", ".join(repr(k) for k in _VALID_PROFILE_KINDS)
            )
        )
    return pkind


def _normalize_kinds(pkind: str | list[str]) -> list[str]:
    """Normalise *pkind* (str or list) to a list of valid profile kinds."""
    if pkind == "all":
        return list(_VALID_PROFILE_KINDS)

    if isinstance(pkind, str):
        pkind = pkind.split()

    stack: list[str] = []
    for el in pkind:
        el = el.lower()
        if el == "optimisation":
            el = "optimization"
        if el in _VALID_PROFILE_KINDS:
            stack.append(el)
    return stack


def import_profiles(zipfile: str, override: bool = False) -> None:
    """
    Import profiles from a zip archive.

    The archive must contain subdirectories matching valid profile kinds
    (e.g. ``spinsystem/``, ``simulation/``).  Files are extracted directly
    into :data:`PROFILE_ROOT`.

    Parameters
    ----------
    zipfile : str
        Path to the zip archive.
    override : bool, optional
        If True, existing profiles with the same name will be overwritten.
        The default is False.

    Raises
    ------
    ValueError
        If the archive contains a top-level entry that does not correspond
        to a valid profile kind.

    Returns
    -------
    None

    """
    archive = Path(zipfile).resolve()
    root = PROFILE_ROOT

    with ZipFile(str(archive), "r") as zf:
        for entry in zf.namelist():
            top = entry.split("/")[0]
            if top and top not in _VALID_PROFILE_KINDS:
                raise ValueError(
                    "Archive entry '{}' does not match any valid profile kind.".format(
                        top
                    )
                )
            target = root / entry
            if target.is_file() and not override:
                continue
            zf.extract(entry, str(root))


def export(
    path: str | None = None,
    pkind: str | list[str] = "all",
    pname: str = "all",
) -> None:
    """
    Export the chosen profile(s) as a zip archive.

    Parameters
    ----------
    path : str, optional
        Directory where the zip file will be stored. If None, the zip file
        will be stored in the current working directory. The default is None.
    pkind : str or list of str, optional
        Define which kind(s) of profiles should be exported. Options are
        'plot', 'spinsystem', 'optimization', 'simulation', 'variation' and
        'all'. The default is 'all'.
    pname : str, optional
        Give the basename of the profile. The default is 'all'.

    Returns
    -------
    None

    """
    root = PROFILE_ROOT
    kinds = _normalize_kinds(pkind)

    file_paths: list[Path] = []
    for kind in kinds:
        file_paths.extend(_get_profile_paths(root / kind, pname))

    if path is None:
        archive = Path.cwd() / "profiles_spinanalysis.zip"
    else:
        archive = Path(path) / "profiles_spinanalysis.zip"

    root_str = str(root)
    with ZipFile(str(archive), "w") as zf:
        for fp in file_paths:
            arcname = str(fp)[len(root_str) + 1 :]
            zf.write(str(fp), arcname)


def _get_profile_paths(pfolder: Path, pname: str = "all") -> list[Path]:
    """
    Get the paths of all profiles in *pfolder*.

    Parameters
    ----------
    pfolder : Path
        Folder with profiles.
    pname : str, optional
        Basename of the profile. The default is "all".

    Returns
    -------
    list of Path
        Paths to matching profile files.

    """
    paths: list[Path] = []

    if not pfolder.is_dir():
        return paths

    for fp in pfolder.rglob("*"):
        if not fp.is_file():
            continue
        if fp.name == "configspec.ini":
            continue
        if pname.lower() == "all":
            paths.append(fp)
        elif pname.lower() == fp.name:
            paths.append(fp)

    return paths


def _validate_profile_sections(profile: dict, pkind: str) -> None:
    """
    Check that all required sections and keys are present in the profile.

    Parameters
    ----------
    profile : dict
        Dictionary with all profile settings.
    pkind : str
        Kind of the profile (already normalised to lower-case).

    Raises
    ------
    ValueError
        If a required section or key is missing.

    Returns
    -------
    None

    """
    if "main" not in profile:
        raise ValueError(
            "Profile for '{}' is missing the 'main' section.".format(pkind)
        )

    match pkind:
        case "optimization" | "simulation":
            routine = profile["main"].get("routine", "")
            if not routine:
                raise ValueError(
                    "Profile for '{}' is missing 'routine' in the 'main' "
                    "section.".format(pkind)
                )
            if routine not in profile:
                raise ValueError(
                    "Profile for '{}' is missing the '{}' section.".format(
                        pkind, routine
                    )
                )


def _set_configobj(profile: dict, config: ConfigObj, pkind: str) -> ConfigObj:
    """
    Write profile parameters into the ConfigObj.

    For profiles that carry a routine (optimization, simulation), the
    routine-specific section is validated and copied in addition to
    ``main``.

    Parameters
    ----------
    profile : dict
        Dictionary with all profile settings.
    config : ConfigObj
        ConfigObj to populate.
    pkind : str
        Normalised profile kind.

    Raises
    ------
    ValueError
        If the routine is not valid for the given profile kind.

    Returns
    -------
    ConfigObj
        The populated ConfigObj.

    """
    main = {k: v for k, v in profile["main"].items() if v is not None}
    config["main"] = main

    match pkind:
        case "optimization":
            routine = profile["main"]["routine"]
            if routine not in _VALID_OPTIMIZATION_ROUTINES:
                raise ValueError("{} is no valid optimization routine!".format(routine))
            config[routine] = profile[routine]
        case "simulation":
            routine = profile["main"]["routine"]
            if routine not in _VALID_SIMULATION_ROUTINES:
                raise ValueError("{} is no valid simulation routine!".format(routine))
            config[routine] = profile[routine]

    return config


def add_profile(profile: dict, pkind: str, pname: str = "") -> None:
    """
    Add a new profile for spinanalysis.

    Parameters
    ----------
    profile : dict
        Dictionary with all profile settings.
    pkind : str
        Kind of the profile. Not case sensitive. Can be 'plot',
        'simulation', 'optimization', 'spinsystem' or 'variation'.
    pname : str, optional
        Name of the profile. If no profile name is given, a default one
        will be generated by :func:`_get_profile_name`. The default is ''.

    Raises
    ------
    ValueError
        If *pkind* is not a valid profile kind, or if the profile fails
        configspec validation.

    Returns
    -------
    None

    Examples
    --------
    Creating and adding a new profile:

    >>> from spinanalysis.epr import Spinsystem
    >>> from spinanalysis import profiles
    >>> Sys = Spinsystem()
    >>> Sys_profile = profiles.new_spinsystem_profile()
    >>> Sys_profile['main']['g1'] = [2.0034, 2.00156, 2.00228]
    >>> profiles.add_profile(Sys_profile, 'spinsystem', 'Sys_prof_1')

    """
    pkind = _normalize_pkind(pkind)

    if pname == "":
        pname = _get_profile_name(pkind)

    root = PROFILE_ROOT
    config_path = root / pkind

    match pkind:
        case "plot":
            _save_config_plot(profile, config_path / pname)
        case "spinsystem" | "variation" | "optimization" | "simulation":
            _validate_profile_sections(profile, pkind)
            config_name = config_path / (pname + ".ini")
            path_to_configspec = config_path / "configspec.ini"
            config_path.mkdir(parents=True, exist_ok=True)
            config = ConfigObj(str(config_name), configspec=str(path_to_configspec))

            config = _set_configobj(profile, config, pkind)

            _validate_config(config, pname)
            config.write()

    return None


def _save_config_plot(profile: dict, path: Path) -> None:
    """
    Write plot profile parameters to a stylesheet file.

    Parameters
    ----------
    profile : dict
        Plot profile.
    path : Path
        Path where the stylesheet will be stored.

    Returns
    -------
    None

    """
    preamble = "## {:*^76s}\n## {:*^76s}\n## {:*^76s}\n".format(
        "", " SPINANALYSIS SPECIAL SETTINGS ", ""
    )
    lines = [preamble, "\n"]
    for key in profile:
        if key.endswith("lim"):
            line = "{}: {}, {}".format(
                str(key), str(profile[key][0]), str(profile[key][1])
            )
        elif isinstance(profile[key], (int, float, bool)):
            line = "{}: {}".format(str(key), str(profile[key]))
        else:
            line = '{}: "{}"'.format(str(key), str(profile[key]))
        lines.append(line + "\n")
    Path(path).write_text("".join(lines))


def _get_profile_name(pkind: str) -> str:
    """
    Search for the smallest number available for the default profile name.

    Scheme for default profile name is 'profile_[number]'.

    Parameters
    ----------
    pkind : str
        Kind of the profile. Not case sensitive.

    Returns
    -------
    str
        Available profile name, e.g. 'profile_12'.

    """
    suffix = "" if pkind == "plot" else ".ini"
    folder = PROFILE_ROOT / pkind

    existing: set[int] = set()
    if folder.is_dir():
        for fp in folder.glob("profile_*" + suffix):
            stem = fp.stem if suffix else fp.name
            num_part = stem.replace("profile_", "")
            try:
                existing.add(int(num_part))
            except ValueError:
                continue

    number = 1
    while number in existing:
        number += 1

    return "profile_{}".format(number)


def _validate_config(config: ConfigObj, pname: str) -> None:
    """Validate *config* against its configspec and raise on failure.

    Only sections and keys present before validation are checked.
    Keys with the value ``"None"`` (serialised Python ``None``) are
    skipped, as they represent optional fields that were not set.
    """
    validator = validate.Validator()
    present = {s: set(config[s].keys()) for s in config.sections}
    result = config.validate(validator, preserve_errors=True)
    if result is not True:
        for section, keys in present.items():
            section_result = result.get(section, True)
            if section_result is True:
                continue
            if isinstance(section_result, dict):
                failed = {
                    k: v
                    for k, v in section_result.items()
                    if k in keys and v is not True and config[section].get(k) != "None"
                }
                if failed:
                    raise ValueError(
                        "Profile '{}' failed configspec validation "
                        "in section '{}': {}".format(pname, section, failed)
                    )
            elif section_result is not True:
                raise ValueError(
                    "Profile '{}' failed configspec validation in section '{}'.".format(
                        pname, section
                    )
                )


def load_profile(pname: str, pkind: str) -> dict:
    """
    Load a profile from a config file.

    Parameters
    ----------
    pname : str
        Name of the profile. Case sensitive. Either with .ini or not.
        E. g.: load_profile('test') or load_profile('test.ini').
    pkind : str
        Kind of the profile. Not case sensitive. Can be 'simulation',
        'optimization', 'spinsystem' or 'variation'.

    Raises
    ------
    ValueError
        If *pkind* is not valid or the profile fails configspec validation.

    Returns
    -------
    dict
        Loaded profile as a dictionary.

    """
    pkind = _normalize_pkind(pkind)

    if pname.endswith(".ini"):
        pname = pname[:-4]

    root = PROFILE_ROOT
    path_to_profile = root / pkind / (pname + ".ini")
    path_to_configspec = root / pkind / "configspec.ini"

    config = ConfigObj(
        str(path_to_profile), configspec=str(path_to_configspec), file_error=True
    )
    _validate_config(config, pname)

    profile: dict[str, dict] = {}
    for section in config.sections:
        profile[section] = config[section]

    return profile


def load_plot_profile(pname: str | None) -> dict:
    """
    Load a plotting profile from an mplstylesheet.

    Parameters
    ----------
    pname : str or None
        Name of the profile. Case sensitive. If None or a built-in
        matplotlib style, the default stylesheet is loaded.

    Returns
    -------
    dict
        Settings for the plotting functions.

    """
    if pname is None or pname in style.available:
        pname = "default_stylesheet"

    path_to_profile = PROFILE_ROOT / "plot" / pname

    profile: dict[str, object] = {}
    valid_keys = (
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
    )

    with open(str(path_to_profile), "r") as file:
        for raw_line in file:
            if raw_line.startswith("#") or not raw_line.strip():
                continue
            parts = raw_line.strip().split(":", 1)
            if len(parts) != 2:
                continue
            key, value = parts[0].strip(), parts[1].split("#")[0].strip()
            if key not in valid_keys:
                continue
            if "lim" in key and len(key) == 4:
                bounds = value.split(",")
                if len(bounds) != 2:
                    raise ValueError("Malformed bounds for '{}': {}".format(key, value))
                profile[key] = [float(bounds[0]), float(bounds[1])]
            elif ("label" in key and len(key) == 6) or key == "title":
                profile[key] = value.strip('"').strip("'")
            else:
                profile[key] = bool(strtobool(value.strip('"').strip("'")))

    return profile


def new_plot_profile() -> dict:
    """
    Get an empty plotting profile.

    Returns
    -------
    dict
        Dictionary with default settings for plotting.

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
    dict
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
    dict
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
            "fit_distribution": False,
        }
    }

    return default_profile


def new_optimization_profile() -> dict:
    """
    Get a default optimization profile.

    Returns
    -------
    dict
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
    dict
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
