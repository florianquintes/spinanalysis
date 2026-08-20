#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read EPR data from BRUKER BES3T, ESP transient, MATLAB, and text formats.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from spinanalysis._utils import strtobool


def get_full_path(directory_name: str, start_directory: str | None = None) -> str:
    """
    Get the full path of a given directory. Search starts at home directory.

    Parameters
    ----------
    directory_name : str
        Name of the directory whose path is to be found.
    start_directory : str, optional
        Directory at which the search starts. If given, the performance
        increases very sharply. The default is None.

        Recommended.

    Raises
    ------
    FileNotFoundError
        If the directory cannot be found.

    Returns
    -------
    str
        The full path of the directory.

    """
    home = Path.home()

    if start_directory is None:
        search_root = home
    else:
        search_root = home / start_directory

    matches: list[Path] = []
    for p in search_root.rglob(directory_name):
        if p.is_dir() and p.name == directory_name:
            matches.append(p)

    if not matches:
        raise FileNotFoundError(
            "Directory '{}' not found below '{}'.".format(directory_name, search_root)
        )

    if len(matches) > 1:
        warnings.warn(
            "Multiple directories named '{}' found; using the first match.".format(
                directory_name
            ),
            stacklevel=2,
        )

    return str(matches[0])


def _find_file(folder: Path, suffix: str) -> Path:
    """
    Find a single file in *folder* with the given *suffix* (case-insensitive).

    Raises
    ------
    FileNotFoundError
        If no matching file is found.

    Returns
    -------
    Path

    """
    matches = [
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == suffix.lower()
    ]
    if not matches:
        raise FileNotFoundError(
            "No file with suffix '{}' found in '{}'.".format(suffix, folder)
        )
    return matches[0]


def get_DSC_parameters(path_to_folder: str) -> dict:
    """
    Extract all parameters from the DSC file.

    The DSC file is found by searching for ``*.DSC`` in the data folder
    rather than assuming it shares the folder's basename.

    Parameters
    ----------
    path_to_folder : str
        Full path to the folder containing the ``.DSC`` and ``.DTA`` files.

    Returns
    -------
    dict
        Dictionary with all parameters. Keys are the same as in the .DSC
        file. An additional key ``'path_to_folder'`` holds the folder path.

    """
    folder = Path(path_to_folder)
    dsc_file = _find_file(folder, ".DSC")

    dsc_parameters: dict[str, object] = {}

    with open(str(dsc_file), "r") as file:
        for line in file.readlines():
            if line.startswith(("*", "#", "'")):
                continue
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "FTAcqModeSlct":
                dsc_parameters[parts[0]] = parts[-1]
            elif len(parts) >= 2:
                dsc_parameters[parts[0]] = convert_parameter_type(parts[1])
            else:
                dsc_parameters[parts[0]] = True

    dsc_parameters["path_to_folder"] = str(folder)

    return dsc_parameters


def convert_parameter_type(value: str) -> bool | int | float | str:
    """
    Convert the type of a given string to bool, int or float if possible.

    Parameters
    ----------
    value : str
        Parameter string which should be converted.

    Returns
    -------
    bool or int or float or str
        Parameter as bool, int or float. If none is possible, the stripped
        string will be returned.

    """
    try:
        return bool(strtobool(value))
    except ValueError:
        pass

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value.strip("'")


def get_byte_mode(DSC_dict: dict, data_key: str = "IRFMT") -> str:
    """
    Get the used byte mode of the BRUKER BES3T data.

    For more information about BES3T go to BRUKER website or easyspin on
    GitHub.

    Parameters
    ----------
    DSC_dict : dict
        Dictionary with all parameters. Key is the same as in .DSC.
    data_key : str, optional
        Key for the data array. Either 'IRFMT' or 'IIFMT'.
        The default is 'IRFMT'.

    Raises
    ------
    ValueError
        If the BSEQ value is not 'BIG' or 'LIT', if the format value is
        not one of C, S, I, F, D, or if the format is 'A' (ASCII).

    Returns
    -------
    str
        Byte mode string for ``numpy.fromfile(dtype=...)``.

    """
    byte_mode = ""

    bseq = DSC_dict["BSEQ"]
    if bseq == "BIG":
        byte_mode += ">"
    elif bseq == "LIT":
        byte_mode += "<"
    else:
        raise ValueError("Unknown BSEQ value '{}' in .DSC file!".format(bseq))

    fmt = DSC_dict[data_key]
    if fmt in ("C", "S", "I"):
        byte_mode += "i"
        if fmt == "I":
            byte_mode += "4"
        elif fmt == "S":
            byte_mode += "2"
    elif fmt in ("F", "D"):
        byte_mode += "f"
        if fmt == "D":
            byte_mode += "8"
    elif fmt == "A":
        raise ValueError("Can't read BES3T data in ASCII format!")
    else:
        raise ValueError(
            "Unknown value for keyword '{}' in .DSC file!".format(data_key)
        )

    return byte_mode


def load_data_vector(DSC_dict: dict) -> np.ndarray:
    """
    Load the binary intensity vector(s) from a BRUKER BES3T file.

    Parameters
    ----------
    DSC_dict : dict
        Dictionary with all parameters. Key is the same as in .DSC.

    Raises
    ------
    ValueError
        If IKKF is not 'CPLX' or 'REAL', or if the dimension is not 1 or 2.
    KeyError
        If an axis is given but the corresponding number of points is
        missing.

    Returns
    -------
    np.ndarray, np.complex128
        Real and imaginary part of the measured intensities. If no
        imaginary part is measured, zeros will be inserted.

    """
    get_data_dimension(DSC_dict)
    byte_mode = get_byte_mode(DSC_dict)

    folder = Path(DSC_dict["path_to_folder"])
    dta_file = folder / (DSC_dict["TITL"] + ".DTA")

    data_vector = np.fromfile(str(dta_file), dtype=byte_mode)

    if DSC_dict["IKKF"] == "CPLX":
        data_vector_real = data_vector[0::2]
        data_vector_imag = data_vector[1::2]
    elif DSC_dict["IKKF"] == "REAL":
        data_vector_real = data_vector
        data_vector_imag = np.zeros(len(data_vector))
    else:
        raise ValueError("Unknown value for keyword IKKF!")

    if DSC_dict["dimensions"] == 2:
        if "XPTS" in DSC_dict and "YPTS" in DSC_dict:
            data_vector_real = data_vector_real.reshape(
                (DSC_dict["YPTS"], DSC_dict["XPTS"])
            )
            data_vector_imag = data_vector_imag.reshape(
                (DSC_dict["YPTS"], DSC_dict["XPTS"])
            )
        elif "XPTS" in DSC_dict and "ZPTS" in DSC_dict:
            data_vector_real = data_vector_real.reshape(
                (DSC_dict["ZPTS"], DSC_dict["XPTS"])
            )
            data_vector_imag = data_vector_imag.reshape(
                (DSC_dict["ZPTS"], DSC_dict["XPTS"])
            )
        else:
            raise KeyError("Can't find needed keys from (XPTS, YPTS, ZPTS)!")
    elif DSC_dict["dimensions"] == 1:
        pass
    else:
        raise ValueError("Can't handle dimensions {}".format(DSC_dict["dimensions"]))

    cmplx_data_vector = data_vector_real + 1j * data_vector_imag

    return cmplx_data_vector.T


def load_axis_vector(axis: str, DSC_dict: dict) -> np.ndarray:
    """
    Load the points of a given axis (x, y, z).

    Parameters
    ----------
    axis : str
        Name of the axis. Needs to start with x, X, y, Y or z, Z.
    DSC_dict : dict
        Dictionary with all parameters. Key is the same as in .DSC.

    Raises
    ------
    ValueError
        If the given axis doesn't start with an allowed character.

    Returns
    -------
    np.ndarray
        Array with all axis points. Empty array if no axis data is
        available.

    """
    axis = axis.upper()
    if axis.startswith("X"):
        axis = "X"
    elif axis.startswith("Y"):
        axis = "Y"
    elif axis.startswith("Z"):
        axis = "Z"
    else:
        raise ValueError("Axis needs to start with 'x', 'y' or 'z'!")

    folder = Path(DSC_dict["path_to_folder"])
    axis_file = folder / (DSC_dict["TITL"] + "." + axis + "GF")
    data_key = axis + "FMT"

    try:
        byte_mode = get_byte_mode(DSC_dict, data_key=data_key)
        axis_vector = np.fromfile(str(axis_file), dtype=byte_mode)
    except (KeyError, FileNotFoundError):
        try:
            minimum = DSC_dict[axis + "MIN"]
        except KeyError:
            warnings.warn(
                "No axis data found for axis '{}'; returning empty array.".format(axis),
                stacklevel=2,
                category=UserWarning,
            )
            axis_vector = np.array([])
        else:
            width = DSC_dict[axis + "WID"]
            points = DSC_dict[axis + "PTS"]
            axis_vector = np.linspace(minimum, minimum + width, points)

    return axis_vector


def get_data_dimension(DSC_dict: dict) -> None:
    """
    Determine the dimension of the measured spectrum (1d/2d/3d).

    The result is stored in-place in *DSC_dict* under the key
    ``'dimensions'``.

    Parameters
    ----------
    DSC_dict : dict
        Dictionary with all parameters. Modified in place.

    Returns
    -------
    None

    """
    dim = 0
    for axis in ("XTYP", "YTYP", "ZTYP"):
        if DSC_dict[axis] != "NODATA":
            dim += 1

    DSC_dict["dimensions"] = dim


def load_epr_bruker_bes3t(
    folder: str, start_directory: str | None = None
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """
    Load the whole dataset from a BRUKER BES3T data folder.

    Time axis will be rescaled for OOP-ESEEM experiments if
    ``'FTAcqModeSlct'`` is ``'Tables'``.

    Parameters
    ----------
    folder : str
        Name of the data folder with the corresponding data files.
    start_directory : str, optional
        Give the path starting from your home folder which the search for
        the data folder should start at. The default is None.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/\\*, then
        you can give this start_directory with start_directory='nice'. So
        the function call would be::

            load_epr_bruker_bes3t(this_folder, 'nice')

        Recommended:
            Without this parameter, the search for the right folder will be
            much longer.

    Returns
    -------
    axis : tuple of np.ndarray
        Tuple of all axis vectors as three numpy arrays (x, y, z).
    data : np.ndarray, np.complex128
        All intensity values as one complex numpy array.

    """
    path = get_full_path(folder, start_directory)
    dsc_parameters = get_DSC_parameters(path)

    data = load_data_vector(dsc_parameters)
    x = load_axis_vector("x", dsc_parameters)
    try:
        if dsc_parameters["FTAcqModeSlct"] == "Tables":
            x /= 2
    except KeyError:
        pass
    y = load_axis_vector("y", dsc_parameters)
    z = load_axis_vector("z", dsc_parameters)
    axis = (x, y, z)

    return axis, data


def load_epr_ESP_transient(
    folder: str, start_directory: str | None = None
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Load data from a transient EPR experiment measured with ESP380E.

    Parameters
    ----------
    folder : str
        Name of the folder with all data files.
    start_directory : str, optional
        Give the path starting from your home folder which the search for
        the data folder should start at. The default is None.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/\\*, then
        you can give this start_directory with start_directory='nice'. So
        the function call would be::

            load_epr_ESP_transient(this_folder, 'nice')

        Recommended:
            Without this parameter, the search for the right folder will be
            much longer.

    Returns
    -------
    axis : tuple of np.ndarray
        Tuple with all axis vectors as two numpy arrays
        (magnetic_field, time).
    data : np.ndarray, np.complex128
        All intensity values as one complex numpy array. Imaginary part is
        always 0.

    """
    path = get_full_path(folder, start_directory)
    axis, data = get_transient_data(path)

    return axis, data


def _validate_info_file(sections_found: set[str]) -> None:
    """Warn if expected sections were not found in the .info file."""
    expected = {"MAGNETIC", "TRANSIENT"}
    missing = expected - sections_found
    if missing:
        warnings.warn(
            "Sections {} not found in .info file. The file may be "
            "corrupted or from a different spectrometer.".format(
                ", ".join(repr(s) for s in sorted(missing))
            ),
            stacklevel=3,
        )


def get_transient_info(fpath: str) -> tuple:
    """
    Get all information about the time axis and magnetic field vector from
    the .info file.

    The .info file is found by searching for ``*.info`` in the data folder.

    Parameters
    ----------
    fpath : str
        Full path to the folder with the experimental data.

    Raises
    ------
    KeyError
        If the section 'MAGNETIC FIELD' or 'TRANSIENT' could not be found
        in the .info file.

    Returns
    -------
    tuple
        Contains (time_length, time_points, mag_field_start,
        mag_field_stop, mag_field_step).

    """
    folder = Path(fpath)
    info_file = _find_file(folder, ".info")

    transient_info: list[float | int | None] = [None] * 5
    sections_found: set[str] = set()

    with open(str(info_file), "r") as file:
        current_section = ""
        for line in file.readlines():
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0].isupper():
                current_section = parts[0]
                sections_found.add(current_section)
                continue
            if current_section == "MAGNETIC":
                if parts[0] == "Start:":
                    transient_info[2] = float(parts[1])
                elif parts[0] == "Stop:":
                    transient_info[3] = float(parts[1])
                elif parts[0] == "Step:":
                    transient_info[4] = float(parts[1])
            elif current_section == "TRANSIENT":
                if parts[0] == "Points:":
                    transient_info[1] = int(parts[1])
                elif parts[0] == "Length:":
                    transient_info[0] = float(parts[1])

            if None not in transient_info:
                break

    _validate_info_file(sections_found)

    if transient_info[0] is None:
        raise KeyError("Couldn't find section 'TRANSIENT' in .info file!")
    if transient_info[2] is None:
        raise KeyError("Couldn't find section 'MAGNETIC FIELD' in .info file!")

    return tuple(transient_info)


def _find_transient_basename(folder: Path) -> str:
    """Find the shared basename of numbered transient data files.

    Files are named ``<basename>.<number>`` where the number width
    depends on the total count of traces (e.g. ``.1``, ``.01``,
    ``.001``). This function globs for files with numeric suffixes and
    returns the common basename.

    Raises
    ------
    FileNotFoundError
        If no numbered data files are found.
    """
    candidates: list[Path] = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        suffix = f.suffix
        if suffix and suffix[1:].isdigit():
            candidates.append(f)

    if not candidates:
        raise FileNotFoundError(
            "No numbered transient data files found in '{}'.".format(folder)
        )

    candidates.sort()
    return candidates[0].stem


def _discover_max_digits(folder: Path, basename: str) -> int:
    """Discover the digit width used for the numbered file suffixes.

    Tries increasing widths (1, 2, 3, ...) until a file
    ``<basename>.<padded-1>`` is found.
    """
    for width in range(1, 10):
        fname = folder / (basename + "." + str(1).zfill(width))
        if fname.is_file():
            return width
    raise FileNotFoundError(
        "No first transient data file could be found in '{}'.".format(folder)
    )


def get_transient_data(fpath: str) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Get the measured intensities of the whole spectrum from a transient
    EPR experiment measured by ESP380E.

    Parameters
    ----------
    fpath : str
        Full path to the folder with the experimental data.

    Returns
    -------
    axis : tuple of np.ndarray
        Tuple with all axis vectors as two numpy arrays
        (magnetic_field, time).
    data : np.ndarray, np.complex128
        Measured real intensities. All imaginary parts are zero.

    """
    folder = Path(fpath)
    basename = _find_transient_basename(folder)
    max_digits = _discover_max_digits(folder, basename)

    data_vector_real: list[np.ndarray] = []
    magnetic_field: list[float] = []

    file_exists = True
    file_num = 1
    time_axis: np.ndarray | None = None

    while file_exists:
        ending = "." + str(file_num).zfill(max_digits)
        fname = folder / (basename + ending)
        if not fname.is_file():
            file_exists = False
            continue

        if file_num == 1:
            (field_point, time_axis), data = read_single_transient_file(
                str(folder), basename, file_num, max_digits, time=True
            )
        else:
            field_point, data = read_single_transient_file(
                str(folder), basename, file_num, max_digits
            )
        data_vector_real.append(data)
        magnetic_field.append(round(field_point, 1))
        file_num += 1

    magnetic_field_arr = np.array(magnetic_field)
    sorting = np.argsort(magnetic_field_arr)
    magnetic_field_arr = magnetic_field_arr[sorting]
    cmplx_data_vector = np.array(data_vector_real, dtype=np.complex128)
    cmplx_data_vector = cmplx_data_vector[sorting]
    axis = (magnetic_field_arr, time_axis)

    return axis, cmplx_data_vector.T


def read_single_transient_file(
    fpath: str,
    basename: str,
    filenumber: int,
    digits: int,
    time: bool = False,
) -> tuple:
    """
    Get the measured intensities of a single field point from a transient
    EPR experiment measured with ESP380E.

    Parameters
    ----------
    fpath : str
        Full path to the folder with the experimental data.
    basename : str
        Shared basename of the numbered data files.
    filenumber : int
        Number of the dataset for the magnetic field point, e. g. 3.
    digits : int
        Number of digits in the file-number suffix.
    time : bool, optional
        If True, the time axis will also be returned.

    Returns
    -------
    field : float
        Magnetic field point.
    time_axis : np.ndarray
        Linear time axis. Only returned if *time* is True.
    data_vector : np.ndarray
        Measured intensities.

    """
    folder = Path(fpath)
    ending = "." + str(filenumber).zfill(digits)
    fname = folder / (basename + ending)

    with open(str(fname), "r") as file:
        lines = file.readlines()

    field: float = 0.0
    time_axis: np.ndarray | None = None

    for i, line in enumerate(lines):
        if i >= 5:
            break
        parts = line.strip().split()
        if parts and parts[0] == "B0":
            field = float(parts[2])
        if i == 3 and time:
            points = int(parts[1])
            start = float(parts[2])
            stop = float(parts[3])
            time_axis = np.linspace(start, stop, points)

    try:
        data_vector = np.genfromtxt(str(fname), skip_header=5)
        data_vector = data_vector.flatten()
    except ValueError:
        data_lines: list[str] = []
        for line in lines[5:]:
            data_lines.extend(line.strip().split())
        data_vector = np.array(data_lines, dtype=np.complex128)

    if time:
        return (field, time_axis), data_vector

    return field, data_vector


def load_simulated_data(
    folder: str, start_directory: str | None = None
) -> tuple[np.ndarray, ...]:
    """
    Load simulated data from spinanalysis or data saved with
    :func:`saving.save_simulation`. Uses :func:`numpy.loadtxt`.

    Parameters
    ----------
    folder : str
        Name of the data folder with the corresponding data files
        (``x_axis.txt``, ``intensity.txt``, optionally ``y_axis.txt``).
    start_directory : str, optional
        Give the path starting from your home folder which the search for
        the data folder should start at. The default is None.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/\\*, then
        you can give this start_directory with start_directory='cooluser/nice'.
        So the function call would be::

            load_simulated_data(this_folder, 'cooluser/nice')

        Recommended:
            Without this parameter, the search for the right folder
            will be much longer.

    Returns
    -------
    x : np.ndarray
        Axis vector for the x-axis.
    y : np.ndarray
        Axis vector for the y-axis. Only returned if the simulated data
        is 2d.
    intensity : np.ndarray, np.complex128
        Simulated intensities. Either 1d or 2d.

    """
    path = get_full_path(folder, start_directory)
    data_folder = Path(path)

    x = np.loadtxt(str(data_folder / "x_axis.txt"))
    try:
        intensity = np.loadtxt(str(data_folder / "intensity.txt"))
    except ValueError:
        intensity = np.loadtxt(str(data_folder / "intensity.txt"), dtype=np.complex_)

    try:
        y = np.loadtxt(str(data_folder / "y_axis.txt"))
    except OSError:
        return (x, intensity)

    return (x, y, intensity)


def load_matlab(
    folder: str,
    start_directory: str | None = None,
    field: str = "field",
    signal: str = "signal",
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Load EPR data from a MATLAB data file (``.mat``).

    The *folder* argument is the name of the directory containing the
    ``.mat`` file. The first ``.mat`` file found in the directory is
    loaded.

    Parameters
    ----------
    folder : str
        Name of the data folder with the ``.mat`` file.
    start_directory : str, optional
        Give the path starting from your home folder which the search for
        the data folder should start at. The default is None.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/\\*, then
        you can give this start_directory with start_directory='cooluser/nice'.
        So the function call would be::

            load_matlab(this_folder, 'cooluser/nice')

    field : str, optional
        Name of the field array in the .mat file. The default is 'field'.
    signal : str, optional
        Name of the signal array in the .mat file. The default is 'signal'.

    Returns
    -------
    axis : tuple of np.ndarray
        Tuple of np.ndarray containing the x and y axis.
    data : np.ndarray
        np.ndarray with the measured intensities.

    """
    path = get_full_path(folder, start_directory)
    data_folder = Path(path)
    mat_file = _find_file(data_folder, ".mat")

    mat_data = loadmat(str(mat_file))
    x = mat_data[field].flatten()
    y = np.array([])
    data = mat_data[signal].flatten()
    axis = (x, y)

    return axis, data


def load_txt(
    folder: str, start_directory: str | None = None
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Load EPR data from a ``.txt`` file.

    The *folder* argument is the name of the directory containing the
    ``.txt`` file. The first ``.txt`` file found in the directory is
    loaded.

    Parameters
    ----------
    folder : str
        Name of the data folder with the ``.txt`` file.
    start_directory : str, optional
        Give the path starting from your home folder which the search for
        the data folder should start at. The default is None.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/\\*, then
        you can give this start_directory with start_directory='cooluser/nice'.
        So the function call would be::

            load_txt(this_folder, 'cooluser/nice')

    Returns
    -------
    axis : tuple of np.ndarray
        Tuple of np.ndarray containing the x and y axis.
    data : np.ndarray
        np.ndarray with the measured intensities.

    """
    path = get_full_path(folder, start_directory)
    data_folder = Path(path)
    txt_file = _find_file(data_folder, ".txt")

    n_cols = 0
    with open(str(txt_file), "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            n_cols = len(stripped.split())
            break

    if n_cols == 0:
        raise ValueError("File '{}' contains no data.".format(txt_file))

    data = np.loadtxt(str(txt_file), unpack=True)

    if n_cols == 2:
        x, intensity = data
        y = np.array([])
    elif n_cols == 3:
        x, y, intensity = data
    else:
        raise ValueError(
            "Expected 2 or 3 columns in '{}', got {}.".format(txt_file, n_cols)
        )

    axis = (x, y)

    return axis, intensity
