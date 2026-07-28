#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import os
import glob
from typing import Tuple
import numpy as np
from scipy.io import loadmat
from spinanalysis._utils import strtobool


def get_full_path(directory_name: str, start_directory: str = None) -> str:
    """
    Get the full path of a given directory. Search starts  at home directory.

    Parameters
    ----------
    directory_name : str
        Name of the directory whose path is to be found.
    start_directory : str, optional
        Directory at which the search starts. If given, the perfomance
        increases very sharply. The default is None.

        Recommended.

    Returns
    -------
    full_path : str
        The full path of the directory.

    """
    if start_directory is None:
        start_directory = os.path.join(os.path.expanduser("~"), "**")
    else:
        start_directory = os.path.join(os.path.expanduser("~"), start_directory, "**")

    all_subdirectories = glob.glob(start_directory, recursive=True)

    full_path = None
    for path in all_subdirectories:
        if path.endswith(directory_name):
            full_path = path

    return full_path


def get_DSC_parameters(path_to_folder: str) -> dict:
    """
    Extract all parameters from the DSC File.

    Parameters
    ----------
    path_to_folder : str
        Full path to the folder with .DSC and .DTA file. Files and path must
        have the same basename (BRUKER STANDARD).

    Returns
    -------
    DSC_parameters : dict
        Dictionary with all parameters. Key is the same as in .DSC.

    """
    DSC_parameters = {}

    basename = os.path.basename(path_to_folder)  # Foldername == Filename
    DSC_file = os.path.join(path_to_folder, basename + ".DSC")

    with open(DSC_file, "r") as file:
        for line in file.readlines():
            if line.startswith(("*", "#", "'")):
                pass
            elif line.startswith("FTAcqModeSlct"):
                line = line.split()
                DSC_parameters[line[0]] = line[-1]
            else:
                line = line.split()
                try:
                    DSC_parameters[line[0]] = convert_parameter_type(line[1])
                except IndexError:
                    pass

    DSC_parameters["path_to_folder"] = path_to_folder

    return DSC_parameters


def convert_parameter_type(value: str) -> Tuple[bool, int, float, str]:
    """
    Convert the type of a given string to bool, int or float if possible.

    Parameters
    ----------
    value : str
        Parameter string which should be converted.

    Returns
    -------
    value : bool or int or float or str
        Parameter as bool, int or float. If none is possible, the stripped
        string will be returned.

    """
    try:
        value = bool(strtobool(value))
        return value
    except ValueError:
        pass

    try:
        value = int(value)
        return value
    except ValueError:
        pass

    try:
        value = float(value)
        return value
    except ValueError:
        value = value.strip("'")
        return value


def get_byte_mode(DSC_dict: dict, data_key: str = "IRFMT") -> str:
    """
    Get the used byte mode of the BRUKER BES3T-data.

    For more information about BES3T go to BRUKER website or easyspin @ github.

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
        Raised if key value is not C, S, I, F or D.

    Returns
    -------
    byte_mode : str
        Return the  used byte mode as one string for
        numpy.fromfile(dtype=byte_mode).

    """
    byte_mode = ""

    if DSC_dict["BSEQ"] == "BIG":
        byte_mode += ">"
    elif DSC_dict["BSEQ"] == "LIT":
        byte_mode += "<"
    else:
        byte_mode += ">"

    if DSC_dict[data_key] in ("C", "S", "I"):
        byte_mode += "i"
        if DSC_dict[data_key] == "I":
            byte_mode += "4"
        elif DSC_dict[data_key] == "S":
            byte_mode += "2"
    elif DSC_dict[data_key] in ("F", "D"):
        byte_mode += "f"
        if DSC_dict[data_key] == "D":
            byte_mode += "8"
    elif DSC_dict[data_key] == "A":
        raise ValueError("Can't read BES3T data in ASCII format!")
    else:
        raise ValueError("Unknown value for keyword {} in .DSC file!".format(data_key))

    return byte_mode


def load_data_vector(DSC_dict: dict) -> np.array:
    """
    Load the binary intensity vector(s) from BRUKER BES3T-File.

    Parameters
    ----------
    DSC_dict : dict
        Dictionary with all parameters. Key is the same as in .DSC.

    Raises
    ------
    ValueError
        Will be raised if IKKF isn't CPLX or REAL or if the dimension isn't 1
        or 2.
    KeyError
        Raised if an axis is given but not the corresponding number of points.

    Returns
    -------
    cmplx_data_vector : np.array, np.complex128
        Real and imaginary part of the measured intensities.  If no imaginary
        part is measured, zeroes will be inserted.

    """
    get_data_dimension(DSC_dict)
    byte_mode = get_byte_mode(DSC_dict)

    DTA_file = os.path.join(DSC_dict["path_to_folder"], DSC_dict["TITL"] + ".DTA")

    data_vector = np.fromfile(DTA_file, dtype=byte_mode)

    if DSC_dict["IKKF"] == "CPLX":
        data_vector_real = data_vector[0::2]
        data_vector_imag = data_vector[1::2]
    elif DSC_dict["IKKF"] == "REAL":
        data_vector_real = 1 * data_vector
        data_vector_imag = np.zeros(len(data_vector))
    else:
        raise ValueError("Unknown value for keyword IKKF!")

    if DSC_dict["dimensions"] == 2:
        if "XPTS" and "YPTS" in DSC_dict:
            data_vector_real = data_vector_real.reshape(
                (DSC_dict["YPTS"], DSC_dict["XPTS"])
            )
            data_vector_imag = data_vector_imag.reshape(
                (DSC_dict["YPTS"], DSC_dict["XPTS"])
            )

        elif "XPTS" and "ZPTS" in DSC_dict:
            data_vector_real = data_vector_real.reshape(
                (DSC_dict["ZPTS"], DSC_dict["XPTS"])
            )
            data_vector_imag = data_vector_imag.reshape(
                (DSC_dict["ZPTS"], DSC_dict["XPTS"])
            )
        else:
            raise KeyError("Can't find needed keys from  (XPTS, YPTS, ZPTS)!")

    elif DSC_dict["dimensions"] == 1:
        pass
    else:
        raise ValueError("Can't handle dimensions {0}".format(DSC_dict["dimensions"]))

    cmplx_data_vector = data_vector_real + 1j * data_vector_imag

    return cmplx_data_vector.T


def load_axis_vector(axis: str, DSC_dict: dict) -> np.array:
    """
    Load the points of a given axis (x, y, z).

    Parameters
    ----------
    axis : str
        Name of the axis. Needs to start with x,X,y,Y or z,Z. E. g.: x-axis.
    DSC_dict : dict
        Dictionary with all parameters. Key is the same as in .DSC.

    Raises
    ------
    ValueError
        Will be raised if the given axis doesn't starts with an allowed
        character.

    Returns
    -------
    axis_vector : np.array
        Array with all axis points.

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

    axis_file = os.path.join(
        DSC_dict["path_to_folder"], DSC_dict["TITL"] + "." + axis + "GF"
    )
    data_key = axis + "FMT"

    try:
        byte_mode = get_byte_mode(DSC_dict, data_key=data_key)
        axis_vector = np.fromfile(axis_file, dtype=byte_mode)
    except (KeyError, FileNotFoundError):
        try:
            minimum = DSC_dict[axis + "MIN"]
        except KeyError:
            axis_vector = np.array([])
        else:
            width = DSC_dict[axis + "WID"]
            points = DSC_dict[axis + "PTS"]
            axis_vector = np.linspace(minimum, minimum + width, points)

    return axis_vector


def get_data_dimension(DSC_dict: dict) -> None:
    """
    Get the dimension of the measured spectrum (1d/2d).

    Parameters
    ----------
    DSC_dict : dict
        Dictionary with all parameters. Key is the same as in .DSC.

    Returns
    -------
    None
        The dimension will be safed in the given dictionary to the key
        'dimensions'.

    """
    dim = 0
    for axis in ("XTYP", "YTYP", "ZTYP"):
        if DSC_dict[axis] != "NODATA":
            dim += 1

    DSC_dict["dimensions"] = dim

    return None


def load_epr_bruker_bes3t(
    folder: str, start_directory: str = None
) -> Tuple[tuple, np.array]:
    """
    Load the whole dataset from BRUKER BES3T data folder into numpy.arrays.

    Time axis will be rescaled for OOP-ESEEM experiments, if 'FTAcqModeSlct'
    is 'Run from Tables'.

    Parameters
    ----------
    folder : str
        Name of the data folder with the corresponding data files.
    start_directory : str, optional
        Give the path starting from your home folder which the search for the
        data folder should start at. The default is None.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/*, then you
        can give this start_directory with start_directory='nice'. So
        the function call would be:
            load_epr_bruker_bes3t(this_folder, 'nice')

        Recommended:
            Without this parameter, the search for the right folder will be
            much longer.

    Returns
    -------
    axis : tuple
        Tuple of all axis vectors as three numpy.arrays (x, y, z).
    data : np.array, np.complex128
        All intensity values as one complex numpy.array.

    """
    path = get_full_path(folder, start_directory)
    DSC_parameters = get_DSC_parameters(path)

    data = load_data_vector(DSC_parameters)
    x = load_axis_vector("x", DSC_parameters)
    try:
        if DSC_parameters["FTAcqModeSlct"] == "Tables":
            x /= 2
    except KeyError:
        pass
    y = load_axis_vector("y", DSC_parameters)
    z = load_axis_vector("z", DSC_parameters)
    axis = (x, y, z)

    return axis, data


def load_epr_ESP_transient(
    folder: str, start_directory: str = None
) -> Tuple[tuple, np.array]:
    """
    Load data from a transient epr experiment measured with ESP380E.

    Parameters
    ----------
    folder : str
        Name of the folder with all data files. Data files need to have the
        same basename as folder, e. g. '/home/user/exp_1/exp_1.001'.
    start_directory : str, optional
        Give the path starting from your home folder which the search for
        the data folder should start at. The default is None.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/*, then you
        can give this start_directory with start_directory='nice'. So
        the function call would be:
            load_epr_ESP_transient(this_folder, 'nice')

        Recommended:
            Without this parameter, the search for the right folder will be
            much longer.

    Returns
    -------
    axis : tuple
        Return a tuple with all axis vectors  as two numpy.arrays
        (magnetic_field, time).
    data : np.array, np.complex128
        All intensity values as one complex numpy.array. Imaginary part is
        always 0.


    """
    path = get_full_path(folder, start_directory)
    # tr_info = get_transient_info(path)
    # time_axis = np.linspace(0, tr_info[0], tr_info[1])
    # points = int((abs(tr_info[3] - tr_info[2])) / tr_info[4] + 1)
    # magnetic_field = np.linspace(tr_info[2], tr_info[3], points)
    # axis = (magnetic_field, time_axis)

    # dimensions = (magnetic_field.shape[0], time_axis.shape[0])
    axis, data = get_transient_data(path)

    return axis, data


def get_transient_info(fpath: str) -> tuple:
    """
    Get all informations about the time axis and magnetic field vector from the
    .info file.

    Parameters
    ----------
    fpath : str
        Full path to the folder with the experimental data.

    Raises
    ------
    KeyError
        Raised if section 'MAGNETIC FIELD' or 'TRANSIENT' could not be found in
        .info file. Take care that the sections must be all upper case.

    Returns
    -------
    transient_info : tuple
        Contains the needed informations in the following order (time_length,
        time_points, mag_field_start, mag_field_stop, mag_field_step).

    """
    fbasename = os.path.basename(fpath)
    fname = os.path.join(fpath, fbasename + ".info")
    transient_info = [None] * 5

    with open(fname, "r") as info_file:
        current_section = ""
        for line in info_file.readlines():
            line = line.strip().split()
            if line == []:
                pass
            else:
                if line[0].isupper():
                    current_section = line[0]
                else:
                    if current_section == "MAGNETIC":
                        if line[0] == "Start:":
                            transient_info[2] = float(line[1])
                        elif line[0] == "Stop:":
                            transient_info[3] = float(line[1])
                        elif line[0] == "Step:":
                            transient_info[4] = float(line[1])
                        else:
                            pass
                    elif current_section == "TRANSIENT":
                        if line[0] == "Points:":
                            transient_info[1] = int(line[1])
                        elif line[0] == "Length:":
                            transient_info[0] = float(line[1])
                        else:
                            pass
                    else:
                        pass

            if None not in transient_info:
                break

    if transient_info[0] is None:
        raise KeyError("Couldn't find section 'TRANSIENT' in .info file!")
    if transient_info[2] is None:
        raise KeyError("Couldn't find section 'MAGNETIC FIELD' in .info file!")

    return tuple(transient_info)


def get_transient_data(fpath: str) -> Tuple[tuple, np.array]:
    """
    Get the measured intensities of the whole spectrum from a transient epr
    experiment measured by ESP380E.

    Parameters
    ----------
    fpath : str
        Full path to the folder with the experimental data.

    Returns
    -------
    axis : tuple
        Return a tuple with all axis vectors  as two numpy.arrays
        (magnetic_field, time).
    cmplx_data_vector : np.array, np.complex128
        Measured real intensities. All imaginary parts are zero.

    """
    data_vector_real = []
    magnetic_field = []

    file_found = False
    max_digits = 1
    while not file_found:
        fname = os.path.join(
            fpath,
            os.path.basename(fpath) + ".{0:0>{1}s}".format("1", str(max_digits)),
        )
        if os.path.isfile(fname):
            file_found = True
        else:
            max_digits += 1

        if max_digits >= 10:
            raise FileNotFoundError("No first ESP file could be found!")

    file_exists = True
    file = 1
    while file_exists:
        try:
            if file == 1:
                (field_point, time_axis), data = read_single_transient_file(
                    fpath, file, max_digits, time=True
                )
            else:
                field_point, data = read_single_transient_file(fpath, file, max_digits)
            file += 1
        except OSError:
            file_exists = False
        else:
            data_vector_real.append(data)
            magnetic_field.append(round(field_point, 1))

    magnetic_field = np.array(magnetic_field)
    sorting = np.argsort(magnetic_field)
    magnetic_field = magnetic_field[sorting]
    cmplx_data_vector = np.array(data_vector_real, dtype=np.complex128)
    cmplx_data_vector = cmplx_data_vector[sorting]
    axis = (magnetic_field, time_axis)

    return axis, cmplx_data_vector.T


def read_single_transient_file(
    fpath: str,
    filenumber: int,
    digits: int,
    time: bool = False,
) -> Tuple[float, np.array]:
    """
    Get the measured intensities of a single field point from a transient epr
    experiment measured by ESP380E.

    Parameters
    ----------
    fpath : str
        Full path to the folder with the experimental data.
    filenumber : int
        Number of the dataset for the magnetic field point e. g. 003.
    digits : int
        Number of digits from the highest filenumber (ESP380E has an increasing
        number as fileextension for each magnetic field point e. g. data.001).
    time : bool, optional
        If True, the time axis will be returned.

    Returns
    -------
    field : float
        Magnetic field point.
    time_axis : np.array
        Linear time axis.
    data_vector : np.array
        Measured intensities.

    """
    fbasename = os.path.basename(fpath)
    ending = ".{0:0>{1}s}".format(str(filenumber), str(digits))
    fname = os.path.join(fpath, fbasename + ending)

    try:
        data_vector = np.genfromtxt(fname, skip_header=5)
        data_vector = data_vector.flatten()
    except ValueError:
        data_vector = []
        with open(fname) as file:
            for i, line in enumerate(file.readlines()):
                if i < 5:
                    continue
                else:
                    line = line.strip().split()
                    data_vector.extend(line)
            data_vector = np.array(data_vector, dtype=np.complex128)

    with open(fname) as file:
        for i, line in enumerate(file.readlines()):
            if line.startswith("B0"):
                line = line.strip().split()
                field = float(line[2])
            if i == 3 and time:
                line = line.strip().split()
                points = int(line[1])
                start = float(line[2])
                stop = float(line[3])
                time_axis = np.linspace(start, stop, points)
            if i == 4:
                break

    if time:
        return (field, time_axis), data_vector

    return field, data_vector


def load_simulated_data(
    folder: str, start_directory: str = None
) -> Tuple[np.array, np.array, np.array]:
    """
    Load the simulated data from EasyPairSpin or data saved with
    saving.save_simulation(). Using numpy.loadtxt().

    Parameters
    ----------
    folder : str
        Name of the data folder with the corresponding data files.
    start_directory : str, optional
        Give the path starting from your home folder which the search for
        the data folder should start at. The default is None.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/*, then you
        can give this start_directory with start_directory='cooluser/nice'. So
        the function call would be:
            load_simulated_data(this_folder, 'cooluser/nice')

        Recommended:
            Without this parameter, the search for the right folder
            will be much longer.

    Returns
    -------
    x : np.array
        Axis vector for the x-axis.
    y : np.array
        Axis vector for the y-axis. Only returned, if the simulated data is 2d.
    intensity : np.array, np.complex128
        Simulated intensities. Either 1d or 2d.

    """
    path = get_full_path(folder, start_directory)

    x = np.loadtxt(os.path.join(path, "x_axis.txt"))
    try:
        intensity = np.loadtxt(os.path.join(path, "intensity.txt"))
    except ValueError:
        intensity = np.loadtxt(os.path.join(path, "intensity.txt"), dtype=np.complex_)

    try:
        y = np.loadtxt(os.path.join(path, "y_axis.txt"))
    except IOError:
        return (x, intensity)
    else:
        return (x, y, intensity)


def load_matlab(
    folder: str,
    start_directory: str = None,
    field: str = "field",
    signal: str = "signal",
) -> Tuple[np.array, np.array, np.array]:
    """
    Load EPR data from a matlab data file .mat

    Parameters
    ----------
    folder : str
        Name of the data folder with the corresponding data files.
    start_directory : str, optional
        Give the path starting from your home folder which the search for
        the data folder should start at. The default is None.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/*, then you
        can give this start_directory with start_directory='cooluser/nice'. So
        the function call would be:
            load_simulated_data(this_folder, 'cooluser/nice')
    field : str, optional
        Name of the field array. The default is 'field'.
    signal : str, optional
        Name of the signal array. The default is 'signal'.

    Returns
    -------
    axis : np.array
        Tuple of np.array containing the x and y axis.
    data : np.array
        np.array with the measured intensities.

    """
    path = get_full_path(folder, start_directory)
    mat_data = loadmat(path)
    x = mat_data[field].flatten()
    y = np.array([])
    data = mat_data[signal].flatten()
    axis = (x, y)

    return axis, data


def load_txt(
    folder: str, start_directory: str = None
) -> Tuple[np.array, np.array, np.array]:
    """
    Load EPR data from a .txt file.

    Parameters
    ----------
    folder : str
        Name of the data folder with the corresponding data files.
    start_directory : str, optional
        Give the path starting from your home folder which the search for
        the data folder should start at. The default is ''.
        E. g.: data is at /home/cooluser/nice/data/this_folder
        and you know, that all your data is in /home/cooluser/nice/*, then you
        can give this start_directory with start_directory='cooluser/nice'. So
        the function call would be:
            load_simulated_data(this_folder, 'cooluser/nice')

    Returns
    -------
    axis : np.array
        Tuple of np.array containing the x and y axis.
    data : np.array
        np.array with the measured intensities.

    """
    path = get_full_path(folder, start_directory)
    try:
        x, data = np.loadtxt(path)
        y = np.array([])
    except ValueError:
        try:
            x, data = np.loadtxt(path, unpack=True)
            y = np.array([])
        except ValueError:
            try:
                x, y, data = np.loadtxt(path)
            except ValueError:
                x, y, data = np.loadtxt(path, unpack=True)
    axis = (x, y)

    return axis, data
