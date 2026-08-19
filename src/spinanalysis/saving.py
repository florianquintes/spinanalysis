#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

from __future__ import annotations

import datetime
from importlib.metadata import metadata as _pkg_metadata
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_G_FIELDS: tuple[str, ...] = ("g1", "g2", "g_tri", "g")
_A_FIELDS: tuple[str, ...] = ("A1", "A2", "A3", "A4", "A5", "A_eseem", "omega_I")
_ZFS_FIELDS: tuple[str, ...] = ("D", "D_tri", "E", "E_tri", "J_ex", "J_0", "beta")
_NUCLEUS_PAIRS: tuple[tuple[str, str], ...] = (
    ("n1", "I1"),
    ("n2", "I2"),
    ("n3", "I3"),
    ("n4", "I4"),
    ("n5", "I5"),
)
_FRAME_FIELDS: tuple[str, ...] = (
    "g1_frame",
    "g2_frame",
    "g_tri_frame",
    "g_frame",
    "A1_frame",
    "A2_frame",
    "A3_frame",
    "A4_frame",
    "A5_frame",
    "D_frame",
    "D_tri_frame",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_authors() -> str:
    """Return the author name(s) from package metadata."""
    try:
        meta = _pkg_metadata("spinanalysis")
    except Exception:
        return "spinanalysis"
    author_email = meta.get("Author-email", "")
    if author_email:
        names: list[str] = []
        for part in author_email.split(","):
            part = part.strip()
            if "<" in part:
                names.append(part.split("<")[0].strip())
            else:
                names.append(part)
        return ", ".join(names)
    return meta.get("Author", "spinanalysis")


def _resolve_output_dir(path: str | Path | None) -> Path:
    """Resolve the output directory for :func:`write_out_file`.

    If *path* is given, use it (creating it if needed).  Otherwise
    create ``spinanalysis_YYYY-MM-DD_N`` in the current working
    directory, where *N* is the lowest available positive integer.
    """
    if path is not None:
        out_dir = Path(path)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    base_name = f"spinanalysis_{date_str}"
    cwd = Path.cwd()

    n = 1
    while True:
        candidate = cwd / f"{base_name}_{n}"
        if not candidate.exists():
            break
        n += 1

    candidate.mkdir(parents=True)
    return candidate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_plot(
    fname: str,
    *figures: object,
    path: str | Path | None = None,
    **kwargs: Any,
) -> None:
    """
    Save the figures plotted with matplotlib.

    Parameters
    ----------
    fname : str
        Filename for the figure(s). If multiple figures are given,
        '_[number]' will be appended to the filename.
    *figures : object
        Matplotlib figure object(s).
    path : str or pathlib.Path, optional
        Directory where the figures will be stored. The default is
        ``~/spinanalysis/plots/``.
    **kwargs : Any
        Other keyword arguments. Will be passed to ``plt.savefig()``.
        See matplotlib documentation for further information.

    Returns
    -------
    None

    """

    if not figures:
        raise ValueError("At least one figure must be provided.")

    p = Path(fname)
    fmt = p.suffix
    base_name = p.stem

    if isinstance(figures[0], list):
        figures = tuple(*figures)

    multi_plot = len(figures) > 1

    if "format" in kwargs:
        fmt = "." + kwargs["format"]

    save_path = (
        Path(path) if path is not None else Path.home() / "spinanalysis" / "plots"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    for n, fig in enumerate(figures):
        if multi_plot:
            save_name = save_path / f"{base_name}_{n + 1}{fmt}"
        else:
            save_name = save_path / f"{base_name}{fmt}"
        fig.savefig(str(save_name), **kwargs)


def save_simulation(
    name: str,
    *data: np.ndarray,
    path: str | Path | None = None,
) -> None:
    """
    Save the simulated data at ``[path]/[name]/[files]`` using
    :func:`numpy.savetxt`.

    Parameters
    ----------
    name : str
        Folder name for the dataset.
    *data : np.ndarray
        Arrays with the simulated data. If 2 arrays: ``x_axis``, ``int``;
        if 3 arrays: ``x_axis``, ``y_axis``, ``int``.
    path : str or pathlib.Path, optional
        Directory where the data will be stored. The default is
        ``~/spinanalysis/simulations/``.

    Raises
    ------
    ValueError
        If the number of data arrays is not 2 or 3.

    Returns
    -------
    None

    """

    if name.lower().endswith(".txt"):
        name = name[:-4]

    save_path = (
        Path(path) if path is not None else Path.home() / "spinanalysis" / "simulations"
    )
    save_path.mkdir(parents=True, exist_ok=True)

    save_folder = save_path / name
    save_folder.mkdir(parents=True, exist_ok=True)

    if len(data) == 2:
        x, intensity = data
    elif len(data) == 3:
        x, y, intensity = data
    else:
        raise ValueError("Can't handle {0}D data. Need 2D or 3D.".format(len(data)))

    np.savetxt(str(save_folder / "x_axis.txt"), x)
    np.savetxt(str(save_folder / "intensity.txt"), intensity)

    if len(data) == 3:
        np.savetxt(str(save_folder / "y_axis.txt"), y)


def write_out_file(
    Sys: Any,
    Exp: Any,
    SimOpt: Any,
    *FitOpt: Any,
    current_best: bool = False,
    path: str | Path | None = None,
) -> Path:
    """
    Write an output file with all data from ``Sys``, ``Exp``, ``SimOpt``
    and, if running in optimization mode, ``FitOpt``.

    Parameters
    ----------
    Sys : object
        :class:`~spinanalysis.epr.Spinsystem` object.
    Exp : object
        :class:`~spinanalysis.epr.Experimental` object.
    SimOpt : object
        :class:`~spinanalysis.epr.SimulationOptions` object.
    *FitOpt : object
        :class:`~spinanalysis.epr.FittingOptions` object (optional, only
        in optimization mode).
    current_best : bool, optional
        ``True`` if *Sys* and *Exp* are the current best while running
        in optimization mode.  ``False`` if they are the final result or
        in normal simulation mode.  The default is ``False``.
    path : str or pathlib.Path, optional
        Directory where the output file will be stored.  If ``None``
        (the default), a new folder ``spinanalysis_YYYY-MM-DD_N`` is
        created in the current working directory, where *N* is the
        lowest available positive integer.

    Returns
    -------
    pathlib.Path
        Path to the written output file.

    """

    title_part = "_current_best" if current_best else "_result"
    fit_mode = len(FitOpt) == 1

    out_dir = _resolve_output_dir(path)

    routine = FitOpt[0].routine if fit_mode else SimOpt.routine
    out_file = out_dir / f"{routine}{title_part}.spinanalysis"

    authors = _get_authors()

    with open(out_file, "w") as out:
        out.write("{:#^50s}\n".format(""))

        header = " CURRENT BEST " if current_best else " OUTPUT-FILE "
        out.write("{:#^50s}\n".format(header))

        out.write("{:#^50s}\n".format(" " + routine + " "))
        out.write("{:#^50s}\n".format(f" AUTHOR: {authors.upper()} "))
        out.write("{:#^50s}\n".format(""))
        out.write("\n")

        # -- Spinsystem ------------------------------------------------
        out.write("\n\n")
        out.write("{:#^50s}\n".format(" SPINSYSTEM "))
        out.write("\n\n")

        # g-Tensor
        out.write("{: ^15s} {: ^20s}\n".format("g-Tensor", "Value"))
        with np.printoptions(formatter={"float": "{:0.7f}".format}):
            for key in _G_FIELDS:
                value = getattr(Sys, key, None)
                if value is not None:
                    out.write("{0:>14s}: {1}\n".format(key, value))

        # A-Tensor
        out.write("\n")
        out.write("{: ^15s} {: ^20s}\n".format("A-Tensor", "Value / MHz"))
        with np.printoptions(formatter={"float": "{:2.2f}".format}):
            for key in _A_FIELDS:
                value = getattr(Sys, key, None)
                if value is not None:
                    out.write("{0:>14s}: {1}\n".format(key, value))

        # ZFS
        out.write("\n")
        out.write("{: ^15s} {: ^20s}\n".format("ZFS", "Value / MHz"))
        for key in _ZFS_FIELDS:
            value = getattr(Sys, key, None)
            if value is not None:
                out.write("{0:>14s}: {1}\n".format(key, value))

        # Orientations
        out.write("\n")
        out.write("{: ^15s} {: ^20s}\n".format("Orientations", "Angle / rad"))
        with np.printoptions(formatter={"float": "{: 0.1f}".format}):
            for key in _FRAME_FIELDS:
                value = getattr(Sys, key, None)
                if value is not None:
                    out.write("{0:>14s}: {1}\n".format(key, value))

        # Nuclear spins
        out.write("\n")
        out.write("{: ^15s}|{: ^20s}\n".format("Nuclear spin", "Number of cores"))
        for core_key, spin_key in _NUCLEUS_PAIRS:
            core_val = getattr(Sys, core_key, None)
            spin_val = getattr(Sys, spin_key, None)
            if core_val is not None and spin_val is not None:
                out.write(
                    "{0:>7s}: {1:<6.1f}|{2:>9s}: {3:<9}\n".format(
                        core_key, core_val, spin_key, spin_val
                    )
                )

        out.write("\n")
        out.write("{:>14s}: {:.2f} mT\n".format("width_gauss", Sys.width_gauss))

        # -- Experimental ----------------------------------------------
        out.write("\n\n")
        out.write("{:#^50s}\n".format(" EXPERIMENTAL "))
        out.write("\n\n")

        out.write("{0:>20s}: {1:<9.6f} mT\n".format("microwave amplitude", Exp.B_mw))
        out.write(
            "{0:>20s}: {1:<9.6f} GHz\n".format("microwave frequency", Exp.freq_mw / 1e9)
        )
        out.write("{0:>20s}: {1:<7.2f} mT\n".format("min B_z", Exp.B_z.min()))
        out.write("{0:>20s}: {1:<7.2f} mT\n".format("max B_z", Exp.B_z.max()))
        out.write("{0:>20s}: {1}\n".format("B_z points", len(Exp.B_z)))

        # -- Simulation options ----------------------------------------
        out.write("\n\n")
        out.write("{:#^50s}\n".format(" SIMULATION OPTIONS "))
        out.write("\n\n")

        out.write("{0:>20s}: {1:<20s}\n".format("routine", SimOpt.routine))
        out.write("{0:>20s}: {1} cores\n".format("run at", SimOpt.cpu_cores))
        out.write("{0:>20s}: {1:<20}\n".format("knots", SimOpt.knots))
        out.write("{0:>20s}: {1} space\n".format("using", SimOpt.space))

        # -- Fitting options -------------------------------------------
        if fit_mode:
            out.write("\n\n")
            out.write("{:#^50s}\n".format(" FITTING OPTIONS "))
            out.write("\n\n")

            for key, value in FitOpt[0].model_dump().items():
                if value is not None:
                    out.write("{0:>20s}: {1:<20s}\n".format(key, str(value)))

    return out_file
