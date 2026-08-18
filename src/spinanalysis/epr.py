#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data models for EPR systems, experiments, and optimization settings.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import numpy as np
import scipy.constants as constant
import warnings
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from spinanalysis import profiles
from typing import Any, ClassVar


class ExperimentalInput(BaseModel):
    """Validated input data for :class:`Experimental`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    magnetic_field: np.ndarray | None = None
    real_int: np.ndarray | None = None
    imag_int: np.ndarray | None = None
    cmplx_int: np.ndarray | None = None
    time_axis: np.ndarray | None = None
    rescale: bool = True

    @field_validator(
        "magnetic_field",
        "real_int",
        "imag_int",
        "cmplx_int",
        "time_axis",
        mode="before",
    )
    @classmethod
    def convert_arrays(cls, value: object) -> np.ndarray | None:
        """Convert array-like input to an independent NumPy array."""
        if value is None:
            return None
        return np.asarray(value).copy()

    @field_validator("real_int", "imag_int", "time_axis", mode="after")
    @classmethod
    def require_real_arrays(cls, value: np.ndarray | None) -> np.ndarray | None:
        """Reject complex-valued real intensity and time-axis inputs."""
        if value is not None and np.iscomplexobj(value):
            raise ValueError("real_int, imag_int, and time_axis must be real-valued")
        return value

    @field_validator("cmplx_int", mode="after")
    @classmethod
    def require_complex_intensity(cls, value: np.ndarray | None) -> np.ndarray | None:
        """Require complex-valued input for ``cmplx_int``."""
        if value is not None and not np.iscomplexobj(value):
            raise ValueError("cmplx_int must contain complex-valued data")
        return value

    @field_validator("magnetic_field", mode="after")
    @classmethod
    def require_real_field(cls, value: np.ndarray | None) -> np.ndarray | None:
        """Reject complex-valued magnetic-field input."""
        if value is not None and np.iscomplexobj(value):
            raise ValueError("magnetic_field must be real-valued")
        return value

    @model_validator(mode="after")
    def validate_intensity_sources(self) -> "ExperimentalInput":
        """Ensure complex and component intensity inputs are not combined."""
        if self.cmplx_int is not None and (
            self.real_int is not None or self.imag_int is not None
        ):
            raise ValueError("cmplx_int cannot be combined with real_int or imag_int")
        if self.real_int is not None and self.imag_int is not None:
            if self.real_int.shape != self.imag_int.shape:
                raise ValueError("real_int and imag_int must have matching shapes")
        return self


class _MutableModel(BaseModel):
    """Compatibility base for mutable runtime models."""

    model_config = ConfigDict(
        extra="allow", arbitrary_types_allowed=True, validate_assignment=True
    )

    @field_validator(
        "g1",
        "g2",
        "g_tri",
        "g",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "population",
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
        mode="before",
        check_fields=False,
    )
    @classmethod
    def arrays(cls, value: object) -> np.ndarray:
        return np.asarray(value, dtype=float)

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)


class EPR_Parameters(_MutableModel):
    """

    A class containing all parameters for various radical pair simulations.

    Attributes
    ----------
    g1 : 1d-Array, np.float64
        g-Tensor of electron 1.
    g2 : 1d-Array, np.float64
        g-Tensor of electron 2.
    g_tri : 1d-Array, np.float64
        g-Tensor of a triplet radical.
    g : 1d-Array, np.float64
        g-Tensor of a radical.
    A1 : 1d-Array, np.float64
        A-Tensor of nuclei 1 in Megahertz.
    A2 : 1d-Array, np.float64
        A-Tensor of nuclei 2 in Megahertz.
    A3 : 1d-Array, np.float64
        A-Tensor of nuclei 3 in Megahertz.
    A4 : 1d-Array, np.float64
        A-Tensor of nuclei 4 in Megahertz.
    A5 : 1d-Array, np.float64
        A-Tensor of nuclei 5 in Megahertz.
    A_eseem : float64
        Hyperfine coupling for the nuclei in OOP-ESEEM in Megahertz.
    omega_I : float64
        Nuclei frequency in OOP-ESEEM in Megahertz.
    D : float64
        Zero field splitting parameter D in Megahertz.
    D_tri : float64
        Zero field splitting parameter D of a triplet in Megahertz.
    E : float64
        Zero field splitting parameter E in Megahertz.
    E_tri : float64
        Zero field splitting parameter E of a triplet in Megahertz.
    beta : float64
        Decay rate of the exchange coupling in Å^-1.
    J_0 : float64
        Zero distance exchange coupling constant in Megahertz.
    J_ex : float64
        Exchange coupling in Megahertz.
    g1_frame : 1d-Array, np.float64
        Orientation of electron spin 1 in radian.
    g2_frame : 1d-Array, np.float64
        Orientation of electron spin 2 in radian.
    g_tri_frame : 1d-Array, np.float64
        Orientation of g_tri in radian.
    g_frame : 1d-Array, np.float64
        Orientation of g in radian.
    A1_frame : 1d-Array, np.float64
        Orientation of nuclei spin 1 in radian.
    A2_frame : 1d-Array, np.float64
        Orientation of nuclei spin 2 in radian.
    A3_frame : 1d-Array, np.float64
        Orientation of nuclei spin 3 in radian.
    A4_frame : 1d-Array, np.float64
        Orientation of nuclei spin 4 in radian.
    A5_frame : 1d-Array, np.float64
        Orientation of nuclei spin 5 in radian.
    D_frame : 1d-Array, np.float64
        Orientation of dipol coupling in radian.
    D_tri_frame : 1d-Array, np.float64
        Orientation of triplet dipol coupling in radian.
    n1 : int
        Number of chemically equivalent atoms.
    I1 : float64
        Corresponding nuclear spin.
    n2 : int
        Number of chemically equivalent atoms.
    I2 : float64
        Corresponding nuclear spin.
    n3 : int
        Number of chemically equivalent atoms.
    I3 : float64
        Corresponding nuclear spin.
    n4 : int
        Number of chemically equivalent atoms.
    I4 : float64
        Corresponding nuclear spin.
    n5 : int
        Number of chemically equivalent atoms.
    I5 : float64
        Corresponding nuclear spin.
    width_gauss : float64
        Gaussian linewidth in mT.
    T_relax_1 : float64
        Longitudinal relaxation time in s.
    T_relax_2 : float64
        Transversal relaxation time in s.
    decay : float64
        Exponential decay time for hilbert space simulations in s.
    T_pm : float64
        Phase memory time (OOP ESEEM).
    population : 1d-Array, np.float64
        Populations of the initial density matrix of a triplet precursor in
        zero field.
    amplitude : float64
        Amplitude of the spectra for OOP-ESEEM.

    """

    g1: np.ndarray = Field(default_factory=lambda: np.array([2.002, 2.002, 2.002]))
    g2: np.ndarray = Field(default_factory=lambda: np.array([2.004, 2.004, 2.004]))
    g_tri: np.ndarray = Field(default_factory=lambda: np.array([2.002, 2.002, 2.002]))
    g: np.ndarray = Field(default_factory=lambda: np.array([2.002, 2.002, 2.002]))
    A1: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A2: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A3: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A4: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A5: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A_eseem: float = 0.0
    omega_I: float = 0.0
    D: float = 1.0
    D_tri: float = 0.0
    E: float = 0.0
    E_tri: float = 0.0
    J_ex: float = 0.0
    J_0: float = 0.0
    beta: float = 1.4
    g1_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    g2_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    g_tri_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    g_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A1_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A2_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A3_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A4_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    A5_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    D_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    D_tri_frame: np.ndarray = Field(default_factory=lambda: np.zeros(3))
    width_gauss: float = 0.5
    T_relax_1: float = 0.0
    T_relax_2: float = 0.0
    T_pm: float = 0.0
    decay: float = 0.0
    population: np.ndarray = Field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    amplitude: float = 0.0

    def __init__(self, **data: Any) -> None:
        """
        Initialize object of class 'Spinsystem' for radical pair simulation.

        Returns
        -------
        None.

        """
        if data:
            super().__init__(**data)
            return
        super().__init__()

        # [SPINSYSTEM]
        self.g1 = np.array([2.002, 2.002, 2.002])
        self.g2 = np.array([2.004, 2.004, 2.004])
        self.g_tri = np.array([2.002, 2.002, 2.002])
        self.g = np.array([2.002, 2.002, 2.002])

        self.A1 = np.zeros(3)  # MHz
        self.A2 = np.zeros(3)  # MHz
        self.A3 = np.zeros(3)  # MHz
        self.A4 = np.zeros(3)  # MHz
        self.A5 = np.zeros(3)  # MHz
        self.A_eseem = 0.0  # MHz
        self.omega_I = 0.0  # MHz
        self.D = 1.0  # MHz
        self.D_tri = 0.0  # MHz
        self.E = 0  # MHz
        self.E_tri = 0  # MHz
        self.J_ex = 0  # MHz
        self.J_0 = 0  # MHz
        self.beta = 1.4  # Å^-1.

        # [ORIENTATIONS]
        self.g1_frame = np.zeros(3)  # Euler angle / rad
        self.g2_frame = np.zeros(3)
        self.g_tri_frame = np.zeros(3)
        self.g_frame = np.zeros(3)
        self.A1_frame = np.zeros(3)
        self.A2_frame = np.zeros(3)
        self.A3_frame = np.zeros(3)
        self.A4_frame = np.zeros(3)
        self.A5_frame = np.zeros(3)
        self.D_frame = np.zeros(3)
        self.D_tri_frame = np.zeros(3)

        # [LINEWIDTHS]
        self.width_gauss = 0.5  # Gaussian linewidth / mT

        # [RELAXATION TIMES]
        self.T_relax_1 = 0.0  # longitudinal
        self.T_relax_2 = 0.0  # transversal
        self.T_pm = 0.0  # phase memory time
        self.decay = 0.0

        # [DENSITY MATRIX]
        self.population = np.array([1.0, 0.0, 0.0])

        # [OTHER]
        self.amplitude = 0.0


class Spinsystem(EPR_Parameters):
    """
    A class containing all parameters for various radical pair simulations.

    Attributes
    ----------
    g1_iso : float64
        Isotropic g value of electron 1.
    g2_iso : float64
        Isotropic g value of electron 2.
    n1 : int
        Number of chemically equivalent atoms.
    I1 : float64
        Corresponding nuclear spin.
    n2 : int
        Number of chemically equivalent atoms.
    I2 : float64
        Corresponding nuclear spin.
    n3 : int
        Number of chemically equivalent atoms.
    I3 : float64
        Corresponding nuclear spin.
    n4 : int
        Number of chemically equivalent atoms.
    I4 : float64
        Corresponding nuclear spin.
    n5 : int
        Number of chemically equivalent atoms.
    I5 : float64
        Corresponding nuclear spin.
    donor_list : np.array
        Defines which atom groups are donor groups.
    acceptor_list : np.array
        Defines which atom groups are acceptor groups.
    frame_group_i : list
        Define a frame_group which will be used in optimization mode. Each
        frame group contains the names of the angle lists which always will
        have same values during optimization. i is a variable and can be
        whatever you want. You can define as many frame groups as you want. An
        example frame group would be: frame_group_1 = ['A1', 'A2', 'D']. This
        list means, that A2_frame and D_frame will always have the same values
        as A1_frame, no matter which values were given to them.
    spin_system : str
        Define the spin system by one out of: "rp" (radical pair), "doub"
        (doublet), "trip" (triplet), "tdp" (triplet-doublet pair).
    precursor : str
        State of the precursor. One out of: "zf", "eigen", "singlet",
        "triplet-zf", "triplet-eigen", "coupled", "basis".
    dynamics : np.array
        Matrix with rate constants of relaxation process in 1/s. For further
        information see the documentation.
    distribution_order : int
        Number of Gaussians used for Multi-Gauss-Fitting.
    distribution : np.array
        Distance distribution of the radical pair.

    Methods
    -------
    load(profile_name: str)
        Load Spinsystem values from a config file (profile_name.ini).

        Recommended
    save(profile_name: str)
        Save the current spinsystem values as a config file (profile_name.ini).

        Recommended
    _get_g_iso()
        Determine both g_iso values. Needed in simulation.

    Examples
    --------
    Initialize a new object of class <Spinsystem>:

    >>> Sys = Spinsystem()
    >>> Sys.g1
    np.array([2.002, 2.002, 2.002])
    >>> Sys.g1_iso
    2.002

    Change values:

    >>> Sys.g1 = np.array([2.0024, 2.00381, 2.0027])
    >>> Sys.get_g_iso()
    >>> Sys.g1
    np.array([2.0024 , 2.00381, 2.0027 ])
    >>> Sys.g1_iso
    2.00297

    Create a new spinsystem profile from an empty template and load it:

    >>> Sys_profile = profiles.new_spinsystem_profile()
    >>> Sys_profile['g_1'] = [2.0034, 2.00156, 2.00228] #  use list not array!
    >>> profiles.add_profile(Sys.profile, 'spinsystem', 'Sys_prof_1')
    >>> Sys_2 = Spinsystem()
    >>> Sys_2.load_profile('Sys_prof_1')
    >>> Sys_2.g1
    np.array([2.0034 , 2.00156, 2.00228])

    You can also save your current spinsystem as a new profile:

    >>> Sys_3 = Spinsystem()
    >>> Sys_3.g1 = np.array([1, 2, 3]) #  either array or list
    >>> Sys_3.save('Sys_prof_2')
    >>> Sys_4 = Spinsystem()
    >>> Sys_4.load('Sys_prof_2')
    >>> Sys_4.g1
    np.array([1., 2., 3.])

    """

    g1_iso: float = Field(default=0.0, exclude=True)
    g2_iso: float = Field(default=0.0, exclude=True)
    spin_system: str = "rp"
    precursor: str = "singlet"
    acceptor_list: np.ndarray = Field(default_factory=lambda: np.array([1, 2, 3]))
    donor_list: np.ndarray = Field(default_factory=lambda: np.array([4, 5]))
    n1: int = 0
    I1: float = 0.0
    n2: int = 0
    I2: float = 0.0
    n3: int = 0
    I3: float = 0.0
    n4: int = 0
    I4: float = 0.0
    n5: int = 0
    I5: float = 0.0
    dynamics: np.ndarray | None = None
    distribution_order: int = 3
    distribution: np.ndarray | None = None

    @field_validator("acceptor_list", "donor_list", mode="before")
    @classmethod
    def convert_nucleus_lists(cls, value: object) -> np.ndarray:
        return np.asarray(value, dtype=int)

    @field_validator("n1", "n2", "n3", "n4", "n5", mode="before")
    @classmethod
    def validate_nuclear_counts(cls, value: object) -> int:
        """Require non-negative integer counts for equivalent nuclei."""
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("nuclear counts must be non-negative integers")
        if value < 0:
            raise ValueError("nuclear counts must be non-negative integers")
        return int(value)

    @field_validator("I1", "I2", "I3", "I4", "I5", mode="before")
    @classmethod
    def validate_nuclear_spins(cls, value: object) -> float:
        """Require non-negative nuclear spins in half-integer steps."""
        if isinstance(value, bool):
            raise ValueError("nuclear spins must be non-negative half-integers")
        try:
            spin = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "nuclear spins must be non-negative half-integers"
            ) from error
        if (
            not np.isfinite(spin)
            or spin < 0
            or not np.isclose(spin * 2, round(spin * 2))
        ):
            raise ValueError("nuclear spins must be non-negative half-integers")
        return spin

    PROFILE_FIELDS: ClassVar[tuple[str, ...]] = (
        "spin_system",
        "precursor",
        "acceptor_list",
        "donor_list",
        "n1",
        "I1",
        "n2",
        "I2",
        "n3",
        "I3",
        "n4",
        "I4",
        "n5",
        "I5",
        "dynamics",
        "distribution_order",
        "distribution",
        "g1",
        "g2",
        "g_tri",
        "g",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A_eseem",
        "omega_I",
        "D",
        "D_tri",
        "E",
        "E_tri",
        "J_ex",
        "J_0",
        "beta",
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
        "width_gauss",
        "T_relax_1",
        "T_relax_2",
        "T_pm",
        "decay",
        "population",
        "amplitude",
    )
    FRAME_FIELDS: ClassVar[tuple[str, ...]] = (
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

    def __init__(self, degree: bool = False, **data: Any) -> None:
        """
        Initialize object of class 'Spinsystem' for radical pair simulation.

        Returns
        -------
        None.

        """
        if data:
            super().__init__(**data)
            self._get_g_iso()
            if degree:
                self._convert_frames_to_radians()
            return
        super().__init__()

        # [SPIN SYSTEM]
        self._get_g_iso()
        self.spin_system = "rp"
        self.precursor = "singlet"

        # [NUCLEI]
        self.acceptor_list = np.array([1, 2, 3])
        self.donor_list = np.array([4, 5])

        self.n1 = 0  # Number of chemically equivalent atoms
        self.I1 = 0  # corresponding nuclear spin
        self.n2 = 0
        self.I2 = 0
        self.n3 = 0
        self.I3 = 0
        self.n4 = 0
        self.I4 = 0
        self.n5 = 0
        self.I5 = 0

        # [DYNAMICS]
        self.dynamics = None

        # [DISTRIBUTIONS]
        self.distribution_order = 3
        self.distribution = None
        if degree:
            self._convert_frames_to_radians()

    def _convert_frames_to_radians(self) -> None:
        """Convert all orientation fields from degrees to radians."""
        for key in self.FRAME_FIELDS:
            setattr(
                self, key, np.asarray(getattr(self, key), dtype=float) * np.pi / 180
            )

    def _get_g_iso(self) -> None:
        """Get g1_iso and g2_iso."""
        self.g1_iso = np.mean(self.g1)
        self.g2_iso = np.mean(self.g2)

    def __setattr__(self, name: str, value: object) -> None:
        """Keep derived isotropic g values in sync with tensor assignments."""
        super().__setattr__(name, value)
        if name in ("g1", "g2") and hasattr(self, "g1") and hasattr(self, "g2"):
            self._get_g_iso()

    def load(self, profile_name: str, degree: bool = False) -> None:
        """
        Load a spinsystem from a profile.

        Load the settings from ~/.config/spinanalysis/profiles/spinsystem/
        [profile_name].ini into the Spinsystem object. Overwrites previous
        settings.

        Parameters
        ----------
        profile_name : str
            Name of the Spinsystem profile which will be loaded.
        degree : bool, optional
            If True, the angle values in the profile are given in degree not
            radian. Thus, they will be converted to radian. If False, the
            angles are given in radian and will therefore not be converted. The
            default is 'False'.

        Returns
        -------
        None.

        """
        spinsystem_profile = profiles.load_profile(profile_name, "spinsystem")

        values = spinsystem_profile["main"]
        for key in self.PROFILE_FIELDS:
            if key not in values:
                continue
            value = values[key]
            if value is None or value == "None":
                value = None
            elif key == "distribution":
                if value is None:
                    value = None
                else:
                    distribution = np.asarray(value, dtype=float)
                    if distribution.size % 2:
                        raise ValueError(
                            "The distribution profile must contain two rows."
                        )
                    value = distribution.reshape(2, -1)
            elif isinstance(value, (list, tuple)):
                value = np.asarray(value)
            if degree and key in self.FRAME_FIELDS:
                value = np.asarray(value, dtype=float) * np.pi / 180
            setattr(self, key, value)

        self._get_g_iso()

    def save(self, profile_name: str = "", degree: bool = False) -> None:
        """
        Save the spinsystem as a profile.

        Save the spinsystem object as a spinsystem profile using
        profile_management.add_profile(). Load the spinsystem using
        Spinsystem.load_profile(<profile_name>).

        Parameters
        ----------
        profile_name : str, optional
            Name of the profile.  If no profile name is given, a default one
            will be generated by get_profile_name(). The default is ''.

        Returns
        -------
        None
            Nothing will be returned.

        """
        spinsys_profile = profiles.new_spinsystem_profile()

        for key in spinsys_profile["main"]:
            value = getattr(self, key)
            if degree and key in self.FRAME_FIELDS:
                value = np.asarray(value, dtype=float) * 180 / np.pi
            spinsys_profile["main"][key] = (
                value.flatten().tolist() if isinstance(value, np.ndarray) else value
            )

        profiles.add_profile(spinsys_profile, "spinsystem", pname=profile_name)


class Experimental(_MutableModel):
    """

    A class containing all experimental parameters and data.

    Attributes
    ----------
    B_z : 1d-Array, np.float64
        External magnetic field points in mT used for simulation. Conversions
        allowed.
    freq_mw : float64
        Frequency of induced microwave radiation in Gigahertz.
    magnetic_field : 1d-Array, np.float64
        Same as B_z, but will never be changed.
    int : np.array, np.complex128
        Real and imaginary part of the measured intensities. 1d or 2d.
    time_axis : 1d-Array, optional
        Contains all experimental time points.
    spec_sim : 1d-Array, np.float64
        Calculated spectrum. At initialisation empty.

    Methods
    -------
    get_linear_time_axis()
        Get a linear time axis using the given boundaries from self.t_scale
        with self.t_points points.

    """

    B_z: np.ndarray = Field(
        default_factory=lambda: np.linspace(240, 260, 100), exclude=True
    )
    magnetic_field: np.ndarray = Field(
        default_factory=lambda: np.linspace(240, 260, 100), exclude=True
    )
    freq_mw: float = Field(default=9.7e9)
    B_mw: float = Field(default=1e-3)
    t_scale: list[float] = Field(default_factory=lambda: [0.0, 2e-6], exclude=True)
    t_points: int = Field(default=2, exclude=True)
    time_axis: np.ndarray | None = Field(default=None, exclude=True)
    int: np.ndarray | None = Field(default=None, exclude=True)
    spec_sim: np.ndarray = Field(default_factory=lambda: np.zeros(100), exclude=True)

    def __init__(
        self,
        magnetic_field: np.array = None,
        real_int: np.array = None,
        imag_int: np.array = None,
        cmplx_int: np.array = None,
        time_axis: np.array = None,
        rescale: bool = True,
        **data: Any,
    ):
        """
        Initialize object of class 'Experimental' for radical pair simulation.

        Parameters
        ----------
        magnetic_field : 1d-Array, np.float64, optional
            Contains all experimental external magnetic field points.
        real_int : np.array, np.float64, optional
            Real part of the measured intensities. 1d or 2d.
        imag_int : np.array, np.float64, optional
            Imaginary part of the measured intensities. 1d or 2d.
        cmplx_int : np.array, np.complex128, optional
            Real and imaginary part of the measured intensities. 1d or 2d. If
            this parameter is given, real_int and imag_int will be ignored.
        time_axis : 1d-Array, optional
            Contains all experimental time points in s.
        rescale : bool, optional
            If True, the experimental intensities will be scaled to a maximum
            of 1. Default is 'True'.

        Returns
        -------
        None.

        """
        super().__init__()
        input_data = ExperimentalInput(
            magnetic_field=magnetic_field,
            real_int=real_int,
            imag_int=imag_int,
            cmplx_int=cmplx_int,
            time_axis=time_axis,
            rescale=rescale,
        )

        if input_data.magnetic_field is None:
            self.B_z = np.linspace(240, 260, 100)
        else:
            self.B_z = np.asarray(input_data.magnetic_field, dtype=float)
        self.magnetic_field = self.B_z.copy()
        self.freq_mw = data.pop("freq_mw", 9.7e9)  # Microwave radiation / Hz
        self.B_mw = data.pop("B_mw", 1e-3)

        self.t_scale = [0, 2e-6]
        self.t_points = 2
        self.time_axis = None
        if input_data.time_axis is not None:
            self.time_axis = np.asarray(input_data.time_axis, dtype=float)
            if self.time_axis.ndim != 1 or self.time_axis.size < 2:
                raise ValueError(
                    "time_axis must be a one-dimensional array with at least two points."
                )
            self.t_scale[0] = self.time_axis.min()
            self.t_scale[1] = self.time_axis.max()
            self.t_points = self.time_axis.size

        self.int = None
        if input_data.cmplx_int is not None:
            self.int = np.asarray(input_data.cmplx_int, dtype=np.complex128)
        else:
            if input_data.real_int is not None:
                real_int = np.asarray(input_data.real_int, dtype=float)
                self.int = np.zeros(real_int.shape, dtype="complex128")
                self.int.real = real_int
            if input_data.imag_int is not None:
                imag_int = np.asarray(input_data.imag_int, dtype=float)
                if self.int is not None:
                    if self.int.shape != imag_int.shape:
                        raise ValueError(
                            "real_int and imag_int must have matching shapes."
                        )
                    self.int.imag = imag_int
                else:
                    self.int = np.zeros(imag_int.shape, dtype="complex128")
                    self.int.imag = imag_int

        if self.int is not None:
            if self.int.ndim == 1 and input_data.magnetic_field is not None:
                if self.int.shape[0] != self.B_z.size:
                    raise ValueError(
                        "The intensity data and magnetic-field axis must have compatible lengths."
                    )
            if input_data.rescale:
                maximum = np.abs(self.int).max()
                if maximum:
                    self.int /= maximum
            self.spec_sim = np.zeros(self.int.shape, dtype="complex128")
        else:
            self.spec_sim = np.zeros(self.B_z.shape, dtype="complex128")

    def get_linear_time_axis(
        self, t_min: float = None, t_max: float = None, t_points: int = None
    ) -> None:
        """
        Get a linear timea axis for transient simulations.

        Get a linear time axis using the given boundaries from self.t_scale
        with self.t_points points.

        Parameters
        ----------
        t_min : float, optional
            Left boundary of the time axis. If None is given, the current value
            of self.t_scale[0] will be used. Else, the value of self.t_scale[0]
            will be replaced. The default is None.
        t_max : float, optional
            Right boundary of the time axis. If None is given, the current
            value of self.t_scale[1] will be used. Else, the value of
            self.t_scale[1] will be replaced. The default is None.
        t_points : int, optional
            Number of time points. If None is given, the current value of
            self.t_points will be used. Else, the value of self.t_points will
            be replaced. The default is None.

        Returns
        -------
        None
            Nothing will be returned.

        """
        if self.time_axis is None and (t_min is None or t_max is None):
            raise ValueError(
                "t_min and t_max are required when no time_axis is available."
            )

        if t_min is not None:
            self.t_scale[0] = t_min
        if t_max is not None:
            self.t_scale[1] = t_max
        if t_points is not None:
            self.t_points = t_points

        if self.t_points < 2:
            raise ValueError("t_points must be at least 2.")
        if self.t_scale[0] >= self.t_scale[1]:
            raise ValueError("t_min must be smaller than t_max.")

        self.time_axis = np.linspace(self.t_scale[0], self.t_scale[1], self.t_points)


class Variation(EPR_Parameters):
    """
    A class containing variation ranges for all possible parameters.

    Attributes
    ----------
    needed_digits : int
        Number of needed digits for chromosomes.
    number_of_genes : int
        Number of varied parameters. Used for fp representation.
    variation_array : 1d-Array, np.float64
        Array with all variation ranges greater 0.
    boundaries : list
        Sequence of tuples containing upper and lower bounds for all varied
        parameters. Used for scipy.optimize.
    freq_mw : float64
        Frequency of induced microwave radiation in Hertz.
    bohr_magneton : float64
        Bohr magneton in Hertz/Tesla.

    Methods
    -------
    load(profile_name: str)
        Load Spinsystem values from a config file (profile_name.ini).

        Recommended
    save(profile_name: str)
        Save the current variation values as a config file (profile_name.ini).

        Recommended
    get_digits_for_one_par(Par, digits_per_True, one_par=True)
        Get the number of needed digits for one varied parameter.
    get_needed_digits()
        Get the total number of needed digits in binary mode.
    get_number_of_genes()
        Get the total number of genes. Is equal to the number of varied
        parameters.
    get_variation_array()
        Create an array with all used variation ranges. Only used in floating
        point representation.
    update_digits()
        Determine number of needed digits. Just calls get_needed_digits()
    get_boundaries(Sys)
        Get a sequence of tuples containing the boundaries for the varied
        parameters.

    Examples
    --------
    Initialize a new object of class <Variation>:

    >>> Var = Variation()
    >>> Var.g1
    np.array([0., 0., 0.])

    Change values:

    >>> Var.g1 = np.array([0.003, 0.004, 0.003])
    >>> Var.g1
    np.array([0.003, 0.004, 0.003])

    Create a new variation profile from an empty template and load it:

    >>> Var_profile = profiles.new_variation_profile()
    >>> Var_profile['g_1'] = [0.001, 0.007, 0.003] #  use list not array!
    >>> profiles.add_profile(Var.profile, 'variation', 'Var_prof_1')
    >>> Var_2 = Variation()
    >>> Var_2.load_profile('Var_prof_1')
    >>> Var_2.g1
    np.array([0.001, 0.007, 0.003])

    You can also save your current variation object as a new profile:

    >>> Var_3 = Variation()
    >>> Var_3.g1 = np.array([1, 2, 3]) #  either array or list
    >>> Var_3.save('Var_prof_2')
    >>> Var_4 = Variation()
    >>> Var_4.load('Var_prof_2')
    >>> Var_4.g1
    np.array([1., 2., 3.])

    """

    bohr_magneton: float = Field(default=0.0, exclude=True)
    fit_distribution: bool = False
    freq_mw: float = 0.0
    non_vars: tuple[str, ...] = Field(default=(), exclude=True)
    single_vars: tuple[str, ...] = Field(default=(), exclude=True)
    needed_digits: int = Field(default=0, exclude=True)
    number_of_genes: int = Field(default=0, exclude=True)
    variation_array: np.ndarray = Field(
        default_factory=lambda: np.array([]), exclude=True
    )
    boundaries: list[tuple[float, float]] = Field(default_factory=list, exclude=True)

    PROFILE_FIELDS: ClassVar[tuple[str, ...]] = (
        "g1",
        "g2",
        "g_tri",
        "g",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A_eseem",
        "omega_I",
        "D",
        "D_tri",
        "E",
        "E_tri",
        "beta",
        "J_0",
        "J_ex",
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
        "width_gauss",
        "T_relax_1",
        "T_relax_2",
        "T_pm",
        "decay",
        "population",
        "freq_mw",
        "amplitude",
    )

    def __init__(self, **data: Any):
        """
        Initialize object of class 'Variation' for radical pairs in EPR.

        Returns
        -------
        None.

        """
        if data:
            super().__init__(**data)
            for key in self.PROFILE_FIELDS:
                if key not in self.model_fields_set:
                    value = getattr(self, key)
                    if isinstance(value, np.ndarray):
                        setattr(self, key, np.zeros_like(value, dtype=float))
                    elif isinstance(value, (float, int)):
                        setattr(self, key, 0.0)
            self._initialize_runtime_fields()
            return
        super().__init__()
        self._initialize_variation_fields()

    def _initialize_variation_fields(self) -> None:
        """Initialize variation values and optimizer metadata."""
        for key in self.PROFILE_FIELDS:
            value = getattr(self, key)
            if isinstance(value, np.ndarray):
                setattr(self, key, np.zeros_like(value, dtype=float))
            elif isinstance(value, (float, int)):
                setattr(self, key, 0.0)

        self.bohr_magneton = constant.value("Bohr magneton in Hz/T")

        self.fit_distribution = False

        # [EXPERIMENTAL SETUP]
        self.freq_mw = 0.0  # Microwave radiation / Hz

        # [Variable Lists]
        self.non_vars = (
            "bohr_magneton",
            "needed_digits",
            "number_of_genes",
            "variation_array",
            "boundaries",
            "isotropic",
            "fit_distribution",
            "non_vars",
            "single_vars",
        )
        self.single_vars = (
            "A_eseem",
            "omega_I",
            "D",
            "D_tri",
            "E",
            "E_tri",
            "beta",
            "J_0",
            "J_ex",
            "T_relax_1",
            "T_relax_2",
            "T_pm",
            "decay",
            "freq_mw",
            "amplitude",
            "width_gauss",
        )

        self.needed_digits = 0
        self.number_of_genes = 0
        self.variation_array = np.array([], dtype=float)
        self.boundaries = []

    def _initialize_runtime_fields(self) -> None:
        """Initialize metadata when constructing from validated data."""
        self.bohr_magneton = constant.value("Bohr magneton in Hz/T")
        self.non_vars = (
            "bohr_magneton",
            "needed_digits",
            "number_of_genes",
            "variation_array",
            "boundaries",
            "isotropic",
            "fit_distribution",
            "non_vars",
            "single_vars",
        )
        self.single_vars = (
            "A_eseem",
            "omega_I",
            "D",
            "D_tri",
            "E",
            "E_tri",
            "beta",
            "J_0",
            "J_ex",
            "T_relax_1",
            "T_relax_2",
            "T_pm",
            "decay",
            "freq_mw",
            "amplitude",
            "width_gauss",
        )

    def load(self, profile_name: str, degree: bool = False) -> None:
        """
        Load a variation object from a profile.

        Load the settings from ~/.config/spinanalysis/profiles/variation/
        [profile_name].ini into the Variation object. Overwrites previous
        settings.

        Parameters
        ----------
        profile_name : str
            Name of the Variation profile which will be loaded.
        degree : bool, optional
            If True, the angle values in the profile are given in degree not
            radian. Thus, they will be converted to radian. If False, the
            angles are given in radian and will therefore not be converted. The
            default is 'False'.

        Returns
        -------
        None.

        """
        variation_profile = profiles.load_profile(profile_name, "variation")

        values = variation_profile["main"]
        for key in self.PROFILE_FIELDS:
            if key not in values:
                continue
            value = values[key]
            if isinstance(value, (list, tuple)):
                value = np.asarray(value)
            if degree and key.endswith("_frame"):
                value = np.asarray(value, dtype=float) * np.pi / 180
            setattr(self, key, value)

    def save(self, profile_name: str = "") -> None:
        """
        Save a variation object as a profile.

        Save the variation object as a variation profile using
        profile_management.add_profile(). Load the variation object using
        Variation.load_profile(<profile_name>).

        Parameters
        ----------
        profile_name : str, optional
            Name of the profile.  If no profile name is given, a default one
            will be generated by get_profile_name(). The default is ''.

        Returns
        -------
        None
            Nothing will be returned.

        """
        var_profile = profiles.new_variation_profile()

        for key in var_profile["main"]:
            value = getattr(self, key)
            var_profile["main"][key] = (
                value.tolist() if isinstance(value, np.ndarray) else value
            )

        profiles.add_profile(var_profile, "variation", pname=profile_name)

    def get_needed_digits(self) -> None:
        """
        Get sum of needed digits for all varied parameters.

        Returns
        -------
        None.

        """
        self.get_number_of_genes()
        self.needed_digits = self.number_of_genes * 12

        return None

    def get_number_of_genes(self) -> None:
        """Determine number of parameters which get varied."""
        self.number_of_genes = 0
        for key, value in self._variation_fields():
            if key in self.single_vars:
                if value > 0.0:
                    self.number_of_genes += 1
            else:
                self.number_of_genes += int(np.count_nonzero(np.asarray(value) > 0.0))

        return None

    def get_variation_array(self) -> None:
        """Put all variation ranges in one 1d-Array."""
        self.get_number_of_genes()
        self.variation_array = np.zeros(self.number_of_genes)
        i = 0
        for key, value in self._variation_fields():
            if key in self.single_vars:
                if value > 0:
                    self.variation_array[i] = value
                    i += 1
            else:
                for parameter in value:
                    if parameter > 0:
                        self.variation_array[i] = parameter
                        i += 1

        return None

    def update_digits(self) -> None:
        """Update number of needed_digits."""
        self.get_needed_digits()

        return None

    def get_boundaries(self, Sys: object) -> None:
        """
        Create a sequence of pairs with all bounds for the varied parameters.

        Used for the scipy optimization routines.

        Parameters
        ----------
        Sys : object
            Spinsystem object.

        Returns
        -------
        None
            Nothing will be returned.

        """
        self.boundaries = []

        for key, value in self._variation_fields():
            if key in self.single_vars:
                if value > 0:
                    mid = getattr(Sys, key)
                    var = value
                    lb = mid - var
                    ub = mid + var

                    if lb > ub:
                        lb = mid + var
                        ub = mid - var

                    bounds = (lb, ub)
                    self.boundaries.append(bounds)
            else:
                for i, parameter in enumerate(value):
                    if parameter > 0:
                        mid = getattr(Sys, key)[i]
                        var = parameter
                        lb = mid - var
                        ub = mid + var

                        bounds = (lb, ub)
                        self.boundaries.append(bounds)

        if self.fit_distribution:
            if Sys.distribution is None:
                raise ValueError(
                    "Sys.distribution is required when fit_distribution is enabled."
                )
            for i in range(Sys.distribution_order):
                b_int = (0, 1)
                b_pos = (Sys.distribution[0].min(), Sys.distribution[0].max())
                b_sigma = (0.01, 0.7)
                self.boundaries.append(b_int)
                self.boundaries.append(b_sigma)
                self.boundaries.append(b_pos)

        return None

    def _variation_fields(self):
        """Return variation fields in their stable declaration order."""
        excluded = set(self.non_vars)
        return (
            (key, getattr(self, key))
            for key in self.PROFILE_FIELDS
            if key not in excluded
        )


class SimulationOptions(_MutableModel):
    """

    A class containing all simulation options.

    Attributes
    ----------
    routine: str
        Name of the simulation routine which will be used by spinanalysis() and
        spinanalysis_optimize().
    knots: int
        Number of knots used for spherical grid. The default is 20.
    grid_points: int
        Deprecated alias for ``knots``. It remains supported for compatibility
        but will be removed in a future release.
    space: str
        Name of the mathematical space used for some calculations.
    pop_evolution : boolean
        If set to True, the population evolution in calculated using teacups.
    eigval_mode : boolean
        If set to True, only the eigenvalues of the system are calculated using
        teacups.
    force_cpu : boolean
        If True, the simulation will be executed on the CPU, even if GPU is
        available. Default is False.
    regularization_mode : int
        Choose the regularization matrix used for the Tikhonov-Regularization.
        0 : Unitary matrix
        1 : First order derivative matrix
        2 : Second order derivative matrix (default)

    Methods
    -------
    load(profile_name: str)
        Load SimulationOptions values from a config file (profile_name.ini).

        Recommended
    save(profile_name: str)
        Save the current simulation options as a config file
        (profile_name.ini).

    Examples
    --------
    Initialize an object of class <SimulationOptions>:

    >>> SimOpt = SimulationOptions()
    >>> SimOpt.grid_points
    500

    Change values:

    >>> SimOpt.grid_points = 1000
    >>> SimOpt.grid_points
    1000

    Save your current values as a new profile:

    >>> SimOpt.save('SimOpt_prof_1')
    >>> SimOpt_2 = SimulationOptions()
    >>> SimOpt_2.grid_points
    500
    >>> SimOpt.load('SimOpt_prof_1')
    >>> SimOpt_2.grid_points
    1000

    You can also create a simulation options profile from an empty template:

    >>> simopt_prof = profiles.new_simulation_profile()
    >>> simopt_prof['static_radpair']['grid_points'] = 1100
    >>> profiles.add_profile(simopt_prof, 'simulation', 'SimOpt_prof_2')
    >>> SimOpt_3 = SimulationOptions()
    >>> SimOpt_3.grid_points
    500
    >>> SimOpt.load('SimOpt_prof_2')
    >>> SimOpt_3.grid_points
    1100


    """

    routine: str = ""
    cpu_cores: int = 0
    knots: int = 20
    grid_points: int = 20
    refinement: int = 1
    space: str = "hilbert"
    pop_evolution: bool = True
    eigval_mode: bool = False
    min_r: float = 10.0
    max_r: float = 50.0
    r_points: int = 401
    fast_mode: bool = False
    GCV: bool = False
    force_cpu: bool = False
    regularization_mode: int = 2

    @field_validator(
        "cpu_cores", "grid_points", "refinement", "r_points", mode="before"
    )
    @classmethod
    def validate_integer_options(cls, value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("option must be an integer")
        return int(value)

    @field_validator("grid_points", mode="before")
    @classmethod
    def validate_grid_points(cls, value: object) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError("grid_points must be a positive integer")
        return int(value)

    @field_validator("knots", mode="before")
    @classmethod
    def validate_knots(cls, value: object) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError("knots must be a positive integer")
        return int(value)

    @field_validator(
        "eigval_mode", "pop_evolution", "fast_mode", "GCV", "force_cpu", mode="before"
    )
    @classmethod
    def validate_boolean_options(cls, value: object) -> bool:
        if value is None:
            return value
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("option must be a boolean")
        return bool(value)

    def __setattr__(self, name: str, value: object) -> None:
        if name == "grid_points":
            warnings.warn(
                "SimulationOptions.grid_points is deprecated; use knots instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        super().__setattr__(name, value)

    def __init__(self, **data: Any):
        """
        Initialize object of class 'Simulation_Options' used for simulations.

        Returns
        -------
        None.

        """
        if "grid_points" in data:
            warnings.warn(
                "SimulationOptions.grid_points is deprecated; use knots instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if data:
            super().__init__(**data)
            return
        super().__init__()

        # [MAIN]
        self.routine = ""
        self.cpu_cores = 0

        # [STATIC_RADICAL_PAIR]
        self.knots = 20
        object.__setattr__(self, "grid_points", 20)
        self.refinement = 1

        # [TEACUPS]
        self.space = "hilbert"
        self.pop_evolution = True
        self.eigval_mode = False

        # [OPOSSUM]

        # [DIDELPHIS]
        self.min_r = 10
        self.max_r = 50
        self.r_points = 401
        self.fast_mode = False
        self.GCV = False
        self.force_cpu = False
        self.regularization_mode = 2

        # [DIDELPHIS_MGF]

    def load(self, profile_name: str) -> None:
        """
        Load simulation options from a profile.

        Load the settings from ~/.config/spinanalysis/profiles/simulation/
        [profile_name].ini into the SimulationOptions object. Overwrites
        previous settings.

        Parameters
        ----------
        profile_name : str
            Name of the simulation profile which will be loaded.

        Returns
        -------
        None.

        """
        simulation_profile = profiles.load_profile(profile_name, "simulation")

        self.routine = simulation_profile["main"]["routine"].lower()
        self.cpu_cores = simulation_profile["main"]["cpu_cores"]
        routine_profile = simulation_profile.get(self.routine, {})
        for key, value in routine_profile.items():
            if key == "grid_points":
                warnings.warn(
                    "SimulationOptions.grid_points is deprecated; use knots instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                object.__setattr__(self, "grid_points", value)
                continue
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self, profile_name: str = "") -> None:
        """
        Save the simulation optionas as a profile.

        Save the simulation options as a simulation options profile using
        profile_management.add_profile(). Load the simulation options using
        SimulationOptions.load_profile(<profile_name>).

        Parameters
        ----------
        profile_name : str, optional
            Name of the profile.  If no profile name is given, a default one
            will be generated by get_profile_name(). The default is ''.

        Returns
        -------
        None
            Nothing will be returned.

        """
        simopt_profile = profiles.new_simulation_profile()

        for section in [
            "main",
            "static_radpair",
            "teacups",
            "opossum",
            "didelphis",
            "didelphis_tikhonov",
        ]:
            for key in simopt_profile[section]:
                if hasattr(self, key):
                    value = getattr(self, key)
                    simopt_profile[section][key] = (
                        value.tolist() if isinstance(value, np.ndarray) else value
                    )

        profiles.add_profile(simopt_profile, "simulation", pname=profile_name)


class FittingOptions(_MutableModel):
    """

    A class containing all optimization options.

    Attributes
    ----------
    routine : str
        Name of the optimization routine which will be used by
        spinanalysis_optimize().
    method : str
        Name of the optimization method used in the scipy.optimize routines.
    x0 : numpy.array, np.float64
        Array containing the initial guess for the optimization routine for
        the parameters which will be varied.
    cpu_cores : int
        Number of cores used for the optimization.
    gui : bool
        Set to True if in GUI mode (PySpin). The default is False.
    window : object
        Plot canvas. Only needed in GUI mode.

    Methods
    -------
    load(profile_name: str)
        Load FittingOptions values from a config file (profile_name.ini).

        Recommended
    save(profile_name: str)
        Save the current fitting options as a config file (profile_name.ini).

        Recommended

    Examples
    --------
     Initialize an object of class <FittingOptions>:

        >>> FitOpt = FittingOptions()
        >>> FitOpt.GAVaPS
        True

        Change values:

        >>> FitOpt.GAVaPS = False
        >>> FitOpt.GAVaPS
        False

        Save your current values as a new profile:

        >>> FitOpt.save_simulationoptions('FitOpt_prof_1')
        >>> FitOpt_2 = FittingOptions()
        >>> FitOpt_2.GAVaPS
        True
        >>> FitOpt.load_profile('FitOpt_prof_1')
        >>> FitOpt_2.GAVaPS
        False

        You can also create a fitting options profile from an empty template:

        >>> fitopt_prof = profiles.new_optimization_profile()
        >>> fitopt_prof['genetic']['GAVaPS'] = False
        >>> profiles.add_profile(fitopt_prof, 'simulation', 'FitOpt_prof_2')
        >>> FitOpt_3 = FittingOptions()
        >>> FitOpt_3.GAVaPS
        True
        >>> FitOpt.load_profile('FitOpt_prof_2')
        >>> FitOpt_3.GAVaPS
        False

    """

    routine: str | None = None
    method: str | None = None
    x0: np.ndarray | None = None
    cpu_cores: int = 0
    gui: bool = False
    window: Any = None
    GAVaPS: bool = True
    representation: str | None = None
    lifetime_mode: str | None = None
    crossover_type: str | None = None
    mutation_type: str | None = None
    min_lifetime: int | None = None
    max_lifetime: int | None = None
    reproduction_ratio: float | None = None
    p_c: float | None = None
    p_m: float | None = None
    pop_size: int | None = None
    min_pop_size: int | None = None
    max_pop_size: int | None = None
    convergence: float | None = None
    peak_prominence: float | None = None
    error_weight: np.ndarray | None = None
    max_generation: int | None = None
    show_status: bool | None = None
    maxiter: int | None = None
    maxiter_minimizer: int | None = None
    initial_temp: float | None = None
    restart_temp_ratio: float | None = None
    visit: float | None = None
    accept: float | None = None
    maxfun: int | None = None
    no_local_search: bool | None = None
    n: int | None = None
    iters: int | None = None
    maxfev: int | None = None
    f_tol: float | None = None
    maxev: int | None = None
    maxtime: float | None = None
    minimize_every_iter: bool | None = None
    local_iter: int | None = None
    sampling_method: str | None = None
    strategy: str | None = None
    popsize: int | None = None
    tol: float | None = None
    mutation: Any = None
    recombination: float | None = None
    seed: int | None = None
    disp: bool | None = None
    polish: bool | None = None
    init: Any = None
    atol: float | None = None
    updating: str | None = None
    T: float | None = None
    niter: int | None = None
    stepsize: float | None = None
    interval: int | None = None
    niter_success: int | None = None
    target_accept_rate: float | None = None
    stepwise_factor: float | None = None
    ftol: float | None = None
    xtol: float | None = None
    gtol: float | None = None
    loss: str | None = None
    f_scale: float | None = None
    max_nfev: int | None = None
    tr_solver: str | None = None
    verbose: int | None = None

    @field_validator(
        "cpu_cores",
        "maxiter",
        "maxiter_minimizer",
        "maxfun",
        "n",
        "iters",
        "maxfev",
        "maxev",
        "max_nfev",
        mode="before",
    )
    @classmethod
    def validate_integer_options(cls, value: object) -> object:
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
        ):
            raise ValueError("option must be an integer")
        return value

    @field_validator(
        "gui",
        "GAVaPS",
        "show_status",
        "no_local_search",
        "disp",
        "polish",
        "minimize_every_iter",
        mode="before",
    )
    @classmethod
    def validate_boolean_options(cls, value: object) -> bool:
        if value is None:
            return value
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError("option must be a boolean")
        return bool(value)

    def __init__(self, **data: Any):
        """
        Initialize object of class 'FittingOptions' used for optimization.

        Returns
        -------
        None.

        """
        if data:
            super().__init__(**data)
            return
        super().__init__()

        # [MAIN]
        self.routine = None
        self.method = None
        self.x0 = None
        self.cpu_cores = 0
        self.gui = False
        self.window = None

        # [GENETIC]
        self.GAVaPS = True
        self.representation = None
        self.lifetime_mode = None
        self.crossover_type = None
        self.mutation_type = None
        self.min_lifetime = None
        self.max_lifetime = None
        self.reproduction_ratio = None
        self.p_c = None
        self.p_m = None
        self.pop_size = None
        self.min_pop_size = None
        self.max_pop_size = None
        self.convergence = None
        self.peak_prominence = None
        self.error_weight = None
        self.max_generation = None
        self.show_status = None

        # [MINIMIZE]
        self.maxiter = None

        # [DUAL_ANNEALING]
        self.maxiter = None
        self.maxiter_minimizer = None
        self.initial_temp = None
        self.restart_temp_ratio = None
        self.visit = None
        self.accept = None
        self.maxfun = None
        self.no_local_search = None

        # [SHGO]
        self.n = None
        self.iters = None
        self.maxiter_minimizer = None
        self.maxfev = None
        self.f_tol = None
        self.maxiter = None
        self.maxev = None
        self.maxtime = None
        self.minimize_every_iter = None
        self.local_iter = None
        self.sampling_method = None

        # [DIFFERENTIAL_EVOLUTION]
        self.strategy = None
        self.maxiter = None
        self.popsize = None
        self.tol = None
        self.mutation = None
        self.recombination = None
        self.seed = None
        self.disp = None
        self.polish = None
        self.init = None
        self.atol = None
        self.updating = None

        # [BASINHOPPING]
        self.T = None
        self.niter = None
        self.stepsize = None
        self.maxiter_minimizer = None
        self.interval = None
        self.disp = None
        self.niter_success = None
        self.seed = None
        self.target_accept_rate = None
        self.stepwise_factor = None

        # [LEAST_SQUARES]
        self.ftol = None
        self.xtol = None
        self.gtol = None
        self.loss = None
        self.f_scale = None
        self.max_nfev = None
        self.tr_solver = None
        self.verbose = None

    def load(self, profile_name: str) -> None:
        """
        Load fitting options from a profile.

        Load the settings from ~/.config/spinanalysis/profiles/optimization/
        [profile_name].ini into the FittingOptions object. Overwrites previous
        settings. Only loads the section given in ['main']['routine'].

        Parameters
        ----------
        profile_name : str
            Name of the optimization profile which will be loaded.

        Returns
        -------
        None.

        """
        fitting_profile = profiles.load_profile(profile_name, "optimization")

        self.routine = fitting_profile["main"]["routine"]
        self.cpu_cores = fitting_profile["main"]["cpu_cores"]
        for key, value in fitting_profile.get(self.routine, {}).items():
            if key == "ftol" and not hasattr(self, key):
                key = "f_tol"
            if hasattr(self, key):
                setattr(self, key, value)

    def save(self, profile_name: str = "") -> None:
        """
        Save the fitting options as a profile.

        Save the fitting options as a fitting options profile using
        profile_management.add_profile(). Load the fitting options using
        FittingOptions.load_profile(<profile_name>).

        Parameters
        ----------
        profile_name : str, optional
            Name of the profile.  If no profile name is given, a default one
            will be generated by get_profile_name(). The default is ''.

        Returns
        -------
        None
            Nothing will be returned.

        """
        fitopt_profile = profiles.new_optimization_profile()

        for section in [
            "main",
            "genetic",
            "minimize",
            "dual_annealing",
            "shgo",
            "differential_evolution",
            "basinhopping",
            "least_squares",
        ]:
            for key in fitopt_profile[section]:
                if key == "ftol" and not hasattr(self, key):
                    value = getattr(self, "f_tol", None)
                elif hasattr(self, key):
                    value = getattr(self, key)
                else:
                    continue
                fitopt_profile[section][key] = (
                    value.tolist() if isinstance(value, np.ndarray) else value
                )

        profiles.add_profile(fitopt_profile, "optimization", pname=profile_name)
