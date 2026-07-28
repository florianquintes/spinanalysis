#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
© M. Sc. Florian Quintes, 2021-2022

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import numpy as np
import scipy.constants as constant
from spinanalysis import profiles


class EPR_Parameters:
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

    def __init__(self) -> None:
        """
        Initialize object of class 'Spinsystem' for radical pair simulation.

        Returns
        -------
        None.

        """
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

    def __init__(self) -> None:
        """
        Initialize object of class 'Spinsystem' for radical pair simulation.

        Returns
        -------
        None.

        """
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

    def _get_g_iso(self) -> None:
        """Get g1_iso and g2_iso."""
        self.g1_iso = self.g1.sum() / 3
        self.g2_iso = self.g2.sum() / 3

    def load(self, profile_name: str, degree: bool = False) -> None:
        """
        Load a spinsystem from a profile.

        Load the settings from [sys.prefix]/easypairspin/profiles/spinsystem/
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

        for key in vars(self):
            if key not in ("g1_iso", "g2_iso"):
                try:
                    vars(self)[key] = spinsystem_profile["main"][key]
                except KeyError:
                    pass

                try:
                    if len(vars(self)[key]) == 3:
                        vars(self)[key] = np.array(vars(self)[key])
                except TypeError:
                    pass

                if key in ("acceptor_list", "donor_list"):
                    vars(self)[key] = np.array(vars(self)[key])

                if key == "distribution":
                    if vars(self)[key] in ("None", None):
                        vars(self)[key] = None
                    else:
                        vars(self)[key] = np.array(vars(self)[key])
                        size = vars(self)[key].size
                        vars(self)[key] = vars(self)[key].reshape((2, size // 2))

                if degree:
                    if key.endswith("_frame"):
                        vars(self)[key] *= np.pi / 180

        self._get_g_iso()

    def save(self, profile_name: str = "") -> None:
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
            if isinstance(vars(self)[key], np.ndarray):
                spinsys_profile["main"][key] = list(vars(self)[key].flatten())
            else:
                spinsys_profile["main"][key] = vars(self)[key]

        profiles.add_profile(spinsys_profile, "spinsystem", pname=profile_name)


class Experimental:
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

    def __init__(
        self,
        magnetic_field: np.array = None,
        real_int: np.array = None,
        imag_int: np.array = None,
        cmplx_int: np.array = None,
        time_axis: np.array = None,
        rescale: bool = True,
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
        self.magnetic_field = magnetic_field
        if self.magnetic_field is None:
            self.B_z = np.linspace(240, 260, 100)
        else:
            self.B_z = 1 * magnetic_field  # external magnetic field / mT
        self.freq_mw = 9.7e9  # Microwave radiation / Hz
        self.B_mw = 1e-3

        self.t_scale = [0, 2e-6]
        self.t_points = 2
        self.time_axis = time_axis
        if time_axis is not None:
            self.t_scale[0] = time_axis.min()
            self.t_scale[1] = time_axis.max()
            self.t_points = time_axis.shape[0]

        self.int = None
        if cmplx_int is not None:
            self.int = cmplx_int
        else:
            if real_int is not None:
                self.int = np.zeros(real_int.shape, dtype="complex128")
                self.int.real = real_int
            if imag_int is not None:
                if self.int is not None:
                    self.int.imag = imag_int
                else:
                    self.int = np.zeros(imag_int.shape, dtype="complex128")
                    self.int.imag = real_int

        if self.int is not None and rescale:
            self.int /= np.abs(self.int).max()
            self.spec_sim = np.zeros(
                self.int.shape, dtype="complex128"
            )  # simulated spectra

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
        if t_min is not None:
            self.t_scale[0] = t_min
        if t_max is not None:
            self.t_scale[1] = t_max
        if t_points is not None:
            self.t_points = t_points

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

    def __init__(self):
        """
        Initialize object of class 'Variation' for radical pairs in EPR.

        Returns
        -------
        None.

        """
        super().__init__()
        for key in vars(self):
            if isinstance(vars(self)[key], float):
                vars(self)[key] = 0.0
            elif isinstance(vars(self)[key], int):
                vars(self)[key] = 0
            else:
                for i in range(len(vars(self)[key])):
                    vars(self)[key][i] = 0.0

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

    def load(self, profile_name: str, degree: bool = False) -> None:
        """
        Laod a variation object from a profile.

        Load the settings from [sys.prefix]/easypairspin/profiles/variation/
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

        for key in vars(self):
            if key not in ("g1_iso", "g2_iso", "bohr_magneton"):
                try:
                    vars(self)[key] = variation_profile["main"][key]
                except KeyError:
                    pass

                try:
                    if len(vars(self)[key]) == 3:
                        vars(self)[key] = np.array(vars(self)[key])
                except TypeError:
                    pass

            if degree:
                if key.endswith("_frame"):
                    vars(self)[key] *= np.pi / 180

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
            if isinstance(vars(self)[key], np.ndarray):
                var_profile["main"][key] = list(vars(self)[key])
            else:
                var_profile["main"][key] = vars(self)[key]

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
        for key in vars(self):
            if key in self.non_vars:
                pass
            elif key in self.single_vars:
                if vars(self)[key] > 0.0:
                    self.number_of_genes += 1
            else:
                for parameter in vars(self)[key]:
                    if parameter > 0.0:
                        self.number_of_genes += 1

        return None

    def get_variation_array(self) -> None:
        """Put all variation ranges in one 1d-Array."""
        self.get_number_of_genes()
        self.variation_array = np.zeros(self.number_of_genes)
        i = 0
        for key in vars(self):
            if key in self.non_vars:
                pass
            elif key in self.single_vars:
                if vars(self)[key] > 0:
                    self.variation_array[i] = vars(self)[key]
                    i += 1
            else:
                for parameter in vars(self)[key]:
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

        for key in vars(self):
            if key in self.non_vars:
                pass
            elif key in self.single_vars:
                if vars(self)[key] > 0:
                    mid = vars(Sys)[key]
                    var = vars(self)[key]
                    lb = mid - var
                    ub = mid + var

                    if lb > ub:
                        lb = mid + var
                        ub = mid - var

                    bounds = (lb, ub)
                    self.boundaries.append(bounds)
            else:
                for i, parameter in enumerate(vars(self)[key]):
                    if parameter > 0:
                        mid = vars(Sys)[key][i]
                        var = parameter
                        lb = mid - var
                        ub = mid + var

                        bounds = (lb, ub)
                        self.boundaries.append(bounds)

        if self.fit_distribution:
            for i in range(Sys.distribution_order):
                b_int = (0, 1)
                b_pos = (Sys.distribution[0].min(), Sys.distribution[0].max())
                b_sigma = (0.01, 0.7)
                self.boundaries.append(b_int)
                self.boundaries.append(b_sigma)
                self.boundaries.append(b_pos)

        return None


class SimulationOptions:
    """
    A class containing all simulation options.

    Attributes
    ----------
    routine: str
        Name of the simulation routine which will be used by easypairspin() and
        easypairspin_optimize().
    grid_points: int
        Number of points used for spherical grid.
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

    def __init__(self):
        """
        Initialize object of class 'Simulation_Options' used for simulations.

        Returns
        -------
        None.

        """
        # [MAIN]
        self.routine = ""
        self.cpu_cores = 0

        # [STATIC_RADICAL_PAIR]
        self.grid_points = 500
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

        Load the settings from [sys.prefix]/easypairspin/profiles/simulation/
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

        self.routine = simulation_profile["main"]["routine"]
        self.cpu_cores = simulation_profile["main"]["cpu_cores"]
        for key in simulation_profile[self.routine].keys():
            vars(self)[key] = simulation_profile[self.routine][key]

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

        for section in ["main", "static_radpair", "teacups", "opossum"]:
            for key in simopt_profile[section]:
                if isinstance(vars(self)[key], np.ndarray):
                    simopt_profile[section][key] = list(vars(self)[key])
                else:
                    simopt_profile[section][key] = vars(self)[key]

        profiles.add_profile(simopt_profile, "simulation", pname=profile_name)


class FittingOptions:
    """
    A class containing all optimization options.

    Attributes
    ----------
    routine : str
        Name of the optimization routine which will be used by
        easypairspin_optimize().
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

    def __init__(self):
        """
        Initialize object of class 'FittingOptions' used for optimization.

        Returns
        -------
        None.

        """
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

        Load the settings from [sys.prefix]/easypairspin/profiles/optimization/
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
        for key in fitting_profile[self.routine].keys():
            vars(self)[key] = fitting_profile[self.routine][key]

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
                if isinstance(vars(self)[key], np.ndarray):
                    fitopt_profile[section][key] = list(vars(self)[key])
                else:
                    fitopt_profile[section][key] = vars(self)[key]

        profiles.add_profile(fitopt_profile, "optimization", pname=profile_name)
