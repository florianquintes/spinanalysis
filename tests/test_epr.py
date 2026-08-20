"""Tests for the Pydantic EPR data models in :mod:`spinanalysis.epr`.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
import scipy.constants as constant

from spinanalysis.epr import Experimental, Spinsystem, Variation
from spinanalysis.epr import FittingOptions, SimulationOptions


class TestExperimental:
    def test_default_values(self):
        experiment = Experimental()

        assert experiment.B_z.shape == (100,)
        assert experiment.B_z.dtype == np.float64
        assert np.array_equal(experiment.magnetic_field, experiment.B_z)
        assert experiment.magnetic_field.dtype == np.float64
        assert experiment.freq_mw == pytest.approx(9.7e9)
        assert experiment.B_mw == pytest.approx(1e-3)
        assert experiment.int is None
        assert np.array_equal(experiment.spec_sim, np.zeros(100))
        assert experiment.spec_sim.dtype == np.complex128

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"real_int": [1, 2, 3]}, [1 + 0j, 2 + 0j, 3 + 0j]),
            ({"imag_int": [1, 2, 3]}, [0 + 1j, 0 + 2j, 0 + 3j]),
            (
                {"real_int": [1, 2], "imag_int": [3, 4]},
                [1 + 3j, 2 + 4j],
            ),
            (
                {"cmplx_int": [1 + 2j, 3 + 4j]},
                [1 / np.sqrt(5) + 2j / np.sqrt(5), 3 / np.sqrt(5) + 4j / np.sqrt(5)],
            ),
            ({"real_int": [-1, -2, -10]}, [-0.1, -0.2, -1.0]),
            ({"imag_int": [-1, -2, -10]}, [-0.1j, -0.2j, -1.0j]),
            (
                {"real_int": [-2, 4], "imag_int": [6, -20]},
                [-0.1 + 0.3j, 0.2 - 1.0j],
            ),
            (
                {"cmplx_int": [-1 - 2j, 3 + 4j, -15 + 8j]},
                [(-1 - 2j) / 17, (3 + 4j) / 17, (-15 + 8j) / 17],
            ),
        ],
    )
    def test_intensity_input_combinations(self, kwargs, expected):
        experiment = Experimental(magnetic_field=np.arange(len(expected)), **kwargs)
        expected = np.asarray(expected, dtype=np.complex128)
        expected /= np.abs(expected).max()

        assert np.allclose(experiment.int, expected)
        assert experiment.int.dtype == np.complex128
        assert experiment.spec_sim.shape == experiment.int.shape
        assert experiment.spec_sim.dtype == np.complex128

    @pytest.mark.parametrize("field", [[1, 2, 3], (1, 2, 3)])
    def test_field_input_sequences_become_float64_arrays(self, field):
        experiment = Experimental(magnetic_field=field)

        assert isinstance(experiment.B_z, np.ndarray)
        assert isinstance(experiment.magnetic_field, np.ndarray)
        assert experiment.B_z.dtype == np.float64
        assert experiment.magnetic_field.dtype == np.float64

    @pytest.mark.parametrize("name", ["real_int", "imag_int", "time_axis"])
    @pytest.mark.parametrize("value", [[1, 2, 3], (1, 2, 3)])
    def test_real_input_sequences_become_float64_arrays(self, name, value):
        kwargs = {name: value}
        if name != "time_axis":
            kwargs["magnetic_field"] = [1, 2, 3]
        experiment = Experimental(**kwargs)

        result = experiment.time_axis if name == "time_axis" else experiment.int
        assert isinstance(result, np.ndarray)
        assert result.dtype == (np.float64 if name == "time_axis" else np.complex128)

    @pytest.mark.parametrize("value", [[1 + 2j, 3 + 4j], (1 + 2j, 3 + 4j)])
    def test_complex_input_sequences_become_complex128_arrays(self, value):
        experiment = Experimental(magnetic_field=[1, 2], cmplx_int=value)

        assert isinstance(experiment.int, np.ndarray)
        assert experiment.int.dtype == np.complex128

    @pytest.mark.parametrize("name", ["real_int", "imag_int", "time_axis"])
    def test_real_inputs_reject_complex_values(self, name):
        with pytest.raises(ValidationError):
            Experimental(**{name: [1 + 1j, 2 + 0j]})

    def test_cmplx_int_rejects_real_values(self):
        with pytest.raises(ValidationError):
            Experimental(cmplx_int=[1, 2])

    def test_cmplx_int_cannot_be_combined_with_component_inputs(self):
        with pytest.raises(ValidationError):
            Experimental(cmplx_int=np.array([1 + 1j]), real_int=[1])
        with pytest.raises(ValidationError):
            Experimental(cmplx_int=np.array([1 + 1j]), imag_int=[1])

    def test_component_intensities_require_matching_shapes(self):
        with pytest.raises(ValidationError):
            Experimental(real_int=[1, 2], imag_int=[1])

    def test_zero_intensity_is_not_changed_by_rescaling(self):
        zeros = np.zeros(3, dtype=np.complex128)
        experiment = Experimental(magnetic_field=[1, 2, 3], cmplx_int=zeros)

        assert np.array_equal(experiment.int, zeros)
        assert np.array_equal(experiment.spec_sim, zeros)

    def test_time_axis_is_converted_and_configured(self):
        experiment = Experimental(time_axis=[0.0, 0.5, 1.0])

        assert isinstance(experiment.time_axis, np.ndarray)
        assert np.array_equal(experiment.time_axis, [0.0, 0.5, 1.0])
        assert experiment.t_scale == [0.0, 1.0]
        assert experiment.t_points == 3

    @pytest.mark.parametrize("name", ["magnetic_field", "time_axis"])
    def test_axes_reject_complex_values(self, name):
        with pytest.raises(ValidationError):
            Experimental(**{name: [1 + 1j, 2 + 0j]})

    def test_get_linear_time_axis_defaults(self):
        experiment = Experimental()

        with pytest.raises(ValueError):
            experiment.get_linear_time_axis()

    @pytest.mark.parametrize(
        "kwargs",
        [{"t_min": 1.0}, {"t_max": 2.0}, {"t_min": None, "t_max": None}],
    )
    def test_get_linear_time_axis_requires_both_bounds_without_time_axis(self, kwargs):
        with pytest.raises(ValueError):
            Experimental().get_linear_time_axis(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [{"t_min": 1.0}, {"t_max": 2.0}, {}],
    )
    def test_get_linear_time_axis_uses_existing_time_axis(self, kwargs):
        experiment = Experimental(time_axis=[0.0, 1.0, 2.0])
        experiment.get_linear_time_axis(**kwargs)

        assert isinstance(experiment.time_axis, np.ndarray)
        assert experiment.time_axis.dtype == np.float64

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t_min": 1.0, "t_max": 2.0},
            {"t_min": 1.0, "t_max": 2.0, "t_points": 5},
        ],
    )
    def test_get_linear_time_axis_accepts_both_bounds_without_time_axis(self, kwargs):
        experiment = Experimental()
        experiment.get_linear_time_axis(**kwargs)

        assert isinstance(experiment.time_axis, np.ndarray)
        assert experiment.time_axis.dtype == np.float64

    def test_get_linear_time_axis_accepts_valid_overrides(self):
        experiment = Experimental(time_axis=[0.0, 1.0])
        experiment.get_linear_time_axis(t_min=1.0, t_max=2.0, t_points=5)

        assert np.array_equal(experiment.time_axis, np.linspace(1.0, 2.0, 5))
        assert experiment.time_axis.dtype == np.float64
        assert experiment.t_scale == [1.0, 2.0]
        assert experiment.t_points == 5

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"t_points": 1},
            {"t_points": 0},
            {"t_min": 2.0, "t_max": 1.0},
            {"t_min": 1.0, "t_max": 1.0},
        ],
    )
    def test_get_linear_time_axis_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            Experimental().get_linear_time_axis(**kwargs)


class TestSimulationOptions:
    def test_default_values(self):
        options = SimulationOptions()

        assert options.routine == ""
        assert options.cpu_cores == 0
        assert options.knots == 20
        assert options.refinement == 1
        assert options.space == "hilbert"
        assert options.pop_evolution is True
        assert options.eigval_mode is False
        assert options.r_points == 401
        assert options.regularization_mode == 2

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("cpu_cores", "one"),
            ("knots", "many"),
            ("space", 2),
            ("eigval_mode", "yes"),
            ("r_points", 401.5),
        ],
    )
    def test_invalid_types_are_rejected(self, name, value):
        with pytest.raises(ValidationError):
            SimulationOptions(**{name: value})

    def test_assignment_is_validated(self):
        options = SimulationOptions()
        options.knots = 1000
        assert options.knots == 1000

        with pytest.raises(ValidationError):
            options.knots = "many"

    def test_grid_points_warns_and_remains_supported(self):
        options = SimulationOptions()

        with pytest.deprecated_call(match="grid_points.*knots"):
            options.grid_points = 1000

        assert options.grid_points == 1000
        assert options.knots == 20

        with pytest.deprecated_call(match="grid_points.*knots"):
            legacy = SimulationOptions(grid_points=750)

        assert legacy.grid_points == 750
        assert legacy.knots == 20

    def test_profile_round_trip(self, tmp_path, monkeypatch):
        _install_option_profiles(tmp_path, monkeypatch, "simulation")
        options = SimulationOptions(routine="static_radpair", cpu_cores=2, knots=25)
        options.save("simulation_options")

        loaded = SimulationOptions()
        loaded.load("simulation_options")

        assert loaded.routine == "static_radpair"
        assert loaded.cpu_cores == 2
        assert loaded.knots == 25

    @pytest.mark.parametrize("name", ["knots", "grid_points"])
    @pytest.mark.parametrize("value", [0, -1])
    def test_grid_values_must_be_positive(self, name, value):
        if name == "grid_points":
            with pytest.deprecated_call(match="grid_points.*knots"):
                with pytest.raises(ValidationError):
                    SimulationOptions(**{name: value})
        else:
            with pytest.raises(ValidationError):
                SimulationOptions(**{name: value})

        options = SimulationOptions()
        if name == "grid_points":
            with pytest.deprecated_call(match="grid_points.*knots"):
                with pytest.raises(ValidationError):
                    setattr(options, name, value)
        else:
            with pytest.raises(ValidationError):
                setattr(options, name, value)


class TestFittingOptions:
    def test_default_values(self):
        options = FittingOptions()

        assert options.routine is None
        assert options.method is None
        assert options.cpu_cores == 0
        assert options.gui is False
        assert options.GAVaPS is True

    @pytest.mark.parametrize(
        ("name", "value"),
        [
            ("cpu_cores", "one"),
            ("gui", "yes"),
            ("maxiter", "many"),
            ("p_c", "half"),
            ("error_weight", 1.0),
        ],
    )
    def test_invalid_types_are_rejected(self, name, value):
        with pytest.raises(ValidationError):
            FittingOptions(**{name: value})

    def test_assignment_is_validated(self):
        options = FittingOptions()
        options.maxiter = 100
        assert options.maxiter == 100

        with pytest.raises(ValidationError):
            options.maxiter = "many"

    def test_profile_round_trip(self, tmp_path, monkeypatch):
        profile_root = _install_option_profiles(tmp_path, monkeypatch, "optimization")
        options = FittingOptions(routine="minimize", cpu_cores=2, maxiter=250)
        options.save("fitting_options")

        loaded = FittingOptions()
        loaded.load("fitting_options")

        assert loaded.routine == "minimize"
        assert loaded.cpu_cores == 2
        assert loaded.maxiter == 250
        assert (profile_root / "optimization" / "fitting_options.ini").exists()


class TestSpinsystem:
    def test_default_values(self):
        system = Spinsystem()

        assert np.array_equal(system.g1, [2.002, 2.002, 2.002])
        assert np.array_equal(system.g2, [2.004, 2.004, 2.004])
        assert np.array_equal(system.g_tri, [2.002, 2.002, 2.002])
        assert np.array_equal(system.g, [2.002, 2.002, 2.002])
        assert system.g1_iso == pytest.approx(2.002)
        assert system.g2_iso == pytest.approx(2.004)
        assert system.spin_system == "rp"
        assert system.precursor == "singlet"
        assert np.array_equal(system.acceptor_list, [1, 2, 3])
        assert np.array_equal(system.donor_list, [4, 5])
        assert system.width_gauss == pytest.approx(0.5)
        assert system.distribution is None
        assert system.distribution_order == 3

    def test_values_can_be_set(self):
        system = Spinsystem()
        values = {
            "g1": [1.0, 2.0, 3.0],
            "g2": (4.0, 5.0, 6.0),
            "spin_system": "trip",
            "precursor": "zf",
            "acceptor_list": [1, 3],
            "donor_list": (2, 4),
            "n1": 2,
            "I1": 0.5,
            "D": 10.0,
            "E": 1.0,
            "width_gauss": 0.25,
        }

        for name, value in values.items():
            setattr(system, name, value)

        assert np.array_equal(system.g1, [1.0, 2.0, 3.0])
        assert system.g1_iso == pytest.approx(2.0)
        assert np.array_equal(system.g2, [4.0, 5.0, 6.0])
        assert system.g2_iso == pytest.approx(5.0)
        assert system.spin_system == "trip"
        assert system.precursor == "zf"
        assert system.n1 == 2
        assert system.I1 == pytest.approx(0.5)
        assert system.D == pytest.approx(10.0)
        assert system.E == pytest.approx(1.0)
        assert system.width_gauss == pytest.approx(0.25)

    @pytest.mark.parametrize("name", ["n1", "n2", "n3", "n4", "n5"])
    @pytest.mark.parametrize("value", [0, 1, 5])
    def test_nuclear_counts_accept_nonnegative_integers(self, name, value):
        system = Spinsystem(**{name: value})
        assert getattr(system, name) == value

    @pytest.mark.parametrize("name", ["n1", "n2", "n3", "n4", "n5"])
    @pytest.mark.parametrize("value", [-1, 0.5, 1.2, "1", True])
    def test_nuclear_counts_reject_invalid_values(self, name, value):
        with pytest.raises(ValidationError):
            Spinsystem(**{name: value})

    @pytest.mark.parametrize("name", ["I1", "I2", "I3", "I4", "I5"])
    @pytest.mark.parametrize("value", [0, 0.5, 1.0, 2.5])
    def test_nuclear_spins_accept_nonnegative_half_integers(self, name, value):
        system = Spinsystem(**{name: value})
        assert getattr(system, name) == pytest.approx(value)

    @pytest.mark.parametrize("name", ["I1", "I2", "I3", "I4", "I5"])
    @pytest.mark.parametrize("value", [-0.5, 0.25, 1.2, "spin", True])
    def test_nuclear_spins_reject_invalid_values(self, name, value):
        with pytest.raises(ValidationError):
            Spinsystem(**{name: value})

    def test_nuclear_values_are_validated_on_assignment(self):
        system = Spinsystem()

        system.n1 = 3
        system.I1 = 1.5
        assert system.n1 == 3
        assert system.I1 == pytest.approx(1.5)

        with pytest.raises(ValidationError):
            system.n1 = -1
        with pytest.raises(ValidationError):
            system.I1 = 0.25

    @pytest.mark.parametrize(
        "attribute",
        [
            "g1",
            "g2",
            "g_tri",
            "g",
            "A1",
            "A1_frame",
            "acceptor_list",
            "donor_list",
            "population",
        ],
    )
    @pytest.mark.parametrize("value", [[1, 2, 3], (1, 2, 3)])
    def test_list_and_tuple_values_are_numpy_arrays(self, attribute, value):
        system = Spinsystem(**{attribute: value})

        assert isinstance(getattr(system, attribute), np.ndarray)

    @pytest.mark.parametrize("profile_name", ["test_Sys_radpair", "test_Sys_triplet"])
    def test_load_profile(self, tmp_path, monkeypatch, profile_name):
        self._install_profile_fixture(tmp_path, monkeypatch)
        system = Spinsystem()
        system.load(profile_name)

        if profile_name == "test_Sys_radpair":
            assert np.array_equal(system.g1, [2.002, 2.002, 2.002])
            assert np.array_equal(system.A1, [20.0, 0.0, 0.0])
            assert system.D == pytest.approx(10.0)
            assert system.n1 == 2
        else:
            assert system.spin_system == "trip"
            assert system.precursor == "zf"
            assert np.array_equal(system.g1_frame, [10.0, 20.0, 30.0])
            assert np.array_equal(system.distribution, [[0.1, 0.2], [0.3, 0.4]])
            assert system.distribution_order == 2
        assert system.I2 == pytest.approx(
            1.0 if profile_name == "test_Sys_triplet" else 1.0
        )

    def test_load_profile_with_degree_angles(self, tmp_path, monkeypatch):
        self._install_profile_fixture(tmp_path, monkeypatch)
        system = Spinsystem()
        system.load("test_Sys_radpair", degree=True)

        assert np.array_equal(system.g1_frame, [0.0, 0.0, 0.0])
        assert np.array_equal(system.D_frame, [0.0, 0.0, 0.0])

    def test_save_contains_all_profile_fields(self, tmp_path, monkeypatch):
        profile_root = self._install_profile_fixture(tmp_path, monkeypatch)

        system = Spinsystem()
        system.g1 = np.array([2.1, 2.2, 2.3])
        system.g2 = np.array([2.4, 2.5, 2.6])
        system.distribution = np.array([[0.1, 0.2], [0.3, 0.4]])
        system.save("saved_system")

        saved = profile_root / "spinsystem" / "saved_system.ini"
        assert saved.exists()

        loaded = Spinsystem()
        loaded.load("saved_system")
        for name in Spinsystem.PROFILE_FIELDS:
            expected = getattr(system, name)
            actual = getattr(loaded, name)
            if isinstance(expected, np.ndarray):
                assert np.array_equal(actual, expected), name
            else:
                assert actual == expected, name

    @pytest.mark.parametrize("init_degree", [False, True])
    @pytest.mark.parametrize("save_degree", [False, True])
    def test_frame_units_round_trip(
        self, tmp_path, monkeypatch, init_degree, save_degree
    ):
        profile_root = self._install_profile_fixture(tmp_path, monkeypatch)
        input_angles = np.array([10.0, 20.0, 30.0])
        expected_angles = input_angles * np.pi / 180 if init_degree else input_angles

        system = Spinsystem(
            degree=init_degree,
            g1_frame=input_angles,
            A1_frame=input_angles,
            D_tri_frame=input_angles,
        )
        system.save("units_system", degree=save_degree)

        loaded = Spinsystem()
        loaded.load("units_system", degree=save_degree)

        assert np.allclose(loaded.g1_frame, expected_angles)
        assert np.allclose(loaded.A1_frame, expected_angles)
        assert np.allclose(loaded.D_tri_frame, expected_angles)
        assert (profile_root / "spinsystem" / "units_system.ini").exists()

    def _install_profile_fixture(self, tmp_path, monkeypatch):
        profile_root = tmp_path / "profiles"
        profile_dir = profile_root / "spinsystem"
        profile_dir.mkdir(parents=True)
        source_dir = Path(__file__).parent / "data" / "test_profiles" / "spinsystem"
        for fixture in source_dir.glob("*.ini"):
            shutil.copy(fixture, profile_dir)
        shutil.copy(
            Path("src/spinanalysis/data/profiles/spinsystem/configspec.ini"),
            profile_dir,
        )
        monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", profile_root)
        return profile_root


class TestVariation:
    def test_default_values(self):
        var = Variation()

        assert np.array_equal(var.g1, [0.0, 0.0, 0.0])
        assert np.array_equal(var.g2, [0.0, 0.0, 0.0])
        assert np.array_equal(var.A1, [0.0, 0.0, 0.0])
        assert var.D == pytest.approx(0.0)
        assert var.E == pytest.approx(0.0)
        assert var.width_gauss == pytest.approx(0.0)
        assert var.freq_mw == pytest.approx(0.0)
        assert var.amplitude == pytest.approx(0.0)
        assert var.fit_distribution is False
        assert var.needed_digits == 0
        assert var.number_of_genes == 0
        assert np.array_equal(var.variation_array, np.array([]))
        assert var.boundaries == []
        assert var.bohr_magneton == pytest.approx(
            constant.value("Bohr magneton in Hz/T")
        )

    def test_default_data_types(self):
        var = Variation()

        assert isinstance(var.g1, np.ndarray)
        assert var.g1.dtype == np.float64
        assert isinstance(var.g2, np.ndarray)
        assert var.g2.dtype == np.float64
        assert isinstance(var.A1, np.ndarray)
        assert var.A1.dtype == np.float64
        assert isinstance(var.D, float)
        assert isinstance(var.E, float)
        assert isinstance(var.width_gauss, float)
        assert isinstance(var.freq_mw, float)
        assert isinstance(var.amplitude, float)
        assert isinstance(var.fit_distribution, bool)
        assert isinstance(var.needed_digits, int)
        assert isinstance(var.number_of_genes, int)
        assert isinstance(var.variation_array, np.ndarray)
        assert var.variation_array.dtype == np.float64
        assert isinstance(var.boundaries, list)
        assert isinstance(var.bohr_magneton, float)
        assert isinstance(var.non_vars, tuple)
        assert isinstance(var.single_vars, tuple)

    def test_values_can_be_set(self):
        var = Variation()
        var.g1 = np.array([0.001, 0.002, 0.003])
        var.D = 2.0
        var.width_gauss = 0.1
        var.fit_distribution = True

        assert np.allclose(var.g1, [0.001, 0.002, 0.003])
        assert var.D == pytest.approx(2.0)
        assert var.width_gauss == pytest.approx(0.1)
        assert var.fit_distribution is True

    @pytest.mark.parametrize(
        "attribute",
        [
            "g1",
            "g2",
            "g_tri",
            "g",
            "A1",
            "A2",
            "A3",
            "A4",
            "A5",
            "g1_frame",
            "A1_frame",
            "D_frame",
            "population",
        ],
    )
    @pytest.mark.parametrize("value", [[1, 2, 3], (1, 2, 3)])
    def test_list_and_tuple_values_are_numpy_arrays(self, attribute, value):
        var = Variation(**{attribute: value})

        assert isinstance(getattr(var, attribute), np.ndarray)
        assert getattr(var, attribute).dtype == np.float64

    def test_init_with_data_zeros_unspecified_profile_fields(self):
        var = Variation(D=2.0)

        assert var.D == pytest.approx(2.0)
        assert np.array_equal(var.g1, [0.0, 0.0, 0.0])
        assert np.array_equal(var.A1, [0.0, 0.0, 0.0])
        assert var.E == pytest.approx(0.0)
        assert var.width_gauss == pytest.approx(0.0)

    def test_assignment_is_validated(self):
        var = Variation()
        var.D = 5.0
        assert var.D == pytest.approx(5.0)

        var.g1 = [0.1, 0.2, 0.3]
        assert isinstance(var.g1, np.ndarray)
        assert var.g1.dtype == np.float64
        assert np.allclose(var.g1, [0.1, 0.2, 0.3])

    def test_get_number_of_genes_counts_single_vars(self):
        var = Variation()
        var.D = 2.0
        var.E = 0.5
        var.width_gauss = 0.1

        var.get_number_of_genes()
        assert var.number_of_genes == 3

    def test_get_number_of_genes_counts_array_elements(self):
        var = Variation()
        var.g1 = np.array([0.001, 0.0, 0.003])
        var.A1 = np.array([5.0, 0.0, 0.0])

        var.get_number_of_genes()
        assert var.number_of_genes == 3

    def test_get_number_of_genes_counts_mixed_vars(self):
        var = Variation()
        var.g1 = np.array([0.001, 0.0, 0.003])
        var.D = 2.0
        var.A1 = np.array([5.0, 0.0, 0.0])
        var.E = 0.0

        var.get_number_of_genes()
        assert var.number_of_genes == 4

    def test_get_number_of_genes_zero_when_nothing_varied(self):
        var = Variation()
        var.get_number_of_genes()
        assert var.number_of_genes == 0

    def test_get_variation_array_collects_values(self):
        var = Variation()
        var.g1 = np.array([0.001, 0.0, 0.003])
        var.D = 2.0
        var.A1 = np.array([5.0, 0.0, 0.0])

        var.get_variation_array()
        assert var.variation_array.shape == (4,)
        assert np.allclose(
            np.sort(var.variation_array), np.sort([0.001, 0.003, 2.0, 5.0])
        )
        assert var.variation_array.dtype == np.float64

    def test_get_needed_digits(self):
        var = Variation()
        var.D = 2.0
        var.E = 0.5

        var.get_needed_digits()
        assert var.number_of_genes == 2
        assert var.needed_digits == 24

    def test_update_digits(self):
        var = Variation()
        var.D = 2.0
        var.E = 0.5
        var.g1 = np.array([0.001, 0.0, 0.0])

        var.update_digits()
        assert var.number_of_genes == 3
        assert var.needed_digits == 36

    def test_get_boundaries_single_vars(self):
        var = Variation()
        var.D = 2.0
        var.E = 0.5

        sys = Spinsystem()
        sys.D = 10.0
        sys.E = 1.0

        var.get_boundaries(sys)
        assert len(var.boundaries) == 2
        assert (8.0, 12.0) in var.boundaries
        assert (0.5, 1.5) in var.boundaries

    def test_get_boundaries_array_vars(self):
        var = Variation()
        var.g1 = np.array([0.001, 0.0, 0.003])

        sys = Spinsystem()
        sys.g1 = np.array([2.002, 2.002, 2.002])

        var.get_boundaries(sys)
        assert len(var.boundaries) == 2
        assert pytest.approx(2.001) in var.boundaries[0]
        assert pytest.approx(2.003) in var.boundaries[0]
        assert pytest.approx(1.999) in var.boundaries[1]
        assert pytest.approx(2.005) in var.boundaries[1]

    def test_get_boundaries_mixed_vars(self):
        var = Variation()
        var.g1 = np.array([0.001, 0.0, 0.003])
        var.D = 2.0

        sys = Spinsystem()
        sys.g1 = np.array([2.0, 2.0, 2.0])
        sys.D = 10.0

        var.get_boundaries(sys)
        assert len(var.boundaries) == 3

    def test_get_boundaries_fit_distribution_without_distribution_raises(self):
        var = Variation()
        var.fit_distribution = True

        sys = Spinsystem()
        sys.distribution = None

        with pytest.raises(ValueError):
            var.get_boundaries(sys)

    def test_get_boundaries_fit_distribution_with_distribution(self):
        var = Variation()
        var.fit_distribution = True

        sys = Spinsystem()
        sys.distribution = np.array([[0.1, 0.5], [0.3, 0.7]])
        sys.distribution_order = 2

        var.get_boundaries(sys)
        assert len(var.boundaries) == 6

    def test_profile_round_trip(self, tmp_path, monkeypatch):
        self._install_profile_fixture(tmp_path, monkeypatch)
        var = Variation()
        var.g1 = np.array([0.001, 0.002, 0.003])
        var.D = 2.0
        var.width_gauss = 0.1
        var.save("test_round_trip")

        loaded = Variation()
        loaded.load("test_round_trip")

        assert np.allclose(loaded.g1, [0.001, 0.002, 0.003])
        assert loaded.D == pytest.approx(2.0)
        assert loaded.width_gauss == pytest.approx(0.1)

    def test_load_profile(self, tmp_path, monkeypatch):
        self._install_profile_fixture(tmp_path, monkeypatch)
        var = Variation()
        var.load("test_Var_radpair")

        assert np.allclose(var.g1, [0.001, 0.002, 0.0])
        assert np.allclose(var.g2, [0.0, 0.003, 0.0])
        assert np.allclose(var.A1, [5.0, 0.0, 0.0])
        assert var.D == pytest.approx(2.0)
        assert var.E == pytest.approx(0.5)
        assert var.width_gauss == pytest.approx(0.1)

    def test_load_profile_with_degree(self, tmp_path, monkeypatch):
        self._install_profile_fixture(tmp_path, monkeypatch)
        var = Variation()
        var.load("test_Var_radpair", degree=True)

        assert np.allclose(var.g1_frame, [0.0, 0.0, 0.0])
        assert np.allclose(var.D_frame, [0.0, 0.0, 0.0])

    def test_save_contains_all_profile_fields(self, tmp_path, monkeypatch):
        profile_root = self._install_profile_fixture(tmp_path, monkeypatch)
        var = Variation()
        var.g1 = np.array([0.01, 0.02, 0.03])
        var.D = 3.0
        var.save("saved_var")

        saved = profile_root / "variation" / "saved_var.ini"
        assert saved.exists()

        loaded = Variation()
        loaded.load("saved_var")
        for name in Variation.PROFILE_FIELDS:
            expected = getattr(var, name)
            actual = getattr(loaded, name)
            if isinstance(expected, np.ndarray):
                assert np.allclose(actual, expected), name
            else:
                assert actual == pytest.approx(expected), name

    def test_single_vars_are_subset_of_profile_fields(self):
        var = Variation()
        for name in var.single_vars:
            assert name in Variation.PROFILE_FIELDS, name

    def test_non_vars_excludes_profile_fields(self):
        var = Variation()
        for name in Variation.PROFILE_FIELDS:
            assert name not in var.non_vars, name

    def _install_profile_fixture(self, tmp_path, monkeypatch):
        profile_root = tmp_path / "profiles"
        profile_dir = profile_root / "variation"
        profile_dir.mkdir(parents=True)
        source_dir = Path(__file__).parent / "data" / "test_profiles" / "variation"
        for fixture in source_dir.glob("*.ini"):
            shutil.copy(fixture, profile_dir)
        shutil.copy(
            Path("src/spinanalysis/data/profiles/variation/configspec.ini"),
            profile_dir,
        )
        monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", profile_root)
        return profile_root


def _install_option_profiles(tmp_path, monkeypatch, kind):
    profile_root = tmp_path / "profiles"
    profile_dir = profile_root / kind
    profile_dir.mkdir(parents=True)
    source = Path("src/spinanalysis/data/profiles") / kind / "configspec.ini"
    shutil.copy(source, profile_dir)
    monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", profile_root)
    return profile_root
