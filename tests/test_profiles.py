import shutil
import zipfile
from pathlib import Path

import pytest

from spinanalysis import profiles
from spinanalysis.profiles import (
    _get_profile_name,
    _get_profile_paths,
    _normalize_kinds,
    _normalize_pkind,
    _save_config_plot,
    _set_configobj,
    _validate_config,
    _validate_profile_sections,
)
from configobj import ConfigObj


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_profile_root(tmp_path, monkeypatch):
    root = tmp_path / "profiles"
    root.mkdir()
    monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", root)
    return root


def _install_configspec(tmp_path, monkeypatch, kind):
    root = _make_profile_root(tmp_path, monkeypatch)
    kind_dir = root / kind
    kind_dir.mkdir(parents=True)
    source = Path("src/spinanalysis/data/profiles") / kind / "configspec.ini"
    shutil.copy(source, kind_dir)
    return root


def _install_all_configspecs(tmp_path, monkeypatch):
    root = _make_profile_root(tmp_path, monkeypatch)
    for kind in profiles._VALID_PROFILE_KINDS:
        kind_dir = root / kind
        kind_dir.mkdir(parents=True)
        spec = Path("src/spinanalysis/data/profiles") / kind / "configspec.ini"
        if spec.exists():
            shutil.copy(spec, kind_dir)
    return root


# ---------------------------------------------------------------------------
# _normalize_pkind
# ---------------------------------------------------------------------------


class TestNormalizePkind:
    @pytest.mark.parametrize("pkind", list(profiles._VALID_PROFILE_KINDS))
    def test_valid_kinds_are_returned_lowercased(self, pkind):
        assert _normalize_pkind(pkind) == pkind

    def test_british_spelling_is_mapped(self):
        assert _normalize_pkind("optimisation") == "optimization"

    @pytest.mark.parametrize("pkind", ["invalid", "save", ""])
    def test_invalid_kind_raises(self, pkind):
        with pytest.raises(ValueError):
            _normalize_pkind(pkind)

    def test_uppercase_is_lowercased(self):
        assert _normalize_pkind("Spinsystem") == "spinsystem"


# ---------------------------------------------------------------------------
# _normalize_kinds
# ---------------------------------------------------------------------------


class TestNormalizeKinds:
    def test_all_returns_all_valid_kinds(self):
        result = _normalize_kinds("all")
        assert result == list(profiles._VALID_PROFILE_KINDS)

    def test_single_string_returns_one_element_list(self):
        assert _normalize_kinds("spinsystem") == ["spinsystem"]

    def test_space_separated_string(self):
        result = _normalize_kinds("spinsystem variation")
        assert result == ["spinsystem", "variation"]

    def test_list_of_kinds(self):
        result = _normalize_kinds(["spinsystem", "variation"])
        assert result == ["spinsystem", "variation"]

    def test_british_spelling_in_string(self):
        result = _normalize_kinds("optimisation")
        assert result == ["optimization"]

    def test_british_spelling_in_list(self):
        result = _normalize_kinds(["optimisation"])
        assert result == ["optimization"]

    def test_invalid_kinds_are_dropped(self):
        result = _normalize_kinds("spinsystem invalid save")
        assert result == ["spinsystem"]

    def test_all_invalid_returns_empty_list(self):
        assert _normalize_kinds(["invalid", "save"]) == []


# ---------------------------------------------------------------------------
# _get_profile_paths
# ---------------------------------------------------------------------------


class TestGetProfilePaths:
    def test_returns_all_files_when_pname_all(self, tmp_path):
        (tmp_path / "a.ini").touch()
        (tmp_path / "b.ini").touch()
        result = _get_profile_paths(tmp_path, "all")
        names = sorted(p.name for p in result)
        assert names == ["a.ini", "b.ini"]

    def test_skips_configspec_ini(self, tmp_path):
        (tmp_path / "configspec.ini").touch()
        (tmp_path / "a.ini").touch()
        result = _get_profile_paths(tmp_path, "all")
        assert len(result) == 1
        assert result[0].name == "a.ini"

    def test_filters_by_pname(self, tmp_path):
        (tmp_path / "a.ini").touch()
        (tmp_path / "b.ini").touch()
        result = _get_profile_paths(tmp_path, "b.ini")
        assert len(result) == 1
        assert result[0].name == "b.ini"

    def test_nonexistent_folder_returns_empty(self, tmp_path):
        result = _get_profile_paths(tmp_path / "nonexistent", "all")
        assert result == []

    def test_empty_folder_returns_empty(self, tmp_path):
        assert _get_profile_paths(tmp_path, "all") == []


# ---------------------------------------------------------------------------
# _validate_profile_sections
# ---------------------------------------------------------------------------


class TestValidateProfileSections:
    def test_missing_main_raises(self):
        with pytest.raises(ValueError, match="main"):
            _validate_profile_sections({}, "spinsystem")

    def test_missing_routine_raises_for_optimization(self):
        with pytest.raises(ValueError, match="routine"):
            _validate_profile_sections({"main": {}}, "optimization")

    def test_missing_routine_raises_for_simulation(self):
        with pytest.raises(ValueError, match="routine"):
            _validate_profile_sections({"main": {}}, "simulation")

    def test_empty_routine_raises_for_optimization(self):
        with pytest.raises(ValueError, match="routine"):
            _validate_profile_sections({"main": {"routine": ""}}, "optimization")

    def test_missing_routine_section_raises(self):
        with pytest.raises(ValueError, match="genetic"):
            _validate_profile_sections({"main": {"routine": "genetic"}}, "optimization")

    def test_valid_spinsystem_passes(self):
        _validate_profile_sections({"main": {"g1": [1, 2, 3]}}, "spinsystem")

    def test_valid_optimization_passes(self):
        _validate_profile_sections(
            {"main": {"routine": "genetic"}, "genetic": {"pop_size": 100}},
            "optimization",
        )

    def test_valid_simulation_passes(self):
        _validate_profile_sections(
            {"main": {"routine": "teacups"}, "teacups": {"knots": 20}},
            "simulation",
        )


# ---------------------------------------------------------------------------
# _set_configobj
# ---------------------------------------------------------------------------


class TestSetConfigobj:
    def _make_config(self, tmp_path, kind):
        spec = Path("src/spinanalysis/data/profiles") / kind / "configspec.ini"
        return ConfigObj(configspec=str(spec))

    def test_simple_profile_copies_main_only(self, tmp_path):
        config = self._make_config(tmp_path, "spinsystem")
        profile = {"main": {"g1": [2.0, 2.0, 2.0], "D": 10.0}}
        result = _set_configobj(profile, config, "spinsystem")
        assert result["main"]["g1"] == [2.0, 2.0, 2.0]
        assert result["main"]["D"] == 10.0

    def test_optimization_copies_routine_section(self, tmp_path):
        config = self._make_config(tmp_path, "optimization")
        profile = {
            "main": {"routine": "genetic", "cpu_cores": 2},
            "genetic": {"pop_size": 500},
        }
        result = _set_configobj(profile, config, "optimization")
        assert result["genetic"]["pop_size"] == 500

    def test_simulation_copies_routine_section(self, tmp_path):
        config = self._make_config(tmp_path, "simulation")
        profile = {
            "main": {"routine": "static_radpair", "cpu_cores": 1},
            "static_radpair": {"knots": 25},
        }
        result = _set_configobj(profile, config, "simulation")
        assert result["static_radpair"]["knots"] == 25

    def test_invalid_optimization_routine_raises(self, tmp_path):
        config = self._make_config(tmp_path, "optimization")
        profile = {"main": {"routine": "bogus"}, "bogus": {}}
        with pytest.raises(ValueError, match="optimization routine"):
            _set_configobj(profile, config, "optimization")

    def test_invalid_simulation_routine_raises(self, tmp_path):
        config = self._make_config(tmp_path, "simulation")
        profile = {"main": {"routine": "bogus"}, "bogus": {}}
        with pytest.raises(ValueError, match="simulation routine"):
            _set_configobj(profile, config, "simulation")

    def test_none_values_filtered_from_main(self, tmp_path):
        config = self._make_config(tmp_path, "spinsystem")
        profile = {"main": {"g1": [2.0, 2.0, 2.0], "D": None, "E": 1.0}}
        result = _set_configobj(profile, config, "spinsystem")
        assert "D" not in result["main"]
        assert "E" in result["main"]

    def test_didelphis_tikhonov_accepted(self, tmp_path):
        config = self._make_config(tmp_path, "simulation")
        profile = {
            "main": {"routine": "didelphis_tikhonov", "cpu_cores": 0},
            "didelphis_tikhonov": {"min_r": 10, "max_r": 50, "r_points": 401},
        }
        result = _set_configobj(profile, config, "simulation")
        assert "didelphis_tikhonov" in result


# ---------------------------------------------------------------------------
# add_profile
# ---------------------------------------------------------------------------


class TestAddProfile:
    def test_invalid_pkind_raises(self, tmp_path, monkeypatch):
        _make_profile_root(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            profiles.add_profile({"main": {}}, "invalid", "test")

    def test_default_name_generated_when_empty(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        prof = profiles.new_spinsystem_profile()
        profiles.add_profile(prof, "spinsystem", "")
        assert (profiles.PROFILE_ROOT / "spinsystem" / "profile_1.ini").exists()

    def test_plot_profile_writes_stylesheet(self, tmp_path, monkeypatch):
        root = _make_profile_root(tmp_path, monkeypatch)
        (root / "plot").mkdir()
        prof = profiles.new_plot_profile()
        profiles.add_profile(prof, "plot", "my_style")
        f = root / "plot" / "my_style"
        assert f.exists()
        content = f.read_text()
        assert "SPINANALYSIS SPECIAL SETTINGS" in content

    def test_spinsystem_round_trip(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        prof = profiles.new_spinsystem_profile()
        prof["main"]["g1"] = [2.1, 2.2, 2.3]
        prof["main"]["D"] = 42.0
        profiles.add_profile(prof, "spinsystem", "test_sys")
        loaded = profiles.load_profile("test_sys", "spinsystem")
        assert loaded["main"]["g1"] == [2.1, 2.2, 2.3]
        assert loaded["main"]["D"] == 42.0

    def test_variation_round_trip(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "variation")
        prof = profiles.new_variation_profile()
        prof["main"]["g1"] = [0.001, 0.002, 0.003]
        prof["main"]["D"] = 2.0
        profiles.add_profile(prof, "variation", "test_var")
        loaded = profiles.load_profile("test_var", "variation")
        assert loaded["main"]["g1"] == [0.001, 0.002, 0.003]
        assert loaded["main"]["D"] == 2.0

    def test_optimization_round_trip(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "optimization")
        prof = profiles.new_optimization_profile()
        prof["main"]["routine"] = "minimize"
        prof["main"]["cpu_cores"] = 4
        prof["minimize"]["maxiter"] = 200
        profiles.add_profile(prof, "optimization", "test_opt")
        loaded = profiles.load_profile("test_opt", "optimization")
        assert loaded["main"]["routine"] == "minimize"
        assert loaded["main"]["cpu_cores"] == 4
        assert loaded["minimize"]["maxiter"] == 200

    def test_simulation_round_trip(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "simulation")
        prof = profiles.new_simulation_profile()
        prof["main"]["routine"] = "static_radpair"
        prof["main"]["cpu_cores"] = 2
        prof["static_radpair"]["knots"] = 30
        profiles.add_profile(prof, "simulation", "test_sim")
        loaded = profiles.load_profile("test_sim", "simulation")
        assert loaded["main"]["routine"] == "static_radpair"
        assert loaded["static_radpair"]["knots"] == 30

    def test_configspec_validation_failure_raises(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        prof = profiles.new_spinsystem_profile()
        prof["main"]["width_gauss"] = -1.0
        with pytest.raises(ValueError, match="configspec validation"):
            profiles.add_profile(prof, "spinsystem", "bad")

    def test_none_fields_not_written(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        prof = profiles.new_spinsystem_profile()
        prof["main"]["distribution"] = None
        profiles.add_profile(prof, "spinsystem", "test_none")
        loaded = profiles.load_profile("test_none", "spinsystem")
        assert "distribution" not in loaded["main"]

    def test_creates_directory_if_missing(self, tmp_path, monkeypatch):
        root = tmp_path / "profiles"
        monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", root)
        spec_src = Path("src/spinanalysis/data/profiles/spinsystem/configspec.ini")
        (root / "spinsystem").mkdir(parents=True)
        shutil.copy(spec_src, root / "spinsystem")
        prof = profiles.new_spinsystem_profile()
        profiles.add_profile(prof, "spinsystem", "test")
        assert (root / "spinsystem" / "test.ini").exists()


# ---------------------------------------------------------------------------
# _save_config_plot
# ---------------------------------------------------------------------------


class TestSaveConfigPlot:
    def test_writes_preamble_header(self, tmp_path):
        path = tmp_path / "style"
        _save_config_plot({}, path)
        content = path.read_text()
        assert "SPINANALYSIS SPECIAL SETTINGS" in content

    def test_lim_keys_formatted_as_pair(self, tmp_path):
        path = tmp_path / "style"
        _save_config_plot({"xlim": [1.0, 2.0]}, path)
        content = path.read_text()
        assert "xlim: 1.0, 2.0" in content

    def test_numeric_values_unquoted(self, tmp_path):
        path = tmp_path / "style"
        _save_config_plot({"percentage_mode": True}, path)
        content = path.read_text()
        assert "percentage_mode: True" in content

    def test_string_values_quoted(self, tmp_path):
        path = tmp_path / "style"
        _save_config_plot({"xlabel": "Field"}, path)
        content = path.read_text()
        assert 'xlabel: "Field"' in content


# ---------------------------------------------------------------------------
# _get_profile_name
# ---------------------------------------------------------------------------


class TestGetProfileName:
    def test_returns_profile_1_for_empty_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", tmp_path)
        (tmp_path / "spinsystem").mkdir()
        assert _get_profile_name("spinsystem") == "profile_1"

    def test_returns_profile_1_for_nonexistent_folder(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", tmp_path)
        assert _get_profile_name("spinsystem") == "profile_1"

    def test_returns_next_available(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", tmp_path)
        d = tmp_path / "spinsystem"
        d.mkdir()
        (d / "profile_1.ini").touch()
        assert _get_profile_name("spinsystem") == "profile_2"

    def test_skips_non_numeric_suffixes(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", tmp_path)
        d = tmp_path / "spinsystem"
        d.mkdir()
        (d / "profile_abc.ini").touch()
        assert _get_profile_name("spinsystem") == "profile_1"

    def test_finds_smallest_gap(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", tmp_path)
        d = tmp_path / "spinsystem"
        d.mkdir()
        (d / "profile_1.ini").touch()
        (d / "profile_3.ini").touch()
        assert _get_profile_name("spinsystem") == "profile_2"

    def test_plot_kind_no_ini_suffix(self, tmp_path, monkeypatch):
        monkeypatch.setattr("spinanalysis.profiles.PROFILE_ROOT", tmp_path)
        d = tmp_path / "plot"
        d.mkdir()
        (d / "profile_1").touch()
        assert _get_profile_name("plot") == "profile_2"


# ---------------------------------------------------------------------------
# _validate_config
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid_config_passes(self, tmp_path):
        spec = Path("src/spinanalysis/data/profiles/spinsystem/configspec.ini")
        config = ConfigObj(configspec=str(spec))
        config["main"] = {"g1": [2.0, 2.0, 2.0], "D": 1.0, "width_gauss": 0.5}
        _validate_config(config, "test")

    def test_invalid_config_raises(self, tmp_path):
        spec = Path("src/spinanalysis/data/profiles/spinsystem/configspec.ini")
        config = ConfigObj(configspec=str(spec))
        config["main"] = {"width_gauss": -1.0}
        with pytest.raises(ValueError, match="configspec validation"):
            _validate_config(config, "test")

    def test_none_string_values_skipped(self, tmp_path):
        spec = Path("src/spinanalysis/data/profiles/spinsystem/configspec.ini")
        config = ConfigObj(configspec=str(spec))
        config["main"] = {"distribution": "None"}
        _validate_config(config, "test")

    def test_whole_section_failure_raises(self, tmp_path):
        spec = Path("src/spinanalysis/data/profiles/optimization/configspec.ini")
        config = ConfigObj(configspec=str(spec))
        config["main"] = {"routine": "bogus"}
        with pytest.raises(ValueError, match="configspec validation"):
            _validate_config(config, "test")


# ---------------------------------------------------------------------------
# load_profile
# ---------------------------------------------------------------------------


class TestLoadProfile:
    def test_normalises_pkind(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        prof = profiles.new_spinsystem_profile()
        profiles.add_profile(prof, "spinsystem", "test")
        loaded = profiles.load_profile("test", "Spinsystem")
        assert "main" in loaded

    def test_strips_ini_suffix(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        prof = profiles.new_spinsystem_profile()
        profiles.add_profile(prof, "spinsystem", "test")
        loaded = profiles.load_profile("test.ini", "spinsystem")
        assert "main" in loaded

    def test_returns_dict_with_sections(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        prof = profiles.new_spinsystem_profile()
        profiles.add_profile(prof, "spinsystem", "test")
        loaded = profiles.load_profile("test", "spinsystem")
        assert isinstance(loaded, dict)
        assert "main" in loaded
        assert isinstance(loaded["main"], dict)

    def test_invalid_pkind_raises(self, tmp_path, monkeypatch):
        _make_profile_root(tmp_path, monkeypatch)
        with pytest.raises(ValueError):
            profiles.load_profile("test", "invalid")

    def test_nonexistent_file_raises(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        with pytest.raises(IOError):
            profiles.load_profile("nonexistent", "spinsystem")

    def test_validation_failure_raises(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        root = profiles.PROFILE_ROOT
        bad = root / "spinsystem" / "bad.ini"
        bad.write_text("[main]\nwidth_gauss = -1.0\n")
        with pytest.raises(ValueError, match="configspec validation"):
            profiles.load_profile("bad", "spinsystem")

    def test_round_trip_load(self, tmp_path, monkeypatch):
        _install_configspec(tmp_path, monkeypatch, "spinsystem")
        prof = profiles.new_spinsystem_profile()
        prof["main"]["g1"] = [2.1, 2.2, 2.3]
        prof["main"]["D"] = 42.0
        prof["main"]["E"] = 1.5
        profiles.add_profile(prof, "spinsystem", "round_trip")
        loaded = profiles.load_profile("round_trip", "spinsystem")
        assert loaded["main"]["g1"] == [2.1, 2.2, 2.3]
        assert loaded["main"]["D"] == 42.0
        assert loaded["main"]["E"] == 1.5


# ---------------------------------------------------------------------------
# load_plot_profile
# ---------------------------------------------------------------------------


class TestLoadPlotProfile:
    def _install_plot_root(self, tmp_path, monkeypatch):
        root = _make_profile_root(tmp_path, monkeypatch)
        plot_dir = root / "plot"
        plot_dir.mkdir()
        shutil.copy(
            Path("src/spinanalysis/data/profiles/plot/default_stylesheet"),
            plot_dir,
        )
        return root

    def test_none_loads_default_stylesheet(self, tmp_path, monkeypatch):
        self._install_plot_root(tmp_path, monkeypatch)
        prof = profiles.load_plot_profile(None)
        assert prof["percentage_mode"] is True

    def test_matplotlib_builtin_style_loads_default(self, tmp_path, monkeypatch):
        self._install_plot_root(tmp_path, monkeypatch)
        prof = profiles.load_plot_profile("seaborn-v0_8-whitegrid")
        assert prof["percentage_mode"] is True

    def test_loads_custom_stylesheet(self, tmp_path, monkeypatch):
        root = self._install_plot_root(tmp_path, monkeypatch)
        custom = root / "plot" / "custom"
        custom.write_text(
            "percentage_mode: False\n"
            "xlim: 1.0, 2.0\n"
            'xlabel: "Field"\n'
            "show_title: True\n"
        )
        prof = profiles.load_plot_profile("custom")
        assert prof["percentage_mode"] is False
        assert prof["xlim"] == [1.0, 2.0]
        assert prof["xlabel"] == "Field"
        assert prof["show_title"] is True

    def test_parses_lim_keys(self, tmp_path, monkeypatch):
        root = self._install_plot_root(tmp_path, monkeypatch)
        custom = root / "plot" / "lim_test"
        custom.write_text("xlim: 0.5, 3.5\n")
        prof = profiles.load_plot_profile("lim_test")
        assert prof["xlim"] == [0.5, 3.5]

    def test_parses_label_keys(self, tmp_path, monkeypatch):
        root = self._install_plot_root(tmp_path, monkeypatch)
        custom = root / "plot" / "label_test"
        custom.write_text('ylabel: "Time"\n')
        prof = profiles.load_plot_profile("label_test")
        assert prof["ylabel"] == "Time"

    def test_parses_title_key(self, tmp_path, monkeypatch):
        root = self._install_plot_root(tmp_path, monkeypatch)
        custom = root / "plot" / "title_test"
        custom.write_text('title: "My Plot"\n')
        prof = profiles.load_plot_profile("title_test")
        assert prof["title"] == "My Plot"

    def test_parses_boolean_keys(self, tmp_path, monkeypatch):
        root = self._install_plot_root(tmp_path, monkeypatch)
        custom = root / "plot" / "bool_test"
        custom.write_text("legend: True\ncolorbar: False\n")
        prof = profiles.load_plot_profile("bool_test")
        assert prof["legend"] is True
        assert prof["colorbar"] is False

    def test_skips_comments_and_blanks(self, tmp_path, monkeypatch):
        root = self._install_plot_root(tmp_path, monkeypatch)
        custom = root / "plot" / "comment_test"
        custom.write_text("# comment\n\npercentage_mode: True\n")
        prof = profiles.load_plot_profile("comment_test")
        assert prof == {"percentage_mode": True}

    def test_skips_unknown_keys(self, tmp_path, monkeypatch):
        root = self._install_plot_root(tmp_path, monkeypatch)
        custom = root / "plot" / "unknown_test"
        custom.write_text("unknown_key: value\npercentage_mode: True\n")
        prof = profiles.load_plot_profile("unknown_test")
        assert "unknown_key" not in prof
        assert prof["percentage_mode"] is True

    def test_malformed_bounds_raise(self, tmp_path, monkeypatch):
        root = self._install_plot_root(tmp_path, monkeypatch)
        custom = root / "plot" / "bad_bounds"
        custom.write_text("xlim: 1.0\n")
        with pytest.raises(ValueError, match="Malformed bounds"):
            profiles.load_plot_profile("bad_bounds")


# ---------------------------------------------------------------------------
# new_plot_profile
# ---------------------------------------------------------------------------


class TestNewPlotProfile:
    def test_returns_dict_with_expected_keys(self):
        prof = profiles.new_plot_profile()
        expected_keys = {
            "percentage_mode",
            "xlim",
            "ylim",
            "zlim",
            "xlabel",
            "ylabel",
            "zlabel",
            "legend",
            "show_title",
            "title",
            "colorbar",
        }
        assert set(prof.keys()) == expected_keys

    def test_default_values(self):
        prof = profiles.new_plot_profile()
        assert prof["percentage_mode"] is False
        assert prof["xlim"] == [0, 0]
        assert prof["xlabel"] == ""


# ---------------------------------------------------------------------------
# new_spinsystem_profile
# ---------------------------------------------------------------------------


class TestNewSpinsystemProfile:
    def test_has_main_section(self):
        prof = profiles.new_spinsystem_profile()
        assert "main" in prof

    def test_contains_all_configspec_fields(self):
        spec_path = Path("src/spinanalysis/data/profiles/spinsystem/configspec.ini")
        spec_text = spec_path.read_text()
        spec_fields = set()
        for line in spec_text.splitlines():
            line = line.strip()
            if not line or line.startswith("[") or line.startswith("#"):
                continue
            key = line.split("=")[0].strip()
            if key:
                spec_fields.add(key)
        prof_fields = set(profiles.new_spinsystem_profile()["main"].keys())
        assert spec_fields <= prof_fields

    def test_key_default_values(self):
        prof = profiles.new_spinsystem_profile()["main"]
        assert prof["g1"] == [2.002, 2.002, 2.002]
        assert prof["D_tri"] == 700.0
        assert prof["beta"] == 1.4
        assert prof["width_gauss"] == 0.5


# ---------------------------------------------------------------------------
# new_variation_profile
# ---------------------------------------------------------------------------


class TestNewVariationProfile:
    def test_has_main_section(self):
        prof = profiles.new_variation_profile()
        assert "main" in prof

    def test_contains_fit_distribution(self):
        prof = profiles.new_variation_profile()
        assert "fit_distribution" in prof["main"]

    def test_all_values_are_zero_defaults(self):
        prof = profiles.new_variation_profile()["main"]
        assert prof["g1"] == [0, 0, 0]
        assert prof["D"] == 0.0
        assert prof["fit_distribution"] is False


# ---------------------------------------------------------------------------
# new_optimization_profile
# ---------------------------------------------------------------------------


class TestNewOptimizationProfile:
    def test_has_main_and_all_routine_sections(self):
        prof = profiles.new_optimization_profile()
        assert "main" in prof
        for r in profiles._VALID_OPTIMIZATION_ROUTINES:
            assert r in prof

    def test_main_routine_is_empty(self):
        prof = profiles.new_optimization_profile()
        assert prof["main"]["routine"] == ""

    def test_genetic_section_has_expected_keys(self):
        prof = profiles.new_optimization_profile()
        assert "pop_size" in prof["genetic"]
        assert "p_c" in prof["genetic"]

    def test_minimize_section_has_expected_keys(self):
        prof = profiles.new_optimization_profile()
        assert "maxiter" in prof["minimize"]


# ---------------------------------------------------------------------------
# new_simulation_profile
# ---------------------------------------------------------------------------


class TestNewSimulationProfile:
    def test_has_main_and_all_routine_sections(self):
        prof = profiles.new_simulation_profile()
        assert "main" in prof
        for r in profiles._VALID_SIMULATION_ROUTINES:
            assert r in prof

    def test_main_routine_is_empty(self):
        prof = profiles.new_simulation_profile()
        assert prof["main"]["routine"] == ""

    def test_static_radpair_has_knots(self):
        prof = profiles.new_simulation_profile()
        assert prof["static_radpair"]["knots"] == 20

    def test_didelphis_tikhonov_has_force_cpu_and_regularization_mode(self):
        prof = profiles.new_simulation_profile()
        assert "force_cpu" in prof["didelphis_tikhonov"]
        assert "regularization_mode" in prof["didelphis_tikhonov"]


# ---------------------------------------------------------------------------
# import_profiles
# ---------------------------------------------------------------------------


class TestImportProfiles:
    def test_extracts_valid_zip(self, tmp_path, monkeypatch):
        root = _make_profile_root(tmp_path, monkeypatch)
        zip_path = tmp_path / "import.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("spinsystem/my_profile.ini", "[main]\ng1 = 1.0, 1.0, 1.0\n")
        profiles.import_profiles(str(zip_path))
        assert (root / "spinsystem" / "my_profile.ini").exists()

    def test_invalid_top_level_raises(self, tmp_path, monkeypatch):
        _make_profile_root(tmp_path, monkeypatch)
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("bogus/file.ini", "[main]\n")
        with pytest.raises(ValueError, match="valid profile kind"):
            profiles.import_profiles(str(zip_path))

    def test_skip_existing_when_override_false(self, tmp_path, monkeypatch):
        root = _make_profile_root(tmp_path, monkeypatch)
        existing = root / "spinsystem" / "my_profile.ini"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("original")
        zip_path = tmp_path / "import.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("spinsystem/my_profile.ini", "new")
        profiles.import_profiles(str(zip_path), override=False)
        assert existing.read_text() == "original"

    def test_overwrite_when_override_true(self, tmp_path, monkeypatch):
        root = _make_profile_root(tmp_path, monkeypatch)
        existing = root / "spinsystem" / "my_profile.ini"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("original")
        zip_path = tmp_path / "import.zip"
        with zipfile.ZipFile(str(zip_path), "w") as zf:
            zf.writestr("spinsystem/my_profile.ini", "new")
        profiles.import_profiles(str(zip_path), override=True)
        assert existing.read_text() == "new"


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


class TestExport:
    def _setup_profiles(self, tmp_path, monkeypatch):
        root = _install_configspec(tmp_path, monkeypatch, "spinsystem")
        prof = profiles.new_spinsystem_profile()
        profiles.add_profile(prof, "spinsystem", "export_test")
        return root

    def test_creates_zip_in_cwd_when_path_none(self, tmp_path, monkeypatch):
        self._setup_profiles(tmp_path, monkeypatch)
        monkeypatch.chdir(tmp_path)
        profiles.export(path=None, pkind="spinsystem")
        assert (tmp_path / "profiles_spinanalysis.zip").exists()

    def test_creates_zip_in_given_path(self, tmp_path, monkeypatch):
        self._setup_profiles(tmp_path, monkeypatch)
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        profiles.export(path=str(out_dir), pkind="spinsystem")
        assert (out_dir / "profiles_spinanalysis.zip").exists()

    def test_all_exports_all_kinds(self, tmp_path, monkeypatch):
        _install_all_configspecs(tmp_path, monkeypatch)
        sp = profiles.new_spinsystem_profile()
        profiles.add_profile(sp, "spinsystem", "a")
        vp = profiles.new_variation_profile()
        profiles.add_profile(vp, "variation", "b")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        profiles.export(path=str(out_dir), pkind="all")
        with zipfile.ZipFile(str(out_dir / "profiles_spinanalysis.zip"), "r") as zf:
            names = zf.namelist()
        assert any("spinsystem/a.ini" in n for n in names)
        assert any("variation/b.ini" in n for n in names)

    def test_specific_kind_exports_only_that_kind(self, tmp_path, monkeypatch):
        _install_all_configspecs(tmp_path, monkeypatch)
        sp = profiles.new_spinsystem_profile()
        profiles.add_profile(sp, "spinsystem", "a")
        vp = profiles.new_variation_profile()
        profiles.add_profile(vp, "variation", "b")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        profiles.export(path=str(out_dir), pkind="spinsystem")
        with zipfile.ZipFile(str(out_dir / "profiles_spinanalysis.zip"), "r") as zf:
            names = zf.namelist()
        assert any("spinsystem/a.ini" in n for n in names)
        assert not any("variation/b.ini" in n for n in names)

    def test_skips_configspec_ini(self, tmp_path, monkeypatch):
        self._setup_profiles(tmp_path, monkeypatch)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        profiles.export(path=str(out_dir), pkind="spinsystem")
        with zipfile.ZipFile(str(out_dir / "profiles_spinanalysis.zip"), "r") as zf:
            names = zf.namelist()
        assert not any("configspec.ini" in n for n in names)

    def test_arcnames_are_relative(self, tmp_path, monkeypatch):
        self._setup_profiles(tmp_path, monkeypatch)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        profiles.export(path=str(out_dir), pkind="spinsystem")
        with zipfile.ZipFile(str(out_dir / "profiles_spinanalysis.zip"), "r") as zf:
            names = zf.namelist()
        for n in names:
            assert not n.startswith("/")
            assert "spinsystem/export_test.ini" in n
