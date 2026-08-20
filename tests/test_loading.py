"""Tests for the EPR data loading routines in :mod:`spinanalysis.loading`.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from spinanalysis.loading import (
    _find_file,
    _find_transient_basename,
    convert_parameter_type,
    get_byte_mode,
    get_data_dimension,
    get_DSC_parameters,
    get_full_path,
    get_transient_data,
    get_transient_info,
    load_axis_vector,
    load_data_vector,
    load_epr_bruker_bes3t,
    load_epr_ESP_transient,
    load_matlab,
    load_txt,
    read_single_transient_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def data_dir(test_loading_dir):
    return test_loading_dir


# ---------------------------------------------------------------------------
# TestConvertParameterType
# ---------------------------------------------------------------------------


class TestConvertParameterType:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("True", True),
            ("False", False),
            ("yes", True),
            ("no", False),
            ("1", True),
            ("0", False),
            ("on", True),
            ("off", False),
        ],
    )
    def test_bool(self, value, expected):
        assert convert_parameter_type(value) is expected

    @pytest.mark.parametrize(
        ("value", "expected"), [("42", 42), ("-7", -7), ("0", True)]
    )
    def test_int(self, value, expected):
        result = convert_parameter_type(value)
        if value == "0":
            assert result is False
        else:
            assert isinstance(result, int)
            assert result == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("3.14", 3.14), ("1e-5", 1e-5), ("-2.5", -2.5)],
    )
    def test_float(self, value, expected):
        result = convert_parameter_type(value)
        assert isinstance(result, float)
        assert result == pytest.approx(expected)

    def test_string(self):
        assert convert_parameter_type("hello") == "hello"

    def test_quoted_string(self):
        assert convert_parameter_type("'quoted'") == "quoted"


# ---------------------------------------------------------------------------
# TestGetByteMode
# ---------------------------------------------------------------------------


class TestGetByteMode:
    @pytest.mark.parametrize(
        ("irfmt", "expected"),
        [("D", ">f8"), ("F", ">f"), ("I", ">i4"), ("S", ">i2"), ("C", ">i")],
    )
    def test_big_endian(self, irfmt, expected):
        dsc = {"BSEQ": "BIG", "IRFMT": irfmt}
        assert get_byte_mode(dsc) == expected

    def test_little_endian(self):
        dsc = {"BSEQ": "LIT", "IRFMT": "D"}
        assert get_byte_mode(dsc) == "<f8"

    def test_iifmt_key(self):
        dsc = {"BSEQ": "BIG", "IIFMT": "D"}
        assert get_byte_mode(dsc, data_key="IIFMT") == ">f8"

    def test_ascii_raises(self):
        dsc = {"BSEQ": "BIG", "IRFMT": "A"}
        with pytest.raises(ValueError, match="ASCII"):
            get_byte_mode(dsc)

    def test_unknown_bseq(self):
        dsc = {"BSEQ": "WEIRD", "IRFMT": "D"}
        with pytest.raises(ValueError, match="BSEQ"):
            get_byte_mode(dsc)

    def test_unknown_format(self):
        dsc = {"BSEQ": "BIG", "IRFMT": "X"}
        with pytest.raises(ValueError, match="IRFMT"):
            get_byte_mode(dsc)


# ---------------------------------------------------------------------------
# TestGetDataDimension
# ---------------------------------------------------------------------------


class TestGetDataDimension:
    def _base_dict(self):
        return {"XTYP": "IDX", "YTYP": "NODATA", "ZTYP": "NODATA"}

    def test_1d(self):
        dsc = self._base_dict()
        get_data_dimension(dsc)
        assert dsc["dimensions"] == 1

    def test_2d(self):
        dsc = self._base_dict()
        dsc["YTYP"] = "IGD"
        get_data_dimension(dsc)
        assert dsc["dimensions"] == 2

    def test_3d(self):
        dsc = self._base_dict()
        dsc["YTYP"] = "IGD"
        dsc["ZTYP"] = "IGD"
        get_data_dimension(dsc)
        assert dsc["dimensions"] == 3


# ---------------------------------------------------------------------------
# TestGetDSCParameters
# ---------------------------------------------------------------------------


class TestGetDSCParameters:
    def test_pulse_eseem_qband(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "pulse_eseem_qband"))
        assert dsc["XPTS"] == 190
        assert dsc["IKKF"] == "CPLX"
        assert dsc["BSEQ"] == "BIG"
        assert dsc["IRFMT"] == "D"
        assert dsc["TITL"] == "pulse_eseem_qband"
        assert "path_to_folder" in dsc

    def test_cw_xband_100k(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "cw_xband_100k"))
        assert dsc["XPTS"] == 3750
        assert dsc["IKKF"] == "REAL"

    def test_trepr_xband_2d(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "trepr_xband_2d"))
        assert dsc["XPTS"] == 4096
        assert dsc["YPTS"] == 400
        assert dsc["YTYP"] == "IGD"

    def test_missing_dsc(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_DSC_parameters(str(tmp_path))


# ---------------------------------------------------------------------------
# TestLoadDataVector
# ---------------------------------------------------------------------------


class TestLoadDataVector:
    def test_pulse_eseem_qband(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "pulse_eseem_qband"))
        data = load_data_vector(dsc)
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.complex128
        assert data.shape == (190,)

    def test_cw_xband_100k(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "cw_xband_100k"))
        data = load_data_vector(dsc)
        assert data.dtype == np.complex128
        assert data.shape == (3750,)
        assert np.all(data.imag == 0)

    def test_trepr_xband_2d(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "trepr_xband_2d"))
        data = load_data_vector(dsc)
        assert data.dtype == np.complex128
        assert data.shape == (4096, 400)


# ---------------------------------------------------------------------------
# TestLoadAxisVector
# ---------------------------------------------------------------------------


class TestLoadAxisVector:
    def test_x_axis_cw(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "cw_xband_100k"))
        x = load_axis_vector("x", dsc)
        assert isinstance(x, np.ndarray)
        assert x.shape == (3750,)
        assert np.all(np.diff(x) > 0)

    def test_y_axis_2d(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "trepr_xband_2d"))
        y = load_axis_vector("y", dsc)
        assert y.shape == (400,)

    def test_z_axis_missing_warns(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "pulse_eseem_qband"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(UserWarning, match="No axis data"):
                load_axis_vector("z", dsc)

    def test_z_axis_missing_returns_empty(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "pulse_eseem_qband"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = load_axis_vector("z", dsc)
        assert z.shape == (0,)

    def test_invalid_axis(self, data_dir):
        dsc = get_DSC_parameters(str(data_dir / "cw_xband_100k"))
        with pytest.raises(ValueError, match="Axis needs to start"):
            load_axis_vector("w", dsc)


# ---------------------------------------------------------------------------
# TestLoadEprBrukerBES3T
# ---------------------------------------------------------------------------


class TestLoadEprBrukerBES3T:
    def test_pulse_eseem_qband(self):
        with pytest.warns(
            UserWarning,
            match=r"No axis data found for axis 'Y'; returning empty array.|No axis data found for axis 'Z'; returning empty array.",
        ):
            axis, data = load_epr_bruker_bes3t("pulse_eseem_qband")
        assert len(axis) == 3
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.complex128
        assert data.shape == (190,)
        assert axis[0].shape == (190,)

    def test_cw_xband_100k(self):
        with pytest.warns(
            UserWarning,
            match=r"No axis data found for axis 'Y'; returning empty array.|No axis data found for axis 'Z'; returning empty array.",
        ):
            axis, data = load_epr_bruker_bes3t("cw_xband_100k")
        assert data.shape == (3750,)
        assert axis[0].shape == (3750,)

    def test_pulse_laserimd_qband(self):
        with pytest.warns(
            UserWarning,
            match=r"No axis data found for axis 'Y'; returning empty array.|No axis data found for axis 'Z'; returning empty array.",
        ):
            axis, data = load_epr_bruker_bes3t("pulse_laserimd_qband")
        assert data.shape == (225,)

    def test_trepr_xband_2d(self):
        with pytest.warns(
            UserWarning,
            match=r"No axis data found for axis 'Y'; returning empty array.|No axis data found for axis 'Z'; returning empty array.",
        ):
            axis, data = load_epr_bruker_bes3t("trepr_xband_2d")
        assert data.shape == (4096, 400)
        assert axis[1].shape == (400,)

    def test_nonexistent_folder(self):
        with pytest.raises(FileNotFoundError):
            load_epr_bruker_bes3t("does_not_exist")


# ---------------------------------------------------------------------------
# TestGetTransientInfo
# ---------------------------------------------------------------------------


class TestGetTransientInfo:
    def test_transient_xband(self, data_dir):
        info = get_transient_info(str(data_dir / "transient_xband"))
        assert len(info) == 5
        time_length, time_points, field_start, field_stop, field_step = info
        assert isinstance(time_length, float)
        assert isinstance(time_points, int)
        assert isinstance(field_start, float)
        assert isinstance(field_stop, float)
        assert isinstance(field_step, float)
        assert field_stop > field_start

    def test_missing_info(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_transient_info(str(tmp_path))


# ---------------------------------------------------------------------------
# TestReadSingleTransientFile
# ---------------------------------------------------------------------------


class TestReadSingleTransientFile:
    def test_with_time(self, data_dir):
        folder = str(data_dir / "transient_xband")
        (field, time_axis), data = read_single_transient_file(
            folder, "transient_xband", 1, 3, time=True
        )
        assert isinstance(field, float)
        assert isinstance(time_axis, np.ndarray)
        assert isinstance(data, np.ndarray)
        assert len(time_axis) == len(data)

    def test_without_time(self, data_dir):
        folder = str(data_dir / "transient_xband")
        field, data = read_single_transient_file(folder, "transient_xband", 1, 3)
        assert isinstance(field, float)
        assert isinstance(data, np.ndarray)

    def test_qband(self, data_dir):
        folder = str(data_dir / "transient_qband")
        (field, time_axis), data = read_single_transient_file(
            folder, "transient_qband", 1, 3, time=True
        )
        assert len(time_axis) == len(data)


# ---------------------------------------------------------------------------
# TestGetTransientData
# ---------------------------------------------------------------------------


class TestGetTransientData:
    def test_transient_xband(self, data_dir):
        axis, data = get_transient_data(str(data_dir / "transient_xband"))
        field, time = axis
        assert isinstance(field, np.ndarray)
        assert isinstance(time, np.ndarray)
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.complex128
        assert data.shape[0] == len(time)
        assert data.shape[1] == len(field)

    def test_transient_qband(self, data_dir):
        axis, data = get_transient_data(str(data_dir / "transient_qband"))
        field, time = axis
        assert data.dtype == np.complex128
        assert data.shape[0] == len(time)
        assert data.shape[1] == len(field)


# ---------------------------------------------------------------------------
# TestLoadEprESPTransient
# ---------------------------------------------------------------------------


class TestLoadEprESPTransient:
    def test_transient_xband(self):
        axis, data = load_epr_ESP_transient("transient_xband")
        assert len(axis) == 2
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.complex128

    def test_transient_qband(self):
        axis, data = load_epr_ESP_transient("transient_qband")
        assert len(axis) == 2
        assert data.dtype == np.complex128

    def test_nonexistent_folder(self):
        with pytest.raises(FileNotFoundError):
            load_epr_ESP_transient("does_not_exist")


# ---------------------------------------------------------------------------
# TestLoadMatlab
# ---------------------------------------------------------------------------


class TestLoadMatlab:
    def test_load_matlab(self):
        axis, data = load_matlab("matlab")
        x, y = axis
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(data, np.ndarray)
        assert len(x) == 256
        assert len(data) == 256
        assert y.shape == (0,)

    @pytest.mark.parametrize("idx", list(range(1, 12)))
    def test_all_mat_files_valid(self, idx, data_dir):
        mat_file = data_dir / "matlab" / f"Test_{idx}.mat"
        d = loadmat(str(mat_file))
        assert "field" in d
        assert "signal" in d
        assert d["field"].shape == (1, 256)
        assert d["signal"].shape == (1, 256)

    def test_wrong_field_key(self):
        with pytest.raises(KeyError):
            load_matlab("matlab", field="nonexistent")

    def test_nonexistent_folder(self):
        with pytest.raises(FileNotFoundError):
            load_matlab("does_not_exist")


# ---------------------------------------------------------------------------
# TestLoadTxt
# ---------------------------------------------------------------------------


class TestLoadTxt:
    def test_two_columns(self, tmp_path):
        folder = tmp_path / "txt_2col"
        folder.mkdir()
        (folder / "data.txt").write_text("1.0 10.0\n2.0 20.0\n3.0 30.0\n")
        axis, data = load_txt("txt_2col", start_directory=str(tmp_path.parent))
        x, y = axis
        assert len(x) == 3
        assert y.shape == (0,)
        assert len(data) == 3

    def test_three_columns(self, tmp_path):
        folder = tmp_path / "txt_3col"
        folder.mkdir()
        (folder / "data.txt").write_text("1.0 5.0 10.0\n2.0 6.0 20.0\n3.0 7.0 30.0\n")
        axis, data = load_txt("txt_3col", start_directory=str(tmp_path.parent))
        x, y = axis
        assert len(x) == 3
        assert len(y) == 3
        assert len(data) == 3

    def test_empty_file(self, tmp_path):
        folder = tmp_path / "txt_empty"
        folder.mkdir()
        (folder / "data.txt").write_text("# only a comment\n\n")
        with pytest.raises(ValueError, match="no data"):
            load_txt("txt_empty", start_directory=str(tmp_path.parent))

    def test_wrong_column_count(self, tmp_path):
        folder = tmp_path / "txt_5col"
        folder.mkdir()
        (folder / "data.txt").write_text("1 2 3 4 5\n6 7 8 9 10\n")
        with pytest.raises(ValueError, match="2 or 3 columns"):
            load_txt("txt_5col", start_directory=str(tmp_path.parent))


# ---------------------------------------------------------------------------
# TestGetFullPath
# ---------------------------------------------------------------------------


class TestGetFullPath:
    def test_existing_folder(self):
        path = get_full_path("pulse_eseem_qband")
        assert "pulse_eseem_qband" in path
        assert Path(path).is_dir()

    def test_nonexistent_folder(self):
        with pytest.raises(FileNotFoundError):
            get_full_path("does_not_exist")

    def test_with_start_directory(self):
        path = get_full_path("pulse_eseem_qband", start_directory=".")
        assert "pulse_eseem_qband" in path


# ---------------------------------------------------------------------------
# TestFindFile
# ---------------------------------------------------------------------------


class TestFindFile:
    def test_find_dsc(self, data_dir):
        result = _find_file(data_dir / "pulse_eseem_qband", ".DSC")
        assert result.suffix == ".DSC"
        assert result.is_file()

    def test_find_not_found(self, data_dir):
        with pytest.raises(FileNotFoundError):
            _find_file(data_dir / "pulse_eseem_qband", ".xyz")


# ---------------------------------------------------------------------------
# TestFindTransientBasename
# ---------------------------------------------------------------------------


class TestFindTransientBasename:
    def test_xband(self, data_dir):
        basename = _find_transient_basename(data_dir / "transient_xband")
        assert basename == "transient_xband"

    def test_qband(self, data_dir):
        basename = _find_transient_basename(data_dir / "transient_qband")
        assert basename == "transient_qband"

    def test_no_numbered_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _find_transient_basename(tmp_path)
