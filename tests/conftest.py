"""Shared pytest fixtures and configuration for the test suite.

© M. Sc. Florian Quintes, 2026

@contact: florian.quintes@pc.uni.freiburg.de

@author: Florian Quintes
"""

import zipfile
from pathlib import Path

import pytest


_ZIP_PATH = Path(__file__).parent / "data" / "test_loading.zip"


@pytest.fixture(scope="session")
def test_loading_dir(tmp_path_factory):
    """Extract the test_loading ZIP archive once per session.

    Returns the path to the extracted ``test_loading`` directory.
    """
    extract_root = tmp_path_factory.mktemp("test_loading_data")
    with zipfile.ZipFile(_ZIP_PATH) as zf:
        zf.extractall(extract_root)
    return extract_root / "test_loading"


@pytest.fixture(autouse=True)
def _mock_loading_home(monkeypatch, test_loading_dir):
    """Point ``Path.home()`` at the extracted test_loading directory.

    This lets ``get_full_path()`` find the test data folders without
    ``start_directory`` for any test module that imports from
    ``spinanalysis.loading``.
    """
    monkeypatch.setattr(Path, "home", lambda: test_loading_dir)
