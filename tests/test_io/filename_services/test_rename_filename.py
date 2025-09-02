"""Test rename operations for converting numeric prefixes to letters.

This module exercises the ``rename_filename`` utilities that rename files by
translating numeric prefixes into alphabetic counterparts and supports dry-run
execution.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from m3c2.io.filename_services import rename_filename


@pytest.mark.parametrize(
    "name, expected",
    [
        ("1-5_cloud", "a-5_cloud"),
        ("foo_1-5_cloud.txt", "foo_a-5_cloud.txt"),
        ("foo_2-10-AI_cloud", "foo_b-10-AI_cloud"),
        ("foo_cloud.txt", "foo_cloud.txt"),
        ("1-1_cloud_2-3_cloud", "a-1_cloud_b-3_cloud"),
    ],
)
def test_transform(name, expected):
    """Ensure numeric prefixes are converted to letters.

    Parameters
    ----------
    name : str
        Original file name.
    expected : str
        Expected result after applying the transformation.
    """

    assert rename_filename.transform(name) == expected


def test_main_renames_and_dry_run(tmp_path, monkeypatch):
    """Validate rename execution and dry-run behavior of the CLI.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used to create example files.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to override command-line arguments.
    """

    f1 = tmp_path / "1-1_cloud.txt"
    f1.touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    f2 = sub / "2-2_cloud.txt"
    f2.touch()

    monkeypatch.setattr(sys, "argv", ["rename_filename", "-r", str(tmp_path)])
    rename_filename.main()

    assert not f1.exists()
    assert not f2.exists()
    assert (tmp_path / "a-1_cloud.txt").exists()
    assert (sub / "b-2_cloud.txt").exists()

    f3 = tmp_path / "1-3_cloud.txt"
    f3.touch()
    monkeypatch.setattr(sys, "argv", ["rename_filename", "-r", "-n", str(tmp_path)])
    rename_filename.main()
    assert f3.exists()
    assert not (tmp_path / "a-3_cloud.txt").exists()
