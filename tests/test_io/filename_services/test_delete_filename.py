import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from m3c2.io.filename_services import delete_filename


@pytest.mark.parametrize(
    "name, expected",
    [
        ("python_file.txt", "file.txt"),
        ("python_python.txt", "python.txt"),
        ("no_prefix.txt", "no_prefix.txt"),
        ("file_python_.txt", "file_.txt"),
        ("python_", ""),
    ],
)
def test_transform(name, expected):
    assert delete_filename.transform(name) == expected


def test_main_renames_and_dry_run(tmp_path, monkeypatch):
    f1 = tmp_path / "python_file1.txt"
    f1.touch()
    sub = tmp_path / "sub"
    sub.mkdir()
    f2 = sub / "python_file2.txt"
    f2.touch()

    monkeypatch.setattr(sys, "argv", ["delete_filename", "-r", str(tmp_path)])
    delete_filename.main()

    assert not f1.exists()
    assert not f2.exists()
    assert (tmp_path / "file1.txt").exists()
    assert (sub / "file2.txt").exists()

    f3 = tmp_path / "python_dry.txt"
    f3.touch()
    monkeypatch.setattr(sys, "argv", ["delete_filename", "-r", "-n", str(tmp_path)])
    delete_filename.main()
    assert f3.exists()
    assert not (tmp_path / "dry.txt").exists()
