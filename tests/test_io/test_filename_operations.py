"""Tests for filename renaming utilities."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

# Ensure repository root on path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from m3c2.importer.filenames import rename_filename, delete_filename


def test_rename_success(tmp_path: Path) -> None:
    """Rename files with numeric group prefixes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by ``pytest``.

    Examples
    --------
    >>> file = tmp_path / "1-42_cloud.txt"
    >>> _ = file.write_text("data")
    >>> new_name = rename_filename.transform(file.name)
    >>> file.rename(tmp_path / new_name)
    """
    file = tmp_path / "1-42_cloud.txt"
    file.write_text("data")

    new_name = rename_filename.transform(file.name)
    file.rename(tmp_path / new_name)

    assert not file.exists()
    assert (tmp_path / new_name).exists()


def test_delete_success(tmp_path: Path) -> None:
    """Remove ``python_`` tokens from filenames.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by ``pytest``.

    Examples
    --------
    >>> file = tmp_path / "python_test.txt"
    >>> _ = file.write_text("data")
    >>> new_name = delete_filename.transform(file.name)
    >>> file.rename(tmp_path / new_name)
    """
    file = tmp_path / "python_test.txt"
    file.write_text("data")

    new_name = delete_filename.transform(file.name)
    file.rename(tmp_path / new_name)

    assert not file.exists()
    assert (tmp_path / new_name).exists()


def test_rename_missing_file(tmp_path: Path) -> None:
    """Raise ``FileNotFoundError`` when source file is absent."""
    missing = tmp_path / "1-42_cloud.txt"
    target = tmp_path / rename_filename.transform(missing.name)

    with pytest.raises(FileNotFoundError):
        missing.rename(target)


def test_delete_missing_file(tmp_path: Path) -> None:
    """Raise ``FileNotFoundError`` for missing files when deleting tokens."""
    missing = tmp_path / "python_test.txt"
    target = tmp_path / delete_filename.transform(missing.name)

    with pytest.raises(FileNotFoundError):
        missing.rename(target)


def test_rename_permission_error(tmp_path: Path, monkeypatch) -> None:
    """Propagate ``PermissionError`` during rename operations."""
    file = tmp_path / "1-42_cloud.txt"
    file.write_text("data")
    target = tmp_path / rename_filename.transform(file.name)

    def fake_rename(self: Path, target_path: Path) -> None:
        raise PermissionError("no permission")

    monkeypatch.setattr(Path, "rename", fake_rename)

    with pytest.raises(PermissionError):
        file.rename(target)


def test_delete_permission_error(tmp_path: Path, monkeypatch) -> None:
    """Propagate ``PermissionError`` during delete operations."""
    file = tmp_path / "python_test.txt"
    file.write_text("data")
    target = tmp_path / delete_filename.transform(file.name)

    def fake_rename(self: Path, target_path: Path) -> None:
        raise PermissionError("no permission")

    monkeypatch.setattr(Path, "rename", fake_rename)

    with pytest.raises(PermissionError):
        file.rename(target)
