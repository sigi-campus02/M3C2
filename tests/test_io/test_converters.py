"""Tests for the :mod:`m3c2.importer.converters` module."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
import m3c2.importer.converters as conv


def test_ensure_xyz_existing_file(tmp_path: Path) -> None:
    """Return the existing XYZ file without conversion."""
    base = tmp_path / "cloud"
    xyz_path = base.with_suffix(".xyz")
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.savetxt(xyz_path, data, fmt="%.6f")

    result = conv.ensure_xyz(base, ("xyz", xyz_path))

    assert result == xyz_path
    assert result.exists()
    assert np.allclose(np.loadtxt(result), data)


def test_ensure_xyz_las_conversion(monkeypatch, tmp_path: Path) -> None:
    """Convert LAS input to XYZ using mocked reader."""
    base = tmp_path / "points"
    las_path = base.with_suffix(".las")
    las_path.write_text("dummy")
    arr = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    monkeypatch.setattr(conv, "read_las", lambda p: arr)

    result = conv.ensure_xyz(base, ("laslike", las_path))

    assert result.exists()
    assert np.allclose(np.loadtxt(result), arr)


def test_ensure_xyz_ply_conversion(monkeypatch, tmp_path: Path) -> None:
    """Convert PLY input to XYZ using mocked reader."""
    base = tmp_path / "model"
    ply_path = base.with_suffix(".ply")
    ply_path.write_text("dummy")
    arr = np.array([[1.0, 0.0, 0.0]])

    monkeypatch.setattr(conv, "read_ply", lambda p: arr)

    result = conv.ensure_xyz(base, ("ply", ply_path))

    assert result.exists()
    assert np.allclose(np.loadtxt(result), arr)


def test_ensure_xyz_obj_conversion(monkeypatch, tmp_path: Path) -> None:
    """Convert OBJ input to XYZ using mocked reader."""
    base = tmp_path / "mesh"
    obj_path = base.with_suffix(".obj")
    obj_path.write_text("dummy")
    arr = np.array([[1.0, 2.0, 3.0]])

    monkeypatch.setattr(conv, "read_obj", lambda p: arr)

    result = conv.ensure_xyz(base, ("obj", obj_path))

    assert result.exists()
    assert np.allclose(np.loadtxt(result), arr)


def test_ensure_xyz_ply_missing_dependency(monkeypatch, tmp_path: Path) -> None:
    """Raise ``RuntimeError`` when ``plyfile`` is unavailable."""
    base = tmp_path / "missing"
    ply_path = base.with_suffix(".ply")
    ply_path.write_text("dummy")

    monkeypatch.setattr(conv, "PlyData", None)
    monkeypatch.setattr(conv, "read_ply", lambda p: np.zeros((0, 3)))

    with pytest.raises(RuntimeError):
        conv.ensure_xyz(base, ("ply", ply_path))


def test_ensure_xyz_missing_file(tmp_path: Path) -> None:
    """Raise ``FileNotFoundError`` when no input file exists."""
    base = tmp_path / "nowhere"

    with pytest.raises(FileNotFoundError):
        conv.ensure_xyz(base, (None, None))
