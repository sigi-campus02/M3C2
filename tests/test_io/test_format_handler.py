"""Tests for the :mod:`m3c2.io.format_handler` module."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
import m3c2.io.format_handler as fh


def test_read_xyz(tmp_path: Path) -> None:
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = tmp_path / "points.xyz"
    np.savetxt(path, data, fmt="%.1f")

    arr = fh.read_xyz(path)

    assert np.allclose(arr, data)


def test_read_ply(tmp_path: Path) -> None:
    path = tmp_path / "points.ply"
    path.write_text(
        """ply
format ascii 1.0
element vertex 2
property float x
property float y
property float z
end_header
1 2 3
4 5 6
"""
    )

    arr = fh.read_ply(path)
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert np.allclose(arr, expected)


def test_read_obj(tmp_path: Path) -> None:
    path = tmp_path / "mesh.obj"
    path.write_text("""v 1 2 3\nv 4 5 6\n""")

    arr = fh.read_obj(path)
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert np.allclose(arr, expected)


def test_read_las_missing_dependency(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "dummy.las"
    path.write_text("dummy")

    def fake_import(name: str) -> None:
        raise ImportError

    monkeypatch.setattr(fh, "import_module", fake_import)

    with pytest.raises(RuntimeError):
        fh.read_las(path)
