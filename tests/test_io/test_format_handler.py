"""Tests for the :mod:`m3c2.io.format_handler` module."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
import m3c2.io.format_handler as fh


def test_read_xyz(tmp_path: Path) -> None:
    """Ensure that ``read_xyz`` correctly parses simple XYZ files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by ``pytest``.

    Returns
    -------
    None
        The test passes if the read array matches the saved data.

    Examples
    --------
    >>> data = np.array([[1.0, 2.0, 3.0]])
    >>> path = tmp_path / "points.xyz"
    >>> np.savetxt(path, data, fmt="%.1f")
    >>> fh.read_xyz(path)
    array([[1., 2., 3.]])
    """
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = tmp_path / "points.xyz"
    np.savetxt(path, data, fmt="%.1f")

    arr = fh.read_xyz(path)

    assert np.allclose(arr, data)


def test_read_ply(tmp_path: Path) -> None:
    """Verify reading of ASCII PLY point clouds.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory supplied by ``pytest``.

    Returns
    -------
    None
        The test passes when the parsed vertices equal the expected array.

    Examples
    --------
    >>> path = tmp_path / "points.ply"
    >>> _ = path.write_text("ply\nformat ascii 1.0\n...")
    >>> fh.read_ply(path)
    array([[1., 2., 3.],
           [4., 5., 6.]])
    """
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
    """Ensure OBJ vertex data are loaded into a NumPy array.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory fixture from ``pytest``.

    Returns
    -------
    None
        The test passes when the returned array matches the expected values.

    Examples
    --------
    >>> path = tmp_path / "mesh.obj"
    >>> _ = path.write_text("v 1 2 3\nv 4 5 6\n")
    >>> fh.read_obj(path)
    array([[1., 2., 3.],
           [4., 5., 6.]])
    """
    path = tmp_path / "mesh.obj"
    path.write_text("""v 1 2 3\nv 4 5 6\n""")

    arr = fh.read_obj(path)
    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert np.allclose(arr, expected)


def test_read_las_missing_dependency(monkeypatch, tmp_path: Path) -> None:
    """Raise ``RuntimeError`` when LAS reading dependencies are absent.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to simulate an unavailable ``laspy`` module.
    tmp_path : pathlib.Path
        Temporary directory where a dummy LAS file is stored.

    Returns
    -------
    None
        The test succeeds if :func:`m3c2.io.format_handler.read_las` raises
        :class:`RuntimeError`.

    Examples
    --------
    >>> monkeypatch.setattr(fh, "import_module", lambda name: (_ for _ in ()).throw(ImportError))
    >>> with pytest.raises(RuntimeError):
    ...     fh.read_las(tmp_path / "dummy.las")
    """
    path = tmp_path / "dummy.las"
    path.write_text("dummy")

    def fake_import(name: str) -> None:
        raise ImportError

    monkeypatch.setattr(fh, "import_module", fake_import)

    with pytest.raises(RuntimeError):
        fh.read_las(path)
