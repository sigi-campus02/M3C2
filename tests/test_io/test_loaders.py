"""Tests for simple point cloud loaders."""

from __future__ import annotations


from pathlib import Path
import io
import sys

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from m3c2.importer.loaders.las import LASLoader
from m3c2.importer.loaders.ply import PLYLoader
from m3c2.importer.loaders.xyz import XYZLoader


class DummyBackend:
    """Minimal backend emulating :mod:`py4dgeo` readers."""

    def read_from_xyz(self, path: str, ref_path: str | None = None):
        """Read arrays from plain XYZ text files."""
        first = np.loadtxt(path)
        if ref_path is None:
            return first
        second = np.loadtxt(ref_path)
        return first, second

    def read_from_las(self, path: str, ref_path: str | None = None):
        """Pretend to read LAS files by parsing whitespace separated floats."""
        first = np.loadtxt(path)
        if ref_path is None:
            return first
        second = np.loadtxt(ref_path)
        return first, second

    def _read_ply(self, path: str) -> np.ndarray:
        """Read a minimal ASCII PLY file."""
        with open(path, "r", encoding="utf8") as f:
            lines = f.readlines()
        start = None
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                start = i + 1
                break
        if start is None:
            raise ValueError("no PLY header")
        data = "".join(lines[start:])
        return np.loadtxt(io.StringIO(data))

    def read_from_ply(self, path: str, ref_path: str | None = None):
        """Read arrays from minimal PLY files."""
        first = self._read_ply(path)
        if ref_path is None:
            return first
        second = self._read_ply(ref_path)
        return first, second


def _write_ply(path: Path, data: np.ndarray) -> None:
    """Write *data* to a simple ASCII PLY file at *path*."""
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(data)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    lines.extend(" ".join(map(str, row)) for row in data)
    path.write_text("\n".join(lines))


def test_xyz_loader(tmp_path: Path) -> None:
    """Load XYZ files and return arrays."""
    backend = DummyBackend()
    loader = XYZLoader(backend)

    data1 = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
    data2 = np.array([[5.0, 4.0, 3.0], [2.0, 1.0, 0.0]])
    path1 = tmp_path / "a.xyz"
    path2 = tmp_path / "b.xyz"
    np.savetxt(path1, data1)
    np.savetxt(path2, data2)

    arr = loader.load_single(path1)
    assert np.allclose(arr, data1)

    arr1, arr2 = loader.load_pair(path1, path2)
    assert np.allclose(arr1, data1)
    assert np.allclose(arr2, data2)


def test_xyz_loader_invalid(tmp_path: Path) -> None:
    """Raise ``ValueError`` on invalid XYZ files."""
    backend = DummyBackend()
    loader = XYZLoader(backend)
    bad = tmp_path / "bad.xyz"
    bad.write_text("invalid")
    with pytest.raises(ValueError):
        loader.load_single(bad)


def test_las_loader(tmp_path: Path) -> None:
    """Load LAS files via :class:`LASLoader`."""
    backend = DummyBackend()
    loader = LASLoader(backend)
    data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    path = tmp_path / "points.las"
    ref = tmp_path / "ref.las"
    np.savetxt(path, data)
    np.savetxt(ref, data + 1)

    arr = loader.load_single(path)
    assert np.allclose(arr, data)

    arr1, arr2 = loader.load_pair(path, ref)
    assert np.allclose(arr1, data)
    assert np.allclose(arr2, data + 1)


def test_las_loader_invalid(tmp_path: Path) -> None:
    """Propagate parsing errors for LAS files."""
    backend = DummyBackend()
    loader = LASLoader(backend)
    bad = tmp_path / "bad.las"
    bad.write_text("not a las file")
    with pytest.raises(ValueError):
        loader.load_single(bad)


def test_ply_loader(tmp_path: Path) -> None:
    """Load PLY files and verify array shapes."""
    backend = DummyBackend()
    loader = PLYLoader(backend)

    data = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    data2 = data + 1
    path1 = tmp_path / "points.ply"
    path2 = tmp_path / "ref.ply"
    _write_ply(path1, data)
    _write_ply(path2, data2)

    arr = loader.load_single(path1)
    assert np.allclose(arr, data)

    arr1, arr2 = loader.load_pair(path1, path2)
    assert np.allclose(arr1, data)
    assert np.allclose(arr2, data2)


def test_ply_loader_invalid(tmp_path: Path) -> None:
    """Invalid PLY content raises ``ValueError``."""
    backend = DummyBackend()
    loader = PLYLoader(backend)
    bad = tmp_path / "bad.ply"
    bad.write_text("ply\nformat ascii 1.0\nend_header\ninvalid\n")
    with pytest.raises(ValueError):
        loader.load_single(bad)


def test_ply_loader_missing_backend() -> None:
    """Ensure ``PLYLoader`` complains when backend lacks support."""

    class NoPly:
        pass

    with pytest.raises(RuntimeError):
        PLYLoader(NoPly())