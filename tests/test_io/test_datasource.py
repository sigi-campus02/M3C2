"""Tests for the :mod:`datasource.datasource` module."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import m3c2.importer.datasource as ds_module
from m3c2.config.datasource_config import DataSourceConfig
from m3c2.importer.file_detection import detect
from m3c2.importer.converters import ensure_xyz


class DummyEpoch:
    """Simple stand-in for :class:`py4dgeo.Epoch`."""

    def __init__(self, arr: np.ndarray) -> None:
        self.cloud = arr


class DummyPy4DGeo:
    """Minimal replacement for :mod:`py4dgeo` used in tests.

    The real :mod:`py4dgeo` package provides various readers for different
    point cloud formats.  This dummy implementation emulates the parts of the
    API that :class:`~m3c2.io.datasource.DataSource` interacts with and records
    which reader was invoked.  The recorded information allows the tests to
    assert that the correct loading routine was selected.
    """

    last_call: str | None = None

    @staticmethod
    def read_from_xyz(m_path: str, r_path: str):  # type: ignore[override]
        DummyPy4DGeo.last_call = "xyz"
        comparison_arr = np.loadtxt(m_path)
        reference_arr = np.loadtxt(r_path)
        return DummyEpoch(comparison_arr), DummyEpoch(reference_arr)

    @staticmethod
    def read_from_ply(m_path: str, r_path: str):  # type: ignore[override]
        DummyPy4DGeo.last_call = "ply"
        comparison_arr = np.array([[0.0, 0.0, 0.0]])
        reference_arr = np.array([[1.0, 1.0, 1.0]])
        return DummyEpoch(comparison_arr), DummyEpoch(reference_arr)


def test_detect_xyz(tmp_path: Path) -> None:
    """Check that XYZ files are detected correctly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by ``pytest``.

    Returns
    -------
    None
        This test only asserts behaviour and does not return anything.

    Examples
    --------
    >>> folder = tmp_path / 'data'
    >>> folder.mkdir()
    >>> (folder / 'comparison.xyz').write_text('0 0 0\n')
    >>> detect(folder / 'comparison')[0]
    'xyz'
    """

    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "comparison.xyz").write_text("0 0 0\n")

    kind, path = detect(folder / "comparison")

    assert kind == "xyz"
    assert path == folder / "comparison.xyz"


def test_ensure_xyz_from_gpc(tmp_path: Path) -> None:
    """Ensure that GPC files are converted to XYZ format.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory containing the GPC file.

    Returns
    -------
    None
        The test verifies file conversion without returning anything.

    Examples
    --------
    >>> gpc_path = tmp_path / 'comparison.gpc'
    >>> gpc_path.write_text('1 2 3\n4 5 6\n')
    >>> xyz_path = ensure_xyz(tmp_path / 'comparison', ('gpc', gpc_path))
    >>> xyz_path.suffix
    '.xyz'
    """

    gpc_path = tmp_path / "comparison.gpc"
    gpc_path.write_text("1 2 3\n4 5 6\n")
    base = tmp_path / "comparison"

    xyz_path = ensure_xyz(base, ("gpc", gpc_path))

    assert xyz_path.exists()
    data = np.loadtxt(xyz_path)
    assert data.shape == (2, 3)


def test_load_points_xyz(tmp_path: Path, monkeypatch) -> None:
    """Load points from XYZ files using the data source.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory with the mock point clouds.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace :mod:`py4dgeo` with a dummy implementation.

    Returns
    -------
    None
        The test asserts correct behaviour and returns nothing.

    Examples
    --------
    >>> comparison = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    >>> reference = np.array([[0, 0, 0], [2, 2, 2]], dtype=float)
    >>> np.savetxt(tmp_path / 'comparison.xyz', comparison, fmt='%.6f')
    >>> np.savetxt(tmp_path / 'reference.xyz', reference, fmt='%.6f')
    >>> monkeypatch.setattr(ds_module, 'py4dgeo', DummyPy4DGeo)
    >>> cfg = DataSourceConfig(str(tmp_path))
    >>> ds = ds_module.DataSource(cfg)
    >>> ds.load_points()[0].cloud.shape
    (2, 3)
    """

    comparison = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    reference = np.array([[0, 0, 0], [2, 2, 2]], dtype=float)
    np.savetxt(tmp_path / "comparison.xyz", comparison, fmt="%.6f")
    np.savetxt(tmp_path / "reference.xyz", reference, fmt="%.6f")

    DummyPy4DGeo.last_call = None
    monkeypatch.setattr(ds_module, "py4dgeo", DummyPy4DGeo)

    cfg = DataSourceConfig(str(tmp_path))
    ds = ds_module.DataSource(cfg)
    comparison_epoch, reference_epoch, corepoints = ds.load_points()

    assert np.allclose(comparison_epoch.cloud, comparison)
    assert np.allclose(reference_epoch.cloud, reference)
    assert np.allclose(corepoints, reference)
    assert DummyPy4DGeo.last_call == "xyz"


def test_load_points_ply(tmp_path: Path, monkeypatch) -> None:
    """Load points from PLY files using the data source.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory containing dummy PLY files.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace :mod:`py4dgeo` with a dummy implementation.

    Returns
    -------
    None
        This test only verifies behaviour and returns nothing.

    Examples
    --------
    >>> (tmp_path / 'comparison.ply').write_text('dummy')
    >>> (tmp_path / 'reference.ply').write_text('dummy')
    >>> monkeypatch.setattr(ds_module, 'py4dgeo', DummyPy4DGeo)
    >>> cfg = DataSourceConfig(str(tmp_path))
    >>> ds = ds_module.DataSource(cfg)
    >>> ds.load_points()[2].shape
    (1, 3)
    """

    (tmp_path / "comparison.ply").write_text("dummy")
    (tmp_path / "reference.ply").write_text("dummy")

    DummyPy4DGeo.last_call = None
    monkeypatch.setattr(ds_module, "py4dgeo", DummyPy4DGeo)

    cfg = DataSourceConfig(str(tmp_path))
    ds = ds_module.DataSource(cfg)
    comparison_epoch, reference_epoch, corepoints = ds.load_points()

    assert DummyPy4DGeo.last_call == "ply"
    assert np.allclose(corepoints, reference_epoch.cloud)
