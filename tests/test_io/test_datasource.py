"""Tests for the :mod:`datasource.datasource` module."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import m3c2.io.datasource as ds_module
from m3c2.config.datasource_config import DataSourceConfig


class DummyEpoch:
    """Simple stand-in for :class:`py4dgeo.Epoch`."""

    def __init__(self, arr: np.ndarray) -> None:
        self.cloud = arr


class DummyPy4DGeo:
    @staticmethod
    def read_from_xyz(m_path: str, r_path: str):  # type: ignore[override]
        mov_arr = np.loadtxt(m_path)
        ref_arr = np.loadtxt(r_path)
        return DummyEpoch(mov_arr), DummyEpoch(ref_arr)


def test_detect_xyz(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "mov.xyz").write_text("0 0 0\n")

    cfg = DataSourceConfig(str(folder))
    ds = ds_module.DataSource(cfg)
    kind, path = ds._detect(ds.mov_base)

    assert kind == "xyz"
    assert path == folder / "mov.xyz"


def test_ensure_xyz_from_gpc(tmp_path: Path) -> None:
    gpc_path = tmp_path / "mov.gpc"
    gpc_path.write_text("1 2 3\n4 5 6\n")
    cfg = DataSourceConfig(str(tmp_path))
    ds = ds_module.DataSource(cfg)

    xyz_path = ds._ensure_xyz(ds.mov_base, ("gpc", gpc_path))

    assert xyz_path.exists()
    data = np.loadtxt(xyz_path)
    assert data.shape == (2, 3)


def test_load_points_xyz(tmp_path: Path, monkeypatch) -> None:
    mov = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    ref = np.array([[0, 0, 0], [2, 2, 2]], dtype=float)
    np.savetxt(tmp_path / "mov.xyz", mov, fmt="%.6f")
    np.savetxt(tmp_path / "ref.xyz", ref, fmt="%.6f")

    monkeypatch.setattr(ds_module, "py4dgeo", DummyPy4DGeo)

    cfg = DataSourceConfig(str(tmp_path))
    ds = ds_module.DataSource(cfg)
    mov_epoch, ref_epoch, corepoints = ds.load_points()

    assert np.allclose(mov_epoch.cloud, mov)
    assert np.allclose(ref_epoch.cloud, ref)
    assert np.allclose(corepoints, mov)
