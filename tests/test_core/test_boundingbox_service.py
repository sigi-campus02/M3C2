"""Test suite for the bounding box service.

These tests exercise helper utilities used for reading point clouds,
frame transformations, and bounding box clipping.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

class DummyPointCloud:
    """Lightweight stand-in for an Open3D ``PointCloud``.

    Parameters
    ----------
    empty : bool, optional
        Whether the point cloud should behave as empty. Defaults to ``True``.
    """

    def __init__(self, empty: bool = True):
        self._empty = empty
        self.points = []
        self.colors = []
        self.normals = []

    def is_empty(self):
        return self._empty

    def has_colors(self):
        return bool(self.colors)

    def has_normals(self):
        return bool(self.normals)

    def get_oriented_bounding_box(self):
        class OBB:
            center = np.zeros(3)
            R = np.eye(3)

        return OBB()


def _dummy_read_point_cloud(path):
    """Generate a dummy point cloud for the given file.

    Parameters
    ----------
    path : path-like
        Path to the PLY file.

    Returns
    -------
    DummyPointCloud
        Placeholder point cloud instance.
    """

    return DummyPointCloud()


dummy_o3d = SimpleNamespace(
    geometry=SimpleNamespace(PointCloud=DummyPointCloud),
    io=SimpleNamespace(read_point_cloud=_dummy_read_point_cloud, write_point_cloud=lambda *args, **kwargs: None),
    utility=SimpleNamespace(Vector3dVector=lambda x: x),
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.modules.setdefault("open3d", dummy_o3d)

from m3c2.core.boundingbox_service import (
    read_ply,
    to_local_frame,
    to_world_frame,
    clip_obbf_aligned_many,
)


def test_read_ply_missing(tmp_path):
    """Ensure ``read_ply`` raises when the source file is absent.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by ``pytest``.
    """

    missing_path = tmp_path / "missing.ply"
    with pytest.raises(FileNotFoundError):
        read_ply(missing_path)


def test_read_ply_empty(tmp_path, monkeypatch):
    """Check that reading an empty cloud triggers a ``RuntimeError``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory in which an empty file is created.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to override the point-cloud reader.
    """

    empty_path = tmp_path / "empty.ply"
    empty_path.touch()

    monkeypatch.setattr(
        dummy_o3d.io, "read_point_cloud", lambda p: DummyPointCloud(empty=True)
    )

    with pytest.raises(RuntimeError):
        read_ply(empty_path)


def test_round_trip_transformations():
    """Verify that frame transformations are inverse operations.

    Random points are rotated and translated to a local frame and back to
    world coordinates, ensuring the round-trip recovers the original data.
    """

    rng = np.random.default_rng(42)
    xyz = rng.random((10, 3))
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    C = rng.random(3)

    local = to_local_frame(xyz, Q, C)
    world = to_world_frame(local, Q, C)

    assert np.allclose(world, xyz)


def test_clip_obbf_aligned_many_mismatched_input_lengths(monkeypatch):
    """Confirm mismatched path lists raise a ``ValueError``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the ``open3d`` module dependency.
    """

    monkeypatch.setattr("m3c2.core.boundingbox_service.o3d", MagicMock())
    with pytest.raises(ValueError):
        clip_obbf_aligned_many(["a.ply", "b.ply"], ["out.ply"])
