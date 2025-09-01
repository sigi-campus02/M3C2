import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest


class DummyPointCloud:
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
    missing_path = tmp_path / "missing.ply"
    with pytest.raises(FileNotFoundError):
        read_ply(missing_path)


def test_read_ply_empty(tmp_path, monkeypatch):
    empty_path = tmp_path / "empty.ply"
    empty_path.touch()

    monkeypatch.setattr(
        dummy_o3d.io, "read_point_cloud", lambda p: DummyPointCloud(empty=True)
    )

    with pytest.raises(RuntimeError):
        read_ply(empty_path)


def test_round_trip_transformations():
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
    monkeypatch.setattr("m3c2.core.boundingbox_service.o3d", MagicMock())
    with pytest.raises(ValueError):
        clip_obbf_aligned_many(["a.ply", "b.ply"], ["out.ply"])
