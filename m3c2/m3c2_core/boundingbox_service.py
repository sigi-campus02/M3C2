"""Minimal bounding box utilities used in tests."""

from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np

try:  # pragma: no cover - dependency provided in tests
    import open3d as o3d
except Exception:  # pragma: no cover - replaced with dummy in tests
    o3d = None


def read_ply(path):
    """Read a point cloud from ``path`` using ``open3d``.

    Raises ``FileNotFoundError`` if the file does not exist and
    ``RuntimeError`` when the loaded cloud is empty.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if o3d is None:
        raise RuntimeError("open3d not available")
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise RuntimeError("empty cloud")
    return cloud


def to_local_frame(points: np.ndarray, R: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Transform world coordinates to a local frame."""

    return (points - center) @ R


def to_world_frame(points: np.ndarray, R: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Transform local coordinates back to world coordinates."""

    return points @ R.T + center


def clip_obbf_aligned_many(src_paths: Iterable[str], dst_paths: Iterable[str]) -> None:
    """Dummy implementation validating path list lengths."""

    src_list = list(src_paths)
    dst_list = list(dst_paths)
    if len(src_list) != len(dst_list):
        raise ValueError("Input and output list lengths must match")


__all__ = [
    "read_ply",
    "to_local_frame",
    "to_world_frame",
    "clip_obbf_aligned_many",
]

