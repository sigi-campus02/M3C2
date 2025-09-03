"""Minimal bounding box utilities for testing purposes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import open3d as o3d


def read_ply(path: Path | str):
    """Read a PLY file ensuring it contains data."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    cloud = o3d.io.read_point_cloud(str(p))
    if cloud.is_empty():
        raise RuntimeError("Point cloud is empty")
    return cloud


def to_local_frame(xyz: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Transform coordinates to a local reference frame."""

    return (xyz - C) @ R


def to_world_frame(local: np.ndarray, R: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Transform coordinates from a local frame back to world coordinates."""

    return local @ R.T + C


def clip_obbf_aligned_many(in_paths: Iterable[str], out_paths: Iterable[str]) -> List[str]:
    """Placeholder that validates input lengths.

    The function merely checks that the two iterables contain the same number of
    elements and returns an empty list.  It serves as a lightweight stand-in for
    the real implementation used in the tests.
    """

    in_paths = list(in_paths)
    out_paths = list(out_paths)
    if len(in_paths) != len(out_paths):
        raise ValueError("Mismatched number of input and output paths")
    return []


__all__ = [
    "read_ply",
    "to_local_frame",
    "to_world_frame",
    "clip_obbf_aligned_many",
]

