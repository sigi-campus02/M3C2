from __future__ import annotations

"""File format handlers for point cloud data."""

from pathlib import Path
from importlib import import_module
from typing import Callable, Dict
from plyfile import PlyData

import numpy as np

# ---------------------------------------------------------------------------
# Reader functions

def read_xyz(path: Path) -> np.ndarray:
    """Read XYZ formatted files using ``numpy``."""
    return np.loadtxt(path, dtype=np.float64, usecols=(0, 1, 2))


def read_las(path: Path) -> np.ndarray:
    """Read LAS/LAZ files using :mod:`laspy` lazily."""
    try:
        laspy = import_module("laspy")
    except Exception as exc:  # pragma: no cover - dependency issue
        raise RuntimeError("LAS/LAZ found, but 'laspy' is not installed.") from exc

    try:
        las = laspy.read(str(path))
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency issue
        raise RuntimeError(
            "LAZ detected, please install 'laspy[lazrs]'."
        ) from exc
    return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)


def read_ply(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)


def read_obj(path: Path) -> np.ndarray:
    """Parse an OBJ file and extract vertex coordinates."""
    vertices: list[list[float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=np.float64)


def read_gpc(path: Path) -> np.ndarray:
    """Read GPC files as plain text."""
    return np.loadtxt(path, dtype=np.float64, usecols=(0, 1, 2))


# Mapping from extension (without dot) to reader function
FORMAT_READERS: Dict[str, Callable[[Path], np.ndarray]] = {
    "xyz": read_xyz,
    "las": read_las,
    "laz": read_las,
    "ply": read_ply,
    "obj": read_obj,
    "gpc": read_gpc,
}

__all__ = ["FORMAT_READERS", "DependencyError"]