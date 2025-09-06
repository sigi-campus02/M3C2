"""Helpers for exporting point clouds to PLY files.

This module contains standalone helper functions kept free of class wrappers,
making imports lightweight and avoiding optional dependencies unless required.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

# ``plyfile`` is an optional dependency.  Importing it lazily allows this module
# to be used in environments where the package is not installed.
try:  # pragma: no cover - exercised in tests via monkeypatch
    from plyfile import PlyData, PlyElement  # type: ignore
except ImportError:  # pragma: no cover - informative logging only
    PlyData = None
    PlyElement = None
    logger.info("Missing optional dependency 'plyfile'.")


def export_xyz_distance(
    points: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
    outply: str,
    binary: bool = True,
) -> None:
    """Export point coordinates with a distance scalar to a PLY file.

    Parameters
    ----------
    points:
        Array of shape ``(N, 3)`` containing point coordinates.
    distances:
        1-D array of length ``N`` with distance values.
    mask:
        1-D array of length ``N`` with uint8 flags marking outliers (1) and
        inliers (0).
    outply:
        Destination path of the PLY file.
    binary:
        If ``True`` the file is written in binary format; otherwise ASCII.

    Raises
    ------
    RuntimeError
        If the :mod:`plyfile` dependency is not installed.
    ValueError
        If the input arrays do not have compatible shapes.
    """

    if PlyData is None or PlyElement is None:
        raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("'points' muss die Form (N, 3) haben")

    if distances.ndim != 1 or distances.shape[0] != points.shape[0]:
        raise ValueError("'distances' muss die Länge von 'points' haben")

    if mask.ndim != 1 or mask.shape[0] != points.shape[0]:
        raise ValueError("'mask' muss die Länge von 'points' haben")

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("distance", "f4"),
        ("mask", "u1"),
    ]

    vertex = np.empty(points.shape[0], dtype=dtype)
    vertex["x"] = points[:, 0].astype(np.float32)
    vertex["y"] = points[:, 1].astype(np.float32)
    vertex["z"] = points[:, 2].astype(np.float32)
    vertex["distance"] = distances.astype(np.float32)
    vertex["mask"] = mask.astype(np.uint8)

    el = PlyElement.describe(vertex, "vertex")

    d = os.path.dirname(outply)
    if d:
        os.makedirs(d, exist_ok=True)

    PlyData([el], text=not binary).write(outply)


__all__ = [
    "export_xyz_distance",
]

