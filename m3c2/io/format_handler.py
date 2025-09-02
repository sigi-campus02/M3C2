"""File format handlers for point cloud data.

This module provides reader functions for a variety of point cloud file
formats, including ``XYZ``, ``LAS/LAZ``, ``PLY``, ``OBJ``, and ``GPC``.
"""

from __future__ import annotations

from pathlib import Path
from importlib import import_module
import logging
from typing import Callable, Dict
from plyfile import PlyData

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reader functions

def read_xyz(path: Path) -> np.ndarray:
    """Read a plain text ``XYZ`` file and return its coordinates.

    Parameters
    ----------
    path:
        Path to the ``.xyz`` file to read.

    Raises
    ------
    OSError
        If the file cannot be accessed. The error is logged and re-raised.
    ValueError
        If the file contents cannot be parsed into ``float`` values.
        The error is logged and re-raised.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` containing ``x``, ``y`` and ``z`` coordinates
        of the point cloud.
    """
    logger.info("Reading XYZ file %s", path)
    try:
        return np.loadtxt(path, dtype=np.float64, usecols=(0, 1, 2))
    except (OSError, ValueError):
        logger.exception("Failed to read XYZ file %s", path)
        raise


def read_las(path: Path) -> np.ndarray:
    """Read a ``.las`` or ``.laz`` file and return its point cloud.

    LAS/LAZ reading requires the :mod:`laspy` package.  For compressed
    ``.laz`` files the optional ``lazrs`` dependency of ``laspy`` must be
    installed as well.

    Parameters
    ----------
    path:
        Path to the LAS/LAZ file to read.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` containing the ``x``, ``y`` and ``z``
        coordinates of the point cloud.
    """
    logger.info("Reading LAS/LAZ file %s", path)
    try:
        laspy = import_module("laspy")
    except Exception as exc:  # pragma: no cover - dependency issue
        logger.exception("LAS/LAZ found, but 'laspy' is not installed.")
        raise RuntimeError("LAS/LAZ found, but 'laspy' is not installed.") from exc

    try:
        las = laspy.read(str(path))
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency issue
        logger.exception("LAZ detected but 'laspy[lazrs]' is missing.")
        raise RuntimeError(
            "LAZ detected, please install 'laspy[lazrs]'.",
        ) from exc
    except Exception:
        logger.exception("Failed to read LAS/LAZ file %s", path)
        raise
    return np.vstack([las.x, las.y, las.z]).T.astype(np.float64)


def read_ply(path: Path) -> np.ndarray:
    """Read a PLY file and return its point cloud.

    Parameters
    ----------
    path:
        Path to the ``.ply`` file to read.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` containing ``x``, ``y`` and ``z`` coordinates
        of the point cloud.
    """
    logger.info("Reading PLY file %s", path)
    try:
        ply = PlyData.read(str(path))
    except (OSError, ValueError):
        logger.exception("Failed to read PLY file %s", path)
        raise
    v = ply["vertex"]
    return np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)


def read_obj(path: Path) -> np.ndarray:
    """Parse an OBJ file and extract vertex coordinates.

    Parameters
    ----------
    path:
        Path to the ``.obj`` file to read.

    Notes
    -----
    Lines beginning with ``"v "`` are interpreted as vertex definitions. If a
    vertex line does not contain three coordinate components, it is skipped and
    a warning is logged.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` containing the ``x``, ``y`` and ``z``
        coordinates of all parsed vertices.
    """
    logger.info("Reading OBJ file %s", path)
    vertices: list[list[float]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    else:
                        logger.warning(
                            "Malformed vertex line %d in %s: %s",
                            line_number,
                            path,
                            line,
                        )
    except Exception:
        logger.exception("Failed to read OBJ file %s", path)
        raise
    return np.asarray(vertices, dtype=np.float64)


def read_gpc(path: Path) -> np.ndarray:
    """Read point coordinates from a GPC file.

    The *GPC* (Geometrie-Punktwolke) format is a simple whitespace separated
    text format where each line stores at least three floating point values:
    ``x``, ``y`` and ``z``.  Additional columns are ignored.

    Parameters
    ----------
    path : pathlib.Path
        Path to the ``.gpc`` file that should be read.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` containing the point coordinates.
    """

    logger.info("Reading GPC file %s", path)
    try:
        return np.loadtxt(path, dtype=np.float64, usecols=(0, 1, 2))
    except Exception:
        logger.exception("Failed to read GPC file %s", path)
        raise


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
