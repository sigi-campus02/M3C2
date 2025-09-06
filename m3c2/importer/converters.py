"""Utilities for converting point cloud formats to XYZ."""

from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
from plyfile import PlyData

from .format_handler import read_ply, read_las, read_obj, read_gpc

logger = logging.getLogger(__name__)


def ensure_xyz(base: Path, detected: tuple[str | None, Path | None]) -> Path:
    """Ensure that an ``.xyz`` file exists for ``base``.

    Parameters
    ----------
    base:
        Path without extension specifying the desired output location.
    detected:
        ``(kind, path)`` tuple as returned by :func:`detect`.

    Returns
    -------
    pathlib.Path
        Path to the resulting ``.xyz`` file.

    Raises
    ------
    FileNotFoundError
        If no supported file for ``base`` exists.
    RuntimeError
        If a required optional dependency is missing during conversion.
    """

    kind, path = detected
    xyz = base.with_suffix(".xyz")
    logger.debug("Ensuring XYZ for %s: kind=%s, path=%s", base, kind, path)

    if kind == "xyz" and path:
        logger.debug("XYZ already present at %s", path)
        return path

    if kind == "laslike" and path:
        logger.info("[%s] Konvertiere LAS/LAZ -> XYZ ...", base)
        try:
            arr = read_las(path)
            np.savetxt(xyz, arr, fmt="%.6f")
        except (OSError, ValueError) as exc:
            logger.exception("LAS/LAZ conversion failed for %s", path)
            raise exc
        return xyz

    if kind == "ply" and path:
        if PlyData is None:
            logger.error("PLY gefunden, aber 'plyfile' ist nicht installiert.")
            raise RuntimeError("PLY gefunden, aber 'plyfile' ist nicht installiert.")
        logger.info("[%s] Konvertiere PLY -> XYZ ...", base)
        try:
            arr = read_ply(path)
            np.savetxt(xyz, arr, fmt="%.6f")
        except (OSError, ValueError) as exc:
            logger.exception("PLY conversion failed for %s", path)
            raise exc
        return xyz

    if kind == "obj" and path:
        logger.info("[%s] Konvertiere OBJ -> XYZ ...", base)
        try:
            arr = read_obj(path)
            np.savetxt(xyz, arr, fmt="%.6f")
        except (OSError, ValueError) as exc:
            logger.exception("OBJ conversion failed for %s", path)
            raise exc
        return xyz

    if kind == "gpc" and path:
        logger.info("[%s] Konvertiere GPC -> XYZ ...", base)
        try:
            arr = read_gpc(path)
            np.savetxt(xyz, arr, fmt="%.6f")
        except (OSError, ValueError) as exc:
            logger.exception("GPC conversion failed for %s", path)
            raise exc
        return xyz

    logger.error("No suitable file found for conversion: %s", base)
    raise FileNotFoundError(f"Fehlt: {base}.xyz/.las/.laz/.ply/.obj/.gpc")


__all__ = ["ensure_xyz"]
