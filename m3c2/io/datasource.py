"""Utilities for loading heterogeneous point cloud data.

This module centralises the logic for reading point cloud pairs from a number
of supported file formats and converting them to a unified XYZ representation.
Only the necessary optional dependencies are imported for the requested file
types, enabling this module to be imported even when those libraries are not
installed.  The :class:`DataSource` class is the public entry point and exposes
the :meth:`DataSource.load_points` method which returns the moving epoch, the
reference epoch and the core points as NumPy arrays.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple
import numpy as np
import py4dgeo
from plyfile import PlyData
import laspy
import config.datasource_config as cfg
from .format_handler import read_xyz, read_las, read_ply, read_obj, read_gpc


@dataclass
class DataSource:
    """Class for loading point cloud pairs from various file formats."""

    config: cfg.DataSourceConfig

    def _detect(self, base: Path) -> tuple[str | None, Path | None]:
        
        # Map each supported extension to its potential file path
        mapping = {
            "xyz": base.with_suffix(".xyz"),
            "las": base.with_suffix(".las"),
            "laz": base.with_suffix(".laz"),
            "ply": base.with_suffix(".ply"),
            "obj": base.with_suffix(".obj"),
            "gpc": base.with_suffix(".gpc"),
        }

        # Check for existing files in order of preference
        if mapping["xyz"].exists():
            return "xyz", mapping["xyz"]
        if mapping["las"].exists() or mapping["laz"].exists():
            return "laslike", mapping["las"] if mapping["las"].exists() else mapping["laz"]
        if mapping["ply"].exists():
            return "ply", mapping["ply"]
        if mapping["obj"].exists():
            return "obj", mapping["obj"]
        if mapping["gpc"].exists():
            return "gpc", mapping["gpc"]
        return None, None

    def _ensure_xyz(self, base: Path, detected: tuple[str | None, Path | None]) -> Path:
        """Ensure that an ``.xyz`` file exists for ``base``.

        Parameters
        ----------
        base:
            Path without extension specifying the desired output location.
        detected:
            ``(kind, path)`` tuple as returned by :meth:`_detect`.

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

        # If an ``.xyz`` file already exists simply return it
        if kind == "xyz" and path:
            return path

        # Convert LAS/LAZ files to ``.xyz`` using ``laspy``
        if kind == "laslike" and path:
            logging.info("[%s] Konvertiere LAS/LAZ → XYZ …", base)
            arr = read_las(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz

        # Convert PLY files to ``.xyz`` using ``plyfile``
        if kind == "ply" and path:
            if PlyData is None:
                raise RuntimeError("PLY gefunden, aber 'plyfile' ist nicht installiert.")
            logging.info("[%s] Konvertiere PLY → XYZ …", base)
            arr = read_ply(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz

        # Convert OBJ files by extracting their vertices
        if kind == "obj" and path:
            logging.info("[%s] Konvertiere OBJ → XYZ …", base)
            arr = read_obj(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz

        # Convert GPC files to ``.xyz`` using plain text loading
        if kind == "gpc" and path:
            logging.info("[%s] Konvertiere GPC → XYZ …", base)
            arr = read_gpc(path)
            np.savetxt(xyz, arr, fmt="%.6f")
            return xyz

        # No suitable file found for conversion
        raise FileNotFoundError(f"Fehlt: {base}.xyz/.las/.laz/.ply/.obj/.gpc")

    # ------------------------------------------------------------------
    # public API
    def load_points(self, config) -> Tuple[object, object, np.ndarray]:
        """Load the moving and reference epochs and derive core points.

        Returns
        -------
        tuple
            ``(mov, ref, corepoints)`` where ``mov`` and ``ref`` are the
            objects returned by :mod:`py4dgeo` and ``corepoints`` is an
            ``np.ndarray`` of shape ``(N, 3)``.

        Raises
        ------
        RuntimeError
            If :mod:`py4dgeo` is not available.
        TypeError
            If the derived core points are not an ``np.ndarray``.
        """

        if py4dgeo is None: 
            raise RuntimeError("'py4dgeo' ist nicht installiert.")

        # Determine the available file types for moving and reference epochs
        m_kind, m_path = self._detect(self.config.mov_basename)
        r_kind, r_path = self._detect(self.config.ref_basename)

        # Choose the appropriate py4dgeo reader based on detected types
        if m_kind == r_kind == "xyz":
            logging.info("Nutze py4dgeo.read_from_xyz")
            mov, ref = py4dgeo.read_from_xyz(str(m_path), str(r_path))

        elif m_kind == "laslike" and r_kind == "laslike":
            logging.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            mov, ref = py4dgeo.read_from_las(str(m_path), str(r_path))

        elif m_kind == r_kind == "ply" and hasattr(py4dgeo, "read_from_ply"):
            logging.info("Nutze py4dgeo.read_from_ply")
            mov, ref = read_ply(str(m_path), str(r_path))

        else:
            # Convert heterogeneous types to XYZ and use the generic reader
            m_xyz = self._ensure_xyz(self.config.mov_basename, (m_kind, m_path))
            r_xyz = self._ensure_xyz(self.config.ref_basename, (r_kind, r_path))

            logging.info("Mischtypen → konvertiert zu XYZ → py4dgeo.read_from_xyz")

            mov, ref = py4dgeo.read_from_xyz(str(m_xyz), str(r_xyz))

        # Extract core points from the configured epoch, optionally subsampled
        if self.config.mov_as_corepoints:
            logging.info(
                "Nutze mov als Corepoints und nutze Subsamling: %s",
                self.config.use_subsampled_corepoints,
            )
            corepoints = (
                mov.cloud[:: self.config.use_subsampled_corepoints]
                if hasattr(mov, "cloud")
                else mov
            )
        else:
            logging.info(
                "Nutze ref als Corepoints und nutze Subsamling: %s",
                self.config.use_subsampled_corepoints,
            )
            corepoints = (
                ref.cloud[:: self.config.use_subsampled_corepoints]
                if hasattr(ref, "cloud")
                else ref
            )

        if not isinstance(corepoints, np.ndarray):
            raise TypeError("Unerwarteter Typ für corepoints; erwarte np.ndarray (Nx3).")

        return mov, ref, corepoints


__all__ = ["DataSource"]

