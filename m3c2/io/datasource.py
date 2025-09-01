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
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import py4dgeo
from plyfile import PlyData
import laspy

from m3c2.config.datasource_config import DataSourceConfig
from .format_handler import read_ply, read_las, read_obj, read_gpc


@dataclass
class DataSource:
    """Class for loading point cloud pairs from various file formats."""

    config: DataSourceConfig

    # ------------------------------------------------------------------
    # path helpers
    @property
    def mov_base(self) -> Path:
        return Path(self.config.folder) / self.config.mov_basename

    @property
    def ref_base(self) -> Path:
        return Path(self.config.folder) / self.config.ref_basename

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
    def load_points(self) -> Tuple[object, object, np.ndarray]:
        """Load the moving and reference epochs and derive core points."""

        if py4dgeo is None:
            raise RuntimeError("'py4dgeo' ist nicht installiert.")

        mov, ref = self._load_epochs()
        corepoints = self._derive_corepoints(mov, ref)

        if not isinstance(corepoints, np.ndarray):
            raise TypeError("Unerwarteter Typ für corepoints; erwarte np.ndarray (Nx3).")

        return mov, ref, corepoints

    # ------------------------------------------------------------------
    # internal helpers
    def _load_epochs(self) -> Tuple[object, object]:
        """Detect input types and read epochs using :mod:`py4dgeo`."""

        m_kind, m_path = self._detect(self.mov_base)
        r_kind, r_path = self._detect(self.ref_base)

        if m_kind == r_kind == "xyz":
            logging.info("Nutze py4dgeo.read_from_xyz")
            
            return py4dgeo.read_from_xyz(str(m_path), str(r_path))

        if m_kind == r_kind == "laslike":
            logging.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            return py4dgeo.read_from_las(str(m_path), str(r_path))

        if m_kind == r_kind == "ply" and hasattr(py4dgeo, "read_from_ply"):
            logging.info("Nutze py4dgeo.read_from_ply")
            return py4dgeo.read_from_ply(str(m_path), str(r_path))

        m_xyz = self._ensure_xyz(self.mov_base, (m_kind, m_path))
        r_xyz = self._ensure_xyz(self.ref_base, (r_kind, r_path))
        logging.info("Mischtypen → konvertiert zu XYZ → py4dgeo.read_from_xyz")
        return py4dgeo.read_from_xyz(str(m_xyz), str(r_xyz))

    def _derive_corepoints(self, mov: object, ref: object) -> np.ndarray:
        """Derive core points from configured epoch with optional subsampling."""

        use_mov = self.config.mov_as_corepoints
        label = "mov" if use_mov else "ref"
        logging.info(
            "Nutze %s als Corepoints und nutze Subsamling: %s",
            label,
            self.config.use_subsampled_corepoints,
        )

        source = mov if use_mov else ref
        data = source.cloud if hasattr(source, "cloud") else source
        return data[:: self.config.use_subsampled_corepoints]


__all__ = ["DataSource"]

