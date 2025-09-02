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


logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Class for loading point cloud pairs from various file formats."""

    config: DataSourceConfig

    # ------------------------------------------------------------------
    # path helpers
    @property
    def mov_base(self) -> Path:
        """Base path of the moving epoch file.

        Combines the configured folder and moving filename base without
        an extension, resulting in the path used to detect the moving
        epoch's data file.
        """

        return Path(self.config.folder) / self.config.mov_basename

    @property
    def ref_base(self) -> Path:
        """Return the base path for the reference epoch.

        The path is constructed by combining the configured data folder with
        the reference file's base name. The resulting path intentionally lacks
        a file extension; downstream loaders append the appropriate suffix when
        searching for the actual reference data file.
        """
        return Path(self.config.folder) / self.config.ref_basename
    

    @property
    def singlecloud_base(self) -> Path:
        """Return the base path for the single cloud epoch.

        The path is constructed by combining the configured data folder with
        the single cloud file's base name. The resulting path intentionally lacks
        a file extension; downstream loaders append the appropriate suffix when
        searching for the actual reference data file.
        """
        return Path(self.config.folder) / self.config.filename_singlecloud


    def _detect(self, base: Path) -> tuple[str | None, Path | None]:
        """Detect available point cloud files for a given base path.

        Parameters
        ----------
        base:
            Path without an extension that serves as the candidate stem for
            supported point cloud file formats.

        Returns
        -------
        tuple[str | None, Path | None]
            A pair ``(kind, path)`` where ``kind`` identifies the detected
            format (e.g. ``"xyz"`` or ``"laslike"``) and ``path`` points to the
            discovered file. ``(None, None)`` is returned if no supported file
            exists.
        """

        logger.debug("Detecting file type for base %s", base)

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
            logger.debug("Detected XYZ file at %s", mapping["xyz"])
            return "xyz", mapping["xyz"]
        if mapping["las"].exists() or mapping["laz"].exists():
            path = mapping["las"] if mapping["las"].exists() else mapping["laz"]
            logger.debug("Detected LAS/LAZ file at %s", path)
            return "laslike", path
        if mapping["ply"].exists():
            logger.debug("Detected PLY file at %s", mapping["ply"])
            return "ply", mapping["ply"]
        if mapping["obj"].exists():
            logger.debug("Detected OBJ file at %s", mapping["obj"])
            return "obj", mapping["obj"]
        if mapping["gpc"].exists():
            logger.debug("Detected GPC file at %s", mapping["gpc"])
            return "gpc", mapping["gpc"]

        logger.debug("No supported file detected for %s", base)
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
        logger.debug("Ensuring XYZ for %s: kind=%s, path=%s", base, kind, path)

        # If an ``.xyz`` file already exists simply return it
        if kind == "xyz" and path:
            logger.debug("XYZ already present at %s", path)
            return path

        # Convert LAS/LAZ files to ``.xyz`` using ``laspy``
        if kind == "laslike" and path:
            logger.info("[%s] Konvertiere LAS/LAZ → XYZ …", base)
            try:
                arr = read_las(path)
                np.savetxt(xyz, arr, fmt="%.6f")
            except Exception as exc:
                logger.error("LAS/LAZ conversion failed for %s: %s", path, exc)
                raise
            return xyz

        # Convert PLY files to ``.xyz`` using ``plyfile``
        if kind == "ply" and path:
            if PlyData is None:
                logger.error("PLY gefunden, aber 'plyfile' ist nicht installiert.")
                raise RuntimeError("PLY gefunden, aber 'plyfile' ist nicht installiert.")
            logger.info("[%s] Konvertiere PLY → XYZ …", base)
            try:
                arr = read_ply(path)
                np.savetxt(xyz, arr, fmt="%.6f")
            except Exception as exc:
                logger.error("PLY conversion failed for %s: %s", path, exc)
                raise
            return xyz

        # Convert OBJ files by extracting their vertices
        if kind == "obj" and path:
            logger.info("[%s] Konvertiere OBJ → XYZ …", base)
            try:
                arr = read_obj(path)
                np.savetxt(xyz, arr, fmt="%.6f")
            except Exception as exc:
                logger.error("OBJ conversion failed for %s: %s", path, exc)
                raise
            return xyz

        # Convert GPC files to ``.xyz`` using plain text loading
        if kind == "gpc" and path:
            logger.info("[%s] Konvertiere GPC → XYZ …", base)
            try:
                arr = read_gpc(path)
                np.savetxt(xyz, arr, fmt="%.6f")
            except Exception as exc:
                logger.error("GPC conversion failed for %s: %s", path, exc)
                raise
            return xyz

        logger.error("No suitable file found for conversion: %s", base)
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
    # public API SINGLE CLOUD
    def load_points_singlecloud(self) -> np.ndarray:
        """Load the single cloud epoch."""

        if py4dgeo is None:
            raise RuntimeError("'py4dgeo' ist nicht installiert.")

        singlecloud = self._load_epochs_singlecloud()
        
        if hasattr(singlecloud, "cloud"):
            singlecloud = singlecloud.cloud
        if not isinstance(singlecloud, np.ndarray):
            raise TypeError("Unerwarteter Typ für singlecloud; erwarte np.ndarray (Nx3). Type: %s", type(singlecloud))
        
        return singlecloud
    
    # ------------------------------------------------------------------
    # internal helpers
    def _load_epochs(self) -> Tuple[object, object]:
        """Detect input types and read epochs using :mod:`py4dgeo`."""

        m_kind, m_path = self._detect(self.mov_base)
        r_kind, r_path = self._detect(self.ref_base)

        if m_kind == r_kind == "xyz":
            logger.info("Nutze py4dgeo.read_from_xyz")
            try:
                return py4dgeo.read_from_xyz(str(m_path), str(r_path))
            except (FileNotFoundError, py4dgeo.Py4DGEOError) as exc:
                logger.error("py4dgeo.read_from_xyz failed: %s", exc)
                raise

        if m_kind == r_kind == "laslike":
            logger.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            try:
                return py4dgeo.read_from_las(str(m_path), str(r_path))
            except (FileNotFoundError, py4dgeo.Py4DGEOError) as exc:
                logger.error("py4dgeo.read_from_las failed: %s", exc)
                raise

        if m_kind == r_kind == "ply" and hasattr(py4dgeo, "read_from_ply"):
            logger.info("Nutze py4dgeo.read_from_ply")
            try:
                return py4dgeo.read_from_ply(str(m_path), str(r_path))
            except (FileNotFoundError, py4dgeo.Py4DGEOError) as exc:
                logger.error("py4dgeo.read_from_ply failed: %s", exc)
                raise

        m_xyz = self._ensure_xyz(self.mov_base, (m_kind, m_path))
        r_xyz = self._ensure_xyz(self.ref_base, (r_kind, r_path))
        logger.info("Mischtypen → konvertiert zu XYZ → py4dgeo.read_from_xyz")
        try:
            return py4dgeo.read_from_xyz(str(m_xyz), str(r_xyz))
        except (FileNotFoundError, py4dgeo.Py4DGEOError) as exc:
            logger.error("py4dgeo.read_from_xyz failed: %s", exc)
            raise
    

    def _load_epochs_singlecloud(self) -> np.ndarray:
        """Detect input types and read epochs using :mod:`py4dgeo`."""

        s_kind, s_path = self._detect(self.singlecloud_base)

        if s_kind == "xyz":
            logger.info("Nutze py4dgeo.read_from_xyz")
            try:
                return py4dgeo.read_from_xyz(str(s_path))
            except (FileNotFoundError, py4dgeo.Py4DGEOError) as exc:
                logger.error("py4dgeo.read_from_xyz failed: %s", exc)
                raise

        if s_kind == "laslike":
            logger.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            try:
                return py4dgeo.read_from_las(str(s_path))
            except (FileNotFoundError, py4dgeo.Py4DGEOError) as exc:
                logger.error("py4dgeo.read_from_las failed: %s", exc)
                raise

        if s_kind == "ply" and hasattr(py4dgeo, "read_from_ply"):
            logger.info("Nutze py4dgeo.read_from_ply")
            try:
                return py4dgeo.read_from_ply(str(s_path))
            except (FileNotFoundError, py4dgeo.Py4DGEOError) as exc:
                logger.error("py4dgeo.read_from_ply failed: %s", exc)
                raise

        s_xyz = self._ensure_xyz(self.singlecloud_base, (s_kind, s_path))
        logger.info("Mischtypen → konvertiert zu XYZ → py4dgeo.read_from_xyz")
        try:
            return py4dgeo.read_from_xyz(str(s_xyz))
        except (FileNotFoundError, py4dgeo.Py4DGEOError) as exc:
            logger.error("py4dgeo.read_from_xyz failed: %s", exc)
            raise
    

    def _derive_corepoints(self, mov: object, ref: object) -> np.ndarray:
        """Derive core points from configured epoch with optional subsampling."""

        use_mov = self.config.mov_as_corepoints
        label = "mov" if use_mov else "ref"
        logger.info(
            "Nutze %s als Corepoints und nutze Subsamling: %s",
            label,
            self.config.use_subsampled_corepoints,
        )

        source = mov if use_mov else ref
        data = source.cloud if hasattr(source, "cloud") else source
        return data[:: self.config.use_subsampled_corepoints]


__all__ = ["DataSource"]

