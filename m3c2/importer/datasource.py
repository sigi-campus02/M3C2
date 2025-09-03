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

from m3c2.config.datasource_config import DataSourceConfig
from .file_detection import detect
from .converters import ensure_xyz
from .loaders import XYZLoader, LASLoader, PLYLoader


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
        """Detect input types and read epochs using format-specific loaders."""

        m_kind, m_path = detect(self.mov_base)
        r_kind, r_path = detect(self.ref_base)

        if m_kind == r_kind == "xyz":
            logger.info("Nutze py4dgeo.read_from_xyz")
            loader = XYZLoader(py4dgeo)
            return loader.load_pair(m_path, r_path)

        if m_kind == r_kind == "laslike":
            logger.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            loader = LASLoader(py4dgeo)
            return loader.load_pair(m_path, r_path)

        if m_kind == r_kind == "ply" and hasattr(py4dgeo, "read_from_ply"):
            logger.info("Nutze py4dgeo.read_from_ply")
            loader = PLYLoader(py4dgeo)
            return loader.load_pair(m_path, r_path)

        m_xyz = ensure_xyz(self.mov_base, (m_kind, m_path))
        r_xyz = ensure_xyz(self.ref_base, (r_kind, r_path))
        logger.info("Mischtypen → konvertiert zu XYZ → py4dgeo.read_from_xyz")
        loader = XYZLoader(py4dgeo)
        return loader.load_pair(m_xyz, r_xyz)
    

    def _load_epochs_singlecloud(self) -> np.ndarray:
        """Detect input types and read epochs using format-specific loaders."""

        s_kind, s_path = detect(self.singlecloud_base)

        if s_kind == "xyz":
            logger.info("Nutze py4dgeo.read_from_xyz")
            loader = XYZLoader(py4dgeo)
            return loader.load_single(s_path)

        if s_kind == "laslike":
            logger.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            loader = LASLoader(py4dgeo)
            return loader.load_single(s_path)

        if s_kind == "ply" and hasattr(py4dgeo, "read_from_ply"):
            logger.info("Nutze py4dgeo.read_from_ply")
            loader = PLYLoader(py4dgeo)
            return loader.load_single(s_path)

        s_xyz = ensure_xyz(self.singlecloud_base, (s_kind, s_path))
        logger.info("Mischtypen → konvertiert zu XYZ → py4dgeo.read_from_xyz")
        loader = XYZLoader(py4dgeo)
        return loader.load_single(s_xyz)
    

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

