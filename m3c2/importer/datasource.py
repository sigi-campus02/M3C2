"""Utilities for loading heterogeneous point cloud data.

This module centralises the logic for reading point cloud pairs from a number
of supported file formats and converting them to a unified XYZ representation.
Only the necessary optional dependencies are imported for the requested file
types, enabling this module to be imported even when those libraries are not
installed.  The :class:`DataSource` class is the public entry point and exposes
the :meth:`DataSource.load_points` method which returns the comparison epoch, the
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
    def comparison_base(self) -> Path:
        """Base path of the comparison epoch file.

        Combines the configured folder and comparison filename base without
        an extension, resulting in the path used to detect the comparison
        epoch's data file.
        """

        return Path(self.config.folder) / self.config.comparison_basename

    @property
    def reference_base(self) -> Path:
        """Return the base path for the reference epoch.

        The path is constructed by combining the configured data folder with
        the reference file's base name. The resulting path intentionally lacks
        a file extension; downstream loaders append the appropriate suffix when
        searching for the actual reference data file.
        """
        return Path(self.config.folder) / self.config.reference_basename
    

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
        """Load the comparison and reference epochs and derive core points."""

        if py4dgeo is None:
            raise RuntimeError("'py4dgeo' ist nicht installiert.")

        comparison, reference = self._load_epochs()

        # Always use reference point cloud for corepoints
        corepoints = self._derive_corepoints(reference)

        if not isinstance(corepoints, np.ndarray):
            raise TypeError("Unerwarteter Typ für corepoints; erwarte np.ndarray (Nx3).")

        return comparison, reference, corepoints
    
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

        # Detect input types and actual file paths for the two epochs
        comparison_kind, comparison_path = detect(self.comparison_base)
        reference_kind, reference_path = detect(self.reference_base)

        if comparison_kind == reference_kind == "xyz":
            logger.info("Nutze py4dgeo.read_from_xyz")
            loader = XYZLoader(py4dgeo)  # select XYZ loader
            return loader.load_pair(comparison_path, reference_path)

        if comparison_kind == reference_kind == "laslike":
            logger.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            loader = LASLoader(py4dgeo)  # select LAS loader
            return loader.load_pair(comparison_path, reference_path)

        if comparison_kind == reference_kind == "ply" and hasattr(py4dgeo, "read_from_ply"):
            logger.info("Nutze py4dgeo.read_from_ply")
            loader = PLYLoader(py4dgeo)  # select PLY loader
            return loader.load_pair(comparison_path, reference_path)

        # Mixed formats: convert both epochs to temporary XYZ files
        comparison_xyz_path = ensure_xyz(self.comparison_base, (comparison_kind, comparison_path))
        reference_xyz_path = ensure_xyz(self.reference_base, (reference_kind, reference_path))
        logger.info("Mischtypen -> konvertiert zu XYZ -> py4dgeo.read_from_xyz")
        loader = XYZLoader(py4dgeo)
        return loader.load_pair(comparison_xyz_path, reference_xyz_path)
    

    def _load_epochs_singlecloud(self) -> np.ndarray:
        """Detect input types and read epochs using format-specific loaders."""

        # Detect input type and file path for the single cloud
        single_kind, single_path = detect(self.singlecloud_base)

        if single_kind == "xyz":
            logger.info("Nutze py4dgeo.read_from_xyz")
            loader = XYZLoader(py4dgeo)  # select XYZ loader
            return loader.load_single(single_path)

        if single_kind == "laslike":
            logger.info("Nutze py4dgeo.read_from_las (unterstützt .las und .laz)")
            loader = LASLoader(py4dgeo)  # select LAS loader
            return loader.load_single(single_path)

        if single_kind == "ply" and hasattr(py4dgeo, "read_from_ply"):
            logger.info("Nutze py4dgeo.read_from_ply")
            loader = PLYLoader(py4dgeo)  # select PLY loader
            return loader.load_single(single_path)

        # Convert input to XYZ when format is mixed/unsupported
        single_xyz_path = ensure_xyz(self.singlecloud_base, (single_kind, single_path))
        logger.info("Mischtypen -> konvertiert zu XYZ -> py4dgeo.read_from_xyz")
        loader = XYZLoader(py4dgeo)
        return loader.load_single(single_xyz_path)
    

    def _derive_corepoints(self, reference: object) -> np.ndarray:
        """Derive core points from reference epoch with optional subsampling."""

        # For logging, label the source of the core points
        reference_label = "reference"
        logger.info(
            "Nutze %s als Corepoints und nutze Subsamling: %s",
            reference_label,
            self.config.use_subsampled_corepoints,
        )

        # Extract the raw NumPy array from the py4dgeo object when needed
        source = reference
        data = source.cloud if hasattr(source, "cloud") else source

        # Apply subsampling step to reduce the number of core points
        return data[:: self.config.use_subsampled_corepoints]


__all__ = ["DataSource"]

