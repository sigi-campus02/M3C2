from __future__ import annotations

"""Loader for LAS/LAZ point cloud files."""

from pathlib import Path


class LASLoader:
    """Read point clouds from LAS or LAZ files."""

    def __init__(self, backend):
        """Initialize loader with a backend.

        Parameters:
            backend: Object providing LAS reading capabilities.

        Returns:
            None
        """
        self.backend = backend

    def load_pair(self, comparison_path: Path, reference_path: Path):
        """Load point clouds for comparison and reference.

        Parameters:
            comparison_path (Path): Path to the point cloud for comparison.
            reference_path (Path): Path to the point cloud for reference.

        Returns:
            Any: Loaded comparison and reference clouds.
        """
        return self.backend.read_from_las(str(comparison_path), str(reference_path))

    def load_single(self, path: Path):
        """Load a single point cloud.

        Parameters:
            path (Path): Path to the point cloud file.

        Returns:
            Any: Loaded point cloud.
        """
        return self.backend.read_from_las(str(path))
