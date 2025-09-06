from __future__ import annotations

"""Loader for PLY point cloud files."""

from pathlib import Path


class PLYLoader:
    """Read point clouds from PLY files using :mod:`py4dgeo`."""

    def __init__(self, backend):
        """Initialize loader with a backend.

        Parameters:
            backend: Object providing PLY reading capabilities.

        Returns:
            None
        """
        if not hasattr(backend, "read_from_ply"):
            raise RuntimeError("'py4dgeo' lacks PLY support")
        self.backend = backend

    def load_pair(self, comparison_path: Path, reference_path: Path):
        """Load point clouds for comparison and reference.

        Parameters:
            comparison_path (Path): Path to the point cloud for comparison.
            reference_path (Path): Path to the point cloud for reference.

        Returns:
            Any: Loaded comparison and reference clouds.
        """
        return self.backend.read_from_ply(str(comparison_path), str(reference_path))

    def load_single(self, path: Path):
        """Load a single point cloud.

        Parameters:
            path (Path): Path to the point cloud file.

        Returns:
            Any: Loaded point cloud.
        """
        return self.backend.read_from_ply(str(path))
