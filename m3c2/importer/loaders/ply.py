"""Loader for PLY point cloud files."""

from __future__ import annotations


from pathlib import Path


class PLYLoader:
    """Read point clouds from PLY files using :mod:`py4dgeo`."""

    def __init__(self, backend):
        """Initialize loader with a backend providing PLY read support.

        Parameters
        ----------
        backend : object
            Object exposing a ``read_from_ply`` function used to read files.
        """
        if not hasattr(backend, "read_from_ply"):
            raise RuntimeError("'py4dgeo' lacks PLY support")
        self.backend = backend

    def load_pair(self, comparison_path: Path, reference_path: Path):
        """Load comparison and reference point clouds from the given paths.

        Parameters
        ----------
        comparison_path : Path
            Path to the comparison epoch.
        reference_path : Path
            Path to the reference epoch.

        Returns
        -------
        Any
            Objects returned by the backend for the two epochs.
        """
        return self.backend.read_from_ply(str(comparison_path), str(reference_path))

    def load_single(self, path: Path):
        """Load a single point cloud from the given path.

        Parameters
        ----------
        path : Path
            Path to the point cloud file.

        Returns
        -------
        Any
            Object returned by the backend for the epoch.
        """
        return self.backend.read_from_ply(str(path))
