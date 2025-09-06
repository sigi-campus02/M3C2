"""Loader for LAS/LAZ point cloud files."""

from __future__ import annotations

from pathlib import Path


class LASLoader:
    """Read point clouds from LAS or LAZ files."""

    def __init__(self, backend):
        self.backend = backend

    def load_pair(self, comparison_path: Path, reference_path: Path):
        return self.backend.read_from_las(str(comparison_path), str(reference_path))

    def load_single(self, path: Path):
        return self.backend.read_from_las(str(path))
