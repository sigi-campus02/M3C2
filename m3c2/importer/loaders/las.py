from __future__ import annotations

"""Loader for LAS/LAZ point cloud files."""

from pathlib import Path


class LASLoader:
    """Read point clouds from LAS or LAZ files."""

    def __init__(self, backend):
        self.backend = backend

    def load_pair(self, mov_path: Path, ref_path: Path):
        return self.backend.read_from_las(str(mov_path), str(ref_path))

    def load_single(self, path: Path):
        return self.backend.read_from_las(str(path))
