from __future__ import annotations

"""Loader for plain XYZ point cloud files."""

from pathlib import Path


class XYZLoader:
    """Read point clouds in the XYZ text format."""

    def __init__(self, backend):
        self.backend = backend

    def load_pair(self, mov_path: Path, ref_path: Path):
        return self.backend.read_from_xyz(str(mov_path), str(ref_path))

    def load_single(self, path: Path):
        return self.backend.read_from_xyz(str(path))
