"""Loader for PLY point cloud files."""

from __future__ import annotations


from pathlib import Path


class PLYLoader:
    """Read point clouds from PLY files using :mod:`py4dgeo`."""

    def __init__(self, backend):
        if not hasattr(backend, "read_from_ply"):
            raise RuntimeError("'py4dgeo' lacks PLY support")
        self.backend = backend

    def load_pair(self, comparison_path: Path, reference_path: Path):
        return self.backend.read_from_ply(str(comparison_path), str(reference_path))

    def load_single(self, path: Path):
        return self.backend.read_from_ply(str(path))