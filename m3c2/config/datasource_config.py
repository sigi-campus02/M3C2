"""Datasource configuration definitions.

This module defines the :class:`DataSourceConfig` dataclass which stores
settings for locating the input point clouds and controlling how core
points are derived.

Attributes
----------
folder:
    Directory containing the point cloud files.
mov_basename:
    Basename for the moving point cloud file.
ref_basename:
    Basename for the reference point cloud file.
mov_as_corepoints:
    If ``True``, the moving cloud provides the core points.
use_subsampled_corepoints:
    Factor by which core points are subsampled.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DataSourceConfig:
    """Configuration for the data source.

    Parameters
    ----------
    folder:
        Directory containing the point cloud files.
    mov_basename:
        Basename for the moving point cloud file (without extension).
    ref_basename:
        Basename for the reference point cloud file (without extension).
    filename_singlecloud:
        Name used when a single point cloud file is provided.
    mov_as_corepoints:
        If ``True``, the moving cloud is used to derive core points.
    use_subsampled_corepoints:
        Factor by which to subsample the core points.
    """

    folder: str
    mov_basename: str = "mov"
    ref_basename: str = "ref"
    filename_singlecloud: str = "mov"
    mov_as_corepoints: bool = True
    use_subsampled_corepoints: int = 1

