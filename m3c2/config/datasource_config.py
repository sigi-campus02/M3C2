"""Datasource configuration definitions.

This module defines the :class:`DataSourceConfig` dataclass which stores
settings for locating the input point clouds and controlling how core
points are derived.

Attributes
----------
folder:
    Directory containing the point cloud files.
comparison_basename:
    Basename for the comparison point cloud file.
reference_basename:
    Basename for the reference point cloud file.
comparison_as_corepoints:
    If ``True``, the comparison cloud provides the core points.
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
    comparison_basename:
        Basename for the comparison point cloud file (without extension).
    reference_basename:
        Basename for the reference point cloud file (without extension).
    filename_singlecloud:
        Name used when a single point cloud file is provided.
    comparison_as_corepoints:
        If ``True``, the comparison cloud is used to derive core points.
    use_subsampled_corepoints:
        Factor by which to subsample the core points.
    """

    folder: str
    comparison_basename: str = "comparison"
    reference_basename: str = "reference"
    filename_singlecloud: str = "comparison"
    comparison_as_corepoints: bool = True
    use_subsampled_corepoints: int = 1

