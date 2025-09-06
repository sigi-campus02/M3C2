"""Compute and export statistics for point cloud analysis.

This subpackage bundles utilities for deriving descriptive metrics, assessing
cloud quality, detecting statistical outliers, and exporting tabular
summaries.  Functions are organised by responsibility to keep the
codebase maintainable and composable.
"""

from .distance_stats import calc_stats, _load_params
from .m3c2_aggregator import compute_m3c2_statistics
from .single_cloud_service import calc_single_cloud_stats
from .path_utils import _resolve
from m3c2.exporter.statistics_exporter import CANONICAL_COLUMNS, write_table, write_cloud_stats

__all__ = [
    "calc_stats",
    "compute_m3c2_statistics",
    "calc_single_cloud_stats",
    "_load_params",
    "_resolve",
    "CANONICAL_COLUMNS",
    "write_table",
    "write_cloud_stats",
]
