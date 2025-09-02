"""Compute and export statistics for point cloud analysis.

This subpackage bundles utilities for deriving descriptive metrics, assessing
cloud quality, detecting statistical outliers, and exporting tabular
summaries.  The :class:`~m3c2.core.statistics.service.StatisticsService`
coordinates these helpers for use in the higher level pipeline.
"""

from .service import StatisticsService
from .exporters import CANONICAL_COLUMNS, write_table, write_cloud_stats

__all__ = [
    "StatisticsService",
    "CANONICAL_COLUMNS",
    "write_table",
    "write_cloud_stats",
]
