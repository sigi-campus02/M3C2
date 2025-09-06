"""Wrapper around statistics export utilities used in tests.

This module exposes the same helper functions as
``m3c2.exporter.statistics_exporter`` but allows test suites to patch
internal helpers like :func:`_append_df_to_excel` via the public
``m3c2.statistics.exporters`` namespace.
"""

from __future__ import annotations

import logging
import pandas as pd

from m3c2.exporter import statistics_exporter as _impl

logger = logging.getLogger(__name__)

# Expose pandas so tests can patch ``pd.ExcelWriter`` via this module.
pd = pd

CANONICAL_COLUMNS = _impl.CANONICAL_COLUMNS


def _append_df_to_excel(df_new: pd.DataFrame, out_xlsx: str, sheet_name: str = "Results") -> None:
    """Delegate to the real implementation used in the application."""
    _impl._append_df_to_excel(df_new, out_xlsx, sheet_name=sheet_name)


def _append_df_to_json(df_new: pd.DataFrame, out_json: str) -> None:
    """Delegate to the real implementation used in the application."""
    _impl._append_df_to_json(df_new, out_json)


def write_table(
    rows: list[dict],
    out_path: str = "m3c2_stats_all.xlsx",
    sheet_name: str = "Results",
    output_format: str = "excel",
) -> None:
    """Write a statistics table dispatching to Excel or JSON helpers."""
    df = pd.DataFrame(rows)
    if df.empty:
        logger.info("Skipping writing table to %s - no data", out_path)
        return
    if output_format.lower() == "json":
        _append_df_to_json(df, out_path)
    else:
        _append_df_to_excel(df, out_path, sheet_name=sheet_name)


def write_cloud_stats(
    rows: "list[dict] | pd.DataFrame",
    out_path: str = "m3c2_stats_clouds.xlsx",
    sheet_name: str = "CloudStats",
    output_format: str = "excel",
) -> None:
    """Write per-cloud statistics to disk."""
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    elif rows:
        df = pd.DataFrame(rows, columns=list(rows[0].keys()))
    else:
        df = pd.DataFrame()

    if df.empty:
        logger.info("Skipping writing cloud stats to %s - no data", out_path)
        return

    if isinstance(rows, pd.DataFrame):
        if output_format.lower() == "json":
            df.to_json(out_path, orient="records", indent=2)
        else:
            with pd.ExcelWriter(out_path) as writer:
                df.to_excel(writer, sheet_name=sheet_name)
    else:
        if output_format.lower() == "json":
            df.to_json(out_path, orient="records", indent=2)
        else:
            with pd.ExcelWriter(out_path) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)


__all__ = [
    "CANONICAL_COLUMNS",
    "_append_df_to_excel",
    "_append_df_to_json",
    "write_table",
    "write_cloud_stats",
    "pd",
]
