"""Aggregate M3C2 statistics across multiple folders."""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .distance_stats import calc_stats
from .exporters import _append_df_to_excel, _append_df_to_json
from .path_utils import _resolve

logger = logging.getLogger(__name__)


def compute_m3c2_statistics(
    folder_ids: List[str],
    filename_ref: str = "",
    process_python_CC: str = "python",
    bins: int = 256,
    range_override: Optional[Tuple[float, float]] = None,
    min_expected: Optional[float] = None,
    out_path: str = "m3c2_stats_all.xlsx",
    sheet_name: str = "Results",
    output_format: str = "excel",
    outlier_multiplicator: float = 3.0,
    outlier_method: str = "rmse",
) -> pd.DataFrame:
    """Gather M3C2 distance statistics from multiple project folders.

    Parameters mirror those of the former ``StatisticsService`` class method.
    The resulting dataframe is optionally appended to an Excel or JSON file.
    """

    logger.info("Starting compute_m3c2_statistics for %d folders", len(folder_ids))
    rows: List[Dict] = []

    for fid in folder_ids:
        py_dist_path = _resolve(fid, f"python_{filename_ref}_m3c2_distances.txt")
        py_params_path = _resolve(fid, f"python_{filename_ref}_m3c2_params.txt")

        if os.path.exists(py_dist_path):
            logger.info("Processing Python distances: %s", py_dist_path)
            logger.info("Using Python params: %s", py_params_path)
            try:
                values = np.loadtxt(py_dist_path)
            except (OSError, ValueError) as exc:
                logger.warning(
                    "Skipping folder %s due to unreadable distances file %s: %s",
                    fid,
                    py_dist_path,
                    exc,
                )
                continue

            stats = calc_stats(
                values,
                params_path=py_params_path if os.path.exists(py_params_path) else None,
                bins=bins,
                range_override=range_override,
                min_expected=min_expected,
                outlier_multiplicator=outlier_multiplicator,
                outlier_method=outlier_method,
            )
            rows.append(
                {
                    "Folder": fid,
                    "Version": filename_ref or "",
                    "Distances Path": py_dist_path,
                    "Params Path": py_params_path if os.path.exists(py_params_path) else "",
                    **stats,
                }
            )

    df_result = pd.DataFrame(rows)

    if out_path and not df_result.empty:
        if output_format.lower() == "json":
            _append_df_to_json(df_result, out_path)
        else:
            _append_df_to_excel(df_result, out_path, sheet_name=sheet_name)

    logger.info("Finished compute_m3c2_statistics for %d folders", len(folder_ids))
    return df_result


__all__ = ["compute_m3c2_statistics"]

