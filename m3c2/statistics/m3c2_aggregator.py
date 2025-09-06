"""Aggregate M3C2 distance statistics across multiple folders."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import logging
import os

import numpy as np
import pandas as pd

from .distance_stats import calc_stats
from m3c2.exporter.statistics_exporter import _append_df_to_excel, _append_df_to_json
from .path_utils import _resolve

logger = logging.getLogger(__name__)


def compute_m3c2_statistics(
    folder_ids: List[str],
    filename_reference: str = "",
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

    Parameters
    ----------
    folder_ids : List[str]
        Identifiers of project folders to process.
    filename_reference : str, optional
        Reference label used to construct filenames, by default ``""``.
    process_python_CC : str, optional
        Indicates whether results stem from the Python or CloudCompare
        implementation.  Currently only the Python variant is handled.
    bins : int, optional
        Number of histogram bins for :func:`calc_stats`.
    range_override : Optional[Tuple[float, float]], optional
        Explicit distance range to use instead of the data bounds.
    min_expected : Optional[float], optional
        Minimum expected count per histogram bin for distribution fitting.
    out_path : str, optional
        Destination file for the aggregated results table.
    sheet_name : str, optional
        Sheet name used when exporting to Excel.
    output_format : str, optional
        Output format, either ``"excel"`` or ``"json"``.
    outlier_multiplicator : float, optional
        Factor applied to the outlier threshold metric.
    outlier_method : str, optional
        Method used for outlier detection.

    Returns
    -------
    pandas.DataFrame
        One row per processed folder with the computed statistics.
    """

    logger.info("Starting compute_m3c2_statistics for %d folders", len(folder_ids))
    rows: List[Dict] = []

    for fid in folder_ids:
        py_dist_path = _resolve(fid, f"python_{filename_reference}_m3c2_distances.txt")

        py_params_path = _resolve(fid, f"python_{filename_reference}_m3c2_params.txt")

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
                    "Version": filename_reference or "",
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

