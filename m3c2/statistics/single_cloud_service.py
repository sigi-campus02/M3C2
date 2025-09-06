"""Compute statistics for individual point clouds."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .singlecloud_metrics import calc_single_cloud_stats as _calc_single_cloud_stats
from m3c2.exporter.statistics_exporter import write_cloud_stats


def calc_single_cloud_stats(
    folder_ids: List[str],
    filename_singlecloud: str,
    singlecloud: Optional[object] = None,
    data_dir: str = "",
    area_m2: Optional[float] = None,
    radius: float = None,
    k: int = 24,
    sample_size: Optional[int] = None,
    use_convex_hull: bool = True,
    out_path: str = "m3c2_stats_clouds.xlsx",
    sheet_name: str = "CloudStats",
    output_format: str = "excel",
) -> pd.DataFrame:
    """Evaluate quality metrics for point clouds in multiple folders.

    Parameters mirror the previous ``StatisticsService.calc_single_cloud_stats``
    class method but the implementation is now a lightweight function.
    """

    rows: List[Dict] = []

    for fid in folder_ids:
        pts = singlecloud.cloud if hasattr(singlecloud, "cloud") else singlecloud

        stats = _calc_single_cloud_stats(
            pts,
            area_m2=area_m2,
            radius=radius,
            k=k,
            sample_size=sample_size,
            use_convex_hull=use_convex_hull,
        )
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats = {
            "Timestamp": ts,
            "Data Dir": data_dir,
            "Folder": fid,
            "File": filename_singlecloud,
            **stats,
        }
        rows.append(stats)

    df_result = pd.DataFrame(rows)
    df_result_t = df_result.set_index("Folder").T
    df_result_t.index.name = "Metric"

    if out_path and rows:
        write_cloud_stats(
            df_result_t,
            out_path=out_path,
            sheet_name=sheet_name,
            output_format=output_format,
        )

    return df_result_t


__all__ = ["calc_single_cloud_stats"]

