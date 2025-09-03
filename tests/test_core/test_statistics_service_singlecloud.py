"""Tests for StatisticsService.calc_single_cloud_stats output ordering."""

import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.core.statistics import service


def test_calc_single_cloud_stats_metadata_first(monkeypatch):
    """Ensure resulting rows start with metadata keys."""

    def fake_calc_single_cloud_stats(
        pts,
        area_m2=None,
        radius=None,
        k=None,
        sample_size=None,
        use_convex_hull=True,
    ) -> Dict:
        return {"metric": 1}

    captured_dfs: List[pd.DataFrame] = []

    def fake_write_cloud_stats(df, out_path, sheet_name, output_format):
        captured_dfs.append(df)

    monkeypatch.setattr(service, "calc_single_cloud_stats", fake_calc_single_cloud_stats)
    monkeypatch.setattr(service, "write_cloud_stats", fake_write_cloud_stats)

    service.StatisticsService.calc_single_cloud_stats(
        folder_ids=["fid"],
        filename_singlecloud="file",
        singlecloud=None,
        data_dir="dd",
        out_path="out.json",
        output_format="json",
    )

    assert captured_dfs, "No data frame was captured"
    df = captured_dfs[0]
    assert list(df.index[:3]) == [
        "Timestamp",
        "Data Dir",
        "File",
    ]

