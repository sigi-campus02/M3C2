"""Tests for StatisticsService.calc_single_cloud_stats output ordering."""

import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.core.statistics import service


def test_calc_single_cloud_stats_file_folder_first(monkeypatch):
    """Ensure resulting rows start with File and Folder keys."""

    def fake_calc_single_cloud_stats(
        pts,
        area_m2=None,
        radius=None,
        k=None,
        sample_size=None,
        use_convex_hull=True,
    ) -> Dict:
        return {"metric": 1}

    captured_rows: List[Dict] = []

    def fake_write_cloud_stats(rows, out_path, sheet_name, output_format):
        captured_rows.extend(rows)

    monkeypatch.setattr(service, "calc_single_cloud_stats", fake_calc_single_cloud_stats)
    monkeypatch.setattr(service, "write_cloud_stats", fake_write_cloud_stats)

    service.StatisticsService.calc_single_cloud_stats(
        folder_ids=["fid"],
        filename_singlecloud="file",
        singlecloud=None,
        out_path="out.json",
        output_format="json",
    )

    assert captured_rows, "No rows were captured"
    assert list(captured_rows[0].keys())[:2] == ["File", "Folder"]

