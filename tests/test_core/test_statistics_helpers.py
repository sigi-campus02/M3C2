import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from m3c2.statistics import path_utils, distance_outlier_metrics as dom
from m3c2.statistics import exporters
from m3c2.statistics import m3c2_aggregator


def test_resolve_prefers_local(monkeypatch, tmp_path):
    """_resolve should prefer files in the given folder before falling back to data/."""
    fid = tmp_path / "fid"
    fid.mkdir()
    monkeypatch.chdir(tmp_path)
    local = fid / "a.txt"
    local.write_text("x")
    assert path_utils._resolve("fid", "a.txt") == os.path.join("fid", "a.txt")
    # When the file is missing fall back to data/<fid>/filename
    assert path_utils._resolve("fid", "missing.txt") == os.path.join("data", "fid", "missing.txt")


@pytest.mark.parametrize("method", ["rmse", "iqr", "std", "nmad"])
def test_get_outlier_mask_methods(method):
    """get_outlier_mask should handle all supported methods."""
    arr = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    mask, threshold = dom.get_outlier_mask(arr, method, factor=1.0)
    assert mask.shape == arr.shape
    assert mask.dtype == bool
    if method == "iqr":
        assert isinstance(threshold, str)
    else:
        assert isinstance(threshold, float)


def test_get_outlier_mask_invalid():
    """Unknown methods raise ValueError."""
    with pytest.raises(ValueError):
        dom.get_outlier_mask(np.array([1, 2]), "foo", 1.0)


def test_compute_outliers_basic():
    """compute_outliers summarises inlier/outlier arrays."""
    inliers = np.array([0.0, 1.0])
    outliers = np.array([2.0, -3.0])
    res = dom.compute_outliers(inliers, outliers)
    assert res["outlier_count"] == 2
    assert res["inlier_count"] == 2
    assert res["pos_out"] == 1
    assert res["neg_out"] == 1
    assert np.isclose(res["mean_out"], np.mean(outliers))
    assert np.isclose(res["std_out"], np.std(outliers))


def test_compute_outliers_empty():
    """Empty arrays yield zero counts and NaN statistics."""
    res = dom.compute_outliers(np.array([]), np.array([]))
    assert res["outlier_count"] == 0
    assert res["inlier_count"] == 0
    assert np.isnan(res["mean_out"])
    assert np.isnan(res["std_out"])


def test_write_table_dispatch(monkeypatch, tmp_path):
    """write_table dispatches based on output_format and skips empty data."""
    called = {}

    def fake_excel(df, out, sheet_name="Results"):
        called["excel"] = (df.copy(), out, sheet_name)

    def fake_json(df, out):
        called["json"] = (df.copy(), out)

    monkeypatch.setattr(exporters, "_append_df_to_excel", fake_excel)
    monkeypatch.setattr(exporters, "_append_df_to_json", fake_json)

    rows = [{"A": 1}]
    exporters.write_table(rows, out_path="out.xlsx", output_format="excel")
    exporters.write_table(rows, out_path="out.json", output_format="json")
    exporters.write_table([], out_path="skip.xlsx")

    assert "excel" in called and called["excel"][1] == "out.xlsx"
    assert "json" in called and called["json"][1] == "out.json"


def test_write_cloud_stats_json(tmp_path):
    """write_cloud_stats writes JSON for list input."""
    rows = [{"A": 1}, {"A": 2}]
    out = tmp_path / "stats.json"
    exporters.write_cloud_stats(rows, out_path=str(out), output_format="json")
    data = pd.read_json(out)
    assert len(data) == 2
    assert list(data.columns) == ["A"]


def test_write_cloud_stats_excel(monkeypatch):
    """write_cloud_stats uses ExcelWriter for DataFrame input."""
    rows = pd.DataFrame({"A": [1, 2]})
    called = {}

    class DummyWriter:
        def __init__(self, path):
            called["path"] = path
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_to_excel(self, writer, sheet_name="CloudStats", index=False):
        called["sheet"] = sheet_name
        called["index"] = index

    monkeypatch.setattr(exporters.pd, "ExcelWriter", lambda path: DummyWriter(path))
    monkeypatch.setattr(pd.DataFrame, "to_excel", fake_to_excel)

    exporters.write_cloud_stats(rows, out_path="out.xlsx", output_format="excel")
    assert called["path"] == "out.xlsx"
    assert called["sheet"] == "CloudStats"
    assert called["index"] is False


def test_compute_m3c2_statistics(tmp_path, monkeypatch):
    """compute_m3c2_statistics aggregates statistics for valid folders."""
    fid = tmp_path / "proj1"
    fid.mkdir()
    (fid / "python__m3c2_distances.txt").write_text("0\n1\n-1\n2\n-2\n")
    (fid / "python__m3c2_params.txt").write_text("NormalScale=1.0\nSearchScale=2.0\n")
    monkeypatch.chdir(tmp_path)

    called = {}

    def fake_json(df, out):
        called["path"] = out

    monkeypatch.setattr(m3c2_aggregator, "_append_df_to_json", fake_json)
    df = m3c2_aggregator.compute_m3c2_statistics(
        ["proj1"],
        out_path="stats.json",
        output_format="json",
        outlier_multiplicator=2.0,
    )
    assert not df.empty and len(df) == 1
    assert called["path"] == "stats.json"
    assert df.loc[0, "Normal Scale"] == 1.0
    assert df.loc[0, "Search Scale"] == 2.0


def test_compute_m3c2_statistics_missing(tmp_path, monkeypatch):
    """Folders without distance files are skipped."""
    fid = tmp_path / "missing"
    fid.mkdir()
    monkeypatch.chdir(tmp_path)
    called = {}
    monkeypatch.setattr(m3c2_aggregator, "_append_df_to_json", lambda df, out: called.setdefault("called", True))
    df = m3c2_aggregator.compute_m3c2_statistics(
        ["missing"], out_path="stats.json", output_format="json"
    )
    assert df.empty
    assert "called" not in called
