"""Tests for the statistics exporters utilities.

These tests validate that the :mod:`m3c2.core.statistics.exporters`
module writes tabular results using the appropriate backend depending on
requested file format and that it gracefully handles empty inputs.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import logging

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.core.statistics import exporters


def test_write_table_calls_excel_by_default():
    """Ensure ``write_table`` uses the Excel backend by default.

    Notes
    -----
    The helper responsible for appending data frames to Excel files should
    be invoked when no explicit output format is provided.
    """

    rows = [{"a": 1}]
    with patch("m3c2.core.statistics.exporters._append_df_to_excel") as m_excel:
        exporters.write_table(rows, out_path="dummy.xlsx")
        m_excel.assert_called_once()
        df_arg = m_excel.call_args.args[0]
        assert isinstance(df_arg, pd.DataFrame)
        assert df_arg.iloc[0]["a"] == 1


def test_write_table_calls_json_when_requested():
    """Ensure ``write_table`` dispatches to JSON when requested.

    Notes
    -----
    When ``output_format`` is set to ``"json"``, the JSON-specific
    append helper should be invoked once.
    """

    rows = [{"a": 1}]
    with patch("m3c2.core.statistics.exporters._append_df_to_json") as m_json:
        exporters.write_table(rows, out_path="dummy.json", output_format="json")
        m_json.assert_called_once()


def test_write_table_empty_rows():
    """Verify ``write_table`` skips writing when rows are empty.

    Notes
    -----
    The Excel append helper must not be invoked if the input ``rows``
    list is empty.
    """

    with patch("m3c2.core.statistics.exporters._append_df_to_excel") as m_excel:
        exporters.write_table([], out_path="dummy.xlsx")
        m_excel.assert_not_called()


def test_append_df_to_json_handles_read_errors(tmp_path, caplog):
    """Existing JSON parse errors are logged and an empty DataFrame is used."""

    out = tmp_path / "data.json"
    out.write_text("not valid json")
    df = pd.DataFrame({"Total Points": [1]})
    with caplog.at_level(logging.WARNING):
        exporters._append_df_to_json(df, str(out))
    assert "Failed to read existing JSON" in caplog.text
    result = pd.read_json(out)
    assert len(result) == 1


def test_write_cloud_stats_excel():
    """Ensure ``write_cloud_stats`` writes rows with metadata columns."""

    rows = [{
        "Timestamp": "ts",
        "Data Dir": "dd",
        "Folder": "run1",
        "File": "f",
        "a": 1,
    }]
    captured = {}

    def fake_to_excel(self, *args, **kwargs):
        captured["df"] = self

    with patch("os.path.exists", return_value=False), \
         patch("m3c2.core.statistics.exporters.pd.ExcelWriter") as m_writer, \
         patch("pandas.DataFrame.to_excel", new=fake_to_excel):
        m_writer.return_value.__enter__.return_value = MagicMock()
        exporters.write_cloud_stats(rows, out_path="dummy.xlsx")

    df_written = captured["df"]
    assert list(df_written.columns) == ["Timestamp", "Data Dir", "Folder", "File", "a"]
    assert df_written.iloc[0]["Folder"] == "run1"


def test_write_cloud_stats_transposed_df(monkeypatch):
    """DataFrame inputs keep index and no timestamp column is injected."""

    df = pd.DataFrame({"run1": [1]}, index=["metric"])
    df.index.name = "Metric"
    captured = {}

    def fake_to_excel(self, *args, **kwargs):
        captured["df"] = self

    with patch("os.path.exists", return_value=False), \
         patch("m3c2.core.statistics.exporters.pd.ExcelWriter") as m_writer, \
         patch("pandas.DataFrame.to_excel", new=fake_to_excel):
        m_writer.return_value.__enter__.return_value = MagicMock()
        exporters.write_cloud_stats(df, out_path="dummy.xlsx")

    df_written = captured["df"]
    assert "Timestamp" not in df_written.columns
    assert list(df_written.index) == ["metric"]


def test_write_cloud_stats_json():
    """Ensure ``write_cloud_stats`` dispatches to JSON when requested.

    Notes
    -----
    The JSON output path should be used when ``output_format`` equals
    ``"json"``, resulting in a call to ``DataFrame.to_json``.
    """

    rows = [{"a": 1}]
    with patch("os.path.exists", return_value=False), \
         patch("pandas.DataFrame.to_json") as m_to_json:
        exporters.write_cloud_stats(rows, out_path="dummy.json", output_format="json")
        m_to_json.assert_called_once()


def test_write_cloud_stats_empty_rows():
    """Verify ``write_cloud_stats`` handles empty inputs gracefully.

    Notes
    -----
    When provided with an empty list of rows, the DataFrame export should
    not be triggered.
    """

    with patch("pandas.DataFrame.to_excel") as m_to_excel:
        exporters.write_cloud_stats([], out_path="dummy.xlsx")
        m_to_excel.assert_not_called()
