import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.core.statistics import exporters


def test_write_table_calls_excel_by_default():
    rows = [{"a": 1}]
    with patch("m3c2.core.statistics.exporters._append_df_to_excel") as m_excel:
        exporters.write_table(rows, out_path="dummy.xlsx")
        m_excel.assert_called_once()
        df_arg = m_excel.call_args.args[0]
        assert isinstance(df_arg, pd.DataFrame)
        assert df_arg.iloc[0]["a"] == 1


def test_write_table_calls_json_when_requested():
    rows = [{"a": 1}]
    with patch("m3c2.core.statistics.exporters._append_df_to_json") as m_json:
        exporters.write_table(rows, out_path="dummy.json", output_format="json")
        m_json.assert_called_once()


def test_write_table_empty_rows():
    with patch("m3c2.core.statistics.exporters._append_df_to_excel") as m_excel:
        exporters.write_table([], out_path="dummy.xlsx")
        m_excel.assert_not_called()


def test_write_cloud_stats_excel():
    rows = [{"a": 1}]
    with patch("os.path.exists", return_value=False), \
         patch("m3c2.core.statistics.exporters.pd.ExcelWriter") as m_writer, \
         patch("pandas.DataFrame.to_excel") as m_to_excel:
        m_writer.return_value.__enter__.return_value = MagicMock()
        exporters.write_cloud_stats(rows, out_path="dummy.xlsx")
        m_to_excel.assert_called_once()


def test_write_cloud_stats_json():
    rows = [{"a": 1}]
    with patch("os.path.exists", return_value=False), \
         patch("pandas.DataFrame.to_json") as m_to_json:
        exporters.write_cloud_stats(rows, out_path="dummy.json", output_format="json")
        m_to_json.assert_called_once()


def test_write_cloud_stats_empty_rows():
    with patch("pandas.DataFrame.to_excel") as m_to_excel:
        exporters.write_cloud_stats([], out_path="dummy.xlsx")
        m_to_excel.assert_not_called()
