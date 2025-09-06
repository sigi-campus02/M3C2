"""Tests for :mod:`m3c2.exporter.statistics_exporter`.

The tests mock out heavy dependencies and filesystem access to ensure
correct path handling, error behaviour and logging without touching the
real disk.
"""

import logging
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.exporter import statistics_exporter as exporter


def test_append_df_to_json_handles_read_error(tmp_path, caplog):
    """Existing JSON parse errors are logged and directories are created."""
    df = pd.DataFrame({"Total Points": [1]})
    out = tmp_path / "sub" / "data.json"

    with patch("os.path.exists", return_value=True), \
         patch("os.makedirs") as m_makedirs, \
         patch("pandas.read_json", side_effect=ValueError("bad")), \
         patch("pandas.DataFrame.to_json") as m_to_json, \
         caplog.at_level(logging.INFO):
        exporter._append_df_to_json(df, str(out))

    m_makedirs.assert_called_once_with(str(out.parent), exist_ok=True)
    m_to_json.assert_called_once()
    assert m_to_json.call_args.args[0] == str(out)
    assert "Failed to read existing JSON" in caplog.text
    assert f"Appended {len(df)} rows to JSON file {out}" in caplog.text


def test_append_df_to_excel_creates_dir_and_logs(tmp_path, caplog):
    """A new Excel file triggers directory creation and info logging."""
    df = pd.DataFrame({"Total Points": [1]})
    out = tmp_path / "sub" / "stats.xlsx"

    fake_ws = MagicMock()
    fake_wb = MagicMock(active=fake_ws)

    openpyxl_module = ModuleType("openpyxl")
    openpyxl_module.__path__ = []
    openpyxl_module.Workbook = MagicMock(return_value=fake_wb)
    openpyxl_module.load_workbook = MagicMock()

    utils_module = ModuleType("openpyxl.utils")
    utils_module.__path__ = []
    dataframe_module = ModuleType("openpyxl.utils.dataframe")
    dataframe_module.dataframe_to_rows = MagicMock(return_value=[[1]])
    utils_module.dataframe = dataframe_module
    openpyxl_module.utils = utils_module

    with patch.dict(sys.modules, {
        "openpyxl": openpyxl_module,
        "openpyxl.utils": utils_module,
        "openpyxl.utils.dataframe": dataframe_module,
    }), patch("os.path.exists", return_value=False), \
         patch("os.makedirs") as m_makedirs, \
         caplog.at_level(logging.INFO):
        exporter._append_df_to_excel(df, str(out))

    m_makedirs.assert_called_once_with(str(out.parent), exist_ok=True)
    openpyxl_module.Workbook.assert_called_once()
    fake_wb.save.assert_called_once_with(str(out))
    assert f"Appended {len(df)} rows to Excel file {out}" in caplog.text


def test_append_df_to_excel_missing_openpyxl_raises(tmp_path):
    """A helpful error is raised when ``openpyxl`` is not installed."""
    df = pd.DataFrame({"Total Points": [1]})
    with patch.dict(sys.modules, {"openpyxl": None}):
        with pytest.raises(ModuleNotFoundError) as exc:
            exporter._append_df_to_excel(df, str(tmp_path / "out.xlsx"))
    assert "openpyxl" in str(exc.value)
