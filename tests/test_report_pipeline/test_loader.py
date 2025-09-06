from pathlib import Path
import logging

import numpy as np
import pytest

from report_pipeline.plotting.loader import load_distance_series


@pytest.mark.parametrize(
    ("filename", "content", "expected"),
    [
        ("data.txt", "1\n2\n3\n", [1.0, 2.0, 3.0]),
        ("data.csv", "1,2,3\n4,5,6\n", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ],
)
def test_load_distance_series_valid(
    tmp_path: Path,
    filename: str,
    content: str,
    expected: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Loading valid text and CSV files returns a float array without logging."""
    file_path = tmp_path / filename
    file_path.write_text(content)

    with caplog.at_level(logging.DEBUG):
        data = load_distance_series(file_path)

    assert isinstance(data, np.ndarray)
    assert data.dtype == float
    assert np.array_equal(data, np.array(expected, dtype=float))
    assert caplog.text == ""


def test_load_distance_series_missing_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Missing files raise ``FileNotFoundError`` without logging."""
    missing = tmp_path / "missing.txt"

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(FileNotFoundError) as excinfo:
            load_distance_series(missing)

    assert f"Distance file not found: {missing}" in str(excinfo.value)
    assert caplog.text == ""


def test_load_distance_series_invalid_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Non-numeric content raises ``ValueError`` and logs nothing."""
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("1\nfoo\n3\n")

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ValueError) as excinfo:
            load_distance_series(bad_file)

    assert f"Could not read numeric data from {bad_file}" == str(excinfo.value)
    assert caplog.text == ""
