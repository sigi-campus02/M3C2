"""Tests for the :mod:`m3c2.importer.file_detection` module."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))
from m3c2.importer.file_detection import detect


@pytest.mark.parametrize(
    "suffix, expected",
    [
        (".xyz", "xyz"),
        (".las", "laslike"),
        (".laz", "laslike"),
        (".ply", "ply"),
        (".obj", "obj"),
        (".gpc", "gpc"),
        (None, None),
    ],
)
def test_detect(tmp_path: Path, suffix: str | None, expected: str | None) -> None:
    """Detect point cloud files with various extensions.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by ``pytest``.
    suffix : str | None
        File extension to create, or ``None`` for no file.
    expected : str | None
        Expected format identifier returned by :func:`detect`.

    Returns
    -------
    None
        This test only asserts behaviour and does not return anything.
    """
    base = tmp_path / "sample"
    if suffix:
        path = base.with_suffix(suffix)
        path.write_text("data")
    kind, detected_path = detect(base)
    assert kind == expected
    if expected is None:
        assert detected_path is None
    else:
        assert detected_path == base.with_suffix(suffix)
