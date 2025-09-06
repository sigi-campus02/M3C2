"""Tests for the :class:`OutlierHandler` utility."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.statistics.outlier_handler import OutlierHandler


def test_detect_marks_outliers():
    """Identify outliers and mark them with ``1`` in the mask."""

    distances = np.array([0.0, 0.0, 0.0, 100.0, 0.0, -100.0, np.nan])
    mask = OutlierHandler.detect(distances, method="iqr", factor=1.5)
    expected = np.array([0, 0, 0, 1, 0, 1, 0], dtype=np.uint8)
    assert np.array_equal(mask, expected)

    inliers = distances[mask == 0]
    assert 100.0 not in inliers
    assert -100.0 not in inliers


def test_detect_empty_array():
    """Return an empty mask for empty input arrays."""

    mask = OutlierHandler.detect(np.array([]), method="std", factor=1.0)
    assert mask.size == 0


def test_detect_invalid_method():
    """Raise ``ValueError`` when an unknown method is provided."""

    with pytest.raises(ValueError):
        OutlierHandler.detect(np.array([1.0]), method="unknown", factor=1.0)

