"""Tests for outlier detection utilities.

This module validates the behavior of helper functions responsible for
identifying outliers and computing outlier statistics.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.core.statistics.outliers import compute_outliers, get_outlier_mask


def test_get_outlier_mask_rmse():
    """Verify RMSE-based outlier masking.

    Ensures that the root mean square error (RMSE) method flags the large
    value as an outlier and returns the calculated threshold.

    Returns
    -------
    None
        This test function does not return a value.
    """

    arr = np.array([0.0, 0.0, 10.0])
    mask, threshold = get_outlier_mask(arr, method="rmse", outlier_multiplicator=1.0)
    rmse = np.sqrt(np.mean(arr ** 2))
    assert mask.tolist() == [False, False, True]
    assert threshold == pytest.approx(rmse)


def test_get_outlier_mask_iqr():
    """Check outlier detection using the IQR method.

    The interquartile range (IQR) is used to identify the extreme value in the
    sample array.

    Returns
    -------
    None
        This test function does not return a value.
    """

    arr = np.array([1.0, 2.0, 3.0, 100.0])
    mask, threshold = get_outlier_mask(arr, method="iqr", outlier_multiplicator=1.0)
    assert mask.tolist() == [False, False, False, True]
    assert isinstance(threshold, str)


def test_get_outlier_mask_std():
    """Validate standard deviation outlier detection.

    Confirms that values exceeding one standard deviation from the mean are
    marked as outliers and that the correct threshold is returned.

    Returns
    -------
    None
        This test function does not return a value.
    """

    arr = np.array([0.0, 0.0, 10.0])
    mask, threshold = get_outlier_mask(arr, method="std", outlier_multiplicator=1.0)
    mu = np.mean(arr)
    std = np.std(arr)
    expected = np.abs(arr - mu) > std
    assert mask.tolist() == expected.tolist()
    assert threshold == pytest.approx(std)


def test_get_outlier_mask_nmad():
    """Assess NMAD-based masking.

    The normalized median absolute deviation (NMAD) method should detect the
    solitary large value and provide a near-zero threshold.

    Returns
    -------
    None
        This test function does not return a value.
    """

    arr = np.array([0.0, 0.0, 0.0, 10.0])
    mask, threshold = get_outlier_mask(arr, method="nmad", outlier_multiplicator=1.0)
    assert mask.tolist() == [False, False, False, True]
    assert threshold == pytest.approx(0.0)


def test_get_outlier_mask_invalid_method():
    """Ensure unsupported methods raise an error.

    Raises
    ------
    ValueError
        If an unknown outlier detection method is requested.

    Returns
    -------
    None
        This test function does not return a value.
    """

    with pytest.raises(ValueError):
        get_outlier_mask(np.array([1.0]), method="foo", outlier_multiplicator=1.0)


def test_compute_outliers_basic():
    """Compute basic outlier statistics.

    Verifies that counts and summary statistics are correctly computed when
    both positive and negative outliers are present.

    Returns
    -------
    None
        This test function does not return a value.
    """

    inliers = np.array([1.0, -2.0, 3.0])
    outliers = np.array([10.0, -20.0])
    res = compute_outliers(inliers, outliers)
    assert res["outlier_count"] == 2
    assert res["inlier_count"] == 3
    assert res["pos_out"] == 1
    assert res["neg_out"] == 1
    assert res["mean_out"] == pytest.approx(np.mean(outliers))
    assert res["std_out"] == pytest.approx(np.std(outliers))


def test_compute_outliers_empty_outliers():
    """Handle cases without outliers.

    When the outlier array is empty, the function should report zero outliers
    and propagate ``nan`` for the mean and standard deviation.

    Returns
    -------
    None
        This test function does not return a value.
    """

    inliers = np.array([1.0, 2.0])
    outliers = np.array([])
    res = compute_outliers(inliers, outliers)
    assert res["outlier_count"] == 0
    assert np.isnan(res["mean_out"])
    assert np.isnan(res["std_out"])
