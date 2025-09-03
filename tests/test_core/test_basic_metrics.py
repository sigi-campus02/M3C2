"""Tests for basic statistical metrics and distribution fitting.

This module validates the behavior of the :mod:`m3c2.core.statistics` basic
metric utilities, ensuring they correctly handle edge cases, compute
statistics, and fit probability distributions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.statistics.distance_basic_metrics import basic_stats, fit_distributions


def test_basic_stats_empty_array():
    """Handle empty input arrays in ``basic_stats``.

    Purpose
    -------
    Ensure that ``basic_stats`` gracefully handles empty arrays without
    producing invalid statistics.

    Expected
    --------
    - ``Valid Count`` is ``0``.
    - ``Mean`` and ``Jaccard Index`` are ``NaN``.
    """

    res = basic_stats(np.array([]), tolerance=0.1)
    assert res["Valid Count"] == 0
    assert np.isnan(res["Mean"])
    assert np.isnan(res["Jaccard Index"])


def test_basic_stats_computation():
    """Compute statistics on a simple numeric array.

    Purpose
    -------
    Verify that ``basic_stats`` returns correct statistical measures for a
    small array of known values.

    Expected
    --------
    - ``Valid Count`` equals the array length.
    - ``Min`` and ``Max`` match the array bounds.
    - ``Mean`` and ``Median`` both equal ``2.0``.
    - ``Std Empirical`` matches ``np.std`` of the array.
    - ``Within-Tolerance`` equals ``2/3`` for a tolerance of ``2.0``.
    - ``Jaccard Index`` equals ``1/3``.
    """

    arr = np.array([1.0, 2.0, 3.0])
    res = basic_stats(arr, tolerance=2.0)
    assert res["Valid Count"] == 3
    assert res["Min"] == 1.0
    assert res["Max"] == 3.0
    assert np.isclose(res["Mean"], 2.0)
    assert np.isclose(res["Median"], 2.0)
    assert np.isclose(res["Std Empirical"], np.std(arr))
    assert np.isclose(res["Within-Tolerance"], 2.0 / 3.0)
    assert np.isclose(res["Jaccard Index"], 1.0 / 3.0)


def test_fit_distributions_basic():
    """Fit Gaussian and Weibull distributions to random data.

    Purpose
    -------
    Confirm that ``fit_distributions`` estimates distribution parameters and
    Pearson correlation coefficients for sample data.

    Expected
    --------
    - Estimated mean and standard deviation are close to sample statistics.
    - Pearson coefficients for Gaussian and Weibull fits are non-negative.
    """

    rng = np.random.default_rng(0)
    data = rng.normal(loc=1.0, scale=2.0, size=500)
    hist, bin_edges = np.histogram(data, bins=20)
    res = fit_distributions(data, hist, bin_edges, None)
    assert np.isclose(res["mu"], np.mean(data), atol=0.1)
    assert np.isclose(res["std"], np.std(data), atol=0.1)
    assert res["pearson_gauss"] >= 0
    assert res["pearson_weib"] >= 0


def test_fit_distributions_length_mismatch():
    """Validate length checks in ``fit_distributions``.

    Purpose
    -------
    Ensure that mismatched histogram inputs trigger an assertion error.

    Expected
    --------
    ``fit_distributions`` raises an :class:`AssertionError` when histogram
    and bin edge arrays are of differing lengths.
    """

    with pytest.raises(AssertionError):
        fit_distributions(np.arange(10), np.ones(5), np.arange(6), None)
